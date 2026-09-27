//! Statistics learned from human-made charts.
//!
//! `itg-charter train <Songs dir>` reads every dance-single chart it finds and
//! records, per difficulty:
//! - an arrow n-gram: P(row | two previous rows, time gap), rows as 4-bit panel masks;
//! - densities (rows per second), jump / hold ratios, hold lengths, rhythmic snaps;
//! - tech rates computed with the ITGmania parity port (crossovers, footswitches...);
//! - a linear fit of the meter against the density of the busiest measures.
//!
//! Generation reads these numbers instead of hand-written constants.

use crate::difficulty::Difficulty;
use crate::parity::{self, Layout, TechCounts};
use crate::simfile::{Cell, NoteRow, Simfile, Timing};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MODEL_VERSION: u32 = 2;
pub const GAP_BUCKETS: usize = 6;
pub const MASKS: usize = 16;
pub const SNAP_CLASSES: usize = 6;
const NDIFF: usize = 5;

/// Tempo prior: histogram of log2(BPM) from `BPM_PRIOR_MIN`, `BPM_PRIOR_BINS_PER_OCTAVE`
/// bins per octave over `BPM_PRIOR_OCTAVES` octaves.
pub const BPM_PRIOR_MIN: f64 = 60.0;
pub const BPM_PRIOR_BINS_PER_OCTAVE: f64 = 24.0;
pub const BPM_PRIOR_OCTAVES: f64 = 2.5;

/// The model shipped with the binary (trained on the ITGmania songs available at build time).
pub const EMBEDDED: &str = include_str!("../model/model.json");

/// Bucket of the time gap between two rows, in 48ths of a beat.
pub fn gap_bucket(gap48: u32) -> usize {
    match gap48 {
        0..=11 => 0,  // faster than 16ths
        12 => 1,      // 16th
        13..=16 => 2, // 12th
        17..=24 => 3, // 8th
        25..=48 => 4, // up to a quarter
        _ => 5,       // longer
    }
}

/// Snap class of a row position in 48ths of a beat: 4th, 8th, 12th, 16th, 24th, other.
pub fn snap_class(pos48: u32) -> usize {
    let p = pos48 % 48;
    if p == 0 {
        0
    } else if p.is_multiple_of(24) {
        1
    } else if p.is_multiple_of(16) {
        2
    } else if p.is_multiple_of(12) {
        3
    } else if p.is_multiple_of(8) {
        4
    } else {
        5
    }
}

/// Mirror left ↔ right of a 4-panel mask (L D U R → R D U L).
pub fn mirror(mask: u8) -> u8 {
    (mask & 0b0110) | ((mask & 1) << 3) | ((mask >> 3) & 1)
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Quantiles {
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
}

impl Quantiles {
    pub fn of(values: &[f64]) -> Quantiles {
        let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return Quantiles::default();
        }
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        Quantiles {
            p10: q(0.1),
            p25: q(0.25),
            p50: q(0.5),
            p75: q(0.75),
            p90: q(0.9),
        }
    }
}

/// Per-difficulty statistics.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DiffStats {
    pub charts: u32,
    /// Rows per second over the charted part of the song.
    pub nps: Quantiles,
    /// Fraction of rows that are jumps (2+ panels).
    pub jump_ratio: Quantiles,
    /// Fraction of rows that start a hold.
    pub hold_ratio: Quantiles,
    /// Hold lengths in beats.
    pub hold_beats: Quantiles,
    /// Fraction of rows falling on each snap class (4th, 8th, 12th, 16th, 24th, other).
    pub snaps: [f64; SNAP_CLASSES],
    /// Tech counts per 100 rows.
    pub crossovers: Quantiles,
    pub footswitches: Quantiles,
    pub jacks: Quantiles,
    pub doublesteps: Quantiles,
    pub brackets: Quantiles,
    pub meter: Quantiles,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub version: u32,
    pub stats: Vec<DiffStats>,
    /// Counts indexed by [difficulty][gap bucket][row-2][row-1][row], rows as masks (0 = start).
    pub ngram: Vec<u32>,
    /// meter ≈ intercept + slope × (75th percentile of per-measure rows per second).
    pub meter_intercept: f64,
    pub meter_slope: f64,
    /// How often human charters chose each tempo (the song's main BPM), smoothed and
    /// scaled to a maximum of 1; used to pick the tempo octave. See `BPM_PRIOR_*`.
    pub bpm_prior: Vec<f64>,
}

fn ngram_index(d: usize, g: usize, p2: usize, p1: usize, cur: usize) -> usize {
    (((d * GAP_BUCKETS + g) * MASKS + p2) * MASKS + p1) * MASKS + cur
}

/// Numbers describing one chart, shared by training and evaluation of generated charts.
#[derive(Clone, Debug, Default)]
pub struct ChartFeatures {
    pub rows: usize,
    pub nps: f64,
    pub measure_nps_p75: f64,
    pub jump_ratio: f64,
    pub hold_ratio: f64,
    pub hold_beats: Vec<f64>,
    pub snaps: [u32; SNAP_CLASSES],
    pub tech: Option<TechCounts>,
    /// (mask, position in 48ths) of every row with at least one tap/hold head.
    pub sequence: Vec<(u8, u32)>,
}

impl ChartFeatures {
    pub fn compute(rows: &[NoteRow], timing: &Timing, layout: &Layout) -> ChartFeatures {
        let mut f = ChartFeatures::default();
        let mut seconds = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            let mut mask = 0u8;
            let mut holds = 0;
            for (c, cell) in r.cells.iter().take(4).enumerate() {
                match cell {
                    Cell::Tap | Cell::Lift => mask |= 1 << c,
                    Cell::HoldHead | Cell::RollHead => {
                        mask |= 1 << c;
                        holds += 1;
                        if let Some(t) = rows[i + 1..].iter().find(|n| n.cells.get(c) == Some(&Cell::Tail)) {
                            f.hold_beats.push(t.beat - r.beat);
                        }
                    }
                    _ => {}
                }
            }
            if mask == 0 {
                continue;
            }
            let pos = (r.beat * 48.0).round().max(0.0) as u32;
            f.sequence.push((mask, pos));
            f.snaps[snap_class(pos)] += 1;
            if mask.count_ones() >= 2 {
                f.jump_ratio += 1.0;
            }
            if holds > 0 {
                f.hold_ratio += 1.0;
            }
            seconds.push(timing.seconds(r.beat));
        }
        f.rows = f.sequence.len();
        if f.rows == 0 {
            return f;
        }
        f.jump_ratio /= f.rows as f64;
        f.hold_ratio /= f.rows as f64;
        let span = seconds.last().unwrap() - seconds[0];
        f.nps = if span > 0.0 { f.rows as f64 / span } else { 0.0 };
        // Per-measure density, over measures that have notes.
        let mut per_measure: Vec<(u32, usize)> = Vec::new();
        for &(_, pos) in &f.sequence {
            let m = pos / 192;
            match per_measure.last_mut() {
                Some((lm, n)) if *lm == m => *n += 1,
                _ => per_measure.push((m, 1)),
            }
        }
        let densities: Vec<f64> = per_measure
            .iter()
            .map(|&(m, n)| {
                let len = timing.seconds((m + 1) as f64 * 4.0) - timing.seconds(m as f64 * 4.0);
                if len > 0.0 { n as f64 / len } else { 0.0 }
            })
            .collect();
        f.measure_nps_p75 = Quantiles::of(&densities).p75;
        let mut prows = parity::rows_from_chart(rows, timing, 4);
        if parity::analyze(layout, &mut prows).is_some() {
            f.tech = Some(TechCounts::from_rows(layout, &prows));
        }
        f
    }
}

/// Finds simfiles below `dir`, preferring `.ssc` over `.sm` in the same folder. Sorted.
pub fn find_simfiles(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        let mut files: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        files.sort();
        let ext = |p: &Path, e: &str| p.extension().is_some_and(|x| x.eq_ignore_ascii_case(e));
        let has_ssc = files.iter().any(|p| ext(p, "ssc"));
        for p in files {
            if p.is_dir() {
                stack.push(p);
            } else if ext(&p, "ssc") || (ext(&p, "sm") && !has_ssc) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

impl Model {
    pub fn embedded() -> Result<Model> {
        Model::from_json(EMBEDDED)
    }

    pub fn from_json(text: &str) -> Result<Model> {
        let m: Model = serde_json::from_str(text).context("parsing model")?;
        anyhow::ensure!(
            m.version == MODEL_VERSION,
            "model version {} unsupported",
            m.version
        );
        Ok(m)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("model serializes")
    }

    /// Trains on every dance-single chart found below `songs_dir`.
    pub fn train(songs_dir: &Path, mut progress: impl FnMut(usize, usize)) -> Result<Model> {
        let files = find_simfiles(songs_dir);
        anyhow::ensure!(!files.is_empty(), "no simfile found in {}", songs_dir.display());
        let layout = Layout::dance_single();
        let mut ngram = vec![0u32; NDIFF * GAP_BUCKETS * MASKS * MASKS * MASKS];
        let mut per_diff: Vec<Vec<ChartFeatures>> = vec![Vec::new(); NDIFF];
        let mut meters: Vec<Vec<f64>> = vec![Vec::new(); NDIFF];
        let mut fit_points = Vec::new();
        let mut main_bpms = Vec::new();
        for (i, path) in files.iter().enumerate() {
            progress(i, files.len());
            let Ok(sim) = Simfile::load(path) else { continue };
            if let Some(bpm) = sim
                .charts
                .iter()
                .find(|c| c.steps_type == "dance-single")
                .and_then(|c| Some(main_bpm(&sim.timing(c).ok()?, &c.rows().ok()?)))
            {
                main_bpms.push(bpm);
            }
            for chart in &sim.charts {
                if chart.steps_type != "dance-single" {
                    continue;
                }
                let Ok(diff) = Difficulty::parse(&chart.difficulty) else {
                    continue;
                };
                let (Ok(timing), Ok(rows)) = (sim.timing(chart), chart.rows()) else {
                    continue;
                };
                let f = ChartFeatures::compute(&rows, &timing, &layout);
                if f.rows < 16 {
                    continue;
                }
                let d = diff.index();
                for mirrored in [false, true] {
                    let (mut p2, mut p1, mut last) = (0usize, 0usize, None);
                    for &(mask, pos) in &f.sequence {
                        let mask = if mirrored { mirror(mask) } else { mask } as usize;
                        let g = last.map_or(GAP_BUCKETS - 1, |l| gap_bucket(pos - l));
                        ngram[ngram_index(d, g, p2, p1, mask)] += 1;
                        (p2, p1, last) = (p1, mask, Some(pos));
                    }
                }
                if chart.meter > 0 {
                    meters[d].push(chart.meter as f64);
                    fit_points.push((f.measure_nps_p75, chart.meter as f64));
                }
                per_diff[d].push(f);
            }
        }
        let stats = per_diff
            .iter()
            .zip(&meters)
            .map(|(charts, meters)| diff_stats(charts, meters))
            .collect();
        let (meter_intercept, meter_slope) = linear_fit(&fit_points);
        Ok(Model {
            version: MODEL_VERSION,
            stats,
            ngram,
            meter_intercept,
            meter_slope,
            bpm_prior: bpm_histogram(&main_bpms),
        })
    }

    /// Prior weight of a tempo (0 < p ≤ 1), linear interpolation of `bpm_prior`.
    pub fn bpm_prior(&self, bpm: f64) -> f64 {
        const FLOOR: f64 = 0.02;
        let x = (bpm / BPM_PRIOR_MIN).log2() * BPM_PRIOR_BINS_PER_OCTAVE;
        let h = &self.bpm_prior;
        if h.is_empty() || !(0.0..(h.len() - 1) as f64).contains(&x) {
            return FLOOR;
        }
        let i = x.floor() as usize;
        let f = x - i as f64;
        (h[i] * (1.0 - f) + h[i + 1] * f).max(FLOOR)
    }

    pub fn stats(&self, d: Difficulty) -> &DiffStats {
        &self.stats[d.index()]
    }

    /// Smoothed probability of `cur` given the two previous rows and the gap bucket.
    /// Backs off from trigram to bigram to unigram (Dirichlet smoothing).
    pub fn prob(&self, d: Difficulty, gap: usize, p2: u8, p1: u8, cur: u8) -> f64 {
        self.marginals(d, gap)
            .prob(p2 as usize, p1 as usize, cur as usize)
    }

    fn marginals(&self, d: Difficulty, gap: usize) -> Marginals<'_> {
        let base = ngram_index(d.index(), gap, 0, 0, 0);
        let counts = &self.ngram[base..base + MASKS * MASKS * MASKS];
        let mut m = Marginals {
            counts,
            uni: [0.0; MASKS],
            uni_t: 0.0,
            bi: [[0.0; MASKS]; MASKS],
            bi_t: [0.0; MASKS],
            tri_t: [[0.0; MASKS]; MASKS],
        };
        for a in 0..MASKS {
            for b in 0..MASKS {
                for c in 1..MASKS {
                    let n = counts[(a * MASKS + b) * MASKS + c] as f64;
                    m.uni[c] += n;
                    m.uni_t += n;
                    m.bi[b][c] += n;
                    m.bi_t[b] += n;
                    m.tri_t[a][b] += n;
                }
            }
        }
        m
    }

    /// Precomputes log-probabilities for fast lookups during generation.
    pub fn table(&self, d: Difficulty) -> ProbTable {
        let mut logp = vec![0f32; GAP_BUCKETS * MASKS * MASKS * MASKS];
        for g in 0..GAP_BUCKETS {
            let m = self.marginals(d, g);
            for p2 in 0..MASKS {
                for p1 in 0..MASKS {
                    for cur in 1..MASKS {
                        let i = ((g * MASKS + p2) * MASKS + p1) * MASKS + cur;
                        logp[i] = m.prob(p2, p1, cur).ln() as f32;
                    }
                }
            }
        }
        ProbTable { logp }
    }

    /// Estimated meter for a chart's busiest-measure density.
    pub fn meter_for(&self, measure_nps_p75: f64) -> f64 {
        self.meter_intercept + self.meter_slope * measure_nps_p75
    }
}

struct Marginals<'a> {
    counts: &'a [u32],
    uni: [f64; MASKS],
    uni_t: f64,
    bi: [[f64; MASKS]; MASKS],
    bi_t: [f64; MASKS],
    tri_t: [[f64; MASKS]; MASKS],
}

impl Marginals<'_> {
    fn prob(&self, p2: usize, p1: usize, cur: usize) -> f64 {
        const ALPHA: f64 = 4.0;
        let pu = (self.uni[cur] + ALPHA / 15.0) / (self.uni_t + ALPHA);
        let pb = (self.bi[p1][cur] + ALPHA * pu) / (self.bi_t[p1] + ALPHA);
        let c = self.counts[(p2 * MASKS + p1) * MASKS + cur] as f64;
        (c + ALPHA * pb) / (self.tri_t[p2][p1] + ALPHA)
    }
}

/// Log-probabilities of the arrow n-gram for one difficulty.
pub struct ProbTable {
    logp: Vec<f32>,
}

impl ProbTable {
    pub fn logp(&self, gap: usize, p2: u8, p1: u8, cur: u8) -> f32 {
        self.logp[((gap * MASKS + p2 as usize) * MASKS + p1 as usize) * MASKS + cur as usize]
    }
}

/// The BPM in effect for most of the chart (the tempo a player would name).
fn main_bpm(timing: &Timing, rows: &[NoteRow]) -> f64 {
    let last = rows.last().map_or(0.0, |r| r.beat);
    let mut best = (timing.bpms[0].1, f64::MIN);
    for (i, &(start, bpm)) in timing.bpms.iter().enumerate() {
        let end = timing.bpms.get(i + 1).map_or(last, |b| b.0).min(last);
        let span = end - start.max(0.0);
        if span > best.1 {
            best = (bpm, span);
        }
    }
    best.0
}

/// Smoothed histogram of log2(BPM), scaled to a maximum of 1.
fn bpm_histogram(bpms: &[f64]) -> Vec<f64> {
    let bins = (BPM_PRIOR_OCTAVES * BPM_PRIOR_BINS_PER_OCTAVE) as usize + 1;
    let mut h = vec![0.0; bins];
    for &b in bpms {
        let x = (b / BPM_PRIOR_MIN).log2() * BPM_PRIOR_BINS_PER_OCTAVE;
        if (0.0..bins as f64).contains(&x) {
            h[x.round() as usize] += 1.0;
        }
    }
    // Gaussian smoothing, sigma = 2 bins (1/12 octave).
    let sigma = 2.0f64;
    let smoothed: Vec<f64> = (0..bins)
        .map(|i| {
            (0..bins)
                .map(|j| h[j] * (-0.5 * ((i as f64 - j as f64) / sigma).powi(2)).exp())
                .sum()
        })
        .collect();
    let max = smoothed.iter().copied().fold(0.0, f64::max).max(1e-12);
    smoothed.iter().map(|v| v / max).collect()
}

/// Tempo prior of the embedded model (loaded once).
pub fn embedded_bpm_prior(bpm: f64) -> f64 {
    static MODEL: std::sync::OnceLock<Model> = std::sync::OnceLock::new();
    MODEL
        .get_or_init(|| Model::embedded().expect("embedded model"))
        .bpm_prior(bpm)
}

fn diff_stats(charts: &[ChartFeatures], meters: &[f64]) -> DiffStats {
    let col = |f: &dyn Fn(&ChartFeatures) -> f64| Quantiles::of(&charts.iter().map(f).collect::<Vec<_>>());
    let tech = |f: &dyn Fn(&TechCounts) -> u32| {
        Quantiles::of(
            &charts
                .iter()
                .filter_map(|c| c.tech.map(|t| 100.0 * f(&t) as f64 / c.rows as f64))
                .collect::<Vec<_>>(),
        )
    };
    let mut snaps = [0f64; SNAP_CLASSES];
    let total: u32 = charts.iter().map(|c| c.snaps.iter().sum::<u32>()).sum();
    for c in charts {
        for (s, n) in snaps.iter_mut().zip(c.snaps) {
            *s += n as f64 / total.max(1) as f64;
        }
    }
    let holds: Vec<f64> = charts.iter().flat_map(|c| c.hold_beats.iter().copied()).collect();
    DiffStats {
        charts: charts.len() as u32,
        nps: col(&|c| c.nps),
        jump_ratio: col(&|c| c.jump_ratio),
        hold_ratio: col(&|c| c.hold_ratio),
        hold_beats: Quantiles::of(&holds),
        snaps,
        crossovers: tech(&|t| t.crossovers),
        footswitches: tech(&|t| t.footswitches),
        jacks: tech(&|t| t.jacks),
        doublesteps: tech(&|t| t.doublesteps),
        brackets: tech(&|t| t.brackets),
        meter: Quantiles::of(meters),
    }
}

fn linear_fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    if n < 2.0 {
        return (1.0, 1.0);
    }
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (my - slope * mx, slope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_and_snaps() {
        assert_eq!(gap_bucket(12), 1);
        assert_eq!(gap_bucket(24), 3);
        assert_eq!(gap_bucket(48), 4);
        assert_eq!(gap_bucket(96), 5);
        assert_eq!(snap_class(0), 0);
        assert_eq!(snap_class(24), 1);
        assert_eq!(snap_class(16), 2);
        assert_eq!(snap_class(36), 3);
        assert_eq!(mirror(0b0001), 0b1000);
        assert_eq!(mirror(0b0110), 0b0110);
    }

    #[test]
    fn embedded_model_loads_and_is_sane() {
        let m = Model::embedded().unwrap();
        let easy = m.stats(Difficulty::Easy);
        let hard = m.stats(Difficulty::Hard);
        assert!(easy.charts > 0 && hard.charts > 0);
        assert!(easy.nps.p50 < hard.nps.p50);
        assert!(m.meter_slope > 0.0);
        // The library is mostly pop/EDM: human tempos peak around 127 BPM.
        let peak = (80..250)
            .max_by(|a, b| m.bpm_prior(*a as f64).total_cmp(&m.bpm_prior(*b as f64)))
            .unwrap();
        assert!((115..140).contains(&peak), "prior peaks at {peak}");
        assert!(m.bpm_prior(75.0) < m.bpm_prior(150.0));
        assert!(m.bpm_prior(300.0) < m.bpm_prior(150.0));
        let s: f64 = (1..16u8).map(|c| m.prob(Difficulty::Medium, 3, 1, 8, c)).sum();
        assert!((s - 1.0).abs() < 1e-6, "probabilities sum to {s}");
    }
}
