//! Chart generation: where the notes go (placement) and which arrows (selection).
//!
//! Placement turns detected onsets into rows on the beat grid, with densities,
//! snaps, jump and hold ratios taken from the trained [`Model`].
//! Selection is a beam search over arrow choices scored by
//! - the n-gram learned from human charts (what looks like ITG),
//! - minus the ITGmania parity cost (what is comfortable to play),
//! - plus seeded Gumbel noise (variety, reproducible),
//! - plus a bonus for repeating the arrows of an earlier, similar measure.

use crate::analysis::{Grid, Layer, SongAnalysis};
use crate::difficulty::Difficulty;
use crate::model::{ChartFeatures, Model, ProbTable, gap_bucket, snap_class};
use crate::parity::{self, Layout, Row, State};
use crate::simfile::{Cell, NoteRow, OutChart, ROWS_PER_BEAT, Timing};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Tunable weights of the generator. The defaults were chosen by comparing the
/// tech rates of generated charts with the human ones (see `examples/eval_charts.rs`).
#[derive(Clone, Copy, Debug)]
pub struct GenOptions {
    pub beam_width: usize,
    /// Nats per unit of parity cost.
    pub parity_weight: f64,
    /// Scale of the Gumbel noise (0 = deterministic arg-max regardless of seed).
    pub temperature: f64,
    /// Bonus (nats) for reusing the arrow of the matching note of a similar measure.
    pub repeat_bonus: f64,
    /// Minimum cosine similarity for two measures to count as a repetition.
    pub repeat_similarity: f64,
    /// Penalty (nats) for an arrow whose cheapest parity makes a footswitch (as the
    /// game's tech counts define it), on top of the parity cost; see
    /// [`footswitch_penalty`] for its scaling per difficulty.
    pub footswitch_penalty: f64,
}

impl Default for GenOptions {
    fn default() -> Self {
        GenOptions {
            beam_width: 12,
            parity_weight: 0.02,
            temperature: 0.4,
            repeat_bonus: 1.5,
            repeat_similarity: 0.9,
            footswitch_penalty: 0.0,
        }
    }
}

/// Share of the rows spread over measures in proportion to their expected human
/// density; the rest go to the best positions of the whole song. Measured with
/// `examples/eval_placement.rs` (test split): 1.0 gives a per-measure density
/// correlation of 0.320 (F natural 70.0%); 0.75 and 0.5 fall back to the global
/// ranking's 0.27 (F 70.4%) — there is no middle ground.
pub const MEASURE_QUOTA_SHARE: f64 = 1.0;

/// Bonus added to the placement score of positions that repeat the rhythm of an
/// earlier, similar-sounding measure (0 = off), per difficulty. Measured with
/// `examples/eval_placement.rs` (test split): no effect on the F-score; in Challenge
/// it brings the share of repeated rhythms closer to the human one (18% -> 29.5%,
/// human 44%) while in easier slots, already at or above the human share, it would
/// only add repetition. Hence Challenge only.
pub const RHYTHM_REUSE_BONUS: [f32; 5] = [0.0, 0.0, 0.0, 0.0, 0.1];

/// Picks rows by decreasing score: first within each measure's quota (if any), then up
/// to `target` overall, never closer than `finest` 48ths to another row.
fn select_rows(
    scored: &[(f32, u32, Candidate, f32)],
    quota: Option<&BTreeMap<u32, usize>>,
    target: usize,
    finest: u32,
) -> BTreeMap<u32, Candidate> {
    let mut chosen: BTreeMap<u32, Candidate> = BTreeMap::new();
    let fits = |chosen: &BTreeMap<u32, Candidate>, pos: u32| {
        let lo = pos.saturating_sub(finest - 1);
        chosen.range(lo..pos + finest).next().is_none()
    };
    if let Some(quota) = quota {
        let mut left = quota.clone();
        for (_, pos, c, _) in scored {
            if let Some(q) = left.get_mut(&(pos / (4 * ROWS_PER_BEAT)))
                && *q > 0
                && fits(&chosen, *pos)
            {
                chosen.insert(*pos, *c);
                *q -= 1;
            }
        }
    }
    for (_, pos, c, _) in scored {
        if chosen.len() >= target {
            break;
        }
        if fits(&chosen, *pos) {
            chosen.insert(*pos, *c);
        }
    }
    chosen
}

/// A placed row, before arrows are chosen.
#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    /// Position in 48ths of a beat.
    pub pos: u32,
    pub strength: f32,
    pub jump: bool,
    /// Hold length in 48ths of a beat.
    pub hold: Option<u32>,
}

/// One candidate position with the evidence for it.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    pos: u32,
    strength: f32,
    kick: f32,
}

/// 90th percentile of the onset strengths of a layer (normalization).
fn strength_scale(layer: &Layer) -> f32 {
    let mut v: Vec<f32> = layer.onsets.iter().map(|o| o.1).collect();
    if v.is_empty() {
        return 1.0;
    }
    v.sort_by(f32::total_cmp);
    v[(v.len() - 1) * 9 / 10].max(1e-6)
}

/// Snaps a time to the coarsest subdivision (4th, 8th, 12th, 16th) within tolerance.
pub fn quantize(grid: &Grid, t: f64) -> Option<u32> {
    let b48 = grid.beat(t) * ROWS_PER_BEAT as f64;
    if b48 < 0.0 {
        return None;
    }
    for step in [48.0, 24.0, 16.0, 12.0] {
        let m = (b48 / step).round() * step;
        let err_s = (b48 - m).abs() / ROWS_PER_BEAT as f64 * grid.period();
        let tol = (0.3 * step / ROWS_PER_BEAT as f64 * grid.period()).min(0.035);
        if err_s <= tol {
            return Some(m as u32);
        }
    }
    None
}

fn layer_rms_at(layer: &Layer, t: f64) -> f32 {
    let i = (layer.envelope.fps * t) as usize;
    layer.rms.get(i).copied().unwrap_or(0.0)
}

/// Part of the song considered "playing": between the first and last hop louder
/// than 10% of the 95th-percentile level.
fn active_range(mix: &Layer, duration: f64) -> (f64, f64) {
    let mut v = mix.rms.clone();
    v.sort_by(f32::total_cmp);
    let loud = v
        .get(v.len().saturating_sub(1) * 95 / 100)
        .copied()
        .unwrap_or(0.0);
    let thr = 0.1 * loud;
    let fps = mix.envelope.fps;
    let first = mix.rms.iter().position(|r| *r > thr).unwrap_or(0) as f64 / fps;
    let last = mix.rms.iter().rposition(|r| *r > thr).unwrap_or(0) as f64 / fps;
    (first.max(0.0), last.min(duration - 0.5).max(first))
}

/// Gathers onset candidates from every available layer, merged per grid position.
fn candidates(a: &SongAnalysis) -> Vec<Candidate> {
    let mut layers: Vec<(&Layer, f32)> = vec![];
    match &a.stems {
        Some(s) => {
            layers.push((&s.drums, 1.0));
            layers.push((&s.bass, 0.6));
            layers.push((&s.vocals, 0.7));
            layers.push((&s.other, 0.6));
            layers.push((&a.mix, 0.5));
        }
        None => layers.push((&a.mix, 1.0)),
    }
    let mut by_pos: BTreeMap<u32, Candidate> = BTreeMap::new();
    for (layer, weight) in layers {
        let scale = strength_scale(layer);
        for &(t, s) in &layer.onsets {
            let Some(pos) = quantize(&a.grid, t) else { continue };
            let strength = weight * (s / scale).min(1.5);
            let c = by_pos.entry(pos).or_insert(Candidate {
                pos,
                strength: 0.0,
                kick: 0.0,
            });
            // Several sources agreeing on a position make it stronger.
            c.strength = c.strength.max(strength) + 0.1 * strength.min(c.strength);
        }
    }
    let kscale = strength_scale(&a.kick);
    for &(t, s) in &a.kick.onsets {
        if let Some(pos) = quantize(&a.grid, t) {
            let k = (s / kscale).min(1.5);
            let c = by_pos.entry(pos).or_insert(Candidate {
                pos,
                strength: 0.5 * k,
                kick: 0.0,
            });
            c.kick = c.kick.max(k);
        }
    }
    by_pos.into_values().collect()
}

/// Chooses the rows of a chart (no arrows yet).
pub fn place_notes(a: &SongAnalysis, model: &Model, d: Difficulty, rng: &mut ChaCha8Rng) -> Vec<Note> {
    place_notes_scaled(a, model, d, 1.0, rng)
}

/// [`place_notes`] with the number of rows multiplied by `density` (style — snaps,
/// jumps, holds — unchanged). Used to aim at a given meter.
pub fn place_notes_scaled(
    a: &SongAnalysis,
    model: &Model,
    d: Difficulty,
    density: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<Note> {
    let stats = model.stats(d);
    let grid = &a.grid;
    let (start, end) = active_range(&a.mix, a.duration);
    let in_range = |pos: u32| {
        let t = grid.time(pos as f64 / ROWS_PER_BEAT as f64);
        t >= start - 0.05 && t <= end
    };
    let mut cands: Vec<Candidate> = candidates(a).into_iter().filter(|c| in_range(c.pos)).collect();

    // Steady quarter notes as a fallback layer, weighted by loudness, so that sparse
    // charts keep a pulse even where onsets are weak.
    let mut rms_sorted = a.mix.rms.clone();
    rms_sorted.sort_by(f32::total_cmp);
    let loud = rms_sorted
        .get(rms_sorted.len().saturating_sub(1) * 95 / 100)
        .copied()
        .unwrap_or(1.0)
        .max(1e-6);
    let known: BTreeSet<u32> = cands.iter().map(|c| c.pos).collect();
    let first_beat = (grid.beat(start).ceil().max(0.0)) as u32;
    let last_beat = grid.beat(end).floor().max(0.0) as u32;
    for b in first_beat..=last_beat {
        let pos = b * ROWS_PER_BEAT;
        if !known.contains(&pos) {
            let level = layer_rms_at(&a.mix, grid.time(b as f64)) / loud;
            cands.push(Candidate {
                pos,
                strength: 0.25 * level.min(1.0),
                kick: 0.0,
            });
        }
    }

    // Target number of rows: the typical density of this difficulty, modulated by how
    // busy the song is compared to a typical one.
    let active = (end - start).max(1.0);
    let onset_rate = a.mix.onsets.iter().filter(|o| o.0 >= start && o.0 <= end).count() as f64 / active;
    let intensity = (onset_rate / 5.0).sqrt().clamp(0.8, 1.25);
    let target = (stats.nps.p50 * intensity * active * density).round().max(4.0) as usize;

    // Allowed subdivisions and their weights come from the human charts.
    let snaps = stats.snaps;
    let class_step = [48u32, 24, 16, 12, 8, 1];
    let allowed = |c: usize| c < 4 && snaps[c] >= 0.01;
    let finest = (0..4)
        .filter(|&c| allowed(c))
        .map(|c| class_step[c])
        .min()
        .unwrap_or(48);
    let weight = |c: usize| {
        if allowed(c) {
            (snaps[c] / snaps[0].max(1e-9)).sqrt() as f32
        } else {
            0.0
        }
    };

    // (score for ranking, position, candidate, P(human row) when the learned model is used)
    let mut scored: Vec<(f32, u32, Candidate, f32)> = match crate::placement::embedded_model() {
        // Learned placement: every allowed 12th/16th position of the active range, ranked
        // by P(a human puts a row here) (see `placement.rs`, measured with
        // `examples/eval_placement.rs`). Onset candidates keep their strength and kick
        // for jumps and holds.
        Some(learned) => {
            let features = crate::placement::PlacementFeatures::new(a);
            let by_pos: BTreeMap<u32, Candidate> = cands.iter().map(|c| (c.pos, *c)).collect();
            let first = (grid.beat(start - 0.05) * ROWS_PER_BEAT as f64).ceil().max(0.0) as u32;
            let last = (grid.beat(end) * ROWS_PER_BEAT as f64).floor().max(0.0) as u32;
            (first..=last)
                .filter(|&pos| crate::placement::is_candidate(pos) && weight(snap_class(pos)) > 0.0)
                .map(|pos| {
                    let c = by_pos.get(&pos).copied().unwrap_or(Candidate {
                        pos,
                        strength: 0.0,
                        kick: 0.0,
                    });
                    let p = learned.prob(d, &features.features(pos)) as f32;
                    let jitter = rng.gen_range(0.85..1.15f32);
                    (p * jitter, pos, c, p)
                })
                .collect()
        }
        None => cands
            .iter()
            .filter_map(|c| {
                let w = weight(snap_class(c.pos));
                (w > 0.0).then(|| {
                    let jitter = rng.gen_range(0.85..1.15f32);
                    (c.strength.max(0.3 * c.kick) * w * jitter, c.pos, *c, 0.0)
                })
            })
            .collect(),
    };
    // With the learned model, the expected number of human rows is the sum of the
    // probabilities (the model is calibrated: fitted by log loss).
    let target = match crate::placement::embedded_model() {
        Some(_) => {
            let expected: f64 = scored.iter().map(|s| s.3 as f64).sum();
            (expected * density).round().max(4.0) as usize
        }
        None => target,
    };
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

    const MEASURE: u32 = 4 * ROWS_PER_BEAT;
    let learned = crate::placement::embedded_model().is_some();
    // Spread the rows over the song like a human would: each measure gets a share of
    // the target proportional to its expected number of human rows (sum of the
    // probabilities), by largest remainder; the best positions of the measure fill it.
    let quota: Option<BTreeMap<u32, usize>> = learned.then(|| {
        let quota_rows = (target as f64 * MEASURE_QUOTA_SHARE).round() as usize;
        let mut expected: BTreeMap<u32, f64> = BTreeMap::new();
        for s in &scored {
            *expected.entry(s.1 / MEASURE).or_default() += s.3 as f64;
        }
        let total: f64 = expected.values().sum::<f64>().max(1e-9);
        let exact: Vec<(u32, f64)> = expected
            .iter()
            .map(|(m, e)| (*m, e / total * quota_rows as f64))
            .collect();
        let mut quota: BTreeMap<u32, usize> = exact.iter().map(|(m, x)| (*m, x.floor() as usize)).collect();
        let mut remainders: Vec<(f64, u32)> = exact.iter().map(|(m, x)| (x - x.floor(), *m)).collect();
        remainders.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let missing = quota_rows.saturating_sub(quota.values().sum());
        for (_, m) in remainders.iter().take(missing) {
            *quota.get_mut(m).unwrap() += 1;
        }
        quota
    });
    let mut chosen = select_rows(&scored, quota.as_ref(), target, finest);
    let reuse_bonus = RHYTHM_REUSE_BONUS[d.index()];
    if learned && reuse_bonus > 0.0 {
        // Second pass: favour, in each measure that sounds like an earlier one, the
        // rhythm chosen for that earlier measure (humans often, not always, repeat it).
        let last = scored.iter().map(|s| s.1).max().unwrap_or(0) / MEASURE;
        let similar = similar_earlier_measures(&measure_fingerprints(a, last), 0.9);
        let mut echo: BTreeSet<u32> = BTreeSet::new();
        for (m, r) in similar.iter().enumerate() {
            if let Some(r) = r {
                for p in chosen.range(r * MEASURE..(r + 1) * MEASURE).map(|(p, _)| *p) {
                    echo.insert(p - r * MEASURE + m as u32 * MEASURE);
                }
            }
        }
        for s in scored.iter_mut() {
            if echo.contains(&s.1) {
                s.0 += reuse_bonus;
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        chosen = select_rows(&scored, quota.as_ref(), target, finest);
    }
    let mut notes: Vec<Note> = chosen
        .values()
        .map(|c| Note {
            pos: c.pos,
            strength: c.strength,
            jump: false,
            hold: None,
        })
        .collect();
    let kicks: HashMap<u32, f32> = chosen.values().map(|c| (c.pos, c.kick)).collect();

    // Jumps: on the strongest kicks, with room around them.
    let n = notes.len();
    let jumps = (stats.jump_ratio.p50 * n as f64).round() as usize;
    let room = (2 * finest).max(24);
    let mut jump_cands: Vec<(f32, usize)> = (0..n)
        .filter(|&i| {
            let before = if i > 0 {
                notes[i].pos - notes[i - 1].pos
            } else {
                u32::MAX
            };
            let after = if i + 1 < n {
                notes[i + 1].pos - notes[i].pos
            } else {
                u32::MAX
            };
            before >= room && after >= room && snap_class(notes[i].pos) <= 1
        })
        .map(|i| {
            (
                kicks[&notes[i].pos] * (0.5 + notes[i].strength) * rng.gen_range(0.9..1.1f32),
                i,
            )
        })
        .collect();
    jump_cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    for &(_, i) in jump_cands.iter().take(jumps) {
        notes[i].jump = true;
    }

    // Holds: on notes followed by at least a beat of silence in the chart, preferring
    // sustained sound; they end before the next note.
    let holds = (stats.hold_ratio.p50 * n as f64).round() as usize;
    let typical = (stats.hold_beats.p50.max(0.5) * ROWS_PER_BEAT as f64) as u32;
    let sustain_layer = a.stems.as_ref().map_or(&a.mix, |s| &s.other);
    let mut hold_cands: Vec<(f32, usize, u32)> = (0..n.saturating_sub(1))
        .filter(|&i| !notes[i].jump && notes[i + 1].pos - notes[i].pos >= ROWS_PER_BEAT)
        .map(|i| {
            let gap = notes[i + 1].pos - notes[i].pos;
            let len = (gap - 12).min(2 * typical).max(24) / 12 * 12;
            let t0 = grid.time(notes[i].pos as f64 / ROWS_PER_BEAT as f64);
            let t1 = grid.time((notes[i].pos + len) as f64 / ROWS_PER_BEAT as f64);
            let sustain = (layer_rms_at(sustain_layer, (t0 + t1) / 2.0)
                / layer_rms_at(sustain_layer, t0).max(1e-6))
            .min(1.0);
            (notes[i].strength * sustain * rng.gen_range(0.9..1.1f32), i, len)
        })
        .collect();
    hold_cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    for &(_, i, len) in hold_cands.iter().take(holds) {
        notes[i].hold = Some(len);
    }
    notes
}

/// For each note, the index of the matching note in an earlier similar measure.
/// Audio fingerprint of measures `0..=last`: onset strength on the 16 sixteenths of
/// the measure, for the mix, the kick and the stems when present.
pub fn measure_fingerprints(a: &SongAnalysis, last: u32) -> Vec<Vec<f32>> {
    let mut layers: Vec<&Layer> = vec![&a.mix, &a.kick];
    if let Some(s) = &a.stems {
        layers.extend([&s.drums, &s.bass, &s.vocals, &s.other]);
    }
    (0..=last)
        .map(|m| {
            let mut v = Vec::new();
            for l in &layers {
                for k in 0..16 {
                    let t = a.grid.time(m as f64 * 4.0 + k as f64 / 4.0);
                    v.push(l.envelope.max_around(t, 0.03));
                }
            }
            v
        })
        .collect()
}

pub fn cosine(x: &[f32], y: &[f32]) -> f64 {
    let dot: f64 = x.iter().zip(y).map(|(a, b)| (*a * *b) as f64).sum();
    let nx: f64 = x.iter().map(|a| (*a * *a) as f64).sum::<f64>().sqrt();
    let ny: f64 = y.iter().map(|a| (*a * *a) as f64).sum::<f64>().sqrt();
    if nx == 0.0 || ny == 0.0 {
        0.0
    } else {
        dot / (nx * ny)
    }
}

/// For each measure, the most similar of the 64 previous measures whose fingerprint
/// has a cosine similarity of at least `min_similarity` (ties: the latest).
pub fn similar_earlier_measures(fingerprints: &[Vec<f32>], min_similarity: f64) -> Vec<Option<u32>> {
    (0..fingerprints.len())
        .map(|m| {
            let mut best: Option<(f64, u32)> = None;
            for p in (m.saturating_sub(64)..m).rev() {
                let sim = cosine(&fingerprints[m], &fingerprints[p]);
                if sim >= min_similarity && best.is_none_or(|b| sim > b.0) {
                    best = Some((sim, p as u32));
                }
            }
            best.map(|b| b.1)
        })
        .collect()
}

pub fn find_repeats(a: &SongAnalysis, notes: &[Note], min_similarity: f64) -> Vec<Option<usize>> {
    let measure_of = |pos: u32| pos / (4 * ROWS_PER_BEAT);
    let last = notes.last().map_or(0, |n| measure_of(n.pos));
    let features = measure_fingerprints(a, last);
    // Rhythm signature of each measure: relative positions + jump/hold flags.
    let mut by_measure: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, n) in notes.iter().enumerate() {
        by_measure.entry(measure_of(n.pos)).or_default().push(i);
    }
    let signature = |idx: &[usize]| -> Vec<(u32, bool, bool)> {
        idx.iter()
            .map(|&i| {
                (
                    notes[i].pos % (4 * ROWS_PER_BEAT),
                    notes[i].jump,
                    notes[i].hold.is_some(),
                )
            })
            .collect()
    };
    let mut out = vec![None; notes.len()];
    let measures: Vec<(u32, Vec<usize>)> = by_measure.into_iter().collect();
    for (k, (m, idx)) in measures.iter().enumerate() {
        let sig = signature(idx);
        let mut best: Option<(f64, &Vec<usize>)> = None;
        for (pm, pidx) in measures[..k].iter().rev().take(64) {
            if signature(pidx) != sig {
                continue;
            }
            let sim = cosine(&features[*m as usize], &features[*pm as usize]);
            if sim >= min_similarity && best.is_none_or(|b| sim > b.0) {
                best = Some((sim, pidx));
            }
        }
        if let Some((_, pidx)) = best {
            for (a, b) in idx.iter().zip(pidx.iter()) {
                out[*a] = Some(*b);
            }
        }
    }
    out
}

const SINGLES: [u8; 4] = [0b0001, 0b0010, 0b0100, 0b1000];
const JUMPS: [u8; 6] = [0b1001, 0b0011, 0b0101, 0b1010, 0b1100, 0b0110];

struct Hyp {
    score: f64,
    /// Reachable parity states with their accumulated cost and the number of
    /// footswitches on their cheapest path (pruned, cheapest first).
    frontier: Vec<(State, f32, u32)>,
    p3: u8,
    p2: u8,
    p1: u8,
    /// Index in the arena of the last choice.
    node: usize,
}

/// Picks the arrows of every note.
/// Footswitch penalty (nats) of the arrow search for difficulty `d`.
pub fn footswitch_penalty(_model: &Model, _d: Difficulty, opts: &GenOptions) -> f64 {
    opts.footswitch_penalty
}

#[allow(clippy::too_many_arguments)]
pub fn select_arrows(
    notes: &[Note],
    grid: &Grid,
    table: &ProbTable,
    repeats: &[Option<usize>],
    opts: &GenOptions,
    fs_penalty: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<u8> {
    let layout = Layout::dance_single();
    // arena of (parent, mask); usize::MAX = root
    let mut arena: Vec<(usize, u8)> = Vec::new();
    let mut beam = vec![Hyp {
        score: 0.0,
        frontier: vec![(State::beginning(), 0.0, 0)],
        p3: 0,
        p2: 0,
        p1: 0,
        node: usize::MAX,
    }];
    let mut prev_row: Option<Row> = None;
    const FRONTIER: usize = 24;
    for (i, note) in notes.iter().enumerate() {
        let beat = note.pos as f64 / ROWS_PER_BEAT as f64;
        let second = grid.time(beat) as f32;
        let gap = if i == 0 {
            crate::model::GAP_BUCKETS - 1
        } else {
            gap_bucket(note.pos - notes[i - 1].pos)
        };
        let masks: &[u8] = if note.jump { &JUMPS } else { &SINGLES };
        let mut next: Vec<Hyp> = Vec::new();
        for h in &beam {
            // Arrow chosen for the reference note by *this* hypothesis.
            let reference = repeats[i].map(|j| {
                let mut node = h.node;
                for _ in 0..(i - 1 - j) {
                    node = arena[node].0;
                }
                arena[node].1
            });
            let (base_cost, base_fs) = (h.frontier[0].1, h.frontier[0].2);
            for &m in masks {
                let mut row = Row::new(4, beat as f32, second);
                for c in 0..4 {
                    if m & (1 << c) != 0 {
                        row.add_note(c);
                    }
                }
                let mut frontier: Vec<(State, f32, u32)> = Vec::new();
                let mut index: HashMap<u64, usize> = HashMap::new();
                // Arrows of the previous row (none before the first one).
                let prev_notes = if i == 0 { 0 } else { h.p1 as u16 };
                for (st, cost, fs_count) in &h.frontier {
                    // As upstream: the node before the first row is one second earlier.
                    let elapsed = prev_row.as_ref().map_or(1.0, |p| second - p.second);
                    for p in layout.placements(row.note_mask, row.hold_mask) {
                        let ns = State::result(st, &row, p);
                        let c =
                            cost + parity::action_cost(&layout, st, &ns, &row, prev_row.as_ref(), p, elapsed);
                        let fs = fs_count
                            + (fs_penalty != 0.0
                                && parity::is_footswitch(&layout, st, prev_notes, &row, p, elapsed))
                                as u32;
                        match index.get(&ns.key()) {
                            Some(&k) => {
                                if c < frontier[k].1 {
                                    frontier[k].1 = c;
                                    frontier[k].2 = fs;
                                }
                            }
                            None => {
                                index.insert(ns.key(), frontier.len());
                                frontier.push((ns, c, fs));
                            }
                        }
                    }
                }
                if frontier.is_empty() {
                    continue;
                }
                frontier.sort_by(|a, b| a.1.total_cmp(&b.1));
                frontier.truncate(FRONTIER);
                let parity_delta = (frontier[0].1 - base_cost) as f64;
                // Footswitches gained (or lost, when the cheapest path is revised) by the
                // cheapest parity path: the game counts those of the cheapest path.
                let fs_delta = frontier[0].2 as f64 - base_fs as f64;
                let u: f64 = rng.gen_range(1e-12..1.0);
                let gumbel = -(-u.ln()).ln();
                let bonus = if reference == Some(m) {
                    opts.repeat_bonus
                } else {
                    0.0
                };
                let score = h.score + table.logp(gap, h.p3, h.p2, h.p1, m) as f64
                    - opts.parity_weight * parity_delta
                    + opts.temperature * gumbel
                    + bonus
                    - fs_penalty * fs_delta;
                arena.push((h.node, m));
                next.push(Hyp {
                    score,
                    frontier,
                    p3: h.p2,
                    p2: h.p1,
                    p1: m,
                    node: arena.len() - 1,
                });
            }
        }
        next.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.node.cmp(&b.node)));
        // Keep diversity: one hypothesis per (last three arrows, best parity state).
        let mut seen = std::collections::HashSet::new();
        next.retain(|h| seen.insert((h.p3, h.p2, h.p1, h.frontier[0].0.key())));
        next.truncate(opts.beam_width);
        beam = next;
        let mut row = Row::new(4, beat as f32, second);
        row.add_note(0);
        prev_row = Some(row);
    }
    let mut out = Vec::with_capacity(notes.len());
    let mut node = beam.first().map_or(usize::MAX, |h| h.node);
    while node != usize::MAX {
        out.push(arena[node].1);
        node = arena[node].0;
    }
    out.reverse();
    out
}

/// Builds the output rows (taps, hold heads and tails).
pub fn to_rows(notes: &[Note], masks: &[u8]) -> Vec<(u32, [Cell; 4])> {
    let mut rows: BTreeMap<u32, [Cell; 4]> = BTreeMap::new();
    for (n, &m) in notes.iter().zip(masks) {
        let head = if n.hold.is_some() {
            Cell::HoldHead
        } else {
            Cell::Tap
        };
        let row = rows.entry(n.pos).or_insert([Cell::Empty; 4]);
        for (c, cell) in row.iter_mut().enumerate() {
            if m & (1 << c) != 0 {
                *cell = head;
            }
        }
        if let Some(len) = n.hold {
            let tail = rows.entry(n.pos + len).or_insert([Cell::Empty; 4]);
            for (c, cell) in tail.iter_mut().enumerate() {
                if m & (1 << c) != 0 {
                    *cell = Cell::Tail;
                }
            }
        }
    }
    rows.into_iter().collect()
}

/// Meter from the density of the generated chart, within the human range of the slot.
pub fn estimate_meter(rows: &[(u32, [Cell; 4])], grid: &Grid, model: &Model, d: Difficulty) -> u32 {
    let s = model.stats(d);
    raw_meter(rows, grid, model)
        .round()
        .clamp(s.meter.p10.max(1.0), s.meter.p90.max(1.0)) as u32
}

/// Meter predicted by the learned fit from the density of the busiest measures
/// (no clamping).
pub fn raw_meter(rows: &[(u32, [Cell; 4])], grid: &Grid, model: &Model) -> f64 {
    let note_rows: Vec<NoteRow> = rows
        .iter()
        .map(|(p, c)| NoteRow {
            beat: *p as f64 / ROWS_PER_BEAT as f64,
            cells: c.to_vec(),
        })
        .collect();
    let timing = Timing::constant(grid.bpm, grid.offset());
    let f = ChartFeatures::compute_density(&note_rows, &timing);
    model.meter_for(f)
}

/// Meters that can be requested (the range of the ITGmania difficulty scale used here).
pub const METER_RANGE: std::ops::RangeInclusive<u32> = 1..=10;

/// The difficulty whose human charts are typically closest to `meter` (ties: the
/// easier one). Its style (snaps, jumps, holds, arrow patterns) is used for the chart.
pub fn style_for_meter(model: &Model, meter: u32) -> Difficulty {
    let mut best = Difficulty::Beginner;
    for d in Difficulty::ALL {
        let dist = |d: Difficulty| (model.stats(d).meter.p50 - meter as f64).abs();
        if dist(d) < dist(best) - 1e-9 {
            best = d;
        }
    }
    best
}

/// Puts each requested meter in its own difficulty slot (the game shows one chart per
/// slot), in increasing order, preferring the slot whose human charts have the closest
/// meter. At most 5 meters.
pub fn assign_slots(model: &Model, meters: &[u32]) -> anyhow::Result<Vec<(Difficulty, u32)>> {
    let mut meters = meters.to_vec();
    meters.sort();
    meters.dedup();
    anyhow::ensure!(!meters.is_empty(), "no meter requested");
    anyhow::ensure!(
        meters.len() <= Difficulty::ALL.len(),
        "at most {} meters (one per difficulty slot), got {}",
        Difficulty::ALL.len(),
        meters.len()
    );
    if let Some(m) = meters.iter().find(|m| !METER_RANGE.contains(m)) {
        anyhow::bail!(
            "meter {m} out of range {}..={}",
            METER_RANGE.start(),
            METER_RANGE.end()
        );
    }
    let n = meters.len();
    let mut out = Vec::with_capacity(n);
    let mut next = 0usize;
    for (i, &m) in meters.iter().enumerate() {
        let preferred = style_for_meter(model, m).index();
        // Leave enough slots for the remaining (harder) meters.
        let latest = Difficulty::ALL.len() - (n - i);
        let slot = preferred.clamp(next, latest);
        out.push((Difficulty::ALL[slot], m));
        next = slot + 1;
    }
    Ok(out)
}

/// Generates a chart aiming at `meter`: the style of [`style_for_meter`], and a
/// density found by bisection so that the busiest measures are as dense as in human
/// charts rated `meter` ([`Model::density_for_meter`]). The chart is labelled with
/// the `slot` difficulty. The written meter is the target when it is reached, else
/// the closest one (e.g. a calm song cannot reach a high meter).
pub fn generate_for_meter(
    a: &SongAnalysis,
    model: &Model,
    slot: Difficulty,
    meter: u32,
    seed: u64,
    opts: &GenOptions,
) -> OutChart {
    let style = style_for_meter(model, meter);
    let salt = slot.seed_salt() ^ (meter as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    let base = ChaCha8Rng::seed_from_u64(seed ^ salt);
    // The meter only depends on where the rows are, not on the arrows: search on the
    // placement alone, each try from the same RNG state (deterministic).
    let timing = Timing::constant(a.grid.bpm, a.grid.offset());
    let place = |density: f64| {
        let mut rng = base.clone();
        let notes = place_notes_scaled(a, model, style, density, &mut rng);
        let rows: Vec<NoteRow> = notes
            .iter()
            .map(|n| NoteRow {
                beat: n.pos as f64 / ROWS_PER_BEAT as f64,
                cells: vec![Cell::Tap],
            })
            .collect();
        let d = ChartFeatures::compute_density(&rows, &timing);
        (d, notes, rng)
    };
    let target = model.density_for_meter(meter);
    let (mut lo, mut hi) = (0.02f64, 6.0f64);
    let mut best = place(1.0);
    for _ in 0..14 {
        let mid = (lo * hi).sqrt();
        let tried = place(mid);
        if (tried.0 - target).abs() < (best.0 - target).abs() {
            best = tried.clone();
        }
        if tried.0 < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (density, notes, mut rng) = best;
    let repeats = find_repeats(a, &notes, opts.repeat_similarity);
    let table = model.table(style);
    let fs_penalty = footswitch_penalty(model, style, opts);
    let masks = select_arrows(&notes, &a.grid, &table, &repeats, opts, fs_penalty, &mut rng);
    let rows = to_rows(&notes, &masks);
    let written = model.meter_for_density(density);
    OutChart {
        difficulty: slot.name().to_string(),
        meter: written,
        description: format!("itg-charter seed {seed}"),
        rows,
    }
}

/// Generates one chart. The RNG is derived from the seed and the difficulty only, so a
/// chart does not depend on which other difficulties are generated.
pub fn generate(a: &SongAnalysis, model: &Model, d: Difficulty, seed: u64, opts: &GenOptions) -> OutChart {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ d.seed_salt());
    let notes = place_notes(a, model, d, &mut rng);
    let repeats = find_repeats(a, &notes, opts.repeat_similarity);
    let table = model.table(d);
    let fs_penalty = footswitch_penalty(model, d, opts);
    let masks = select_arrows(&notes, &a.grid, &table, &repeats, opts, fs_penalty, &mut rng);
    let rows = to_rows(&notes, &masks);
    let meter = estimate_meter(&rows, &a.grid, model, d);
    OutChart {
        difficulty: d.name().to_string(),
        meter,
        description: format!("itg-charter seed {seed}"),
        rows,
    }
}
