//! Learned note placement: P(a human charter puts a row here | what we hear here).
//!
//! Every position of the 4th/8th/12th/16th grid is described by a few audio and
//! metric features ([`PlacementFeatures`]); a logistic regression per difficulty,
//! fitted on human charts of the train split (`examples/fit_placement.rs`), turns them
//! into a probability. The generator ranks candidate positions by this probability;
//! how many rows it keeps is still decided by the density target.

use crate::analysis::{Envelope, Grid, SongAnalysis};
use crate::chart::quantize;
use crate::difficulty::Difficulty;
use crate::model::snap_class;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PLACEMENT_VERSION: u32 = 2;
/// snap (4) + beat in bar (4) + mix + 4 bands + loudness + local density + salience + onset
/// + section context (measure / phrase loudness, kick band, onset count, change, position)
pub const FEATURES: usize = 23;
pub const FEATURE_NAMES: [&str; FEATURES] = [
    "snap 4th",
    "snap 8th",
    "snap 12th",
    "snap 16th",
    "beat 1",
    "beat 2",
    "beat 3",
    "beat 4",
    "mix onset",
    "kick",
    "low-mid",
    "mid",
    "high",
    "loudness",
    "local density",
    "salience",
    "onset here",
    "measure loudness",
    "phrase loudness",
    "measure kick",
    "measure onsets",
    "section change",
    "song position",
];

/// Model shipped with the binary.
pub const EMBEDDED: &str = include_str!("../model/placement.json");

/// Grid positions considered as candidates (48ths of a beat): 12ths and 16ths.
pub fn is_candidate(pos: u32) -> bool {
    pos.is_multiple_of(12) || pos.is_multiple_of(16)
}

fn percentile(v: &[f32], p: f64) -> f32 {
    let mut s: Vec<f32> = v.iter().copied().filter(|x| *x > 0.0).collect();
    if s.is_empty() {
        return 1.0;
    }
    s.sort_by(f32::total_cmp);
    s[((s.len() - 1) as f64 * p) as usize].max(1e-9)
}

/// Per-song normalization and lookups to compute features at any grid position.
pub struct PlacementFeatures<'a> {
    grid: Grid,
    mix: &'a Envelope,
    mix_ref: f32,
    bands: Vec<(&'a Envelope, f32)>,
    rms: &'a [f32],
    rms_fps: f64,
    rms_ref: f32,
    onsets: BTreeSet<u32>,
    /// Section context per measure (see [`section_context`]).
    measures: Vec<[f32; 5]>,
    duration: f64,
}

/// Measure-level context, from the mix only (so that it is the same with and without
/// stems): loudness of the measure and of its 8-measure phrase, kick-band energy and
/// onset count, each relative to the song's median measure (capped at 3), and how much
/// the 4 measures from here differ from the 4 before (1 - cosine of their onset
/// fingerprints), which is high at section boundaries.
fn section_context(a: &SongAnalysis) -> Vec<[f32; 5]> {
    let grid = &a.grid;
    let n = (grid.beat(a.duration) / 4.0).ceil().max(1.0) as usize;
    let span = |m: usize| (grid.time(4.0 * m as f64), grid.time(4.0 * (m + 1) as f64));
    let fps = a.mix.envelope.fps;
    let mean = |v: &[f32], m: usize| -> f32 {
        let (t0, t1) = span(m);
        let (i0, i1) = ((t0 * fps).max(0.0) as usize, (t1 * fps).max(0.0) as usize);
        let s = v.get(i0.min(v.len())..i1.min(v.len())).unwrap_or(&[]);
        if s.is_empty() {
            0.0
        } else {
            s.iter().sum::<f32>() / s.len() as f32
        }
    };
    let relative = |v: Vec<f32>| -> Vec<f32> {
        let r = percentile(&v, 0.5);
        v.into_iter().map(|x| (x / r).min(3.0)).collect()
    };
    let loud = relative((0..n).map(|m| mean(&a.mix.rms, m)).collect());
    let empty = Vec::new();
    let kick = relative(
        (0..n)
            .map(|m| mean(a.bands.first().map_or(&empty, |e| &e.values), m))
            .collect(),
    );
    let mut counts = vec![0f32; n];
    for &(t, _) in &a.mix.onsets {
        let m = (grid.beat(t) / 4.0).floor();
        if m >= 0.0 && (m as usize) < n {
            counts[m as usize] += 1.0;
        }
    }
    let counts = relative(counts);
    let fingerprint: Vec<Vec<f32>> = (0..n)
        .map(|m| {
            let mut v = Vec::with_capacity(16 * (1 + a.bands.len()));
            for env in std::iter::once(&a.mix.envelope).chain(&a.bands) {
                for k in 0..16 {
                    v.push(env.max_around(grid.time(4.0 * m as f64 + k as f64 / 4.0), 0.03));
                }
            }
            v
        })
        .collect();
    let sum = |r: std::ops::Range<usize>| -> Vec<f32> {
        let mut out = vec![0f32; fingerprint[0].len()];
        for f in &fingerprint[r] {
            for (o, x) in out.iter_mut().zip(f) {
                *o += x;
            }
        }
        out
    };
    (0..n)
        .map(|m| {
            let phrase = m.saturating_sub(3)..(m + 5).min(n);
            let phrase_loud = loud[phrase.clone()].iter().sum::<f32>() / phrase.len() as f32;
            let change = if m == 0 {
                1.0
            } else {
                1.0 - crate::chart::cosine(&sum(m.saturating_sub(4)..m), &sum(m..(m + 4).min(n))) as f32
            };
            [loud[m], phrase_loud, kick[m], counts[m], change]
        })
        .collect()
}

impl<'a> PlacementFeatures<'a> {
    pub fn new(a: &'a SongAnalysis) -> PlacementFeatures<'a> {
        let onsets = a
            .mix
            .onsets
            .iter()
            .filter_map(|&(t, _)| quantize(&a.grid, t))
            .collect();
        PlacementFeatures {
            grid: a.grid,
            mix: &a.mix.envelope,
            mix_ref: percentile(&a.mix.envelope.values, 0.9),
            bands: a.bands.iter().map(|e| (e, percentile(&e.values, 0.9))).collect(),
            rms: &a.mix.rms,
            rms_fps: a.mix.envelope.fps,
            rms_ref: percentile(&a.mix.rms, 0.95),
            onsets,
            measures: section_context(a),
            duration: a.duration.max(1.0),
        }
    }

    fn strength(&self, env: &Envelope, reference: f32, pos: u32) -> f32 {
        let t = self.grid.time(pos as f64 / 48.0);
        (1.0 + env.max_around(t, 0.03) / reference).ln()
    }

    pub fn features(&self, pos: u32) -> [f32; FEATURES] {
        let mut f = [0f32; FEATURES];
        let snap = match snap_class(pos) {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 3,
        };
        f[snap] = 1.0;
        f[4 + ((pos / 48) % 4) as usize] = 1.0;
        let mix = self.strength(self.mix, self.mix_ref, pos);
        f[8] = mix;
        for (k, (env, r)) in self.bands.iter().enumerate().take(4) {
            f[9 + k] = self.strength(env, *r, pos);
        }
        let t = self.grid.time(pos as f64 / 48.0);
        let i = (t * self.rms_fps).max(0.0) as usize;
        f[13] = self.rms.get(i).copied().unwrap_or(0.0) / self.rms_ref;
        // Mean onset strength over the surrounding beat (8 sixteenths), and how much this
        // position stands out from its two 16th-note neighbours.
        let around: Vec<f32> = (-4i32..=4)
            .filter(|k| *k != 0)
            .map(|k| self.strength(self.mix, self.mix_ref, (pos as i32 + 12 * k).max(0) as u32))
            .collect();
        f[14] = around.iter().sum::<f32>() / around.len() as f32;
        let neighbours = (around[3] + around[4]) / 2.0;
        f[15] = mix - neighbours;
        f[16] = self.onsets.contains(&pos) as u8 as f32;
        let m = ((pos / 192) as usize).min(self.measures.len() - 1);
        f[17..22].copy_from_slice(&self.measures[m]);
        f[22] = (t / self.duration).clamp(0.0, 1.0) as f32;
        f
    }
}

/// Logistic regressions, one per difficulty: weights[d] = [bias, w_1 … w_FEATURES].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlacementModel {
    pub version: u32,
    pub weights: Vec<Vec<f64>>,
    /// Log loss / accuracy on the train and test splits when fitted, for the record.
    pub report: String,
}

/// The embedded model, loaded once; `None` if it has no weights (not fitted yet).
pub fn embedded_model() -> Option<&'static PlacementModel> {
    static MODEL: std::sync::OnceLock<Option<PlacementModel>> = std::sync::OnceLock::new();
    MODEL
        .get_or_init(|| PlacementModel::embedded().ok().filter(|m| m.weights.len() == 5))
        .as_ref()
}

impl PlacementModel {
    pub fn embedded() -> Result<PlacementModel> {
        let m: PlacementModel = serde_json::from_str(EMBEDDED).context("parsing placement model")?;
        anyhow::ensure!(
            m.version == PLACEMENT_VERSION,
            "placement model version {}",
            m.version
        );
        Ok(m)
    }

    pub fn prob(&self, d: Difficulty, f: &[f32; FEATURES]) -> f64 {
        logistic(&self.weights[d.index()], f)
    }
}

pub fn logistic(w: &[f64], f: &[f32; FEATURES]) -> f64 {
    let z = w[0] + f.iter().zip(&w[1..]).map(|(x, w)| *x as f64 * w).sum::<f64>();
    1.0 / (1.0 + (-z).exp())
}

/// Fits a logistic regression by Newton's method (deterministic), with an L2 penalty
/// `l2` on the weights (not the bias). Returns [bias, weights…].
#[allow(clippy::needless_range_loop)] // matrix algebra reads best with indices
pub fn fit_logistic(xs: &[[f32; FEATURES]], ys: &[bool], l2: f64) -> Vec<f64> {
    const N: usize = FEATURES + 1;
    let mut w = vec![0.0; N];
    for _ in 0..25 {
        let mut grad = [0.0f64; N];
        let mut hess = [[0.0f64; N]; N];
        for (x, &y) in xs.iter().zip(ys) {
            let p = logistic(&w, x);
            let err = p - y as u8 as f64;
            let s = p * (1.0 - p);
            let mut v = [0.0f64; N];
            v[0] = 1.0;
            for k in 0..FEATURES {
                v[k + 1] = x[k] as f64;
            }
            for a in 0..N {
                grad[a] += err * v[a];
                for b in a..N {
                    hess[a][b] += s * v[a] * v[b];
                }
            }
        }
        for a in 0..N {
            for b in 0..a {
                hess[a][b] = hess[b][a];
            }
            if a > 0 {
                grad[a] += l2 * w[a];
                hess[a][a] += l2;
            }
            hess[a][a] += 1e-9;
        }
        let step = solve(hess, grad);
        let mut change = 0.0f64;
        for a in 0..N {
            w[a] -= step[a];
            change = change.max(step[a].abs());
        }
        if change < 1e-7 {
            break;
        }
    }
    w
}

/// Solves `m x = b` by Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)] // matrix algebra reads best with indices
fn solve<const N: usize>(mut m: [[f64; N]; N], mut b: [f64; N]) -> [f64; N] {
    for col in 0..N {
        let piv = (col..N)
            .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
            .unwrap();
        m.swap(col, piv);
        b.swap(col, piv);
        let d = m[col][col];
        if d.abs() < 1e-300 {
            continue;
        }
        for row in col + 1..N {
            let factor = m[row][col] / d;
            for k in col..N {
                m[row][k] -= factor * m[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let s: f64 = (row + 1..N).map(|k| m[row][k] * x[k]).sum();
        x[row] = if m[row][row].abs() < 1e-300 {
            0.0
        } else {
            (b[row] - s) / m[row][row]
        };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_recovers_a_known_logistic() {
        // y = 1 iff feature 8 (mix onset) > 0.5, with a little noise-free margin.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..400 {
            let mut f = [0f32; FEATURES];
            f[8] = (i % 20) as f32 / 20.0;
            f[0] = 1.0;
            xs.push(f);
            ys.push(f[8] > 0.5);
        }
        let w = fit_logistic(&xs, &ys, 1e-3);
        let acc = xs
            .iter()
            .zip(&ys)
            .filter(|(x, y)| (logistic(&w, x) > 0.5) == **y)
            .count();
        assert!(acc >= 390, "accuracy {acc}/400, weights {w:?}");
        assert!(w[9] > 0.0);
    }

    #[test]
    fn embedded_model_is_current() {
        // Otherwise the generator silently falls back to onset-strength placement.
        let m = PlacementModel::embedded().expect("model/placement.json matches PLACEMENT_VERSION");
        assert_eq!(m.weights.len(), 5);
        assert!(m.weights.iter().all(|w| w.len() == FEATURES + 1));
    }

    #[test]
    fn candidates_are_12ths_and_16ths() {
        assert!(is_candidate(0) && is_candidate(12) && is_candidate(16) && is_candidate(24));
        assert!(!is_candidate(8) && !is_candidate(6));
    }
}
