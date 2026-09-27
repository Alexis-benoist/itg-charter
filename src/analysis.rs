//! Audio analysis: onset envelopes and onsets (aubio), tempo and beat phase.
//!
//! aubio provides the signal processing (spectral-flux onset detection function,
//! peak picking, beat tracking). On top of it we fit a *constant* BPM and a phase
//! over the whole song, because a simfile needs one exact grid rather than a
//! list of beat times.

use crate::audio::Audio;
use aubio_rs::{Onset, OnsetMode, Tempo};

/// aubio phase-vocoder window and hop, in samples.
pub const WIN: usize = 1024;
pub const HOP: usize = 256;

/// Delay between an attack and the peak of the onset functions, in seconds,
/// subtracted from envelope times. Calibrated with `examples/eval_sync.rs` on the
/// human-synced ITGmania library: without it, detected beats land a median 17 ms
/// after the charts' beats (tight spread: 80% within 7–25 ms). This includes the
/// detector latency and the sync convention of the packs.
pub const ENVELOPE_LATENCY: f64 = 0.017;

/// An onset detection function sampled once per hop.
#[derive(Clone, Debug, Default)]
pub struct Envelope {
    /// Frames per second.
    pub fps: f64,
    pub values: Vec<f32>,
}

impl Envelope {
    /// Time of frame `i`: aubio's frame `i` covers the samples up to `(i + 1) * HOP`.
    pub fn time(&self, i: usize) -> f64 {
        (i + 1) as f64 / self.fps - ENVELOPE_LATENCY
    }

    fn index(&self, t: f64) -> f64 {
        (t + ENVELOPE_LATENCY) * self.fps - 1.0
    }

    /// Linear interpolation at time `t` (0 outside).
    pub fn at(&self, t: f64) -> f32 {
        let x = self.index(t);
        if x < 0.0 || x >= (self.values.len() - 1) as f64 {
            return 0.0;
        }
        let i = x.floor() as usize;
        let f = (x - i as f64) as f32;
        self.values[i] * (1.0 - f) + self.values[i + 1] * f
    }

    /// Maximum over `[t - w, t + w]`.
    pub fn max_around(&self, t: f64, w: f64) -> f32 {
        let a = self.index(t - w).ceil().max(0.0) as usize;
        let b = (self.index(t + w).floor().max(-1.0) + 1.0) as usize;
        self.values
            .get(a..b.min(self.values.len()))
            .and_then(|s| s.iter().copied().reduce(f32::max))
            .unwrap_or(0.0)
    }

    pub fn duration(&self) -> f64 {
        self.values.len() as f64 / self.fps
    }

    /// Returns a copy where every value is the max over ±`radius` frames.
    pub fn dilated(&self, radius: usize) -> Envelope {
        let n = self.values.len();
        let values = (0..n)
            .map(|i| {
                self.values[i.saturating_sub(radius)..(i + radius + 1).min(n)]
                    .iter()
                    .copied()
                    .fold(0.0, f32::max)
            })
            .collect();
        Envelope {
            fps: self.fps,
            values,
        }
    }

    pub fn mean(&self) -> f64 {
        self.values.iter().map(|v| *v as f64).sum::<f64>() / self.values.len().max(1) as f64
    }

    pub fn std(&self) -> f64 {
        let m = self.mean();
        (self.values.iter().map(|v| (*v as f64 - m).powi(2)).sum::<f64>() / self.values.len().max(1) as f64)
            .sqrt()
    }
}

/// Onset information for one signal (full mix, or one stem).
#[derive(Clone, Debug, Default)]
pub struct Layer {
    pub envelope: Envelope,
    /// Detected onsets: (time in seconds, strength of the onset function).
    pub onsets: Vec<(f64, f32)>,
    /// RMS level per hop.
    pub rms: Vec<f32>,
}

impl Layer {
    pub fn analyze(samples: &[f32], sample_rate: u32) -> Layer {
        let mut onset = Onset::new(OnsetMode::SpecFlux, WIN, HOP, sample_rate)
            .expect("aubio onset")
            .with_threshold(0.2)
            .with_minioi_ms(40.0);
        let mut layer = Layer {
            envelope: Envelope {
                fps: sample_rate as f64 / HOP as f64,
                values: Vec::with_capacity(samples.len() / HOP + 1),
            },
            ..Layer::default()
        };
        let mut chunk = [0f32; HOP];
        for block in samples.chunks(HOP) {
            chunk[..block.len()].copy_from_slice(block);
            chunk[block.len()..].fill(0.0);
            let detected = onset.do_result(chunk).expect("aubio onset frame") > 0.0;
            let d = onset.get_descriptor();
            layer.envelope.values.push(d);
            if detected {
                layer.onsets.push((onset.get_last_s() as f64, d));
            }
            let e: f32 = chunk.iter().map(|s| s * s).sum::<f32>() / HOP as f32;
            layer.rms.push(e.sqrt());
        }
        // aubio reports the peak a few frames late; use the envelope maximum near it.
        let env = layer.envelope.clone();
        for o in &mut layer.onsets {
            o.1 = o.1.max(env.max_around(o.0, 0.03));
        }
        layer
    }
}

/// Low-frequency (kick) layer: positive flux of the RMS energy of the low-passed
/// signal. Spectral flux is a poor kick detector because broadband noise (snares)
/// still adds up across bins after filtering; plain energy is not fooled.
pub fn kick_layer(samples: &[f32], sample_rate: u32) -> Layer {
    let low = low_pass(samples, sample_rate, 150.0);
    let fps = sample_rate as f64 / HOP as f64;
    let n = low.len() / HOP + 1;
    let mut rms = Vec::with_capacity(n);
    for i in 0..n {
        let end = ((i + 1) * HOP).min(low.len());
        let start = end.saturating_sub(WIN / 2);
        let e: f32 = low[start..end].iter().map(|x| x * x).sum::<f32>() / (end - start).max(1) as f32;
        rms.push(e.sqrt());
    }
    let values: Vec<f32> = (0..n)
        .map(|i| {
            if i == 0 {
                0.0
            } else {
                (rms[i] - rms[i - 1]).max(0.0)
            }
        })
        .collect();
    let envelope = Envelope { fps, values };
    let onsets = pick_peaks(&envelope, 0.06);
    Layer {
        envelope,
        onsets,
        rms,
    }
}

/// Local maxima above mean + 1 std, at least `min_gap` seconds apart.
pub fn pick_peaks(env: &Envelope, min_gap: f64) -> Vec<(f64, f32)> {
    let threshold = (env.mean() + env.std()) as f32;
    let radius = (min_gap * env.fps).ceil() as usize;
    let v = &env.values;
    let mut out = Vec::new();
    for i in 0..v.len() {
        if v[i] <= threshold {
            continue;
        }
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(v.len());
        // strict maximum on the left, non-strict on the right: one peak per plateau
        if v[lo..i].iter().all(|x| *x < v[i]) && v[i + 1..hi].iter().all(|x| *x <= v[i]) {
            out.push((env.time(i), v[i]));
        }
    }
    out
}

/// Simple one-pole low-pass filter (for the kick band).
pub fn low_pass(samples: &[f32], sample_rate: u32, cutoff_hz: f32) -> Vec<f32> {
    let dt = 1.0 / sample_rate as f32;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let a = dt / (rc + dt);
    let mut y = 0.0;
    // Two passes for a steeper slope.
    let first: Vec<f32> = samples
        .iter()
        .map(|&x| {
            y += a * (x - y);
            y
        })
        .collect();
    y = 0.0;
    first
        .iter()
        .map(|&x| {
            y += a * (x - y);
            y
        })
        .collect()
}

/// aubio's beat tracker: returns beat times in seconds.
pub fn aubio_beats(samples: &[f32], sample_rate: u32) -> Vec<f64> {
    let hop = 512;
    let mut tempo = Tempo::new(OnsetMode::SpecFlux, 1024, hop, sample_rate).expect("aubio tempo");
    let mut beats = Vec::new();
    let mut chunk = vec![0f32; hop];
    for block in samples.chunks(hop) {
        chunk[..block.len()].copy_from_slice(block);
        chunk[block.len()..].fill(0.0);
        if tempo.do_result(&chunk).expect("aubio tempo frame") > 0.0 {
            beats.push(tempo.get_last_s() as f64);
        }
    }
    beats
}

/// Constant tempo grid: beat `k` happens at `beat0 + k * 60 / bpm` seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub bpm: f64,
    pub beat0: f64,
}

impl Grid {
    pub fn period(&self) -> f64 {
        60.0 / self.bpm
    }
    /// `#OFFSET` value of the simfile.
    pub fn offset(&self) -> f64 {
        -self.beat0
    }
    pub fn time(&self, beat: f64) -> f64 {
        self.beat0 + beat * self.period()
    }
    pub fn beat(&self, time: f64) -> f64 {
        (time - self.beat0) / self.period()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TempoOptions {
    pub min_bpm: f64,
    pub max_bpm: f64,
    /// Center of the tempo prior (log scale), ITG songs cluster around here.
    pub prior_bpm: f64,
}

impl Default for TempoOptions {
    fn default() -> Self {
        TempoOptions {
            min_bpm: 70.0,
            max_bpm: 250.0,
            prior_bpm: 140.0,
        }
    }
}

/// Mean envelope value on the beats of (period, phase), normalized.
pub fn comb(env: &Envelope, period: f64, phase: f64, stats: (f64, f64)) -> f64 {
    let dur = env.duration();
    let (mut sum, mut n) = (0.0, 0usize);
    let mut t = phase;
    while t < dur {
        sum += env.at(t) as f64;
        n += 1;
        t += period;
    }
    if n == 0 {
        return 0.0;
    }
    (sum / n as f64 - stats.0) / stats.1.max(1e-9)
}

pub fn best_phase(env: &Envelope, period: f64, step: f64, stats: (f64, f64)) -> (f64, f64) {
    let mut best = (0.0, f64::MIN);
    let mut p = 0.0;
    while p < period {
        let s = comb(env, period, p, stats);
        if s > best.1 {
            best = (p, s);
        }
        p += step;
    }
    best
}

/// Best (bpm, phase, score) in [lo, hi] with the given bpm step.
fn search(
    env: &Envelope,
    lo: f64,
    hi: f64,
    step: f64,
    phase_step: f64,
    stats: (f64, f64),
) -> (f64, f64, f64) {
    let mut best = (lo, 0.0, f64::MIN);
    let mut bpm = lo;
    while bpm <= hi + 1e-9 {
        let (phase, score) = best_phase(env, 60.0 / bpm, phase_step, stats);
        if score > best.2 {
            best = (bpm, phase, score);
        }
        bpm += step;
    }
    best
}

fn prior(bpm: f64, center: f64) -> f64 {
    let x = (bpm / center).log2() / 0.6;
    (-0.5 * x * x).exp()
}

/// Fits a constant BPM and beat phase.
///
/// The comb score of the true tempo is a very narrow peak (a 0.05 BPM error drifts
/// by a quarter beat over a song), so the search is multi-resolution:
/// 1. the whole [min, max] range at 0.1 BPM on a strongly dilated envelope (±40 ms);
/// 2. the best local maxima (plus aubio's estimate) refined at 0.01 BPM (±12 ms);
/// 3. the winner refined to 0.001 BPM on the raw envelope.
///
/// The octave is chosen with a log-normal tempo prior, and the BPM is snapped to an
/// integer when that fits almost as well (most songs are at integer tempos).
pub fn fit_tempo(env: &Envelope, aubio_bpm: f64, opts: &TempoOptions) -> (f64, f64) {
    let stats = (env.mean(), env.std());
    let wide = env.dilated((0.04 * env.fps).round() as usize);
    let wstats = (wide.mean(), wide.std());
    let mid = env.dilated((0.012 * env.fps).round() as usize);
    let mstats = (mid.mean(), mid.std());

    // Stage 1: full range.
    let mut scan = Vec::new();
    let mut bpm = opts.min_bpm;
    while bpm <= opts.max_bpm {
        let (_, sc) = best_phase(&wide, 60.0 / bpm, 0.008, wstats);
        scan.push((bpm, sc * prior(bpm, opts.prior_bpm)));
        bpm += 0.1;
    }
    let mut peaks: Vec<(f64, f64)> = (1..scan.len().saturating_sub(1))
        .filter(|&i| scan[i].1 >= scan[i - 1].1 && scan[i].1 > scan[i + 1].1)
        .map(|i| scan[i])
        .collect();
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut candidates: Vec<f64> = peaks.iter().take(6).map(|p| p.0).collect();
    for m in [0.5, 1.0, 2.0] {
        let c = aubio_bpm * m;
        if (opts.min_bpm..=opts.max_bpm).contains(&c) {
            candidates.push(c);
        }
    }

    // Stage 2: refine each candidate.
    let mut best = (candidates[0], f64::MIN);
    for c in candidates {
        let (b, _, sc) = search(&mid, c - 0.15, c + 0.15, 0.01, 0.003, mstats);
        let weighted = sc * prior(b, opts.prior_bpm);
        if weighted > best.1 {
            best = (b, weighted);
        }
    }

    // Stage 3: fine.
    let (b, p, sc) = search(env, best.0 - 0.012, best.0 + 0.012, 0.001, 0.001, stats);
    let round = b.round();
    let (rp, rs) = best_phase(env, 60.0 / round, 0.001, stats);
    if rs >= sc * 0.97 && (round - b).abs() < 0.2 {
        (round, rp)
    } else {
        ((b * 1000.0).round() / 1000.0, p)
    }
}

/// Picks beat 0 (the downbeat) using the kick layer.
///
/// Broadband onset functions are often dominated by off-beat hi-hats, so the
/// fitted phase can be half a beat off. We try the 8 half-beat shifts of a bar
/// and keep the one whose every-4th beat carries the most low-end energy, with
/// a bonus for kick energy on all 4 beats (resolves the half-beat ambiguity).
pub fn choose_downbeat(kick: &Envelope, bpm: f64, phase: f64) -> f64 {
    let period = 60.0 / bpm;
    // Tolerate small timing offsets: peaks are only one frame wide.
    let kick = &kick.dilated((0.02 * kick.fps) as usize);
    let stats = (kick.mean(), kick.std());
    let mut best = (phase, f64::MIN);
    for k in 0..8 {
        let p = phase + k as f64 * period / 2.0;
        let s = comb(kick, 4.0 * period, p, stats) + 2.0 * comb(kick, period, p, stats);
        if s > best.1 + 1e-9 {
            best = (p, s);
        }
    }
    // Keep beat 0 close to the start of the audio (it may be slightly negative).
    let mut beat0 = best.0;
    while beat0 > 2.0 * period {
        beat0 -= 4.0 * period;
    }
    beat0
}

/// Separated sources of a song (from Demucs), each mono at `sample_rate`.
#[derive(Clone, Debug)]
pub struct Stems {
    pub sample_rate: u32,
    pub drums: Vec<f32>,
    pub bass: Vec<f32>,
    pub vocals: Vec<f32>,
    pub other: Vec<f32>,
}

/// Onset layers of the stems.
#[derive(Clone, Debug)]
pub struct StemLayers {
    pub drums: Layer,
    pub bass: Layer,
    pub vocals: Layer,
    pub other: Layer,
}

/// Everything the chart generator needs to know about the audio.
#[derive(Clone, Debug)]
pub struct SongAnalysis {
    pub grid: Grid,
    pub duration: f64,
    /// aubio's own tempo estimate (before our fit), for diagnostics.
    pub aubio_bpm: f64,
    pub mix: Layer,
    pub kick: Layer,
    pub stems: Option<StemLayers>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AnalysisOptions {
    pub tempo: TempoOptions,
    /// Forces the BPM (skips detection).
    pub bpm: Option<f64>,
    /// Forces the simfile `#OFFSET` (skips phase detection).
    pub offset: Option<f64>,
}

fn median_interval(beats: &[f64]) -> Option<f64> {
    let mut d: Vec<f64> = beats
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0.0)
        .collect();
    if d.is_empty() {
        return None;
    }
    d.sort_by(f64::total_cmp);
    Some(d[d.len() / 2])
}

/// Beat period from aubio's beat times by least squares.
///
/// aubio reports beats on a 512-sample hop, so single intervals are quantized to
/// ~11.6 ms (≈2% of a beat); a regression over the whole song removes that bias.
pub fn regression_period(beats: &[f64]) -> Option<f64> {
    let p0 = median_interval(beats)?;
    let mut k = 0.0;
    let mut pts = vec![(0.0, beats[0])];
    for w in beats.windows(2) {
        k += ((w[1] - w[0]) / p0).round().max(1.0);
        pts.push((k, w[1]));
    }
    let n = pts.len() as f64;
    let mk = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let mt = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let skk: f64 = pts.iter().map(|p| (p.0 - mk).powi(2)).sum();
    let skt: f64 = pts.iter().map(|p| (p.0 - mk) * (p.1 - mt)).sum();
    (skk > 0.0).then(|| skt / skk)
}

impl SongAnalysis {
    pub fn compute(audio: &Audio, stems: Option<&Stems>, opts: &AnalysisOptions) -> SongAnalysis {
        let sr = audio.sample_rate;
        let mix = Layer::analyze(&audio.samples, sr);
        let layers = stems.map(|s| StemLayers {
            drums: Layer::analyze(&s.drums, s.sample_rate),
            bass: Layer::analyze(&s.bass, s.sample_rate),
            vocals: Layer::analyze(&s.vocals, s.sample_rate),
            other: Layer::analyze(&s.other, s.sample_rate),
        });
        // The kick is best found on the isolated drums when we have them.
        let kick = match stems {
            Some(s) => kick_layer(&s.drums, s.sample_rate),
            None => kick_layer(&audio.samples, sr),
        };
        let beats = aubio_beats(&audio.samples, sr);
        let aubio_bpm = regression_period(&beats).map_or(opts.tempo.prior_bpm, |p| 60.0 / p);
        let env = &mix.envelope;
        let (bpm, phase) = match opts.bpm {
            Some(b) => {
                let stats = (env.mean(), env.std());
                (b, best_phase(env, 60.0 / b, 0.001, stats).0)
            }
            None => fit_tempo(env, aubio_bpm, &opts.tempo),
        };
        let beat0 = match opts.offset {
            Some(o) => -o,
            None => choose_downbeat(&kick.envelope, bpm, phase),
        };
        // Round like the simfile will, so that the grid used for charting is exactly
        // the one the game will play.
        let grid = Grid {
            bpm: (bpm * 1000.0).round() / 1000.0,
            beat0: (beat0 * 1000.0).round() / 1000.0,
        };
        SongAnalysis {
            grid,
            duration: audio.duration(),
            aubio_bpm,
            mix,
            kick,
            stems: layers,
        }
    }
}

impl SongAnalysis {
    /// Start of the loudest `length`-second window, snapped down to a beat (music preview).
    pub fn sample_start(&self, length: f64) -> f64 {
        let fps = self.mix.envelope.fps;
        let w = (length * fps) as usize;
        let rms = &self.mix.rms;
        if rms.len() <= w {
            return 0.0;
        }
        let mut sum: f64 = rms[..w].iter().map(|v| *v as f64).sum();
        let mut best = (sum, 0);
        for i in w..rms.len() {
            sum += rms[i] as f64 - rms[i - w] as f64;
            if sum > best.0 + 1e-9 {
                best = (sum, i + 1 - w);
            }
        }
        let t = best.1 as f64 / fps;
        let beat = self.grid.beat(t).floor();
        self.grid.time(beat).max(0.0)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::synth::drum_loop;

    #[test]
    fn tempo_and_phase_on_synthetic_loops() {
        let sr = 44100;
        for &(bpm, first) in &[(128.0, 0.35), (150.0, 0.12), (174.0, 0.8), (100.0, 0.5)] {
            let s = drum_loop(bpm, first, 30.0, sr);
            let layer = Layer::analyze(&s, sr);
            let beats = aubio_beats(&s, sr);
            let aubio_bpm = 60.0 / regression_period(&beats).unwrap();
            let (fit, phase) = fit_tempo(&layer.envelope, aubio_bpm, &TempoOptions::default());
            // Octave errors are accepted: the grid still lines up with the beats.
            let octave_ok = [0.5, 1.0, 2.0].iter().any(|m| (fit - bpm * m).abs() < 0.01);
            assert!(octave_ok, "bpm {bpm}: got {fit} (aubio {aubio_bpm})");
            let period = 60.0 / fit.max(bpm);
            let kick = kick_layer(&s, sr);
            let beat0 = choose_downbeat(&kick.envelope, fit, phase);
            let err = ((beat0 - first) / period).rem_euclid(1.0);
            let err_ms = err.min(1.0 - err) * period * 1000.0;
            assert!(
                err_ms < 25.0,
                "bpm {bpm}: phase error {err_ms:.1} ms (beat0 {beat0})"
            );
            if (fit - bpm).abs() < 0.01 {
                let bar_err = ((beat0 - first) / (4.0 * period)).rem_euclid(1.0);
                assert!(bar_err < 0.02 || bar_err > 0.98, "downbeat off by {bar_err} bars");
            }
        }
    }
}
