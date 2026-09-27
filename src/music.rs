//! Musical cues at each note, used to relate arrows to the music.
//!
//! For a note at time t we describe *what happens in the music* with three coarse
//! cues, each measured on the full mix:
//! - pitch movement since the previous note (aubio yinfast pitch track, in semitones);
//! - hit type: which frequency band (kick, low-mid, mid, high) carries the attack;
//! - accent: whether the attack is strong compared to the song's other onsets.
//!
//! Whether these cues predict human arrows is measured, not assumed
//! (`examples/music_signal.rs`, `examples/eval_arrows.rs`).

use crate::analysis::{Envelope, HALF_BEAT_BANDS, band_layer};
use crate::audio::Audio;
use aubio_rs::{Pitch, PitchMode, PitchUnit};

/// Pitch analysis window and hop, in samples.
const PITCH_WIN: usize = 2048;
const PITCH_HOP: usize = 256;
/// Pitch estimates below this aubio confidence count as unvoiced.
const PITCH_MIN_CONFIDENCE: f32 = 0.7;
/// Pitch is read over this window after the note, in seconds.
const PITCH_WINDOW: f64 = 0.12;
/// Minimum pitch change (semitones) that counts as up or down.
const PITCH_STEP: f32 = 2.0;

pub const PITCH_MOVES: usize = 4; // down, same, up, unknown
pub const HIT_TYPES: usize = HALF_BEAT_BANDS.len();
pub const ACCENTS: usize = 2; // weak, strong
pub const CONTEXTS: usize = PITCH_MOVES * HIT_TYPES * ACCENTS;

/// FNV-1a, stable across Rust versions (unlike `DefaultHasher`).
fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// Deterministic train/test split of the song library by normalized title, so that
/// copies of a song in several packs stay on the same side.
pub fn split_of(title: &str) -> &'static str {
    let norm: String = title
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    if fnv1a(&norm).is_multiple_of(2) {
        "train"
    } else {
        "test"
    }
}

/// Pitch (MIDI note number) per hop; NaN where unvoiced.
#[derive(Clone, Debug, Default)]
pub struct PitchTrack {
    pub fps: f64,
    pub midi: Vec<f32>,
}

impl PitchTrack {
    pub fn compute(samples: &[f32], sample_rate: u32) -> PitchTrack {
        // yinfast: FFT-based YIN. (The bundled aubio reports no confidence for yinfft.)
        let mut pitch = Pitch::new(PitchMode::Yinfast, PITCH_WIN, PITCH_HOP, sample_rate)
            .expect("aubio pitch")
            .with_unit(PitchUnit::Midi)
            .with_silence(-50.0);
        let mut midi = Vec::with_capacity(samples.len() / PITCH_HOP + 1);
        let mut chunk = [0f32; PITCH_HOP];
        for block in samples.chunks(PITCH_HOP) {
            chunk[..block.len()].copy_from_slice(block);
            chunk[block.len()..].fill(0.0);
            let p = pitch.do_result(chunk).expect("aubio pitch frame");
            let voiced = p > 0.0 && pitch.get_confidence() >= PITCH_MIN_CONFIDENCE;
            midi.push(if voiced { p } else { f32::NAN });
        }
        PitchTrack {
            fps: sample_rate as f64 / PITCH_HOP as f64,
            midi,
        }
    }

    /// Median voiced pitch over [t, t + w], or NaN.
    pub fn median(&self, t: f64, w: f64) -> f32 {
        let a = (t * self.fps).max(0.0) as usize;
        let b = (((t + w) * self.fps).max(0.0) as usize + 1).min(self.midi.len());
        let mut v: Vec<f32> = self
            .midi
            .get(a..b)
            .unwrap_or(&[])
            .iter()
            .copied()
            .filter(|p| !p.is_nan())
            .collect();
        if v.is_empty() {
            return f32::NAN;
        }
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    }
}

/// Everything needed to describe the music at any note of a song.
#[derive(Clone, Debug, Default)]
pub struct MusicCues {
    pub pitch: PitchTrack,
    /// Energy-flux envelopes of `HALF_BEAT_BANDS`, with their 90th percentiles.
    bands: Vec<(Envelope, f32)>,
    /// Broadband envelope and the threshold above which an attack is "strong".
    mix: Envelope,
    strong: f32,
}

fn percentile(v: &[f32], p: f64) -> f32 {
    let mut s: Vec<f32> = v.iter().copied().filter(|x| *x > 0.0).collect();
    if s.is_empty() {
        return 1.0;
    }
    s.sort_by(f32::total_cmp);
    s[((s.len() - 1) as f64 * p) as usize].max(1e-9)
}

/// Musical context of one note (see [`MusicCues::context`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    /// 0 down, 1 same, 2 up, 3 unknown.
    pub pitch_move: u8,
    /// Index into `HALF_BEAT_BANDS` (0 kick … 3 high).
    pub hit: u8,
    /// 0 weak, 1 strong.
    pub accent: u8,
}

impl Context {
    pub fn index(self) -> usize {
        (self.pitch_move as usize * HIT_TYPES + self.hit as usize) * ACCENTS + self.accent as usize
    }
}

impl MusicCues {
    /// `mix` is the broadband onset envelope of the song (e.g. `SongAnalysis::mix`).
    pub fn compute(audio: &Audio, mix: &Envelope) -> MusicCues {
        let bands = HALF_BEAT_BANDS
            .iter()
            .map(|&(lo, hi)| {
                let e = band_layer(&audio.samples, audio.sample_rate, lo, hi).envelope;
                let p90 = percentile(&e.values, 0.9);
                (e, p90)
            })
            .collect();
        MusicCues {
            pitch: PitchTrack::compute(&audio.samples, audio.sample_rate),
            bands,
            mix: mix.clone(),
            strong: percentile(&mix.values, 0.95),
        }
    }

    /// Pitch at a note (NaN if unvoiced).
    pub fn pitch_at(&self, t: f64) -> f32 {
        self.pitch.median(t, PITCH_WINDOW)
    }

    /// Context of a note at `t`, given the pitch at the previous note.
    pub fn context(&self, t: f64, previous_pitch: f32) -> Context {
        let p = self.pitch_at(t);
        let pitch_move = if p.is_nan() || previous_pitch.is_nan() {
            3
        } else if p - previous_pitch <= -PITCH_STEP {
            0
        } else if p - previous_pitch >= PITCH_STEP {
            2
        } else {
            1
        };
        let mut hit = (0u8, f32::MIN);
        for (k, (env, p90)) in self.bands.iter().enumerate() {
            let v = env.max_around(t, 0.03) / p90;
            if v > hit.1 {
                hit = (k as u8, v);
            }
        }
        let accent = (self.mix.max_around(t, 0.03) >= self.strong) as u8;
        Context {
            pitch_move,
            hit: hit.0,
            accent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_track_follows_a_rising_tone() {
        let sr = 44100;
        let tone = |freq: f32, secs: f32| -> Vec<f32> {
            (0..(secs * sr as f32) as usize)
                .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
                .collect()
        };
        let mut s = tone(220.0, 0.5); // A3 = MIDI 57
        s.extend(tone(440.0, 0.5)); // A4 = MIDI 69
        let track = PitchTrack::compute(&s, sr);
        let low = track.median(0.2, 0.1);
        let high = track.median(0.7, 0.1);
        assert!((low - 57.0).abs() < 0.5, "low {low}");
        assert!((high - 69.0).abs() < 0.5, "high {high}");
    }

    #[test]
    fn split_is_stable() {
        assert_eq!(split_of("Call Me Maybe"), split_of("call me  maybe!"));
        let train = ["a", "b", "c", "d", "e", "f", "g", "h"]
            .iter()
            .filter(|t| split_of(t) == "train")
            .count();
        assert!(train > 0 && train < 8);
    }
}
