//! itg-charter: generates In The Groove / ITGmania step charts from audio.
//!
//! Pipeline: [`audio`] decoding → optional [`stems`] separation (Demucs) →
//! [`analysis`] (aubio onsets, tempo and phase fit) → [`chart`] generation driven by
//! the [`model`] learned from human charts and the [`parity`] port of ITGmania →
//! [`simfile`] output.

pub mod analysis;
pub mod audio;
pub mod chart;
pub mod difficulty;
pub mod model;
pub mod parity;
pub mod simfile;
pub mod stems;
pub mod synth;
