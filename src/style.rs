//! Charting styles: one learned arrow model per style, embedded in the binary.
//!
//! The styles are groups of packs with similar habits at equal meter (see
//! `training/STYLES.md`); the pack of each style is listed in `training/styles.tsv` and
//! the models are rebuilt with the `train_styles` example. The placement model
//! (`model/placement.json`) is shared: it is fitted on every pack.

use crate::chart::GenOptions;
use crate::model::{self, Model};
use anyhow::Result;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Style {
    /// Classic ITG (the game's Songs, ITG3, Rebirth…): average density, jumps, some
    /// crossovers, few holds.
    #[default]
    Classic,
    /// Stream / modern pad (TalonMix, Valex, FlightMix…): denser, faster, few jumps,
    /// almost no crossovers.
    Stream,
    /// Tech (Crossover Spectrum, DDR edits, Stomp Exceed…): many crossovers,
    /// footswitches, jacks and brackets, few holds.
    Tech,
}

impl Style {
    pub const ALL: [Style; 3] = [Style::Classic, Style::Stream, Style::Tech];

    /// Name used on the command line and in `training/styles.tsv`.
    pub fn name(self) -> &'static str {
        match self {
            Style::Classic => "classic",
            Style::Stream => "stream",
            Style::Tech => "tech",
        }
    }

    /// Model file, relative to the repository root.
    pub fn model_path(self) -> &'static str {
        match self {
            Style::Classic => "model/model.json",
            Style::Stream => "model/styles/stream.json",
            Style::Tech => "model/styles/tech.json",
        }
    }

    fn json(self) -> &'static str {
        match self {
            Style::Classic => model::EMBEDDED,
            Style::Stream => include_str!("../model/styles/stream.json"),
            Style::Tech => include_str!("../model/styles/tech.json"),
        }
    }

    /// Generation options of this style. The parity weight (0.02, `eval_charts`) was
    /// calibrated on classic charts; at that weight the tech model makes 0.3 crossovers
    /// per 100 rows where tech charters make 4.7 (Medium median). Swept with
    /// `eval_charts` on the tech packs: 0.01 → 1.9, 0.005 → 4.2, 0 → 12.6 (above
    /// p90).
    pub fn gen_options(self) -> GenOptions {
        let parity_weight = match self {
            Style::Classic | Style::Stream => GenOptions::default().parity_weight,
            Style::Tech => 0.005,
        };
        GenOptions {
            parity_weight,
            ..GenOptions::default()
        }
    }

    /// The embedded model of this style.
    pub fn model(self) -> Result<Model> {
        Model::from_json(self.json())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::difficulty::Difficulty;

    #[test]
    fn every_style_model_loads_and_matches_its_style() {
        let models: Vec<Model> = Style::ALL.iter().map(|s| s.model().unwrap()).collect();
        let [classic, stream, tech] = &models[..] else {
            unreachable!()
        };
        for m in &models {
            assert!(m.stats(Difficulty::Hard).charts > 0);
            let s: f64 = (1..16u8).map(|c| m.prob(Difficulty::Hard, 3, 1, 8, c)).sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
        assert_eq!(
            Style::default().model().unwrap().to_json(),
            Model::embedded().unwrap().to_json()
        );
        // The habits that tell the styles apart (see training/STYLES.md).
        let xo = |m: &Model| m.stats(Difficulty::Hard).crossovers.p50;
        let jumps = |m: &Model| m.stats(Difficulty::Hard).jump_ratio.p50;
        assert!(xo(tech) > xo(classic) && xo(classic) > xo(stream));
        assert!(jumps(stream) < jumps(classic));
    }
}
