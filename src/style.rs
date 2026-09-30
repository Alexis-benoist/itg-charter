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

    /// The model, packed by `build.rs` from [`Style::model_path`].
    fn packed(self) -> &'static [u8] {
        match self {
            Style::Classic => model::EMBEDDED,
            Style::Stream => include_bytes!(concat!(env!("OUT_DIR"), "/stream.bin")),
            Style::Tech => include_bytes!(concat!(env!("OUT_DIR"), "/tech.bin")),
        }
    }

    /// Generation options of this style, measured with `eval_charts` and
    /// `eval_patterns` on 40 songs of the style (`eval/*grid*`, `eval/README.md`).
    ///
    /// ITGmania's parity cost penalises every foot movement: at the default weight
    /// (0.02) and temperature (0.4) the beam settles into drills (Challenge classic: 46
    /// drill rows / 100, human median 5.4) and avoids stairs (2.9, human 7.6), while
    /// sampling the order-4 model alone matches humans on every pattern. Temperature
    /// 0.7 with half the parity weight keeps the patterns human (0 flag in every
    /// style), but lets footswitches through (Medium: 0.3 / 100, humans 0 in every
    /// style; stream Hard 0.97, human p90 0.94; tech Hard 1.28, p90 1.14).
    /// Tech charts need less parity still for their crossovers (weight 0.02: 0.3 / 100
    /// in Medium, humans 4.7).
    ///
    /// Hence a separate footswitch penalty (see [`crate::chart::footswitch_penalty`]):
    /// 2 nats, divided by 1 + the human footswitch p90 of the difficulty. Medium falls
    /// to 0 footswitch in every style, Hard / Challenge stay between the human p50 and
    /// p90 (classic 0.33 / 0.67, stream 0.27 / 0.69, tech 0.40 / 0.97), the patterns
    /// stay human (0 flag; a base of 4 pushes classic Challenge drills just out of the
    /// human range) and so do crossovers (`eval/*fsp90*`).
    pub fn gen_options(self) -> GenOptions {
        let parity_weight = match self {
            Style::Classic | Style::Stream => 0.01,
            Style::Tech => 0.0025,
        };
        GenOptions {
            parity_weight,
            temperature: 0.7,
            footswitch_penalty: 2.0,
            ..GenOptions::default()
        }
    }

    /// The embedded model of this style.
    pub fn model(self) -> Result<Model> {
        Model::from_packed(self.packed())
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
            let s: f64 = (1..16u8).map(|c| m.prob(Difficulty::Hard, 3, 2, 1, 8, c)).sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
        assert_eq!(
            Style::default().model().unwrap().to_json(),
            Model::embedded().unwrap().to_json()
        );
        // The packed models are the JSON files, exactly.
        for (style, m) in Style::ALL.iter().zip(&models) {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(style.model_path());
            let json = Model::from_json(&std::fs::read_to_string(path).unwrap()).unwrap();
            assert!(json == *m, "{} differs from its JSON", style.name());
        }
        // The habits that tell the styles apart (see training/STYLES.md).
        let xo = |m: &Model| m.stats(Difficulty::Hard).crossovers.p50;
        let jumps = |m: &Model| m.stats(Difficulty::Hard).jump_ratio.p50;
        assert!(xo(tech) > xo(classic) && xo(classic) > xo(stream));
        assert!(jumps(stream) < jumps(classic));
    }
}
