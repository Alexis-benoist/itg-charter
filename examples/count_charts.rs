//! How many charts would `itg-charter train` learn from?
//!
//! Applies the same filters as `Model::train` (dance-single, parsable difficulty, at
//! least 16 note rows, our generated simfiles excluded) and reports the total per
//! difficulty, the number of simfiles, and songs present more than once (same
//! title + artist) across packs.
//!
//! Usage: cargo run --release --example count_charts -- SONGS_DIR [SONGS_DIR...]

use itg_charter::difficulty::Difficulty;
use itg_charter::model::{ChartFeatures, find_simfiles};
use itg_charter::parity::Layout;
use itg_charter::simfile::Simfile;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() {
    let dirs: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    assert!(!dirs.is_empty(), "usage: count_charts SONGS_DIR [SONGS_DIR...]");
    let layout = Layout::dance_single();
    for dir in dirs {
        let files = find_simfiles(&dir);
        let (mut simfiles, mut songs_with_charts) = (0usize, 0usize);
        let mut per_diff = [0usize; 5];
        let mut titles: BTreeMap<String, usize> = BTreeMap::new();
        for path in &files {
            let Ok(sim) = Simfile::load(path) else { continue };
            if sim.is_generated() {
                continue;
            }
            simfiles += 1;
            let mut n = 0;
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
                if ChartFeatures::compute(&rows, &timing, &layout).rows < 16 {
                    continue;
                }
                per_diff[diff.index()] += 1;
                n += 1;
            }
            if n > 0 {
                songs_with_charts += 1;
                let tag = |k| sim.tag(k).unwrap_or("").trim().to_lowercase();
                let key = format!("{} / {}", tag("TITLE"), tag("ARTIST"));
                *titles.entry(key).or_default() += 1;
            }
        }
        let total: usize = per_diff.iter().sum();
        let dups: usize = titles.values().map(|&c| c - 1).sum();
        println!("{}", dir.display());
        println!("  simfiles {simfiles}, songs with charts {songs_with_charts} (duplicates {dups})");
        let by_diff: Vec<String> = Difficulty::ALL
            .iter()
            .map(|d| format!("{} {}", d.name(), per_diff[d.index()]))
            .collect();
        println!("  charts {total}: {}", by_diff.join(", "));
    }
}
