//! Fits the learned placement model (`model/placement.json`).
//!
//! Each 12th/16th grid position of a human dance-single chart (between its first and
//! last row, on the human grid) is one example: its features ([`PlacementFeatures`])
//! and whether the human put a row there. One logistic regression per difficulty is
//! fitted on the train split and scored (log loss, accuracy) on the test split, next to
//! a constant-rate baseline. Writes the model and `eval/placement-fit.txt`.
//!
//! Usage: cargo run --release --example fit_placement -- SONGS_DIR [--max-train N] [--max-test N]

use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::audio::decode_file;
use itg_charter::difficulty::Difficulty;
use itg_charter::model::find_simfiles;
use itg_charter::music::split_of;
use itg_charter::placement::{
    FEATURE_NAMES, FEATURES, PLACEMENT_VERSION, PlacementFeatures, PlacementModel, fit_logistic,
    is_candidate, logistic,
};
use itg_charter::simfile::{Cell, Simfile};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

type Data = Vec<Vec<([f32; FEATURES], bool)>>; // per difficulty

struct Song {
    sim: Simfile,
    music: PathBuf,
    bpm: f64,
    offset: f64,
}

fn load_songs(dir: &std::path::Path, split: &str, max: usize) -> Vec<Song> {
    let mut songs = Vec::new();
    for path in find_simfiles(dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let title = sim
            .tag("TITLE")
            .filter(|t| !t.is_empty())
            .unwrap_or(&name)
            .to_string();
        if split_of(&title) != split {
            continue;
        }
        let Some(chart) = sim.charts.first() else { continue };
        let Ok(timing) = sim.timing(chart) else { continue };
        if !timing.is_constant() || sim.charts.iter().any(|c| c.tags.iter().any(|(k, _)| k == "BPMS")) {
            continue;
        }
        let Some(music) = sim.tag("MUSIC").filter(|m| !m.is_empty()) else {
            continue;
        };
        let music = path.parent().unwrap().join(music);
        if music.exists() {
            let (bpm, offset) = (timing.bpms[0].1, timing.offset);
            songs.push(Song {
                sim,
                music,
                bpm,
                offset,
            });
        }
    }
    let step = (songs.len() as f64 / max as f64).max(1.0);
    let picked: BTreeSet<usize> = (0..max.min(songs.len()))
        .map(|i| (i as f64 * step) as usize)
        .collect();
    songs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| picked.contains(i))
        .map(|(_, s)| s)
        .collect()
}

fn collect(songs: &[Song]) -> Data {
    let data = Mutex::new(vec![Vec::new(); 5]);
    let next = Mutex::new(0usize);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| {
                loop {
                    let k = {
                        let mut n = next.lock().unwrap();
                        *n += 1;
                        *n - 1
                    };
                    let Some(song) = songs.get(k) else { break };
                    let Ok(audio) = decode_file(&song.music) else {
                        continue;
                    };
                    let opts = AnalysisOptions {
                        bpm: Some(song.bpm),
                        offset: Some(song.offset),
                        ..AnalysisOptions::default()
                    };
                    let a = SongAnalysis::compute(&audio, None, &opts);
                    let pf = PlacementFeatures::new(&a);
                    let mut local = vec![Vec::new(); 5];
                    for chart in &song.sim.charts {
                        if chart.steps_type != "dance-single" {
                            continue;
                        }
                        let Ok(d) = Difficulty::parse(&chart.difficulty) else {
                            continue;
                        };
                        let Ok(rows) = chart.rows() else { continue };
                        let human: BTreeSet<u32> = rows
                            .iter()
                            .filter(|r| {
                                r.cells.iter().take(4).any(|c| {
                                    matches!(c, Cell::Tap | Cell::HoldHead | Cell::RollHead | Cell::Lift)
                                })
                            })
                            .map(|r| (r.beat * 48.0).round().max(0.0) as u32)
                            .collect();
                        if human.len() < 16 {
                            continue;
                        }
                        let (first, last) = (*human.first().unwrap(), *human.last().unwrap());
                        for pos in (first..=last).filter(|p| is_candidate(*p)) {
                            local[d.index()].push((pf.features(pos), human.contains(&pos)));
                        }
                    }
                    let mut all = data.lock().unwrap();
                    for (a, l) in all.iter_mut().zip(local) {
                        a.extend(l);
                    }
                }
            });
        }
    });
    data.into_inner().unwrap()
}

fn log_loss(w: &[f64], data: &[([f32; FEATURES], bool)]) -> (f64, f64) {
    let (mut ll, mut ok) = (0.0, 0usize);
    for (x, y) in data {
        let p = logistic(w, x).clamp(1e-9, 1.0 - 1e-9);
        ll -= if *y { p.ln() } else { (1.0 - p).ln() };
        ok += ((p > 0.5) == *y) as usize;
    }
    (
        ll / data.len().max(1) as f64,
        ok as f64 / data.len().max(1) as f64,
    )
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(
        args.first()
            .expect("usage: fit_placement SONGS_DIR [--max-train N] [--max-test N]"),
    );
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let max_train: usize = opt("--max-train").and_then(|m| m.parse().ok()).unwrap_or(300);
    let max_test: usize = opt("--max-test").and_then(|m| m.parse().ok()).unwrap_or(150);
    let train = collect(&load_songs(&dir, "train", max_train));
    let test = collect(&load_songs(&dir, "test", max_test));

    let mut report = String::new();
    let _ = writeln!(
        report,
        "train songs {max_train}, test songs {max_test} (max), L2 = 1.0"
    );
    let mut weights = Vec::new();
    for d in Difficulty::ALL {
        let (tr, te) = (&train[d.index()], &test[d.index()]);
        let xs: Vec<[f32; FEATURES]> = tr.iter().map(|e| e.0).collect();
        let ys: Vec<bool> = tr.iter().map(|e| e.1).collect();
        let w = fit_logistic(&xs, &ys, 1.0);
        // Constant-rate baseline: bias only.
        let rate = ys.iter().filter(|y| **y).count() as f64 / ys.len().max(1) as f64;
        let mut base = vec![0.0; FEATURES + 1];
        base[0] = (rate / (1.0 - rate)).ln();
        let (ll_tr, acc_tr) = log_loss(&w, tr);
        let (ll_te, acc_te) = log_loss(&w, te);
        let (bl_te, bacc_te) = log_loss(&base, te);
        let _ = writeln!(
            report,
            "{:<9}: {} train / {} test positions, row rate {:.1}% | log loss train {:.4}, test {:.4} (constant {:.4}) | accuracy test {:.1}% (constant {:.1}%)",
            d.name(),
            tr.len(),
            te.len(),
            100.0 * rate,
            ll_tr,
            ll_te,
            bl_te,
            100.0 * acc_te,
            100.0 * bacc_te
        );
        let _ = writeln!(
            report,
            "           weights: bias {:+.3}, {}",
            w[0],
            FEATURE_NAMES
                .iter()
                .zip(&w[1..])
                .map(|(n, v)| format!("{n} {v:+.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = acc_tr;
        weights.push(w);
    }
    print!("{report}");
    let model = PlacementModel {
        version: PLACEMENT_VERSION,
        weights,
        report: report.clone(),
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::write(
        root.join("model/placement.json"),
        serde_json::to_string_pretty(&model)?,
    )?;
    std::fs::write(root.join("eval/placement-fit.txt"), &report)?;
    Ok(())
}
