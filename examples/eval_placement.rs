//! Do our notes fall where a human charter puts them?
//!
//! For songs of the chosen split (single BPM, no stops), every human dance-single chart
//! is compared with the rows our generator places for the same difficulty slot, on the
//! **human grid** (BPM and offset forced from the simfile, so sync errors do not count).
//! Only the timing of rows is compared, not the arrows.
//!
//! - precision: share of our rows at a position where the human has a row;
//! - recall: share of the human rows we reproduce;
//! - F-score: harmonic mean.
//!
//! Two variants: "natural" (the generator's own number of rows) and "equal count" (as
//! many rows as the human chart, which judges *where* independently of *how many*).
//! As a reference, "chance" is the F-score of placing the same number of rows at random
//! on the 16th-note grid between the first and last human rows. (A human-vs-human
//! reference is not measurable on the ITGmania library: songs present in several packs
//! with the same audio carry copies of the same chart.)
//!
//! Summary written to `eval/placement-<TAG>.txt`.
//!
//! Usage: cargo run --release --example eval_placement -- SONGS_DIR
//!        [--max N] [--split train|test|all] [--tag TAG]

use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::audio::decode_file;
use itg_charter::chart::place_notes_scaled;
use itg_charter::difficulty::Difficulty;
use itg_charter::model::{Model, find_simfiles};
use itg_charter::music::split_of;
use itg_charter::simfile::{Cell, NoteRow, Simfile};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

const SEED: u64 = 1;

/// Row positions (48ths of a beat) of the rows with at least one note.
fn positions(rows: &[NoteRow]) -> BTreeSet<u32> {
    rows.iter()
        .filter(|r| {
            r.cells
                .iter()
                .take(4)
                .any(|c| matches!(c, Cell::Tap | Cell::HoldHead | Cell::RollHead | Cell::Lift))
        })
        .map(|r| (r.beat * 48.0).round().max(0.0) as u32)
        .collect()
}

#[derive(Clone, Copy, Default)]
struct Counts {
    ours: u64,
    human: u64,
    matched: u64,
    charts: u64,
}

impl Counts {
    fn add(&mut self, ours: usize, human: usize, matched: usize) {
        self.ours += ours as u64;
        self.human += human as u64;
        self.matched += matched as u64;
        self.charts += 1;
    }
    fn prf(&self) -> (f64, f64, f64) {
        let p = self.matched as f64 / self.ours.max(1) as f64;
        let r = self.matched as f64 / self.human.max(1) as f64;
        let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        (p, r, f)
    }
}

struct Song {
    sim: Simfile,
    music: PathBuf,
    title: String,
    bpm: f64,
    offset: f64,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: eval_placement SONGS_DIR [--max N] [--split S] [--tag T]";
    let dir = PathBuf::from(args.first().expect(usage));
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let max: usize = opt("--max").and_then(|m| m.parse().ok()).unwrap_or(300);
    let split = opt("--split").unwrap_or_else(|| "test".into());
    let tag = opt("--tag").unwrap_or_else(|| split.clone());
    let model = Model::embedded()?;

    let mut songs = Vec::new();
    for path in find_simfiles(&dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let title = sim
            .tag("TITLE")
            .filter(|t| !t.is_empty())
            .unwrap_or(&name)
            .to_string();
        if split != "all" && split_of(&title) != split {
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
        if !music.exists() {
            continue;
        }
        let (bpm, offset) = (timing.bpms[0].1, timing.offset);
        songs.push(Song {
            sim,
            music,
            title,
            bpm,
            offset,
        });
    }

    let step = (songs.len() as f64 / max as f64).max(1.0);
    let picked: Vec<usize> = (0..max.min(songs.len()))
        .map(|i| (i as f64 * step) as usize)
        .collect();
    eprintln!(
        "{} constant-BPM songs in split {split}, evaluating {}",
        songs.len(),
        picked.len()
    );

    let natural = Mutex::new([Counts::default(); 5]);
    let equal = Mutex::new([Counts::default(); 5]);
    // chance: expected matches = ours × human / grid positions
    let chance = Mutex::new([Counts::default(); 5]);
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
                    let Some(&i) = picked.get(k) else { break };
                    let song = &songs[i];
                    let Ok(audio) = decode_file(&song.music) else {
                        continue;
                    };
                    let opts = AnalysisOptions {
                        bpm: Some(song.bpm),
                        offset: Some(song.offset),
                        ..AnalysisOptions::default()
                    };
                    let a = SongAnalysis::compute(&audio, None, &opts);
                    for chart in &song.sim.charts {
                        if chart.steps_type != "dance-single" {
                            continue;
                        }
                        let Ok(d) = Difficulty::parse(&chart.difficulty) else {
                            continue;
                        };
                        let Ok(rows) = chart.rows() else { continue };
                        let human = positions(&rows);
                        if human.len() < 16 {
                            continue;
                        }
                        let place = |density: f64| -> BTreeSet<u32> {
                            let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ d.seed_salt());
                            place_notes_scaled(&a, &model, d, density, &mut rng)
                                .iter()
                                .map(|n| n.pos)
                                .collect()
                        };
                        let ours = place(1.0);
                        let m = ours.intersection(&human).count();
                        natural.lock().unwrap()[d.index()].add(ours.len(), human.len(), m);
                        // Equal count: rescale the density (two refinements of the ratio).
                        let mut density = human.len() as f64 / ours.len().max(1) as f64;
                        let mut same = place(density);
                        for _ in 0..2 {
                            density *= human.len() as f64 / same.len().max(1) as f64;
                            same = place(density);
                        }
                        let m = same.intersection(&human).count();
                        equal.lock().unwrap()[d.index()].add(same.len(), human.len(), m);
                        let span = (human.last().unwrap() - human.first().unwrap()) / 12 + 1;
                        let expected = (human.len() * human.len()) as f64 / span as f64;
                        chance.lock().unwrap()[d.index()].add(
                            human.len(),
                            human.len(),
                            expected.round() as usize,
                        );
                    }
                    eprintln!("[{k}] {}", song.title);
                }
            });
        }
    });

    let natural = natural.into_inner().unwrap();
    let equal = equal.into_inner().unwrap();
    let human_vs_human = chance.into_inner().unwrap();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "split {split}, songs {}, seed {SEED}, human grid forced",
        picked.len()
    );
    let _ = writeln!(
        out,
        "{:<10} | {:^29} | {:^29} | {:^22}",
        "", "natural (P / R / F)", "equal count (P / R / F)", "chance F (equal count)"
    );
    let mut totals = [Counts::default(); 3];
    for d in Difficulty::ALL {
        let i = d.index();
        let (np, nr, nf) = natural[i].prf();
        let (ep, er, ef) = equal[i].prf();
        let (_, _, hf) = human_vs_human[i].prf();
        let _ = writeln!(
            out,
            "{:<10} | {:>5.1}% {:>5.1}% {:>5.1}% ({:>4}) | {:>5.1}% {:>5.1}% {:>5.1}% ({:>4}) | {:>5.1}%",
            d.name(),
            100.0 * np,
            100.0 * nr,
            100.0 * nf,
            natural[i].charts,
            100.0 * ep,
            100.0 * er,
            100.0 * ef,
            equal[i].charts,
            100.0 * hf,
        );
        for (t, c) in totals.iter_mut().zip([natural[i], equal[i], human_vs_human[i]]) {
            t.ours += c.ours;
            t.human += c.human;
            t.matched += c.matched;
            t.charts += c.charts;
        }
    }
    let _ = writeln!(
        out,
        "{:<10} | F natural {:.1}% | F equal count {:.1}% | F chance {:.1}%",
        "All",
        100.0 * totals[0].prf().2,
        100.0 * totals[1].prf().2,
        100.0 * totals[2].prf().2
    );
    print!("{out}");
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("eval");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(format!("placement-{tag}.txt")), &out)?;
    Ok(())
}
