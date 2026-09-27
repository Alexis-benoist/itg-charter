//! Measures BPM / offset detection against human-synced simfiles.
//!
//! For every song with a single BPM and no stops, the audio is analyzed and the
//! detected grid is compared with the simfile's: exact BPM rate, octave errors,
//! and beat phase error (ms) when the BPM is right.
//!
//! Usage: cargo run --release --example eval_sync -- SONGS_DIR [MAX_SONGS] [--stems]

use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::audio::decode_file;
use itg_charter::model::find_simfiles;
use itg_charter::simfile::Simfile;
use itg_charter::stems;
use std::path::PathBuf;
use std::sync::Mutex;

struct Case {
    name: String,
    music: PathBuf,
    bpm: f64,
    beat0: f64,
}

struct Outcome {
    name: String,
    truth_bpm: f64,
    bpm: f64,
    aubio_bpm: f64,
    phase_ms: Option<f64>,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(args.first().expect("usage: eval_sync SONGS_DIR [MAX] [--stems]"));
    let max: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(150);
    let use_stems = args.iter().any(|a| a == "--stems");

    let mut cases = Vec::new();
    for path in find_simfiles(&dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
        let Some(chart) = sim.charts.first() else { continue };
        let Ok(timing) = sim.timing(chart) else { continue };
        if !timing.is_constant()
            || sim
                .charts
                .iter()
                .any(|c| !c.tags.is_empty() && c.tags.iter().any(|(k, _)| k == "BPMS"))
        {
            continue;
        }
        let Some(music) = sim.tag("MUSIC").filter(|m| !m.is_empty()) else {
            continue;
        };
        let music = path.parent().unwrap().join(music);
        if !music.exists() {
            continue;
        }
        cases.push(Case {
            name: path.file_stem().unwrap().to_string_lossy().into_owned(),
            music,
            bpm: timing.bpms[0].1,
            beat0: -timing.offset,
        });
    }
    // Spread the sample evenly over packs.
    let step = (cases.len() as f64 / max as f64).max(1.0);
    let picked: Vec<&Case> = (0..max.min(cases.len()))
        .map(|i| &cases[(i as f64 * step) as usize])
        .collect();
    eprintln!("{} constant-BPM songs, evaluating {}", cases.len(), picked.len());

    let results = Mutex::new(Vec::new());
    let next = Mutex::new(0usize);
    let threads = if use_stems {
        1
    } else {
        std::thread::available_parallelism().map_or(4, |n| n.get())
    };
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = {
                        let mut n = next.lock().unwrap();
                        *n += 1;
                        *n - 1
                    };
                    let Some(case) = picked.get(i) else { break };
                    let Ok(audio) = decode_file(&case.music) else {
                        continue;
                    };
                    let st = if use_stems {
                        stems::separate(&case.music, &stems::StemOptions::default()).ok()
                    } else {
                        None
                    };
                    let a = SongAnalysis::compute(&audio, st.as_ref(), &AnalysisOptions::default());
                    let phase_ms = ((a.grid.bpm - case.bpm).abs() < 0.05).then(|| {
                        let period = 60.0 / case.bpm;
                        let e = ((a.grid.beat0 - case.beat0) / period).rem_euclid(1.0);
                        let e = if e > 0.5 { e - 1.0 } else { e };
                        e * period * 1000.0
                    });
                    eprintln!(
                        "[{i:>3}] {:<40} truth {:>7.2} ours {:>7.2} (aubio {:>6.1}) phase {}",
                        case.name.chars().take(40).collect::<String>(),
                        case.bpm,
                        a.grid.bpm,
                        a.aubio_bpm,
                        phase_ms.map_or("-".into(), |p| format!("{p:+.1} ms"))
                    );
                    results.lock().unwrap().push(Outcome {
                        name: case.name.clone(),
                        truth_bpm: case.bpm,
                        bpm: a.grid.bpm,
                        aubio_bpm: a.aubio_bpm,
                        phase_ms,
                    });
                }
            });
        }
    });
    let results = results.into_inner().unwrap();
    let n = results.len() as f64;
    let exact = results
        .iter()
        .filter(|r| (r.bpm - r.truth_bpm).abs() < 0.05)
        .count();
    let octave = results
        .iter()
        .filter(|r| {
            [0.5, 2.0, 2.0 / 3.0, 1.5]
                .iter()
                .any(|m| (r.bpm - r.truth_bpm * m).abs() < 0.05 * m)
        })
        .count();
    let aubio_close = results
        .iter()
        .filter(|r| (r.aubio_bpm - r.truth_bpm).abs() < 1.0)
        .count();
    let mut phases: Vec<f64> = results.iter().filter_map(|r| r.phase_ms).collect();
    phases.sort_by(f64::total_cmp);
    let mut abs: Vec<f64> = phases.iter().map(|p| p.abs()).collect();
    abs.sort_by(f64::total_cmp);
    let q = |v: &[f64], p: f64| {
        v.get(((v.len().max(1) - 1) as f64 * p) as usize)
            .copied()
            .unwrap_or(f64::NAN)
    };
    println!("songs: {}", results.len());
    println!("exact BPM (±0.05): {exact} ({:.1}%)", 100.0 * exact as f64 / n);
    println!(
        "octave-type errors (×½ ×2 ×⅔ ×1.5): {octave} ({:.1}%)",
        100.0 * octave as f64 / n
    );
    println!(
        "aubio raw estimate within 1 BPM: {aubio_close} ({:.1}%)",
        100.0 * aubio_close as f64 / n
    );
    println!(
        "phase error on exact-BPM songs: median signed {:+.1} ms, |err| median {:.1} ms, p90 {:.1} ms, within 20 ms: {:.1}%",
        q(&phases, 0.5),
        q(&abs, 0.5),
        q(&abs, 0.9),
        100.0 * abs.iter().filter(|a| **a < 20.0).count() as f64 / abs.len().max(1) as f64
    );
    let wrong: Vec<&Outcome> = results
        .iter()
        .filter(|r| (r.bpm - r.truth_bpm).abs() >= 0.05)
        .collect();
    println!("wrong BPM examples:");
    for r in wrong.iter().take(15) {
        println!(
            "  {:<40} truth {:.3} ours {:.3} aubio {:.1}",
            r.name, r.truth_bpm, r.bpm, r.aubio_bpm
        );
    }
    Ok(())
}
