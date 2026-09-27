//! Measures BPM / offset detection against human-synced simfiles.
//!
//! For every song with a single BPM and no stops, the audio is analyzed and the
//! detected grid is compared with the simfile's: exact BPM rate, octave errors,
//! beat phase error (ms) when the BPM is right, broken down by audio format.
//!
//! Songs are split deterministically into a train and a test half by a hash of
//! their normalized title (copies of a song in several packs stay on the same
//! side). Calibrations are fitted on `train` (see `fit_sync.rs`) and reported on
//! `test`.
//!
//! Per-song results go to `target/eval/sync-<TAG>.csv` and the summary to
//! `eval/sync-<TAG>.txt` (committed with the change it measures).
//!
//! Usage: cargo run --release --example eval_sync -- SONGS_DIR
//!        [--max N] [--split train|test|all] [--stems] [--tag TAG]

use itg_charter::analysis::{AnalysisOptions, HALF_BEAT_FEATURES, SongAnalysis};
use itg_charter::audio::decode_file;
use itg_charter::model::find_simfiles;
use itg_charter::simfile::Simfile;
use itg_charter::stems;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

struct Case {
    name: String,
    music: PathBuf,
    format: String,
    bpm: f64,
    beat0: f64,
}

struct Outcome {
    name: String,
    format: String,
    truth_bpm: f64,
    bpm: f64,
    aubio_bpm: f64,
    /// Signed phase error of the final grid, ms (exact-BPM songs only).
    phase_ms: Option<f64>,
    /// Whether the phase before the half-beat decision was on the beat (exact-BPM songs only).
    raw_on_beat: Option<bool>,
    on_beat_prob: f64,
    features: [f64; HALF_BEAT_FEATURES],
    tempo_candidates: Vec<(f64, f64)>,
}

/// FNV-1a, stable across Rust versions (unlike `DefaultHasher`).
fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// "train" or "test" for a song title.
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

/// Signed distance to the nearest beat of the truth grid, in beats, in [-0.5, 0.5).
fn beat_error(t: f64, beat0: f64, period: f64) -> f64 {
    let e = ((t - beat0) / period).rem_euclid(1.0);
    if e >= 0.5 { e - 1.0 } else { e }
}

fn category(o: &Outcome) -> &'static str {
    if (o.bpm - o.truth_bpm).abs() >= 0.05 {
        let octave = [0.5, 2.0, 2.0 / 3.0, 1.5]
            .iter()
            .any(|m| (o.bpm - o.truth_bpm * m).abs() < 0.05 * m);
        return if octave { "octave" } else { "bpm" };
    }
    let period = 60_000.0 / o.truth_bpm;
    let p = o.phase_ms.unwrap_or(0.0).abs();
    if p < 30.0 {
        "ok"
    } else if (p / period - 0.5).abs() < 0.08 {
        "half"
    } else if (p / period - 0.25).abs() < 0.08 {
        "quarter"
    } else {
        "phase"
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: eval_sync SONGS_DIR [--max N] [--split train|test|all] [--stems] [--tag TAG]";
    let dir = PathBuf::from(args.first().expect(usage));
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let max: usize = opt("--max").and_then(|m| m.parse().ok()).unwrap_or(150);
    let split = opt("--split").unwrap_or_else(|| "test".into());
    let use_stems = args.iter().any(|a| a == "--stems");
    let tag = opt("--tag").unwrap_or_else(|| format!("{split}{}", if use_stems { "-stems" } else { "" }));

    let mut cases = Vec::new();
    for path in find_simfiles(&dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
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
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let title = sim
            .tag("TITLE")
            .filter(|t| !t.is_empty())
            .unwrap_or(&name)
            .to_string();
        if split != "all" && split_of(&title) != split {
            continue;
        }
        let format = music
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        cases.push(Case {
            name,
            music,
            format,
            bpm: timing.bpms[0].1,
            beat0: -timing.offset,
        });
    }
    // Spread the sample evenly over packs.
    let step = (cases.len() as f64 / max as f64).max(1.0);
    let picked: Vec<&Case> = (0..max.min(cases.len()))
        .map(|i| &cases[(i as f64 * step) as usize])
        .collect();
    eprintln!(
        "{} constant-BPM songs in split {split}, evaluating {}",
        cases.len(),
        picked.len()
    );

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
                    let exact = (a.grid.bpm - case.bpm).abs() < 0.05;
                    let period = 60.0 / case.bpm;
                    let phase_ms =
                        exact.then(|| beat_error(a.grid.beat0, case.beat0, period) * period * 1000.0);
                    let raw_on_beat =
                        exact.then(|| beat_error(a.diagnostics.raw_phase, case.beat0, period).abs() < 0.25);
                    let o = Outcome {
                        name: case.name.clone(),
                        format: case.format.clone(),
                        truth_bpm: case.bpm,
                        bpm: a.grid.bpm,
                        aubio_bpm: a.aubio_bpm,
                        phase_ms,
                        raw_on_beat,
                        on_beat_prob: a.diagnostics.on_beat,
                        features: a.diagnostics.half_features,
                        tempo_candidates: a.diagnostics.tempo_candidates.clone(),
                    };
                    eprintln!(
                        "[{i:>3}] {:<40} truth {:>7.2} ours {:>7.2} {:<7} {}",
                        case.name.chars().take(40).collect::<String>(),
                        case.bpm,
                        a.grid.bpm,
                        category(&o),
                        phase_ms.map_or("-".into(), |p| format!("{p:+.1} ms"))
                    );
                    results.lock().unwrap().push(o);
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by(|a, b| a.name.cmp(&b.name));

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/eval");
    std::fs::create_dir_all(&out_dir)?;
    let mut csv = String::from(
        "name,format,truth_bpm,bpm,aubio_bpm,category,phase_ms,raw_on_beat,on_beat_prob,f_kick,f_lowmid,f_mid,f_high,f_mix,tempo_candidates\n",
    );
    for r in &results {
        let _ = write!(
            csv,
            "\"{}\",{},{},{},{:.3},{},{},{},{:.4}",
            r.name.replace('"', "'"),
            r.format,
            r.truth_bpm,
            r.bpm,
            r.aubio_bpm,
            category(r),
            r.phase_ms.map_or(String::new(), |p| format!("{p:.2}")),
            r.raw_on_beat.map_or(String::new(), |b| (b as u8).to_string()),
            r.on_beat_prob
        );
        for f in r.features {
            let _ = write!(csv, ",{f:.5}");
        }
        // "bpm:score;bpm:score;..." (no commas, one CSV cell)
        let cands: Vec<String> = r
            .tempo_candidates
            .iter()
            .map(|(b, s)| format!("{b:.3}:{s:.5}"))
            .collect();
        let _ = write!(csv, ",{}", cands.join(";"));
        csv.push('\n');
    }
    std::fs::write(out_dir.join(format!("sync-{tag}.csv")), csv)?;

    let mut report = String::new();
    let _ = writeln!(
        report,
        "split {split}, stems {use_stems}, songs {}",
        results.len()
    );
    for (label, filter) in [("all", None), ("mp3", Some("mp3")), ("ogg", Some("ogg"))] {
        let rs: Vec<&Outcome> = results
            .iter()
            .filter(|r| filter.is_none_or(|f| r.format == f))
            .collect();
        if rs.is_empty() {
            continue;
        }
        let n = rs.len() as f64;
        let count = |c: &str| rs.iter().filter(|r| category(r) == c).count();
        let pct = |k: usize| 100.0 * k as f64 / n;
        let exact = rs.len() - count("octave") - count("bpm");
        let mut phases: Vec<f64> = rs.iter().filter_map(|r| r.phase_ms).collect();
        phases.sort_by(f64::total_cmp);
        let ok: Vec<f64> = phases.iter().copied().filter(|p| p.abs() < 30.0).collect();
        let mut abs: Vec<f64> = phases.iter().map(|p| p.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let q = |v: &[f64], p: f64| {
            v.get(((v.len().max(1) - 1) as f64 * p) as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        let _ = writeln!(
            report,
            "[{label}] n={} | BPM exact {:.1}% octave {:.1}% other {:.1}% | on exact BPM: ok {:.1}% half-beat {:.1}% quarter {:.1}% other {:.1}% | |phase| median {:.1} ms, <20 ms {:.1}% | signed median of ok songs {:+.1} ms",
            rs.len(),
            pct(exact),
            pct(count("octave")),
            pct(count("bpm")),
            100.0 * count("ok") as f64 / exact.max(1) as f64,
            100.0 * count("half") as f64 / exact.max(1) as f64,
            100.0 * count("quarter") as f64 / exact.max(1) as f64,
            100.0 * count("phase") as f64 / exact.max(1) as f64,
            q(&abs, 0.5),
            100.0 * abs.iter().filter(|a| **a < 20.0).count() as f64 / abs.len().max(1) as f64,
            q(&ok, 0.5),
        );
    }
    print!("{report}");
    // Summaries are versioned (eval/), per-song CSVs are not (target/eval/).
    let summary_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("eval");
    std::fs::create_dir_all(&summary_dir)?;
    std::fs::write(summary_dir.join(format!("sync-{tag}.txt")), &report)?;
    eprintln!("wrote {}", out_dir.join(format!("sync-{tag}.csv")).display());
    Ok(())
}
