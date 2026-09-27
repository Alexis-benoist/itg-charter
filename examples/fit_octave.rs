//! Fits the log-normal tempo prior (`TEMPO_PRIOR_CENTER`, `TEMPO_PRIOR_OCTAVES` in
//! analysis.rs) on the CSV written by `eval_sync`.
//!
//! For every song, `eval_sync` records the refined candidate tempos with their
//! audio-only comb score; the analysis keeps the candidate with the best
//! score × prior. This grid-searches the prior's centre and width that maximize the
//! number of songs whose chosen candidate is the human BPM (train CSV), and reports
//! the result on the test CSV, next to the current constants and to the prior learned
//! from the tempos human charters chose (model).
//!
//! Usage: cargo run --release --example fit_octave -- target/eval/sync-train.csv target/eval/sync-test.csv

use itg_charter::analysis::{TEMPO_PRIOR_CENTER, TEMPO_PRIOR_OCTAVES, choose_candidate};
use itg_charter::model::embedded_bpm_prior;

struct Song {
    truth: f64,
    candidates: Vec<(f64, f64)>,
}

fn load(path: &str) -> anyhow::Result<Vec<Song>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or_default().split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).expect(name);
    let (truth, cands) = (col("truth_bpm"), col("tempo_candidates"));
    let mut out = Vec::new();
    for line in lines {
        // The name is the only quoted field and comes first.
        let rest = &line[line.rfind('"').map_or(0, |i| i + 2)..];
        let cells: Vec<&str> = std::iter::once("").chain(rest.split(',')).collect();
        let candidates: Vec<(f64, f64)> = cells[cands]
            .split(';')
            .filter_map(|c| {
                let (b, s) = c.split_once(':')?;
                Some((b.parse().ok()?, s.parse().ok()?))
            })
            .collect();
        if candidates.is_empty() {
            continue;
        }
        out.push(Song {
            truth: cells[truth].parse()?,
            candidates,
        });
    }
    Ok(out)
}

fn accuracy(songs: &[Song], prior: impl Fn(f64) -> f64 + Copy) -> f64 {
    let ok = songs
        .iter()
        .filter(|s| (choose_candidate(&s.candidates, prior) - s.truth).abs() < 0.06)
        .count();
    100.0 * ok as f64 / songs.len().max(1) as f64
}

fn lognormal(center: f64, octaves: f64) -> impl Fn(f64) -> f64 + Copy {
    move |bpm: f64| {
        let x = (bpm / center).log2() / octaves;
        (-0.5 * x * x).exp()
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let train = load(args.first().expect("usage: fit_octave TRAIN.csv [TEST.csv]"))?;
    let test = match args.get(1) {
        Some(p) => load(p)?,
        None => Vec::new(),
    };
    println!("train songs: {}, test songs: {}", train.len(), test.len());
    let report = |name: &str, prior: &dyn Fn(f64) -> f64| {
        let p = |b: f64| prior(b);
        println!(
            "{name:<42} train {:>5.1}%  test {:>5.1}%",
            accuracy(&train, p),
            accuracy(&test, p)
        );
    };
    report(
        &format!("current (centre {TEMPO_PRIOR_CENTER}, {TEMPO_PRIOR_OCTAVES} oct)"),
        &lognormal(TEMPO_PRIOR_CENTER, TEMPO_PRIOR_OCTAVES),
    );
    report("no prior", &|_| 1.0);
    report("learned tempo histogram (model)", &embedded_bpm_prior);
    let mut best = (0.0, 0.0, f64::MIN);
    for c in (90..=200).step_by(2) {
        for w in (4..=30).map(|k| k as f64 * 0.05) {
            let acc = accuracy(&train, lognormal(c as f64, w));
            // Ties: prefer the widest prior (least assumption).
            if acc > best.2 + 1e-9 || ((acc - best.2).abs() < 1e-9 && w > best.1) {
                best = (c as f64, w, acc);
            }
        }
    }
    report(
        &format!("best on train (centre {}, {:.2} oct)", best.0, best.1),
        &lognormal(best.0, best.1),
    );
    Ok(())
}
