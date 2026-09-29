//! Longer arrow patterns: generated charts vs the human charts of a style.
//!
//! Human reference: every dance-single chart below SONGS_DIR. Ours: every difficulty
//! generated (seed 1, the style's model and options) for MAX songs of SONGS_DIR. For
//! each chart, on runs of single-panel rows (consecutive rows at most an 8th apart):
//! - stairs /100 rows: 4 rows on the 4 panels, from one side panel to the other
//!   (L D U R, L U D R and mirrors);
//! - drills /100 rows: rows inside alternations of two panels at least 5 rows long;
//! - candles /100 rows: up ↔ down with a side panel in between (U L D, D R U…);
//! - 16th run length: mean length of the runs of rows a 16th apart (at least 2 rows);
//! - 4-row diversity, on all rows: distinct 4-row windows / windows;
//! - 8-row repeat, on all rows: share of 8-row windows already seen in the chart.
//!
//! Per difficulty: our median against the human p10–p50–p90.
//!
//! Summary written to `eval/patterns-<STYLE>.txt`.
//!
//! Usage: cargo run --release --example eval_patterns -- SONGS_DIR [--style S] [--max N]

use clap::ValueEnum;
use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::audio::decode_file;
use itg_charter::chart::generate;
use itg_charter::difficulty::Difficulty;
use itg_charter::model::{ChartFeatures, Quantiles, find_simfiles};
use itg_charter::parity::Layout;
use itg_charter::simfile::{NoteRow, Simfile, Timing};
use itg_charter::style::Style;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

const NAMES: [&str; 6] = [
    "stairs/100",
    "drills/100",
    "candles/100",
    "16th run",
    "4-row diversity",
    "8-row repeat",
];
const NF: usize = NAMES.len();
const L: u8 = 1;
const D: u8 = 2;
const U: u8 = 4;
const R: u8 = 8;

/// Pattern features of a chart from its (mask, position in 48ths) sequence.
fn patterns(seq: &[(u8, u32)]) -> [f64; NF] {
    let n = seq.len().max(1) as f64;
    // Runs of single-panel rows at most an 8th apart.
    let mut runs: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    for (i, &(m, pos)) in seq.iter().enumerate() {
        let close = i > 0 && pos - seq[i - 1].1 <= 24;
        if m.count_ones() != 1 || !close {
            if cur.len() > 1 {
                runs.push(std::mem::take(&mut cur));
            }
            cur.clear();
        }
        if m.count_ones() == 1 {
            cur.push(m);
        }
    }
    if cur.len() > 1 {
        runs.push(cur);
    }
    let (mut stairs, mut drill_rows, mut candles) = (0usize, 0usize, 0usize);
    for r in &runs {
        for w in r.windows(4) {
            let distinct = w.iter().fold(0u8, |a, &m| a | m) == 15;
            let sides = (w[0] == L && w[3] == R) || (w[0] == R && w[3] == L);
            stairs += (distinct && sides) as usize;
        }
        for w in r.windows(3) {
            let ud = (w[0] == U && w[2] == D) || (w[0] == D && w[2] == U);
            candles += (ud && (w[1] == L || w[1] == R)) as usize;
        }
        // Maximal alternations a b a b … (a ≠ b).
        let mut start = 0;
        for i in 1..=r.len() {
            let alt = i < r.len() && r[i] != r[i - 1] && (i - start < 2 || r[i] == r[i - 2]);
            if !alt {
                if i - start >= 5 {
                    drill_rows += i - start;
                }
                start = if i < r.len() && r[i] != r[i - 1] { i - 1 } else { i };
            }
        }
    }
    // Runs of rows a 16th apart.
    let mut sixteenth = Vec::new();
    let mut len = 1;
    for i in 1..=seq.len() {
        if i < seq.len() && seq[i].1 - seq[i - 1].1 == 12 {
            len += 1;
        } else {
            if len >= 2 {
                sixteenth.push(len as f64);
            }
            len = 1;
        }
    }
    let masks: Vec<u8> = seq.iter().map(|s| s.0).collect();
    let windows4: Vec<&[u8]> = masks.windows(4).collect();
    let distinct4 = windows4.iter().collect::<HashSet<_>>().len();
    let mut seen = HashSet::new();
    let (mut repeats, mut windows8) = (0usize, 0usize);
    for w in masks.windows(8) {
        windows8 += 1;
        repeats += (!seen.insert(w)) as usize;
    }
    [
        100.0 * stairs as f64 / n,
        100.0 * drill_rows as f64 / n,
        100.0 * candles as f64 / n,
        if sixteenth.is_empty() {
            0.0
        } else {
            sixteenth.iter().sum::<f64>() / sixteenth.len() as f64
        },
        distinct4 as f64 / windows4.len().max(1) as f64,
        repeats as f64 / windows8.max(1) as f64,
    ]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: eval_patterns SONGS_DIR [--style S] [--max N]";
    let dir = PathBuf::from(args.first().expect(usage));
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1));
    let style = opt("--style")
        .map_or(Ok(Style::default()), |s| Style::from_str(s, true))
        .map_err(anyhow::Error::msg)?;
    let max: usize = opt("--max").and_then(|m| m.parse().ok()).unwrap_or(40);
    let (model, gen_opts) = (style.model()?, style.gen_options());
    let layout = Layout::dance_single();

    // Human charts, and the audio files of the songs.
    let mut human: Vec<Vec<[f64; NF]>> = vec![Vec::new(); 5];
    let mut songs = Vec::new();
    for path in find_simfiles(&dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
        if sim.is_generated() {
            continue;
        }
        for chart in sim.charts.iter().filter(|c| c.steps_type == "dance-single") {
            let Ok(diff) = Difficulty::parse(&chart.difficulty) else {
                continue;
            };
            let (Ok(timing), Ok(rows)) = (sim.timing(chart), chart.rows()) else {
                continue;
            };
            let f = ChartFeatures::compute(&rows, &timing, &layout);
            if f.rows >= 16 {
                human[diff.index()].push(patterns(&f.sequence));
            }
        }
        if let Some(m) = sim.tag("MUSIC").filter(|m| !m.is_empty()) {
            let music = path.parent().unwrap().join(m);
            if music.exists() {
                songs.push(music);
            }
        }
    }
    let step = (songs.len() as f64 / max as f64).max(1.0);
    let picked: Vec<PathBuf> = (0..max.min(songs.len()))
        .map(|i| songs[(i as f64 * step) as usize].clone())
        .collect();

    // Generated charts (song order does not matter: only medians are reported).
    let ours: Mutex<Vec<Vec<[f64; NF]>>> = Mutex::new(vec![Vec::new(); 5]);
    let next = Mutex::new(0usize);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| {
                loop {
                    let i = {
                        let mut n = next.lock().unwrap();
                        *n += 1;
                        *n - 1
                    };
                    let Some(path) = picked.get(i) else { break };
                    let Ok(audio) = decode_file(path) else { continue };
                    let a = SongAnalysis::compute(&audio, None, &AnalysisOptions::default());
                    let timing = Timing::constant(a.grid.bpm, a.grid.offset());
                    for d in Difficulty::ALL {
                        let c = generate(&a, &model, d, 1, &gen_opts);
                        let rows: Vec<NoteRow> = c
                            .rows
                            .iter()
                            .map(|(p, cells)| NoteRow {
                                beat: *p as f64 / 48.0,
                                cells: cells.to_vec(),
                            })
                            .collect();
                        let f = ChartFeatures::compute(&rows, &timing, &layout);
                        ours.lock().unwrap()[d.index()].push(patterns(&f.sequence));
                    }
                    eprintln!("[{i}] {}", path.display());
                }
            });
        }
    });
    let ours = ours.into_inner().unwrap();

    let mut report = String::new();
    writeln!(
        report,
        "{} --style {}: {} songs generated",
        dir.display(),
        style.name(),
        picked.len()
    )?;
    let mut flags = 0;
    for d in Difficulty::ALL {
        let i = d.index();
        writeln!(
            report,
            "== {} (human {} charts, ours {})",
            d.name(),
            human[i].len(),
            ours[i].len()
        )?;
        for (f, name) in NAMES.iter().enumerate() {
            let h = Quantiles::of(&human[i].iter().map(|x| x[f]).collect::<Vec<_>>());
            let o = Quantiles::of(&ours[i].iter().map(|x| x[f]).collect::<Vec<_>>()).p50;
            let out = o < h.p10 || o > h.p90;
            flags += out as usize;
            writeln!(
                report,
                "{name:>16}: ours median {o:7.3} | human p10 {:7.3} p50 {:7.3} p90 {:7.3}{}",
                h.p10,
                h.p50,
                h.p90,
                if out { "  <-- outside human p10-p90" } else { "" }
            )?;
        }
    }
    writeln!(report, "outside human range: {flags}")?;
    print!("{report}");
    std::fs::write(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("eval/patterns-{}.txt", style.name())),
        &report,
    )?;
    Ok(())
}
