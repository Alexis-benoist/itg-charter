//! How much do earlier rows tell about the next arrow? (context length of the n-gram)
//!
//! On the dance-single charts of SONGS_DIR, split by title (`music::split_of`), arrow
//! models of order 1 to 5 (the current row given 0 to 4 previous rows, and the gap
//! bucket before it, per difficulty) are counted on the train split and score the test
//! split in bits per row. Same smoothing as `Model` (order 3 = the embedded model's
//! n-gram): Dirichlet back-off to the next shorter context, α = 4, and every train
//! chart also counted mirrored. A learning curve (1/4, 1/2, all of the train charts,
//! every k-th chart) shows whether longer contexts need more data.
//!
//! Summary written to `eval/context-<TAG>.txt`.
//!
//! Usage: cargo run --release --example eval_context -- SONGS_DIR [--tag TAG]

use itg_charter::difficulty::Difficulty;
use itg_charter::model::{ChartFeatures, GAP_BUCKETS, find_simfiles, gap_bucket, mirror};
use itg_charter::music::split_of;
use itg_charter::parity::Layout;
use itg_charter::simfile::Simfile;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

const MAX_ORDER: usize = 5;
const ALPHA: f64 = 4.0;

/// (difficulty, gap bucket, mask) of every row of a chart.
type Seq = (usize, Vec<(usize, u8)>);

/// Counts of (context, arrow) and of contexts, per order. The key packs the difficulty,
/// the gap bucket and the `order - 1` previous masks.
struct Counts {
    joint: Vec<HashMap<u64, u32>>,
    total: Vec<HashMap<u64, u32>>,
}

fn key(d: usize, g: usize, prev: &[u8], order: usize) -> u64 {
    let mut k = (d * GAP_BUCKETS + g) as u64;
    for &m in &prev[prev.len() - (order - 1)..] {
        k = k << 4 | m as u64;
    }
    k << 4
}

/// Previous masks before row `i`, padded with 0 (no row) at the start.
fn history(rows: &[(usize, u8)], i: usize) -> [u8; MAX_ORDER - 1] {
    let mut h = [0u8; MAX_ORDER - 1];
    for (j, slot) in h.iter_mut().rev().enumerate() {
        if i > j {
            *slot = rows[i - 1 - j].1;
        }
    }
    h
}

impl Counts {
    fn new(train: &[&Seq]) -> Counts {
        let mut c = Counts {
            joint: vec![HashMap::new(); MAX_ORDER + 1],
            total: vec![HashMap::new(); MAX_ORDER + 1],
        };
        for (d, rows) in train {
            for mirrored in [false, true] {
                let rows: Vec<(usize, u8)> = rows
                    .iter()
                    .map(|&(g, m)| (g, if mirrored { mirror(m) } else { m }))
                    .collect();
                for i in 0..rows.len() {
                    let (g, m) = rows[i];
                    let h = history(&rows, i);
                    for order in 1..=MAX_ORDER {
                        let k = key(*d, g, &h, order);
                        *c.joint[order].entry(k | m as u64).or_default() += 1;
                        *c.total[order].entry(k).or_default() += 1;
                    }
                }
            }
        }
        c
    }

    fn prob(&self, d: usize, g: usize, h: &[u8], m: u8, order: usize) -> f64 {
        let mut p = 1.0 / 15.0;
        for o in 1..=order {
            let k = key(d, g, h, o);
            let n = self.joint[o].get(&(k | m as u64)).copied().unwrap_or(0) as f64;
            let t = self.total[o].get(&k).copied().unwrap_or(0) as f64;
            p = (n + ALPHA * p) / (t + ALPHA);
        }
        p
    }

    /// Bits per row of `test` at every order (index 0 unused).
    fn score(&self, test: &[&Seq]) -> [f64; MAX_ORDER + 1] {
        let mut bits = [0.0; MAX_ORDER + 1];
        let mut n = 0usize;
        for (d, rows) in test {
            for i in 0..rows.len() {
                let (g, m) = rows[i];
                let h = history(rows, i);
                for (order, b) in bits.iter_mut().enumerate().skip(1) {
                    *b -= self.prob(*d, g, &h, m, order).log2();
                }
                n += 1;
            }
        }
        bits.map(|b| b / n.max(1) as f64)
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: eval_context SONGS_DIR [--tag TAG]";
    let dir = PathBuf::from(args.first().expect(usage));
    let tag = match args.get(1).map(String::as_str) {
        Some("--tag") => args.get(2).expect(usage).clone(),
        _ => "test".into(),
    };
    let layout = Layout::dance_single();
    let (mut train, mut test): (Vec<Seq>, Vec<Seq>) = (Vec::new(), Vec::new());
    for path in find_simfiles(&dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
        if sim.is_generated() {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let title = sim
            .tag("TITLE")
            .filter(|t| !t.is_empty())
            .unwrap_or(&name)
            .to_string();
        let split = split_of(&title);
        for chart in sim.charts.iter().filter(|c| c.steps_type == "dance-single") {
            let Ok(diff) = Difficulty::parse(&chart.difficulty) else {
                continue;
            };
            let (Ok(timing), Ok(rows)) = (sim.timing(chart), chart.rows()) else {
                continue;
            };
            let f = ChartFeatures::compute(&rows, &timing, &layout);
            if f.rows < 16 {
                continue;
            }
            let mut last = None;
            let seq: Vec<(usize, u8)> = f
                .sequence
                .iter()
                .map(|&(m, pos)| {
                    let g = last.map_or(GAP_BUCKETS - 1, |l| gap_bucket(pos - l));
                    last = Some(pos);
                    (g, m)
                })
                .collect();
            let s = (diff.index(), seq);
            if split == "train" {
                train.push(s)
            } else {
                test.push(s)
            }
        }
    }
    let test_refs: Vec<&Seq> = test.iter().collect();
    let rows: usize = test.iter().map(|s| s.1.len()).sum();
    let mut report = String::new();
    writeln!(
        report,
        "{}: train {} charts, test {} charts ({rows} rows); bits per row, uniform = {:.3}",
        dir.display(),
        train.len(),
        test.len(),
        15f64.log2()
    )?;
    let header: Vec<String> = (1..=MAX_ORDER)
        .map(|o| format!("order {o} ({} prev)", o - 1))
        .collect();
    writeln!(report, "train share | {}", header.join(" | "))?;
    for step in [4, 2, 1] {
        let sub: Vec<&Seq> = train.iter().step_by(step).collect();
        let bits = Counts::new(&sub).score(&test_refs);
        let cols: Vec<String> = (1..=MAX_ORDER).map(|o| format!("{:>14.3}", bits[o])).collect();
        writeln!(
            report,
            "1/{step} ({:>5} charts) | {}",
            sub.len(),
            cols.join(" | ")
        )?;
    }
    print!("{report}");
    std::fs::write(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("eval/context-{tag}.txt")),
        &report,
    )?;
    Ok(())
}
