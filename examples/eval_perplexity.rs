//! How well does the arrow n-gram predict held-out human charts?
//!
//! Scores every dance-single chart of the test split of TEST_DIR (see `music::split_of`)
//! in bits per arrow, -log2 P(row | two previous rows, gap), under models trained with
//! `Model::train` on the train split only of each TRAIN_DIR. The simfiles of that split
//! are copied to `target/eval/perplexity/` first, so nothing of the test split leaks in.
//! With `--curve`, the last TRAIN_DIR is also trained on 1/8, 1/4 and 1/2 of its train
//! songs (every k-th song, in path order): a learning curve.
//!
//! The embedded model is scored too, for reference only: it was trained on every song,
//! test split included.
//!
//! Summary written to `eval/perplexity-<TAG>.txt`.
//!
//! Usage: cargo run --release --example eval_perplexity -- TEST_DIR
//!        [--train DIR]... [--curve] [--tag TAG]

use itg_charter::difficulty::Difficulty;
use itg_charter::model::{ChartFeatures, GAP_BUCKETS, MASKS, Model, find_simfiles, gap_bucket};
use itg_charter::music::split_of;
use itg_charter::parity::Layout;
use itg_charter::simfile::Simfile;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// A human chart: difficulty and (mask, position in 48ths) of each row.
struct TestChart {
    diff: Difficulty,
    sequence: Vec<(u8, u32)>,
}

/// Title used for the split, as in the other evaluations.
fn title_of(sim: &Simfile, path: &Path) -> String {
    let name = path.file_stem().unwrap().to_string_lossy().into_owned();
    sim.tag("TITLE")
        .filter(|t| !t.is_empty())
        .unwrap_or(&name)
        .to_string()
}

fn load_split(dir: &Path, split: &str) -> Vec<(PathBuf, Simfile)> {
    find_simfiles(dir)
        .into_iter()
        .filter_map(|p| Simfile::load(&p).ok().map(|s| (p, s)))
        .filter(|(p, s)| !s.is_generated() && split_of(&title_of(s, p)) == split)
        .collect()
}

fn test_charts(dir: &Path) -> Vec<TestChart> {
    let layout = Layout::dance_single();
    let mut out = Vec::new();
    for (_, sim) in load_split(dir, "test") {
        for chart in sim.charts.iter().filter(|c| c.steps_type == "dance-single") {
            let Ok(diff) = Difficulty::parse(&chart.difficulty) else {
                continue;
            };
            let (Ok(timing), Ok(rows)) = (sim.timing(chart), chart.rows()) else {
                continue;
            };
            let f = ChartFeatures::compute(&rows, &timing, &layout);
            if f.rows >= 16 {
                out.push(TestChart {
                    diff,
                    sequence: f.sequence,
                });
            }
        }
    }
    out
}

/// Copies every `step`-th train simfile of `files` into `dir` (one folder per song, so
/// `find_simfiles` keeps its .ssc-over-.sm choice) and trains on them.
fn train_on(files: &[(PathBuf, Simfile)], step: usize, dir: &Path) -> anyhow::Result<(Model, usize)> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    let mut n = 0;
    for (i, (path, _)) in files.iter().enumerate().filter(|(i, _)| i % step == 0) {
        let song = dir.join(format!("{i:06}"));
        std::fs::create_dir_all(&song)?;
        std::fs::copy(path, song.join(path.file_name().unwrap()))?;
        n += 1;
    }
    Ok((Model::train(dir, |_, _| {})?, n))
}

/// Bits per arrow, per difficulty and overall: (sum of bits, arrows).
fn score(model: &Model, charts: &[TestChart]) -> [(f64, usize); 6] {
    let mut acc = [(0.0, 0usize); 6];
    for c in charts {
        let (mut p2, mut p1, mut last) = (0u8, 0u8, None);
        for &(mask, pos) in &c.sequence {
            let g = last.map_or(GAP_BUCKETS - 1, |l| gap_bucket(pos - l));
            let bits = -model.prob(c.diff, g, p2, p1, mask).log2();
            for k in [c.diff.index(), 5] {
                acc[k].0 += bits;
                acc[k].1 += 1;
            }
            (p2, p1, last) = (p1, mask, Some(pos));
        }
    }
    acc
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: eval_perplexity TEST_DIR [--train DIR]... [--curve] [--tag TAG]";
    let test_dir = PathBuf::from(args.first().expect(usage));
    let mut train_dirs = Vec::new();
    let (mut curve, mut tag) = (false, "test".to_string());
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--train" => {
                train_dirs.push(PathBuf::from(args.get(i + 1).expect(usage)));
                i += 1;
            }
            "--curve" => curve = true,
            "--tag" => {
                tag = args.get(i + 1).expect(usage).clone();
                i += 1;
            }
            a => panic!("unknown argument {a}; {usage}"),
        }
        i += 1;
    }
    if train_dirs.is_empty() {
        train_dirs.push(test_dir.clone());
    }

    let charts = test_charts(&test_dir);
    let arrows: usize = charts.iter().map(|c| c.sequence.len()).sum();
    let mut report = String::new();
    writeln!(
        report,
        "test split of {}: {} charts, {arrows} rows; uniform over the {} masks = {:.3} bits",
        test_dir.display(),
        charts.len(),
        MASKS - 1,
        ((MASKS - 1) as f64).log2()
    )?;
    let header: Vec<&str> = Difficulty::ALL.iter().map(|d| d.name()).collect();
    writeln!(
        report,
        "bits per row | {:>9} | all      | trained on",
        header.join(" | ")
    )?;

    let mut runs: Vec<(String, Model)> = Vec::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/eval/perplexity");
    for (k, dir) in train_dirs.iter().enumerate() {
        let files = load_split(dir, "train");
        let last = k + 1 == train_dirs.len();
        let steps: &[usize] = if curve && last { &[8, 4, 2, 1] } else { &[1] };
        for &step in steps {
            let (model, n) = train_on(&files, step, &root.join(format!("{k}-{step}")))?;
            let charts_trained: u32 = model.stats.iter().map(|s| s.charts).sum();
            let name = format!(
                "{} train 1/{step} ({n} simfiles, {charts_trained} charts)",
                dir.display()
            );
            eprintln!("trained: {name}");
            runs.push((name, model));
        }
    }
    runs.push((
        "embedded model (all songs, test split included)".into(),
        Model::embedded()?,
    ));

    for (name, model) in &runs {
        let acc = score(model, &charts);
        let cols: Vec<String> = acc[..5]
            .iter()
            .map(|(b, n)| format!("{:>9.3}", b / *n as f64))
            .collect();
        writeln!(
            report,
            "             | {} | {:>8.3} | {name}",
            cols.join(" | "),
            acc[5].0 / acc[5].1 as f64
        )?;
    }
    print!("{report}");
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("eval/perplexity-{tag}.txt"));
    std::fs::write(out, &report)?;
    Ok(())
}
