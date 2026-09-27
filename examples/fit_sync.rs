//! Fits `HALF_BEAT_WEIGHTS` (analysis.rs) on the CSV written by `eval_sync`.
//!
//! Each exact-BPM song gives one example: the half-beat features of the phase found
//! by the tempo fit, labelled with whether that phase is on the human beat grid.
//! A logistic regression (L2-regularized, batch gradient descent, deterministic) is
//! fitted on the train CSV and its accuracy reported on the test CSV, next to the
//! accuracy of the weights currently in the code.
//!
//! Usage: cargo run --release --example fit_sync -- target/eval/sync-train.csv target/eval/sync-test-baseline.csv

use itg_charter::analysis::{HALF_BEAT_FEATURES, HALF_BEAT_WEIGHTS, on_beat_probability};

type Example = ([f64; HALF_BEAT_FEATURES], bool);

fn load(path: &str) -> anyhow::Result<Vec<Example>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or_default().split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).expect(name);
    let label = col("raw_on_beat");
    let first = col("f_kick");
    let mut out = Vec::new();
    for line in lines {
        // The name is the only quoted field and comes first: split after it.
        let rest = &line[line.rfind('"').map_or(0, |i| i + 2)..];
        let cells: Vec<&str> = std::iter::once("").chain(rest.split(',')).collect();
        let Some(l) = cells.get(label).filter(|l| !l.is_empty()) else {
            continue;
        };
        let mut f = [0.0; HALF_BEAT_FEATURES];
        for (k, v) in f.iter_mut().enumerate() {
            *v = cells[first + k].parse()?;
        }
        out.push((f, *l == "1"));
    }
    Ok(out)
}

fn accuracy(data: &[Example], w: &[f64; HALF_BEAT_FEATURES + 1]) -> f64 {
    let ok = data
        .iter()
        .filter(|(f, y)| (on_beat_probability(f, w) >= 0.5) == *y)
        .count();
    100.0 * ok as f64 / data.len().max(1) as f64
}

fn fit(data: &[Example], l2: f64) -> [f64; HALF_BEAT_FEATURES + 1] {
    let mut w = [0.0; HALF_BEAT_FEATURES + 1];
    let n = data.len() as f64;
    for _ in 0..20_000 {
        let mut grad = [0.0; HALF_BEAT_FEATURES + 1];
        for (f, y) in data {
            let err = on_beat_probability(f, &w) - if *y { 1.0 } else { 0.0 };
            grad[0] += err;
            for k in 0..HALF_BEAT_FEATURES {
                grad[k + 1] += err * f[k];
            }
        }
        for k in 0..w.len() {
            let reg = if k == 0 { 0.0 } else { l2 * w[k] };
            w[k] -= 0.5 * (grad[k] / n + reg);
        }
    }
    w
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let train = load(args.first().expect("usage: fit_sync TRAIN.csv [TEST.csv]"))?;
    let test = match args.get(1) {
        Some(p) => load(p)?,
        None => Vec::new(),
    };
    let on = train.iter().filter(|e| e.1).count();
    println!(
        "train examples: {} ({on} on-beat), test examples: {}",
        train.len(),
        test.len()
    );
    println!(
        "current weights {HALF_BEAT_WEIGHTS:?}: train {:.1}%, test {:.1}%",
        accuracy(&train, &HALF_BEAT_WEIGHTS),
        accuracy(&test, &HALF_BEAT_WEIGHTS)
    );
    for l2 in [0.0, 0.01, 0.1] {
        let w = fit(&train, l2);
        println!(
            "l2 {l2}: weights [{}]: train {:.1}%, test {:.1}%",
            w.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(", "),
            accuracy(&train, &w),
            accuracy(&test, &w)
        );
    }
    Ok(())
}
