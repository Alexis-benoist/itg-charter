//! Do musical cues predict the arrows human charters chose?
//!
//! On the train half of the library (see `music::split_of`), every note of every
//! dance-single chart is aligned with the audio and described by its musical context
//! (pitch movement, hit type, accent — see `music.rs`). The arrow is summarised as a
//! class: L, D, U, R (single panel), same panel again, or jump.
//!
//! Reported, per difficulty:
//! - mutual information I(context; arrow) and, since the n-gram already knows the
//!   previous arrow, I(context; arrow | previous arrow), in bits, with the
//!   entropy H(arrow | previous) for scale; also per cue;
//! - P(arrow | pitch movement) and P(arrow | hit type) for single notes;
//! - correlation between pitch change (semitones) and vertical / horizontal movement.
//!
//! Summary written to `eval/music-signal-<split>.txt`.
//!
//! Usage: cargo run --release --example music_signal -- SONGS_DIR [--max N] [--split train|test]

use itg_charter::analysis::Layer;
use itg_charter::audio::decode_file;
use itg_charter::difficulty::Difficulty;
use itg_charter::model::find_simfiles;
use itg_charter::music::{CONTEXTS, Context, MusicCues, split_of};
use itg_charter::simfile::{Cell, Simfile};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

const CLASSES: usize = 6; // L D U R same jump
const CLASS_NAMES: [&str; CLASSES] = ["L", "D", "U", "R", "same", "jump"];
const PREV: usize = CLASSES + 1; // + start

#[derive(Clone)]
struct Stats {
    /// [previous class][context][class]
    joint: Vec<u64>,
    /// Pitch change vs vertical / horizontal movement, single → single, both voiced.
    pairs: Vec<(f64, f64, f64)>,
}

impl Stats {
    fn new() -> Stats {
        Stats {
            joint: vec![0; PREV * CONTEXTS * CLASSES],
            pairs: Vec::new(),
        }
    }
    fn add(&mut self, other: &Stats) {
        for (a, b) in self.joint.iter_mut().zip(&other.joint) {
            *a += b;
        }
        self.pairs.extend_from_slice(&other.pairs);
    }
}

fn class_of(mask: u8, prev_mask: Option<u8>) -> usize {
    if mask.count_ones() >= 2 {
        5
    } else if prev_mask == Some(mask) {
        4
    } else {
        mask.trailing_zeros() as usize
    }
}

/// (x, y) of a panel: L (-1,0) D (0,-1) U (0,1) R (1,0).
fn xy(mask: u8) -> (f64, f64) {
    match mask {
        1 => (-1.0, 0.0),
        2 => (0.0, -1.0),
        4 => (0.0, 1.0),
        _ => (1.0, 0.0),
    }
}

/// Mutual information (bits) between the middle and last index of a
/// [cond][x][y] table, conditioned on the first.
fn cond_mi(t: &[f64], nc: usize, nx: usize, ny: usize) -> f64 {
    let total: f64 = t.iter().sum();
    let mut mi = 0.0;
    for c in 0..nc {
        let block = &t[c * nx * ny..(c + 1) * nx * ny];
        let n: f64 = block.iter().sum();
        if n == 0.0 {
            continue;
        }
        let px: Vec<f64> = (0..nx)
            .map(|x| block[x * ny..(x + 1) * ny].iter().sum::<f64>() / n)
            .collect();
        let py: Vec<f64> = (0..ny)
            .map(|y| (0..nx).map(|x| block[x * ny + y]).sum::<f64>() / n)
            .collect();
        for x in 0..nx {
            for y in 0..ny {
                let p = block[x * ny + y] / n;
                if p > 0.0 {
                    mi += (n / total) * p * (p / (px[x] * py[y])).log2();
                }
            }
        }
    }
    mi
}

/// Conditional entropy H(y | cond) in bits of a [cond][y] table.
fn cond_entropy(t: &[f64], nc: usize, ny: usize) -> f64 {
    let total: f64 = t.iter().sum();
    let mut h = 0.0;
    for c in 0..nc {
        let row = &t[c * ny..(c + 1) * ny];
        let n: f64 = row.iter().sum();
        for &v in row {
            if v > 0.0 {
                h -= (v / total) * (v / n).log2();
            }
        }
    }
    h
}

fn correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let (mx, my) = (
        pairs.iter().map(|p| p.0).sum::<f64>() / n,
        pairs.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let sxy: f64 = pairs.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let sxx: f64 = pairs.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let syy: f64 = pairs.iter().map(|p| (p.1 - my).powi(2)).sum();
    sxy / (sxx * syy).sqrt()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(
        args.first()
            .expect("usage: music_signal SONGS_DIR [--max N] [--split S]"),
    );
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let max: usize = opt("--max").and_then(|m| m.parse().ok()).unwrap_or(400);
    let split = opt("--split").unwrap_or_else(|| "train".into());

    let mut songs = Vec::new();
    for path in find_simfiles(&dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let title = sim
            .tag("TITLE")
            .filter(|t| !t.is_empty())
            .unwrap_or(&name)
            .to_string();
        let Some(music) = sim.tag("MUSIC").filter(|m| !m.is_empty()) else {
            continue;
        };
        let music = path.parent().unwrap().join(music);
        if split_of(&title) == split && music.exists() {
            songs.push((sim, music));
        }
    }
    let step = (songs.len() as f64 / max as f64).max(1.0);
    let picked: Vec<usize> = (0..max.min(songs.len()))
        .map(|i| (i as f64 * step) as usize)
        .collect();
    eprintln!("{} songs in split {split}, using {}", songs.len(), picked.len());

    let per_diff = Mutex::new(vec![Stats::new(); 5]);
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
                    let (sim, music) = &songs[i];
                    let Ok(audio) = decode_file(music) else { continue };
                    let mix = Layer::analyze(&audio.samples, audio.sample_rate);
                    let cues = MusicCues::compute(&audio, &mix.envelope);
                    let mut local = vec![Stats::new(); 5];
                    for chart in &sim.charts {
                        if chart.steps_type != "dance-single" {
                            continue;
                        }
                        let Ok(d) = Difficulty::parse(&chart.difficulty) else {
                            continue;
                        };
                        let (Ok(timing), Ok(rows)) = (sim.timing(chart), chart.rows()) else {
                            continue;
                        };
                        let st = &mut local[d.index()];
                        let (mut prev_mask, mut prev_class, mut prev_pitch) = (None::<u8>, CLASSES, f32::NAN);
                        for r in &rows {
                            let mask = r
                                .cells
                                .iter()
                                .take(4)
                                .enumerate()
                                .filter(|(_, c)| {
                                    matches!(c, Cell::Tap | Cell::HoldHead | Cell::RollHead | Cell::Lift)
                                })
                                .fold(0u8, |m, (i, _)| m | 1 << i);
                            if mask == 0 {
                                continue;
                            }
                            let t = timing.seconds(r.beat);
                            let ctx: Context = cues.context(t, prev_pitch);
                            let class = class_of(mask, prev_mask);
                            st.joint[(prev_class * CONTEXTS + ctx.index()) * CLASSES + class] += 1;
                            let pitch = cues.pitch_at(t);
                            if let Some(pm) = prev_mask
                                && mask.count_ones() == 1
                                && pm.count_ones() == 1
                                && pm != mask
                                && !pitch.is_nan()
                                && !prev_pitch.is_nan()
                            {
                                let (a, b) = (xy(pm), xy(mask));
                                st.pairs.push(((pitch - prev_pitch) as f64, b.1 - a.1, b.0 - a.0));
                            }
                            (prev_mask, prev_class, prev_pitch) = (Some(mask), class, pitch);
                        }
                    }
                    let mut all = per_diff.lock().unwrap();
                    for (a, l) in all.iter_mut().zip(&local) {
                        a.add(l);
                    }
                    eprintln!("[{k}] {}", music.display());
                }
            });
        }
    });
    let per_diff = per_diff.into_inner().unwrap();

    let mut out = String::new();
    let _ = writeln!(out, "split {split}, songs {}", picked.len());
    let mut total = Stats::new();
    for s in &per_diff {
        total.add(s);
    }
    let labelled: Vec<(&str, &Stats)> = Difficulty::ALL
        .iter()
        .map(|d| d.name())
        .zip(per_diff.iter())
        .chain(std::iter::once(("All", &total)))
        .collect();
    for (name, st) in labelled {
        let j: Vec<f64> = st.joint.iter().map(|v| *v as f64).collect();
        let n: f64 = j.iter().sum();
        if n == 0.0 {
            continue;
        }
        // [prev][ctx][class] → I(ctx; class | prev); marginal over prev → I(ctx; class)
        let cmi = cond_mi(&j, PREV, CONTEXTS, CLASSES);
        let mut marg = vec![0.0; CONTEXTS * CLASSES];
        let mut prev_class = vec![0.0; PREV * CLASSES];
        for p in 0..PREV {
            for c in 0..CONTEXTS {
                for y in 0..CLASSES {
                    let v = j[(p * CONTEXTS + c) * CLASSES + y];
                    marg[c * CLASSES + y] += v;
                    prev_class[p * CLASSES + y] += v;
                }
            }
        }
        let mi = cond_mi(&marg, 1, CONTEXTS, CLASSES);
        let h = cond_entropy(&prev_class, PREV, CLASSES);
        // Per cue: collapse the context to one cue.
        let cue_mi = |f: &dyn Fn(usize) -> usize, k: usize| {
            let mut t = vec![0.0; PREV * k * CLASSES];
            for p in 0..PREV {
                for c in 0..CONTEXTS {
                    for y in 0..CLASSES {
                        t[(p * k + f(c)) * CLASSES + y] += j[(p * CONTEXTS + c) * CLASSES + y];
                    }
                }
            }
            cond_mi(&t, PREV, k, CLASSES)
        };
        let (hits, accents) = (itg_charter::music::HIT_TYPES, itg_charter::music::ACCENTS);
        let pitch_mi = cue_mi(&|c| c / (hits * accents), itg_charter::music::PITCH_MOVES);
        let hit_mi = cue_mi(&|c| (c / accents) % hits, hits);
        let accent_mi = cue_mi(&|c| c % accents, accents);
        // Small-sample bias of the plug-in estimate, for scale.
        let bias = ((CONTEXTS - 1) * (CLASSES - 1) * PREV) as f64 / (2.0 * n * std::f64::consts::LN_2);
        let pv: Vec<(f64, f64)> = st.pairs.iter().map(|p| (p.0, p.1)).collect();
        let ph: Vec<(f64, f64)> = st.pairs.iter().map(|p| (p.0, p.2)).collect();
        let _ = writeln!(
            out,
            "\n== {name}: {n} notes | H(arrow|prev) {h:.3} bits | I(ctx;arrow) {mi:.4} | I(ctx;arrow|prev) {cmi:.4} (bias ~{bias:.4}) | per cue given prev: pitch {pitch_mi:.4}, hit {hit_mi:.4}, accent {accent_mi:.4} | corr(dpitch, dy) {:+.3}, corr(dpitch, dx) {:+.3} over {} single moves",
            pearson_fmt(correlation(&pv)),
            pearson_fmt(correlation(&ph)),
            st.pairs.len()
        );
        let table = |label: &str, rows: usize, f: &dyn Fn(usize) -> usize, names: &[&str]| -> String {
            let mut t = vec![[0.0f64; CLASSES]; rows];
            for p in 0..PREV {
                for c in 0..CONTEXTS {
                    for y in 0..CLASSES {
                        t[f(c)][y] += j[(p * CONTEXTS + c) * CLASSES + y];
                    }
                }
            }
            let mut s = format!(
                "  P(arrow | {label})   {}\n",
                CLASS_NAMES.map(|c| format!("{c:>6}")).join("")
            );
            for (r, row) in t.iter().enumerate() {
                let n: f64 = row.iter().sum();
                let _ = writeln!(
                    s,
                    "  {:<18} {}  (n={n})",
                    names[r],
                    row.iter()
                        .map(|v| format!("{:>6.3}", v / n.max(1.0)))
                        .collect::<String>()
                );
            }
            s
        };
        out.push_str(&table(
            "pitch move",
            4,
            &|c| c / (hits * accents),
            &["down", "same", "up", "unknown"],
        ));
        out.push_str(&table(
            "hit type",
            hits,
            &|c| (c / accents) % hits,
            &["kick", "low-mid", "mid", "high"],
        ));
        out.push_str(&table("accent", accents, &|c| c % accents, &["weak", "strong"]));
    }
    print!("{out}");
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("eval");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(format!("music-signal-{split}.txt")), &out)?;
    Ok(())
}

fn pearson_fmt(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x }
}
