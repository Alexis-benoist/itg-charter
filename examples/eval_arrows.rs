//! Do repeated measures repeat their arrows? Human charts vs ours.
//!
//! For every measure that sounds like an earlier one and has the same rhythm (the
//! pairing of `find_repeats`, used by the generator's `repeat_bonus`), the arrows of the
//! later measure are compared with the earlier ones: identical, left-right mirror,
//! up-down flip, both (180° rotation), or something else. Human charts on the human
//! grid; ours generated on the same grid with the default options.
//!
//! Summary written to `eval/arrows-<TAG>.txt`.
//!
//! Usage: cargo run --release --example eval_arrows -- SONGS_DIR
//!        [--max N] [--split train|test|all] [--tag TAG]

use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::audio::decode_file;
use itg_charter::chart::{GenOptions, Note, find_repeats, generate};
use itg_charter::difficulty::Difficulty;
use itg_charter::model::{Model, find_simfiles};
use itg_charter::music::split_of;
use itg_charter::simfile::{Cell, Simfile};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

const SEED: u64 = 1;
const KINDS: [&str; 5] = ["same", "mirror L-R", "flip U-D", "rotate", "other"];

fn swap(m: u8, a: u8, b: u8) -> u8 {
    let (x, y) = ((m >> a) & 1, (m >> b) & 1);
    (m & !(1 << a) & !(1 << b)) | (x << b) | (y << a)
}

/// The arrow transforms, in the order of `KINDS` (columns: left, down, up, right).
fn transform(k: usize, m: u8) -> u8 {
    match k {
        0 => m,
        1 => swap(m, 0, 3),
        2 => swap(m, 1, 2),
        _ => swap(swap(m, 0, 3), 1, 2),
    }
}

/// Notes and arrow masks of a chart (taps, hold / roll heads, lifts).
fn notes_of(rows: impl Iterator<Item = (u32, [Cell; 4])>) -> (Vec<Note>, Vec<u8>) {
    let mut notes = Vec::new();
    let mut masks = Vec::new();
    for (pos, cells) in rows {
        let mut mask = 0u8;
        let mut hold = false;
        for (c, cell) in cells.iter().enumerate() {
            match cell {
                Cell::Tap | Cell::Lift => mask |= 1 << c,
                Cell::HoldHead | Cell::RollHead => {
                    mask |= 1 << c;
                    hold = true;
                }
                _ => {}
            }
        }
        if mask != 0 {
            notes.push(Note {
                pos,
                strength: 0.0,
                jump: mask.count_ones() >= 2,
                hold: hold.then_some(48),
            });
            masks.push(mask);
        }
    }
    (notes, masks)
}

/// Counts per kind of the repeated measures of one chart.
fn classify(a: &SongAnalysis, notes: &[Note], masks: &[u8], counts: &mut [u64; 5]) {
    let repeats = find_repeats(a, notes, 0.9);
    // later measure -> (note, matching earlier note) pairs
    let mut pairs: BTreeMap<u32, Vec<(usize, usize)>> = BTreeMap::new();
    for (i, r) in repeats.iter().enumerate() {
        if let Some(j) = r {
            pairs.entry(notes[i].pos / 192).or_default().push((i, *j));
        }
    }
    for p in pairs.values() {
        let kind = (0..4)
            .find(|&k| p.iter().all(|&(i, j)| masks[i] == transform(k, masks[j])))
            .unwrap_or(4);
        counts[kind] += 1;
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
    let usage = "usage: eval_arrows SONGS_DIR [--max N] [--split S] [--tag T] [--repeat-bonus B,B,…]";
    let dir = PathBuf::from(args.first().expect(usage));
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let max: usize = opt("--max").and_then(|m| m.parse().ok()).unwrap_or(150);
    let split = opt("--split").unwrap_or_else(|| "test".into());
    let tag = opt("--tag").unwrap_or_else(|| split.clone());
    let model = Model::embedded()?;
    // One generation per repeat_bonus value (default: the generator's).
    let gen_opts: Vec<GenOptions> = match opt("--repeat-bonus") {
        Some(list) => list
            .split(',')
            .map(|b| {
                Ok(GenOptions {
                    repeat_bonus: [b.parse()?; 5],
                    ..GenOptions::default()
                })
            })
            .collect::<anyhow::Result<_>>()?,
        None => vec![GenOptions::default()],
    };

    let mut songs = Vec::new();
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

    // [option set][difficulty] -> (human, ours) counts per kind
    type Counts = [([u64; 5], [u64; 5]); 5];
    let counts: Mutex<Vec<Counts>> = Mutex::new(vec![[([0u64; 5], [0u64; 5]); 5]; gen_opts.len()]);
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
                    let mut local: Vec<Counts> = vec![[([0u64; 5], [0u64; 5]); 5]; gen_opts.len()];
                    let mut done = [false; 5];
                    for chart in &song.sim.charts {
                        if chart.steps_type != "dance-single" {
                            continue;
                        }
                        let Ok(d) = Difficulty::parse(&chart.difficulty) else {
                            continue;
                        };
                        let Ok(rows) = chart.rows() else { continue };
                        let (notes, masks) = notes_of(rows.iter().filter(|r| r.cells.len() >= 4).map(|r| {
                            (
                                (r.beat * 48.0).round().max(0.0) as u32,
                                [r.cells[0], r.cells[1], r.cells[2], r.cells[3]],
                            )
                        }));
                        if notes.len() < 16 {
                            continue;
                        }
                        let mut human = [0u64; 5];
                        classify(&a, &notes, &masks, &mut human);
                        for l in local.iter_mut() {
                            for (t, h) in l[d.index()].0.iter_mut().zip(human) {
                                *t += h;
                            }
                        }
                        if !done[d.index()] {
                            done[d.index()] = true;
                            for (l, o) in local.iter_mut().zip(&gen_opts) {
                                let ours = generate(&a, &model, d, SEED, o);
                                let (notes, masks) = notes_of(ours.rows.iter().copied());
                                classify(&a, &notes, &masks, &mut l[d.index()].1);
                            }
                        }
                    }
                    let mut all = counts.lock().unwrap();
                    for (ta, la) in all.iter_mut().zip(local) {
                        for (t, l) in ta.iter_mut().zip(la) {
                            for k in 0..5 {
                                t.0[k] += l.0[k];
                                t.1[k] += l.1[k];
                            }
                        }
                    }
                    eprintln!("[{k}] {}", song.title);
                }
            });
        }
    });

    let counts = counts.into_inner().unwrap();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "split {split}, songs {}, seed {SEED}, human grid forced; repeated measures (same rhythm, audio cosine >= 0.9)",
        picked.len()
    );
    let header: Vec<String> = KINDS.iter().map(|k| format!("{k:>21}")).collect();
    for (c, o) in counts.iter().zip(&gen_opts) {
        let _ = writeln!(out, "\nrepeat_bonus {:?}", o.repeat_bonus);
        let _ = writeln!(out, "{:<10} | {:>13} | {}", "", "pairs h / o", header.join(" |"));
        let mut total = ([0u64; 5], [0u64; 5]);
        for d in Difficulty::ALL {
            let (h, o) = c[d.index()];
            for k in 0..5 {
                total.0[k] += h[k];
                total.1[k] += o[k];
            }
            let _ = writeln!(out, "{}", line(d.name(), &h, &o));
        }
        let _ = writeln!(out, "{}", line("All", &total.0, &total.1));
    }
    let _ = writeln!(out, "(each cell: human % / ours %)");
    print!("{out}");
    std::fs::create_dir_all("eval")?;
    std::fs::write(format!("eval/arrows-{tag}.txt"), &out)?;
    Ok(())
}

fn line(name: &str, h: &[u64; 5], o: &[u64; 5]) -> String {
    let (nh, no) = (h.iter().sum::<u64>(), o.iter().sum::<u64>());
    let cells: Vec<String> = (0..5)
        .map(|k| {
            format!(
                "{:>9.1}% / {:>5.1}%",
                100.0 * h[k] as f64 / nh.max(1) as f64,
                100.0 * o[k] as f64 / no.max(1) as f64
            )
        })
        .collect();
    format!("{name:<10} | {nh:>6} / {no:<5} | {}", cells.join(" |"))
}
