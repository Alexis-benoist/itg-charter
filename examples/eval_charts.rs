//! Compares generated charts with the human charts the model was trained on.
//!
//! Generates every difficulty for a sample of songs (no stems, for speed) and prints,
//! per difficulty, the median of each statistic next to the human p10–p50–p90.
//!
//! Usage: cargo run --release --example eval_charts -- SONGS_DIR [MAX_SONGS]
//!        [parity_weight temperature repeat_bonus]   (to tune `GenOptions`)

use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::audio::decode_file;
use itg_charter::chart::{GenOptions, generate};
use itg_charter::difficulty::Difficulty;
use itg_charter::model::{ChartFeatures, Model, Quantiles, find_simfiles};
use itg_charter::parity::Layout;
use itg_charter::simfile::{NoteRow, Simfile, Timing};
use std::path::PathBuf;
use std::sync::Mutex;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(args.first().expect("usage: eval_charts SONGS_DIR [MAX]"));
    let max: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(40);
    let model = Model::embedded()?;
    let mut gen_opts = GenOptions::default();
    if let [pw, t, rb] = &args.get(2..5).unwrap_or(&[]) {
        gen_opts.parity_weight = pw.parse()?;
        gen_opts.temperature = t.parse()?;
        gen_opts.repeat_bonus = [rb.parse()?; 5];
    }
    println!("{gen_opts:?}");
    let mut songs = Vec::new();
    for path in find_simfiles(&dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
        if sim.is_generated() {
            continue;
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

    // per difficulty: list of feature vectors
    let feats: Mutex<Vec<Vec<[f64; 8]>>> = Mutex::new(vec![Vec::new(); 5]);
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
                    let layout = Layout::dance_single();
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
                        let f = ChartFeatures::compute(
                            &rows,
                            &Timing::constant(a.grid.bpm, a.grid.offset()),
                            &layout,
                        );
                        let t = f.tech.unwrap_or_default();
                        let per100 = |x: u32| 100.0 * x as f64 / f.rows.max(1) as f64;
                        feats.lock().unwrap()[d.index()].push([
                            f.nps,
                            f.jump_ratio,
                            f.hold_ratio,
                            per100(t.crossovers),
                            per100(t.footswitches),
                            per100(t.jacks),
                            per100(t.doublesteps),
                            c.meter as f64,
                        ]);
                    }
                    eprintln!("[{i}] {}", path.display());
                }
            });
        }
    });
    let feats = feats.into_inner().unwrap();
    let mut flags = 0;
    let names = [
        "nps",
        "jump_ratio",
        "hold_ratio",
        "xo/100",
        "fs/100",
        "jacks/100",
        "dblstep/100",
        "meter",
    ];
    for d in Difficulty::ALL {
        let s = model.stats(d);
        let human = [
            &s.nps,
            &s.jump_ratio,
            &s.hold_ratio,
            &s.crossovers,
            &s.footswitches,
            &s.jacks,
            &s.doublesteps,
            &s.meter,
        ];
        println!("== {} ({} charts)", d.name(), feats[d.index()].len());
        for (k, name) in names.iter().enumerate() {
            let ours = Quantiles::of(&feats[d.index()].iter().map(|f| f[k]).collect::<Vec<_>>());
            let h = human[k];
            let outside = ours.p50 < h.p10 || ours.p50 > h.p90;
            flags += outside as u32;
            let flag = if outside {
                "  <-- outside human p10-p90"
            } else {
                ""
            };
            println!(
                "  {name:>12}: ours median {:>7.3} | human p10 {:>7.3} p50 {:>7.3} p90 {:>7.3}{flag}",
                ours.p50, h.p10, h.p50, h.p90
            );
        }
    }
    println!("outside human range: {flags}");
    Ok(())
}
