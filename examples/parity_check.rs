//! Compares our parity port with the tech counts computed by ITGmania itself.
//!
//! ITGmania stores `#TECHCOUNTS` for every chart in its song cache
//! (`~/.itgmania/Cache/Songs/*`). This recomputes them with `itg_charter::parity`
//! and reports how many charts match exactly.
//!
//! The cache has no note data for `.sm` songs, so notes are read from the original
//! file (`#STEPFILENAME`, relative to the game directory).
//!
//! Usage: cargo run --release --example parity_check -- [CACHE_DIR [GAME_DIR]]
//! (CACHE_DIR defaults to ~/.itgmania/Cache/Songs, GAME_DIR to $ITGMANIA_DIR).

use itg_charter::parity::{Layout, TechCounts, analyze, rows_from_chart};
use itg_charter::simfile::Simfile;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs_home().join(".itgmania/Cache/Songs"));
    let game_dir = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("ITGMANIA_DIR").map(PathBuf::from))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "game folder unknown: pass it as second argument or set ITGMANIA_DIR \
                 (the folder that contains ITGmania's Songs/)"
            )
        })?;
    let layout = Layout::dance_single();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    let (mut total, mut exact, mut skipped) = (0, 0, 0);
    let mut per_category_errors = [0u32; 10];
    let mut shown = 0;
    for path in entries {
        let text = String::from_utf8_lossy(&std::fs::read(&path)?).into_owned();
        let Ok(sim) = Simfile::parse(&text, true) else {
            continue;
        };
        for chart in &sim.charts {
            if chart.steps_type != "dance-single" {
                continue;
            }
            let Some(expected) = chart.tags.iter().find(|(k, _)| k == "TECHCOUNTS") else {
                continue;
            };
            let expected: Vec<f32> = expected
                .1
                .split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect();
            if expected.len() < 10 {
                continue;
            }
            let original;
            let (sim, chart) = if chart.notes.is_empty() {
                let Some((_, file)) = chart.tags.iter().find(|(k, _)| k == "STEPFILENAME") else {
                    continue;
                };
                let Ok(o) = Simfile::load(&game_dir.join(file.trim_start_matches('/'))) else {
                    skipped += 1;
                    continue;
                };
                original = o;
                let Some(c) = original.charts.iter().find(|c| {
                    c.steps_type == chart.steps_type
                        && c.difficulty.eq_ignore_ascii_case(&chart.difficulty)
                        && c.description == chart.description
                }) else {
                    skipped += 1;
                    continue;
                };
                (&original, c)
            } else {
                (&sim, chart)
            };
            let (Ok(timing), Ok(note_rows)) = (sim.timing(chart), chart.rows()) else {
                skipped += 1;
                continue;
            };
            let mut rows = rows_from_chart(&note_rows, &timing, 4);
            if analyze(&layout, &mut rows).is_none() {
                skipped += 1;
                continue;
            }
            let tc = TechCounts::from_rows(&layout, &rows);
            let ours = [
                tc.crossovers,
                tc.half_crossovers,
                tc.full_crossovers,
                tc.footswitches,
                tc.up_footswitches,
                tc.down_footswitches,
                tc.sideswitches,
                tc.jacks,
                tc.brackets,
                tc.doublesteps,
            ];
            total += 1;
            let mut ok = true;
            for i in 0..10 {
                if ours[i] as f32 != expected[i] {
                    per_category_errors[i] += 1;
                    ok = false;
                }
            }
            if ok {
                exact += 1;
            } else if shown < 10 {
                shown += 1;
                println!(
                    "MISMATCH {} [{}]\n  game: {:?}\n  ours: {:?}",
                    path.file_name().unwrap().to_string_lossy(),
                    chart.difficulty,
                    &expected[..10],
                    ours
                );
            }
        }
    }
    println!(
        "charts: {total}, exact match: {exact} ({:.1}%), skipped (warps/unsupported): {skipped}",
        100.0 * exact as f64 / total.max(1) as f64
    );
    let names = [
        "crossovers",
        "half_xo",
        "full_xo",
        "footswitch",
        "up_fs",
        "down_fs",
        "sideswitch",
        "jacks",
        "brackets",
        "doublesteps",
    ];
    for (n, e) in names.iter().zip(per_category_errors) {
        println!("  {n:>12}: {e} charts differ");
    }
    Ok(())
}

fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}
