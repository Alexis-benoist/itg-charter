//! Do two human charters put notes at the same places? (ceiling for `eval_placement`)
//!
//! Songs present in several simfiles (same title + artist, alphanumerics only, any
//! case) are paired; for every difficulty both simfiles have in dance-single, the note
//! rows are compared in seconds. The two syncs can differ (offset, a different audio
//! edit, a slightly different BPM), so B is first aligned on A using all the rows of
//! both simfiles: the most common time difference (5 ms bins, within ±3 s), then a
//! least-squares line a = shift + rate·b through the rows it matches. Each pair of
//! charts is then matched one-to-one within ±25 ms, over the time span both cover:
//! F = 2·matched / (rows A + rows B). The key includes the subtitle (remixes).
//!
//! Pairs with F ≥ 99 % and the same number of rows are the same chart copied between
//! packs: counted apart, left out of the agreement figures.
//!
//! Summary written to `eval/human-agreement.txt`.
//!
//! Usage: cargo run --release --example eval_human_agreement -- SONGS_DIR

use itg_charter::difficulty::Difficulty;
use itg_charter::model::find_simfiles;
use itg_charter::simfile::{Cell, Simfile};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

const TOLERANCE: f64 = 0.025;
const MAX_SHIFT: f64 = 3.0;
const BIN: f64 = 0.005;

struct Chart {
    meter: i32,
    /// Seconds of every row with a note.
    times: Vec<f64>,
    /// Seconds of the rows on a measure start.
    downbeats: Vec<f64>,
}

/// One simfile of a song.
struct Version {
    path: PathBuf,
    /// Size and hash of the first MiB of the audio file, if found.
    audio: Option<(u64, u64)>,
    /// Charts per difficulty index.
    charts: BTreeMap<usize, Chart>,
}

fn audio_id(path: &std::path::Path) -> Option<(u64, u64)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let size = f.metadata().ok()?.len();
    let mut buf = vec![0u8; 1 << 20];
    let n = f.read(&mut buf).ok()?;
    let hash = buf[..n].iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    });
    Some((size, hash))
}

fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// One-to-one greedy matching of sorted `a` and mapped `b` within the tolerance.
fn matches(a: &[f64], b: &[f64], map: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let d = a[i] - map(b[j]);
        if d.abs() <= TOLERANCE {
            out.push((a[i], b[j]));
            i += 1;
            j += 1;
        } else if d < 0.0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Most common difference a - b in 5 ms bins within ±`max` around `center`; ties go
/// to the one closest to `center`, then the lowest.
fn mode_shift(a: &[f64], b: &[f64], center: f64, max: f64) -> f64 {
    let half = (max / BIN).round() as i64;
    let mut bins = vec![0u32; (2 * half + 1) as usize];
    for &x in a {
        for &y in b {
            let k = ((x - y - center) / BIN).round() as i64;
            if k.abs() <= half {
                bins[(k + half) as usize] += 1;
            }
        }
    }
    let bin = (-half..=half)
        .max_by(|&x, &y| {
            let (cx, cy) = (bins[(x + half) as usize], bins[(y + half) as usize]);
            cx.cmp(&cy).then(y.abs().cmp(&x.abs())).then(y.cmp(&x))
        })
        .unwrap();
    center + bin as f64 * BIN
}

/// Maps B's times onto A's, from all the rows of both simfiles (so that sparse charts
/// are aligned too). Same audio file: only the offsets differ, the shift is searched
/// within ±100 ms. Otherwise, a coarse shift from the rows on measure starts (within
/// ±3 s; all rows would lock onto an 8th-note error in dense streams), refined within
/// ±30 ms on all rows. Then a least-squares line a = shift + rate·b through the
/// matched rows (BPM drift).
fn align(a: &Version, b: &Version) -> (f64, f64) {
    let rows = |v: &Version, down: bool| {
        let mut t: Vec<f64> = v
            .charts
            .values()
            .flat_map(|c| if down { &c.downbeats } else { &c.times }.iter().copied())
            .collect();
        t.sort_by(|x, y| x.total_cmp(y));
        t.dedup_by(|x, y| (*x - *y).abs() < 1e-4);
        t
    };
    let (ta, tb) = (rows(a, false), rows(b, false));
    let shift = if a.audio.is_some() && a.audio == b.audio {
        mode_shift(&ta, &tb, 0.0, 0.1)
    } else {
        let coarse = mode_shift(&rows(a, true), &rows(b, true), 0.0, MAX_SHIFT);
        mode_shift(&ta, &tb, coarse, 0.03)
    };
    let m = matches(&ta, &tb, |t| t + shift);
    if m.len() < 8 {
        return (shift, 1.0);
    }
    let n = m.len() as f64;
    let (ma, mb) = m
        .iter()
        .fold((0.0, 0.0), |(sa, sb), &(x, y)| (sa + x / n, sb + y / n));
    let (cov, var) = m.iter().fold((0.0, 0.0), |(c, v), &(x, y)| {
        (c + (y - mb) * (x - ma), v + (y - mb).powi(2))
    });
    let r = if var > 0.0 { cov / var } else { 1.0 };
    let (s, r) = (ma - r * mb, r);
    if matches(&ta, &tb, |t| s + r * t).len() >= m.len() {
        (s, r)
    } else {
        (shift, 1.0)
    }
}

/// F between two charts once aligned, over the time span both cover (different audio
/// edits: a short cut against the full song).
fn agreement(a: &[f64], b: &[f64], (s, r): (f64, f64)) -> f64 {
    let b: Vec<f64> = b.iter().map(|&t| s + r * t).collect();
    let lo = a[0].max(b[0]) - TOLERANCE;
    let hi = a[a.len() - 1].min(b[b.len() - 1]) + TOLERANCE;
    let a: Vec<f64> = a.iter().copied().filter(|t| (lo..=hi).contains(t)).collect();
    let b: Vec<f64> = b.into_iter().filter(|t| (lo..=hi).contains(t)).collect();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    2.0 * matches(&a, &b, |t| t).len() as f64 / (a.len() + b.len()) as f64
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: eval_human_agreement SONGS_DIR"),
    );
    // Song key -> simfiles, each with its charts per difficulty.
    let mut songs: BTreeMap<String, Vec<Version>> = BTreeMap::new();
    for path in find_simfiles(&dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
        if sim.is_generated() {
            continue;
        }
        let key = format!(
            "{} / {} / {}",
            norm(sim.tag("TITLE").unwrap_or("")),
            norm(sim.tag("SUBTITLE").unwrap_or("")),
            norm(sim.tag("ARTIST").unwrap_or(""))
        );
        if key.starts_with(" / ") {
            continue;
        }
        let mut charts = BTreeMap::new();
        for chart in sim.charts.iter().filter(|c| c.steps_type == "dance-single") {
            let Ok(diff) = Difficulty::parse(&chart.difficulty) else {
                continue;
            };
            let (Ok(timing), Ok(rows)) = (sim.timing(chart), chart.rows()) else {
                continue;
            };
            let mut times = Vec::new();
            let mut downbeats = Vec::new();
            for r in rows.iter().filter(|r| {
                r.cells
                    .iter()
                    .take(4)
                    .any(|c| matches!(c, Cell::Tap | Cell::HoldHead | Cell::RollHead | Cell::Lift))
            }) {
                let t = timing.seconds(r.beat);
                times.push(t);
                if ((r.beat * 48.0).round() as i64).rem_euclid(192) == 0 {
                    downbeats.push(t);
                }
            }
            times.sort_by(|a, b| a.total_cmp(b));
            downbeats.sort_by(|a, b| a.total_cmp(b));
            if times.len() >= 16 {
                charts.entry(diff.index()).or_insert(Chart {
                    meter: chart.meter,
                    times,
                    downbeats,
                });
            }
        }
        if !charts.is_empty() {
            let audio = sim
                .tag("MUSIC")
                .filter(|m| !m.is_empty())
                .and_then(|m| audio_id(&path.parent()?.join(m)));
            songs
                .entry(key)
                .or_default()
                .push(Version { path, audio, charts });
        }
    }

    // Per group (all distinct pairs, same audio, same audio and meters within 1) and
    // difficulty: F of each pair of charts; copies apart.
    const GROUPS: [&str; 3] = ["all pairs", "same audio", "same audio, meters within 1"];
    let mut f: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); 6]; GROUPS.len()];
    let mut copies = [0usize; 6];
    let mut songs_paired = 0;
    let debug = std::env::var_os("DEBUG_PAIRS").is_some();
    for versions in songs.values().filter(|v| v.len() >= 2) {
        songs_paired += 1;
        for i in 0..versions.len() {
            for j in i + 1..versions.len() {
                let (va, vb) = (&versions[i], &versions[j]);
                let map = align(va, vb);
                let same_audio = va.audio.is_some() && va.audio == vb.audio;
                for (&d, a) in &va.charts {
                    let Some(b) = vb.charts.get(&d) else { continue };
                    let fab = agreement(&a.times, &b.times, map);
                    if debug && fab < 0.3 {
                        eprintln!(
                            "{fab:.2} d{d} rows {}/{} meter {}/{} same audio {same_audio}\n  {}\n  {}",
                            a.times.len(),
                            b.times.len(),
                            a.meter,
                            b.meter,
                            va.path.display(),
                            vb.path.display()
                        );
                    }
                    if fab >= 0.99 && a.times.len() == b.times.len() {
                        copies[d] += 1;
                        copies[5] += 1;
                        continue;
                    }
                    let groups = [true, same_audio, same_audio && (a.meter - b.meter).abs() <= 1];
                    for g in (0..groups.len()).filter(|&g| groups[g]) {
                        f[g][d].push(fab);
                        f[g][5].push(fab);
                    }
                }
            }
        }
    }

    let mut report = String::new();
    writeln!(
        report,
        "{}: {songs_paired} songs in several simfiles (title + subtitle + artist); \
         tolerance ±{:.0} ms after alignment; copies (F ≥ 99 %, same rows) left out",
        dir.display(),
        TOLERANCE * 1000.0
    )?;
    let names: Vec<&str> = Difficulty::ALL.iter().map(|d| d.name()).chain(["All"]).collect();
    writeln!(
        report,
        "copies: {}",
        (0..6)
            .map(|d| format!("{} {}", names[d], copies[d]))
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    for (g, group) in GROUPS.iter().enumerate() {
        writeln!(report, "\n{group}: pairs, F p25 / median / p75, mean")?;
        for (d, name) in names.iter().enumerate() {
            let v = &mut f[g][d];
            let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
            writeln!(
                report,
                "  {name:<10} {:>4}  {:5.1}% / {:5.1}% / {:5.1}%  {:5.1}%",
                v.len(),
                100.0 * quantile(v, 0.25),
                100.0 * quantile(v, 0.5),
                100.0 * quantile(v, 0.75),
                100.0 * mean
            )?;
        }
    }
    print!("{report}");
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("eval/human-agreement.txt");
    std::fs::write(out, &report)?;
    Ok(())
}
