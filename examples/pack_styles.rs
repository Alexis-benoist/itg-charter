//! Charting styles of the packs of a songs folder.
//!
//! Every dance-single chart is described by a few style features (density, 16th
//! streams, triplets, jumps, holds, rolls, mines, crossovers, footswitches, jacks,
//! brackets, BPM changes and stops, main BPM). Each feature is turned into a z-score
//! against the charts of the same meter, so that a pack of hard charts does not look
//! "dense" only because it is hard. A pack's profile is the mean z-score of its charts
//! (packs with at least 20 charts); the profiles are grouped by k-means (deterministic:
//! farthest-point initialisation, packs in name order). For each group: its packs, its
//! profile, and the median raw features of its meter 8–10 charts.
//!
//! Summary written to `eval/pack-styles.txt`.
//!
//! Usage: cargo run --release --example pack_styles -- SONGS_DIR [K]

use itg_charter::model::{ChartFeatures, find_simfiles};
use itg_charter::parity::Layout;
use itg_charter::simfile::{Cell, Simfile};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

const NAMES: [&str; 13] = [
    "nps", "stream16", "triplets", "jumps", "holds", "rolls", "mines", "xo", "fs", "jacks", "brackets",
    "gimmicks", "bpm",
];
const NF: usize = NAMES.len();
const MIN_CHARTS: usize = 20;
const MAX_METER: i32 = 16;

struct ChartStyle {
    pack: String,
    meter: i32,
    x: [f64; NF],
}

fn style(sim: &Simfile, chart: &itg_charter::simfile::ChartData, layout: &Layout) -> Option<[f64; NF]> {
    let timing = sim.timing(chart).ok()?;
    let rows = chart.rows().ok()?;
    let f = ChartFeatures::compute(&rows, &timing, layout);
    if f.rows < 16 {
        return None;
    }
    let n = f.rows as f64;
    let per100 = |c: u32| 100.0 * c as f64 / n;
    let count = |cell: Cell| {
        rows.iter()
            .flat_map(|r| r.cells.iter().take(4))
            .filter(|&&c| c == cell)
            .count() as u32
    };
    let gaps: Vec<u32> = f.sequence.windows(2).map(|w| w[1].1 - w[0].1).collect();
    let stream16 = gaps.iter().filter(|&&g| g == 12).count() as f64 / gaps.len().max(1) as f64;
    let tech =
        |g: &dyn Fn(&itg_charter::parity::TechCounts) -> u32| f.tech.map_or(f64::NAN, |t| per100(g(&t)));
    // Main BPM: the one covering the most beats up to the last note.
    let last = rows.last().map_or(0.0, |r| r.beat);
    let bpm = (0..timing.bpms.len())
        .map(|i| {
            let end = timing.bpms.get(i + 1).map_or(last, |b| b.0).min(last);
            (end - timing.bpms[i].0.max(0.0), timing.bpms[i].1)
        })
        .fold((f64::MIN, 0.0), |best, s| if s.0 > best.0 { s } else { best })
        .1;
    let changes = timing.bpms.len().saturating_sub(1) + timing.stops.len() + timing.delays.len();
    Some([
        f.nps,
        stream16,
        (f.snaps[2] + f.snaps[4]) as f64 / n,
        f.jump_ratio,
        f.hold_ratio,
        per100(count(Cell::RollHead)),
        per100(count(Cell::Mine)),
        tech(&|t| t.crossovers),
        tech(&|t| t.footswitches),
        tech(&|t| t.jacks),
        tech(&|t| t.brackets),
        changes as f64,
        bpm,
    ])
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn dist(a: &[f64; NF], b: &[f64; NF]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(args.first().expect("usage: pack_styles SONGS_DIR [K]"));
    let k: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(5);
    let layout = Layout::dance_single();

    let mut charts = Vec::new();
    for path in find_simfiles(&dir) {
        let Ok(sim) = Simfile::load(&path) else { continue };
        if sim.is_generated() {
            continue;
        }
        let Some(pack) = path.strip_prefix(&dir).ok().and_then(|p| p.iter().next()) else {
            continue;
        };
        let pack = pack.to_string_lossy().into_owned();
        for chart in sim.charts.iter().filter(|c| c.steps_type == "dance-single") {
            if chart.meter < 1 {
                continue;
            }
            if let Some(x) = style(&sim, chart, &layout) {
                charts.push(ChartStyle {
                    pack: pack.clone(),
                    meter: chart.meter.min(MAX_METER),
                    x,
                });
            }
        }
    }

    // Mean and standard deviation of each feature per meter.
    let mut by_meter: BTreeMap<i32, Vec<&ChartStyle>> = BTreeMap::new();
    for c in &charts {
        by_meter.entry(c.meter).or_default().push(c);
    }
    let norm: BTreeMap<i32, [(f64, f64); NF]> = by_meter
        .iter()
        .map(|(&m, cs)| {
            let mut out = [(0.0, 1.0); NF];
            for (f, o) in out.iter_mut().enumerate() {
                let v: Vec<f64> = cs.iter().map(|c| c.x[f]).filter(|x| x.is_finite()).collect();
                let n = v.len().max(1) as f64;
                let mean = v.iter().sum::<f64>() / n;
                let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
                *o = (mean, if sd > 1e-9 { sd } else { 1.0 });
            }
            (m, out)
        })
        .collect();

    // Pack profiles: mean z-score per feature.
    let mut packs: BTreeMap<&str, Vec<&ChartStyle>> = BTreeMap::new();
    for c in &charts {
        packs.entry(&c.pack).or_default().push(c);
    }
    packs.retain(|_, cs| cs.len() >= MIN_CHARTS);
    let names: Vec<&str> = packs.keys().copied().collect();
    let profiles: Vec<[f64; NF]> = packs
        .values()
        .map(|cs| {
            let mut p = [0.0; NF];
            for (f, v) in p.iter_mut().enumerate() {
                let z: Vec<f64> = cs
                    .iter()
                    .map(|c| {
                        let (mean, sd) = norm[&c.meter][f];
                        (c.x[f] - mean) / sd
                    })
                    .filter(|z| z.is_finite())
                    .collect();
                *v = z.iter().sum::<f64>() / z.len().max(1) as f64;
            }
            p
        })
        .collect();

    // k-means, farthest-point initialisation from the most atypical pack.
    let k = k.min(profiles.len());
    let origin = [0.0; NF];
    let first = (0..profiles.len())
        .max_by(|&a, &b| {
            dist(&profiles[a], &origin)
                .total_cmp(&dist(&profiles[b], &origin))
                .then(b.cmp(&a))
        })
        .unwrap();
    let mut centers = vec![profiles[first]];
    while centers.len() < k {
        let next = (0..profiles.len())
            .max_by(|&a, &b| {
                let da = centers
                    .iter()
                    .map(|c| dist(&profiles[a], c))
                    .fold(f64::MAX, f64::min);
                let db = centers
                    .iter()
                    .map(|c| dist(&profiles[b], c))
                    .fold(f64::MAX, f64::min);
                da.total_cmp(&db).then(b.cmp(&a))
            })
            .unwrap();
        centers.push(profiles[next]);
    }
    let mut assign = vec![0; profiles.len()];
    for _ in 0..100 {
        let new: Vec<usize> = profiles
            .iter()
            .map(|p| {
                (0..k)
                    .min_by(|&a, &b| {
                        dist(p, &centers[a])
                            .total_cmp(&dist(p, &centers[b]))
                            .then(a.cmp(&b))
                    })
                    .unwrap()
            })
            .collect();
        for (g, c) in centers.iter_mut().enumerate() {
            let members: Vec<&[f64; NF]> = profiles
                .iter()
                .zip(&new)
                .filter(|&(_, &a)| a == g)
                .map(|(p, _)| p)
                .collect();
            if !members.is_empty() {
                for f in 0..NF {
                    c[f] = members.iter().map(|p| p[f]).sum::<f64>() / members.len() as f64;
                }
            }
        }
        if new == assign {
            break;
        }
        assign = new;
    }

    let mut report = String::new();
    writeln!(
        report,
        "{}: {} charts, {} packs with at least {MIN_CHARTS} charts, {k} groups; profile = mean z-score vs charts of the same meter",
        dir.display(),
        charts.len(),
        names.len()
    )?;
    for (g, center) in centers.iter().enumerate() {
        let members: Vec<usize> = (0..names.len()).filter(|&i| assign[i] == g).collect();
        let group_charts: Vec<&ChartStyle> = members
            .iter()
            .flat_map(|&i| packs[names[i]].iter().copied())
            .collect();
        writeln!(
            report,
            "\n== group {} ({} packs, {} charts)",
            g + 1,
            members.len(),
            group_charts.len()
        )?;
        let mut profile: Vec<(usize, f64)> = center.iter().copied().enumerate().collect();
        profile.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then(a.0.cmp(&b.0)));
        let prof: Vec<String> = profile
            .iter()
            .map(|(f, z)| format!("{} {z:+.2}", NAMES[*f]))
            .collect();
        writeln!(report, "profile: {}", prof.join(", "))?;
        let mid: Vec<&&ChartStyle> = group_charts
            .iter()
            .filter(|c| (8..=10).contains(&c.meter))
            .collect();
        let raw: Vec<String> = (0..NF)
            .map(|f| format!("{} {:.3}", NAMES[f], median(mid.iter().map(|c| c.x[f]).collect())))
            .collect();
        writeln!(
            report,
            "meter 8-10 medians ({} charts): {}",
            mid.len(),
            raw.join(", ")
        )?;
        for &i in &members {
            let mut top: Vec<(usize, f64)> = profiles[i].iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then(a.0.cmp(&b.0)));
            let top: Vec<String> = top
                .iter()
                .take(4)
                .map(|(f, z)| format!("{} {z:+.1}", NAMES[*f]))
                .collect();
            writeln!(
                report,
                "  {:<55} {:>4} charts  {}",
                names[i],
                packs[names[i]].len(),
                top.join(", ")
            )?;
        }
    }
    let all_mid: Vec<&ChartStyle> = charts.iter().filter(|c| (8..=10).contains(&c.meter)).collect();
    let raw: Vec<String> = (0..NF)
        .map(|f| {
            format!(
                "{} {:.3}",
                NAMES[f],
                median(all_mid.iter().map(|c| c.x[f]).collect())
            )
        })
        .collect();
    writeln!(
        report,
        "\nall packs, meter 8-10 medians ({} charts): {}",
        all_mid.len(),
        raw.join(", ")
    )?;
    print!("{report}");
    std::fs::write(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("eval/pack-styles.txt"),
        &report,
    )?;
    Ok(())
}
