//! Trains one arrow model per charting style (`--style` of `itg-charter gen`).
//!
//! `training/styles.tsv` gives the style of each pack (groups of `pack_styles`, see
//! `training/STYLES.md`); a pack can be moved by editing its line. For each embedded
//! style, the simfiles of its packs under SONGS_DIR are copied to
//! `target/styles/<style>/<pack>/…` (same relative paths, hence the same training
//! order as a folder of links to the packs) and trained on with `Model::train`. The
//! classic model is written to `model/model.json` (the default embedded model), the
//! others to `model/styles/<style>.json`.
//!
//! Usage: cargo run --release --example train_styles -- SONGS_DIR

use itg_charter::model::{Model, find_simfiles};
use itg_charter::style::Style;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let songs = PathBuf::from(std::env::args().nth(1).expect("usage: train_styles SONGS_DIR"));
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let list = std::fs::read_to_string(root.join("training/styles.tsv"))?;
    let mut packs: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for line in list.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let (pack, style) = line.split_once('\t').expect("pack<TAB>style");
        packs.entry(style.trim()).or_default().push(pack);
    }
    for style in Style::ALL {
        let members = packs.get(style.name()).map_or(&[][..], |v| v.as_slice());
        anyhow::ensure!(!members.is_empty(), "no pack for style {}", style.name());
        let dir = root.join("target/styles").join(style.name());
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        let mut files = 0;
        for pack in members {
            let src = songs.join(pack);
            anyhow::ensure!(src.exists(), "pack not found: {}", src.display());
            for path in find_simfiles(&src) {
                let dst = dir.join(pack).join(path.strip_prefix(&src)?);
                std::fs::create_dir_all(dst.parent().unwrap())?;
                std::fs::copy(&path, &dst)?;
                files += 1;
            }
        }
        let model = Model::train(&dir, |_, _| {})?;
        let charts: u32 = model.stats.iter().map(|s| s.charts).sum();
        let out = root.join(style.model_path());
        std::fs::create_dir_all(out.parent().unwrap())?;
        std::fs::write(&out, model.to_json())?;
        println!(
            "{:<8} {:>3} packs, {files:>5} simfiles, {charts:>5} charts -> {}",
            style.name(),
            members.len(),
            style.model_path()
        );
    }
    Ok(())
}
