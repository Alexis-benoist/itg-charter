//! A song folder for the game: the simfile, its audio and its visuals.
//!
//! This is what `itg-charter gen` and `itg-charter decorate` do, as a library so
//! that other programs (e.g. itg-yt) can drive it.

use crate::analysis::{AnalysisOptions, SongAnalysis, Stems};
use crate::audio::decode_file;
use crate::chart::{GenOptions, assign_slots, generate, generate_for_meter};
use crate::difficulty::Difficulty;
use crate::model::Model;
use crate::simfile::{SongInfo, Visuals, apply_visuals, render_sm, sanitize};
use crate::stems::{self, StemOptions};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Image and movie files to put in the song folder (`None` = not provided).
#[derive(Clone, Debug, Default)]
pub struct VisualFiles {
    pub banner: Option<PathBuf>,
    pub background: Option<PathBuf>,
    pub jacket: Option<PathBuf>,
    /// Background movie, started at the beginning of the audio.
    pub bg_video: Option<PathBuf>,
}

/// Which charts to generate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Charts {
    /// One chart per difficulty slot, with the typical density of human charts of that
    /// slot (the meter follows).
    Slots(Vec<Difficulty>),
    /// One chart per meter (ITGmania scale, 1–10): the density is tuned to reach the
    /// meter; the charts fill the slots in increasing order (at most 5).
    Meters(Vec<u32>),
}

/// Named sets of meters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Profile {
    /// Meters 2, 3, 4, 5: for players who are starting.
    #[default]
    Beginner,
    /// Meters 2, 4, 6, 8, 10: the whole range of the game, one chart per slot.
    Full,
}

impl Profile {
    pub fn meters(self) -> Vec<u32> {
        match self {
            Profile::Beginner => vec![2, 3, 4, 5],
            Profile::Full => vec![2, 4, 6, 8, 10],
        }
    }
}

impl Default for Charts {
    /// The meters of the default profile.
    fn default() -> Self {
        Charts::Meters(Profile::default().meters())
    }
}

/// Parses meters like "2-5", "2,4,6" or "1-3,8" (validated by [`assign_slots`]).
pub fn parse_meters(s: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let parse = |x: &str| {
            x.trim()
                .parse::<u32>()
                .with_context(|| format!("invalid meter {x:?} in {s:?}"))
        };
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (parse(a)?, parse(b)?);
                anyhow::ensure!(a <= b, "invalid meter range {part:?}");
                out.extend(a..=b);
            }
            None => out.push(parse(part)?),
        }
    }
    anyhow::ensure!(!out.is_empty(), "no meter in {s:?}");
    Ok(out)
}

/// Options of [`create_song`].
#[derive(Clone, Debug)]
pub struct SongOptions {
    pub charts: Charts,
    pub seed: u64,
    /// Folder in which the song folder is created.
    pub output: PathBuf,
    /// Forced BPM / `#OFFSET` (detected when `None`).
    pub bpm: Option<f64>,
    pub offset: Option<f64>,
    /// Title and artist (default: the audio file tags, then its file name).
    pub title: Option<String>,
    pub artist: Option<String>,
    /// Use Demucs source separation.
    pub stems: bool,
    /// Torch device for Demucs.
    pub device: Option<String>,
    /// Model file (default: the embedded model).
    pub model: Option<PathBuf>,
    pub visuals: VisualFiles,
}

impl Default for SongOptions {
    fn default() -> Self {
        SongOptions {
            charts: Charts::default(),
            seed: 0,
            output: PathBuf::from("."),
            bpm: None,
            offset: None,
            title: None,
            artist: None,
            stems: true,
            device: None,
            model: None,
            visuals: VisualFiles::default(),
        }
    }
}

/// Name of the song folder (and of its `.sm`) for a title.
pub fn folder_name(title: &str) -> String {
    let f = sanitize(title).replace(['/', '?', '*', '"', '<', '>', '|'], "_");
    if f.is_empty() { "song".into() } else { f }
}

/// Stems from Demucs, or `None` (with a warning) when it is unavailable.
pub fn load_stems(audio: &Path, device: Option<String>) -> Option<Stems> {
    let opts = StemOptions {
        device,
        ..StemOptions::default()
    };
    eprintln!("separating sources with Demucs (cached after the first run)...");
    match stems::separate(audio, &opts) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("warning: no stems ({e:#}); analysing the full mix only");
            None
        }
    }
}

/// Puts `src` into `dir` (hard link when possible, else copy) under a name that is
/// safe in simfile tags, and returns that name. A file already in place is kept.
pub fn install_file(src: &Path, dir: &Path) -> Result<String> {
    let name = src
        .file_name()
        .with_context(|| format!("no file name in {}", src.display()))?
        .to_string_lossy();
    // '=' and ',' separate the fields of #BGCHANGES.
    let name = sanitize(&name).replace(['=', ','], "_");
    let dest = dir.join(&name);
    if std::fs::canonicalize(src).ok() == std::fs::canonicalize(&dest).ok() {
        return Ok(name);
    }
    let _ = std::fs::remove_file(&dest);
    if std::fs::hard_link(src, &dest).is_err() {
        std::fs::copy(src, &dest)
            .with_context(|| format!("copying {} to {}", src.display(), dest.display()))?;
    }
    Ok(name)
}

/// Installs the provided visual files into `dir`; returns their names.
pub fn install_visuals(files: &VisualFiles, dir: &Path) -> Result<Visuals> {
    let install = |f: &Option<PathBuf>| f.as_deref().map_or(Ok(String::new()), |p| install_file(p, dir));
    Ok(Visuals {
        banner: install(&files.banner)?,
        background: install(&files.background)?,
        jacket: install(&files.jacket)?,
        bg_video: install(&files.bg_video)?,
    })
}

/// Analyzes `audio`, generates the charts and writes the song folder
/// (`<output>/<title>/<title>.sm` + audio + visuals). Returns the `.sm` path.
pub fn create_song(audio_path: &Path, opts: &SongOptions) -> Result<PathBuf> {
    let model = match &opts.model {
        Some(p) => Model::from_json(&std::fs::read_to_string(p)?)?,
        None => Model::embedded()?,
    };
    // (slot, target meter) of each chart, checked before the (slow) analysis.
    let plan: Vec<(Difficulty, Option<u32>)> = match &opts.charts {
        Charts::Slots(ds) => ds.iter().map(|&d| (d, None)).collect(),
        Charts::Meters(ms) => assign_slots(&model, ms)?
            .into_iter()
            .map(|(d, m)| (d, Some(m)))
            .collect(),
    };
    let audio = decode_file(audio_path)?;
    let stems = if opts.stems {
        load_stems(audio_path, opts.device.clone())
    } else {
        None
    };
    let analysis_opts = AnalysisOptions {
        bpm: opts.bpm,
        offset: opts.offset,
        ..AnalysisOptions::default()
    };
    let analysis = SongAnalysis::compute(&audio, stems.as_ref(), &analysis_opts);
    eprintln!(
        "BPM {:.3}, offset {:.3}",
        analysis.grid.bpm,
        analysis.grid.offset()
    );

    let stem_name = audio_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let title = opts.title.clone().or(audio.title.clone()).unwrap_or(stem_name);
    let artist = opts.artist.clone().or(audio.artist.clone()).unwrap_or_default();
    let folder = folder_name(&title);
    let dir = opts.output.join(&folder);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let music = install_file(audio_path, &dir)?;
    let visuals = install_visuals(&opts.visuals, &dir)?;

    let gen_opts = GenOptions::default();
    let charts: Vec<_> = plan
        .iter()
        .map(|&(d, meter)| {
            let c = match meter {
                None => generate(&analysis, &model, d, opts.seed, &gen_opts),
                Some(m) => generate_for_meter(&analysis, &model, d, m, opts.seed, &gen_opts),
            };
            let target = meter.map_or(String::new(), |m| format!(" (target {m})"));
            eprintln!(
                "{:>9}: {:>4} rows, meter {}{target}",
                d.name(),
                c.rows.len(),
                c.meter
            );
            c
        })
        .collect();
    let info = SongInfo {
        title,
        artist,
        music,
        credit: format!("itg-charter {} (seed {})", env!("CARGO_PKG_VERSION"), opts.seed),
        bpm: analysis.grid.bpm,
        offset: analysis.grid.offset(),
        sample_start: analysis.sample_start(12.0),
        sample_length: 12.0,
        visuals,
    };
    let sm = dir.join(format!("{folder}.sm"));
    std::fs::write(&sm, render_sm(&info, &charts))?;
    Ok(sm)
}

/// Adds visuals to an existing song: installs the files next to the `.sm` and
/// declares them in it. Tags of files not provided are left unchanged.
pub fn decorate(sm_path: &Path, files: &VisualFiles) -> Result<()> {
    let dir = sm_path.parent().context("simfile has no folder")?;
    let visuals = install_visuals(files, dir)?;
    let text = std::fs::read_to_string(sm_path).with_context(|| format!("reading {}", sm_path.display()))?;
    std::fs::write(sm_path, apply_visuals(&text, &visuals)?)?;
    Ok(())
}
