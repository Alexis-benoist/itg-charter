use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::chart::{GenOptions, generate};
use itg_charter::difficulty::{Difficulty, parse_list};
use itg_charter::model::Model;
use itg_charter::simfile::{SongInfo, render_sm, sanitize};
use itg_charter::stems::{self, StemOptions};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    version,
    about = "Generates ITGmania (In The Groove) step charts from an audio file"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a song folder (.sm + audio) with one chart per difficulty.
    Gen(GenArgs),
    /// Learn pattern statistics from human-made charts (e.g. the game's Songs folder).
    Train {
        /// Folder searched recursively for .sm/.ssc files.
        songs: PathBuf,
        /// Output model file.
        #[arg(short, long, default_value = "model/model.json")]
        output: PathBuf,
    },
    /// Print the detected BPM and offset of an audio file.
    Analyze {
        audio: PathBuf,
        #[arg(long)]
        no_stems: bool,
    },
}

#[derive(clap::Args)]
struct GenArgs {
    /// Audio file (mp3, ogg, flac, wav).
    audio: PathBuf,
    /// Difficulties, comma separated: beginner,easy,medium,hard,challenge or "all".
    #[arg(short, long, default_value = "all")]
    difficulties: String,
    /// Random seed: the same audio, seed and options always give the same .sm file.
    #[arg(short, long, default_value_t = 0)]
    seed: u64,
    /// Output folder; the song folder is created inside it (e.g. ~/.itgmania/Songs/Generated).
    #[arg(short, long, default_value = ".")]
    output: PathBuf,
    /// Force the BPM instead of detecting it.
    #[arg(long)]
    bpm: Option<f64>,
    /// Force the simfile #OFFSET (seconds) instead of detecting it.
    #[arg(long, allow_hyphen_values = true)]
    offset: Option<f64>,
    #[arg(long)]
    title: Option<String>,
    #[arg(long)]
    artist: Option<String>,
    /// Skip Demucs source separation (faster, less precise).
    #[arg(long)]
    no_stems: bool,
    /// Torch device for Demucs (cuda or cpu).
    #[arg(long)]
    device: Option<String>,
    /// Model file (default: the model embedded in the binary).
    #[arg(long)]
    model: Option<PathBuf>,
}

fn load_stems(audio: &Path, no_stems: bool, device: Option<String>) -> Option<itg_charter::analysis::Stems> {
    if no_stems {
        return None;
    }
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

fn run_gen(args: GenArgs) -> Result<()> {
    let difficulties: Vec<Difficulty> = parse_list(&args.difficulties)?;
    let model = match &args.model {
        Some(p) => Model::from_json(&std::fs::read_to_string(p)?)?,
        None => Model::embedded()?,
    };
    let audio = itg_charter::audio::decode_file(&args.audio)?;
    let stems = load_stems(&args.audio, args.no_stems, args.device.clone());
    let opts = AnalysisOptions {
        bpm: args.bpm,
        offset: args.offset,
        ..AnalysisOptions::default()
    };
    let analysis = SongAnalysis::compute(&audio, stems.as_ref(), &opts);
    eprintln!(
        "BPM {:.3}, offset {:.3}",
        analysis.grid.bpm,
        analysis.grid.offset()
    );

    let stem_name = args
        .audio
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let title = args.title.or(audio.title.clone()).unwrap_or(stem_name);
    let artist = args.artist.or(audio.artist.clone()).unwrap_or_default();
    let folder = sanitize(&title).replace(['/', '?', '*', '"', '<', '>', '|'], "_");
    let dir = args.output.join(if folder.is_empty() {
        "song".into()
    } else {
        folder.clone()
    });
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let music = args
        .audio
        .file_name()
        .context("audio file name")?
        .to_string_lossy()
        .into_owned();
    let dest = dir.join(&music);
    if std::fs::canonicalize(&args.audio).ok() != std::fs::canonicalize(&dest).ok() {
        std::fs::copy(&args.audio, &dest).with_context(|| format!("copying audio to {}", dest.display()))?;
    }

    let gen_opts = GenOptions::default();
    let charts: Vec<_> = difficulties
        .iter()
        .map(|&d| {
            let c = generate(&analysis, &model, d, args.seed, &gen_opts);
            eprintln!("{:>9}: {:>4} rows, meter {}", d.name(), c.rows.len(), c.meter);
            c
        })
        .collect();
    let info = SongInfo {
        title,
        artist,
        music,
        credit: format!("itg-charter {} (seed {})", env!("CARGO_PKG_VERSION"), args.seed),
        bpm: analysis.grid.bpm,
        offset: analysis.grid.offset(),
        sample_start: analysis.sample_start(12.0),
        sample_length: 12.0,
    };
    let sm = dir.join(format!("{}.sm", if folder.is_empty() { "song" } else { &folder }));
    std::fs::write(&sm, render_sm(&info, &charts))?;
    println!("{}", sm.display());
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Gen(args) => run_gen(args)?,
        Command::Train { songs, output } => {
            let model = Model::train(&songs, |i, n| {
                if i % 100 == 0 {
                    eprintln!("[{i}/{n}] simfiles");
                }
            })?;
            for (d, s) in Difficulty::ALL.iter().zip(&model.stats) {
                eprintln!(
                    "{:>9}: {:>5} charts, nps median {:.2}, jumps {:.1}%, holds {:.1}%, meter median {}",
                    d.name(),
                    s.charts,
                    s.nps.p50,
                    100.0 * s.jump_ratio.p50,
                    100.0 * s.hold_ratio.p50,
                    s.meter.p50
                );
            }
            if let Some(dir) = output.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&output, model.to_json())?;
            eprintln!("model written to {}", output.display());
        }
        Command::Analyze { audio, no_stems } => {
            let a = itg_charter::audio::decode_file(&audio)?;
            let stems = load_stems(&audio, no_stems, None);
            let r = SongAnalysis::compute(&a, stems.as_ref(), &AnalysisOptions::default());
            println!(
                "bpm {:.3}\noffset {:.3}\naubio_bpm {:.2}",
                r.grid.bpm,
                r.grid.offset(),
                r.aubio_bpm
            );
        }
    }
    Ok(())
}
