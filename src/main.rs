use anyhow::Result;
use clap::{Parser, Subcommand};
use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::difficulty::{Difficulty, parse_list};
use itg_charter::model::Model;
use itg_charter::song::{self, Charts, SongOptions, VisualFiles};
use itg_charter::style::Style;
use std::path::PathBuf;

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
    /// Generate a song folder (.sm + audio + visuals) with one chart per difficulty.
    Gen(Box<GenArgs>),
    /// Add banner / background / jacket / background movie to an existing song folder.
    Decorate {
        /// The song's .sm file; the files are put next to it.
        sm: PathBuf,
        #[command(flatten)]
        visuals: VisualArgs,
    },
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
struct VisualArgs {
    /// Banner image (418×164), copied into the song folder.
    #[arg(long)]
    banner: Option<PathBuf>,
    /// Background image, copied into the song folder.
    #[arg(long)]
    background: Option<PathBuf>,
    /// Jacket image (square), copied into the song folder.
    #[arg(long)]
    jacket: Option<PathBuf>,
    /// Background movie, copied into the song folder and started with the audio.
    #[arg(long)]
    bg_video: Option<PathBuf>,
}

impl From<VisualArgs> for VisualFiles {
    fn from(v: VisualArgs) -> Self {
        VisualFiles {
            banner: v.banner,
            background: v.background,
            jacket: v.jacket,
            bg_video: v.bg_video,
        }
    }
}

#[derive(clap::Args)]
struct GenArgs {
    /// Audio file (mp3, ogg, flac, wav).
    audio: PathBuf,
    /// Meters to generate on the ITGmania scale (1-10), e.g. "2-5" or "1,3,6"; at most
    /// 5, one per difficulty slot. Default: 2,4,6,8,10 (unless --difficulties is given).
    #[arg(short, long, conflicts_with = "difficulties")]
    meters: Option<String>,
    /// Named set of meters (used when neither --meters nor --difficulties is given):
    /// beginner = 2,3,4,5; full = 2,4,6,8,10.
    #[arg(short, long, value_enum, default_value_t = song::Profile::Full)]
    profile: song::Profile,
    /// Instead of meters: difficulty slots with the typical density of human charts,
    /// comma separated: beginner,easy,medium,hard,challenge or "all".
    #[arg(short, long)]
    difficulties: Option<String>,
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
    /// Model file (default: the model of --style embedded in the binary).
    #[arg(long, conflicts_with = "style")]
    model: Option<PathBuf>,
    /// Charting style learned from human packs (see training/STYLES.md).
    #[arg(long, value_enum, default_value_t)]
    style: Style,
    #[command(flatten)]
    visuals: VisualArgs,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Gen(args) => {
            let charts = match (&args.difficulties, &args.meters) {
                (Some(d), _) => Charts::Slots(parse_list(d)?),
                (None, Some(m)) => Charts::Meters(song::parse_meters(m)?),
                (None, None) => Charts::Meters(args.profile.meters()),
            };
            let opts = SongOptions {
                charts,
                seed: args.seed,
                output: args.output,
                bpm: args.bpm,
                offset: args.offset,
                title: args.title,
                artist: args.artist,
                stems: !args.no_stems,
                device: args.device,
                model: args.model,
                style: args.style,
                visuals: args.visuals.into(),
            };
            println!("{}", song::create_song(&args.audio, &opts)?.display());
        }
        Command::Decorate { sm, visuals } => {
            song::decorate(&sm, &visuals.into())?;
            println!("{}", sm.display());
        }
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
            let stems = if no_stems {
                None
            } else {
                song::load_stems(&audio, None)
            };
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
