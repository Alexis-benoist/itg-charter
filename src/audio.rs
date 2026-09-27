//! Audio decoding (symphonia) to mono f32 samples.

use anyhow::{Context, Result, anyhow};
use std::fs::File;
use std::path::Path;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, MetadataRevision, StandardTag};

/// Decoded mono audio.
#[derive(Clone, Debug)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub title: Option<String>,
    pub artist: Option<String>,
    /// Silence added in front of the decoded audio to match what the game plays
    /// (see [`decode_file`]). Stems decoded by other tools need the same padding.
    pub game_padding: usize,
}

impl Audio {
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32) -> Audio {
        Audio {
            samples,
            sample_rate,
            title: None,
            artist: None,
            game_padding: 0,
        }
    }

    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

fn read_tags(rev: &MetadataRevision, title: &mut Option<String>, artist: &mut Option<String>) {
    for tag in &rev.media.tags {
        match &tag.std {
            Some(StandardTag::TrackTitle(v)) if !v.trim().is_empty() => *title = Some(v.to_string()),
            Some(StandardTag::Artist(v)) if !v.trim().is_empty() => *artist = Some(v.to_string()),
            _ => {}
        }
    }
}

/// Decodes any format supported by symphonia (mp3, ogg/vorbis, flac, wav) to mono.
pub fn decode_file(path: &Path) -> Result<Audio> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .with_context(|| format!("unsupported audio format: {}", path.display()))?;
    let (mut title, mut artist) = (None, None);
    if let Some(rev) = format.metadata().current() {
        read_tags(rev, &mut title, &mut artist);
    }
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("no audio track in {}", path.display()))?;
    let track_id = track.id;
    let is_mp3 = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("mp3"));
    let lame_delay = track.delay.filter(|_| is_mp3);
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| anyhow!("no audio codec parameters"))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .context("unsupported codec")?;

    let mut mono = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    let mut sample_rate = 0;
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e).context("reading audio"),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let channels = buf.spec().channels().count().max(1);
                sample_rate = buf.spec().rate();
                interleaved.resize(buf.samples_interleaved(), 0.0);
                buf.copy_to_slice_interleaved(&mut interleaved);
                mono.extend(
                    interleaved
                        .chunks(channels)
                        .map(|f| f.iter().sum::<f32>() / channels as f32),
                );
            }
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decoding audio"),
        }
    }
    if let Some(rev) = format.metadata().current() {
        read_tags(rev, &mut title, &mut artist);
    }
    anyhow::ensure!(
        !mono.is_empty() && sample_rate > 0,
        "no audio decoded from {}",
        path.display()
    );
    // symphonia is gapless for MP3s with a Xing/Info + LAME tag: it drops the tag frame
    // and trims the encoder delay. ITGmania plays both (the tag frame decodes to
    // silence), so the music starts later in the game. Measured with
    // `examples/eval_sync.rs`: without this, human-synced LAME-tagged MP3s had a median
    // phase error of -47 ms (1152 + 1105 samples = 51 ms at 44.1 kHz).
    let game_padding = lame_delay.map_or(0, |d| {
        let frame = if sample_rate >= 32000 { 1152 } else { 576 };
        d as usize + frame
    });
    if game_padding > 0 {
        mono.splice(0..0, std::iter::repeat_n(0.0, game_padding));
    }
    Ok(Audio {
        samples: mono,
        sample_rate,
        title,
        artist,
        game_padding,
    })
}
