//! Stem separation: cache behaviour (no Demucs needed) and a real Demucs run (ignored).

use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::audio::decode_file;
use itg_charter::chart::{GenOptions, generate};
use itg_charter::difficulty::Difficulty;
use itg_charter::model::Model;
use itg_charter::stems::{STEM_NAMES, StemOptions, cache_path, separate};
use itg_charter::synth::{drum_loop, test_song, wav_bytes};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_song.mp3")
}

#[test]
fn cached_stems_are_used_without_running_demucs() {
    let cache = std::env::temp_dir().join(format!("itg-charter-stems-{}", std::process::id()));
    let opts = StemOptions {
        // A python that does not exist: the test fails if Demucs is invoked.
        python: Some(PathBuf::from("/nonexistent/python")),
        cache_dir: Some(cache.clone()),
        device: None,
    };
    let dir = cache_path(&fixture(), &opts).unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    let sr = 44100;
    let drums = drum_loop(128.0, 0.35, 24.0, sr);
    let song = test_song(128.0, 0.35, 24.0, sr);
    let melody: Vec<f32> = song.iter().zip(&drums).map(|(s, d)| s - d).collect();
    let silence = vec![0f32; drums.len()];
    for (name, samples) in STEM_NAMES.iter().zip([&drums, &silence, &silence, &melody]) {
        std::fs::write(dir.join(format!("{name}.wav")), wav_bytes(samples, sr)).unwrap();
    }
    let stems = separate(&fixture(), &opts).expect("stems from cache");
    assert_eq!(stems.drums.len(), drums.len());

    let audio = decode_file(&fixture()).unwrap();
    let a = SongAnalysis::compute(&audio, Some(&stems), &AnalysisOptions::default());
    assert_eq!(a.grid.bpm, 128.0);
    assert!(a.stems.is_some());
    let model = Model::embedded().unwrap();
    let hard = generate(&a, &model, Difficulty::Hard, 1, &GenOptions::default());
    assert!(hard.rows.len() > 20);
    let _ = std::fs::remove_dir_all(cache);
}

#[test]
fn missing_demucs_is_a_clear_error() {
    let opts = StemOptions {
        python: Some(PathBuf::from("/nonexistent/python")),
        cache_dir: Some(std::env::temp_dir().join("itg-charter-empty-cache")),
        device: None,
    };
    let err = separate(&fixture(), &opts).unwrap_err().to_string();
    assert!(err.contains("Demucs python not found"), "{err}");
}

/// Runs the real Demucs (needs the venv, see CLAUDE.md): `cargo test -- --ignored`.
#[test]
#[ignore]
fn real_demucs_separation() {
    let cache = std::env::temp_dir().join(format!("itg-charter-demucs-{}", std::process::id()));
    let opts = StemOptions {
        cache_dir: Some(cache.clone()),
        ..StemOptions::default()
    };
    let stems = separate(&fixture(), &opts).expect("demucs run");
    let audio = decode_file(&fixture()).unwrap();
    assert!(stems.drums.len().abs_diff(audio.samples.len()) < 4410);
    // Second call reads the cache.
    let again = separate(&fixture(), &opts).unwrap();
    assert_eq!(again.drums, stems.drums);
    let _ = std::fs::remove_dir_all(cache);
}
