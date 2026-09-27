//! Generation invariants on a synthetic song (no audio file, no Demucs).

use itg_charter::analysis::{AnalysisOptions, SongAnalysis};
use itg_charter::audio::Audio;
use itg_charter::chart::{GenOptions, generate};
use itg_charter::difficulty::Difficulty;
use itg_charter::model::Model;
use itg_charter::simfile::{Cell, OutChart, ROWS_PER_BEAT, SongInfo, render_sm};
use itg_charter::synth::test_song;
use std::sync::OnceLock;

fn analysis() -> &'static SongAnalysis {
    static A: OnceLock<SongAnalysis> = OnceLock::new();
    A.get_or_init(|| {
        let audio = Audio::from_samples(test_song(128.0, 0.35, 40.0, 44100), 44100);
        SongAnalysis::compute(&audio, None, &AnalysisOptions::default())
    })
}

fn chart(d: Difficulty, seed: u64) -> OutChart {
    generate(
        analysis(),
        &Model::embedded().unwrap(),
        d,
        seed,
        &GenOptions::default(),
    )
}

fn note_rows(c: &OutChart) -> usize {
    c.rows
        .iter()
        .filter(|(_, r)| r.iter().any(|x| matches!(x, Cell::Tap | Cell::HoldHead)))
        .count()
}

#[test]
fn detects_tempo_of_synthetic_song() {
    let g = analysis().grid;
    assert_eq!(g.bpm, 128.0);
    let period = 60.0 / 128.0;
    let err = ((g.beat0 - 0.35) / period).rem_euclid(1.0);
    assert!(err < 0.06 || err > 0.94, "beat0 {} is off the beat", g.beat0);
}

#[test]
fn same_seed_same_chart_and_different_seed_differs() {
    let a = chart(Difficulty::Hard, 7);
    let b = chart(Difficulty::Hard, 7);
    assert_eq!(a.rows, b.rows);
    let c = chart(Difficulty::Hard, 8);
    assert_ne!(a.rows, c.rows, "seed should change the chart");
}

#[test]
fn rendered_file_is_byte_identical_for_same_seed() {
    let info = SongInfo {
        title: "t".into(),
        artist: "a".into(),
        music: "m.mp3".into(),
        credit: String::new(),
        bpm: analysis().grid.bpm,
        offset: analysis().grid.offset(),
        sample_start: 0.0,
        sample_length: 12.0,
    };
    let render = || render_sm(&info, &Difficulty::ALL.map(|d| chart(d, 3)));
    assert_eq!(render(), render());
}

#[test]
fn density_increases_with_difficulty() {
    let counts: Vec<usize> = Difficulty::ALL.iter().map(|&d| note_rows(&chart(d, 1))).collect();
    for w in counts.windows(2) {
        assert!(w[0] < w[1], "row counts should increase: {counts:?}");
    }
    let meters: Vec<u32> = Difficulty::ALL.iter().map(|&d| chart(d, 1).meter).collect();
    for w in meters.windows(2) {
        assert!(w[0] <= w[1], "meters should not decrease: {meters:?}");
    }
}

#[test]
fn holds_are_well_formed_and_never_overlapped() {
    for d in Difficulty::ALL {
        for seed in 0..3 {
            let c = chart(d, seed);
            let mut open = [false; 4];
            for (pos, row) in &c.rows {
                for col in 0..4 {
                    match row[col] {
                        Cell::HoldHead => {
                            assert!(!open[col], "{d:?}: hold starts inside a hold at {pos}");
                            open[col] = true;
                        }
                        Cell::Tail => {
                            assert!(open[col], "{d:?}: tail without head at {pos}");
                            open[col] = false;
                        }
                        Cell::Tap => assert!(!open[col], "{d:?}: tap inside a hold at {pos}"),
                        _ => {}
                    }
                }
            }
            assert_eq!(open, [false; 4], "{d:?}: unterminated hold");
        }
    }
}

#[test]
fn rows_are_on_standard_snaps_and_jumps_are_pairs() {
    for d in Difficulty::ALL {
        let c = chart(d, 5);
        for (pos, row) in &c.rows {
            let p = pos % ROWS_PER_BEAT;
            assert!(
                p.is_multiple_of(12) || p.is_multiple_of(16),
                "{d:?}: row at {pos} is not a 4th/8th/12th/16th"
            );
            let hits = row
                .iter()
                .filter(|x| matches!(x, Cell::Tap | Cell::HoldHead))
                .count();
            assert!(hits <= 2, "{d:?}: more than two arrows at {pos}");
        }
    }
    let beginner = chart(Difficulty::Beginner, 5);
    assert!(
        beginner.rows.iter().all(|(_, r)| !r.contains(&Cell::HoldHead)),
        "human Beginner charts have no holds (median 0), neither should ours"
    );
}

#[test]
fn difficulties_are_independent_of_each_other() {
    // Each difficulty draws from its own random stream: generating Hard alone or
    // after the others gives the same chart.
    let alone = chart(Difficulty::Hard, 11);
    let _ = chart(Difficulty::Easy, 11);
    let again = chart(Difficulty::Hard, 11);
    assert_eq!(alone.rows, again.rows);
}
