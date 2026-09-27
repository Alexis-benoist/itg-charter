//! End-to-end tests of the binary on the mp3 fixture.

use itg_charter::simfile::Simfile;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_song.mp3")
}

fn out_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("itg-charter-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn run(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_itg-charter"))
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn generate_song(dir: &Path, extra: &[&str]) -> PathBuf {
    let audio = fixture();
    let mut args = vec![
        "gen",
        audio.to_str().unwrap(),
        "--no-stems",
        "-o",
        dir.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    PathBuf::from(run(&args))
}

#[test]
fn generates_a_song_folder_readable_as_simfile() {
    let dir = out_dir("folder");
    let sm = generate_song(&dir, &["-d", "easy,hard", "-s", "42"]);
    assert_eq!(sm, dir.join("Test Song/Test Song.sm"));
    assert!(
        dir.join("Test Song/test_song.mp3").exists(),
        "audio copied next to the .sm"
    );
    let sim = Simfile::load(&sm).unwrap();
    assert_eq!(sim.tag("TITLE"), Some("Test Song"));
    assert_eq!(sim.tag("ARTIST"), Some("itg-charter"));
    assert_eq!(sim.tag("MUSIC"), Some("test_song.mp3"));
    assert_eq!(sim.tag("BPMS"), Some("0.000=128.000"));
    let diffs: Vec<&str> = sim.charts.iter().map(|c| c.difficulty.as_str()).collect();
    assert_eq!(diffs, ["Easy", "Hard"]);
    assert!(
        sim.charts
            .iter()
            .all(|c| c.steps_type == "dance-single" && !c.rows().unwrap().is_empty())
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn output_is_reproducible_with_a_seed() {
    let (a, b, c) = (out_dir("a"), out_dir("b"), out_dir("c"));
    let read = |p: PathBuf| std::fs::read(p).unwrap();
    let first = read(generate_song(&a, &["-s", "123"]));
    let second = read(generate_song(&b, &["-s", "123"]));
    let other = read(generate_song(&c, &["-s", "124"]));
    assert_eq!(first, second, "same seed must give byte-identical .sm files");
    assert_ne!(first, other);
    for d in [a, b, c] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn chart_does_not_depend_on_other_selected_difficulties() {
    let (a, b) = (out_dir("hard"), out_dir("all"));
    let hard_only = Simfile::load(&generate_song(&a, &["-d", "hard", "-s", "9"])).unwrap();
    let all = Simfile::load(&generate_song(&b, &["-d", "all", "-s", "9"])).unwrap();
    let pick = |s: &Simfile| {
        s.charts
            .iter()
            .find(|c| c.difficulty == "Hard")
            .unwrap()
            .notes
            .clone()
    };
    assert_eq!(pick(&hard_only), pick(&all));
    let _ = std::fs::remove_dir_all(a);
    let _ = std::fs::remove_dir_all(b);
}

#[test]
fn forced_bpm_and_offset_are_written() {
    let dir = out_dir("forced");
    let sim = Simfile::load(&generate_song(
        &dir,
        &["-d", "medium", "--bpm", "64", "--offset", "-0.5"],
    ))
    .unwrap();
    assert_eq!(sim.tag("BPMS"), Some("0.000=64.000"));
    assert_eq!(sim.tag("OFFSET"), Some("-0.500"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rejects_unknown_difficulty() {
    let out = Command::new(env!("CARGO_BIN_EXE_itg-charter"))
        .args([
            "gen",
            fixture().to_str().unwrap(),
            "--no-stems",
            "-d",
            "nightmare",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown difficulty"));
}
