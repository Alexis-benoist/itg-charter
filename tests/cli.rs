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

/// Dummy visual files (content does not matter to itg-charter).
fn visual_files(dir: &Path) -> [PathBuf; 4] {
    std::fs::create_dir_all(dir).unwrap();
    ["bn.png", "bg.png", "jacket.png", "My Song=v1, final-bg.mp4"].map(|f| {
        let p = dir.join(f);
        std::fs::write(&p, f).unwrap();
        p
    })
}

#[test]
fn visuals_given_to_gen_or_decorate_give_the_same_song() {
    let (a, b) = (out_dir("visuals-gen"), out_dir("visuals-decorate"));
    let files = visual_files(&a.join("src"));
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    let flags = [
        "--banner".to_string(),
        s(&files[0]),
        "--background".into(),
        s(&files[1]),
        "--jacket".into(),
        s(&files[2]),
        "--bg-video".into(),
        s(&files[3]),
    ];
    let flag_refs: Vec<&str> = flags.iter().map(String::as_str).collect();

    let mut gen_args = vec!["-d", "easy", "-s", "5"];
    gen_args.extend(&flag_refs);
    let with = generate_song(&a, &gen_args);
    let sim = Simfile::load(&with).unwrap();
    let song_dir = with.parent().unwrap();
    assert_eq!(sim.tag("BANNER"), Some("bn.png"));
    assert_eq!(sim.tag("BACKGROUND"), Some("bg.png"));
    assert_eq!(sim.tag("JACKET"), Some("jacket.png"));
    // '=' and ',' would break #BGCHANGES: the installed movie is renamed.
    let movie = "My Song_v1_ final-bg.mp4";
    assert!(song_dir.join(movie).exists());
    let bg = sim.tag("BGCHANGES").unwrap();
    assert!(
        bg.ends_with(&format!("={movie}=1.000=0=0=0=StretchNoLoop====")),
        "{bg}"
    );
    // The movie starts at audio time 0: beat = OFFSET × BPM / 60.
    let timing = sim.timing(&sim.charts[0]).unwrap();
    let beat: f64 = bg.split('=').next().unwrap().parse().unwrap();
    assert!((beat - timing.offset * timing.bpms[0].1 / 60.0).abs() < 0.001);
    for f in ["bn.png", "bg.png", "jacket.png"] {
        assert!(song_dir.join(f).exists(), "{f} copied");
    }

    // gen without visuals, then decorate: same simfile.
    let plain = generate_song(&b, &["-d", "easy", "-s", "5"]);
    let mut dec_args = vec!["decorate", plain.to_str().unwrap()];
    dec_args.extend(&flag_refs);
    run(&dec_args);
    assert_eq!(
        std::fs::read_to_string(&plain).unwrap(),
        std::fs::read_to_string(&with).unwrap()
    );
    let _ = std::fs::remove_dir_all(a);
    let _ = std::fs::remove_dir_all(b);
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

#[test]
fn default_charts_are_meters_2_to_5_and_meters_are_configurable() {
    let dir = out_dir("meters");
    let sim = Simfile::load(&generate_song(&dir, &["-s", "3"])).unwrap();
    let got: Vec<(&str, i32)> = sim
        .charts
        .iter()
        .map(|c| (c.difficulty.as_str(), c.meter))
        .collect();
    assert_eq!(got, [("Beginner", 2), ("Easy", 3), ("Medium", 4), ("Hard", 5)]);

    let sim = Simfile::load(&generate_song(&dir, &["-m", "1,3-4"])).unwrap();
    let meters: Vec<i32> = sim.charts.iter().map(|c| c.meter).collect();
    assert_eq!(meters, [1, 3, 4]);

    let bad = Command::new(env!("CARGO_BIN_EXE_itg-charter"))
        .args(["gen", fixture().to_str().unwrap(), "--no-stems", "-m", "0-4"])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("out of range 1..=10"));
    let _ = std::fs::remove_dir_all(dir);
}
