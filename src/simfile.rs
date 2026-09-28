//! Reading and writing StepMania simfiles (`.sm`, and `.ssc` for reading).

use anyhow::{Context, Result, bail};
use std::path::Path;

/// A parsed simfile: song-level tags plus its charts.
#[derive(Debug, Clone, Default)]
pub struct Simfile {
    pub tags: Vec<(String, String)>,
    pub charts: Vec<ChartData>,
}

/// One chart as found in a simfile, notes still in textual form.
#[derive(Debug, Clone, Default)]
pub struct ChartData {
    pub steps_type: String,
    pub description: String,
    pub difficulty: String,
    pub meter: i32,
    pub notes: String,
    /// Per-chart tags (`.ssc` split timing: BPMS, STOPS, OFFSET, ...).
    pub tags: Vec<(String, String)>,
}

/// One cell of a note row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cell {
    Empty,
    Tap,
    HoldHead,
    RollHead,
    Tail,
    Mine,
    Lift,
    Fake,
    Other,
}

impl Cell {
    pub fn from_char(c: char) -> Cell {
        match c {
            '0' => Cell::Empty,
            '1' => Cell::Tap,
            '2' => Cell::HoldHead,
            '4' => Cell::RollHead,
            '3' => Cell::Tail,
            'M' | 'm' => Cell::Mine,
            'L' | 'l' => Cell::Lift,
            'F' | 'f' => Cell::Fake,
            _ => Cell::Other,
        }
    }

    pub fn to_char(self) -> char {
        match self {
            Cell::Empty => '0',
            Cell::Tap => '1',
            Cell::HoldHead => '2',
            Cell::RollHead => '4',
            Cell::Tail => '3',
            Cell::Mine => 'M',
            Cell::Lift => 'L',
            Cell::Fake => 'F',
            Cell::Other => 'K',
        }
    }
}

/// A non-empty note row with its beat position.
#[derive(Debug, Clone, PartialEq)]
pub struct NoteRow {
    pub beat: f64,
    pub cells: Vec<Cell>,
}

fn strip_comments(text: &str) -> String {
    text.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Splits `#KEY:VALUE;` entries. Keys are upper-cased; values are trimmed.
fn split_tags(text: &str) -> Vec<(String, String)> {
    let text = strip_comments(text);
    let mut out = Vec::new();
    let mut rest = text.as_str();
    while let Some(start) = rest.find('#') {
        rest = &rest[start + 1..];
        let Some(colon) = rest.find(':') else { break };
        let key = rest[..colon].trim().to_ascii_uppercase();
        rest = &rest[colon + 1..];
        // A value ends at ';' or, for malformed files, at the next line starting with '#'.
        let end_semi = rest.find(';');
        let end_hash = rest.find("\n#");
        let end = match (end_semi, end_hash) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => rest.len(),
        };
        out.push((key, rest[..end].trim().to_string()));
        rest = &rest[end..];
    }
    out
}

impl Simfile {
    pub fn parse(text: &str, is_ssc: bool) -> Result<Simfile> {
        let text = text.trim_start_matches('\u{feff}');
        let mut sim = Simfile::default();
        let mut current: Option<ChartData> = None;
        for (key, value) in split_tags(text) {
            if is_ssc {
                match key.as_str() {
                    "NOTEDATA" => {
                        if let Some(c) = current.take() {
                            sim.charts.push(c);
                        }
                        current = Some(ChartData::default());
                    }
                    _ => match current.as_mut() {
                        None => sim.tags.push((key, value)),
                        Some(c) => match key.as_str() {
                            "STEPSTYPE" => c.steps_type = value,
                            "DESCRIPTION" => c.description = value,
                            "DIFFICULTY" => c.difficulty = value,
                            "METER" => c.meter = value.parse().unwrap_or(0),
                            "NOTES" | "NOTES2" => c.notes = value,
                            _ => c.tags.push((key, value)),
                        },
                    },
                }
            } else if key == "NOTES" {
                let parts: Vec<&str> = value.splitn(6, ':').collect();
                if parts.len() < 6 {
                    bail!("malformed #NOTES section");
                }
                sim.charts.push(ChartData {
                    steps_type: parts[0].trim().to_string(),
                    description: parts[1].trim().to_string(),
                    difficulty: parts[2].trim().to_string(),
                    meter: parts[3].trim().parse().unwrap_or(0),
                    notes: parts[5].trim().to_string(),
                    tags: Vec::new(),
                });
            } else {
                sim.tags.push((key, value));
            }
        }
        if let Some(c) = current.take() {
            sim.charts.push(c);
        }
        Ok(sim)
    }

    pub fn load(path: &Path) -> Result<Simfile> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let text = String::from_utf8_lossy(&bytes);
        let is_ssc = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ssc"));
        Simfile::parse(&text, is_ssc)
    }

    /// Whether this simfile was written by itg-charter (credit or chart description).
    /// Such files must never be used as human ground truth or training data, e.g. when
    /// generated songs are linked into the game's Songs folder.
    pub fn is_generated(&self) -> bool {
        self.tag("CREDIT").is_some_and(|c| c.contains("itg-charter"))
            || self.charts.iter().any(|c| c.description.contains("itg-charter"))
    }

    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Timing data for a chart, honouring `.ssc` per-chart overrides.
    pub fn timing(&self, chart: &ChartData) -> Result<Timing> {
        let get = |k: &str| {
            chart
                .tags
                .iter()
                .find(|(key, v)| key == k && !v.is_empty())
                .map(|(_, v)| v.as_str())
                .or_else(|| self.tag(k))
        };
        Timing::from_tags(
            get("OFFSET").unwrap_or("0"),
            get("BPMS").unwrap_or(""),
            get("STOPS").or_else(|| get("FREEZES")).unwrap_or(""),
            get("DELAYS").unwrap_or(""),
            get("WARPS").unwrap_or(""),
        )
    }
}

impl ChartData {
    /// Parses the note block into non-empty rows.
    pub fn rows(&self) -> Result<Vec<NoteRow>> {
        let mut out = Vec::new();
        for (m, measure) in self.notes.split(',').enumerate() {
            let lines: Vec<&str> = measure.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
            let n = lines.len();
            for (i, line) in lines.iter().enumerate() {
                let cells: Vec<Cell> = line.chars().map(Cell::from_char).collect();
                if cells.iter().all(|c| *c == Cell::Empty) {
                    continue;
                }
                out.push(NoteRow {
                    beat: m as f64 * 4.0 + 4.0 * i as f64 / n as f64,
                    cells,
                });
            }
        }
        Ok(out)
    }
}

fn parse_pairs(s: &str) -> Result<Vec<(f64, f64)>> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    for item in s.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (a, b) = item
            .split_once('=')
            .with_context(|| format!("bad timing entry {item:?}"))?;
        out.push((a.trim().parse()?, b.trim().parse()?));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(out)
}

/// Beat ↔ seconds conversion (BPM changes, stops and delays; warps are rejected).
#[derive(Debug, Clone)]
pub struct Timing {
    pub offset: f64,
    pub bpms: Vec<(f64, f64)>,
    pub stops: Vec<(f64, f64)>,
    pub delays: Vec<(f64, f64)>,
}

impl Timing {
    pub fn constant(bpm: f64, offset: f64) -> Timing {
        Timing {
            offset,
            bpms: vec![(0.0, bpm)],
            stops: Vec::new(),
            delays: Vec::new(),
        }
    }

    fn from_tags(offset: &str, bpms: &str, stops: &str, delays: &str, warps: &str) -> Result<Timing> {
        let bpms = parse_pairs(bpms)?;
        if bpms.is_empty() {
            bail!("no BPMS");
        }
        if bpms.iter().any(|(_, b)| *b <= 0.0) {
            bail!("negative or zero BPM (warp) not supported");
        }
        if parse_pairs(warps)?.iter().any(|(_, w)| *w > 0.0) {
            bail!("WARPS not supported");
        }
        Ok(Timing {
            offset: offset.trim().parse().unwrap_or(0.0),
            bpms,
            stops: parse_pairs(stops)?,
            delays: parse_pairs(delays)?,
        })
    }

    pub fn is_constant(&self) -> bool {
        self.bpms.len() == 1 && self.stops.is_empty() && self.delays.is_empty()
    }

    /// Time in seconds at which a note on `beat` must be hit.
    pub fn seconds(&self, beat: f64) -> f64 {
        let mut t = -self.offset;
        for (i, &(start, bpm)) in self.bpms.iter().enumerate() {
            // The first segment also covers negative beats.
            let from = if i == 0 { 0.0 } else { start };
            if i > 0 && beat <= start {
                break;
            }
            let end = self.bpms.get(i + 1).map_or(f64::INFINITY, |b| b.0);
            t += (beat.min(end) - from) * 60.0 / bpm;
        }
        // A stop on a beat delays the notes after it; a delay also delays notes on it.
        t += self.stops.iter().filter(|s| s.0 < beat).map(|s| s.1).sum::<f64>();
        t += self
            .delays
            .iter()
            .filter(|d| d.0 <= beat)
            .map(|d| d.1)
            .sum::<f64>();
        t
    }
}

/// Everything needed to write a `.sm` file.
#[derive(Debug, Clone, Default)]
pub struct SongInfo {
    pub title: String,
    pub artist: String,
    pub music: String,
    pub credit: String,
    pub bpm: f64,
    pub offset: f64,
    pub sample_start: f64,
    pub sample_length: f64,
    pub visuals: Visuals,
}

/// Image and movie files of a song, as file names inside the song folder
/// (empty = none).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Visuals {
    pub banner: String,
    pub background: String,
    pub jacket: String,
    /// Background movie, started at the beginning of the audio.
    pub bg_video: String,
}

/// Beat at which the audio time is 0 in a constant-BPM simfile
/// (time(beat) = beat × 60 / BPM − OFFSET): where a background movie must start.
pub fn movie_start_beat(bpm: f64, offset: f64) -> f64 {
    offset * bpm / 60.0
}

/// `#BGCHANGES` value playing `movie` from `beat` (format used by ITG packs).
pub fn bg_changes(movie: &str, beat: f64) -> String {
    format!("{}={movie}=1.000=0=0=0=StretchNoLoop====", fmt3(beat))
}

/// Replaces the value of `#KEY:...;`, or inserts the tag at the end of the header
/// (before the first chart).
pub fn set_tag(sm: &str, key: &str, value: &str) -> String {
    set_tag_after(sm, key, value, None)
}

/// Like [`set_tag`], but a missing tag is inserted right after the `#AFTER:...;` tag
/// when there is one (to follow the order in which [`render_sm`] writes tags).
pub fn set_tag_after(sm: &str, key: &str, value: &str, after: Option<&str>) -> String {
    let value_range = |key: &str| {
        let marker = format!("#{key}:");
        let start = sm.find(&marker)? + marker.len();
        let end = start + sm[start..].find(';')?;
        Some(start..end)
    };
    if let Some(r) = value_range(key) {
        return format!("{}{value}{}", &sm[..r.start], &sm[r.end..]);
    }
    let at = after
        .and_then(value_range)
        .map(|r| r.end + 1)
        .or_else(|| sm.find("\n\n//---").or_else(|| sm.find("\n//---")))
        .or_else(|| sm.find("#NOTES").map(|i| i.saturating_sub(1)))
        .unwrap_or(sm.len());
    format!("{}\n#{key}:{value};{}", &sm[..at], &sm[at..])
}

/// Declares the non-empty `visuals` in an existing `.sm` text. The movie start is
/// computed from the file's own `#BPMS` / `#OFFSET` (constant tempo required).
pub fn apply_visuals(sm: &str, visuals: &Visuals) -> Result<String> {
    let mut out = sm.to_string();
    for (key, value, after) in [
        ("BANNER", &visuals.banner, None),
        ("BACKGROUND", &visuals.background, None),
        // render_sm writes JACKET (only when set) right after BACKGROUND.
        ("JACKET", &visuals.jacket, Some("BACKGROUND")),
    ] {
        if !value.is_empty() {
            out = set_tag_after(&out, key, &sanitize(value), after);
        }
    }
    if !visuals.bg_video.is_empty() {
        let parsed = Simfile::parse(sm, false)?;
        let chart = parsed.charts.first().context("no chart in the simfile")?;
        let timing = parsed.timing(chart)?;
        if timing.bpms.len() != 1 {
            bail!("a background movie needs a constant BPM");
        }
        let beat = movie_start_beat(timing.bpms[0].1, timing.offset);
        out = set_tag(&out, "BGCHANGES", &bg_changes(&sanitize(&visuals.bg_video), beat));
    }
    Ok(out)
}

/// A chart to write: rows positioned in 1/48ths of a beat (192 per measure).
#[derive(Debug, Clone)]
pub struct OutChart {
    pub difficulty: String,
    pub meter: u32,
    pub description: String,
    /// (position in 48ths of a beat, 4 cells), sorted, unique positions.
    pub rows: Vec<(u32, [Cell; 4])>,
}

pub const ROWS_PER_BEAT: u32 = 48;
const ROWS_PER_MEASURE: u32 = ROWS_PER_BEAT * 4;

/// Removes characters that would break the `#TAG:VALUE;` syntax.
pub fn sanitize(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, ';' | '#' | '\\' | '\n' | '\r'))
        .map(|c| if c == ':' { '-' } else { c })
        .collect::<String>()
        .trim()
        .to_string()
}

fn fmt3(x: f64) -> String {
    let s = format!("{x:.3}");
    if s == "-0.000" { "0.000".into() } else { s }
}

fn render_notes(rows: &[(u32, [Cell; 4])]) -> String {
    let last = rows.last().map_or(0, |r| r.0);
    let measures = last / ROWS_PER_MEASURE + 1;
    let mut out = String::new();
    let mut idx = 0;
    for m in 0..measures {
        let start = m * ROWS_PER_MEASURE;
        let end = start + ROWS_PER_MEASURE;
        let mut in_measure = Vec::new();
        while idx < rows.len() && rows[idx].0 < end {
            in_measure.push(rows[idx]);
            idx += 1;
        }
        // Smallest standard resolution that represents every row exactly.
        let res = [4u32, 8, 12, 16, 24, 32, 48, 64, 96, 192]
            .into_iter()
            .find(|r| {
                let step = ROWS_PER_MEASURE / r;
                in_measure.iter().all(|(p, _)| (p - start).is_multiple_of(step))
            })
            .unwrap_or(192);
        let step = ROWS_PER_MEASURE / res;
        let mut lines = vec![String::from("0000"); res as usize];
        for (p, cells) in &in_measure {
            lines[((p - start) / step) as usize] = cells.iter().map(|c| c.to_char()).collect();
        }
        if m > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!("  // measure {}\n", m + 1));
        for l in lines {
            out.push_str(&l);
            out.push('\n');
        }
    }
    out.push_str(";\n");
    out
}

/// Renders a complete `.sm` file.
pub fn render_sm(info: &SongInfo, charts: &[OutChart]) -> String {
    let mut s = String::new();
    let tag = |s: &mut String, k: &str, v: &str| s.push_str(&format!("#{k}:{v};\n"));
    tag(&mut s, "TITLE", &sanitize(&info.title));
    tag(&mut s, "SUBTITLE", "");
    tag(&mut s, "ARTIST", &sanitize(&info.artist));
    tag(&mut s, "TITLETRANSLIT", "");
    tag(&mut s, "SUBTITLETRANSLIT", "");
    tag(&mut s, "ARTISTTRANSLIT", "");
    tag(&mut s, "GENRE", "");
    tag(&mut s, "CREDIT", &sanitize(&info.credit));
    let v = &info.visuals;
    tag(&mut s, "BANNER", &sanitize(&v.banner));
    tag(&mut s, "BACKGROUND", &sanitize(&v.background));
    // Only written when set, so that files without visuals stay as before.
    if !v.jacket.is_empty() {
        tag(&mut s, "JACKET", &sanitize(&v.jacket));
    }
    tag(&mut s, "LYRICSPATH", "");
    tag(&mut s, "CDTITLE", "");
    tag(&mut s, "MUSIC", &sanitize(&info.music));
    tag(&mut s, "OFFSET", &fmt3(info.offset));
    tag(&mut s, "SAMPLESTART", &fmt3(info.sample_start));
    tag(&mut s, "SAMPLELENGTH", &fmt3(info.sample_length));
    tag(&mut s, "SELECTABLE", "YES");
    tag(&mut s, "BPMS", &format!("0.000={}", fmt3(info.bpm)));
    tag(&mut s, "STOPS", "");
    let movie = if v.bg_video.is_empty() {
        String::new()
    } else {
        bg_changes(&sanitize(&v.bg_video), movie_start_beat(info.bpm, info.offset))
    };
    tag(&mut s, "BGCHANGES", &movie);
    tag(&mut s, "KEYSOUNDS", "");
    for c in charts {
        s.push_str(&format!(
            "\n//---------------dance-single - {}----------------\n#NOTES:\n     dance-single:\n     {}:\n     {}:\n     {}:\n     0.000,0.000,0.000,0.000,0.000:\n",
            sanitize(&c.description),
            sanitize(&c.description),
            c.difficulty,
            c.meter
        ));
        s.push_str(&render_notes(&c.rows));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tap(c: usize) -> [Cell; 4] {
        let mut r = [Cell::Empty; 4];
        r[c] = Cell::Tap;
        r
    }

    #[test]
    fn visuals_are_rendered_and_movie_starts_at_audio_zero() {
        let info = SongInfo {
            title: "t".into(),
            music: "t.ogg".into(),
            bpm: 120.0,
            offset: -0.25,
            visuals: Visuals {
                banner: "bn.png".into(),
                background: "bg.png".into(),
                jacket: "jacket.png".into(),
                bg_video: "t-bg.mp4".into(),
            },
            ..SongInfo::default()
        };
        let chart = OutChart {
            difficulty: "Easy".into(),
            meter: 1,
            description: "d".into(),
            rows: vec![(0, tap(0))],
        };
        let text = render_sm(&info, std::slice::from_ref(&chart));
        let sim = Simfile::parse(&text, false).unwrap();
        assert_eq!(sim.tag("BANNER"), Some("bn.png"));
        assert_eq!(sim.tag("BACKGROUND"), Some("bg.png"));
        assert_eq!(sim.tag("JACKET"), Some("jacket.png"));
        // offset -0.25 s at 120 BPM: audio time 0 is beat -0.5.
        assert_eq!(
            sim.tag("BGCHANGES"),
            Some("-0.500=t-bg.mp4=1.000=0=0=0=StretchNoLoop====")
        );

        // Adding the visuals afterwards to a plain file gives the same tags.
        let plain_info = SongInfo {
            visuals: Visuals::default(),
            ..info.clone()
        };
        let plain = render_sm(&plain_info, &[chart]);
        assert!(!plain.contains("#JACKET"), "no JACKET tag without a jacket");
        let decorated = apply_visuals(&plain, &info.visuals).unwrap();
        let d = Simfile::parse(&decorated, false).unwrap();
        for key in ["BANNER", "BACKGROUND", "JACKET", "BGCHANGES"] {
            assert_eq!(d.tag(key), sim.tag(key), "{key}");
        }
        assert_eq!(d.charts[0].notes, sim.charts[0].notes);
        assert!(decorated.find("#JACKET").unwrap() < decorated.find("#NOTES").unwrap());
    }

    #[test]
    fn set_tag_replaces_or_inserts() {
        let sm = "#TITLE:a;\n#BANNER:;\n\n//--- chart\n#NOTES:x;\n";
        assert_eq!(
            set_tag(sm, "BANNER", "b.png"),
            "#TITLE:a;\n#BANNER:b.png;\n\n//--- chart\n#NOTES:x;\n"
        );
        assert_eq!(
            set_tag(sm, "JACKET", "j.png"),
            "#TITLE:a;\n#BANNER:;\n#JACKET:j.png;\n\n//--- chart\n#NOTES:x;\n"
        );
        assert_eq!(
            set_tag_after(sm, "JACKET", "j.png", Some("TITLE")),
            "#TITLE:a;\n#JACKET:j.png;\n#BANNER:;\n\n//--- chart\n#NOTES:x;\n"
        );
    }

    #[test]
    fn roundtrip_rows_and_resolution() {
        let rows = vec![(0, tap(0)), (24, tap(1)), (48, tap(2)), (192 + 16, tap(3))];
        let chart = OutChart {
            difficulty: "Easy".into(),
            meter: 3,
            description: "test".into(),
            rows: rows.clone(),
        };
        let info = SongInfo {
            title: "A: b;c".into(),
            artist: "X".into(),
            music: "a.mp3".into(),
            credit: "c".into(),
            bpm: 128.0,
            offset: -0.25,
            sample_start: 10.0,
            sample_length: 12.0,
            ..SongInfo::default()
        };
        let text = render_sm(&info, &[chart]);
        assert!(text.contains("#TITLE:A- bc;"));
        assert!(text.contains("#OFFSET:-0.250;"));
        let sim = Simfile::parse(&text, false).unwrap();
        assert_eq!(sim.charts.len(), 1);
        let c = &sim.charts[0];
        assert_eq!(c.difficulty, "Easy");
        assert_eq!(c.meter, 3);
        let parsed = c.rows().unwrap();
        let beats: Vec<f64> = parsed.iter().map(|r| r.beat).collect();
        assert_eq!(beats, vec![0.0, 0.5, 1.0, 4.0 + 1.0 / 3.0]);
        // measure 1 needs 8ths → 8 lines; measure 2 needs 12ths → 12 lines
        let measures: Vec<&str> = c.notes.split(',').collect();
        assert_eq!(measures[0].lines().filter(|l| l.trim().len() == 4).count(), 8);
        assert_eq!(measures[1].lines().filter(|l| l.trim().len() == 4).count(), 12);
    }

    #[test]
    fn timing_with_changes_and_stops() {
        let t = Timing::from_tags("0.1", "0=120,4=60", "2=0.5", "", "").unwrap();
        assert!((t.seconds(0.0) - -0.1).abs() < 1e-9);
        assert!((t.seconds(2.0) - 0.9).abs() < 1e-9); // stop on beat 2 not applied yet
        assert!((t.seconds(3.0) - 1.9).abs() < 1e-9);
        assert!((t.seconds(5.0) - 3.4).abs() < 1e-9);
    }

    #[test]
    fn generated_simfiles_are_recognized() {
        let generated = "#TITLE:x;\n#CREDIT:itg-charter 0.1.0 (seed 0);\n#BPMS:0=120;\n";
        assert!(Simfile::parse(generated, false).unwrap().is_generated());
        let human = "#TITLE:x;\n#CREDIT:J. Frederick;\n#BPMS:0=120;\n";
        assert!(!Simfile::parse(human, false).unwrap().is_generated());
    }

    #[test]
    fn parse_ssc_charts() {
        let text = "#TITLE:x;\n#BPMS:0=150;\n#OFFSET:0;\n#NOTEDATA:;\n#STEPSTYPE:dance-single;\n#DIFFICULTY:Hard;\n#METER:9;\n#BPMS:0=175;\n#NOTES:\n1000\n0100\n0010\n0001\n;\n";
        let sim = Simfile::parse(text, true).unwrap();
        assert_eq!(sim.charts.len(), 1);
        let c = &sim.charts[0];
        assert_eq!((c.difficulty.as_str(), c.meter), ("Hard", 9));
        assert_eq!(sim.timing(c).unwrap().bpms, vec![(0.0, 175.0)]);
        assert_eq!(c.rows().unwrap().len(), 4);
    }
}
