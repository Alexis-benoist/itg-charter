//! Source separation with Demucs (drums / bass / vocals / other).
//!
//! Demucs runs as a Python subprocess from a dedicated virtualenv. Results are
//! cached by the SHA-256 of the audio file, so a song is separated once and every
//! later run (any seed, any difficulty) reads the same stems: this is what keeps
//! the output byte-for-byte reproducible even though GPU inference is not
//! guaranteed to be bit-exact between runs.

use crate::analysis::Stems;
use crate::audio::decode_file;
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const STEM_NAMES: [&str; 4] = ["drums", "bass", "vocals", "other"];
const MODEL: &str = "htdemucs";

#[derive(Clone, Debug, Default)]
pub struct StemOptions {
    /// Python interpreter with `demucs` installed. Default: `$ITG_CHARTER_PYTHON`, then
    /// `~/.local/share/itg-charter/demucs-venv/bin/python` (Linux/macOS) or
    /// `%LOCALAPPDATA%\itg-charter\demucs-venv\Scripts\python.exe` (Windows).
    pub python: Option<PathBuf>,
    /// Torch device ("cuda" or "cpu"); Demucs picks one when unset.
    pub device: Option<String>,
    /// Stem cache directory. Default: `$XDG_CACHE_HOME/itg-charter/stems` or `~/.cache/...`.
    pub cache_dir: Option<PathBuf>,
}

/// Home directory: `$HOME`, then `%USERPROFILE%` (Windows usually has no `HOME`), else ".".
fn home() -> PathBuf {
    home_from(std::env::var_os("HOME"), std::env::var_os("USERPROFILE"))
}

fn home_from(home: Option<OsString>, userprofile: Option<OsString>) -> PathBuf {
    home.or(userprofile)
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
}

/// Default Demucs interpreter: the venv's `bin/python` under `~/.local/share` on Unix,
/// `Scripts\python.exe` under `%LOCALAPPDATA%` (or `<home>\AppData\Local`) on Windows.
fn default_python(windows: bool, home: &Path, local_app_data: Option<OsString>) -> PathBuf {
    if windows {
        local_app_data
            .map_or_else(|| home.join("AppData").join("Local"), PathBuf::from)
            .join("itg-charter")
            .join("demucs-venv")
            .join("Scripts")
            .join("python.exe")
    } else {
        home.join(".local/share/itg-charter/demucs-venv/bin/python")
    }
}

impl StemOptions {
    pub fn python(&self) -> PathBuf {
        self.python
            .clone()
            .or_else(|| std::env::var_os("ITG_CHARTER_PYTHON").map(PathBuf::from))
            .unwrap_or_else(|| default_python(cfg!(windows), &home(), std::env::var_os("LOCALAPPDATA")))
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir.clone().unwrap_or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| home().join(".cache"))
                .join("itg-charter/stems")
        })
    }
}

/// SHA-256 of a file, hex encoded.
pub fn file_hash(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Cache folder for the stems of `audio`.
pub fn cache_path(audio: &Path, opts: &StemOptions) -> Result<PathBuf> {
    Ok(opts.cache_dir().join(format!("{}-{MODEL}", file_hash(audio)?)))
}

/// Loads cached stems from `dir` (all four wav files must exist).
pub fn load_cached(dir: &Path) -> Result<Stems> {
    let mut tracks = Vec::new();
    let mut sample_rate = 0;
    for name in STEM_NAMES {
        let a = decode_file(&dir.join(format!("{name}.wav")))?;
        sample_rate = a.sample_rate;
        tracks.push(a.samples);
    }
    let other = tracks.pop().unwrap();
    let vocals = tracks.pop().unwrap();
    let bass = tracks.pop().unwrap();
    let drums = tracks.pop().unwrap();
    Ok(Stems {
        sample_rate,
        drums,
        bass,
        vocals,
        other,
    })
}

/// Returns the stems of `audio`, running Demucs if they are not cached yet.
pub fn separate(audio: &Path, opts: &StemOptions) -> Result<Stems> {
    let dir = cache_path(audio, opts)?;
    if STEM_NAMES.iter().all(|n| dir.join(format!("{n}.wav")).exists()) {
        return load_cached(&dir);
    }
    let python = opts.python();
    if !python.exists() {
        bail!(
            "Demucs python not found at {} (see the README's Demucs section to install it, set ITG_CHARTER_PYTHON, or use --no-stems)",
            python.display()
        );
    }
    let tmp = dir.with_extension("partial");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    let mut cmd = Command::new(&python);
    cmd.args(["-m", "demucs", "-n", MODEL])
        // shifts > 0 applies a *random* time shift: must stay 0 for reproducibility.
        .args(["--shifts", "0"])
        .args(["--filename", "{stem}.{ext}"])
        .arg("-o")
        .arg(&tmp);
    if let Some(d) = &opts.device {
        cmd.args(["-d", d]);
    }
    cmd.arg(audio);
    let out = cmd.output().context("running demucs")?;
    if !out.status.success() {
        bail!("demucs failed:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    let produced = tmp.join(MODEL);
    std::fs::create_dir_all(&dir)?;
    for n in STEM_NAMES {
        let f = format!("{n}.wav");
        std::fs::rename(produced.join(&f), dir.join(&f))
            .with_context(|| format!("demucs did not produce {f}"))?;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    load_cached(&dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_falls_back_to_userprofile_then_dot() {
        let h = |a: Option<&str>, b: Option<&str>| home_from(a.map(Into::into), b.map(Into::into));
        assert_eq!(h(Some("/home/u"), Some("C:/Users/u")), PathBuf::from("/home/u"));
        assert_eq!(h(None, Some("C:/Users/u")), PathBuf::from("C:/Users/u"));
        assert_eq!(h(None, None), PathBuf::from("."));
    }

    #[test]
    fn default_python_per_os() {
        let home = Path::new("/home/u");
        assert_eq!(
            default_python(false, home, Some("/ignored".into())),
            PathBuf::from("/home/u/.local/share/itg-charter/demucs-venv/bin/python")
        );
        let venv = |base: &Path| {
            base.join("itg-charter")
                .join("demucs-venv")
                .join("Scripts")
                .join("python.exe")
        };
        assert_eq!(
            default_python(true, home, Some("/lad".into())),
            venv(Path::new("/lad"))
        );
        assert_eq!(
            default_python(true, home, None),
            venv(&home.join("AppData").join("Local"))
        );
    }
}
