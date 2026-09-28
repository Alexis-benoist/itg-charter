# Installing itg-charter

## Prebuilt binaries

Every push to `main` publishes a release with binaries for Linux x86_64, macOS (Apple silicon and
Intel) and Windows x86_64: download the archive for your system from the
[Releases](https://github.com/Alexis-benoist/itg-charter/releases) page, check it against its
`.sha256` file, extract it and put `itg-charter` (or `itg-charter.exe`) in your `PATH`.

## From source

Requirements: [Rust](https://rustup.rs) (stable) and a C compiler (aubio's C sources are compiled
into the binary: `build-essential` on Debian/Ubuntu, Xcode command-line tools on macOS, the Visual
Studio C++ build tools on Windows).

```sh
git clone https://github.com/Alexis-benoist/itg-charter
cd itg-charter
cargo build --release          # binary: target/release/itg-charter
```

## Demucs (optional, recommended)

Charts are better when drums, bass and vocals are separated with
[Demucs](https://github.com/facebookresearch/demucs). itg-charter runs it from a Python 3.11
virtual environment, by default `~/.local/share/itg-charter/demucs-venv`:

```sh
uv venv --python 3.11 ~/.local/share/itg-charter/demucs-venv
VIRTUAL_ENV=~/.local/share/itg-charter/demucs-venv uv pip install demucs torch torchaudio soundfile
```

On Windows the default is `%LOCALAPPDATA%\itg-charter\demucs-venv` (interpreter
`Scripts\python.exe`):

```powershell
uv venv --python 3.11 $env:LOCALAPPDATA\itg-charter\demucs-venv
$env:VIRTUAL_ENV="$env:LOCALAPPDATA\itg-charter\demucs-venv"; uv pip install demucs torch torchaudio soundfile
```

(`uv`: https://docs.astral.sh/uv/. Another interpreter can be given with
`ITG_CHARTER_PYTHON=/path/to/python`.) With an NVIDIA GPU a song takes about 20 s; `--device cpu`
works without one, slower. Separated tracks are cached in `~/.cache/itg-charter/stems/`
(`%USERPROFILE%\.cache\...` on Windows, or `$XDG_CACHE_HOME/itg-charter/stems/`). Without
Demucs, itg-charter prints a warning and charts the full mix; `--no-stems` skips it silently.
