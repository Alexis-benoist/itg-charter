# itg-charter

Generates **ITGmania / In The Groove** charts (`.sm`, dance-single) from an audio file (mp3, ogg,
flac, wav), at the difficulty levels you choose, reproducibly with a seed.

```sh
itg-charter gen song.mp3 -o ~/.itgmania/Songs/itg-charter
```

This creates `~/.itgmania/Songs/itg-charter/<Title>/` with the `.sm` and a copy of the audio;
restart ITGmania (or reload the songs) to see it.

To make songs straight from YouTube videos (download, encoding, banner, background video…), see
the companion tool **[itg-yt](https://github.com/Alexis-benoist/itg-yt)**, which uses itg-charter.

## Installation

### Prebuilt binaries

Every push to `main` publishes a release with binaries for Linux x86_64, macOS (Apple silicon and
Intel) and Windows x86_64: download the archive for your system from the
[Releases](https://github.com/Alexis-benoist/itg-charter/releases) page, check it against its
`.sha256` file, extract it and put `itg-charter` (or `itg-charter.exe`) in your `PATH`.

### From source

Requirements: [Rust](https://rustup.rs) (stable) and a C compiler (aubio's C sources are compiled
into the binary: `build-essential` on Debian/Ubuntu, Xcode command-line tools on macOS, the Visual
Studio C++ build tools on Windows).

```sh
git clone https://github.com/Alexis-benoist/itg-charter
cd itg-charter
cargo build --release          # binary: target/release/itg-charter
```

### Demucs (optional, recommended)

Charts are better when drums, bass and vocals are separated with
[Demucs](https://github.com/facebookresearch/demucs). itg-charter runs it from a Python 3.11
virtual environment, by default `~/.local/share/itg-charter/demucs-venv`:

```sh
uv venv --python 3.11 ~/.local/share/itg-charter/demucs-venv
VIRTUAL_ENV=~/.local/share/itg-charter/demucs-venv uv pip install demucs torch torchaudio soundfile
```

(`uv`: https://docs.astral.sh/uv/. Another interpreter can be given with
`ITG_CHARTER_PYTHON=/path/to/python`.) With an NVIDIA GPU a song takes about 20 s; `--device cpu`
works without one, slower. Separated tracks are cached in `~/.cache/itg-charter/stems/`. Without
Demucs, itg-charter prints a warning and charts the full mix; `--no-stems` skips it silently.

## Usage

```sh
itg-charter gen song.ogg                     # default profile "full": meters 2, 4, 6, 8, 10
itg-charter gen song.ogg -p beginner         # meters 2, 3, 4, 5
itg-charter gen song.ogg -m 1,3-5            # any meters on the 1-10 scale (at most 5)
itg-charter gen song.ogg -d easy,hard        # difficulty slots with the density of human charts
itg-charter gen song.ogg -s 42               # another seed: other arrows, same sync
itg-charter analyze song.ogg                 # detected BPM and offset only
```

Other options of `gen`: `-o DIR` (output folder), `--bpm` / `--offset` (force the sync),
`--title`, `--artist`, `--no-stems`, `--device cpu`, `--model FILE`, and the visuals `--banner`,
`--background`, `--jacket`, `--bg-video` (copied into the song folder and declared in the `.sm`).
`itg-charter decorate SONG.sm --bg-video clip.mp4 …` adds visuals to an existing song.
`itg-charter <command> --help` lists everything.

The same audio, options and seed always give byte-identical `.sm` files.

## How it works

1. **Source separation** with Demucs (drums, bass, vocals, other; optional).
2. **Analysis** with [aubio](https://aubio.org): onsets and tempo, then fitting a constant BPM and
   the offset over the whole song.
3. **Note placement** on the grid (quarter notes → sixteenths), with the density needed for the
   requested meter and the jumps and holds of human charts of that level.
4. **Arrow choice**: a beam search combining a pattern model learned from human charts with
   **ITGmania's parity computation** (foot comfort: crossovers, footswitches, double-steps…), with
   noise controlled by the seed.

The pattern model (`model/model.json`, embedded in the binary) was trained with
`itg-charter train <Songs folder>` on a library of ~1,600 community simfiles (~6,800 dance-single
charts). You can retrain it on your own songs and pass the result with `--model`.

## Measured accuracy

On human-synced simfiles never used for tuning (test half of the library, 400 songs): exact BPM
89 %, and when the BPM is right 97.5 % of songs are synced within 30 ms (median error 5 ms). The
port of ITGmania's parity gives the game's own tech counts on 97 % of charts. Generated charts
have statistics (density, jumps, holds, crossovers, footswitches, meter) within the range of human
charts of the same difficulty. Details and evaluation tools: [`eval/`](eval/README.md) and the
`examples/` folder.

## Development

`cargo test`, `cargo fmt --all --check` and `cargo clippy --all-targets -- -D warnings` must pass
(CI). The evaluation examples need a folder of ITGmania songs, e.g.
`cargo run --release --example eval_sync -- /path/to/ITGmania/Songs`. Project conventions (in
French): [`CLAUDE.md`](CLAUDE.md).

## License

GPL-3.0 (aubio and the parity code ported from ITGmania are GPL-3.0).
