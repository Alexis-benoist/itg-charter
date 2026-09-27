# itg-charter

Generates **ITGmania / In The Groove** charts (`.sm`, dance-single) from an audio file (mp3, ogg,
flac, wav), for the chosen difficulties, reproducibly with a seed.

```sh
cargo build --release
./target/release/itg-charter gen song.mp3 -d easy,medium,hard -s 42 -o ~/.itgmania/Songs/itg-charter
```

The folder `~/.itgmania/Songs/itg-charter/<Title>/` contains the `.sm` and a copy of the audio;
restart ITGmania to see it.

## How it works

1. **Source separation** with [Demucs](https://github.com/facebookresearch/demucs)
   (drums, bass, vocals, other; optional, `--no-stems`).
2. **Analysis** with [aubio](https://aubio.org): onsets and tempo, then fitting a constant BPM and
   the offset over the whole song.
3. **Note placement** on the grid (quarter notes → sixteenths) with the densities, jumps and holds
   learned from real charts.
4. **Arrow choice**: a beam search combining a pattern model learned from the game's human charts
   with **ITGmania's parity computation** (foot comfort: crossovers, footswitches,
   double-steps…), with noise controlled by the seed.

Useful options: `--bpm`, `--offset` (force the sync), `--title`, `--artist`, `--device cpu`,
`--model` (another model, see `itg-charter train`).

## Measured accuracy

On the ITGmania library (charts synced by humans): exact BPM in 83 % of cases (98.6 % up to the
octave), 85 % of songs synced within 20 ms. Generated charts have statistics (density, jumps,
holds, crossovers, footswitches, meter) within the distribution of human charts of each
difficulty.

## License

GPL-3.0 (aubio and the parity code ported from ITGmania are GPL-3.0).
