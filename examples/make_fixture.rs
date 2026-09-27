//! Writes the synthetic test song as WAV (convert it to the mp3 fixture with ffmpeg).
//! Usage: cargo run --example make_fixture -- OUT.wav
fn main() -> anyhow::Result<()> {
    let out = std::env::args().nth(1).expect("usage: make_fixture OUT.wav");
    let s = itg_charter::synth::test_song(128.0, 0.35, 24.0, 44100);
    std::fs::write(out, itg_charter::synth::wav_bytes(&s, 44100))?;
    Ok(())
}
