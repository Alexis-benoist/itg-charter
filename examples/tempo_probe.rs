//! Prints comb scores of candidate BPMs for one audio file (tempo debugging).
//! Usage: tempo_probe AUDIO BPM1 BPM2 ...
use itg_charter::analysis::{Layer, best_phase};
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = itg_charter::audio::decode_file(std::path::Path::new(&args[0]))?;
    let layer = Layer::analyze(&a.samples, a.sample_rate);
    let env = &layer.envelope;
    let coarse = env.dilated(2);
    for b in &args[1..] {
        let bpm: f64 = b.parse()?;
        let (p, s) = best_phase(env, 60.0 / bpm, 0.001, (env.mean(), env.std()));
        let (_, cs) = best_phase(&coarse, 60.0 / bpm, 0.004, (coarse.mean(), coarse.std()));
        println!("bpm {bpm:>8.3}: fine {s:.3} (phase {p:.3}) coarse {cs:.3}");
    }
    Ok(())
}
