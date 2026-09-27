//! Prints the decoded duration and sample rate of an audio file (decoder sanity check).
fn main() -> anyhow::Result<()> {
    for p in std::env::args().skip(1) {
        let a = itg_charter::audio::decode_file(std::path::Path::new(&p))?;
        println!(
            "{p}: {} Hz, {} samples, {:.3} s",
            a.sample_rate,
            a.samples.len(),
            a.duration()
        );
    }
    Ok(())
}
