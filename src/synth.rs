//! Synthetic test signals (used by tests and to build the test fixture).

/// Synthetic drum loop: kick on beats, hat on off-beats, snare on 2 and 4.
pub fn drum_loop(bpm: f64, first_beat: f64, seconds: f64, sr: u32) -> Vec<f32> {
    let n = (seconds * sr as f64) as usize;
    let mut out = vec![0f32; n];
    let period = 60.0 / bpm;
    let mut noise = 12345u32;
    let mut rnd = move || {
        noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
        (noise >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
    };
    let mut k = 0;
    loop {
        let t0 = first_beat + k as f64 * period / 2.0;
        if t0 >= seconds {
            break;
        }
        let start = (t0 * sr as f64) as usize;
        let on_beat = k % 2 == 0;
        let snare = k % 4 == 2;
        for i in 0..(0.15 * sr as f64) as usize {
            if start + i >= n {
                break;
            }
            let t = i as f32 / sr as f32;
            let v = if on_beat {
                // Accent the first beat of each bar so the downbeat is identifiable.
                let accent = if (k / 2) % 4 == 0 { 1.0 } else { 0.6 };
                let kick = accent * (2.0 * std::f32::consts::PI * 55.0 * t).sin() * (-t * 25.0).exp();
                let sn = if snare {
                    rnd() * 0.5 * (-t * 30.0).exp()
                } else {
                    0.0
                };
                kick * 0.9 + sn
            } else {
                rnd() * 0.25 * (-t * 60.0).exp()
            };
            out[start + i] += v;
        }
        k += 1;
    }
    out
}

/// A small "song": the drum loop plus a melody of sustained notes, so that charts
/// get holds and varied rhythms. Deterministic.
pub fn test_song(bpm: f64, first_beat: f64, seconds: f64, sr: u32) -> Vec<f32> {
    let mut out = drum_loop(bpm, first_beat, seconds, sr);
    let period = 60.0 / bpm;
    // (start in beats within a 4-bar phrase, length in beats, frequency)
    let phrase = [
        (0.0, 1.5, 440.0),
        (1.5, 0.5, 494.0),
        (2.0, 2.0, 523.0),
        (4.0, 0.5, 587.0),
        (4.5, 0.5, 523.0),
        (5.0, 1.0, 494.0),
        (6.0, 2.0, 440.0),
        (8.0, 0.75, 392.0),
        (8.75, 0.75, 440.0),
        (9.5, 2.5, 494.0),
        (12.0, 4.0, 330.0),
    ];
    let mut bar0 = 0.0;
    while first_beat + bar0 * period < seconds {
        for &(start, len, freq) in &phrase {
            let t0 = first_beat + (bar0 + start) * period;
            let n0 = (t0 * sr as f64) as usize;
            let n = (len * period * sr as f64) as usize;
            for i in 0..n {
                let k = n0 + i;
                if k >= out.len() {
                    break;
                }
                let t = i as f32 / sr as f32;
                let env = (t * 200.0).min(1.0) * (1.0 - i as f32 / n as f32).powf(0.3);
                out[k] += 0.25 * env * (2.0 * std::f32::consts::PI * freq as f32 * t).sin();
            }
        }
        bar0 += 16.0;
    }
    out
}

/// 16-bit mono WAV encoding (for fixtures).
pub fn wav_bytes(samples: &[f32], sr: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = (samples.len() * 2) as u32;
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&sr.to_le_bytes());
    b.extend_from_slice(&(sr * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    b
}
