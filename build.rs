//! Packs the embedded arrow models into a compact binary form (3.6 MB of JSON -> 0.34 MB).
//!
//! The JSON files in `model/` stay the source (readable, diffable); each is written to
//! `OUT_DIR/<name>.bin` as the deflate of:
//! - varint `ngram.len()`, then every `ngram` count as a varint (mostly zeros: 1 byte);
//! - varint `ngram4.len()`, then per entry the varint index delta from the previous
//!   entry and the varint count;
//! - the other fields of the model as JSON, to the end.
//!
//! `Model::from_packed` (src/model.rs) reads it back; a test checks that every packed
//! model equals its JSON file.

use serde_json::Value;

const MODELS: [(&str, &str); 3] = [
    ("model/model.json", "model.bin"),
    ("model/styles/stream.json", "stream.bin"),
    ("model/styles/tech.json", "tech.bin"),
];

fn varint(mut n: u64, out: &mut Vec<u8>) {
    loop {
        let b = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn number(v: &Value) -> u64 {
    v.as_u64().expect("non-negative integer count")
}

fn pack(json: &str) -> Vec<u8> {
    let mut model: Value = serde_json::from_str(json).expect("model JSON");
    let fields = model.as_object_mut().expect("model object");
    let ngram = fields.remove("ngram").expect("ngram");
    let ngram4 = fields.remove("ngram4").expect("ngram4");
    let mut out = Vec::new();
    let ngram = ngram.as_array().expect("ngram array");
    varint(ngram.len() as u64, &mut out);
    for c in ngram {
        varint(number(c), &mut out);
    }
    let ngram4 = ngram4.as_array().expect("ngram4 array");
    varint(ngram4.len() as u64, &mut out);
    let mut prev = 0;
    for e in ngram4 {
        let (i, c) = (number(&e[0]), number(&e[1]));
        assert!(i >= prev, "ngram4 not sorted by index");
        varint(i - prev, &mut out);
        varint(c, &mut out);
        prev = i;
    }
    out.extend(serde_json::to_vec(&model).expect("model fields"));
    miniz_oxide::deflate::compress_to_vec(&out, 10)
}

fn main() {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    for (src, name) in MODELS {
        println!("cargo::rerun-if-changed={src}");
        let json = std::fs::read_to_string(src).unwrap_or_else(|e| panic!("{src}: {e}"));
        std::fs::write(out_dir.join(name), pack(&json)).expect("writing packed model");
    }
}
