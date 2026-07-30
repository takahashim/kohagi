//! Inspect, validate and edit a CoreML `weight.bin`.
//!
//!     milblob dump      <weight.bin|bundle>            list the blobs
//!     milblob verify    <weight.bin|bundle>            check the structure, exit 1 on a problem
//!     milblob roundtrip <weight.bin|bundle> [out.bin]  re-emit and compare bytes
//!     milblob diff      <a> <b>                        compare two blob files
//!     milblob negate    <weight.bin> <meta-offset>      negate an fp16 blob in place
//!
//! A bundle path (`.mlmodelc` or `.mlpackage`) is resolved to the `weight.bin`
//! inside it.

use std::path::{Path, PathBuf};
use std::process::exit;

use coreml_jigs::blob::{self, Blob};

fn resolve(arg: &str) -> PathBuf {
    let p = Path::new(arg);
    if p.is_dir() {
        if let Some(w) = blob::weight_path(p) {
            return w;
        }
        eprintln!("milblob: {arg} holds no weights/weight.bin");
        exit(2);
    }
    p.to_path_buf()
}

fn read(arg: &str) -> (PathBuf, Vec<u8>, Vec<Blob>, u32) {
    let path = resolve(arg);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("milblob: reading {}: {e}", path.display());
            exit(2);
        }
    };
    match blob::parse(&bytes) {
        Ok((count, blobs)) => (path, bytes, blobs, count),
        Err(problem) => {
            eprintln!("milblob: {} is not a valid MIL blob file", path.display());
            eprintln!("  {problem}");
            exit(1);
        }
    }
}

fn dump(arg: &str) {
    let (path, bytes, blobs, count) = read(arg);
    let data: u64 = blobs.iter().map(|b| b.size_in_bytes).sum();
    let end = blobs
        .last()
        .map_or(blob::HEADER_SIZE, |b| b.data_offset + b.size_in_bytes);
    println!("file    : {} ({} bytes)", path.display(), bytes.len());
    println!("header  : count={count} version={}", blob::VERSION);
    println!(
        "blobs   : {} entries, {data} bytes of data, {} bytes of overhead",
        blobs.len(),
        bytes.len() as u64 - data
    );
    println!("tail    : last data ends at {end}");

    let mut by_dtype: std::collections::BTreeMap<u32, (usize, u64)> = Default::default();
    for b in &blobs {
        let e = by_dtype.entry(b.dtype).or_default();
        e.0 += 1;
        e.1 += b.size_in_bytes;
    }
    for (d, (n, bytes)) in by_dtype {
        println!("dtype   : {} x{n} ({bytes} bytes)", blob::dtype_name(d));
    }

    println!(
        "\n{:>4}  {:>12}  {:>9}  {:>12}  {:>12}",
        "idx", "meta_offset", "dtype", "size_bytes", "elements"
    );
    for (i, b) in blobs.iter().enumerate() {
        let elems = b
            .elements()
            .map_or_else(|| "-".to_string(), |n| n.to_string());
        println!(
            "{i:>4}  {:>12}  {:>9}  {:>12}  {elems:>12}",
            b.meta_offset,
            blob::dtype_name(b.dtype),
            b.size_in_bytes
        );
    }
}

/// Structural validation. The interesting failures are silent ones: a blob file
/// whose chain is intact but whose dtype or padding is wrong loads and produces
/// wrong numbers, so every check reports rather than stopping at the first.
fn verify(arg: &str) {
    let (path, bytes, blobs, count) = read(arg);
    let mut problems = blob::lint(&bytes, &blobs);
    if count as usize != blobs.len() {
        // parse() walks exactly `count` records, so this cannot currently trip;
        // keep it so a future reader change cannot disagree with the header
        // silently.
        eprintln!(
            "milblob: header declares {count} blobs but {} were walked",
            blobs.len()
        );
        exit(1);
    }
    if problems.is_empty() {
        println!(
            "OK: {} — {} blobs, {} bytes, chain and alignment intact",
            path.display(),
            blobs.len(),
            bytes.len()
        );
        return;
    }
    eprintln!(
        "milblob: {} has {} problems",
        path.display(),
        problems.len()
    );
    problems.sort_by_key(|p| format!("{p}"));
    for p in &problems {
        eprintln!("  {p}");
    }
    exit(1);
}

fn roundtrip(arg: &str, out: Option<&str>) {
    let (path, bytes, blobs, _) = read(arg);
    let rewritten = blob::write(&blobs, &bytes);
    if rewritten.len() != bytes.len() {
        eprintln!(
            "MISMATCH: re-emitted {} bytes, original is {}",
            rewritten.len(),
            bytes.len()
        );
        exit(1);
    }
    match rewritten.iter().zip(&bytes).position(|(a, b)| a != b) {
        Some(at) => {
            eprintln!("MISMATCH: first differing byte at {at}");
            exit(1);
        }
        None => println!(
            "OK: {} — {} bytes, byte-identical ({} blobs)",
            path.display(),
            bytes.len(),
            blobs.len()
        ),
    }
    if let Some(out) = out {
        std::fs::write(out, &rewritten).expect("write output");
        println!("wrote {out}");
    }
}

/// Compare two blob files record by record. Used to tell "the weights changed"
/// from "the layout changed", which a plain `cmp` cannot.
fn diff(a: &str, b: &str) {
    let (pa, ba, la, _) = read(a);
    let (pb, bb, lb, _) = read(b);
    println!(
        "a: {} ({} bytes, {} blobs)",
        pa.display(),
        ba.len(),
        la.len()
    );
    println!(
        "b: {} ({} bytes, {} blobs)",
        pb.display(),
        bb.len(),
        lb.len()
    );

    if la.len() != lb.len() {
        println!("\nblob counts differ; not comparing records");
        exit(1);
    }
    let mut layout = 0;
    let mut content = 0;
    for (i, (x, y)) in la.iter().zip(&lb).enumerate() {
        if (x.meta_offset, x.dtype, x.size_in_bytes, x.padding_bits)
            != (y.meta_offset, y.dtype, y.size_in_bytes, y.padding_bits)
        {
            println!(
                "{i:>4}: layout differs — a {} {}B @{}, b {} {}B @{}",
                blob::dtype_name(x.dtype),
                x.size_in_bytes,
                x.meta_offset,
                blob::dtype_name(y.dtype),
                y.size_in_bytes,
                y.meta_offset
            );
            layout += 1;
            continue;
        }
        let (sa, sb) = (x.data_offset as usize, y.data_offset as usize);
        let n = x.size_in_bytes as usize;
        if ba[sa..sa + n] != bb[sb..sb + n] {
            let differing = ba[sa..sa + n]
                .iter()
                .zip(&bb[sb..sb + n])
                .filter(|(p, q)| p != q)
                .count();
            println!(
                "{i:>4}: data differs — {differing}/{n} bytes ({} @{})",
                blob::dtype_name(x.dtype),
                x.meta_offset
            );
            content += 1;
        }
    }
    if layout == 0 && content == 0 {
        println!("\nidentical");
    } else {
        println!("\n{layout} blobs differ in layout, {content} in data");
        exit(1);
    }
}

/// Negate one fp16 blob in place. Not a general editor: this exists because
/// negating a `layer_norm` gamma changes a model's output in a way that can be
/// predicted exactly, which makes it a
/// usable end-to-end check on writing a blob CoreML will load.
fn negate(arg: &str, target: u64) {
    let (path, bytes, blobs, _) = read(arg);
    let Some(b) = blobs.iter().find(|b| b.meta_offset == target) else {
        eprintln!("milblob: no blob at metadata offset {target}");
        eprintln!(
            "  offsets present: {}",
            blobs
                .iter()
                .map(|b| b.meta_offset.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        exit(2);
    };
    if b.dtype != 1 {
        eprintln!(
            "milblob: blob at {target} is {}, only fp16 is supported here",
            blob::dtype_name(b.dtype)
        );
        exit(2);
    }
    let mut out = bytes;
    let start = b.data_offset as usize;
    let n = (b.size_in_bytes / 2) as usize;
    for i in 0..n {
        let o = start + i * 2;
        let v = half::f16::from_bits(u16::from_le_bytes([out[o], out[o + 1]]));
        out[o..o + 2].copy_from_slice(&half::f16::from_f32(-v.to_f32()).to_bits().to_le_bytes());
    }
    std::fs::write(&path, &out).expect("write in place");
    println!("negated {n} fp16 values at {target} in {}", path.display());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: milblob <dump|verify|roundtrip|diff|negate> <path> [arg]";
    if args.len() < 3 {
        eprintln!("{usage}");
        exit(2);
    }
    match (args[1].as_str(), args.get(3)) {
        ("dump", _) => dump(&args[2]),
        ("verify", _) => verify(&args[2]),
        ("roundtrip", out) => roundtrip(&args[2], out.map(String::as_str)),
        ("diff", Some(b)) => diff(&args[2], b),
        ("negate", Some(off)) => match off.parse() {
            Ok(o) => negate(&args[2], o),
            Err(_) => {
                eprintln!("milblob: `{off}` is not a metadata offset");
                exit(2);
            }
        },
        ("diff" | "negate", None) => {
            eprintln!("milblob: `{}` needs a second argument\n{usage}", args[1]);
            exit(2);
        }
        (other, _) => {
            eprintln!("milblob: unknown subcommand `{other}`\n{usage}");
            exit(2);
        }
    }
}
