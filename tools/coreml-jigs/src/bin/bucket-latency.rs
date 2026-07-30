//! Measure each bucket's forward-pass latency, with the discipline that makes
//! the numbers reproducible.
//!
//!     bucket-latency <dir|bundle> [--iters N] [--warmup N] [--rounds N]
//!
//! Bucket composition has to be decided per model, because ANE latency is not
//! monotonic in sequence length: on bekko-a25m, 96 tokens measured slower than
//! 128, and 192 slower than 256 on some runs.
//! Deciding that from a single timing is how you pick the wrong buckets, so
//! this jig fixes the three things that made the doc's numbers trustworthy:
//!
//! - **warm up generously.** Too few iterations and the error approaches 50%.
//! - **interleave.** Every round times every bucket in turn, so a machine that
//!   gets busy halfway through penalises all of them rather than the ones that
//!   happened to run late.
//! - **take the minimum.** The fastest observed pass is the one least polluted
//!   by everything else on the machine. The spread is reported next to it so a
//!   noisy run is visible rather than averaged into the answer.
//!
//! Load time is reported too: buckets are nearly free to add in a
//! multi-function bundle by size, but each one costs ~0.17s to load.

use std::path::PathBuf;
use std::process::exit;
use std::time::{Duration, Instant};

use coreml_jigs::bundle;

struct Measured {
    label: String,
    seq: usize,
    dim: usize,
    load: Duration,
    best: Duration,
    worst: Duration,
    median: Duration,
}

fn parse_flag(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|v| match v.parse() {
            Ok(n) => n,
            Err(_) => {
                eprintln!("bucket-latency: `{v}` is not a number for {name}");
                exit(2);
            }
        })
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: bucket-latency <dir|bundle> [--iters N] [--warmup N] [--rounds N]\n\
             \n\
             defaults: --iters 20 --warmup 30 --rounds 5\n\
             reports ms/text per bucket (minimum over rounds), and load time"
        );
        exit(2);
    }
    let root = PathBuf::from(&args[1]);
    let iters = parse_flag(&args, "--iters", 20);
    let warmup = parse_flag(&args, "--warmup", 30);
    let rounds = parse_flag(&args, "--rounds", 5);
    if iters == 0 || rounds == 0 {
        eprintln!("bucket-latency: --iters and --rounds must be at least 1");
        exit(2);
    }

    let targets = match bundle::targets(&root) {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => {
            eprintln!(
                "bucket-latency: no .mlpackage or .mlmodelc under {}",
                root.display()
            );
            exit(2);
        }
        Err(e) => {
            eprintln!("bucket-latency: {e}");
            exit(1);
        }
    };

    // A `.mlpackage` has to be compiled before it can be loaded; do that once per
    // package rather than once per function, and outside the timed section.
    let targets = match bundle::compile_once(&targets) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("bucket-latency: {e}");
            exit(1);
        }
    };

    // Load everything first, so the reported total is what a process actually
    // pays at startup for this bucket set.
    let total_load = Instant::now();
    let mut loaded = Vec::new();
    for target in &targets {
        let (model, load) = match bundle::load(target) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bucket-latency: {e}");
                exit(1);
            }
        };
        let Some((seq, dim)) = bundle::hidden_shape(&model) else {
            eprintln!(
                "bucket-latency: {} has no `hidden` output of shape [1, seq, dim]; \
                 run coreml-inspect on it",
                target.label()
            );
            exit(1);
        };
        loaded.push((target.clone(), model, load, seq, dim));
    }
    let total_load = total_load.elapsed();

    // One reusable input per bucket: allocation is not what we are timing.
    let mut prepared = Vec::new();
    for (target, model, load, seq, dim) in loaded {
        let ids = bundle::fake_ids(seq, 1000);
        let mask = vec![1i32; seq];
        match bundle::inputs(&ids, &mask) {
            Ok(inputs) => prepared.push((target, model, load, seq, dim, inputs)),
            Err(e) => {
                eprintln!("bucket-latency: {e}");
                exit(1);
            }
        }
    }

    eprintln!(
        "warming up {} buckets x {warmup} passes ...",
        prepared.len()
    );
    for (target, model, _, _, _, inputs) in &prepared {
        for _ in 0..warmup {
            if let Err(e) = bundle::predict(model, inputs) {
                eprintln!("bucket-latency: {}: {e}", target.label());
                exit(1);
            }
        }
    }

    // Interleaved rounds: bucket order inside a round is fixed, so no bucket is
    // systematically first or last.
    eprintln!("timing {rounds} rounds x {iters} passes ...");
    let mut samples: Vec<Vec<Duration>> = vec![Vec::with_capacity(rounds); prepared.len()];
    for _ in 0..rounds {
        for (i, (target, model, _, _, _, inputs)) in prepared.iter().enumerate() {
            let start = Instant::now();
            for _ in 0..iters {
                if let Err(e) = bundle::predict(model, inputs) {
                    eprintln!("bucket-latency: {}: {e}", target.label());
                    exit(1);
                }
            }
            samples[i].push(start.elapsed() / iters as u32);
        }
    }

    let mut measured: Vec<Measured> = prepared
        .iter()
        .zip(&mut samples)
        .map(|((target, _, load, seq, dim, _), s)| {
            s.sort();
            Measured {
                label: target.label(),
                seq: *seq,
                dim: *dim,
                load: *load,
                best: s[0],
                worst: s[s.len() - 1],
                median: s[s.len() / 2],
            }
        })
        .collect();
    measured.sort_by_key(|m| m.seq);

    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!("root    : {}", root.display());
    println!(
        "protocol: warmup {warmup}, {rounds} interleaved rounds of {iters} passes, minimum taken"
    );
    // The first load in a process pays a one-off cost the others do not (CoreML
    // and the ANE service coming up, plus compiling the program if the OS has no
    // cached one). Averaging it across the buckets would attribute that to the
    // bucket that happened to be loaded first, so report the two separately: the
    // marginal figure is what deciding to add a bucket actually costs.
    let first = prepared.first().map(|p| p.2).unwrap_or_default();
    let rest: Duration = prepared.iter().skip(1).map(|p| p.2).sum();
    println!(
        "load    : {:.2}s total, cold — first bucket {:.2}s, then {:.2}s each",
        total_load.as_secs_f64(),
        first.as_secs_f64(),
        if prepared.len() > 1 {
            rest.as_secs_f64() / (prepared.len() - 1) as f64
        } else {
            0.0
        }
    );
    println!();
    println!(
        "{:<34} {:>5} {:>5} {:>10} {:>10} {:>9} {:>10}",
        "bucket", "seq", "dim", "ms/text", "µs/token", "spread", "load ms"
    );
    for m in &measured {
        // The per-token column is what tells a near-linear model (ruri) from one
        // with a cliff (bekko at 192): a bucket whose µs/token jumps is paying
        // for something other than its length.
        println!(
            "{:<34} {:>5} {:>5} {:>10.2} {:>10.1} {:>8.0}% {:>10.0}",
            m.label,
            m.seq,
            m.dim,
            ms(m.best),
            ms(m.best) * 1e3 / m.seq as f64,
            (ms(m.worst) / ms(m.best) - 1.0) * 100.0,
            ms(m.load),
        );
    }

    let noisy: Vec<&Measured> = measured
        .iter()
        .filter(|m| ms(m.worst) / ms(m.best) > 1.25)
        .collect();
    if !noisy.is_empty() {
        println!();
        println!(
            "{} buckets varied by more than 25% across rounds: {}",
            noisy.len(),
            noisy
                .iter()
                .map(|m| m.label.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("   the machine is busy or the warmup was short; raise --warmup/--rounds");
        println!("   before reading anything into the differences between buckets");
    }
    println!();
    println!("median vs minimum, as a second opinion on the same runs:");
    for m in &measured {
        println!(
            "  {:<32} min {:>8.2} ms   median {:>8.2} ms",
            m.label,
            ms(m.best),
            ms(m.median)
        );
    }
}
