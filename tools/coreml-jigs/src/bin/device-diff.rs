//! Run one model on the ANE and on the CPU, and compare the numbers.
//!
//!     device-diff <bundle> [function] [--tokens N]
//!
//! The same file, the same weights, the same operations — only the hardware
//! changes. That separates two explanations for a model whose CoreML output does
//! not match a float32 reference:
//!
//! **This does not work for the models it was written for.** A graph masked with
//! fp16 `-inf` — which is what `scripts/convert_coreml.py` emits and what this
//! crate reproduces — comes back entirely NaN under `CPUOnly`, on the reference
//! conversion as much as on ours. `All` runs it, but `All` includes the Neural
//! Engine, so agreeing with it proves nothing. The tool says so rather than
//! drawing the conclusion anyway.
//!
//! What it was meant to separate:
//!
//! - **the arithmetic.** The ANE computes in fp16 and the CPU has more headroom,
//!   so a model with large activations diverges here while a well-conditioned one
//!   does not. Nothing to fix in the graph; the conversion itself costs precision
//!   for this checkpoint.
//! - **the graph or the weights.** Both devices run the same wrong thing, so they
//!   agree with each other and disagree with the reference together.
//!
//! Written for `nomic-ai/modernbert-embed-base`, whose emitted model matches
//! Kohagi's CPU path to 4.8e-3 where four other checkpoints match to 4e-6
//! than one that only reports a number.

use std::path::PathBuf;
use std::process::exit;

use coreml_jigs::bundle::{self, Target};
use objc2_core_ml::MLComputeUnits;

fn flag_usize(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map_or(default, |v| match v.parse() {
            Ok(n) => n,
            Err(_) => {
                eprintln!("device-diff: `{v}` is not a number for {name}");
                exit(2);
            }
        })
}

/// Summary of two runs of the same model.
struct Diff {
    max_abs: f32,
    mean_abs: f32,
    cosine_distance: f64,
    max_magnitude: f32,
    /// Counted rather than folded into the max: `f32::max` returns the other
    /// operand when one is NaN, so a non-finite value hides inside a maximum and
    /// only shows up as a NaN in a mean. Which is exactly how this was missed the
    /// first time.
    nonfinite: usize,
    /// The first row holding a non-finite value, which says whether they are
    /// confined to the padding.
    first_nonfinite_row: Option<usize>,
}

fn compare(ane: &[f32], cpu: &[f32], dim: usize) -> Diff {
    let dot: f64 = ane
        .iter()
        .zip(cpu)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let norm = |v: &[f32]| {
        v.iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt()
    };
    let diffs: Vec<f32> = ane.iter().zip(cpu).map(|(a, b)| (a - b).abs()).collect();
    Diff {
        max_abs: diffs.iter().copied().fold(0.0, f32::max),
        mean_abs: diffs.iter().sum::<f32>() / diffs.len() as f32,
        cosine_distance: 1.0 - dot / (norm(ane) * norm(cpu)),
        // How close the output comes to fp16's limits, which is the thing that
        // makes an arithmetic explanation plausible or not.
        max_magnitude: ane
            .iter()
            .chain(cpu)
            .filter(|v| v.is_finite())
            .map(|v| v.abs())
            .fold(0.0, f32::max),
        nonfinite: ane.iter().chain(cpu).filter(|v| !v.is_finite()).count(),
        first_nonfinite_row: ane
            .iter()
            .chain(cpu)
            .position(|v| !v.is_finite())
            .map(|i| (i % ane.len()) / dim),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: device-diff <bundle> [function] [--tokens N]");
        exit(2);
    }
    let root = PathBuf::from(&args[1]);
    let wanted = args.get(2).filter(|a| !a.starts_with("--")).cloned();
    let tokens = flag_usize(&args, "--tokens", 24);

    let targets = match bundle::targets(&root) {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => {
            eprintln!(
                "device-diff: no .mlpackage or .mlmodelc under {}",
                root.display()
            );
            exit(2);
        }
        Err(e) => {
            eprintln!("device-diff: {e}");
            exit(1);
        }
    };
    let targets: Vec<Target> = match bundle::compile_once(&targets) {
        Ok(t) => t
            .into_iter()
            .filter(|t| {
                wanted
                    .as_ref()
                    .is_none_or(|w| t.function.as_deref() == Some(w))
            })
            .collect(),
        Err(e) => {
            eprintln!("device-diff: {e}");
            exit(1);
        }
    };

    println!("model   : {}", root.display());
    println!("input   : {tokens} real tokens, the rest padding");
    println!();
    println!(
        "{:<34} {:>11} {:>11} {:>11} {:>9} {:>10}",
        "bucket", "1 - cosine", "max |a-b|", "mean |a-b|", "max |v|", "non-finite"
    );

    let mut worst = 0.0f64;
    let mut unusable = false;
    // Only `CPUOnly` keeps the Neural Engine out of the comparison. `All` does not,
    // so an agreement with it is not evidence about fp16 arithmetic.
    let mut cpu_excluded_the_ane = false;
    for target in &targets {
        let (model, _) = match bundle::load_on(target, MLComputeUnits::CPUAndNeuralEngine) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("device-diff: {e}");
                exit(1);
            }
        };
        let Some((seq, dim)) = bundle::hidden_shape(&model) else {
            eprintln!(
                "device-diff: {} has no `hidden` output of shape [1, seq, dim]",
                target.label()
            );
            exit(1);
        };
        // A sentence-shaped input: real ids up front, padding after, so the mask
        // path is exercised the way it is in use.
        let mut ids = bundle::fake_ids(seq, 30_000);
        let mut mask = vec![1i32; seq];
        for i in tokens.min(seq)..seq {
            ids[i] = 0;
            mask[i] = 0;
        }
        let inputs = match bundle::inputs(&ids, &mask) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("device-diff: {e}");
                exit(1);
            }
        };

        let run = |units| -> Vec<f32> {
            let (m, _) = bundle::load_on(target, units).unwrap_or_else(|e| {
                eprintln!("device-diff: {e}");
                exit(1);
            });
            bundle::predict_output(&m, &inputs, "hidden").unwrap_or_else(|e| {
                eprintln!("device-diff: {e}");
                exit(1);
            })
        };
        let bad = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
        let ane = run(MLComputeUnits::CPUAndNeuralEngine);

        // The other side, whichever of these can run the model at all. A CoreML
        // graph masked with fp16 -inf comes back entirely NaN on some compute
        // units, so the fallback is tried before concluding anything.
        let alternatives = [
            ("CPUOnly", MLComputeUnits::CPUOnly),
            ("All", MLComputeUnits::All),
        ];
        let mut other = None;
        for (name, units) in alternatives {
            let v = run(units);
            if bad(&v) < v.len() {
                other = Some((name, v));
                break;
            }
            println!("  {name}: all {} values non-finite", v.len());
        }
        let Some((other_name, cpu)) = other else {
            println!(
                "{:<34} only CPUAndNeuralEngine ran this model",
                target.label()
            );
            unusable = true;
            continue;
        };
        println!("  comparing CPUAndNeuralEngine against {other_name}");
        cpu_excluded_the_ane = other_name == "CPUOnly";
        if bad(&ane) == ane.len() {
            println!("  ANE: all {} values non-finite", ane.len());
            unusable = true;
            continue;
        }

        // Only the unpadded rows: a padded row holds whatever the mask left, and
        // Kohagi's pooling discards it.
        let keep = tokens.min(seq) * dim;
        let d = compare(&ane[..keep], &cpu[..keep], dim);
        if d.cosine_distance.is_finite() {
            worst = worst.max(d.cosine_distance);
        }
        println!(
            "{:<34} {:>11.3e} {:>11.3e} {:>11.3e} {:>9.1} {:>10}",
            target.label(),
            d.cosine_distance,
            d.max_abs,
            d.mean_abs,
            d.max_magnitude,
            d.nonfinite
        );
        if d.nonfinite > 0 {
            println!(
                "  {} non-finite values among the {tokens} unpadded rows, first in row {:?} \
                 — a softmax over an entirely masked row produces NaN",
                d.nonfinite, d.first_nonfinite_row
            );
        }
        // The whole output, to say whether the padded rows are the only ones.
        let all = compare(&ane, &cpu, dim);
        if all.nonfinite > d.nonfinite {
            println!(
                "  {} non-finite values over all {seq} rows, so {} of them are in the \
                 padding",
                all.nonfinite,
                all.nonfinite - d.nonfinite
            );
        }
    }

    println!();
    if unusable {
        println!(
            "No conclusion: this comparison needs both devices to run the model, and one\n\
             of them did not."
        );
        return;
    }
    if !cpu_excluded_the_ane {
        println!(
            "No conclusion about the arithmetic: the only compute-unit setting that\n\
             excludes the Neural Engine is CPUOnly, and it could not run this model.\n\
             `All` includes the Neural Engine, so agreeing with it says nothing.\n\
             \n\
             Comparing against a float32 reference needs one outside CoreML — the\n\
             Python conversion, or Kohagi's own candle path through `parity`."
        );
        return;
    }
    // fp16 has ~3 decimal digits, so agreement below 1e-5 between two devices
    // running the same graph means the arithmetic is not where a larger
    // disagreement with a float32 reference comes from.
    if worst < 1e-5 {
        println!(
            "The two devices agree to {worst:.1e}. If this model disagrees with a float32\n\
             reference by much more than that, the cause is the graph or the weights,\n\
             not the Neural Engine's arithmetic."
        );
    } else {
        println!(
            "The two devices disagree by {worst:.1e} on the same graph, so the arithmetic\n\
             differs between them — an fp16 conversion of this checkpoint loses precision\n\
             the CPU path keeps."
        );
    }
}
