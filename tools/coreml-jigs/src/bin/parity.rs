//! Compare two Kohagi backends over the same texts.
//!
//!     parity --kohagi ./target/release/kohagi --texts texts.txt \
//!            --common "--max-seq-length 128" \
//!            --a "" \
//!            --b "--device coreml --coreml-model-id takahashim/ruri-v3-130m-coreml"
//!
//! Unlike `tools/parity_check.py`, which checks Kohagi against the PyTorch
//! reference, this compares Kohagi against itself on two devices. The CoreML
//! path should match the CPU path to fp16 rounding, and `1 - cosine` above about
//! `1e-5` means something other than precision changed.
//!
//! The settings that invalidate the comparison go in `--common`, not in `--a` or
//! `--b`, and both invocations are printed. A differing `--prefix` (a missing
//! trailing space is enough) or `--max-seq-length` moves `1 - cosine` by orders
//! of magnitude, and the result then measures the configuration rather than the
//! backends.

use std::io::Write;
use std::process::{exit, Command, Stdio};

struct Row {
    id: String,
    embedding: Vec<f64>,
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Split a flag string the way a shell would for the simple cases these
/// invocations need: whitespace, with double quotes to keep a value together
/// (`--prefix "検索文書: "`). No escapes, no single quotes — anything more and
/// the caller should be writing a script instead.
fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut started = false;
    for c in s.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                started = true;
            }
            c if c.is_whitespace() && !quoted => {
                if started {
                    out.push(std::mem::take(&mut cur));
                    started = false;
                }
            }
            c => {
                cur.push(c);
                started = true;
            }
        }
    }
    if started {
        out.push(cur);
    }
    out
}

/// Run Kohagi over `input` and parse its JSONL output.
///
/// stdin comes from a temporary file rather than a pipe: the protocol warns that
/// writing all of stdin before reading stdout can deadlock both processes, and a
/// file sidesteps that without a reader thread.
fn run(bin: &str, args: &[String], input: &str, label: &str) -> Vec<Row> {
    let tmp = std::env::temp_dir().join(format!("parity-{label}-{}.jsonl", std::process::id()));
    let mut f = std::fs::File::create(&tmp).expect("create temp input");
    f.write_all(input.as_bytes()).expect("write temp input");
    drop(f);

    let stdin = std::fs::File::open(&tmp).expect("reopen temp input");
    let out = Command::new(bin)
        .args(args)
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output();
    let _ = std::fs::remove_file(&tmp);

    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!("parity: running {bin}: {e}");
            exit(2);
        }
    };
    if !out.status.success() {
        eprintln!("parity: side {label} exited with {}", out.status);
        exit(1);
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("parity: side {label} wrote a line that is not JSON: {e}");
                exit(1);
            }
        };
        let embedding: Vec<f64> = v["embedding"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
            .unwrap_or_default();
        rows.push(Row {
            id: v["id"].to_string(),
            embedding,
        });
    }
    rows
}

struct Compared {
    id: String,
    cosine_distance: f64,
    max_abs: f64,
    mean_abs: f64,
    nonfinite: usize,
}

fn compare(a: &Row, b: &Row) -> Compared {
    let dot: f64 = a
        .embedding
        .iter()
        .zip(&b.embedding)
        .map(|(x, y)| x * y)
        .sum();
    let na: f64 = a.embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.embedding.iter().map(|y| y * y).sum::<f64>().sqrt();
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        f64::NAN
    };
    let diffs: Vec<f64> = a
        .embedding
        .iter()
        .zip(&b.embedding)
        .map(|(x, y)| (x - y).abs())
        .collect();
    Compared {
        id: a.id.clone(),
        cosine_distance: 1.0 - cosine,
        max_abs: diffs.iter().copied().fold(0.0, f64::max),
        mean_abs: if diffs.is_empty() {
            f64::NAN
        } else {
            diffs.iter().sum::<f64>() / diffs.len() as f64
        },
        nonfinite: a
            .embedding
            .iter()
            .chain(&b.embedding)
            .filter(|v| !v.is_finite())
            .count(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: parity --kohagi <bin> --texts <file> [--common \"...\"] \
                 [--a \"...\"] --b \"...\" [--threshold 1e-4]";
    let (Some(bin), Some(texts), Some(b_args)) = (
        flag(&args, "--kohagi"),
        flag(&args, "--texts"),
        flag(&args, "--b"),
    ) else {
        eprintln!("{usage}");
        eprintln!(
            "\n--common carries the settings that must match on both sides \
             (--prefix, --max-seq-length, --pooling).\n\
             A difference there measures the configuration, not the backends."
        );
        exit(2);
    };
    let a_args = flag(&args, "--a").unwrap_or_default();
    let common = flag(&args, "--common").unwrap_or_default();
    let threshold: f64 = flag(&args, "--threshold")
        .map(|t| match t.parse() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("parity: `{t}` is not a number for --threshold");
                exit(2);
            }
        })
        .unwrap_or(1e-4);

    let raw = match std::fs::read_to_string(&texts) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("parity: reading {texts}: {e}");
            exit(2);
        }
    };
    let lines: Vec<&str> = raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        eprintln!("parity: {texts} holds no non-blank lines");
        exit(2);
    }
    // One JSONL record per input line, ids assigned here so both sides agree.
    let input: String = lines
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{}\n", serde_json::json!({"id": i, "text": t})))
        .collect();

    let build = |side: &str| {
        let mut v = split_args(&common);
        v.extend(split_args(side));
        v
    };
    let (av, bv) = (build(&a_args), build(&b_args));
    println!("texts   : {} ({} records)", texts, lines.len());
    println!("a       : {bin} {}", av.join(" "));
    println!("b       : {bin} {}", bv.join(" "));
    println!();

    let ra = run(&bin, &av, &input, "a");
    let rb = run(&bin, &bv, &input, "b");

    if ra.len() != rb.len() {
        eprintln!(
            "parity: a returned {} rows, b returned {} — one side skipped records \
             (check its stderr above)",
            ra.len(),
            rb.len()
        );
        exit(1);
    }
    if ra.len() != lines.len() {
        eprintln!(
            "parity: {} texts in, {} rows out on both sides — records were skipped",
            lines.len(),
            ra.len()
        );
        exit(1);
    }
    // The protocol says to map by id, not by order.
    let mut by_id: std::collections::HashMap<&str, &Row> =
        rb.iter().map(|r| (r.id.as_str(), r)).collect();
    let mut results = Vec::new();
    for a in &ra {
        let Some(b) = by_id.remove(a.id.as_str()) else {
            eprintln!("parity: id {} is missing from side b", a.id);
            exit(1);
        };
        if a.embedding.len() != b.embedding.len() {
            eprintln!(
                "parity: id {} has dim {} on a and {} on b — different models, not \
                 different backends",
                a.id,
                a.embedding.len(),
                b.embedding.len()
            );
            exit(1);
        }
        results.push(compare(a, b));
    }

    let dim = ra.first().map_or(0, |r| r.embedding.len());
    let mut distances: Vec<f64> = results.iter().map(|r| r.cosine_distance).collect();
    distances.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let worst = results
        .iter()
        .max_by(|x, y| {
            x.cosine_distance
                .partial_cmp(&y.cosine_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("at least one row");
    let nonfinite: usize = results.iter().map(|r| r.nonfinite).sum();

    println!("dim     : {dim}");
    println!("1 - cosine");
    println!("  min    : {:.3e}", distances[0]);
    println!("  median : {:.3e}", distances[distances.len() / 2]);
    println!(
        "  max    : {:.3e}  (id {})",
        worst.cosine_distance, worst.id
    );
    println!("worst row (id {})", worst.id);
    println!("  max |a-b|  : {:.3e}", worst.max_abs);
    println!("  mean |a-b| : {:.3e}", worst.mean_abs);
    println!("non-finite values: {nonfinite}");

    let over: Vec<&Compared> = results
        .iter()
        // NaN spelled out rather than left to a negated comparison: a zero-norm
        // vector on either side produces one, and it must not pass as "small".
        .filter(|r| r.cosine_distance.is_nan() || r.cosine_distance > threshold)
        .collect();
    println!();
    if nonfinite > 0 {
        println!("FAIL: {nonfinite} non-finite values in the output");
        exit(1);
    }
    if over.is_empty() {
        println!(
            "OK: all {} rows within 1 - cosine <= {:.0e}",
            results.len(),
            threshold
        );
    } else {
        println!(
            "FAIL: {}/{} rows exceed 1 - cosine {:.0e}",
            over.len(),
            results.len(),
            threshold
        );
        for r in over.iter().take(10) {
            println!("  id {}: {:.3e}", r.id, r.cosine_distance);
        }
        if over.len() > 10 {
            println!("  ... and {} more", over.len() - 10);
        }
        println!(
            "\nfp16 conversion alone lands near 1e-5. An order of magnitude above that \
             usually means\nthe two sides were configured differently (--prefix, \
             --max-seq-length, --pooling)\nrather than that a backend is wrong; put those \
             in --common and re-run."
        );
        exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_values_survive_splitting() {
        assert_eq!(
            split_args("--device coreml --coreml-dir /tmp/m"),
            vec!["--device", "coreml", "--coreml-dir", "/tmp/m"]
        );
        // A trailing space inside a prefix is exactly the difference that moves
        // 1 - cosine by three orders of magnitude, so it has to survive.
        assert_eq!(
            split_args("--prefix \"検索文書: \""),
            vec!["--prefix", "検索文書: "]
        );
        assert_eq!(split_args("   "), Vec::<String>::new());
        assert_eq!(split_args("--flag \"\""), vec!["--flag", ""]);
    }

    #[test]
    fn identical_vectors_have_zero_distance() {
        let v: Vec<f64> = (0..8).map(|i| (i as f64) / 8.0 - 0.5).collect();
        let a = Row {
            id: "0".into(),
            embedding: v.clone(),
        };
        let b = Row {
            id: "0".into(),
            embedding: v,
        };
        let c = compare(&a, &b);
        assert!(c.cosine_distance.abs() < 1e-15, "{}", c.cosine_distance);
        assert_eq!(c.max_abs, 0.0);
        assert_eq!(c.nonfinite, 0);
    }

    #[test]
    fn a_negated_vector_is_the_far_end() {
        let a = Row {
            id: "0".into(),
            embedding: vec![0.6, 0.8],
        };
        let b = Row {
            id: "0".into(),
            embedding: vec![-0.6, -0.8],
        };
        let c = compare(&a, &b);
        assert!((c.cosine_distance - 2.0).abs() < 1e-12);
        assert!((c.max_abs - 1.6).abs() < 1e-12);
    }
}
