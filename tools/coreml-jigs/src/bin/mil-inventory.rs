//! What operations a MIL program is made of, and how two programs differ.
//!
//!     mil-inventory <bundle|model.mil|model.mlmodel> [--json]
//!     mil-inventory <a> --diff <b>
//!
//! This is the jig for the question an emitter keeps asking: does the graph I
//! wrote consist of the same operations as the one the Python converter wrote?
//! Both forms of a MIL program are read — the text inside a `.mlmodelc` and the
//! protobuf inside a `.mlpackage` — because they are different programs and only
//! one of them is what an emitter has to produce.

use std::path::PathBuf;
use std::process::exit;

use coreml_jigs::mil::{self, Form, Inventory};

fn report(inv: &Inventory) {
    println!("source  : {}", inv.source.display());
    println!("form    : {}", inv.form.label());
    println!(
        "ops     : {} total, {} const, {} compute, {} kinds",
        inv.total(),
        inv.consts(),
        inv.compute(),
        inv.kinds()
    );
    if let Some(n) = inv.blob_refs {
        println!("blobs   : {n} weight.bin references");
    }
    println!();
    let width = inv.counts.keys().map(|k| k.len()).max().unwrap_or(8).max(8);
    let mut rows: Vec<(&String, &usize)> = inv.counts.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (op, n) in rows {
        println!("{op:<width$}  {n:>6}");
    }
}

fn report_diff(a: &Inventory, b: &Inventory) {
    println!("a       : {} [{}]", a.source.display(), a.form.label());
    println!("b       : {} [{}]", b.source.display(), b.form.label());
    println!();
    println!(
        "totals  : a {} ops ({} compute, {} kinds), b {} ops ({} compute, {} kinds)",
        a.total(),
        a.compute(),
        a.kinds(),
        b.total(),
        b.compute(),
        b.kinds()
    );
    println!();

    match mil::first_divergence(a, b) {
        None => println!(
            "sequence: identical, all {} operations in the same order",
            a.sequence.len()
        ),
        Some((at, x, y)) => {
            println!("sequence: first differs at index {at} — a has `{x}`, b has `{y}`")
        }
    }
    println!();

    let deltas = mil::diff(a, b);
    let width = deltas.iter().map(|d| d.op.len()).max().unwrap_or(8).max(8);
    println!("{:<width$}  {:>8} {:>8} {:>8}", "op", "a", "b", "b - a");
    let mut differing = 0;
    for d in &deltas {
        let mark = if d.a == d.b {
            " "
        } else {
            differing += 1;
            "*"
        };
        println!(
            "{:<width$}  {:>8} {:>8} {:>+8} {mark}",
            d.op,
            d.a,
            d.b,
            d.b as i64 - d.a as i64
        );
    }
    println!();
    if differing == 0 {
        println!("identical inventories ({} op kinds)", deltas.len());
    } else {
        println!("{differing} of {} op kinds differ", deltas.len());
        let only_a: Vec<&str> = deltas
            .iter()
            .filter(|d| d.b == 0)
            .map(|d| d.op.as_str())
            .collect();
        let only_b: Vec<&str> = deltas
            .iter()
            .filter(|d| d.a == 0)
            .map(|d| d.op.as_str())
            .collect();
        if !only_a.is_empty() {
            println!("only in a: {}", only_a.join(", "));
        }
        if !only_b.is_empty() {
            println!("only in b: {}", only_b.join(", "));
        }
        // Comparing the two forms of the *same* model is the expected case for
        // this tool, and there the differences are coremlc's doing rather than a
        // sign that anything is wrong. Say so, so the output is not read as a
        // failure.
        if a.form != b.form {
            println!();
            println!(
                "the two sides are different forms of a MIL program, so these differences\n\
                 are what coremlc does between them, not a discrepancy. An emitter targets\n\
                 the package form."
            );
        }
    }
}

fn json(inv: &Inventory) -> serde_json::Value {
    serde_json::json!({
        "source": inv.source.display().to_string(),
        "form": match inv.form { Form::CompiledText => "compiled-text", Form::PackageProto => "package-proto" },
        "total": inv.total(),
        "const": inv.consts(),
        "compute": inv.compute(),
        "kinds": inv.kinds(),
        "blob_refs": inv.blob_refs,
        "counts": inv.counts,
        "sequence": inv.sequence,
    })
}

fn load(path: &str) -> Inventory {
    match mil::read(&PathBuf::from(path)) {
        Ok(inv) => inv,
        Err(e) => {
            eprintln!("mil-inventory: {e}");
            exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: mil-inventory <bundle|model.mil|model.mlmodel> [--json]\n\
             \x20      mil-inventory <a> --diff <b>"
        );
        exit(2);
    }
    let other = args
        .iter()
        .position(|a| a == "--diff")
        .and_then(|i| args.get(i + 1));
    let as_json = args.iter().any(|a| a == "--json");

    let a = load(&args[1]);
    match other {
        None if as_json => println!("{}", serde_json::to_string_pretty(&json(&a)).unwrap()),
        None => report(&a),
        Some(b) => {
            let b = load(b);
            if as_json {
                let deltas: Vec<serde_json::Value> = mil::diff(&a, &b)
                    .iter()
                    .map(|d| serde_json::json!({"op": d.op, "a": d.a, "b": d.b}))
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "a": json(&a), "b": json(&b), "deltas": deltas,
                    }))
                    .unwrap()
                );
            } else {
                report_diff(&a, &b);
            }
        }
    }
}
