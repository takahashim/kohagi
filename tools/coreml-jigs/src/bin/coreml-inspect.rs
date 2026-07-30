//! Report what a converted CoreML directory actually contains, and check it
//! against what Kohagi will assume of it.
//!
//!     coreml-inspect <dir|bundle> [--json]
//!
//! Everything here is read from the model's own description rather than derived
//! from file names, so the two can be compared. The three disagreements that
//! matter, because each one is silent at runtime:
//!
//! - a bundle named `seq-<N>` whose model is not `N` long,
//! - a `config.json` whose `hidden_size` is not the model's output width,
//! - a missing `input_ids` / `attention_mask` / `hidden` feature.
//!
//! Neither form is compiled to read it, so a directory inspects in well under the
//! ~20s a first real load would cost. A `.mlmodelc` is read through
//! `MLModelAsset`, which reports every function; a `.mlpackage` is decoded from
//! its protobuf, because `MLModelAsset` only opens compiled models.

use std::path::{Path, PathBuf};
use std::process::exit;

use coreml_jigs::bundle::{collect, lengths_in_name};
use coreml_jigs::spec;
use coreml_jigs::{await_handler, blob};
use objc2::rc::Retained;
use objc2_core_ml::{MLFeatureDescription, MLModelAsset, MLModelDescription, MLMultiArrayDataType};
use objc2_foundation::{NSArray, NSDictionary, NSString, NSURL};

/// Name an `MLMultiArrayDataType`. The enum is a bit pattern (`0x10000` for
/// float, `0x20000` for int, or-ed with the width), and its `Debug` prints the
/// raw number, which is not what a report should show.
fn multiarray_dtype(d: MLMultiArrayDataType) -> String {
    match d {
        MLMultiArrayDataType::Float16 => "fp16".to_string(),
        MLMultiArrayDataType::Float32 => "fp32".to_string(),
        MLMultiArrayDataType::Double => "fp64".to_string(),
        MLMultiArrayDataType::Int32 => "int32".to_string(),
        other => format!("dtype {}", other.0),
    }
}

/// One input or output of a CoreML function.
struct Feature {
    name: String,
    shape: Vec<usize>,
    dtype: String,
}

/// One function inside a bundle: `main` for a single-length bundle, `seq_<N>`
/// for each length of a multi-function one.
struct Function {
    /// As shown. May carry a note such as `(default)`, so it is not what the
    /// sequence-length check parses — [`Self::real_name`] is.
    name: String,
    inputs: Vec<Feature>,
    outputs: Vec<Feature>,
    /// Converter provenance from the model's `userDefined` metadata, when the
    /// form being read carries it. A model card is expected to state these
    /// versions in the model card.
    provenance: Vec<(String, String)>,
}

impl Function {
    /// The name without any display note, which is what Kohagi addresses and what
    /// carries the sequence length.
    fn real_name(&self) -> &str {
        self.name.split_whitespace().next().unwrap_or(&self.name)
    }

    fn output(&self, name: &str) -> Option<&Feature> {
        self.outputs.iter().find(|f| f.name == name)
    }

    fn has_input(&self, name: &str) -> bool {
        self.inputs.iter().any(|f| f.name == name)
    }

    /// `[1, seq, dim]` -> `(seq, dim)`. `None` for any other rank, which is
    /// itself a finding rather than something to index into.
    fn hidden_shape(&self) -> Option<(usize, usize)> {
        match self.output("hidden").map(|f| f.shape.as_slice()) {
            Some([1, seq, dim]) => Some((*seq, *dim)),
            _ => None,
        }
    }
}

struct Bundle {
    path: PathBuf,
    form: &'static str,
    bytes: u64,
    functions: Vec<Function>,
    /// Blob summary, when the bundle has a readable `weight.bin`.
    weights: Option<(usize, u64)>,
}

fn features(d: &NSDictionary<NSString, MLFeatureDescription>) -> Vec<Feature> {
    let mut out: Vec<Feature> = d
        .keys()
        .map(|key| {
            let desc = d.objectForKey(&key).expect("key from the same dictionary");
            let (shape, dtype) = match unsafe { desc.multiArrayConstraint() } {
                Some(c) => (
                    unsafe { c.shape() }
                        .iter()
                        .map(|n| n.as_isize().max(0) as usize)
                        .collect(),
                    multiarray_dtype(unsafe { c.dataType() }),
                ),
                None => (Vec::new(), "non-multiarray".to_string()),
            };
            Feature {
                name: key.to_string(),
                shape,
                dtype,
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn describe(desc: &MLModelDescription, name: String) -> Function {
    let (inputs, outputs) = unsafe {
        (
            desc.inputDescriptionsByName(),
            desc.outputDescriptionsByName(),
        )
    };
    Function {
        name,
        inputs: features(&inputs),
        outputs: features(&outputs),
        provenance: Vec::new(),
    }
}

/// Read a `.mlpackage` from its protobuf. `MLModelAsset` refuses the portable
/// form, and compiling it just to read its shapes would cost ~20s per bucket —
/// which is exactly the wrong trade for a jig meant to check a directory before
/// it is published.
///
/// A multi-function bundle describes each function separately, and all of them are
/// checked. The MIL program's own function names are not consulted: the
/// description is what a caller can actually address.
fn read_package(path: &Path) -> Result<Vec<Function>, String> {
    let proto = path.join("Data/com.apple.CoreML/model.mlmodel");
    let bytes = std::fs::read(&proto).map_err(|e| format!("reading {}: {e}", proto.display()))?;
    let spec = spec::read(&bytes)?;
    let convert = |f: &spec::Feature| Feature {
        name: f.name.clone(),
        shape: f.shape.iter().map(|&d| d.max(0) as usize).collect(),
        dtype: spec::dtype_name(f.dtype),
    };
    // A multi-function package describes each function separately; a
    // single-function one describes its interface at the top level.
    if !spec.described_functions.is_empty() {
        let mut out: Vec<Function> = spec
            .described_functions
            .iter()
            .map(|f| Function {
                name: if f.name == spec.default_function {
                    format!("{} (default)", f.name)
                } else {
                    f.name.clone()
                },
                inputs: f.inputs.iter().map(convert).collect(),
                outputs: f.outputs.iter().map(convert).collect(),
                provenance: Vec::new(),
            })
            .collect();
        if let Some(first) = out.first_mut() {
            first.provenance = spec.metadata;
        }
        return Ok(out);
    }
    Ok(vec![Function {
        name: "main".to_string(),
        inputs: spec.inputs.iter().map(convert).collect(),
        outputs: spec.outputs.iter().map(convert).collect(),
        provenance: spec.metadata,
    }])
}

/// Read every function of a compiled bundle. A single-function bundle reports no
/// function names, so it is described through the whole-asset call and labelled
/// `main`.
fn read_bundle(path: &Path) -> Result<Vec<Function>, String> {
    if path.extension().and_then(|e| e.to_str()) == Some("mlpackage") {
        return read_package(path);
    }
    let url = NSURL::fileURLWithPath(&NSString::from_str(
        path.to_str().ok_or("path is not valid UTF-8")?,
    ));
    let asset = unsafe { MLModelAsset::modelAssetWithURL_error(&url) }
        .map_err(|e| format!("opening the model asset: {e}"))?;

    let names: Retained<NSArray<NSString>> =
        await_handler(|h| unsafe { asset.functionNamesWithCompletionHandler(h) })
            .map_err(|e| format!("reading function names: {e}"))?;

    if names.is_empty() {
        let desc = await_handler(|h| unsafe { asset.modelDescriptionWithCompletionHandler(h) })
            .map_err(|e| format!("reading the model description: {e}"))?;
        return Ok(vec![describe(&desc, "main".to_string())]);
    }

    let mut out = Vec::new();
    for name in names.iter() {
        let desc = await_handler(|h| unsafe {
            asset.modelDescriptionOfFunctionNamed_completionHandler(&name, h)
        })
        .map_err(|e| format!("reading function {name}: {e}"))?;
        out.push(describe(&desc, name.to_string()));
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn dir_bytes(path: &Path) -> u64 {
    if !path.is_dir() {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    std::fs::read_dir(path)
        .map(|entries| entries.flatten().map(|e| dir_bytes(&e.path())).sum::<u64>())
        .unwrap_or(0)
}

fn weights_summary(bundle: &Path) -> Option<(usize, u64)> {
    let path = blob::weight_path(bundle)?;
    let bytes = std::fs::read(path).ok()?;
    let (_, blobs) = blob::parse(&bytes).ok()?;
    Some((blobs.len(), blobs.iter().map(|b| b.size_in_bytes).sum()))
}

/// The `hidden_size` a sibling `config.json` declares, which is where Kohagi
/// gets the width it pools over.
fn declared_hidden_size(root: &Path) -> Option<u64> {
    let dir = if root.is_dir() && collect(root).first() != Some(&root.to_path_buf()) {
        root.to_path_buf()
    } else {
        root.parent()?.to_path_buf()
    };
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()?
        .get("hidden_size")?
        .as_u64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: coreml-inspect <dir|bundle> [--json]");
        exit(2);
    }
    let root = PathBuf::from(&args[1]);
    let as_json = args.iter().any(|a| a == "--json");

    let paths = collect(&root);
    if paths.is_empty() {
        eprintln!(
            "coreml-inspect: no .mlpackage or .mlmodelc under {}",
            root.display()
        );
        exit(2);
    }
    let declared = declared_hidden_size(&root);

    let mut bundles = Vec::new();
    let mut findings: Vec<String> = Vec::new();
    for path in paths {
        let form = match path.extension().and_then(|e| e.to_str()) {
            Some("mlmodelc") => "compiled",
            _ => "package",
        };
        let functions = match read_bundle(&path) {
            Ok(f) => f,
            Err(e) => {
                findings.push(format!("{}: {e}", path.display()));
                continue;
            }
        };
        let named = lengths_in_name(&path);
        let label = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        for f in &functions {
            if f.inputs.is_empty() && f.outputs.is_empty() {
                // Listed but not described (a non-default function of a
                // .mlpackage); saying nothing is better than a false finding.
                continue;
            }
            for input in ["input_ids", "attention_mask"] {
                if !f.has_input(input) {
                    findings.push(format!(
                        "{label} function {}: no input `{input}` (has: {})",
                        f.name,
                        f.inputs
                            .iter()
                            .map(|x| x.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
            let Some((seq, dim)) = f.hidden_shape() else {
                findings.push(match f.output("hidden") {
                    Some(h) => format!(
                        "{label} function {}: output `hidden` has shape {:?}, expected [1, seq, dim]",
                        f.name, h.shape
                    ),
                    None => format!(
                        "{label} function {}: no output `hidden` (has: {})",
                        f.name,
                        f.outputs.iter().map(|x| x.name.as_str()).collect::<Vec<_>>().join(", ")
                    ),
                });
                continue;
            };
            if let Some(declared) = declared {
                if declared != dim as u64 {
                    findings.push(format!(
                        "{label} function {}: outputs width {dim}, but config.json \
                         declares hidden_size {declared} — Kohagi would pool over the \
                         wrong stride",
                        f.name
                    ));
                }
            }
            if let Some(named) = &named {
                // A `seq-<N>` bundle must be N long; in a multi-function bundle
                // the function name carries the length, so check that instead.
                let expect = if named.len() == 1 {
                    Some(named[0])
                } else {
                    f.real_name()
                        .strip_prefix("seq_")
                        .and_then(|s| s.parse().ok())
                };
                match expect {
                    Some(n) if n != seq => findings.push(format!(
                        "{label} function {}: model is {seq} long but its name says {n} \
                         — Kohagi routes and pads by the name",
                        f.name
                    )),
                    None => findings.push(format!(
                        "{label} function {}: name carries no sequence length; Kohagi \
                         reads lengths from `seq_<N>` function names in a multi-function \
                         bundle",
                        f.name
                    )),
                    Some(_) => {}
                }
            }
        }

        bundles.push(Bundle {
            bytes: dir_bytes(&path),
            weights: weights_summary(&path),
            path,
            form,
            functions,
        });
    }

    if as_json {
        let value = serde_json::json!({
            "root": root.display().to_string(),
            "declared_hidden_size": declared,
            "bundles": bundles.iter().map(|b| serde_json::json!({
                "path": b.path.display().to_string(),
                "form": b.form,
                "bytes": b.bytes,
                "blobs": b.weights.map(|(n, _)| n),
                "weight_bytes": b.weights.map(|(_, n)| n),
                "functions": b.functions.iter().map(|f| serde_json::json!({
                    "name": f.name,
                    "inputs": f.inputs.iter().map(|x| serde_json::json!({
                        "name": x.name, "shape": x.shape, "dtype": x.dtype })).collect::<Vec<_>>(),
                    "outputs": f.outputs.iter().map(|x| serde_json::json!({
                        "name": x.name, "shape": x.shape, "dtype": x.dtype })).collect::<Vec<_>>(),
                    "provenance": f.provenance.iter().cloned().collect::<std::collections::BTreeMap<_,_>>(),
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "findings": findings,
        });
        println!("{}", serde_json::to_string_pretty(&value).unwrap());
    } else {
        println!("root    : {}", root.display());
        match declared {
            Some(d) => println!("config  : hidden_size {d}"),
            None => println!("config  : no readable config.json beside the bundles"),
        }
        for b in &bundles {
            println!(
                "\n{} [{}]  {:.1} MB",
                b.path.file_name().unwrap_or_default().to_string_lossy(),
                b.form,
                b.bytes as f64 / 1e6
            );
            if let Some((n, bytes)) = b.weights {
                println!("  weights : {n} blobs, {:.1} MB", bytes as f64 / 1e6);
            }
            for f in &b.functions {
                let shape = f
                    .hidden_shape()
                    .map(|(s, d)| format!("seq {s}, dim {d}"))
                    .unwrap_or_else(|| "no [1, seq, dim] `hidden` output".to_string());
                println!("  {:<12} {shape}", f.name);
                for x in f.inputs.iter().chain(&f.outputs) {
                    println!("      {:<16} {:?} {}", x.name, x.shape, x.dtype);
                }
                for (k, v) in &f.provenance {
                    println!("      {:<16} {v}", k.rsplit('.').next().unwrap_or(k));
                }
            }
        }
        if findings.is_empty() {
            println!("\nOK: every function takes input_ids/attention_mask and returns hidden [1, seq, dim]");
            println!("    consistent with its name and with config.json");
        } else {
            println!("\n{} findings:", findings.len());
            for f in &findings {
                println!("  - {f}");
            }
        }
    }

    if !findings.is_empty() {
        exit(1);
    }
}
