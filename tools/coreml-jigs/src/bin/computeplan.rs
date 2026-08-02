//! Per-operation compute-device assignment for a CoreML model, and a regression
//! check against a recorded baseline.
//!
//!     computeplan <model> [function] [--tsv out.tsv] [--baseline in.tsv]
//!
//! Without `--baseline` this prints where CoreML intends to run each operation.
//! With one, it compares against a recorded run and exits non-zero on a
//! difference, which is what turns the measurement into a test: the same model
//! can be placed differently by a new macOS, and nothing else in the repo would
//! notice.
//!
//! What the summary reports, and why it is not just a percentage:
//!
//! - **prologue length** — how many leading operations run off the ANE. All-CPU
//!   at the front is one handoff; CPU operations *between* ANE ones mean the
//!   graph is being partitioned repeatedly, which costs a tensor round trip each
//!   time.
//! - **stragglers** — ANE-capable operations placed on CPU outside the prologue.
//! - **ceiling** — the share of operations that list ANE as supported at all. An
//!   assignment at the ceiling cannot be improved by rewriting the graph.
//!
//! `estimatedCost` is a static estimate, not measured latency. It is printed for
//! comparing two models, not for deciding what to optimise.
//!
//! A `.mlpackage` is compiled to a temporary `.mlmodelc` on the way in, because
//! `MLComputePlan` accepts only the compiled form and aborts the process rather
//! than returning an error when given the other one.

use std::collections::BTreeMap;
use std::process::exit;

use coreml_jigs::{await_handler, bundle, device_name};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{MLComputeDeviceProtocol, MLComputePlan, MLComputeUnits, MLModelConfiguration};
use objc2_foundation::{NSString, NSURL};

struct Op {
    index: usize,
    op_type: String,
    preferred: String,
    supported: Vec<String>,
    cost: f64,
}

impl Op {
    fn ane_capable(&self) -> bool {
        self.supported.iter().any(|d| d == "ANE")
    }
}

/// Name a compute device by its Objective-C class. `ProtocolObject` does not
/// expose `-class` through a trait for this protocol, so read it off the shared
/// `NSObject` layout every CoreML compute device has, rather than downcasting to
/// each of the three concrete classes in turn.
fn device_label(obj: &ProtocolObject<dyn MLComputeDeviceProtocol>) -> String {
    let any: &AnyObject = unsafe { &*(obj as *const _ as *const AnyObject) };
    let class = any.class().name();
    device_name(class.to_str().unwrap_or("unnamed")).to_string()
}

fn plan(model: &str, function: &str) -> Result<Vec<Op>, String> {
    // `MLComputePlan` only accepts a compiled model, and handed a `.mlpackage` it
    // aborts the process from C++ rather than returning an error.
    let path = std::path::Path::new(model);
    if path.extension().and_then(|e| e.to_str()) == Some("mlpackage") {
        eprintln!("compiling {model} first (a compute plan needs a .mlmodelc) ...");
    }
    let ready = bundle::compiled_path(path)?;
    let model = ready
        .to_str()
        .ok_or("the compiled path is not valid UTF-8")?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(model));
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
    // A multi-function bundle plans one function at a time. Without this the
    // plan is built for the default function, and every operation of the one
    // asked about comes back with no device at all — which reads as "placement
    // changed" against a baseline rather than as "nothing was measured".
    unsafe { config.setFunctionName(Some(&NSString::from_str(function))) };

    let plan = await_handler(|h| unsafe {
        MLComputePlan::loadContentsOfURL_configuration_completionHandler(&url, &config, h)
    })?;

    let structure = unsafe { plan.modelStructure() };
    let program = unsafe { structure.program() }
        .ok_or("this model is not an ML Program (a NeuralNetwork or pipeline has no MIL ops)")?;
    let functions = unsafe { program.functions() };
    let f = functions
        .objectForKey(&NSString::from_str(function))
        .ok_or_else(|| {
            let names: Vec<String> = functions.keys().map(|k| k.to_string()).collect();
            format!(
                "no function named `{function}`; this model has: {}",
                names.join(", ")
            )
        })?;

    let mut out = Vec::new();
    for (i, op) in unsafe { f.block().operations() }.iter().enumerate() {
        let op_type = unsafe { op.operatorName() }.to_string();
        // `const` is not scheduled; counting it would make every graph look
        // mostly-CPU regardless of where the real work goes.
        if op_type == "const" || op_type.ends_with(".const") {
            continue;
        }
        let usage = unsafe { plan.computeDeviceUsageForMLProgramOperation(&op) };
        let (preferred, supported) = match &usage {
            Some(u) => {
                let pref = unsafe { u.preferredComputeDevice() };
                let mut sup: Vec<String> = unsafe { u.supportedComputeDevices() }
                    .iter()
                    .map(|d| device_label(&d))
                    .collect();
                sup.sort();
                (device_label(&pref), sup)
            }
            None => ("unknown".to_string(), Vec::new()),
        };
        let cost = unsafe { plan.estimatedCostOfMLProgramOperation(&op) }
            .map(|c| unsafe { c.weight() })
            .unwrap_or(0.0);
        out.push(Op {
            index: i,
            op_type,
            preferred,
            supported,
            cost,
        });
    }
    Ok(out)
}

fn to_tsv(ops: &[Op]) -> String {
    let mut s = String::from("op_type\tpreferred\tsupported\tcost\n");
    for op in ops {
        s.push_str(&format!(
            "{}\t{}\t{}\t{:.6}\n",
            op.op_type,
            op.preferred,
            if op.supported.is_empty() {
                "-".to_string()
            } else {
                op.supported.join("|")
            },
            op.cost
        ));
    }
    s
}

/// Compare against a recorded TSV. Only the columns that describe placement are
/// compared: `cost` is a static estimate that can drift without the assignment
/// changing, so a change in it is not a regression.
fn compare(ops: &[Op], baseline: &str) -> Vec<String> {
    let rows: Vec<(&str, &str, &str)> = baseline
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            let mut c = l.split('\t');
            Some((c.next()?, c.next()?, c.next()?))
        })
        .collect();

    let mut diffs = Vec::new();
    if rows.len() != ops.len() {
        diffs.push(format!(
            "operation count changed: baseline {}, now {}",
            rows.len(),
            ops.len()
        ));
    }
    for (i, (op, row)) in ops.iter().zip(&rows).enumerate() {
        let supported = if op.supported.is_empty() {
            "-".to_string()
        } else {
            op.supported.join("|")
        };
        if op.op_type != row.0 {
            diffs.push(format!(
                "op {i}: type changed, baseline `{}`, now `{}`",
                row.0, op.op_type
            ));
        } else if op.preferred != row.1 {
            diffs.push(format!(
                "op {i} ({}): preferred device changed, baseline {}, now {}",
                op.op_type, row.1, op.preferred
            ));
        } else if supported != row.2 {
            diffs.push(format!(
                "op {i} ({}): supported devices changed, baseline {}, now {supported}",
                op.op_type, row.2
            ));
        }
    }
    diffs
}

fn report(model: &str, function: &str, ops: &[Op]) {
    let total = ops.len();
    let mut per_device: BTreeMap<&str, usize> = BTreeMap::new();
    let mut cost_by_device: BTreeMap<&str, f64> = BTreeMap::new();
    let mut per_type: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    for op in ops {
        *per_device.entry(&op.preferred).or_default() += 1;
        *cost_by_device.entry(&op.preferred).or_default() += op.cost;
        *per_type
            .entry(&op.op_type)
            .or_default()
            .entry(&op.preferred)
            .or_default() += 1;
    }

    let prologue = ops.iter().take_while(|op| op.preferred != "ANE").count();
    let stragglers: Vec<&Op> = ops[prologue.min(total)..]
        .iter()
        .filter(|op| op.preferred != "ANE")
        .collect();
    let ceiling = ops.iter().filter(|op| op.ane_capable()).count();

    println!("model    : {model}  function={function}");
    println!("ops      : {total} (const excluded)");
    println!();
    println!("-- preferred device --");
    for (d, n) in per_device.iter() {
        println!(
            "{d:<8} {n:>5} ops ({:>5.1}%)  estimated cost share {:>6.3}",
            *n as f64 / total as f64 * 100.0,
            cost_by_device.get(*d).copied().unwrap_or(0.0)
        );
    }
    println!();
    println!("-- placement structure --");
    println!(
        "prologue   : {prologue} leading ops off the ANE ({})",
        if prologue == 0 {
            "none".to_string()
        } else {
            ops[..prologue]
                .iter()
                .map(|o| o.op_type.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!(
        "stragglers : {} ANE-capable ops on CPU after the prologue{}",
        stragglers.iter().filter(|o| o.ane_capable()).count(),
        if stragglers.is_empty() {
            " (the graph is handed over once)".to_string()
        } else {
            format!(
                " — at op {}",
                stragglers
                    .iter()
                    .take(8)
                    .map(|o| format!("{} ({})", o.index, o.op_type))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    );
    println!(
        "ceiling    : {ceiling}/{total} ops list ANE as supported ({:.1}%)",
        ceiling as f64 / total as f64 * 100.0
    );
    println!();
    println!("-- by op type --");
    for (t, m) in &per_type {
        let detail: Vec<String> = m.iter().map(|(d, n)| format!("{d}={n}")).collect();
        println!("{t:<24} {}", detail.join(" "));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: computeplan <model.mlmodelc|model.mlpackage> [function] \
             [--tsv out.tsv] [--baseline in.tsv]"
        );
        exit(2);
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model = args[1].clone();
    let function = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "main".to_string());

    let ops = match plan(&model, &function) {
        Ok(ops) => ops,
        Err(e) => {
            eprintln!("computeplan: {e}");
            exit(1);
        }
    };
    if ops.is_empty() {
        eprintln!("computeplan: function `{function}` has no schedulable operations");
        exit(1);
    }
    // CoreML answers "no device" for every operation when it planned something
    // other than what was asked about. Reported as a placement, that reads as
    // "everything moved off the ANE"; against a baseline it reads as every
    // operation having changed. Neither is true — nothing was measured.
    if ops.iter().all(|op| op.preferred == "unknown") {
        eprintln!(
            "computeplan: CoreML returned no device for any of the {} operations in \
             `{function}`, so nothing was measured. This is what a compute plan built \
             for a different function looks like.",
            ops.len()
        );
        exit(1);
    }

    if let Some(path) = flag("--tsv") {
        std::fs::write(&path, to_tsv(&ops)).expect("write TSV");
        eprintln!("wrote {path}");
    }

    match flag("--baseline") {
        None => report(&model, &function, &ops),
        Some(path) => {
            let baseline = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("computeplan: reading baseline {path}: {e}");
                    exit(2);
                }
            };
            let diffs = compare(&ops, &baseline);
            if diffs.is_empty() {
                println!("OK: {} ops place exactly as recorded in {path}", ops.len());
            } else {
                eprintln!("computeplan: {} differences from {path}", diffs.len());
                for d in diffs.iter().take(40) {
                    eprintln!("  {d}");
                }
                if diffs.len() > 40 {
                    eprintln!("  ... and {} more", diffs.len() - 40);
                }
                exit(1);
            }
        }
    }
}
