//! Operation inventories for a MIL program, in either of the two forms it is
//! stored in.
//!
//! A `.mlmodelc` keeps its MIL as text (`model.mil`); a `.mlpackage` keeps it as
//! protobuf (`Data/com.apple.CoreML/model.mlmodel`). The two are **not** the same
//! program: the package holds what the converter wrote, and the compiled bundle
//! holds what `coremlc` made of it. An emitter has to produce the former, so
//! reading only the latter would be writing to the wrong target.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Operation counts keyed by type, plus where they came from.
#[derive(Debug, Clone)]
pub struct Inventory {
    pub source: PathBuf,
    pub form: Form,
    pub counts: BTreeMap<String, usize>,
    /// Operation types in program order. Counts can match while the programs
    /// differ, so the order is kept as the stronger comparison.
    pub sequence: Vec<String>,
    /// `weight.bin` references, only visible in the text form.
    pub blob_refs: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// `model.mil` inside a `.mlmodelc`: post-`coremlc`, what CoreML runs.
    CompiledText,
    /// `model.mlmodel` inside a `.mlpackage`: what the converter wrote.
    PackageProto,
}

impl Form {
    pub fn label(&self) -> &'static str {
        match self {
            Self::CompiledText => "compiled MIL text (post-coremlc)",
            Self::PackageProto => "package protobuf (as written by the converter)",
        }
    }
}

impl Inventory {
    pub fn total(&self) -> usize {
        self.counts.values().sum()
    }

    /// `const` is not a scheduled operation; separating it keeps a graph's real
    /// size from being buried under its literals.
    pub fn consts(&self) -> usize {
        self.counts
            .iter()
            .filter(|(k, _)| *k == "const" || k.ends_with(".const"))
            .map(|(_, v)| *v)
            .sum()
    }

    pub fn compute(&self) -> usize {
        self.total() - self.consts()
    }

    pub fn kinds(&self) -> usize {
        self.counts.len()
    }
}

/// Resolve a path a caller might reasonably pass: a bundle directory, or the
/// MIL file itself.
pub fn locate(path: &Path) -> Option<(PathBuf, Form)> {
    if path.is_file() {
        return match path.extension().and_then(|e| e.to_str()) {
            Some("mil") => Some((path.to_path_buf(), Form::CompiledText)),
            Some("mlmodel") => Some((path.to_path_buf(), Form::PackageProto)),
            _ => None,
        };
    }
    for (rel, form) in [
        ("model.mil", Form::CompiledText),
        ("Data/com.apple.CoreML/model.mlmodel", Form::PackageProto),
    ] {
        let p = path.join(rel);
        if p.is_file() {
            return Some((p, form));
        }
    }
    None
}

pub fn read(path: &Path) -> Result<Inventory, String> {
    let (file, form) = locate(path)
        .ok_or_else(|| format!("{} holds no model.mil or model.mlmodel", path.display()))?;
    let bytes = std::fs::read(&file).map_err(|e| format!("reading {}: {e}", file.display()))?;
    let (sequence, blob_refs) = match form {
        Form::CompiledText => {
            let text = String::from_utf8_lossy(&bytes);
            (from_text(&text), Some(text.matches("BLOBFILE").count()))
        }
        Form::PackageProto => (from_proto(&bytes)?, None),
    };
    if sequence.is_empty() {
        return Err(format!("{} yielded no operations", file.display()));
    }
    Ok(Inventory {
        source: file,
        form,
        counts: tally(&sequence),
        sequence,
        blob_refs,
    })
}

pub fn tally(sequence: &[String]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for op in sequence {
        *counts.entry(op.clone()).or_default() += 1;
    }
    counts
}

/// Operations in MIL text, in program order.
///
/// Every statement is one line of the form
/// `<type> <name> = <op>(<args>)[name = string("...")];`, so the operation is
/// the **first** `= <ident>(` on the line. Taking any match instead would also
/// count the type constructors inside the arguments (`val = string(...)`,
/// `val = tensor<...>(...)`), which outnumber the operations.
pub fn from_text(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(first_call)
        .map(str::to_string)
        .collect()
}

/// The identifier in the first `= <ident>(` of a line, if any.
fn first_call(line: &str) -> Option<&str> {
    let mut rest = line;
    while let Some(eq) = rest.find('=') {
        let after = rest[eq + 1..].trim_start();
        let ident: &str = after
            .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
            .next()
            .unwrap_or("");
        if !ident.is_empty() && after[ident.len()..].starts_with('(') {
            return Some(ident);
        }
        rest = &rest[eq + 1..];
    }
    None
}

/// Field-number path to `Operation.type` in a serialized CoreML model:
///
/// ```text
/// 502  Model.mlProgram
///   2  Program.functions          (map entry)
///     2  map value                (Function)
///       3  Function.block_specializations  (map entry)
///         2  map value            (Block)
///           3  Block.operations
///             1  Operation.type
/// ```
///
/// Read off the wire rather than taken on faith: walking `model.mlmodel` and
/// histogramming every string by its field path leaves exactly one path whose
/// values are all MIL operation names, and it is this one. The check below
/// refuses to guess if a future format moves it.
const OP_TYPE_PATH: &[u32] = &[502, 2, 2, 3, 2, 3, 1];

/// Count operations in a serialized model, by walking the protobuf wire format
/// without its schema.
///
/// Only field numbers are needed, so there is no generated code here and no
/// dependency on a `.proto` file. Nested operations (an `Operation` holding
/// `Block`s, as a control-flow op does) are not reached by the flat path; a
/// ModernBERT encoder has none, and [`from_proto`] says so rather than
/// pretending otherwise when the totals disagree with the declared structure.
pub fn from_proto(bytes: &[u8]) -> Result<Vec<String>, String> {
    let mut ops = Vec::new();
    walk(bytes, &mut Vec::new(), 0, &mut |path, value| {
        if path == OP_TYPE_PATH {
            ops.push(value.to_string());
        }
    });
    if ops.is_empty() {
        return Err(format!(
            "no strings at the Operation.type path {OP_TYPE_PATH:?}; this is not an \
             ML Program, or the format moved and the path needs re-deriving \
             (see tools/coreml-jigs/src/mil.rs)"
        ));
    }
    Ok(ops)
}

fn varint(b: &[u8], i: &mut usize) -> Option<u64> {
    let mut r = 0u64;
    let mut shift = 0;
    loop {
        let c = *b.get(*i)?;
        *i += 1;
        r |= u64::from(c & 0x7f) << shift;
        if c & 0x80 == 0 {
            return Some(r);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

/// Walk protobuf wire format, calling `on_string` for every length-delimited
/// field that is not itself a readable submessage.
///
/// Length-delimited fields are ambiguous on the wire: bytes, string and
/// submessage share a type. The rule here is to try a submessage first and fall
/// back to a string, which is what makes a schema unnecessary. It can misread a
/// string that happens to be valid wire format, so a path is only trusted after
/// its values have been checked (see [`OP_TYPE_PATH`]).
fn walk(
    b: &[u8],
    path: &mut Vec<u32>,
    depth: usize,
    on_string: &mut impl FnMut(&[u32], &str),
) -> bool {
    let mut i = 0;
    while i < b.len() {
        let Some(key) = varint(b, &mut i) else {
            return false;
        };
        let (field, wire) = ((key >> 3) as u32, key & 7);
        if field == 0 {
            return false;
        }
        match wire {
            0 => {
                if varint(b, &mut i).is_none() {
                    return false;
                }
            }
            1 => i += 8,
            5 => i += 4,
            2 => {
                let Some(len) = varint(b, &mut i) else {
                    return false;
                };
                let len = len as usize;
                if i + len > b.len() {
                    return false;
                }
                let sub = &b[i..i + len];
                i += len;
                path.push(field);
                let as_message =
                    depth < 12 && !sub.is_empty() && walk(sub, path, depth + 1, on_string);
                if !as_message {
                    if let Ok(text) = std::str::from_utf8(sub) {
                        if !text.is_empty() && text.chars().all(|c| !c.is_control()) {
                            on_string(path, text);
                        }
                    }
                }
                path.pop();
            }
            _ => return false,
        }
        if i > b.len() {
            return false;
        }
    }
    true
}

/// One row of a two-inventory comparison.
pub struct Delta {
    pub op: String,
    pub a: usize,
    pub b: usize,
}

/// Where the two programs' operation sequences first disagree, or `None` if they
/// are the same sequence. A stronger statement than matching counts: the same
/// multiset of operations can be wired into a different program.
pub fn first_divergence(a: &Inventory, b: &Inventory) -> Option<(usize, String, String)> {
    let none = "<end>".to_string();
    for i in 0..a.sequence.len().max(b.sequence.len()) {
        let (x, y) = (a.sequence.get(i), b.sequence.get(i));
        if x != y {
            return Some((
                i,
                x.cloned().unwrap_or_else(|| none.clone()),
                y.cloned().unwrap_or(none),
            ));
        }
    }
    None
}

pub fn diff(a: &Inventory, b: &Inventory) -> Vec<Delta> {
    let mut ops: Vec<&String> = a.counts.keys().chain(b.counts.keys()).collect();
    ops.sort();
    ops.dedup();
    ops.into_iter()
        .map(|op| Delta {
            op: op.clone(),
            a: a.counts.get(op).copied().unwrap_or(0),
            b: b.counts.get(op).copied().unwrap_or(0),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"program(1.3)
{
    func main<ios18>(tensor<int32, [1, 128]> attention_mask, tensor<int32, [1, 128]> input_ids) {
            int32 var_6 = const()[name = string("op_6"), val = int32(-1)];
            tensor<int32, [1]> var_71_axes_0 = const()[name = string("op_71_axes_0"), val = tensor<int32, [1]>([1])];
            tensor<int32, [1, 1, 128]> var_71 = expand_dims(axes = var_71_axes_0, x = attention_mask)[name = string("op_71")];
            tensor<fp16, [512]> w = const()[name = string("w"), val = tensor<fp16, [512]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
            tensor<fp16, [1, 128, 512]> h = layer_norm(axes = a, epsilon = e, gamma = w, x = x)[name = string("h")];
            tensor<fp16, [1, 128, 2048]> p, tensor<fp16, [1, 128, 2048]> q = split(axis = ax, x = y)[name = string("s")];
    } -> (h);
}"#;

    #[test]
    fn text_counts_operations_not_type_constructors() {
        let counts = tally(&from_text(SAMPLE));
        // Three consts, and one each of the rest. `string(...)`, `int32(...)`,
        // `uint64(...)` and `tensor<...>(...)` inside the arguments must not be
        // counted: they outnumber the operations several times over.
        assert_eq!(counts.get("const"), Some(&3));
        assert_eq!(counts.get("expand_dims"), Some(&1));
        assert_eq!(counts.get("layer_norm"), Some(&1));
        assert_eq!(counts.get("split"), Some(&1));
        assert_eq!(counts.get("string"), None);
        assert_eq!(counts.get("uint64"), None);
        assert_eq!(counts.len(), 4);
    }

    #[test]
    fn a_multi_output_statement_counts_once() {
        // `a, b = split(...)` is one operation, and the op is the first call on
        // the line even with a comma-separated left side.
        assert_eq!(
            first_call("tensor<fp16, [2]> p, tensor<fp16, [2]> q = split(axis = ax)[name = x];"),
            Some("split")
        );
        assert_eq!(first_call("} -> (hidden);"), None);
        assert_eq!(first_call("program(1.3)"), None);
    }

    fn inv(sequence: Vec<String>, form: Form) -> Inventory {
        Inventory {
            source: PathBuf::from("x"),
            form,
            counts: tally(&sequence),
            sequence,
            blob_refs: None,
        }
    }

    #[test]
    fn inventory_separates_consts_from_work() {
        let inv = Inventory {
            source: PathBuf::from("x"),
            form: Form::CompiledText,
            counts: tally(&from_text(SAMPLE)),
            sequence: from_text(SAMPLE),
            blob_refs: Some(1),
        };
        assert_eq!(inv.total(), 6);
        assert_eq!(inv.consts(), 3);
        assert_eq!(inv.compute(), 3);
        assert_eq!(inv.kinds(), 4);
    }

    /// A length-delimited field, hand-encoded.
    fn field(num: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut key = u64::from(num) << 3 | 2;
        loop {
            let mut byte = (key & 0x7f) as u8;
            key >>= 7;
            if key != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if key == 0 {
                break;
            }
        }
        let mut len = payload.len() as u64;
        loop {
            let mut byte = (len & 0x7f) as u8;
            len >>= 7;
            if len != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if len == 0 {
                break;
            }
        }
        out.extend_from_slice(payload);
        out
    }

    /// Wrap `op_types` at OP_TYPE_PATH, innermost first.
    fn synth(op_types: &[&str]) -> Vec<u8> {
        let mut ops = Vec::new();
        for t in op_types {
            // Operation { type = 1 }, then Block.operations = 3
            ops.extend(field(3, &field(1, t.as_bytes())));
        }
        let block_map = field(2, &ops); // map value -> Block
        let function = field(3, &block_map); // Function.block_specializations
        let fn_map = field(2, &function); // map value -> Function
        let program = field(2, &fn_map); // Program.functions
        field(502, &program) // Model.mlProgram
    }

    #[test]
    fn proto_walk_finds_operation_types() {
        let bytes = synth(&["const", "linear", "linear", "layer_norm"]);
        let counts = tally(&from_proto(&bytes).expect("finds the op path"));
        assert_eq!(counts.get("linear"), Some(&2));
        assert_eq!(counts.get("const"), Some(&1));
        assert_eq!(counts.get("layer_norm"), Some(&1));
        assert_eq!(counts.len(), 3);
    }

    #[test]
    fn proto_walk_refuses_to_guess_on_a_foreign_model() {
        // A neural-network model has no field 502 at all; reporting zero ops
        // would read as "an empty graph" rather than "wrong kind of model".
        let bytes = field(500, &field(1, b"someLayer"));
        assert!(from_proto(&bytes).is_err());
    }

    fn ops(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn diff_lines_up_both_sides_including_ops_only_one_has() {
        let a = inv(
            ops(&["linear", "linear", "cast", "cast", "cast", "cast", "cast"]),
            Form::PackageProto,
        );
        let b = inv(ops(&["linear", "linear", "squeeze"]), Form::CompiledText);
        let d = diff(&a, &b);
        let by_op: BTreeMap<&str, (usize, usize)> =
            d.iter().map(|x| (x.op.as_str(), (x.a, x.b))).collect();
        assert_eq!(by_op["linear"], (2, 2));
        assert_eq!(by_op["cast"], (5, 0));
        assert_eq!(by_op["squeeze"], (0, 1));
    }

    #[test]
    fn identical_counts_can_still_be_different_programs() {
        // The point of keeping the order: these two have the same inventory.
        let a = inv(ops(&["linear", "gelu", "linear"]), Form::PackageProto);
        let b = inv(ops(&["linear", "linear", "gelu"]), Form::PackageProto);
        assert!(diff(&a, &b).iter().all(|d| d.a == d.b));
        assert_eq!(
            first_divergence(&a, &b),
            Some((1, "gelu".to_string(), "linear".to_string()))
        );
        assert_eq!(first_divergence(&a, &a), None);
    }

    #[test]
    fn a_truncated_sequence_diverges_at_the_end() {
        let a = inv(ops(&["linear", "gelu"]), Form::PackageProto);
        let b = inv(ops(&["linear"]), Form::PackageProto);
        assert_eq!(
            first_divergence(&a, &b),
            Some((1, "gelu".to_string(), "<end>".to_string()))
        );
    }
}
