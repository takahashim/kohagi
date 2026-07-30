//! Building a MIL program.
//!
//! Small on purpose: enough to place typed values and operations into one
//! function's block, with the conventions a real converted model uses. Those were
//! read off `takahashim/ruri-v3-130m-coreml` rather than guessed, and the ones
//! that are not obvious from the schema:
//!
//! - `Program.version` is 1, and the model's `specificationVersion` is 9 for a
//!   macOS 15 deployment target.
//! - The opset is the string `"CoreML8"`, and it is also the key under which the
//!   function stores its one block specialization. (The textual MIL inside a
//!   `.mlmodelc` renders the same thing as `ios18`.)
//! - A function declares its inputs; the block declares only its outputs.
//! - Every operation carries a `name` attribute holding a string tensor, and a
//!   `const` carries its payload in a `val` attribute — either an immediate value
//!   or a reference into `weights/weight.bin` at a blob's *metadata* offset.
//!
//! No shape inference: every value's type is stated where it is created. An
//! operation that disagrees with its
//! inputs is therefore a bug in the caller, and CoreML will say so at compile
//! time rather than producing wrong numbers.

use std::collections::BTreeMap;

use prost::Message;

use crate::coreml_proto::mil_spec::{
    argument, tensor_value, value, Argument, Block, Dimension, Function, NamedValueType, Operation,
    Program, TensorType, TensorValue, Value, ValueType,
};

/// The opset a macOS 15 deployment target uses, and the block-specialization key.
pub const OPSET: &str = "CoreML8";
/// `Model.specificationVersion` for that target.
pub const SPECIFICATION_VERSION: i32 = 9;
/// Where a `const` looks for its bytes, relative to the package.
pub const WEIGHT_FILE: &str = "@model_path/weights/weight.bin";

/// A tensor element type, limited to what this crate emits.
///
/// [`Bool`](Self::Bool) and [`Str`](Self::Str) only ever appear on operation
/// parameters inside the graph (`transpose_x`, `gelu`'s `mode`); a model's inputs
/// and outputs are numeric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Fp16,
    Fp32,
    Int32,
    Int8,
    Bool,
    Str,
}

impl DType {
    fn proto(self) -> i32 {
        use crate::coreml_proto::mil_spec::DataType;
        match self {
            Self::Fp16 => DataType::Float16 as i32,
            Self::Fp32 => DataType::Float32 as i32,
            Self::Int32 => DataType::Int32 as i32,
            Self::Int8 => DataType::Int8 as i32,
            Self::Bool => DataType::Bool as i32,
            Self::Str => DataType::String as i32,
        }
    }
}

/// A named tensor and its type: the unit everything in a block refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(name: impl Into<String>, dtype: DType, shape: &[usize]) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    fn value_type(&self) -> ValueType {
        tensor_type(self.dtype, &self.shape)
    }

    fn declaration(&self) -> NamedValueType {
        NamedValueType {
            name: self.name.clone(),
            r#type: Some(self.value_type()),
        }
    }
}

fn tensor_type(dtype: DType, shape: &[usize]) -> ValueType {
    use crate::coreml_proto::mil_spec::{dimension, value_type};
    ValueType {
        r#type: Some(value_type::Type::TensorType(TensorType {
            data_type: dtype.proto(),
            rank: shape.len() as i64,
            dimensions: shape
                .iter()
                .map(|&d| Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size: d as u64 },
                    )),
                })
                .collect(),
            attributes: BTreeMap::new(),
        })),
    }
}

/// Wrap a payload as an immediate value of the given type.
fn immediate(vtype: ValueType, payload: tensor_value::Value) -> Value {
    use crate::coreml_proto::mil_spec::value::{immediate_value, ImmediateValue};
    Value {
        doc_string: String::new(),
        r#type: Some(vtype),
        value: Some(value::Value::ImmediateValue(ImmediateValue {
            value: Some(immediate_value::Value::Tensor(TensorValue {
                value: Some(payload),
            })),
        })),
    }
}

/// A scalar string, the form every `name` attribute takes.
fn string_value(text: &str) -> Value {
    use crate::coreml_proto::mil_spec::tensor_value::RepeatedStrings;
    immediate(
        tensor_type(DType::Str, &[]),
        tensor_value::Value::Strings(RepeatedStrings {
            values: vec![text.to_string()],
        }),
    )
}

/// An operation input bound to a blob reference directly, rather than to the name
/// of a preceding `const`. `constexpr_blockwise_shift_scale` takes its operands
/// this way.
fn bound_blob(vtype: ValueType, offset: u64) -> Argument {
    use crate::coreml_proto::mil_spec::value::BlobFileValue;
    Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Value(Value {
                doc_string: String::new(),
                r#type: Some(vtype),
                value: Some(value::Value::BlobFileValue(BlobFileValue {
                    file_name: WEIGHT_FILE.to_string(),
                    offset,
                })),
            })),
        }],
    }
}

/// One operation input: a list of argument bindings. Every input this crate emits
/// binds exactly one name.
fn bound(name: &str) -> Argument {
    Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Name(name.to_string())),
        }],
    }
}

/// Collects operations for one function's block.
pub struct Builder {
    inputs: Vec<Tensor>,
    ops: Vec<Operation>,
    outputs: Vec<String>,
    /// Immediate `const`s already emitted, keyed by their encoded payload, so a
    /// value used by every layer is written once.
    ///
    /// This was meant to bring the emitted MIL closer to the reference conversion,
    /// which has 804 `const` operations where a naive emit has 929. It overshot:
    /// pooling every repeated axis, flag and shape across 19 blocks leaves 261, so
    /// the two still differ, in the other direction. What it does buy is a smaller
    /// specification — 263KB to 167KB for `ruri-v3-130m` at sequence length 128 —
    /// which is noise beside a 264MB blob. The output stays bit-identical to the
    /// reference either way.
    ///
    /// Kept because a smaller specification is worth having for free, not because it
    /// achieved what it was written for.
    ///
    /// Blob-backed `const`s are not pooled: two of them differ only by offset, and
    /// what goes into the blob is the caller's decision.
    interned: BTreeMap<Vec<u8>, Tensor>,
}

impl Builder {
    /// A function taking `inputs`. Their names are what a caller feeds at
    /// prediction time.
    pub fn new(inputs: &[Tensor]) -> Self {
        Self {
            inputs: inputs.to_vec(),
            ops: Vec::new(),
            outputs: Vec::new(),
            interned: BTreeMap::new(),
        }
    }

    /// An immediate `const` holding fp32 values. Small literals only: anything
    /// weight-sized belongs in the blob file, where it is memory-mapped rather
    /// than parsed.
    pub fn const_f32(&mut self, out: Tensor, values: &[f32]) -> Tensor {
        use crate::coreml_proto::mil_spec::tensor_value::RepeatedFloats;
        self.check_count(&out, values.len());
        let val = immediate(
            out.value_type(),
            tensor_value::Value::Floats(RepeatedFloats {
                values: values.to_vec(),
            }),
        );
        self.push_const(out, val)
    }

    /// A `const` whose bytes live in `weights/weight.bin`. `offset` is the blob's
    /// **metadata** offset, which is what [`super::blob::Writer::write`] returns.
    pub fn const_blob(&mut self, out: Tensor, offset: u64) -> Tensor {
        use crate::coreml_proto::mil_spec::value::BlobFileValue;
        let val = Value {
            doc_string: String::new(),
            r#type: Some(out.value_type()),
            value: Some(value::Value::BlobFileValue(BlobFileValue {
                file_name: WEIGHT_FILE.to_string(),
                offset,
            })),
        };
        self.push_const(out, val)
    }

    /// An immediate `const` holding int32 values, which is how MIL passes axes,
    /// dimensions and other small integer parameters.
    pub fn const_i32(&mut self, out: Tensor, values: &[i32]) -> Tensor {
        use crate::coreml_proto::mil_spec::tensor_value::RepeatedInts;
        self.check_count(&out, values.len());
        let val = immediate(
            out.value_type(),
            tensor_value::Value::Ints(RepeatedInts {
                values: values.to_vec(),
            }),
        );
        self.push_const(out, val)
    }

    /// An immediate fp16 `const`. The payload is raw little-endian fp16 bytes
    /// rather than a float list: that is how a converted model stores `epsilon`
    /// and other half-precision literals, and `TensorValue` has no fp16 case.
    pub fn const_fp16(&mut self, out: Tensor, values: &[f32]) -> Tensor {
        use crate::coreml_proto::mil_spec::tensor_value::RepeatedBytes;
        self.check_count(&out, values.len());
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let val = immediate(
            out.value_type(),
            tensor_value::Value::Bytes(RepeatedBytes { values: raw }),
        );
        self.push_const(out, val)
    }

    /// A boolean `const` array, as `slice_by_index`'s `end_mask`.
    pub fn const_bools(&mut self, out: Tensor, values: &[bool]) -> Tensor {
        use crate::coreml_proto::mil_spec::tensor_value::RepeatedBools;
        self.check_count(&out, values.len());
        let val = immediate(
            out.value_type(),
            tensor_value::Value::Bools(RepeatedBools {
                values: values.to_vec(),
            }),
        );
        self.push_const(out, val)
    }

    /// An fp16 `const` from a value that has no `f32` spelling, such as an
    /// infinity: `f16::from_f32(f32::NEG_INFINITY)` is fine, but going through
    /// `f32` for a bit pattern that is already `f16` is a detour worth avoiding.
    pub fn const_fp16_bits(&mut self, out: Tensor, value: half::f16) -> Tensor {
        use crate::coreml_proto::mil_spec::tensor_value::RepeatedBytes;
        self.check_count(&out, 1);
        let val = immediate(
            out.value_type(),
            tensor_value::Value::Bytes(RepeatedBytes {
                values: value.to_bits().to_le_bytes().to_vec(),
            }),
        );
        self.push_const(out, val)
    }

    /// A boolean `const` of a declared shape, for a mask rather than a flag list.
    pub fn const_bools_shaped(&mut self, out: Tensor, values: &[bool]) -> Tensor {
        self.const_bools(out, values)
    }

    /// An immediate int8 `const`, for a quantization zero point. Like fp16, the
    /// payload is raw bytes: `TensorValue` has no int8 case, and writing an `ints`
    /// list under an int8 type is rejected as "storage and type have different
    /// number of elements".
    pub fn const_int8(&mut self, out: Tensor, values: &[i8]) -> Tensor {
        use crate::coreml_proto::mil_spec::tensor_value::RepeatedBytes;
        self.check_count(&out, values.len());
        let val = immediate(
            out.value_type(),
            tensor_value::Value::Bytes(RepeatedBytes {
                values: values.iter().map(|&v| v as u8).collect(),
            }),
        );
        self.push_const(out, val)
    }

    /// A scalar boolean `const`, as an operation flag like `transpose_x`.
    pub fn const_bool(&mut self, name: &str, value: bool) -> Tensor {
        use crate::coreml_proto::mil_spec::tensor_value::RepeatedBools;
        let out = Tensor::new(name, DType::Bool, &[]);
        let val = immediate(
            out.value_type(),
            tensor_value::Value::Bools(RepeatedBools {
                values: vec![value],
            }),
        );
        self.push_const(out, val)
    }

    /// A scalar string `const`, as an operation mode like `gelu`'s `EXACT`.
    pub fn const_str(&mut self, name: &str, text: &str) -> Tensor {
        let out = Tensor::new(name, DType::Str, &[]);
        let val = string_value(text);
        self.push_const(out, val)
    }

    fn check_count(&self, out: &Tensor, given: usize) {
        assert_eq!(
            out.elements(),
            given,
            "const {} declares {:?} ({} elements) but was given {given}",
            out.name,
            out.shape,
            out.elements()
        );
    }

    fn push_const(&mut self, out: Tensor, val: Value) -> Tensor {
        // An immediate is identified by its type and payload; the name is not part
        // of it, so two layers asking for `axes = [-1]` get the same value back.
        let key = matches!(val.value, Some(value::Value::ImmediateValue(_))).then(|| {
            let mut key = val.encode_to_vec();
            key.extend(out.dtype.proto().to_le_bytes());
            key.extend(out.shape.iter().flat_map(|d| d.to_le_bytes()));
            key
        });
        if let Some(key) = &key {
            if let Some(existing) = self.interned.get(key) {
                return existing.clone();
            }
        }

        let mut attributes = BTreeMap::new();
        attributes.insert("name".to_string(), string_value(&out.name));
        attributes.insert("val".to_string(), val);
        self.ops.push(Operation {
            r#type: "const".to_string(),
            inputs: BTreeMap::new(),
            outputs: vec![out.declaration()],
            blocks: Vec::new(),
            attributes,
        });
        if let Some(key) = key {
            self.interned.insert(key, out.clone());
        }
        out
    }

    /// Any single-output operation. `inputs` are `(parameter name, value name)`
    /// pairs, as MIL binds arguments by parameter rather than position.
    pub fn op(&mut self, kind: &str, out: Tensor, inputs: &[(&str, &Tensor)]) -> Tensor {
        let bindings = inputs
            .iter()
            .map(|(param, value)| ((*param).to_string(), bound(&value.name)))
            .collect();
        self.push_op(kind, bindings, vec![out.clone()], &out.name);
        out
    }

    /// An operation one of whose inputs takes a list of values, as `concat`'s
    /// `values` does. MIL expresses that as one argument with several bindings
    /// rather than several arguments.
    pub fn op_variadic(
        &mut self,
        kind: &str,
        out: Tensor,
        inputs: &[(&str, &Tensor)],
        variadic: (&str, &[&Tensor]),
    ) -> Tensor {
        let mut bindings = BTreeMap::new();
        for (param, value) in inputs {
            bindings.insert((*param).to_string(), bound(&value.name));
        }
        bindings.insert(
            variadic.0.to_string(),
            Argument {
                arguments: variadic
                    .1
                    .iter()
                    .map(|t| argument::Binding {
                        binding: Some(argument::binding::Binding::Name(t.name.clone())),
                    })
                    .collect(),
            },
        );
        self.push_op(kind, bindings, vec![out.clone()], &out.name);
        out
    }

    /// An operation with several outputs, as `split` has. Their order is the
    /// operation's, so a caller reads them positionally.
    pub fn op_multi_output(
        &mut self,
        kind: &str,
        outs: &[Tensor],
        inputs: &[(&str, &Tensor)],
    ) -> Vec<Tensor> {
        assert!(
            outs.len() > 1,
            "{kind} was given {} outputs; use op() for one",
            outs.len()
        );
        let bindings = inputs
            .iter()
            .map(|(param, value)| ((*param).to_string(), bound(&value.name)))
            .collect();
        // The operation's `name` attribute names the operation, not each output;
        // the reference model uses the first output's stem for it.
        let name = outs[0].name.clone();
        self.push_op(kind, bindings, outs.to_vec(), &name);
        outs.to_vec()
    }

    fn push_op(
        &mut self,
        kind: &str,
        inputs: BTreeMap<String, Argument>,
        outs: Vec<Tensor>,
        name: &str,
    ) {
        let mut attributes = BTreeMap::new();
        attributes.insert("name".to_string(), string_value(name));
        self.ops.push(Operation {
            r#type: kind.to_string(),
            inputs,
            outputs: outs.iter().map(Tensor::declaration).collect(),
            blocks: Vec::new(),
            attributes,
        });
    }

    /// An int8 weight the graph dequantizes at load, as
    /// `constexpr_affine_dequantize`.
    ///
    /// Its arguments are **attributes**, not inputs: a `constexpr_*` operation is a
    /// compile-time constant expression, so its operands are baked in the way
    /// `const`'s `val` is. Passing them as bound inputs is rejected with
    /// "Attribute quantized_data is undefined", which is how that was established —
    /// no published bundle uses this operation to copy from.
    ///
    /// `offset` is the int8 blob's metadata offset. One `scale` is per-tensor; a
    /// vector is per-channel along `axis`, which is what a large embedding table
    /// wants — one scale for 50 million values would throw away most of the range.
    pub fn dequantize_int8(
        &mut self,
        out: Tensor,
        offset: u64,
        zero_point: i8,
        scales: &[f32],
        axis: i32,
    ) -> Tensor {
        use crate::coreml_proto::mil_spec::tensor_value::{RepeatedBytes, RepeatedInts};
        use crate::coreml_proto::mil_spec::value::BlobFileValue;

        let quantized = Value {
            doc_string: String::new(),
            r#type: Some(tensor_type(DType::Int8, &out.shape)),
            value: Some(value::Value::BlobFileValue(BlobFileValue {
                file_name: WEIGHT_FILE.to_string(),
                offset,
            })),
        };
        // `zero_point` and `scale` have to agree in rank: a per-channel scale needs
        // a zero point per channel too.
        let channels = scales.len();
        let shape: &[usize] = if channels == 1 { &[] } else { &[channels] };
        let zero = immediate(
            tensor_type(DType::Int8, shape),
            tensor_value::Value::Bytes(RepeatedBytes {
                values: vec![zero_point as u8; channels],
            }),
        );
        let scale = immediate(
            tensor_type(out.dtype, shape),
            tensor_value::Value::Bytes(RepeatedBytes {
                values: scales
                    .iter()
                    .flat_map(|&s| half::f16::from_f32(s).to_bits().to_le_bytes())
                    .collect(),
            }),
        );
        let axis = immediate(
            tensor_type(DType::Int32, &[]),
            tensor_value::Value::Ints(RepeatedInts { values: vec![axis] }),
        );

        let mut attributes = BTreeMap::new();
        attributes.insert("name".to_string(), string_value(&out.name));
        attributes.insert("quantized_data".to_string(), quantized);
        attributes.insert("zero_point".to_string(), zero);
        attributes.insert("scale".to_string(), scale);
        attributes.insert("axis".to_string(), axis);
        self.ops.push(Operation {
            r#type: "constexpr_affine_dequantize".to_string(),
            inputs: BTreeMap::new(),
            outputs: vec![out.declaration()],
            blocks: Vec::new(),
            attributes,
        });
        out
    }

    /// An int8 weight dequantized blockwise, as iOS18's
    /// `constexpr_blockwise_shift_scale`.
    ///
    /// The scale has the same rank as the data with each axis dividing it: a
    /// `[256, 32]` scale over a `[256, 512]` weight is one scale per 16 values along
    /// the last axis. Per-channel (a `[out, 1]` scale) is the special case, so this
    /// subsumes [`Self::dequantize_int8`]; the older `constexpr_affine_dequantize`
    /// stays because it is what the reference conversion emits.
    ///
    /// **Its operands are inputs, not attributes** — the opposite of
    /// `constexpr_affine_dequantize` — and they are bound as values rather than by
    /// name, so the two blob references sit inside the operation itself with no
    /// `const` in front of them. Writing them as attributes instead does not produce
    /// an error: it crashes `coremlc` with SIGSEGV. That was established by
    /// quantizing a small model with `coremltools.optimize` and reading the operation
    /// back, after guessing had cost a segfault.
    ///
    /// Both offsets are blob metadata offsets: the scale lives in `weight.bin` too,
    /// not as an immediate.
    pub fn dequantize_blockwise(
        &mut self,
        out: Tensor,
        data_offset: u64,
        scale_shape: &[usize],
        scale_offset: u64,
    ) -> Tensor {
        assert_eq!(
            scale_shape.len(),
            out.shape.len(),
            "a blockwise scale has the same rank as the weight"
        );
        for (axis, (&dim, &blocks)) in out.shape.iter().zip(scale_shape).enumerate() {
            assert!(
                blocks > 0 && dim % blocks == 0,
                "axis {axis}: {dim} does not divide into {blocks} blocks"
            );
        }

        let mut attributes = BTreeMap::new();
        attributes.insert("name".to_string(), string_value(&out.name));
        let inputs = [
            ("data", tensor_type(DType::Int8, &out.shape), data_offset),
            ("scale", tensor_type(out.dtype, scale_shape), scale_offset),
        ]
        .into_iter()
        .map(|(name, vtype, offset)| (name.to_string(), bound_blob(vtype, offset)))
        .collect();
        self.ops.push(Operation {
            r#type: "constexpr_blockwise_shift_scale".to_string(),
            inputs,
            outputs: vec![out.declaration()],
            blocks: Vec::new(),
            attributes,
        });
        out
    }

    /// Mark a value as one of the function's results.
    pub fn returns(&mut self, tensor: &Tensor) {
        self.outputs.push(tensor.name.clone());
    }

    /// Wrap the block into a one-function program.
    pub fn finish(self) -> Program {
        assert!(
            !self.outputs.is_empty(),
            "a function with no outputs cannot be predicted from"
        );
        let block = Block {
            inputs: Vec::new(),
            outputs: self.outputs,
            attributes: BTreeMap::new(),
            operations: self.ops,
        };
        let function = Function {
            inputs: self.inputs.iter().map(Tensor::declaration).collect(),
            opset: OPSET.to_string(),
            block_specializations: [(OPSET.to_string(), block)].into_iter().collect(),
            attributes: BTreeMap::new(),
        };
        Program {
            version: 1,
            functions: [("main".to_string(), function)].into_iter().collect(),
            doc_string: String::new(),
            attributes: BTreeMap::new(),
        }
    }

    /// The inputs this function declares, for building the model description.
    pub fn declared_inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_program_carries_the_conventions_coreml_expects() {
        let x = Tensor::new("x", DType::Fp16, &[1, 3]);
        let mut b = Builder::new(std::slice::from_ref(&x));
        let y = b.op(
            "identity",
            Tensor::new("y", DType::Fp16, &[1, 3]),
            &[("x", &x)],
        );
        b.returns(&y);
        let program = b.finish();

        assert_eq!(program.version, 1);
        let f = program.functions.get("main").expect("a main function");
        assert_eq!(f.opset, OPSET);
        // The block lives under the opset key, not an arbitrary one.
        assert_eq!(f.block_specializations.keys().collect::<Vec<_>>(), [OPSET]);
        assert_eq!(f.inputs.len(), 1);
        let block = &f.block_specializations[OPSET];
        // Inputs are declared on the function; the block declares only results.
        assert!(block.inputs.is_empty());
        assert_eq!(block.outputs, ["y"]);
        assert_eq!(block.operations.len(), 1);
        assert_eq!(block.operations[0].r#type, "identity");
        assert!(block.operations[0].attributes.contains_key("name"));
    }

    #[test]
    fn a_blob_const_points_at_the_weight_file() {
        let mut b = Builder::new(&[]);
        let w = b.const_blob(Tensor::new("w", DType::Fp16, &[2, 3]), 4096);
        b.returns(&w);
        let program = b.finish();
        let block = &program.functions["main"].block_specializations[OPSET];
        let val = block.operations[0].attributes.get("val").expect("a val");
        match &val.value {
            Some(value::Value::BlobFileValue(bf)) => {
                assert_eq!(bf.file_name, WEIGHT_FILE);
                assert_eq!(bf.offset, 4096);
            }
            other => panic!("expected a blob reference, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "declares [2, 3] (6 elements) but was given 5")]
    fn an_immediate_const_that_disagrees_with_its_shape_is_caught_here() {
        // Not left to CoreML: a shape it cannot check is a wrong answer, and one
        // it can is a compile error with no line number.
        let mut b = Builder::new(&[]);
        b.const_f32(Tensor::new("w", DType::Fp32, &[2, 3]), &[1.0; 5]);
    }
}
