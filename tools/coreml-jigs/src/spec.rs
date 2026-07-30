//! A model's declared inputs, outputs and provenance, read straight out of a
//! `.mlpackage`'s `model.mlmodel`.
//!
//! `MLModelAsset` only opens compiled models, so the portable form — the one a
//! publisher actually checks before uploading — cannot be described through the
//! framework without compiling it first, at ~20 s per bucket. The protobuf holds
//! everything needed, so it is decoded here instead.
//!
//! This is a targeted reader, not a generated one: six field numbers, each
//! confirmed against a real `model.mlmodel` rather than assumed. Descending
//! explicitly (rather than flattening every field path) is what keeps a feature's
//! name attached to its own shape.
//!
//! ```text
//! Model
//!   2  description  (ModelDescription)
//!      1  input     (repeated FeatureDescription)
//!     10  output    (repeated FeatureDescription)
//!    100  metadata  (Metadata)
//!       100  userDefined (map<string, string>)
//!            1 key   2 value
//!     20  functions (repeated FunctionDescription)
//!     21  defaultFunctionName (string)
//!  502  mlProgram    (Program)
//!       2  functions (map<string, Function>) — 1 key
//!
//! FunctionDescription     1 name   2 input   3 output
//! FeatureDescription      1 name   3 type
//! FeatureType             5 multiArrayType
//! ArrayFeatureType        1 shape (repeated int64)   2 dataType
//! ```

/// One declared input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Feature {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: u64,
}

/// One function of a multi-function model, with its own interface.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub inputs: Vec<Feature>,
    pub outputs: Vec<Feature>,
}

#[derive(Debug, Clone, Default)]
pub struct Spec {
    /// The top-level interface. Empty for a multi-function model, which describes
    /// each function separately in [`Self::described_functions`] instead.
    pub inputs: Vec<Feature>,
    pub outputs: Vec<Feature>,
    /// Per-function interfaces, from `ModelDescription.functions`.
    pub described_functions: Vec<Function>,
    pub default_function: String,
    /// `userDefined` metadata, which for a converted model records the toolchain
    /// that produced it. A model card is expected to state those versions in
    /// the model card, and this is where they can be read back off the artifact.
    pub metadata: Vec<(String, String)>,
    /// MIL function names. Empty for a single-function model, whose only function
    /// is the default one.
    pub functions: Vec<String>,
}

/// `MLMultiArrayDataType` is a bit pattern: `0x10000` for float or `0x20000` for
/// int, or-ed with the width in bits.
pub fn dtype_name(d: u64) -> String {
    match d {
        65552 => "fp16".to_string(),
        65568 => "fp32".to_string(),
        65600 => "fp64".to_string(),
        131104 => "int32".to_string(),
        other => format!("dtype {other}"),
    }
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }

    fn varint(&mut self) -> Option<u64> {
        let mut r = 0u64;
        let mut shift = 0;
        loop {
            let c = *self.b.get(self.i)?;
            self.i += 1;
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

    /// Next `(field number, value)`, skipping the wire types this format does not
    /// use for anything read here.
    fn next(&mut self) -> Option<(u32, Value<'a>)> {
        while self.i < self.b.len() {
            let key = self.varint()?;
            let (field, wire) = ((key >> 3) as u32, key & 7);
            if field == 0 {
                return None;
            }
            match wire {
                0 => return Some((field, Value::Varint(self.varint()?))),
                1 => self.i += 8,
                5 => self.i += 4,
                2 => {
                    let len = self.varint()? as usize;
                    let end = self.i.checked_add(len)?;
                    if end > self.b.len() {
                        return None;
                    }
                    let sub = &self.b[self.i..end];
                    self.i = end;
                    return Some((field, Value::Bytes(sub)));
                }
                _ => return None,
            }
        }
        None
    }
}

/// A `repeated int64`, which protobuf may write packed into one field or as one
/// field per element.
fn shape_of(bytes: &[u8]) -> Vec<i64> {
    let mut r = Reader::new(bytes);
    let mut out = Vec::new();
    while let Some(v) = r.varint() {
        out.push(v as i64);
    }
    out
}

fn feature(bytes: &[u8]) -> Option<Feature> {
    let mut name = None;
    let mut shape = Vec::new();
    let mut dtype = 0;
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next() {
        match (field, value) {
            (1, Value::Bytes(b)) => name = std::str::from_utf8(b).ok().map(str::to_string),
            (3, Value::Bytes(b)) => {
                // FeatureType.multiArrayType
                let mut t = Reader::new(b);
                while let Some((f, v)) = t.next() {
                    if let (5, Value::Bytes(arr)) = (f, v) {
                        let mut a = Reader::new(arr);
                        while let Some((af, av)) = a.next() {
                            match (af, av) {
                                (1, Value::Bytes(packed)) => shape.extend(shape_of(packed)),
                                (1, Value::Varint(v)) => shape.push(v as i64),
                                (2, Value::Varint(v)) => dtype = v,
                                _ => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Some(Feature {
        name: name?,
        shape,
        dtype,
    })
}

fn function_desc(bytes: &[u8]) -> Option<Function> {
    let mut name = None;
    let (mut inputs, mut outputs) = (Vec::new(), Vec::new());
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next() {
        let Value::Bytes(b) = value else { continue };
        match field {
            1 => name = std::str::from_utf8(b).ok().map(str::to_string),
            2 => inputs.extend(feature(b)),
            3 => outputs.extend(feature(b)),
            _ => {}
        }
    }
    inputs.sort_by(|a, b| a.name.cmp(&b.name));
    outputs.sort_by(|a, b| a.name.cmp(&b.name));
    Some(Function {
        name: name?,
        inputs,
        outputs,
    })
}

fn map_entry(bytes: &[u8]) -> Option<(String, String)> {
    let mut key = None;
    let mut val = String::new();
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next() {
        if let Value::Bytes(b) = value {
            let text = std::str::from_utf8(b).ok()?.to_string();
            match field {
                1 => key = Some(text),
                2 => val = text,
                _ => {}
            }
        }
    }
    Some((key?, val))
}

pub fn read(bytes: &[u8]) -> Result<Spec, String> {
    let mut spec = Spec::default();
    let mut top = Reader::new(bytes);
    while let Some((field, value)) = top.next() {
        let Value::Bytes(b) = value else { continue };
        match field {
            2 => {
                let mut d = Reader::new(b);
                while let Some((f, v)) = d.next() {
                    let Value::Bytes(sub) = v else { continue };
                    match f {
                        1 => spec.inputs.extend(feature(sub)),
                        10 => spec.outputs.extend(feature(sub)),
                        20 => spec.described_functions.extend(function_desc(sub)),
                        21 => {
                            if let Ok(name) = std::str::from_utf8(sub) {
                                spec.default_function = name.to_string();
                            }
                        }
                        100 => {
                            let mut m = Reader::new(sub);
                            while let Some((mf, mv)) = m.next() {
                                if let (100, Value::Bytes(entry)) = (mf, mv) {
                                    spec.metadata.extend(map_entry(entry));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            502 => {
                let mut p = Reader::new(b);
                while let Some((f, v)) = p.next() {
                    if let (2, Value::Bytes(entry)) = (f, v) {
                        let mut e = Reader::new(entry);
                        while let Some((ef, ev)) = e.next() {
                            if let (1, Value::Bytes(name)) = (ef, ev) {
                                if let Ok(n) = std::str::from_utf8(name) {
                                    spec.functions.push(n.to_string());
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if spec.inputs.is_empty() && spec.outputs.is_empty() && spec.described_functions.is_empty() {
        return Err(
            "no inputs or outputs in the model description; this may not be a CoreML \
             model.mlmodel, or the format moved (see tools/coreml-jigs/src/spec.rs)"
                .to_string(),
        );
    }
    spec.functions.sort();
    spec.functions.dedup();
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(field: u32, wire: u64) -> Vec<u8> {
        let mut v = u64::from(field) << 3 | wire;
        let mut out = Vec::new();
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                return out;
            }
        }
    }

    fn varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                return out;
            }
        }
    }

    fn bytes_field(field: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = tag(field, 2);
        out.extend(varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }

    fn varint_field(field: u32, v: u64) -> Vec<u8> {
        let mut out = tag(field, 0);
        out.extend(varint(v));
        out
    }

    /// An ArrayFeatureType with a packed shape, as coremltools writes it.
    fn array_type(shape: &[i64], dtype: u64) -> Vec<u8> {
        let packed: Vec<u8> = shape.iter().flat_map(|&d| varint(d as u64)).collect();
        let mut arr = bytes_field(1, &packed);
        arr.extend(varint_field(2, dtype));
        bytes_field(5, &arr)
    }

    fn feature_desc(name: &str, shape: &[i64], dtype: u64) -> Vec<u8> {
        let mut f = bytes_field(1, name.as_bytes());
        f.extend(bytes_field(3, &array_type(shape, dtype)));
        f
    }

    fn model(features: Vec<Vec<u8>>, outputs: Vec<Vec<u8>>, meta: &[(&str, &str)]) -> Vec<u8> {
        let mut desc = Vec::new();
        for f in features {
            desc.extend(bytes_field(1, &f));
        }
        for o in outputs {
            desc.extend(bytes_field(10, &o));
        }
        if !meta.is_empty() {
            let mut m = Vec::new();
            for (k, v) in meta {
                let mut entry = bytes_field(1, k.as_bytes());
                entry.extend(bytes_field(2, v.as_bytes()));
                m.extend(bytes_field(100, &entry));
            }
            desc.extend(bytes_field(100, &m));
        }
        bytes_field(2, &desc)
    }

    #[test]
    fn reads_the_shapes_kohagi_depends_on() {
        let bytes = model(
            vec![
                feature_desc("input_ids", &[1, 128], 131104),
                feature_desc("attention_mask", &[1, 128], 131104),
            ],
            vec![feature_desc("hidden", &[1, 128, 512], 65552)],
            &[("com.github.apple.coremltools.version", "9.0")],
        );
        let spec = read(&bytes).expect("reads");
        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.inputs[0].name, "input_ids");
        assert_eq!(spec.inputs[0].shape, vec![1, 128]);
        assert_eq!(dtype_name(spec.inputs[0].dtype), "int32");
        assert_eq!(spec.outputs.len(), 1);
        assert_eq!(spec.outputs[0].shape, vec![1, 128, 512]);
        assert_eq!(dtype_name(spec.outputs[0].dtype), "fp16");
        assert_eq!(
            spec.metadata,
            vec![(
                "com.github.apple.coremltools.version".to_string(),
                "9.0".to_string()
            )]
        );
    }

    #[test]
    fn a_name_stays_with_its_own_shape() {
        // The reason for descending explicitly instead of flattening field paths:
        // two features of different rank must not have their shapes swapped or
        // merged.
        let bytes = model(
            vec![feature_desc("short", &[1, 64], 131104)],
            vec![feature_desc("long", &[1, 512, 768], 65552)],
            &[],
        );
        let spec = read(&bytes).unwrap();
        assert_eq!(spec.inputs[0].shape, vec![1, 64]);
        assert_eq!(spec.outputs[0].shape, vec![1, 512, 768]);
    }

    #[test]
    fn an_unpacked_shape_reads_the_same() {
        // protobuf may write `repeated int64` either way; a model written by a
        // different producer must not read as rank 0.
        let mut arr = varint_field(1, 1);
        arr.extend(varint_field(1, 256));
        arr.extend(varint_field(2, 65552));
        let mut f = bytes_field(1, b"hidden");
        f.extend(bytes_field(3, &bytes_field(5, &arr)));
        let bytes = model(vec![], vec![f], &[]);
        let spec = read(&bytes).unwrap();
        assert_eq!(spec.outputs[0].shape, vec![1, 256]);
    }

    #[test]
    fn a_model_without_a_description_is_an_error_not_an_empty_spec() {
        assert!(read(&bytes_field(999, b"nothing here")).is_err());
        assert!(read(&[]).is_err());
    }
}
