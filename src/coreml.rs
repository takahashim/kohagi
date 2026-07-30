//! CoreML / Apple Neural Engine encoder backend.
//!
//! Unlike the CPU and Metal paths (which run candle's [`ModernBert`] over
//! bucketed batches), the ANE wants one thing: a *fixed-shape, batch=1*
//! forward. Batching collapses ANE throughput by ~18x, and flexible
//! (enumerated) input shapes disable the ANE compute plan entirely. So this
//! backend loads a **set of pre-converted, fixed-length models** — one
//! `seq-<N>.mlpackage` per bucket length, e.g. `seq-128 / seq-256 / seq-512` —
//! and routes each text to the smallest bucket that fits, padded to that exact
//! length, one row per prediction.
//!
//! Everything else — tokenization, prefixing, pooling, L2 normalization — stays
//! in Rust, exactly as for the candle paths; CoreML only replaces the encoder
//! forward. Output therefore matches the candle path to fp16 rounding
//! (cosine ~0.99999).
//!
//! This module runs the loaded models; [`provision`] handles getting them onto
//! disk (Hub download) and into memory (locate + compile + load).
//!
//! [`ModernBert`]: crate::encoder::ModernBert

mod provision;

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use half::f16;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AllocAnyThread;
use objc2_core_ml::{
    MLDictionaryFeatureProvider, MLFeatureDescription, MLFeatureProvider, MLFeatureValue, MLModel,
    MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString};

use crate::UnsupportedRequest;

pub use provision::fetch_from_hub;

/// One loaded fixed-length model plus the sequence length it was compiled for.
struct Bucket {
    seq: usize,
    model: Retained<MLModel>,
}

/// A set of fixed-shape ANE models, one per bucket length, sharing a tokenizer
/// and hidden dimension. Not `Send`/`Sync`: the ANE is a single shared engine,
/// so [`crate::Embedder`] drives it from one thread anyway.
pub struct CoreMlEncoder {
    buckets: Vec<Bucket>,
    dim: usize,
}

impl CoreMlEncoder {
    /// Load the `seq-<N>` bucket models in `dir`, pinned to the Neural Engine.
    /// `dim` is the model's hidden size (from `config.json`).
    ///
    /// Layout: portable `seq-<N>.mlpackage` bundles sit at the top level, and a
    /// bucket may additionally ship a compiled `seq-<N>.mlmodelc` under
    /// `compiled/` (a flat `.mlmodelc` at the top level is also accepted). When
    /// both forms are present [`provision::load_bucket`] prefers the `.mlmodelc`
    /// and falls back to compiling the `.mlpackage`.
    ///
    /// A single `buckets-<N>-<N>…` bundle holding one CoreML function per length
    /// works the same way and is the compact form: the lengths share one copy of
    /// the weights, which for a large-vocabulary model is most of the bytes.
    pub fn load(dir: &std::path::Path, dim: usize) -> Result<Self> {
        // seq -> (compiled .mlmodelc, portable .mlpackage)
        let mut found: BTreeMap<usize, provision::BucketForms> = BTreeMap::new();
        provision::collect_buckets(dir, &mut found)
            .with_context(|| format!("reading CoreML model dir {}", dir.display()))?;
        let compiled_dir = dir.join("compiled");
        if compiled_dir.is_dir() {
            provision::collect_buckets(&compiled_dir, &mut found)
                .with_context(|| format!("reading {}", compiled_dir.display()))?;
        }
        if found.is_empty() {
            return Err(UnsupportedRequest::new(format!(
                "no `seq-<N>.mlpackage` or `buckets-<N>-<N>….mlpackage` bucket models \
                 found in {} — CoreML needs pre-converted fixed-shape models (see \
                 scripts/convert_coreml.py)",
                dir.display()
            ))
            .into());
        }

        // BTreeMap iterates in ascending seq order, so buckets end up sorted.
        let mut buckets = Vec::new();
        for (seq, forms) in found {
            let model =
                provision::load_bucket(seq, forms.compiled.as_ref(), forms.package.as_ref())?;
            check_io(&model, seq, dim)?;
            buckets.push(Bucket { seq, model });
        }
        Ok(Self { buckets, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The longest sequence any loaded bucket can serve.
    pub fn max_bucket(&self) -> usize {
        self.buckets.last().map_or(0, |b| b.seq)
    }

    /// Smallest bucket length that fits `len` tokens. `None` if `len` exceeds
    /// the largest bucket — the caller guarantees this cannot happen by
    /// validating `max_seq_length <= max_bucket()` at load.
    pub fn bucket_for(&self, len: usize) -> Option<usize> {
        self.buckets.iter().map(|b| b.seq).find(|&s| s >= len)
    }

    /// One forward pass for a single row already padded to bucket length `seq`.
    /// `ids`/`mask` are length `seq`. Returns flat `[seq * dim]` hidden states.
    pub fn forward(&self, ids: &[i64], mask: &[i64], seq: usize) -> Result<Vec<f32>> {
        debug_assert_eq!(ids.len(), seq);
        debug_assert_eq!(mask.len(), seq);
        let bucket = self
            .buckets
            .iter()
            .find(|b| b.seq == seq)
            .with_context(|| format!("no CoreML bucket for seq={seq}"))?;
        // SAFETY: single-threaded use; arrays and feature provider live for the
        // duration of the prediction call.
        unsafe {
            let ids_arr = i32_multiarray(seq, ids)?;
            let mask_arr = i32_multiarray(seq, mask)?;
            let ids_fv = MLFeatureValue::featureValueWithMultiArray(&ids_arr);
            let mask_fv = MLFeatureValue::featureValueWithMultiArray(&mask_arr);

            let k_ids = NSString::from_str("input_ids");
            let k_mask = NSString::from_str("attention_mask");
            let v_ids: &AnyObject = &ids_fv;
            let v_mask: &AnyObject = &mask_fv;
            let dict: Retained<NSDictionary<NSString, AnyObject>> =
                NSDictionary::from_slices(&[&*k_ids, &*k_mask], &[v_ids, v_mask]);
            let provider = MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
            .map_err(|e| anyhow::anyhow!("building CoreML feature provider: {e}"))?;
            let provider_obj: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*provider);

            let out = bucket
                .model
                .predictionFromFeatures_error(provider_obj)
                .map_err(|e| anyhow::anyhow!("CoreML prediction failed: {e}"))?;
            let hidden = out
                .featureValueForName(&NSString::from_str("hidden"))
                .context("CoreML output has no 'hidden' feature")?;
            let arr = hidden
                .multiArrayValue()
                .context("CoreML 'hidden' is not a multiarray")?;
            read_f32(&arr)
        }
    }
}

/// Check a loaded bucket against what this backend is about to assume of it:
/// the two input names it feeds, the output name it reads, and the output shape
/// `[1, seq, dim]`.
///
/// The `dim` comes from the directory's `config.json`, which nothing else ties to
/// the model itself. A directory whose `config.json` belonged to a
/// different checkpoint would either panic in [`crate::model`] (declared `dim`
/// too large) or silently return vectors pooled over the wrong stride
/// (declared `dim` too small). The same goes for `seq`, which comes from the
/// *file name*: a `seq-128.mlpackage` holding a 256-long model would pad and
/// pool against the wrong length. Both are cheap to rule out here, where the
/// model's own description is at hand and no input has been read yet.
fn check_io(model: &MLModel, seq: usize, dim: usize) -> Result<()> {
    let describe = |kind: &str, name: &str| {
        format!("this CoreML bucket (seq-{seq}) has no {kind} named `{name}`")
    };
    unsafe {
        let desc = model.modelDescription();

        let inputs = desc.inputDescriptionsByName();
        for name in ["input_ids", "attention_mask"] {
            if inputs.objectForKey(&NSString::from_str(name)).is_none() {
                let have = feature_names(&inputs);
                return Err(UnsupportedRequest::new(format!(
                    "{} (it has: {have}) — CoreML models for kohagi take \
                     `input_ids` and `attention_mask`; see scripts/convert_coreml.py",
                    describe("input", name)
                ))
                .into());
            }
        }

        let outputs = desc.outputDescriptionsByName();
        let Some(hidden) = outputs.objectForKey(&NSString::from_str("hidden")) else {
            let have = feature_names(&outputs);
            return Err(UnsupportedRequest::new(format!(
                "{} (it has: {have}) — kohagi reads the encoder's hidden states \
                 from an output named `hidden` and pools them itself",
                describe("output", "hidden")
            ))
            .into());
        };
        let shape: Vec<usize> = hidden
            .multiArrayConstraint()
            .context("CoreML output `hidden` is not a multiarray")?
            .shape()
            .iter()
            .map(|n| n.as_isize().max(0) as usize)
            .collect();

        // A fixed-shape encoder output is [1, seq, dim]. Anything else means the
        // bundle is not what this backend runs, so say what was found rather
        // than indexing into it.
        if shape != [1, seq, dim] {
            return Err(UnsupportedRequest::new(format!(
                "CoreML bucket seq-{seq} outputs `hidden` with shape {shape:?}, \
                 but kohagi expected [1, {seq}, {dim}] — {seq} comes from the bundle's \
                 file name and {dim} from `hidden_size` in config.json, so one of the \
                 three disagrees with the model. Check that config.json belongs to the \
                 converted checkpoint and that the bundle is named for its real length"
            ))
            .into());
        }
    }
    Ok(())
}

/// The feature names in a model description dictionary, for an error message.
unsafe fn feature_names(d: &NSDictionary<NSString, MLFeatureDescription>) -> String {
    let mut names: Vec<String> = d.keys().map(|k| k.to_string()).collect();
    names.sort();
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

/// Build a `[1, seq]` Int32 MLMultiArray from `i64` token ids/mask.
unsafe fn i32_multiarray(seq: usize, values: &[i64]) -> Result<Retained<MLMultiArray>> {
    let dims = [NSNumber::new_isize(1), NSNumber::new_isize(seq as isize)];
    let shape = NSArray::from_retained_slice(&dims);
    let arr = MLMultiArray::initWithShape_dataType_error(
        MLMultiArray::alloc(),
        &shape,
        MLMultiArrayDataType::Int32,
    )
    .map_err(|e| anyhow::anyhow!("allocating a {seq}-wide MLMultiArray: {e}"))?;
    // `dataPointer` is deprecated in favour of the block-based getBytes API, but
    // it is correct for the contiguous arrays we allocate here, and avoids the
    // RcBlock ceremony. Revisit if a future objc2-core-ml drops it.
    #[allow(deprecated)]
    let ptr = arr.dataPointer().as_ptr() as *mut i32;
    for (i, &v) in values.iter().enumerate() {
        *ptr.add(i) = v as i32;
    }
    Ok(arr)
}

/// Copy an output MLMultiArray into a flat `Vec<f32>`, converting from whatever
/// element type CoreML produced (fp16 in practice, but be defensive).
unsafe fn read_f32(arr: &MLMultiArray) -> Result<Vec<f32>> {
    let count = arr.count() as usize;
    #[allow(deprecated)]
    let ptr = arr.dataPointer().as_ptr();
    let out = match arr.dataType() {
        MLMultiArrayDataType::Float32 => {
            std::slice::from_raw_parts(ptr as *const f32, count).to_vec()
        }
        MLMultiArrayDataType::Float16 => std::slice::from_raw_parts(ptr as *const u16, count)
            .iter()
            .map(|&b| f16::from_bits(b).to_f32())
            .collect(),
        MLMultiArrayDataType::Double => std::slice::from_raw_parts(ptr as *const f64, count)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        other => {
            return Err(anyhow::anyhow!(
                "unexpected CoreML output dtype {other:?} (expected float)"
            ))
        }
    };
    Ok(out)
}
