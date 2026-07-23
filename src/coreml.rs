//! CoreML / Apple Neural Engine encoder backend.
//!
//! Unlike the CPU and Metal paths (which run candle's [`ModernBert`] over
//! bucketed batches), the ANE wants one thing: a *fixed-shape, batch=1*
//! forward. Batching collapses ANE throughput by ~18x, and flexible
//! (enumerated) input shapes disable the ANE compute plan entirely (see
//! `notes/coreml-feasibility.md`). So this backend loads a **set of
//! pre-converted, fixed-length models** — one `.mlmodelc` per bucket length,
//! e.g. `seq-128 / seq-256 / seq-512` — and routes each text to the smallest
//! bucket that fits, padded to that exact length, one row per prediction.
//!
//! Everything else — tokenization, prefixing, pooling, L2 normalization — stays
//! in Rust, exactly as for the candle paths; CoreML only replaces the encoder
//! forward. Output therefore matches the candle path to fp16 rounding
//! (cosine ~0.99999).
//!
//! [`ModernBert`]: crate::encoder::ModernBert

use std::path::Path;

use anyhow::{Context, Result};
use half::f16;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AllocAnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

use crate::model::UnsupportedRequest;

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
    /// Load every `seq-<N>.mlmodelc` in `dir`, pinned to the Neural Engine.
    /// `dim` is the model's hidden size (from `config.json`).
    pub fn load(dir: &Path, dim: usize) -> Result<Self> {
        let mut buckets = Vec::new();
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("reading CoreML model dir {}", dir.display()))?
        {
            let path = entry?.path();
            let Some(seq) = bucket_seq_of(&path) else {
                continue;
            };
            let model = load_model(&path)
                .with_context(|| format!("loading CoreML model {}", path.display()))?;
            buckets.push(Bucket { seq, model });
        }
        if buckets.is_empty() {
            return Err(UnsupportedRequest::new(format!(
                "no `seq-<N>.mlmodelc` bucket models found in {} — CoreML needs \
                 pre-converted fixed-shape models (see notes/coreml-feasibility.md)",
                dir.display()
            ))
            .into());
        }
        buckets.sort_by_key(|b| b.seq);
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
            let ids_arr = i32_multiarray(seq, ids);
            let mask_arr = i32_multiarray(seq, mask);
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

/// Parse a `seq-<N>.mlmodelc` directory name into its bucket length.
fn bucket_seq_of(path: &Path) -> Option<usize> {
    if path.extension().and_then(|e| e.to_str()) != Some("mlmodelc") {
        return None;
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("seq-"))
        .and_then(|n| n.parse::<usize>().ok())
}

/// Load one compiled `.mlmodelc`, pinned to CPU+ANE.
fn load_model(path: &Path) -> Result<Retained<MLModel>> {
    unsafe {
        let url = NSURL::fileURLWithPath(&NSString::from_str(
            path.to_str().context("model path is not valid UTF-8")?,
        ));
        let config = MLModelConfiguration::new();
        config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
        MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Build a `[1, seq]` Int32 MLMultiArray from `i64` token ids/mask.
unsafe fn i32_multiarray(seq: usize, values: &[i64]) -> Retained<MLMultiArray> {
    let dims = [NSNumber::new_isize(1), NSNumber::new_isize(seq as isize)];
    let shape = NSArray::from_retained_slice(&dims);
    let arr = MLMultiArray::initWithShape_dataType_error(
        MLMultiArray::alloc(),
        &shape,
        MLMultiArrayDataType::Int32,
    )
    .expect("allocate MLMultiArray");
    // `dataPointer` is deprecated in favour of the block-based getBytes API, but
    // it is correct for the contiguous arrays we allocate here, and avoids the
    // RcBlock ceremony. Revisit if a future objc2-core-ml drops it.
    #[allow(deprecated)]
    let ptr = arr.dataPointer().as_ptr() as *mut i32;
    for (i, &v) in values.iter().enumerate() {
        *ptr.add(i) = v as i32;
    }
    arr
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
