//! CoreML / Apple Neural Engine encoder backend.
//!
//! Unlike the CPU and Metal paths (which run candle's [`ModernBert`] over
//! bucketed batches), the ANE wants one thing: a *fixed-shape, batch=1*
//! forward. Batching collapses ANE throughput by ~18x, and flexible
//! (enumerated) input shapes disable the ANE compute plan entirely. So this
//! backend loads a **set of
//! pre-converted, fixed-length models** — one `seq-<N>.mlpackage` per bucket
//! length, e.g. `seq-128 / seq-256 / seq-512` — and routes each text to the smallest
//! bucket that fits, padded to that exact length, one row per prediction.
//!
//! Everything else — tokenization, prefixing, pooling, L2 normalization — stays
//! in Rust, exactly as for the candle paths; CoreML only replaces the encoder
//! forward. Output therefore matches the candle path to fp16 rounding
//! (cosine ~0.99999).
//!
//! [`ModernBert`]: crate::encoder::ModernBert

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
    /// Load the `seq-<N>` bucket models in `dir`, pinned to the Neural Engine.
    /// `dim` is the model's hidden size (from `config.json`).
    ///
    /// A bucket may be shipped as a compiled `.mlmodelc`, a portable
    /// `.mlpackage`, or both. When both are present we prefer the `.mlmodelc`
    /// (no per-run compile cost) and fall back to compiling the `.mlpackage`
    /// only if the compiled form is missing or fails to load — e.g. it was
    /// built for a different OS.
    pub fn load(dir: &Path, dim: usize) -> Result<Self> {
        // seq -> (compiled .mlmodelc, portable .mlpackage)
        let mut found: BTreeMap<usize, (Option<PathBuf>, Option<PathBuf>)> = BTreeMap::new();
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("reading CoreML model dir {}", dir.display()))?
        {
            let path = entry?.path();
            let Some(seq) = bucket_seq_of(&path) else {
                continue;
            };
            let slot = found.entry(seq).or_default();
            match path.extension().and_then(|e| e.to_str()) {
                Some("mlmodelc") => slot.0 = Some(path),
                Some("mlpackage") => slot.1 = Some(path),
                _ => {}
            }
        }
        if found.is_empty() {
            return Err(UnsupportedRequest::new(format!(
                "no `seq-<N>.mlpackage` bucket models found in {} — CoreML needs \
                 pre-converted fixed-shape models (see scripts/convert_coreml.py)",
                dir.display()
            ))
            .into());
        }

        // BTreeMap iterates in ascending seq order, so buckets end up sorted.
        let mut buckets = Vec::new();
        for (seq, (compiled, package)) in found {
            let model = load_bucket(seq, compiled.as_deref(), package.as_deref())?;
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

/// Parse a `seq-<N>.mlpackage` / `seq-<N>.mlmodelc` name into its bucket
/// length. `.mlpackage` is the portable artifact we distribute; `.mlmodelc` is
/// its already-compiled form, accepted so a local dir can hold either.
fn bucket_seq_of(path: &Path) -> Option<usize> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("mlpackage") | Some("mlmodelc") => {}
        _ => return None,
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("seq-"))
        .and_then(|n| n.parse::<usize>().ok())
}

/// Load one bucket, preferring the compiled `.mlmodelc` and falling back to the
/// portable `.mlpackage`. At least one of the two is `Some` (the caller only
/// inserts a bucket when it finds a file).
fn load_bucket(
    seq: usize,
    compiled: Option<&Path>,
    package: Option<&Path>,
) -> Result<Retained<MLModel>> {
    if let Some(c) = compiled {
        match load_model(c) {
            Ok(model) => return Ok(model),
            Err(e) if package.is_some() => {
                eprintln!(
                    "kohagi: seq-{seq}.mlmodelc did not load ({e:#}); \
                     compiling seq-{seq}.mlpackage instead"
                );
            }
            Err(e) => return Err(e).with_context(|| format!("loading {}", c.display())),
        }
    }
    let package = package.expect("load_bucket called with neither model form");
    load_model(package).with_context(|| format!("loading {}", package.display()))
}

/// Load one model, pinned to CPU+ANE. A `.mlpackage` is compiled to a
/// (temporary) `.mlmodelc` first; a `.mlmodelc` is loaded directly.
fn load_model(path: &Path) -> Result<Retained<MLModel>> {
    let compiled;
    let target = if path.extension().and_then(|e| e.to_str()) == Some("mlpackage") {
        compiled = compile_package(path)?;
        compiled.as_path()
    } else {
        path
    };
    unsafe {
        let url = file_url(target)?;
        let config = MLModelConfiguration::new();
        config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
        MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))
    }
}

/// A `file://` URL for a local path.
unsafe fn file_url(path: &Path) -> Result<Retained<NSURL>> {
    Ok(NSURL::fileURLWithPath(&NSString::from_str(
        path.to_str().context("model path is not valid UTF-8")?,
    )))
}

/// Compile a `.mlpackage` to a `.mlmodelc` and return its (temporary) path.
///
/// The Hugging Face cache stores a package as a tree of symlinks into its blob
/// store, which the CoreML compiler cannot follow — it fails with a spurious
/// "file doesn't exist". So if the direct compile fails we retry from a
/// dereferenced, symlink-free copy.
fn compile_package(pkg: &Path) -> Result<PathBuf> {
    if let Ok(out) = compile_at(pkg) {
        return Ok(out);
    }
    let staging = unique_temp_dir("kohagi-coreml-src");
    std::fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;
    let name = pkg.file_name().context("model path has no file name")?;
    let copy = staging.join(name);
    let result = copy_deref(pkg, &copy)
        .with_context(|| format!("dereferencing {}", pkg.display()))
        .and_then(|()| compile_at(&copy).with_context(|| format!("compiling {}", pkg.display())));
    let _ = std::fs::remove_dir_all(&staging);
    result
}

/// One `compileModelAtURL:` call; returns the compiled model's path.
fn compile_at(pkg: &Path) -> Result<PathBuf> {
    unsafe {
        let src = file_url(pkg)?;
        // The async compileModelAtURL:completionHandler: is the current API, but
        // the synchronous one is simpler and fine for a batch CLI.
        #[allow(deprecated)]
        let compiled =
            MLModel::compileModelAtURL_error(&src).map_err(|e| anyhow::anyhow!("{e}"))?;
        let path = compiled.path().context("compiled model URL has no path")?;
        Ok(PathBuf::from(path.to_string()))
    }
}

/// Recursively copy `src` to `dst`, following symlinks so the result has no
/// links — turns a symlinked HF-cache package into a real one the compiler can
/// read.
fn copy_deref(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            copy_deref(&entry.path(), &dst.join(entry.file_name()))?;
        }
        Ok(())
    } else {
        std::fs::copy(src, dst).map(|_| ())
    }
}

/// A process-unique path under the system temp dir (a per-process counter is
/// enough — one process compiles a handful of buckets).
fn unique_temp_dir(prefix: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{}-{}", std::process::id(), n))
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
