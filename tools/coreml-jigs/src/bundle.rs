//! Finding, loading and running the bucket models in a converted directory.
//!
//! Shared by the jigs that need a live `MLModel` rather than just the asset's
//! description. The prediction path mirrors `src/coreml.rs` in Kohagi itself;
//! it is repeated here rather than exported so the jigs cannot influence the
//! shipped backend's shape, and it is small enough that the duplication is
//! visible in one screen.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AllocAnyThread;
use objc2_core_ml::{
    MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLModelAsset,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

use crate::await_handler;

/// One loadable unit: a bundle plus, for a multi-function bundle, the function
/// inside it. `None` means the model's default function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    /// The bundle as the caller named it. Reports use this, so a compiled
    /// temporary does not leak into the output.
    pub path: PathBuf,
    pub function: Option<String>,
    /// Where to load from, when that is not `path` — set by [`compile_once`].
    pub compiled: Option<PathBuf>,
}

impl Target {
    /// How this target is written in a report: `seq-128.mlmodelc` or
    /// `buckets-128-256.mlpackage:seq_256`.
    pub fn label(&self) -> String {
        let file = self.path.file_name().unwrap_or_default().to_string_lossy();
        match &self.function {
            Some(f) => format!("{file}:{f}"),
            None => file.to_string(),
        }
    }
}

pub fn is_bundle(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()),
        Some("mlpackage" | "mlmodelc")
    )
}

/// Every bundle at `root`, including the `compiled/` subdirectory a converted
/// layout puts `.mlmodelc`s in. A path to a bundle is accepted as-is.
pub fn collect(root: &Path) -> Vec<PathBuf> {
    if is_bundle(root) {
        return vec![root.to_path_buf()];
    }
    let mut out = Vec::new();
    for dir in [root.to_path_buf(), root.join("compiled")] {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            out.extend(entries.flatten().map(|e| e.path()).filter(|p| is_bundle(p)));
        }
    }
    out.sort();
    out
}

/// `seq-128.mlmodelc` -> `[128]`, `buckets-128-256.mlpackage` -> `[128, 256]`.
/// Only ever used to compare a name against the model, never as the truth about
/// it: a bundle can be misnamed, and that is one of the things worth catching.
pub fn lengths_in_name(path: &Path) -> Option<Vec<usize>> {
    let stem = path.file_stem()?.to_str()?;
    if let Some(one) = stem.strip_prefix("seq-") {
        return Some(vec![one.parse().ok()?]);
    }
    stem.strip_prefix("buckets-")?
        .split('-')
        .map(|s| s.parse().ok())
        .collect()
}

fn file_url(path: &Path) -> Result<Retained<NSURL>, String> {
    let s = path.to_str().ok_or("path is not valid UTF-8")?;
    Ok(NSURL::fileURLWithPath(&NSString::from_str(s)))
}

/// The function names in a bundle. Empty for a single-function bundle, which is
/// loaded through its default function instead.
///
/// A `.mlpackage` is read from its protobuf, because `MLModelAsset` only opens
/// compiled models — the same split `coreml-inspect` makes.
pub fn function_names(path: &Path) -> Result<Vec<String>, String> {
    if path.extension().and_then(|e| e.to_str()) == Some("mlpackage") {
        let proto = path.join("Data/com.apple.CoreML/model.mlmodel");
        let bytes =
            std::fs::read(&proto).map_err(|e| format!("reading {}: {e}", proto.display()))?;
        let spec = crate::spec::read(&bytes)?;
        return Ok(spec
            .described_functions
            .into_iter()
            .map(|f| f.name)
            .collect());
    }
    let url = file_url(path)?;
    let asset = unsafe { MLModelAsset::modelAssetWithURL_error(&url) }
        .map_err(|e| format!("opening {}: {e}", path.display()))?;
    let names: Retained<NSArray<NSString>> =
        await_handler(|h| unsafe { asset.functionNamesWithCompletionHandler(h) })
            .map_err(|e| format!("reading function names of {}: {e}", path.display()))?;
    Ok(names.iter().map(|n| n.to_string()).collect())
}

/// Every loadable target under `root`, one per function of every bundle.
pub fn targets(root: &Path) -> Result<Vec<Target>, String> {
    let mut out = Vec::new();
    for path in collect(root) {
        let names = function_names(&path)?;
        if names.is_empty() {
            out.push(Target {
                path,
                function: None,
                compiled: None,
            });
        } else {
            for f in names {
                out.push(Target {
                    path: path.clone(),
                    function: Some(f),
                    compiled: None,
                });
            }
        }
    }
    Ok(out)
}

/// Compile a `.mlpackage` and return the compiled path; pass a `.mlmodelc`
/// through unchanged.
///
/// CoreML will not load a package directly, and several of its APIs abort the
/// process rather than returning an error when handed one, so this is where the
/// distinction is handled once. Compiling costs ~20s for a large model, and the
/// result is a temporary the OS cleans up — Kohagi's own path caches it
/// (`src/coreml/provision.rs`), but a jig has no reason to.
pub fn compiled_path(path: &Path) -> Result<PathBuf, String> {
    if path.extension().and_then(|e| e.to_str()) != Some("mlpackage") {
        return Ok(path.to_path_buf());
    }
    let url = file_url(path)?;
    let compiled = unsafe {
        #[allow(deprecated)]
        MLModel::compileModelAtURL_error(&url)
    }
    .map_err(|e| format!("compiling {}: {e}", path.display()))?;
    compiled
        .path()
        .map(|p| PathBuf::from(p.to_string()))
        .ok_or_else(|| "the compiled model has no path".to_string())
}

/// Rewrite every target to point at a compiled model, compiling each distinct
/// package once.
///
/// Compiling per target would repeat the work for every function of a
/// multi-function bundle, and each fresh compile lands in a new temporary
/// directory, so the load that follows misses the OS's program cache as well.
pub fn compile_once(targets: &[Target]) -> Result<Vec<Target>, String> {
    let mut compiled: BTreeMap<PathBuf, PathBuf> = BTreeMap::new();
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let ready = match compiled.get(&target.path) {
            Some(path) => path.clone(),
            None => {
                let path = compiled_path(&target.path)?;
                compiled.insert(target.path.clone(), path.clone());
                path
            }
        };
        out.push(Target {
            compiled: Some(ready),
            ..target.clone()
        });
    }
    Ok(out)
}

/// Load one target pinned to CPU+ANE, and report how long it took.
///
/// The duration is a **cold** load: CoreML builds the Neural Engine program on
/// first load of a given compiled model, and that is included. Kohagi's own path
/// is warm after the first run because it caches the compiled model in a stable
/// place (`src/coreml/provision.rs`), which measured 0.09s per bucket.
pub fn load(target: &Target) -> Result<(Retained<MLModel>, Duration), String> {
    load_on(target, objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine)
}

/// The same, on a chosen set of compute units.
///
/// `CPUOnly` is what tells an arithmetic difference from a graph difference: the
/// same file, the same weights, the same operations, only the hardware changes.
pub fn load_on(
    target: &Target,
    units: objc2_core_ml::MLComputeUnits,
) -> Result<(Retained<MLModel>, Duration), String> {
    let url = file_url(target.compiled.as_deref().unwrap_or(&target.path))?;
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(units) };
    if let Some(f) = &target.function {
        unsafe { config.setFunctionName(Some(&NSString::from_str(f))) };
    }
    let start = Instant::now();
    let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
        .map_err(|e| format!("loading {}: {e}", target.label()))?;
    Ok((model, start.elapsed()))
}

/// The `[1, seq, dim]` of a model's `hidden` output.
pub fn hidden_shape(model: &MLModel) -> Option<(usize, usize)> {
    let desc = unsafe { model.modelDescription() };
    let outputs = unsafe { desc.outputDescriptionsByName() };
    let hidden = outputs.objectForKey(&NSString::from_str("hidden"))?;
    let constraint = unsafe { hidden.multiArrayConstraint() }?;
    let shape: Vec<usize> = unsafe { constraint.shape() }
        .iter()
        .map(|n| n.as_isize().max(0) as usize)
        .collect();
    match shape.as_slice() {
        [1, seq, dim] => Some((*seq, *dim)),
        _ => None,
    }
}

fn int32_array(values: &[i32]) -> Result<Retained<MLMultiArray>, String> {
    let dims = [
        NSNumber::new_isize(1),
        NSNumber::new_isize(values.len() as isize),
    ];
    let shape = NSArray::from_retained_slice(&dims);
    let arr = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape,
            MLMultiArrayDataType::Int32,
        )
    }
    .map_err(|e| format!("allocating a {}-wide MLMultiArray: {e}", values.len()))?;
    #[allow(deprecated)]
    let ptr = unsafe { arr.dataPointer() }.as_ptr() as *mut i32;
    for (i, &v) in values.iter().enumerate() {
        unsafe { *ptr.add(i) = v };
    }
    Ok(arr)
}

/// One `input_ids`/`attention_mask` pair, built once and reused across
/// predictions so a timing loop measures the forward pass and not allocation.
pub struct Inputs(Retained<ProtocolObject<dyn MLFeatureProvider>>);

pub fn inputs(ids: &[i32], mask: &[i32]) -> Result<Inputs, String> {
    let ids_arr = int32_array(ids)?;
    let mask_arr = int32_array(mask)?;
    let ids_fv = unsafe { MLFeatureValue::featureValueWithMultiArray(&ids_arr) };
    let mask_fv = unsafe { MLFeatureValue::featureValueWithMultiArray(&mask_arr) };
    let keys = [
        NSString::from_str("input_ids"),
        NSString::from_str("attention_mask"),
    ];
    let values: [&AnyObject; 2] = [&ids_fv, &mask_fv];
    let dict: Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_slices(&[&*keys[0], &*keys[1]], &values);
    let provider = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &dict,
        )
    }
    .map_err(|e| format!("building a feature provider: {e}"))?;
    Ok(Inputs(ProtocolObject::from_retained(provider)))
}

/// One forward pass. The output is dropped: these jigs time the pass and check
/// that it succeeds.
pub fn predict(model: &MLModel, inputs: &Inputs) -> Result<(), String> {
    unsafe { model.predictionFromFeatures_error(&inputs.0) }
        .map(|_| ())
        .map_err(|e| format!("prediction failed: {e}"))
}

/// One forward pass, returning the named output in logical row-major order.
///
/// Read through `shape` and `strides` rather than off `dataPointer`: CoreML pads
/// an axis, and taking `count()` elements straight from the pointer interleaves
/// values with padding above rank 2 (the same trap `src/coreml.rs` had).
pub fn predict_output(model: &MLModel, inputs: &Inputs, name: &str) -> Result<Vec<f32>, String> {
    let out = unsafe { model.predictionFromFeatures_error(&inputs.0) }
        .map_err(|e| format!("prediction failed: {e}"))?;
    let feature = unsafe { out.featureValueForName(&NSString::from_str(name)) }
        .ok_or_else(|| format!("the model has no output named `{name}`"))?;
    let arr = unsafe { feature.multiArrayValue() }
        .ok_or_else(|| format!("output `{name}` is not a multiarray"))?;

    let numbers = |a: Retained<NSArray<NSNumber>>| -> Vec<usize> {
        a.iter().map(|n| n.as_isize().max(0) as usize).collect()
    };
    let shape = numbers(unsafe { arr.shape() });
    let strides = numbers(unsafe { arr.strides() });
    let total: usize = shape.iter().product();
    let mut values = Vec::with_capacity(total);
    let mut index = vec![0usize; shape.len()];
    unsafe {
        #[allow(deprecated)]
        let ptr = arr.dataPointer().as_ptr() as *const u16;
        for _ in 0..total {
            let offset: usize = index.iter().zip(&strides).map(|(i, s)| i * s).sum();
            values.push(half::f16::from_bits(*ptr.add(offset)).to_f32());
            for axis in (0..shape.len()).rev() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }
    Ok(values)
}

/// Token ids that are deterministic across runs, so two runs of a jig differ
/// only in timing. The values themselves do not matter to a fixed-shape model,
/// but reproducibility does.
pub fn fake_ids(len: usize, vocab: i32) -> Vec<i32> {
    let mut state: u32 = 0x9E37_79B9;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            5 + (state >> 16) as i32 % (vocab - 5).max(1)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_parsed_but_never_trusted_alone() {
        assert_eq!(
            lengths_in_name(Path::new("seq-128.mlmodelc")),
            Some(vec![128])
        );
        assert_eq!(
            lengths_in_name(Path::new("buckets-128-256-512.mlpackage")),
            Some(vec![128, 256, 512])
        );
        assert_eq!(lengths_in_name(Path::new("seq-xyz.mlpackage")), None);
        assert_eq!(lengths_in_name(Path::new("config.json")), None);
    }

    #[test]
    fn labels_name_the_function_when_there_is_one() {
        let plain = Target {
            path: PathBuf::from("/m/seq-128.mlmodelc"),
            function: None,
            compiled: None,
        };
        assert_eq!(plain.label(), "seq-128.mlmodelc");
        let multi = Target {
            path: PathBuf::from("/m/buckets-128-256.mlpackage"),
            function: Some("seq_256".into()),
            compiled: None,
        };
        assert_eq!(multi.label(), "buckets-128-256.mlpackage:seq_256");
        // A compiled temporary must not change how a target is reported.
        let ready = Target {
            compiled: Some(PathBuf::from("/tmp/whatever_ABC.mlmodelc")),
            ..multi
        };
        assert_eq!(ready.label(), "buckets-128-256.mlpackage:seq_256");
    }

    #[test]
    fn fake_ids_are_deterministic_and_in_range() {
        let a = fake_ids(64, 1000);
        assert_eq!(a, fake_ids(64, 1000));
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|&v| (5..1000).contains(&v)));
    }
}
