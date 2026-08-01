//! Writing CoreML `.mlpackage` models from Rust.
//!
//! Behind the `coreml-export` feature, so that a converted model comes out of the
//! same code that reads the checkpoint at inference time rather than out of
//! `scripts/convert_coreml.py` and a PyTorch install. [`blob`] writes
//! `weights/weight.bin`, [`mil`] the program, [`encoder`] the ModernBERT graph,
//! and [`write_package`] the bundle around them.
//!
//! The layout, which [`write_package`] produces and `src/coreml.rs` consumes:
//!
//! ```text
//! seq-128.mlpackage/
//! ├── Manifest.json
//! └── Data/
//!     └── com.apple.CoreML/
//!         ├── model.mlmodel
//!         └── weights/
//!             └── weight.bin
//! ```

pub mod blob;
pub mod encoder;
pub mod mil;
pub mod modernbert;
pub mod safetensors;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use prost::Message;

use modernbert::Activation;

use crate::coreml_proto::{
    array_feature_type, feature_type, model, ArrayFeatureType, FeatureDescription, FeatureType,
    FunctionDescription, Metadata, Model, ModelDescription,
};

/// `Manifest.json`, the 617-byte index at the root of a `.mlpackage`.
///
/// The two identifiers are opaque: CoreML only needs `rootModelIdentifier` to
/// name an entry that exists. A real package carries freshly generated UUIDs;
/// these are fixed so that generating the same model twice produces the same
/// bytes.
const MODEL_ID: &str = "6B0C4B1A-1E7C-4B7E-9E3D-000000000001";
const WEIGHTS_ID: &str = "6B0C4B1A-1E7C-4B7E-9E3D-000000000002";

fn manifest() -> String {
    // Written out rather than serialized from a map: the file is four fixed keys,
    // and matching a reference byte for byte is easier to see this way.
    format!(
        r#"{{
    "fileFormatVersion": "1.0.0",
    "itemInfoEntries": {{
        "{WEIGHTS_ID}": {{
            "author": "com.apple.CoreML",
            "description": "CoreML Model Weights",
            "name": "weights",
            "path": "com.apple.CoreML/weights"
        }},
        "{MODEL_ID}": {{
            "author": "com.apple.CoreML",
            "description": "CoreML Model Specification",
            "name": "model.mlmodel",
            "path": "com.apple.CoreML/model.mlmodel"
        }}
    }},
    "rootModelIdentifier": "{MODEL_ID}"
}}
"#
    )
}

/// One input or output in the model description, which is what a caller feeds and
/// reads at prediction time. Kept separate from [`mil::Tensor`]: the program's
/// internal values are not part of the model's interface.
fn feature(tensor: &mil::Tensor) -> FeatureDescription {
    let dtype = match tensor.dtype {
        mil::DType::Fp16 => array_feature_type::ArrayDataType::Float16,
        mil::DType::Fp32 => array_feature_type::ArrayDataType::Float32,
        mil::DType::Int32 => array_feature_type::ArrayDataType::Int32,
        mil::DType::Int8 => array_feature_type::ArrayDataType::Int8,
        // A model's interface is numeric arrays. Bool and Str exist for operation
        // parameters inside the graph and have no place in a feature description,
        // so this is a caller error rather than something to encode.
        other => panic!(
            "{} is declared {other:?}, which cannot be a model input or output",
            tensor.name
        ),
    };
    FeatureDescription {
        name: tensor.name.clone(),
        short_description: String::new(),
        r#type: Some(FeatureType {
            r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                shape: tensor.shape.iter().map(|&d| d as i64).collect(),
                data_type: dtype as i32,
            })),
            is_optional: false,
        }),
    }
}

/// What produced a model, recorded in its `userDefined` metadata.
///
/// A model card is expected to state the toolchain and the source revision; a
/// converted model that carries them can be asked instead of trusted.
/// `coreml-inspect` reads these back.
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    /// The checkpoint this was converted from, as a Hub id or a path.
    pub source: String,
    /// Which sequence lengths the bundle serves.
    pub lengths: Vec<usize>,
    /// Whether the embedding table is int8. Recorded because a quantized bundle's
    /// vectors are not interchangeable with an fp16 one's.
    pub quantized_embeddings: bool,
    /// Whether the projections are int8 too.
    pub quantized_projections: bool,
    /// The gate activation the graph was built with, when it is not the
    /// ModernBERT default. Recorded because it is the one graph shape a config
    /// changes, so a bundle should be able to say which one it is.
    pub activation: Option<&'static str>,
}

impl Provenance {
    fn entries(&self) -> Vec<(String, String)> {
        let mut out = vec![
            (
                "com.github.takahashim.kohagi.version".to_string(),
                env!("CARGO_PKG_VERSION").to_string(),
            ),
            (
                "com.github.takahashim.kohagi.emitter".to_string(),
                "coreml-export".to_string(),
            ),
        ];
        if !self.source.is_empty() {
            out.push((
                "com.github.takahashim.kohagi.source".to_string(),
                self.source.clone(),
            ));
        }
        if self.quantized_embeddings {
            out.push((
                "com.github.takahashim.kohagi.quantization".to_string(),
                if self.quantized_projections {
                    "all-int8".to_string()
                } else {
                    "embeddings-int8".to_string()
                },
            ));
        }
        if let Some(act) = self.activation {
            out.push((
                "com.github.takahashim.kohagi.activation".to_string(),
                act.to_string(),
            ));
        }
        if !self.lengths.is_empty() {
            out.push((
                "com.github.takahashim.kohagi.sequence_lengths".to_string(),
                self.lengths
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ));
        }
        out
    }

    fn metadata(&self) -> Metadata {
        Metadata {
            short_description: String::new(),
            version_string: String::new(),
            author: String::new(),
            license: String::new(),
            user_defined: self.entries().into_iter().collect(),
        }
    }
}

/// Assemble a `Model` around a program.
pub fn model(program: crate::coreml_proto::mil_spec::Program, io: Io) -> Model {
    model_with(program, io, &Provenance::default())
}

/// The same, recording what produced it.
pub fn model_with(
    program: crate::coreml_proto::mil_spec::Program,
    io: Io,
    provenance: &Provenance,
) -> Model {
    Model {
        specification_version: mil::SPECIFICATION_VERSION,
        description: Some(ModelDescription {
            input: io.inputs.iter().map(feature).collect(),
            output: io.outputs.iter().map(feature).collect(),
            functions: Vec::new(),
            default_function_name: String::new(),
            metadata: Some(provenance.metadata()),
        }),
        is_updatable: false,
        r#type: Some(model::Type::MlProgram(program)),
    }
}

/// Assemble a `Model` whose program holds several functions, one per bucket
/// length.
///
/// A multi-function model describes each function separately rather than through
/// the top-level `input`/`output`, and names one as the default. `functions` pairs
/// each MIL function name with the interface it presents.
pub fn multi_function_model(
    program: crate::coreml_proto::mil_spec::Program,
    functions: &[(String, Vec<mil::Tensor>, Vec<mil::Tensor>)],
    default: &str,
    provenance: &Provenance,
) -> Model {
    Model {
        specification_version: mil::SPECIFICATION_VERSION,
        description: Some(ModelDescription {
            input: Vec::new(),
            output: Vec::new(),
            functions: functions
                .iter()
                .map(|(name, inputs, outputs)| FunctionDescription {
                    name: name.clone(),
                    input: inputs.iter().map(feature).collect(),
                    output: outputs.iter().map(feature).collect(),
                })
                .collect(),
            default_function_name: default.to_string(),
            metadata: Some(provenance.metadata()),
        }),
        is_updatable: false,
        r#type: Some(model::Type::MlProgram(program)),
    }
}

/// The model's interface: which of the program's values a caller supplies and
/// which it reads back.
pub struct Io {
    pub inputs: Vec<mil::Tensor>,
    pub outputs: Vec<mil::Tensor>,
}

/// Write a `.mlpackage` at `path`, replacing anything already there.
///
/// `weights` may be empty, in which case the file is still written: a package
/// whose manifest names a weights directory that does not exist fails to compile.
pub fn write_package(path: &Path, model: &Model, weights: &[u8]) -> Result<()> {
    let data = path.join("Data/com.apple.CoreML");
    if path.exists() {
        std::fs::remove_dir_all(path).with_context(|| format!("clearing {}", path.display()))?;
    }
    std::fs::create_dir_all(data.join("weights"))
        .with_context(|| format!("creating {}", data.display()))?;

    std::fs::write(path.join("Manifest.json"), manifest())
        .with_context(|| format!("writing {}/Manifest.json", path.display()))?;
    std::fs::write(data.join("model.mlmodel"), model.encode_to_vec())
        .with_context(|| format!("writing {}/model.mlmodel", data.display()))?;
    std::fs::write(data.join("weights/weight.bin"), weights)
        .with_context(|| format!("writing {}/weights/weight.bin", data.display()))?;
    Ok(())
}

/// Compile a written `.mlpackage` into `<dir>/compiled/<name>.mlmodelc`.
///
/// Optional: Kohagi compiles a package on first use and caches the result
/// (`src/coreml/provision.rs`), so shipping the compiled form only moves ~20 s off
/// a user's first run, at the cost of doubling what they download. A publisher
/// deciding to pay that can call this.
///
/// Needs the `coreml` feature as well, since compiling goes through the framework.
#[cfg(feature = "coreml")]
pub fn compile_beside(bundle: &Path) -> Result<std::path::PathBuf> {
    use objc2_core_ml::MLModel;
    use objc2_foundation::{NSString, NSURL};

    let parent = bundle
        .parent()
        .context("the bundle has no parent directory")?;
    let stem = bundle
        .file_stem()
        .and_then(|s| s.to_str())
        .context("the bundle has no name")?;
    let out = parent.join("compiled").join(format!("{stem}.mlmodelc"));

    let url = NSURL::fileURLWithPath(&NSString::from_str(
        bundle.to_str().context("bundle path is not valid UTF-8")?,
    ));
    let compiled = unsafe {
        // The async form is current, but a batch converter has nothing to do while
        // it waits.
        #[allow(deprecated)]
        MLModel::compileModelAtURL_error(&url)
    }
    .map_err(|e| anyhow::anyhow!("compiling {}: {e}", bundle.display()))?;
    let from = std::path::PathBuf::from(
        compiled
            .path()
            .context("the compiled model has no path")?
            .to_string(),
    );

    if out.exists() {
        std::fs::remove_dir_all(&out).with_context(|| format!("clearing {}", out.display()))?;
    }
    std::fs::create_dir_all(out.parent().expect("compiled/ has a parent"))?;
    // CoreML leaves the result in a temporary directory for the caller to move.
    if std::fs::rename(&from, &out).is_err() {
        copy_tree(&from, &out).with_context(|| format!("copying into {}", out.display()))?;
        let _ = std::fs::remove_dir_all(&from);
    }
    Ok(out)
}

#[cfg(feature = "coreml")]
fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    if from.is_dir() {
        std::fs::create_dir_all(to)?;
        for entry in std::fs::read_dir(from)? {
            let entry = entry?;
            copy_tree(&entry.path(), &to.join(entry.file_name()))?;
        }
        Ok(())
    } else {
        std::fs::copy(from, to).map(|_| ())
    }
}

/// A checkpoint's files, resolved on disk, and how to name it.
///
/// Both callers of [`convert`] arrive at these differently — the CLI downloads or
/// takes paths, `--device coreml` reuses whatever the candle path resolved — but
/// what the conversion needs from a checkpoint is the same either way.
pub struct Checkpoint {
    /// `model.safetensors`.
    pub weights: PathBuf,
    /// `config.json`.
    pub config: PathBuf,
    /// `tokenizer.json`.
    pub tokenizer: PathBuf,
    /// `1_Pooling/config.json`, when the checkpoint ships one. A reranker or a
    /// base LM does not, and Kohagi falls back to mean pooling with a warning.
    pub pooling: Option<PathBuf>,
    /// The Hub id, or the path for a local checkpoint. Recorded in the bundle's
    /// provenance.
    pub source: String,
}

/// The file name a bundle of these lengths gets.
///
/// One function because a cached conversion is found by this name: spelling it
/// differently in two places would turn every cache hit into a silent miss.
pub fn bundle_name(lengths: &[usize]) -> String {
    format!(
        "buckets-{}.mlpackage",
        lengths
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-")
    )
}

/// Convert a checkpoint into `dir`, leaving a directory Kohagi can load.
///
/// The whole procedure in one place: read the config, refuse a bucket the
/// checkpoint was not trained for before opening 500MB of weights, emit, and copy
/// the metadata a loader needs beside the bundle. `--device coreml` and
/// `coreml-convert` both go through here, so a bundle is the same artifact
/// whichever produced it.
///
/// Returns the emitted config, which the caller has already had to read to report
/// anything about the model.
pub fn convert(
    dir: &Path,
    checkpoint: &Checkpoint,
    lengths: &[usize],
    opts: &encoder::Options,
) -> Result<encoder::EncoderConfig> {
    let text = std::fs::read_to_string(&checkpoint.config)
        .with_context(|| format!("reading {}", checkpoint.config.display()))?;
    let cfg = encoder::EncoderConfig::from_json(&text)?;

    // Before opening 500MB of weights.
    cfg.check_lengths(lengths)
        .with_context(|| format!("converting {}", checkpoint.source))?;

    let weights = safetensors::Checkpoint::open(&checkpoint.weights)?;
    let provenance = Provenance {
        source: checkpoint.source.clone(),
        lengths: lengths.to_vec(),
        quantized_embeddings: opts.quantize_embeddings,
        quantized_projections: opts.quantize_projections,
        activation: (cfg.activation != Activation::default()).then(|| cfg.activation.name()),
    };
    let (model, blob) = encoder::emit_with(&cfg, &weights, lengths, &provenance, opts)?;

    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    write_package(&dir.join(bundle_name(lengths)), &model, &blob)?;

    // A converted directory has to be self-contained: the checkpoint it came from
    // may be a Hugging Face cache entry that is cleared independently.
    std::fs::copy(&checkpoint.config, dir.join("config.json"))
        .with_context(|| format!("copying {}", checkpoint.config.display()))?;
    std::fs::copy(&checkpoint.tokenizer, dir.join("tokenizer.json"))
        .with_context(|| format!("copying {}", checkpoint.tokenizer.display()))?;
    if let Some(pooling) = &checkpoint.pooling {
        let into = dir.join("1_Pooling");
        std::fs::create_dir_all(&into).with_context(|| format!("creating {}", into.display()))?;
        std::fs::copy(pooling, into.join("config.json"))
            .with_context(|| format!("copying {}", pooling.display()))?;
    }
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_matches_the_shape_coreml_writes() {
        let text = manifest();
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(parsed["fileFormatVersion"], "1.0.0");
        assert_eq!(parsed["rootModelIdentifier"], MODEL_ID);
        let entries = parsed["itemInfoEntries"].as_object().expect("entries");
        assert_eq!(entries.len(), 2);
        // The root identifier must name an entry, and that entry must be the spec.
        assert_eq!(entries[MODEL_ID]["name"], "model.mlmodel");
        assert_eq!(entries[MODEL_ID]["path"], "com.apple.CoreML/model.mlmodel");
        assert_eq!(entries[WEIGHTS_ID]["name"], "weights");
        assert_eq!(entries[WEIGHTS_ID]["path"], "com.apple.CoreML/weights");
    }

    #[test]
    fn a_written_package_has_the_layout_the_backend_looks_for() {
        let dir = std::env::temp_dir().join(format!("kohagi-pkg-{}.mlpackage", std::process::id()));
        let x = mil::Tensor::new("x", mil::DType::Fp16, &[1, 3]);
        let mut b = mil::Builder::new(std::slice::from_ref(&x));
        let y = b.op(
            "identity",
            mil::Tensor::new("y", mil::DType::Fp16, &[1, 3]),
            &[("x", &x)],
        );
        b.returns(&y);
        let m = model(
            b.finish(),
            Io {
                inputs: vec![x],
                outputs: vec![y],
            },
        );
        let mut w = blob::Writer::new();
        w.write_f32_as_fp16(&[1.0]);
        write_package(&dir, &m, &w.finish()).expect("writes");

        assert!(dir.join("Manifest.json").is_file());
        assert!(dir.join("Data/com.apple.CoreML/model.mlmodel").is_file());
        assert!(dir
            .join("Data/com.apple.CoreML/weights/weight.bin")
            .is_file());

        // And the description declares the interface, not the internals.
        let bytes = std::fs::read(dir.join("Data/com.apple.CoreML/model.mlmodel")).unwrap();
        let back = Model::decode(bytes.as_slice()).expect("decodes");
        let d = back.description.expect("a description");
        assert_eq!(d.input.len(), 1);
        assert_eq!(d.input[0].name, "x");
        assert_eq!(d.output[0].name, "y");
        assert_eq!(back.specification_version, mil::SPECIFICATION_VERSION);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writing_twice_produces_the_same_bytes() {
        // Generation is reproducible: the manifest identifiers are fixed and the
        // program's maps are sorted, so nothing here varies between runs.
        let build = || {
            let x = mil::Tensor::new("x", mil::DType::Fp16, &[1, 2]);
            let mut b = mil::Builder::new(std::slice::from_ref(&x));
            let w = b.const_blob(mil::Tensor::new("w", mil::DType::Fp16, &[2, 2]), 64);
            let y = b.op(
                "matmul",
                mil::Tensor::new("y", mil::DType::Fp16, &[1, 2]),
                &[("x", &x), ("y", &w)],
            );
            b.returns(&y);
            model(
                b.finish(),
                Io {
                    inputs: vec![x],
                    outputs: vec![y],
                },
            )
            .encode_to_vec()
        };
        assert_eq!(build(), build());
    }
}
