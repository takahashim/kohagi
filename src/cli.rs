//! What `kohagi` and `kohagi-rerank` share above the library: the CLI
//! spellings of the library's enums, how a model source is named on the command
//! line, and how a run's outcome becomes an exit code.
//!
//! Both binaries load the same models onto the same devices and answer to the
//! same exit codes; only the record they read and write differs. Keeping the
//! mapping here means `--device`, `--precision` and the exit codes cannot come
//! to mean two things — and that a third binary would inherit them rather than
//! copy them.
//!
//! Flag *definitions* stay in each binary. Their help text is about that
//! binary's job (`--coreml-buckets` defaults differ, because a pair fills more
//! of a bucket than a text does), and the two `Args` structs share no fields
//! worth naming in common.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::ValueEnum;

use crate::{
    Backend, CoreMlForm, CoreMlQuantize, ModelInfo, ModelSource, Pooling, Precision,
    UnsupportedRequest,
};

/// CLI spellings of the library enums, so `--help` lists the valid values and
/// clap rejects anything else before we do any work.
#[derive(Clone, Copy, ValueEnum)]
pub enum PoolingArg {
    /// Mask-aware mean over tokens (Ruri v3, modernbert-embed).
    Mean,
    /// First token only.
    Cls,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum PrecisionArg {
    /// Matches the PyTorch reference; works on every CPU.
    F32,
    /// ~2x faster on x86_64 CPUs with AVX512-BF16; not bit-identical to f32.
    Bf16,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendArg {
    /// Apple Accelerate on macOS, candle's own gemm elsewhere.
    Cpu,
    /// Apple GPU. Needs a binary built with `--features metal`.
    Metal,
    /// NVIDIA GPU via CUDA. Needs a binary built with `--features cuda`.
    Cuda,
    /// Apple Neural Engine. Needs `--features coreml`; converts --model-id
    /// itself unless given a converted bundle.
    Coreml,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum CoreMlFormArg {
    /// Compiled `.mlmodelc` — no per-run compile (default).
    Compiled,
    /// Portable `.mlpackage` — compiled on load, robust across OS versions.
    Package,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum CoreMlQuantizeArg {
    /// The embedding table int8, one scale per row.
    Embeddings,
    /// The embedding table and every projection int8.
    All,
}

impl From<PoolingArg> for Pooling {
    fn from(p: PoolingArg) -> Self {
        match p {
            PoolingArg::Mean => Pooling::Mean,
            PoolingArg::Cls => Pooling::Cls,
        }
    }
}

impl From<PrecisionArg> for Precision {
    fn from(p: PrecisionArg) -> Self {
        match p {
            PrecisionArg::F32 => Precision::F32,
            PrecisionArg::Bf16 => Precision::Bf16,
        }
    }
}

impl From<BackendArg> for Backend {
    fn from(b: BackendArg) -> Self {
        match b {
            BackendArg::Cpu => Backend::Cpu,
            BackendArg::Metal => Backend::Metal,
            BackendArg::Cuda => Backend::Cuda,
            BackendArg::Coreml => Backend::CoreML,
        }
    }
}

impl From<CoreMlFormArg> for CoreMlForm {
    fn from(f: CoreMlFormArg) -> Self {
        match f {
            CoreMlFormArg::Compiled => CoreMlForm::Compiled,
            CoreMlFormArg::Package => CoreMlForm::Package,
        }
    }
}

impl From<CoreMlQuantizeArg> for CoreMlQuantize {
    fn from(q: CoreMlQuantizeArg) -> Self {
        match q {
            CoreMlQuantizeArg::Embeddings => CoreMlQuantize::Embeddings,
            CoreMlQuantizeArg::All => CoreMlQuantize::All,
        }
    }
}

/// A short display label for a model path: its file name, or the full path.
///
/// Except when that file name is `model.safetensors`, which every checkpoint's
/// is. The directory is the part a caller chose — `alpha05`, `exec9` — so a
/// checkpoint reports that instead; otherwise a summary line says
/// `model=model.safetensors` for every fine-tune on the machine, which is one
/// of the ways two runs get mistaken for each other in the first place.
pub fn label_of(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    if name == "model.safetensors" {
        if let Some(dir) = path.parent().and_then(Path::file_name) {
            return dir.to_string_lossy().into_owned();
        }
    }
    name
}

/// The safetensors checkpoint a run names, with its display label. Used by the
/// candle backends directly, and by CoreML as the thing it converts.
///
/// `model` and `tokenizer` arrive together or not at all — clap's `requires`
/// sees to that — so anything else falls back to the Hub repo.
pub fn checkpoint_source(
    model: Option<&PathBuf>,
    tokenizer: Option<&PathBuf>,
    repo: &str,
) -> (ModelSource, String) {
    match (model, tokenizer) {
        (Some(model), Some(tokenizer)) => (
            ModelSource::Files {
                model: model.clone(),
                tokenizer: tokenizer.clone(),
            },
            label_of(model),
        ),
        _ => (
            ModelSource::Hub {
                repo: repo.to_string(),
            },
            repo.to_string(),
        ),
    }
}

/// Where `--device coreml` gets its model: a converted directory, a Hub repo
/// holding one, or the checkpoint it converts on first use.
///
/// The last is why one `--model-id` serves every device.
pub fn coreml_source(
    dir: Option<&PathBuf>,
    repo: Option<&str>,
    buckets: &[usize],
    quantize: CoreMlQuantize,
    checkpoint: (ModelSource, String),
) -> anyhow::Result<(ModelSource, String)> {
    if let Some(dir) = dir {
        let label = label_of(dir);
        return Ok((ModelSource::CoreMl { dir: dir.clone() }, label));
    }
    if let Some(repo) = repo {
        return Ok((
            ModelSource::CoreMlHub {
                repo: repo.to_string(),
            },
            repo.to_string(),
        ));
    }
    let mut buckets = buckets.to_vec();
    buckets.sort_unstable();
    buckets.dedup();
    if buckets.is_empty() {
        return Err(UnsupportedRequest::new(
            "`--coreml-buckets` is empty; give at least one sequence length",
        )
        .into());
    }
    let (checkpoint, label) = checkpoint;
    Ok((
        ModelSource::CoreMlConvert {
            checkpoint: Box::new(checkpoint),
            buckets,
            quantize,
        },
        label,
    ))
}

/// `--print-model-info`: one JSON line naming the model a run would use.
///
/// One line so that a caller can `json.loads` the whole of stdout, and on
/// stdout rather than stderr because it is this mode's output rather than a
/// remark about it.
pub fn print_model_info(label: &str, info: &ModelInfo) -> anyhow::Result<()> {
    /// The model's own facts, plus the name the caller used for it — which the
    /// model does not know, since one model has many names.
    #[derive(serde::Serialize)]
    struct Printed<'a> {
        model: &'a str,
        #[serde(flatten)]
        info: &'a ModelInfo,
    }

    println!(
        "{}",
        serde_json::to_string(&Printed { model: label, info })?
    );
    Ok(())
}

/// The protocol's exit codes, from a run's skipped-line count.
///
/// 0 every record answered, 2 finished with lines skipped, 3 the CoreML backend
/// cannot serve this request (so the caller can retry on `--device cpu`), 1
/// anything else. See PROTOCOL.md; both binaries owe callers the same table.
pub fn exit_code(program: &str, outcome: anyhow::Result<usize>) -> ExitCode {
    match outcome {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(2),
        Err(e) => {
            eprintln!("{program}: error: {e:#}");
            if e.chain().any(|c| c.is::<UnsupportedRequest>()) {
                ExitCode::from(3)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_checkpoint_is_labelled_by_its_directory() {
        // The name every fine-tune's weights file has, so the directory is the
        // only part that distinguishes them.
        assert_eq!(
            label_of(Path::new("/m/interp/alpha05/model.safetensors")),
            "alpha05"
        );
        // Anything else names itself.
        assert_eq!(
            label_of(Path::new("/m/exec9-fp16.safetensors")),
            "exec9-fp16.safetensors"
        );
        // A CoreML directory is already the name a caller chose.
        assert_eq!(label_of(Path::new("/m/coreml/ruri-130m")), "ruri-130m");
        assert_eq!(
            label_of(Path::new("model.safetensors")),
            "model.safetensors"
        );
    }

    /// `--coreml-dir` wins over `--coreml-model-id`, and with neither the
    /// checkpoint is converted — the rule that lets one `--model-id` serve
    /// every device.
    #[test]
    fn coreml_takes_a_bundle_before_it_converts_one() {
        let checkpoint = || {
            (
                ModelSource::Hub {
                    repo: "org/model".to_string(),
                },
                "org/model".to_string(),
            )
        };
        let dir = PathBuf::from("/m/coreml/ruri-130m");

        let (source, label) = coreml_source(
            Some(&dir),
            Some("org/coreml"),
            &[128],
            CoreMlQuantize::None,
            checkpoint(),
        )
        .unwrap();
        assert!(matches!(source, ModelSource::CoreMl { .. }));
        assert_eq!(label, "ruri-130m");

        let (source, label) = coreml_source(
            None,
            Some("org/coreml"),
            &[128],
            CoreMlQuantize::None,
            checkpoint(),
        )
        .unwrap();
        assert!(matches!(source, ModelSource::CoreMlHub { .. }));
        assert_eq!(label, "org/coreml");

        // Duplicates and order are the caller's convenience, not the bundle's:
        // one bucket per length, ascending, is what gets converted.
        let (source, label) = coreml_source(
            None,
            None,
            &[512, 128, 128],
            CoreMlQuantize::None,
            checkpoint(),
        )
        .unwrap();
        match source {
            ModelSource::CoreMlConvert { buckets, .. } => assert_eq!(buckets, vec![128, 512]),
            _ => panic!("expected a conversion"),
        }
        assert_eq!(label, "org/model");

        assert!(coreml_source(None, None, &[], CoreMlQuantize::None, checkpoint()).is_err());
    }
}
