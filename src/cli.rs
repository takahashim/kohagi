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

use crate::program::remark;

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

/// `--expect-sha256`: refuse to work with weights whose digest does not start
/// with what the caller pinned.
///
/// The summary line and `--print-model-info` make a run's digest recordable;
/// this makes the record enforceable. A caller that wrote the digest beside
/// its results pastes it back — the summary's 12 digits or the full 64 — and a
/// renamed directory, a mixed-up interpolation, or a stale download then stops
/// the run before it answers anything, instead of surviving into the numbers.
///
/// A CoreML bundle has no weights file of its own, so the claim checked there
/// is the `source_sha256` its converter recorded — the checkpoint's digest,
/// which is the value a caller has. A bundle that recorded none cannot satisfy
/// the expectation and is refused rather than waved through.
///
/// `None` means the flag was not given: no expectation, nothing to enforce.
/// Taking the `Option` here keeps that decision in one place instead of at
/// every load site in every binary.
pub fn verify_fingerprint(expected: Option<&str>, info: &ModelInfo) -> anyhow::Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let want = expected.to_ascii_lowercase();
    anyhow::ensure!(
        !want.is_empty() && want.len() <= 64 && want.bytes().all(|b| b.is_ascii_hexdigit()),
        "--expect-sha256 takes a hex prefix of the digest (up to 64 digits), not `{expected}`"
    );
    let Some((claim, actual)) = info.digest() else {
        anyhow::bail!(
            "--expect-sha256 was given, but this model has no digest to check it against \
             (the weights could not be hashed, or this CoreML bundle was converted before \
             its provenance was recorded); an expectation that cannot be verified is \
             refused rather than assumed"
        )
    };
    anyhow::ensure!(
        actual.starts_with(&want),
        "these are not the expected weights: --expect-sha256 {want}, but the loaded \
         model's {claim} is {actual}"
    );
    Ok(())
}

/// The protocol's exit codes, from a run's skipped-line count.
///
/// 0 every record answered, 2 finished with lines skipped, 3 the CoreML backend
/// cannot serve this request (so the caller can retry on `--device cpu`), 1
/// anything else. See PROTOCOL.md; both binaries owe callers the same table.
pub fn exit_code(outcome: anyhow::Result<usize>) -> ExitCode {
    match outcome {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(2),
        Err(e) => {
            remark!("error: {e:#}");
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

    fn info(sha256: Option<&str>, source_sha256: Option<&str>) -> ModelInfo {
        ModelInfo {
            backend: "cpu",
            precision: "f32",
            sha256: sha256.map(str::to_string),
            // `source_sha256` applies only to converted bundles.
            bundle: source_sha256.map(|sha| crate::Bundle {
                source: None,
                source_sha256: Some(sha.to_string()),
                buckets: vec![512],
                quantization: "none".to_string(),
                graph_version: None,
            }),
            pooling: "mean",
            dim: 512,
            max_seq_length: 512,
            declared_max_seq_length: None,
            output: crate::Output::Embedding {
                output_dim: None,
                normalized: true,
            },
        }
    }

    /// The whole point of the flag: a digest prefix either matches the loaded
    /// weights or the run stops, and the summary's 12 digits are enough.
    #[test]
    fn an_expected_digest_is_matched_by_prefix() {
        let loaded = info(Some("1c342581efc23d0b50b92fb11ac1eeb0"), None);
        assert!(verify_fingerprint(Some("1c342581efc2"), &loaded).is_ok());
        assert!(verify_fingerprint(Some("1c342581efc23d0b50b92fb11ac1eeb0"), &loaded).is_ok());
        // Case is presentation, not identity: an uppercased paste still matches.
        assert!(verify_fingerprint(Some("1C342581EFC2"), &loaded).is_ok());

        let e = verify_fingerprint(Some("e831a463bddb"), &loaded)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("e831a463bddb") && e.contains("1c342581efc2"),
            "the error should show both digests: {e}"
        );
    }

    /// No flag, no expectation: nothing is enforced, even against a model
    /// that has no digest at all.
    #[test]
    fn no_expectation_passes_whatever_is_loaded() {
        assert!(verify_fingerprint(None, &info(None, None)).is_ok());
        assert!(verify_fingerprint(None, &info(Some("1c342581efc2"), None)).is_ok());
    }

    /// A CoreML bundle is checked against the checkpoint it was converted
    /// from, which is the digest a caller recorded; a bundle that recorded
    /// nothing cannot be verified and must not pass.
    #[test]
    fn a_bundle_is_checked_by_its_recorded_source() {
        let bundle = info(None, Some("e831a463bddb00112233445566778899"));
        assert!(verify_fingerprint(Some("e831a463bddb"), &bundle).is_ok());
        assert!(verify_fingerprint(Some("1c342581efc2"), &bundle).is_err());

        let unknown = info(None, None);
        let e = verify_fingerprint(Some("1c342581efc2"), &unknown)
            .unwrap_err()
            .to_string();
        assert!(e.contains("no digest"), "unexpected error: {e}");
    }

    /// A value that cannot be a digest prefix is a pasting mistake, not a
    /// mismatch, and the message should say so.
    #[test]
    fn a_non_hex_expectation_is_refused_as_such() {
        let loaded = info(Some("1c342581efc2"), None);
        for bad in ["", "sha256=1c342581", "1c34 2581", &"f".repeat(65)] {
            let e = verify_fingerprint(Some(bad), &loaded)
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("hex prefix"),
                "unexpected error for {bad:?}: {e}"
            );
        }
    }
}
