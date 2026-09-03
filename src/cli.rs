//! What the binaries share above the library: the CLI spellings of the
//! library's enums, the flags that choose and load an embedding model
//! ([`ModelArgs`]), how a model source is named on the command line, and how a
//! run's outcome becomes an exit code.
//!
//! `kohagi`, `kohagi-serve` and `kohagi-rerank` load models onto the same
//! devices and answer to the same exit codes; what differs is the record they
//! read and write, or whether they read one at all. Keeping the mapping here
//! means `--device`, `--precision` and the exit codes cannot come to mean two
//! things, and a new binary inherits them rather than copying them.
//!
//! `kohagi` and `kohagi-serve` load the very same model, so they share the
//! flag definitions too, flattened from [`ModelArgs`]. A cross-encoder has
//! other defaults and another `Options` type, so `kohagi-rerank` and
//! `kohagi-serve` keep their own spellings for it (`--model-id` against
//! `--rerank-model-id`); what they do not keep their own copy of is what those
//! flags mean, which is [`RerankModel`].

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::ValueEnum;

use crate::program::remark;
use crate::rerank::{self, Reranker};

use crate::{
    Backend, CoreMlForm, CoreMlQuantize, Embedder, ModelInfo, ModelSource, Options, Pooling,
    Precision, UnsupportedRequest,
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

/// The flags that decide which model is loaded, and how: shared by `kohagi`
/// and `kohagi-serve` through `#[command(flatten)]`, so the two cannot come to
/// load the same checkpoint differently, and `--help` describes each flag in
/// one place. `kohagi-rerank` keeps its own: a cross-encoder has other
/// defaults (`--coreml-buckets` run longer, a pair fills more of a bucket than
/// a text does) and another `Options` type.
#[derive(Clone, clap::Args)]
pub struct ModelArgs {
    /// Hugging Face model repo to download (ignored with --model-path).
    #[arg(long, default_value = "cl-nagoya/ruri-v3-130m")]
    pub model_id: String,
    /// Local safetensors weights (offline mode; config.json must sit next to
    /// it). Requires --tokenizer-path.
    #[arg(long, requires = "tokenizer_path")]
    pub model_path: Option<PathBuf>,
    /// Local tokenizer.json (offline mode).
    #[arg(long, requires = "model_path")]
    pub tokenizer_path: Option<PathBuf>,
    /// How to reduce token embeddings to one vector per text. Omit to take the
    /// model's own choice from its 1_Pooling/config.json (mean if it ships
    /// none); pass this only to override that.
    #[arg(long, value_enum)]
    pub pooling: Option<PoolingArg>,
    /// Numeric precision of the forward pass. f32 is identical everywhere;
    /// bf16 is faster but not bit-identical.
    #[arg(long, value_enum, default_value_t = PrecisionArg::F32)]
    pub precision: PrecisionArg,
    /// Device for the forward pass. cuda requires an NVIDIA GPU and a binary
    /// built with `--features cuda`. metal requires a binary built with
    /// `--features metal`, and runs ~1.2x faster than cpu on Apple Silicon.
    /// coreml (Apple Neural Engine) requires `--features coreml`; with no
    /// --coreml-dir or --coreml-model-id it converts --model-id itself and
    /// caches the result.
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    pub device: BackendArg,
    /// Directory of pre-converted CoreML models for `--device coreml`: one
    /// `seq-<N>.mlpackage` per bucket length, plus tokenizer.json and
    /// config.json. Produce one with the coreml-convert binary or
    /// scripts/convert_coreml.py. Omit it to convert --model-id on first use.
    #[arg(long)]
    pub coreml_dir: Option<PathBuf>,
    /// Hugging Face repo holding the CoreML models (same layout as
    /// --coreml-dir), downloaded and cached on first use. Alternative to
    /// --coreml-dir for `--device coreml`; --coreml-dir wins if both are set.
    #[arg(long)]
    pub coreml_model_id: Option<String>,
    /// Fixed sequence lengths to emit when `--device coreml` converts a
    /// checkpoint itself (that is, when neither --coreml-dir nor
    /// --coreml-model-id is given). Each becomes one CoreML function over a
    /// single shared copy of the weights, so the set costs no disk; what it
    /// costs is one model to open per length. Match it to the lengths your
    /// texts actually are: a bucket nothing lands in is pure overhead.
    #[arg(long, value_delimiter = ',', default_values_t = [64usize, 128, 256, 512])]
    pub coreml_buckets: Vec<usize>,
    /// Quantize the model when `--device coreml` converts it. `embeddings`
    /// halves a large-vocabulary bundle at no measured retrieval cost;
    /// `all` roughly halves it again for a small one. Omit for fp16; a
    /// quantized bundle's vectors are not interchangeable with an fp16 one's.
    #[arg(long, value_enum)]
    pub coreml_quantize: Option<CoreMlQuantizeArg>,
    /// When a --coreml-model-id repo ships both forms of a bucket, which to
    /// download: `compiled` (.mlmodelc, faster) or `package` (.mlpackage,
    /// portable). Only the chosen form is fetched.
    #[arg(long, value_enum, default_value_t = CoreMlFormArg::Compiled)]
    pub coreml_prefer: CoreMlFormArg,
    /// Skip L2 normalization (normalized output is the default; unit vectors
    /// make dot product = cosine).
    #[arg(long)]
    pub no_normalize: bool,
    /// Keep only the first N dimensions of each embedding and re-normalize
    /// (Matryoshka truncation, meaningful for models trained for it). dot =
    /// cosine still holds on the shorter vectors, and they must not share an
    /// index with full-dimension ones. Refused if N is 0 or exceeds the model
    /// dimension, or combined with --no-normalize.
    #[arg(long, value_name = "N")]
    pub dims: Option<usize>,
    /// Refuse to embed anything unless the loaded weights' sha256 starts with
    /// this hex prefix: paste the 12 digits from a summary line, or the full
    /// digest from --print-model-info or /v1/models. A mismatch exits 1 before
    /// anything is answered, so the wrong checkpoint cannot survive into
    /// results. With --device coreml the bundle's recorded source_sha256 is
    /// checked instead, and a bundle that recorded none is refused.
    #[arg(long, value_name = "HEX")]
    expect_sha256: Option<String>,
    /// Token-level truncation length.
    #[arg(long, default_value_t = 512)]
    pub max_seq_length: usize,
    /// Bucketing granularity; memory stays bounded regardless (see model.rs).
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
}

impl ModelArgs {
    pub fn options(&self) -> Options {
        Options {
            pooling: self.pooling.map(Into::into),
            normalize: !self.no_normalize,
            dims: self.dims,
            max_seq_length: self.max_seq_length,
            batch_size: self.batch_size,
            precision: self.precision.into(),
            backend: self.device.into(),
            coreml_form: self.coreml_prefer.into(),
        }
    }

    /// Where to load the model from, plus the name to show for it (in a
    /// summary line, or a reply): the `--model-id` repo, or the directory of a
    /// `--model-path` checkpoint.
    pub fn source(&self) -> anyhow::Result<(ModelSource, String)> {
        let checkpoint = checkpoint_source(
            self.model_path.as_ref(),
            self.tokenizer_path.as_ref(),
            &self.model_id,
        );
        // CoreML loads converted fixed-shape models rather than safetensors.
        if self.device == BackendArg::Coreml {
            return coreml_source(
                self.coreml_dir.as_ref(),
                self.coreml_model_id.as_deref(),
                &self.coreml_buckets,
                self.coreml_quantize.map(Into::into).unwrap_or_default(),
                checkpoint,
            );
        }
        Ok(checkpoint)
    }

    /// Load the model from `source` and, when `--expect-sha256` pinned a
    /// digest, refuse weights that do not carry it, before anything is
    /// embedded, whichever binary loads.
    pub fn load(&self, source: &ModelSource) -> anyhow::Result<Embedder> {
        let embedder = Embedder::load(source, self.options())?;
        verify_fingerprint(self.expect_sha256.as_deref(), &embedder.info())?;
        Ok(embedder)
    }
}

/// The sequence lengths `--device coreml` converts a cross-encoder to when it
/// converts one itself. A pair fills more of a bucket than a single text does,
/// so these run longer than [`ModelArgs`]'s embedding defaults.
pub const RERANK_COREML_BUCKETS: [usize; 3] = [128, 256, 512];

/// How a cross-encoder is loaded, from whichever binary's flags carry it.
///
/// `kohagi-rerank` spells these `--model-id`, `--device` and so on;
/// `kohagi-serve` spells them `--rerank-model-id` and takes the device and the
/// precision from the embedder it runs beside. The spellings are each binary's,
/// because a cross-encoder's defaults are not an embedder's, but which
/// checkpoint they name and how it is loaded is decided once, here: the same
/// flags through either binary must load the same reranker.
pub struct RerankModel<'a> {
    /// Local safetensors weights, with [`Self::tokenizer_path`] (offline mode).
    pub model_path: Option<&'a PathBuf>,
    pub tokenizer_path: Option<&'a PathBuf>,
    /// The Hugging Face repo, used unless the two paths above are given.
    pub model_id: &'a str,
    pub device: BackendArg,
    pub precision: PrecisionArg,
    pub coreml_dir: Option<&'a PathBuf>,
    pub coreml_model_id: Option<&'a str>,
    /// What to convert when `--device coreml` converts the checkpoint itself;
    /// [`RERANK_COREML_BUCKETS`] is what both binaries default it to.
    pub coreml_buckets: &'a [usize],
    pub coreml_prefer: CoreMlFormArg,
    pub max_seq_length: usize,
    pub batch_size: usize,
    /// Report the sigmoid rather than the raw logit.
    pub sigmoid: bool,
    /// `--expect-sha256`, when the caller pinned a digest.
    pub expect_sha256: Option<&'a str>,
}

impl RerankModel<'_> {
    pub fn options(&self) -> rerank::Options {
        rerank::Options {
            max_seq_length: self.max_seq_length,
            batch_size: self.batch_size,
            precision: self.precision.into(),
            backend: self.device.into(),
            sigmoid: self.sigmoid,
            coreml_form: self.coreml_prefer.into(),
        }
    }

    /// Where the reranker comes from, plus the name to show for it.
    pub fn source(&self) -> anyhow::Result<(ModelSource, String)> {
        let checkpoint = checkpoint_source(self.model_path, self.tokenizer_path, self.model_id);
        // CoreML loads converted fixed-shape models rather than safetensors.
        // Never quantized: a reranker's output is one number compared against a
        // threshold, and int8 moves it further than fp16 already does.
        if self.device == BackendArg::Coreml {
            return coreml_source(
                self.coreml_dir,
                self.coreml_model_id,
                self.coreml_buckets,
                CoreMlQuantize::None,
                checkpoint,
            );
        }
        Ok(checkpoint)
    }

    /// Load from `source` and, when `--expect-sha256` pinned a digest, refuse
    /// weights that do not carry it, before any pair is scored or served. A
    /// threshold belongs to the weights it was tuned on, so both binaries owe
    /// the caller this check.
    pub fn load(&self, source: &ModelSource) -> anyhow::Result<Reranker> {
        let reranker = Reranker::load(source, self.options())?;
        verify_fingerprint(self.expect_sha256, &reranker.info())?;
        Ok(reranker)
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
