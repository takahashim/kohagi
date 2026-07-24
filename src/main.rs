//! Kohagi CLI: JSONL embedding over stdin/stdout, plus a `--text` one-shot
//! mode for quick checks. See PROTOCOL.md for the full contract.
//!
//! Exit codes: 0 = all input embedded, 2 = finished but some lines were
//! skipped (see stderr), 1 = fatal (model load, I/O, bad flags), 3 = the
//! requested CoreML backend cannot serve this request (built without the
//! feature, no `--coreml-dir`, or `--max-seq-length` beyond the largest
//! converted bucket) — caught before any input is read, so the caller can
//! retry on `--device cpu`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};

use kohagi::{stdio, Backend, CoreMlForm, Embedder, ModelSource, Options, Pooling, Precision};

/// CLI spellings of the library enums, so `--help` lists the valid values and
/// clap rejects anything else before we do any work.
#[derive(Clone, Copy, ValueEnum)]
enum PoolingArg {
    /// Mask-aware mean over tokens (Ruri v3, modernbert-embed).
    Mean,
    /// First token only.
    Cls,
}

#[derive(Clone, Copy, ValueEnum)]
enum PrecisionArg {
    /// Matches the PyTorch reference; works on every CPU.
    F32,
    /// ~2x faster on x86_64 CPUs with AVX512-BF16, at cosine ~0.99999.
    Bf16,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    /// Apple Accelerate on macOS, candle's own gemm elsewhere.
    Cpu,
    /// Apple GPU. Needs a binary built with `--features metal`.
    Metal,
    /// Apple Neural Engine. Needs a binary built with `--features coreml` and
    /// `--coreml-dir` pointing at pre-converted fixed-shape models.
    Coreml,
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

#[derive(Clone, Copy, ValueEnum)]
enum CoreMlFormArg {
    /// Compiled `.mlmodelc` — no per-run compile (default).
    Compiled,
    /// Portable `.mlpackage` — compiled on load, robust across OS versions.
    Package,
}

impl From<CoreMlFormArg> for CoreMlForm {
    fn from(f: CoreMlFormArg) -> Self {
        match f {
            CoreMlFormArg::Compiled => CoreMlForm::Compiled,
            CoreMlFormArg::Package => CoreMlForm::Package,
        }
    }
}

impl From<BackendArg> for Backend {
    fn from(b: BackendArg) -> Self {
        match b {
            BackendArg::Cpu => Backend::Cpu,
            BackendArg::Metal => Backend::Metal,
            BackendArg::Coreml => Backend::CoreML,
        }
    }
}

/// Local sentence embeddings for Ruri v3 / ModernBERT models.
///
/// Reads {"id","text"} JSONL on stdin and writes {"id","embedding"} JSONL on
/// stdout; or embeds --text arguments directly. The model is downloaded from
/// the Hugging Face Hub on first use and cached (~/.cache/huggingface).
#[derive(Parser)]
#[command(name = "kohagi", version)]
struct Args {
    /// Hugging Face model repo to download (ignored with --model-path).
    #[arg(long, default_value = "cl-nagoya/ruri-v3-130m")]
    model_id: String,
    /// Local safetensors weights (offline mode; config.json must sit next to
    /// it). Requires --tokenizer-path.
    #[arg(long, requires = "tokenizer_path")]
    model_path: Option<PathBuf>,
    /// Local tokenizer.json (offline mode).
    #[arg(long, requires = "model_path")]
    tokenizer_path: Option<PathBuf>,
    /// Prefix prepended to every text before embedding. Ruri v3 task
    /// prefixes: "検索文書: ", "検索クエリ: ", "トピック: ", or "" (plain
    /// sentence similarity).
    #[arg(long, default_value = "")]
    prefix: String,
    /// How to reduce token embeddings to one vector per text. Omit to take the
    /// model's own choice from its 1_Pooling/config.json (mean if it ships
    /// none); pass this only to override that.
    #[arg(long, value_enum)]
    pooling: Option<PoolingArg>,
    /// Numeric precision of the forward pass. f32 is identical everywhere;
    /// bf16 is faster but not bit-identical.
    #[arg(long, value_enum, default_value_t = PrecisionArg::F32)]
    precision: PrecisionArg,
    /// Device for the forward pass. metal requires a binary built with
    /// `--features metal`, and runs ~1.2x faster than cpu on Apple Silicon.
    /// coreml (Apple Neural Engine) requires `--features coreml` and
    /// `--coreml-dir`.
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    device: BackendArg,
    /// Directory of pre-converted CoreML models for `--device coreml`: one
    /// `seq-<N>.mlpackage` per bucket length, plus tokenizer.json and
    /// config.json. Produce one with scripts/convert_coreml.py.
    #[arg(long)]
    coreml_dir: Option<PathBuf>,
    /// Hugging Face repo holding the CoreML models (same layout as
    /// --coreml-dir), downloaded and cached on first use. Alternative to
    /// --coreml-dir for `--device coreml`; --coreml-dir wins if both are set.
    #[arg(long)]
    coreml_model_id: Option<String>,
    /// When a --coreml-model-id repo ships both forms of a bucket, which to
    /// download: `compiled` (.mlmodelc, faster) or `package` (.mlpackage,
    /// portable). Only the chosen form is fetched.
    #[arg(long, value_enum, default_value_t = CoreMlFormArg::Compiled)]
    coreml_prefer: CoreMlFormArg,
    /// Skip L2 normalization (normalized output is the default; unit vectors
    /// make dot product = cosine).
    #[arg(long)]
    no_normalize: bool,
    /// Token-level truncation length.
    #[arg(long, default_value_t = 512)]
    max_seq_length: usize,
    /// Bucketing granularity; memory stays bounded regardless (see model.rs).
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// Embed these texts and exit, instead of reading stdin. Repeatable;
    /// output ids are the argument positions (0, 1, …).
    #[arg(long)]
    text: Vec<String>,
}

impl Args {
    fn options(&self) -> Options {
        Options {
            pooling: self.pooling.map(Into::into),
            normalize: !self.no_normalize,
            max_seq_length: self.max_seq_length,
            batch_size: self.batch_size,
            precision: self.precision.into(),
            backend: self.device.into(),
            coreml_form: self.coreml_prefer.into(),
        }
    }

    /// Where to load the model from, plus the name to show in the summary.
    fn source(&self) -> anyhow::Result<(ModelSource, String)> {
        // CoreML loads pre-converted fixed-shape models — a local directory or
        // a Hub repo — not safetensors.
        if self.device == BackendArg::Coreml {
            if let Some(dir) = self.coreml_dir.clone() {
                let label = label_of(&dir);
                return Ok((ModelSource::CoreMl { dir }, label));
            }
            if let Some(repo) = self.coreml_model_id.clone() {
                let label = repo.clone();
                return Ok((ModelSource::CoreMlHub { repo }, label));
            }
            return Err(kohagi::UnsupportedRequest(
                "`--device coreml` requires `--coreml-dir <DIR>` or `--coreml-model-id <REPO>`"
                    .to_string(),
            )
            .into());
        }
        let out = match (&self.model_path, &self.tokenizer_path) {
            // clap's `requires` guarantees these two arrive together.
            (Some(model), Some(tokenizer)) => {
                let label = label_of(model);
                let source = ModelSource::Files {
                    model: model.clone(),
                    tokenizer: tokenizer.clone(),
                };
                (source, label)
            }
            _ => {
                let source = ModelSource::Hub {
                    repo: self.model_id.clone(),
                };
                (source, self.model_id.clone())
            }
        };
        Ok(out)
    }
}

/// A short display label for a model path: its file name, or the full path.
fn label_of(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// `--text` mode: embed the arguments and print the same JSONL that stdio
/// mode would, with the argument positions as ids.
fn embed_arguments(args: &Args, source: &ModelSource) -> anyhow::Result<()> {
    #[derive(serde::Serialize)]
    struct Out<'a> {
        id: usize,
        embedding: &'a [f32],
    }

    let embedder = Embedder::load(source, args.options())?;
    let prefixed: Vec<String> = args
        .text
        .iter()
        .map(|t| format!("{}{t}", args.prefix))
        .collect();
    let texts: Vec<&str> = prefixed.iter().map(String::as_str).collect();
    for (id, embedding) in embedder.embed(&texts)?.iter().enumerate() {
        println!("{}", serde_json::to_string(&Out { id, embedding })?);
    }
    Ok(())
}

/// Returns the number of skipped input lines (0 in `--text` mode).
fn run(args: Args) -> anyhow::Result<usize> {
    let (source, label) = args.source()?;

    if !args.text.is_empty() {
        embed_arguments(&args, &source)?;
        return Ok(0);
    }

    let opts = args.options();
    stdio::run(|| Embedder::load(&source, opts), &args.prefix, &label)
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(2),
        Err(e) => {
            eprintln!("kohagi: error: {e:#}");
            // A CoreML-unsupported request gets its own code so callers can
            // tell "retry on --device cpu" apart from a genuine failure.
            if e.chain().any(|c| c.is::<kohagi::UnsupportedRequest>()) {
                ExitCode::from(3)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}
