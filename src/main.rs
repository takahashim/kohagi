//! kohagi CLI: JSONL embedding over stdin/stdout, plus a `--text` one-shot
//! mode for quick checks. See PROTOCOL.md for the full contract.
//!
//! Exit codes: 0 = all input embedded, 2 = finished but some lines were
//! skipped (see stderr), 1 = fatal (model load, I/O, bad flags).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};

use kohagi::{stdio, Backend, Embedder, ModelSource, Options, Pooling, Precision};

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

#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    /// Apple Accelerate on macOS, candle's own gemm elsewhere.
    Cpu,
    /// Apple GPU. Needs a binary built with `--features metal`.
    Metal,
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
    /// How to reduce token embeddings to one vector per text.
    #[arg(long, value_enum, default_value_t = PoolingArg::Mean)]
    pooling: PoolingArg,
    /// Numeric precision of the forward pass. f32 is identical everywhere;
    /// bf16 is faster but not bit-identical.
    #[arg(long, value_enum, default_value_t = PrecisionArg::F32)]
    precision: PrecisionArg,
    /// Device for the forward pass. metal requires a binary built with
    /// `--features metal`, and runs ~1.2x faster than cpu on Apple Silicon.
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    device: BackendArg,
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
            pooling: self.pooling.into(),
            normalize: !self.no_normalize,
            max_seq_length: self.max_seq_length,
            batch_size: self.batch_size,
            precision: self.precision.into(),
            backend: self.device.into(),
        }
    }

    /// Where to load the model from, plus the name to show in the summary.
    fn source(&self) -> (ModelSource, String) {
        match (&self.model_path, &self.tokenizer_path) {
            // clap's `requires` guarantees these two arrive together.
            (Some(model), Some(tokenizer)) => {
                let label = model.file_name().map_or_else(
                    || model.display().to_string(),
                    |n| n.to_string_lossy().into_owned(),
                );
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
        }
    }
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
    let (source, label) = args.source();

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
            ExitCode::FAILURE
        }
    }
}
