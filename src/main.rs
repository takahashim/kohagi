//! kohagi CLI: JSONL embedding over stdin/stdout, plus a `--text` one-shot
//! mode for quick checks. See PROTOCOL.md for the full contract.
//!
//! Exit codes: 0 = all input embedded, 2 = finished but some lines were
//! skipped (see stderr), 1 = fatal (model load, I/O, bad flags).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use kohagi::{stdio, Embedder, ModelSource, Options, Pooling};

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
    /// Pooling: mean (Ruri v3, modernbert-embed) or cls.
    #[arg(long, default_value = "mean")]
    pooling: String,
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

fn run(args: Args) -> anyhow::Result<usize> {
    let pooling = match args.pooling.as_str() {
        "mean" => Pooling::Mean,
        "cls" => Pooling::Cls,
        other => anyhow::bail!("invalid pooling '{other}' (expected mean or cls)"),
    };
    let opts = Options {
        pooling,
        normalize: !args.no_normalize,
        max_seq_length: args.max_seq_length,
        batch_size: args.batch_size,
    };
    let (source, label) = match (&args.model_path, &args.tokenizer_path) {
        (Some(model), Some(tokenizer)) => {
            let label = model.file_name().map_or_else(
                || model.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            (ModelSource::Files { model: model.clone(), tokenizer: tokenizer.clone() }, label)
        }
        _ => (ModelSource::Hub { repo: args.model_id.clone() }, args.model_id.clone()),
    };

    if !args.text.is_empty() {
        // One-shot mode: embed the arguments, print the same JSONL as stdio
        // mode with positional ids.
        #[derive(serde::Serialize)]
        struct Out<'a> {
            id: usize,
            embedding: &'a [f32],
        }
        let embedder = Embedder::load(&source, opts)?;
        let prefixed: Vec<String> =
            args.text.iter().map(|t| format!("{}{t}", args.prefix)).collect();
        let texts: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        for (id, vec) in embedder.embed(&texts)?.iter().enumerate() {
            println!("{}", serde_json::to_string(&Out { id, embedding: vec })?);
        }
        return Ok(0);
    }

    stdio::run(move || Embedder::load(&source, opts), &args.prefix, &label)
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
