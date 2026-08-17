//! Kohagi CLI: JSONL embedding over stdin/stdout, plus a `--text` one-shot
//! mode for quick checks. See PROTOCOL.md for the full contract.
//!
//! Exit codes: 0 = all input embedded, 2 = finished but some lines were
//! skipped (see stderr), 1 = fatal (model load, I/O, bad flags), 3 = the
//! requested CoreML backend cannot serve this request (built without the
//! feature, no `--coreml-dir`, or `--max-seq-length` beyond the largest
//! converted bucket) — caught before any input is read, so the caller can
//! retry on `--device cpu`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};

use kohagi::cli::{self, BackendArg, CoreMlFormArg, CoreMlQuantizeArg, PoolingArg, PrecisionArg};
use kohagi::{stdio, Embedder, ModelSource, Options};

/// This binary's own output shape; the rest of the CLI value enums are shared
/// with `kohagi-rerank` in [`kohagi::cli`].
#[derive(Copy, Clone, ValueEnum)]
enum FormatArg {
    /// Kohagi's JSONL protocol: one record per line, ids echoed.
    Jsonl,
    /// One OpenAI `/v1/embeddings` response object for the whole run.
    Openai,
}

impl From<FormatArg> for stdio::Format {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Jsonl => stdio::Format::Jsonl,
            FormatArg::Openai => stdio::Format::OpenAi,
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
    /// Device for the forward pass. cuda requires an NVIDIA GPU and a binary
    /// built with `--features cuda`. metal requires a binary built with
    /// `--features metal`, and runs ~1.2x faster than cpu on Apple Silicon.
    /// coreml (Apple Neural Engine) requires `--features coreml`; with no
    /// --coreml-dir or --coreml-model-id it converts --model-id itself and
    /// caches the result.
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    device: BackendArg,
    /// Directory of pre-converted CoreML models for `--device coreml`: one
    /// `seq-<N>.mlpackage` per bucket length, plus tokenizer.json and
    /// config.json. Produce one with the coreml-convert binary or
    /// scripts/convert_coreml.py. Omit it to convert --model-id on first use.
    #[arg(long)]
    coreml_dir: Option<PathBuf>,
    /// Hugging Face repo holding the CoreML models (same layout as
    /// --coreml-dir), downloaded and cached on first use. Alternative to
    /// --coreml-dir for `--device coreml`; --coreml-dir wins if both are set.
    #[arg(long)]
    coreml_model_id: Option<String>,
    /// Fixed sequence lengths to emit when `--device coreml` converts a
    /// checkpoint itself (that is, when neither --coreml-dir nor
    /// --coreml-model-id is given). Each becomes one CoreML function over a
    /// single shared copy of the weights, so the set costs no disk; what it
    /// costs is one model to open per length. Match it to the lengths your
    /// texts actually are — a bucket nothing lands in is pure overhead.
    #[arg(long, value_delimiter = ',', default_values_t = [64usize, 128, 256, 512])]
    coreml_buckets: Vec<usize>,
    /// Quantize the model when `--device coreml` converts it. `embeddings`
    /// halves a large-vocabulary bundle at no measured retrieval cost;
    /// `all` roughly halves it again for a small one. Omit for fp16 — a
    /// quantized bundle's vectors are not interchangeable with an fp16 one's.
    #[arg(long, value_enum)]
    coreml_quantize: Option<CoreMlQuantizeArg>,
    /// When a --coreml-model-id repo ships both forms of a bucket, which to
    /// download: `compiled` (.mlmodelc, faster) or `package` (.mlpackage,
    /// portable). Only the chosen form is fetched.
    #[arg(long, value_enum, default_value_t = CoreMlFormArg::Compiled)]
    coreml_prefer: CoreMlFormArg,
    /// Shape of stdout. `jsonl` is Kohagi's protocol, one record per line,
    /// echoing each input's id. `openai` is one `/v1/embeddings` response
    /// object for the whole run, so code written against that API can read it
    /// unchanged — it identifies embeddings by position rather than by id, and
    /// an aborted run leaves an incomplete JSON document.
    #[arg(long, value_enum, default_value_t = FormatArg::Jsonl)]
    format: FormatArg,
    /// Add `n_tokens` (tokens embedded, specials included) and `truncated`
    /// (text ran past --max-seq-length, so its tail was dropped) to each output
    /// record. Off by default, keeping the plain protocol-1 output. The summary
    /// line always reports how many records were truncated regardless.
    #[arg(long)]
    report_tokens: bool,
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
    /// Load the model, print one line of JSON describing it on stdout, and
    /// exit without reading stdin. The weights' sha256 is in there: record it
    /// beside a result and the question "which checkpoint produced this" has
    /// an answer that a renamed directory cannot change.
    #[arg(long, conflicts_with = "text")]
    print_model_info: bool,
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
        let checkpoint = cli::checkpoint_source(
            self.model_path.as_ref(),
            self.tokenizer_path.as_ref(),
            &self.model_id,
        );
        // CoreML loads converted fixed-shape models rather than safetensors.
        if self.device == BackendArg::Coreml {
            return cli::coreml_source(
                self.coreml_dir.as_ref(),
                self.coreml_model_id.as_deref(),
                &self.coreml_buckets,
                self.coreml_quantize.map(Into::into).unwrap_or_default(),
                checkpoint,
            );
        }
        Ok(checkpoint)
    }
}

/// `--text` mode: embed the arguments and print what stdio mode would, with the
/// argument positions as ids.
fn embed_arguments(args: &Args, source: &ModelSource, label: &str) -> anyhow::Result<()> {
    let embedder = Embedder::load(source, args.options())?;
    let prefixed: Vec<String> = args
        .text
        .iter()
        .map(|t| format!("{}{t}", args.prefix))
        .collect();
    let texts: Vec<&str> = prefixed.iter().map(String::as_str).collect();
    let (vectors, tokens) = embedder.embed_with_tokens(&texts)?;

    // Same output shape as stdio mode; ids are the argument positions.
    let mut out = stdio::Writer::new(
        std::io::stdout().lock(),
        args.format.into(),
        label,
        args.report_tokens,
    );
    for (id, (embedding, info)) in vectors.iter().zip(&tokens).enumerate() {
        out.record(&serde_json::Value::from(id), embedding, info)?;
    }
    out.finish()
}

/// Returns the number of skipped input lines (0 in `--text` mode).
fn run(args: Args) -> anyhow::Result<usize> {
    let (source, label) = args.source()?;

    // The OpenAI item shape has nowhere to put per-record token counts, and
    // dropping what was asked for silently is worse than saying so. The total is
    // still reported, as `usage.prompt_tokens`.
    if args.report_tokens && matches!(args.format, FormatArg::Openai) {
        return Err(kohagi::UnsupportedRequest(
            "--report-tokens has no place in the OpenAI response shape; \
             usage.prompt_tokens carries the total, and per-record counts need \
             --format jsonl"
                .to_string(),
        )
        .into());
    }

    if args.print_model_info {
        let embedder = Embedder::load(&source, args.options())?;
        cli::print_model_info(&label, &embedder.info())?;
        return Ok(0);
    }

    if !args.text.is_empty() {
        embed_arguments(&args, &source, &label)?;
        return Ok(0);
    }

    let opts = args.options();
    stdio::run(
        || Embedder::load(&source, opts),
        &args.prefix,
        args.report_tokens,
        &label,
        args.format.into(),
    )
}

fn main() -> ExitCode {
    cli::exit_code("kohagi", run(Args::parse()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_help_and_flags_are_well_formed() {
        use clap::CommandFactory;
        Args::command().debug_assert();
    }
}
