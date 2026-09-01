//! Kohagi's HTTP face: the same model as `kohagi`, answering an
//! OpenAI-compatible `POST /v1/embeddings` instead of stdin. See
//! PROTOCOL-http.md.
//!
//! A third binary, like `kohagi-rerank`: `kohagi` is a pipe and stays one.
//! This is for the caller that wants one model loaded per host rather than
//! one per process, which is what a pipe cannot give a Rails cluster or a
//! sidecar. The model flags are `kohagi`'s; what is new is where to listen
//! and the limits on a request.
//!
//! Exit codes at load are the CLI's: 1 = fatal (model load, bad flags, the
//! address in use), 3 = the requested CoreML backend cannot serve this
//! request. Once listening it runs until SIGTERM or SIGINT, prints one
//! summary line, and exits 0.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;

use kohagi::cli::{self, BackendArg, CoreMlFormArg, CoreMlQuantizeArg, PoolingArg, PrecisionArg};
use kohagi::serve::{self, Listen};
use kohagi::{Embedder, ModelSource, Options};

/// Serve local sentence embeddings over HTTP, OpenAI-compatible.
///
/// Loads the model, listens on --listen, and answers POST /v1/embeddings (plus
/// GET /v1/models and GET /health) until SIGTERM or SIGINT. The model is
/// downloaded from the Hugging Face Hub on first use and cached
/// (~/.cache/huggingface).
#[derive(Parser)]
#[command(name = "kohagi-serve", version)]
struct Args {
    /// Where to listen: `host:port`, or `unix:///path` for a Unix socket (mode
    /// 0600, replaced if a previous run left it). Loopback by default; this
    /// server has no authentication, so keep it off the open network as you
    /// would a database.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: Listen,
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
    /// Prefix prepended to every input text before embedding, as `kohagi
    /// --prefix` does. One server has one prefix: run it with "検索クエリ: "
    /// for queries, or with none and have callers prepend their own.
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
    /// `--features metal`. coreml (Apple Neural Engine) requires `--features
    /// coreml`; with no --coreml-dir or --coreml-model-id it converts
    /// --model-id itself and caches the result.
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    device: BackendArg,
    /// Directory of pre-converted CoreML models for `--device coreml`: one
    /// `seq-<N>.mlpackage` per bucket length, plus tokenizer.json and
    /// config.json. Omit it to convert --model-id on first use.
    #[arg(long)]
    coreml_dir: Option<PathBuf>,
    /// Hugging Face repo holding the CoreML models (same layout as
    /// --coreml-dir), downloaded and cached on first use. --coreml-dir wins if
    /// both are set.
    #[arg(long)]
    coreml_model_id: Option<String>,
    /// Fixed sequence lengths to emit when `--device coreml` converts a
    /// checkpoint itself. Match them to the lengths your texts actually are.
    #[arg(long, value_delimiter = ',', default_values_t = [64usize, 128, 256, 512])]
    coreml_buckets: Vec<usize>,
    /// Quantize the model when `--device coreml` converts it. Omit for fp16;
    /// a quantized bundle's vectors are not interchangeable with an fp16 one's.
    #[arg(long, value_enum)]
    coreml_quantize: Option<CoreMlQuantizeArg>,
    /// When a --coreml-model-id repo ships both forms of a bucket, which to
    /// download: `compiled` (.mlmodelc, faster) or `package` (.mlpackage,
    /// portable).
    #[arg(long, value_enum, default_value_t = CoreMlFormArg::Compiled)]
    coreml_prefer: CoreMlFormArg,
    /// Skip L2 normalization (normalized output is the default; unit vectors
    /// make dot product = cosine). Also refuses a request's `dimensions`,
    /// which re-normalizes.
    #[arg(long)]
    no_normalize: bool,
    /// Keep only the first N dimensions of each embedding and re-normalize
    /// (Matryoshka truncation) on every reply; a request's `dimensions` may
    /// then only go lower. Refused if N is 0 or exceeds the model dimension,
    /// or combined with --no-normalize.
    #[arg(long, value_name = "N")]
    dims: Option<usize>,
    /// Refuse to start unless the loaded weights' sha256 starts with this hex
    /// prefix (the 12 digits from a summary line, or the full digest). A
    /// mismatch exits 1 before listening, so the wrong checkpoint never
    /// answers a request.
    #[arg(long, value_name = "HEX")]
    expect_sha256: Option<String>,
    /// Token-level truncation length.
    #[arg(long, default_value_t = 512)]
    max_seq_length: usize,
    /// Bucketing granularity; memory stays bounded regardless (see model.rs).
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// The most `input` items one request may carry; more is refused with
    /// 400. OpenAI's own limit, and it bounds one reply's size.
    #[arg(long, default_value_t = 2048)]
    max_inputs: usize,
    /// The longest request body read; longer is refused with 413.
    #[arg(long, default_value_t = 32 * 1024 * 1024, value_name = "BYTES")]
    max_body_bytes: usize,
    /// Requests allowed to wait for the model at once. The model answers one
    /// request at a time (one forward pass uses every core); a request that
    /// finds this many already waiting is refused with 503 and Retry-After
    /// rather than queued behind them.
    #[arg(long, default_value_t = 64)]
    max_queue: usize,
    /// After SIGTERM or SIGINT, how long to let open connections finish the
    /// reply in flight before closing them.
    #[arg(long, default_value_t = 30, value_name = "SECONDS")]
    shutdown_timeout: u64,
}

impl Args {
    fn options(&self) -> Options {
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

    /// Where to load the model from, plus the name to show in replies.
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

fn run(args: Args) -> anyhow::Result<()> {
    let (source, label) = args.source()?;
    let config = serve::Config {
        listen: args.listen.clone(),
        label,
        prefix: args.prefix.clone(),
        normalize: !args.no_normalize,
        max_inputs: args.max_inputs,
        max_body_bytes: args.max_body_bytes,
        max_queue: args.max_queue,
        shutdown_timeout: Duration::from_secs(args.shutdown_timeout),
    };
    let options = args.options();
    let expected = args.expect_sha256.clone();

    // Runs on the model's own thread; the model never leaves it. The digest
    // check happens there too, before anything can be answered.
    serve::run(config, move || {
        let embedder = Embedder::load(&source, options)?;
        cli::verify_fingerprint(expected.as_deref(), &embedder.info())?;
        Ok(embedder)
    })
}

fn main() -> ExitCode {
    kohagi::program::set("kohagi-serve");
    cli::exit_code(run(Args::parse()).map(|()| 0))
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
