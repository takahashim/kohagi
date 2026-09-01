//! Kohagi's HTTP face: the same model as `kohagi`, answering an
//! OpenAI-compatible `POST /v1/embeddings` instead of stdin, and, when asked
//! to load one, the same cross-encoder as `kohagi-rerank` behind
//! `POST /v1/rerank`. See PROTOCOL-http.md.
//!
//! A third binary, like `kohagi-rerank`: `kohagi` is a pipe and stays one.
//! This is for the caller that wants one model loaded per host rather than
//! one per process, which is what a pipe cannot give a Rails cluster or a
//! sidecar. The model flags are `kohagi`'s; what is new is where to listen,
//! the limits on a request, and the optional reranker.
//!
//! Exit codes at load are the CLI's: 1 = fatal (model load, bad flags, the
//! address in use), 3 = the requested CoreML backend cannot serve this
//! request. Once listening it runs until SIGTERM or SIGINT, prints one
//! summary line, and exits 0.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;

use kohagi::cli::{self, BackendArg, ModelArgs};
use kohagi::rerank::{self, Reranker};
use kohagi::serve::{self, Listen, Load};
use kohagi::{CoreMlQuantize, ModelSource};

/// Serve local sentence embeddings over HTTP, OpenAI-compatible.
///
/// Loads the model, listens on --listen, and answers POST /v1/embeddings (plus
/// GET /v1/models and GET /health) until SIGTERM or SIGINT; with
/// --rerank-model-id, POST /v1/rerank as well. Models are downloaded from the
/// Hugging Face Hub on first use and cached (~/.cache/huggingface).
#[derive(Parser)]
#[command(name = "kohagi-serve", version)]
struct Args {
    /// Where to listen: `host:port`, or `unix:///path` for a Unix socket (mode
    /// 0600, replaced if a previous run left it). Loopback by default; this
    /// server has no authentication, so keep it off the open network as you
    /// would a database.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: Listen,
    #[command(flatten)]
    model: ModelArgs,
    /// Prefix prepended to every input text before embedding, as `kohagi
    /// --prefix` does. One server has one prefix: run it with "検索クエリ: "
    /// for queries, or with none and have callers prepend their own. Not
    /// applied to /v1/rerank, whose pairs the reranker takes raw.
    #[arg(long, default_value = "")]
    prefix: String,
    #[command(flatten)]
    rerank: RerankArgs,
    /// The most `input` items (or `documents`) one request may carry; more is
    /// refused with 400. OpenAI's own limit, and it bounds one reply's size.
    #[arg(long, default_value_t = 2048)]
    max_inputs: usize,
    /// The longest request body read; longer is refused with 413.
    #[arg(long, default_value_t = 32 * 1024 * 1024, value_name = "BYTES")]
    max_body_bytes: usize,
    /// Requests allowed to wait for a model at once. A model answers one
    /// request at a time (one forward pass uses every core); a request that
    /// finds this many already waiting is refused with 503 and Retry-After
    /// rather than queued behind them.
    #[arg(long, default_value_t = 64, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    max_queue: usize,
    /// After SIGTERM or SIGINT, how long to let open connections finish the
    /// reply in flight before closing them.
    #[arg(long, default_value_t = 30, value_name = "SECONDS")]
    shutdown_timeout: u64,
}

/// The reranker, loaded beside the embedder when one of these names it. Off
/// by default: a cross-encoder is a second model (310M parameters for
/// ruri-v3-reranker-310m), and only a caller of /v1/rerank wants it. It runs
/// on the same --device at the same --precision as the embedder.
#[derive(Clone, clap::Args)]
struct RerankArgs {
    /// Hugging Face reranker repo to load beside the embedder, which turns
    /// /v1/rerank on: cl-nagoya/ruri-v3-reranker-310m, or any ModernBERT
    /// sequence-classification checkpoint with one label. Ignored with
    /// --rerank-model-path.
    #[arg(long, value_name = "REPO")]
    rerank_model_id: Option<String>,
    /// Local safetensors weights of the reranker (offline mode; config.json
    /// must sit next to it), which also turns /v1/rerank on. Requires
    /// --rerank-tokenizer-path.
    #[arg(long, requires = "rerank_tokenizer_path")]
    rerank_model_path: Option<PathBuf>,
    /// The reranker's local tokenizer.json (offline mode).
    #[arg(long, requires = "rerank_model_path")]
    rerank_tokenizer_path: Option<PathBuf>,
    /// Token-level truncation length for a query/document pair. The longer
    /// half is trimmed first, as in the reference implementation.
    #[arg(long, default_value_t = 512)]
    rerank_max_seq_length: usize,
    /// Directory of the reranker's converted CoreML bundle for `--device
    /// coreml`, as `coreml-convert` writes it. Omit it to convert
    /// --rerank-model-id on first use.
    #[arg(long)]
    rerank_coreml_dir: Option<PathBuf>,
    /// Hugging Face repo holding that same layout, downloaded and cached on
    /// first use. --rerank-coreml-dir wins if both are set.
    #[arg(long)]
    rerank_coreml_model_id: Option<String>,
}

impl RerankArgs {
    /// Whether a reranker was asked for at all.
    fn wanted(&self) -> bool {
        self.rerank_model_id.is_some() || self.rerank_model_path.is_some()
    }

    fn options(&self, model: &ModelArgs) -> rerank::Options {
        rerank::Options {
            max_seq_length: self.rerank_max_seq_length,
            batch_size: model.batch_size,
            precision: model.precision.into(),
            backend: model.device.into(),
            // The sigmoid, as `kohagi-rerank` reports by default and as
            // CrossEncoder does, so published thresholds carry over.
            sigmoid: true,
            coreml_form: model.coreml_prefer.into(),
        }
    }

    /// Where the reranker comes from, plus the name to show for it.
    fn source(&self, model: &ModelArgs) -> anyhow::Result<(ModelSource, String)> {
        let checkpoint = cli::checkpoint_source(
            self.rerank_model_path.as_ref(),
            self.rerank_tokenizer_path.as_ref(),
            self.rerank_model_id.as_deref().unwrap_or_default(),
        );
        // CoreML loads converted fixed-shape models rather than safetensors.
        // Never quantized, and with `kohagi-rerank`'s bucket lengths: a pair
        // fills more of a bucket than a text does.
        if model.device == BackendArg::Coreml {
            return cli::coreml_source(
                self.rerank_coreml_dir.as_ref(),
                self.rerank_coreml_model_id.as_deref(),
                &[128, 256, 512],
                CoreMlQuantize::None,
                checkpoint,
            );
        }
        Ok(checkpoint)
    }

    fn load(&self, model: &ModelArgs, source: &ModelSource) -> anyhow::Result<Reranker> {
        Reranker::load(source, self.options(model))
    }
}

fn run(args: Args) -> anyhow::Result<()> {
    let (source, label) = args.model.source()?;
    let reranker = if args.rerank.wanted() {
        let (rerank_source, rerank_label) = args.rerank.source(&args.model)?;
        let (rerank, model) = (args.rerank.clone(), args.model.clone());
        Some(Load {
            label: rerank_label,
            load: move || rerank.load(&model, &rerank_source),
        })
    } else {
        None
    };
    let config = serve::Config {
        listen: args.listen.clone(),
        prefix: args.prefix.clone(),
        max_inputs: args.max_inputs,
        max_body_bytes: args.max_body_bytes,
        max_queue: args.max_queue,
        shutdown_timeout: Duration::from_secs(args.shutdown_timeout),
    };
    let model = args.model;

    // Each runs on its model's own thread; the model never leaves it. The
    // digest check happens there too, before anything can be answered.
    serve::run(
        config,
        Load {
            label,
            load: move || model.load(&source),
        },
        reranker,
    )
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
