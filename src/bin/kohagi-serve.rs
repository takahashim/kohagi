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

use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;

use kohagi::cli::{self, ModelArgs};
use kohagi::serve::{self, Listen};

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
    #[command(flatten)]
    model: ModelArgs,
    /// Prefix prepended to every input text before embedding, as `kohagi
    /// --prefix` does. One server has one prefix: run it with "検索クエリ: "
    /// for queries, or with none and have callers prepend their own.
    #[arg(long, default_value = "")]
    prefix: String,
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
    #[arg(long, default_value_t = 64, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    max_queue: usize,
    /// After SIGTERM or SIGINT, how long to let open connections finish the
    /// reply in flight before closing them.
    #[arg(long, default_value_t = 30, value_name = "SECONDS")]
    shutdown_timeout: u64,
}

fn run(args: Args) -> anyhow::Result<()> {
    let (source, label) = args.model.source()?;
    let config = serve::Config {
        listen: args.listen.clone(),
        label,
        prefix: args.prefix.clone(),
        max_inputs: args.max_inputs,
        max_body_bytes: args.max_body_bytes,
        max_queue: args.max_queue,
        shutdown_timeout: Duration::from_secs(args.shutdown_timeout),
    };
    let model = args.model;

    // Runs on the model's own thread; the model never leaves it. The digest
    // check happens there too, before anything can be answered.
    serve::run(config, move || model.load(&source))
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
