//! `kohagi-serve`: the same [`Embedder`] behind an OpenAI-compatible
//! `POST /v1/embeddings`, for callers that keep one model loaded per host
//! rather than one per process.
//!
//! The stdio protocol (PROTOCOL.md) stays the contract for batches: it streams
//! a corpus with flat memory, which a request cannot. This is the second face
//! for everything else, and what differs is only how a request arrives. The
//! model, its flags and the exit codes at load are the CLI's; the HTTP
//! envelope is written down in PROTOCOL-http.md.
//!
//! One thread owns the model (`worker`); a current-thread tokio runtime
//! accepts connections (`server`) and answers requests (`http`), which are
//! checked and written in OpenAI's shapes (`openai`). A handler hands the
//! worker a batch of texts and waits for the vectors; nothing else touches
//! the model.

mod http;
mod listen;
mod openai;
mod server;
mod worker;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::program::remark;
use crate::{Embedder, ModelInfo, TokenInfo};

pub use listen::Listen;

/// What the server needs beyond the model: where to listen, what to call the
/// model, and the limits that keep one request from taking the host down.
pub struct Config {
    pub listen: Listen,
    /// The model's name in replies and in the summary. The model does not know
    /// it (one model has many names), so the caller supplies it.
    pub label: String,
    /// Prepended to every input text, as the CLI's `--prefix` is.
    pub prefix: String,
    /// The most `input` items one request may carry.
    pub max_inputs: usize,
    /// The largest request body read; anything longer is refused with 413.
    pub max_body_bytes: usize,
    /// Requests allowed to wait for the model, at least 1; beyond that a
    /// request is answered 503 at once rather than queued.
    pub max_queue: usize,
    /// How long to let open connections finish after a stop signal.
    pub shutdown_timeout: Duration,
}

/// What the server asks of a model. [`Embedder`] is the one that ships; tests
/// supply a stub, so the HTTP layer is checked without loading any weights.
pub trait Engine: 'static {
    fn info(&self) -> ModelInfo;
    fn embed_with_tokens(&self, texts: &[&str]) -> Result<(Vec<Vec<f32>>, Vec<TokenInfo>)>;
}

impl Engine for Embedder {
    fn info(&self) -> ModelInfo {
        Embedder::info(self)
    }

    fn embed_with_tokens(&self, texts: &[&str]) -> Result<(Vec<Vec<f32>>, Vec<TokenInfo>)> {
        Embedder::embed_with_tokens(self, texts)
    }
}

/// Load the model, listen, and answer until told to stop.
///
/// `load` runs on the model's own thread, so the model it returns never has to
/// cross one; that is what lets a CoreML encoder, which cannot, serve here.
/// Loading happens before listening: a supervisor then sees a bad checkpoint
/// as a failed start, and `/health` answering means ready. Returns when a
/// SIGTERM or SIGINT has been handled (`Ok`), or when the model thread died
/// (`Err`), so the process exits 1 and the supervisor restarts it.
pub fn run<E: Engine>(
    config: Config,
    load: impl FnOnce() -> Result<E> + Send + 'static,
) -> Result<()> {
    let (handle, info, alive) = worker::spawn(load, config.max_queue)?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?;

    runtime.block_on(async move {
        let state = Arc::new(http::State::new(&config, info, handle));
        let mut server = server::Server::start(&config.listen, state.clone()).await?;
        remark!(
            "listening on {} model={} {}",
            server.describe(),
            state.label(),
            state.info().summary_facts()
        );

        let outcome = server.serve_until(stop_reason(alive)).await;
        server.stop(config.shutdown_timeout).await;
        state.summarize();
        outcome
    })
}

/// Why the server stops: a signal asked it to (`Ok`), or the model thread
/// died and nothing can answer (`Err`).
async fn stop_reason(alive: worker::Alive) -> Result<()> {
    tokio::select! {
        biased;
        _ = shutdown_signal() => Ok(()),
        _ = alive => Err(anyhow::anyhow!("the model thread exited; nothing can answer")),
    }
}

/// Completes on SIGTERM or SIGINT (Ctrl-C), which is how a supervisor and a
/// terminal each ask for a stop. On Windows, on Ctrl-C, Ctrl-Break, or the
/// console closing.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("registering a SIGTERM handler");
        let mut int = signal(SignalKind::interrupt()).expect("registering a SIGINT handler");
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
    }
    #[cfg(windows)]
    {
        use tokio::signal::windows::{ctrl_break, ctrl_c, ctrl_close, ctrl_shutdown};
        let mut c = ctrl_c().expect("registering a Ctrl-C handler");
        let mut b = ctrl_break().expect("registering a Ctrl-Break handler");
        let mut close = ctrl_close().expect("registering a close handler");
        let mut down = ctrl_shutdown().expect("registering a shutdown handler");
        tokio::select! {
            _ = c.recv() => {}
            _ = b.recv() => {}
            _ = close.recv() => {}
            _ = down.recv() => {}
        }
    }
}
