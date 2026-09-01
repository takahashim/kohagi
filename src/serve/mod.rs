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
//! One thread owns the model (see `worker`), and a current-thread tokio
//! runtime reads requests and writes replies. A handler hands the worker a
//! batch of texts and waits for the vectors; nothing else touches the model.

mod http;
mod listen;
mod openai;
mod worker;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::watch;
use tokio::task::JoinSet;

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
    /// Whether the embedder normalizes its output, which decides whether a
    /// request's `dimensions` can be honoured (truncation re-normalizes).
    pub normalize: bool,
    /// The most `input` items one request may carry.
    pub max_inputs: usize,
    /// The largest request body read; anything longer is refused with 413.
    pub max_body_bytes: usize,
    /// Requests allowed to wait for the model; beyond that a request is
    /// answered 503 at once rather than queued.
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
    let (handle, info, mut alive) = worker::spawn(load, config.max_queue)?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?;

    runtime.block_on(async move {
        let bound = listen::Bound::bind(&config.listen).await?;
        remark!(
            "listening on {} model={} {}",
            bound.describe(),
            config.label,
            info.summary_facts()
        );

        let state = Arc::new(http::State::new(&config, info.clone(), handle));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut connections = JoinSet::new();
        let mut signal = std::pin::pin!(listen::shutdown_signal());

        let outcome = loop {
            tokio::select! {
                biased;
                _ = &mut signal => break Ok(()),
                _ = &mut alive => {
                    break Err(anyhow::anyhow!("the model thread exited; nothing can answer"));
                }
                accepted = bound.accept() => match accepted {
                    Ok(io) => {
                        connections.spawn(http::serve_connection(io, state.clone(), shutdown_rx.clone()));
                    }
                    Err(e) => {
                        // Out of descriptors, or a connection that went away
                        // between arriving and being accepted. Neither is a
                        // reason to stop; a pause keeps a persistent one from
                        // spinning.
                        remark!("accept failed: {e}");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                },
            }
        };

        // Stop accepting, let open connections finish what they are answering,
        // and abandon whatever is still open after the timeout.
        let _ = shutdown_tx.send(true);
        let drained = tokio::time::timeout(config.shutdown_timeout, async {
            while connections.join_next().await.is_some() {}
        })
        .await;
        if drained.is_err() {
            remark!(
                "{} connections still open after {:?}; closing them",
                connections.len(),
                config.shutdown_timeout
            );
            connections.abort_all();
        }

        state.summarize(&config.label, &info);
        drop(bound);
        outcome
    })
}
