//! `kohagi-serve`: the same [`Embedder`] behind an OpenAI-compatible
//! `POST /v1/embeddings`, and optionally a [`Reranker`] behind
//! `POST /v1/rerank`, for callers that keep one model loaded per host rather
//! than one per process.
//!
//! The stdio protocol (PROTOCOL.md) stays the contract for batches: it streams
//! a corpus with flat memory, which a request cannot. This is the second face
//! for everything else, and what differs is only how a request arrives. The
//! models, their flags and the exit codes at load are the CLIs'; the HTTP
//! envelope is written down in PROTOCOL-http.md.
//!
//! Each model has a thread of its own (`worker`); a current-thread tokio
//! runtime accepts connections (`server`) and answers requests (`http`),
//! which are checked and written in the shapes clients expect (`openai`). A
//! handler hands a worker one question and waits for the answer; nothing
//! else touches a model.

mod http;
mod listen;
mod openai;
mod server;
mod worker;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::program::remark;
use crate::rerank::Reranker;
use crate::{Embedder, ModelInfo, TokenInfo};

pub use listen::Listen;

/// What the server needs beyond the models: where to listen, and the limits
/// that keep one request from taking the host down.
pub struct Config {
    pub listen: Listen,
    /// Prepended to every text sent for embedding, as the CLI's `--prefix`
    /// is. Not applied to reranking, whose pairs the model takes raw.
    pub prefix: String,
    /// The most `input` items (or `documents`) one request may carry.
    pub max_inputs: usize,
    /// The largest request body read; anything longer is refused with 413.
    pub max_body_bytes: usize,
    /// Requests allowed to wait for a model, at least 1; beyond that a
    /// request is answered 503 at once rather than queued.
    pub max_queue: usize,
    /// How long to let open connections finish after a stop signal.
    pub shutdown_timeout: Duration,
}

/// What the server asks of a model: one kind of question, one kind of
/// answer. [`Embedder`] and [`Reranker`] are the ones that ship; tests supply
/// stubs, so the HTTP layer is checked without loading any weights.
pub trait Engine: 'static {
    type Input: Send + 'static;
    type Output: Send + 'static;
    fn info(&self) -> ModelInfo;
    fn answer(&self, input: Self::Input) -> Result<Self::Output>;
}

/// One request's worth of vectors, in input order.
pub struct Batch {
    pub vectors: Vec<Vec<f32>>,
    pub tokens: Vec<TokenInfo>,
}

impl Engine for Embedder {
    type Input = Vec<String>;
    type Output = Batch;

    fn info(&self) -> ModelInfo {
        Embedder::info(self)
    }

    fn answer(&self, texts: Vec<String>) -> Result<Batch> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let (vectors, tokens) = self.embed_with_tokens(&refs)?;
        Ok(Batch { vectors, tokens })
    }
}

/// A query and the documents to order for it.
pub struct Pairs {
    pub query: String,
    pub documents: Vec<String>,
}

/// One score per document, in input order.
pub struct Scores {
    pub scores: Vec<f32>,
    pub tokens: Vec<TokenInfo>,
}

impl Engine for Reranker {
    type Input = Pairs;
    type Output = Scores;

    fn info(&self) -> ModelInfo {
        Reranker::info(self)
    }

    fn answer(&self, pairs: Pairs) -> Result<Scores> {
        let refs: Vec<(&str, &str)> = pairs
            .documents
            .iter()
            .map(|d| (pairs.query.as_str(), d.as_str()))
            .collect();
        let (scores, tokens) = self.score(&refs)?;
        Ok(Scores { scores, tokens })
    }
}

/// A model to load: what to call it in replies and the summary (the model
/// does not know its own name; one model has many), and how to load it. The
/// loading runs on the model's own thread, so what it returns never has to
/// cross one; that is what lets a CoreML encoder, which cannot, serve here.
pub struct Load<F> {
    pub label: String,
    pub load: F,
}

/// Load the models, listen, and answer until told to stop.
///
/// Loading happens before listening: a supervisor then sees a bad checkpoint
/// as a failed start, and `/health` answering means ready. Returns when a
/// SIGTERM or SIGINT has been handled (`Ok`), or when a model's thread died
/// (`Err`), so the process exits 1 and the supervisor restarts it.
pub fn run<E, R, F, G>(config: Config, embedder: Load<F>, reranker: Option<Load<G>>) -> Result<()>
where
    E: Engine<Input = Vec<String>, Output = Batch>,
    R: Engine<Input = Pairs, Output = Scores>,
    F: FnOnce() -> Result<E> + Send + 'static,
    G: FnOnce() -> Result<R> + Send + 'static,
{
    let embedder = worker::spawn("kohagi-model", embedder, config.max_queue)?;
    let reranker = match reranker {
        Some(reranker) => Some(worker::spawn(
            "kohagi-reranker",
            reranker,
            config.max_queue,
        )?),
        None => None,
    };
    let (embedder, embedder_alive) = (embedder.loaded, embedder.alive);
    let (reranker, reranker_alive) = match reranker {
        Some(spawned) => (Some(spawned.loaded), Some(spawned.alive)),
        None => (None, None),
    };

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?;

    runtime.block_on(async move {
        let state = Arc::new(http::State::new(&config, embedder, reranker));
        let mut server = server::Server::start(&config.listen, state.clone()).await?;
        remark!(
            "listening on {} model={} {}",
            server.describe(),
            state.embedder().label,
            state.embedder().info.summary_facts()
        );
        if let Some(reranker) = state.reranker() {
            remark!(
                "reranker={} {}",
                reranker.label,
                reranker.info.summary_facts()
            );
        }

        let outcome = server
            .serve_until(stop_reason(embedder_alive, reranker_alive))
            .await;
        server.stop(config.shutdown_timeout).await;
        state.summarize();
        outcome
    })
}

/// Why the server stops: a signal asked it to (`Ok`), or a model's thread
/// died and nothing can answer (`Err`).
async fn stop_reason(embedder: worker::Alive, reranker: Option<worker::Alive>) -> Result<()> {
    let reranker = async move {
        match reranker {
            Some(alive) => {
                let _ = alive.await;
            }
            None => std::future::pending().await,
        }
    };
    tokio::select! {
        biased;
        _ = shutdown_signal() => Ok(()),
        _ = embedder => Err(anyhow::anyhow!("the embedding model's thread exited; nothing can answer")),
        _ = reranker => Err(anyhow::anyhow!("the reranker's thread exited; nothing can answer /v1/rerank")),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The other reason to stop, beside a signal: a model's thread died, and
    /// the error names which, since that is what the operator reads.
    #[test]
    fn a_dead_model_thread_is_a_reason_to_stop_with_an_error() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let (_held, embedder) = tokio::sync::oneshot::channel::<()>();
            let (gone, reranker) = tokio::sync::oneshot::channel::<()>();
            drop(gone);
            let e = stop_reason(embedder, Some(reranker))
                .await
                .expect_err("the reranker died");
            assert!(e.to_string().contains("reranker"), "{e}");

            let (gone, embedder) = tokio::sync::oneshot::channel::<()>();
            drop(gone);
            let e = stop_reason(embedder, None)
                .await
                .expect_err("the embedder died");
            assert!(e.to_string().contains("embedding"), "{e}");
        });
    }
}
