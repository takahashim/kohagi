//! The thread that owns the model.
//!
//! One thread loads the model and runs every forward pass; handlers send it a
//! batch of texts over a bounded channel and wait for the vectors. This is the
//! one shape that works on every backend: a CoreML encoder cannot cross
//! threads (`src/coreml.rs`), and on the CPU a single forward pass already
//! fans out over every physical core, so answering requests one at a time
//! costs no throughput. If cross-request batching is ever wanted, it is this
//! loop draining more than one job at a time, and nothing else changes.

use std::panic::{catch_unwind, AssertUnwindSafe};

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, oneshot};

use super::Engine;
use crate::program::remark;
use crate::{ModelInfo, TokenInfo};

/// One request's worth of vectors, in input order.
pub(crate) struct Batch {
    pub vectors: Vec<Vec<f32>>,
    pub tokens: Vec<TokenInfo>,
}

struct Job {
    texts: Vec<String>,
    reply: oneshot::Sender<Result<Batch>>,
}

/// The handlers' end: a bounded queue into the model thread.
#[derive(Clone)]
pub(crate) struct Handle {
    queue: mpsc::Sender<Job>,
}

/// Why a batch was not answered.
pub(crate) enum EmbedError {
    /// The queue is full; the caller may retry later.
    Busy,
    /// The model thread is gone, and the server is on its way down.
    Gone,
    /// The forward pass failed.
    Failed(anyhow::Error),
}

impl Handle {
    /// Queue `texts` and wait for their vectors. Refuses at once, rather than
    /// waiting, when the queue is full: a caller is better served by a 503 it
    /// can act on than by a request that sits behind the queue's depth.
    pub(crate) async fn embed(&self, texts: Vec<String>) -> Result<Batch, EmbedError> {
        let (reply, answer) = oneshot::channel();
        self.queue
            .try_send(Job { texts, reply })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => EmbedError::Busy,
                mpsc::error::TrySendError::Closed(_) => EmbedError::Gone,
            })?;
        match answer.await {
            Ok(Ok(batch)) => Ok(batch),
            Ok(Err(e)) => Err(EmbedError::Failed(e)),
            Err(_) => Err(EmbedError::Gone),
        }
    }
}

/// Completes when the model thread has exited. The thread holds the sending
/// end for its whole life and only lets go by ending, so a completion is the
/// thread's death (a panic outside a forward pass, which `run_job` catches).
pub(crate) type Alive = oneshot::Receiver<()>;

/// Start the model thread and wait for it to load. Returns the handle the
/// handlers use, the loaded model's facts, and the liveness signal. A load
/// error comes back here, so the caller can map it to the CLI's exit codes.
/// `queue` is how many jobs may wait, at least 1; the flag enforces that.
pub(crate) fn spawn<E: Engine>(
    load: impl FnOnce() -> Result<E> + Send + 'static,
    queue: usize,
) -> Result<(Handle, ModelInfo, Alive)> {
    let (queue_tx, mut queue_rx) = mpsc::channel::<Job>(queue);
    let (loaded_tx, loaded_rx) = std::sync::mpsc::channel::<Result<ModelInfo>>();
    let (alive_tx, alive_rx) = oneshot::channel::<()>();

    std::thread::Builder::new()
        .name("kohagi-model".to_string())
        .spawn(move || {
            let _alive = alive_tx;
            let engine = match load() {
                Ok(engine) => {
                    let _ = loaded_tx.send(Ok(engine.info()));
                    engine
                }
                Err(e) => {
                    let _ = loaded_tx.send(Err(e));
                    return;
                }
            };
            while let Some(job) = queue_rx.blocking_recv() {
                let result = run_job(&engine, &job.texts);
                // Logged here, where it happened, once; the reply carries it
                // to the client as well.
                if let Err(e) = &result {
                    remark!("error: {e:#}");
                }
                // A caller that gave up meanwhile (client disconnected) has
                // dropped its end; that is not an error here.
                let _ = job.reply.send(result);
            }
        })
        .map_err(|e| anyhow!("spawning the model thread: {e}"))?;

    let info = loaded_rx
        .recv()
        .unwrap_or_else(|_| Err(anyhow!("the model thread died while loading")))?;
    Ok((Handle { queue: queue_tx }, info, alive_rx))
}

/// One forward pass, with a panic turned into this request's error rather
/// than the server's end. A shape assertion deep in a backend should cost the
/// request that tripped it, not every request after it.
fn run_job<E: Engine>(engine: &E, texts: &[String]) -> Result<Batch> {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    match catch_unwind(AssertUnwindSafe(|| engine.embed_with_tokens(&refs))) {
        Ok(Ok((vectors, tokens))) => Ok(Batch { vectors, tokens }),
        Ok(Err(e)) => Err(e),
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(anyhow!("the forward pass panicked: {what}"))
        }
    }
}
