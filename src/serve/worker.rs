//! The thread that owns a model.
//!
//! One thread loads the model and answers every question put to it; handlers
//! send it one over a bounded channel and wait for the answer. This is the
//! one shape that works on every backend: a CoreML encoder cannot cross
//! threads (`src/coreml.rs`), and on the CPU a single forward pass already
//! fans out over every physical core, so answering requests one at a time
//! costs no throughput. If cross-request batching is ever wanted, it is this
//! loop draining more than one job at a time, and nothing else changes.
//!
//! Generic over the question and the answer, so the embedder (texts in,
//! vectors out) and the reranker (pairs in, scores out) share it.

use std::panic::{catch_unwind, AssertUnwindSafe};

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, oneshot};

use super::{Engine, Load};
use crate::program::remark;
use crate::ModelInfo;

struct Job<I, O> {
    input: I,
    reply: oneshot::Sender<Result<O>>,
}

/// The handlers' end: a bounded queue into the model's thread.
pub(crate) struct Handle<I, O> {
    queue: mpsc::Sender<Job<I, O>>,
}

impl<I, O> Clone for Handle<I, O> {
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
        }
    }
}

/// A model that is up: its name, its facts, and the way to ask it.
pub(crate) struct Loaded<I, O> {
    pub label: String,
    pub info: ModelInfo,
    pub handle: Handle<I, O>,
}

/// Why a question was not answered.
pub(crate) enum WorkerError {
    /// The queue is full; the caller may retry later.
    Busy,
    /// The model's thread is gone, and the server is on its way down.
    Gone,
    /// The forward pass failed.
    Failed(anyhow::Error),
}

impl<I: Send + 'static, O: Send + 'static> Handle<I, O> {
    /// Queue `input` and wait for its answer. Refuses at once, rather than
    /// waiting, when the queue is full: a caller is better served by a 503 it
    /// can act on than by a request that sits behind the queue's depth.
    pub(crate) async fn ask(&self, input: I) -> Result<O, WorkerError> {
        let (reply, answer) = oneshot::channel();
        self.queue
            .try_send(Job { input, reply })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => WorkerError::Busy,
                mpsc::error::TrySendError::Closed(_) => WorkerError::Gone,
            })?;
        match answer.await {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(e)) => Err(WorkerError::Failed(e)),
            Err(_) => Err(WorkerError::Gone),
        }
    }
}

/// Completes when the model's thread has exited. The thread holds the
/// sending end for its whole life and only lets go by ending, so a completion
/// is the thread's death (a panic outside a forward pass, which `run_job`
/// catches).
pub(crate) type Alive = oneshot::Receiver<()>;

/// A model's thread, up: the model as the handlers see it, and the signal
/// that the thread has died.
pub(crate) struct Spawned<I, O> {
    pub loaded: Loaded<I, O>,
    pub alive: Alive,
}

/// Start a model's thread and wait for it to load. A load error comes back
/// here, so the caller can map it to the CLI's exit codes. `queue` is how
/// many jobs may wait, at least 1; the flag enforces that.
pub(crate) fn spawn<E: Engine>(
    thread: &'static str,
    model: Load<impl FnOnce() -> Result<E> + Send + 'static>,
    queue: usize,
) -> Result<Spawned<E::Input, E::Output>> {
    let (queue_tx, mut queue_rx) = mpsc::channel::<Job<E::Input, E::Output>>(queue);
    let (loaded_tx, loaded_rx) = std::sync::mpsc::channel::<Result<ModelInfo>>();
    let (alive_tx, alive_rx) = oneshot::channel::<()>();

    let load = model.load;
    std::thread::Builder::new()
        .name(thread.to_string())
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
                let result = run_job(&engine, job.input);
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
        .map_err(|e| anyhow!("spawning the {thread} thread: {e}"))?;

    let info = loaded_rx
        .recv()
        .unwrap_or_else(|_| Err(anyhow!("the {thread} thread died while loading")))?;
    Ok(Spawned {
        loaded: Loaded {
            label: model.label,
            info,
            handle: Handle { queue: queue_tx },
        },
        alive: alive_rx,
    })
}

/// One forward pass, with a panic turned into this request's error rather
/// than the server's end. A shape assertion deep in a backend should cost the
/// request that tripped it, not every request after it.
fn run_job<E: Engine>(engine: &E, input: E::Input) -> Result<E::Output> {
    match catch_unwind(AssertUnwindSafe(|| engine.answer(input))) {
        Ok(result) => result,
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

#[cfg(test)]
mod tests {
    use super::super::{testing, Batch, Engine, Load};
    use super::*;
    use crate::TokenInfo;

    /// Panics when told to, so the promise under test is `run_job`'s: a panic
    /// costs the request that tripped it, not every request after it.
    struct Fragile;

    impl Engine for Fragile {
        type Input = Vec<String>;
        type Output = Batch;

        fn info(&self) -> ModelInfo {
            testing::embedding_info(1)
        }

        fn answer(&self, texts: Vec<String>) -> Result<Batch> {
            if texts.first().is_some_and(|t| t == "boom") {
                panic!("kaboom");
            }
            Ok(Batch {
                vectors: vec![vec![1.0]; texts.len()],
                tokens: vec![
                    TokenInfo {
                        n_tokens: 1,
                        truncated: false
                    };
                    texts.len()
                ],
            })
        }
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    #[test]
    fn a_panicking_forward_pass_costs_the_request_not_the_server() {
        let spawned = spawn(
            "test-fragile",
            Load {
                label: "f".to_string(),
                load: || Ok(Fragile),
            },
            4,
        )
        .unwrap();
        block_on(async {
            match spawned.loaded.handle.ask(vec!["boom".to_string()]).await {
                Err(WorkerError::Failed(e)) => {
                    assert!(e.to_string().contains("kaboom"), "{e}");
                }
                _ => panic!("the panic should come back as this request's error"),
            }
            // The thread survived it and answers the next request.
            let next = spawned.loaded.handle.ask(vec!["fine".to_string()]).await;
            assert!(matches!(next, Ok(batch) if batch.vectors.len() == 1));
        });
    }

    #[test]
    fn a_load_error_comes_back_to_the_spawn_caller() {
        let e = spawn(
            "test-noload",
            Load {
                label: "f".to_string(),
                load: || -> Result<Fragile> { anyhow::bail!("no such checkpoint") },
            },
            4,
        )
        .err()
        .expect("the load failed");
        assert!(e.to_string().contains("no such checkpoint"), "{e}");
    }

    #[test]
    fn a_load_panic_reads_as_a_death_while_loading() {
        let e = spawn(
            "test-panicload",
            Load {
                label: "f".to_string(),
                load: || -> Result<Fragile> { panic!("torn") },
            },
            4,
        )
        .err()
        .expect("the load died");
        assert!(e.to_string().contains("died while loading"), "{e}");
    }
}
