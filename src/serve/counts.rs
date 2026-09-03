//! What the run summary is written from.
//!
//! The numbers a server reports at shutdown, in the shape the stdio protocol
//! reports its own (`src/protocol.rs`): the same operator reads both, and a
//! run through HTTP should not have to be counted differently from a run
//! through a pipe.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use hyper::StatusCode;

use super::{Batch, Scores};
use crate::program::remark;

/// The summary's numbers. `requests` counts everything that arrived;
/// `rejected` the 4xx among them (the client's mistake) and `failed` the 5xx
/// (this side's). `inputs` and `truncated` are the stdio summary's `in` and
/// `truncated`, over the requests that were answered: texts embedded, and
/// texts or pairs that ran past a model's length. `scored` is the documents
/// reranked. A request is answered whole or not at all, so there is no
/// separate `out`.
#[derive(Default)]
pub(crate) struct Counts {
    // Read by `http`'s tests, which check what a refusal counted as.
    pub(super) requests: AtomicUsize,
    pub(super) rejected: AtomicUsize,
    pub(super) failed: AtomicUsize,
    pub(super) inputs: AtomicUsize,
    pub(super) truncated: AtomicUsize,
    pub(super) scored: AtomicUsize,
}

impl Counts {
    pub(crate) fn saw(&self, status: StatusCode) {
        self.requests.fetch_add(1, Relaxed);
        if status.is_client_error() {
            self.rejected.fetch_add(1, Relaxed);
        } else if status.is_server_error() {
            self.failed.fetch_add(1, Relaxed);
        }
    }

    pub(crate) fn embedded(&self, batch: &Batch) {
        self.inputs.fetch_add(batch.vectors.len(), Relaxed);
        self.truncated
            .fetch_add(batch.tokens.iter().filter(|t| t.truncated).count(), Relaxed);
    }

    pub(crate) fn reranked(&self, scores: &Scores) {
        self.scored.fetch_add(scores.scores.len(), Relaxed);
        self.truncated.fetch_add(
            scores.tokens.iter().filter(|t| t.truncated).count(),
            Relaxed,
        );
    }

    /// The run's one summary line, on stderr at shutdown, reading like the
    /// CLI's: which weights answered, and how much they answered. `facts` is
    /// [`crate::ModelInfo::summary_facts`] for the model that answered.
    pub(crate) fn summarize(&self, label: &str, facts: &str) {
        remark!(
            "model={label} {facts} requests={} in={} truncated={} scored={} rejected={} failed={}",
            self.requests.load(Relaxed),
            self.inputs.load(Relaxed),
            self.truncated.load(Relaxed),
            self.scored.load(Relaxed),
            self.rejected.load(Relaxed),
            self.failed.load(Relaxed)
        );
    }
}
