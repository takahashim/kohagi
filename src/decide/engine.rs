//! How a typed question becomes scored candidates: the head.
//!
//! The request and the answer are model-independent; a head is the choice of
//! how a checkpoint reads a question, selected with `--head` and independent of
//! the checkpoint selected with `--model-id`. Heads register themselves in
//! [`factories`], so adding or removing one (the packed Laya head, behind
//! `head-laya`) is a module plus a registry entry, not an arm in a central
//! enum.

use anyhow::Result;

use super::Request;
use crate::rerank::Options as RerankOptions;
use crate::{Backend, CoreMlForm, ModelInfo, ModelSource, Precision, TokenInfo};

/// A loaded head: how a checkpoint reads a request and returns canonical,
/// calibrated logits. The rest of decide — validation, softmax, the answer —
/// sees only this.
pub trait DecisionHead: Send + Sync {
    /// Per request, per question, canonical `(label, calibrated logit)` pairs.
    /// One [`TokenInfo`] per request, for the summary and `--report-tokens`.
    fn score(&self, requests: &[Request]) -> Result<(Vec<super::RequestScore>, Vec<TokenInfo>)>;

    /// What was loaded, for `--print-model-info` and the fingerprint check.
    fn info(&self) -> ModelInfo;
}

/// Builds one [`DecisionHead`] from a source and the loader knobs. One static
/// per head, listed in [`factories`].
pub trait HeadFactory: Sync {
    /// The `--head` value this factory answers to.
    fn name(&self) -> &'static str;

    /// Read the checkpoint the way this head reads it. `subfolder` is the
    /// nested directory inside the repo or local dir, when the checkpoint has
    /// one (Laya's multilingual weights do).
    fn load(
        &self,
        source: &ModelSource,
        options: EngineOptions,
        subfolder: Option<&str>,
    ) -> Result<Box<dyn DecisionHead>>;
}

/// Every head this build can load, in help order. The cross-encoder is always
/// present; the packed head is gated by `head-laya`.
pub fn factories() -> &'static [&'static dyn HeadFactory] {
    #[cfg(feature = "head-laya")]
    {
        static FACTORIES: &[&dyn HeadFactory] = &[super::cross::FACTORY, super::laya::FACTORY];
        FACTORIES
    }
    #[cfg(not(feature = "head-laya"))]
    {
        static FACTORIES: &[&dyn HeadFactory] = &[super::cross::FACTORY];
        FACTORIES
    }
}

/// The `--head` names this build accepts, for help and error messages.
pub fn names() -> Vec<&'static str> {
    factories().iter().map(|factory| factory.name()).collect()
}

/// Loader knobs shared by the heads, so a `--model-id` and `--head` pair loads
/// the same way whichever binary asked for it.
#[derive(Clone, Copy)]
pub struct EngineOptions {
    pub max_seq_length: usize,
    pub batch_size: usize,
    pub precision: Precision,
    pub backend: Backend,
    pub coreml_form: CoreMlForm,
}

/// A loaded decision head, chosen by name from [`factories`].
pub struct Engine {
    head: Box<dyn DecisionHead>,
}

impl Engine {
    pub fn load(
        source: &ModelSource,
        options: EngineOptions,
        head: &str,
        subfolder: Option<&str>,
    ) -> Result<Self> {
        let factory = factories()
            .iter()
            .find(|factory| factory.name() == head)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown head `{head}`; expected one of: {}",
                    names().join(", ")
                )
            })?;
        Ok(Self {
            head: factory.load(source, options, subfolder)?,
        })
    }

    pub fn info(&self) -> ModelInfo {
        self.head.info()
    }

    /// Per request, per question, canonical `(label, calibrated logit)` pairs.
    /// One [`TokenInfo`] per request, for the summary and `--report-tokens`.
    pub fn score(
        &self,
        requests: &[Request],
    ) -> Result<(Vec<super::RequestScore>, Vec<TokenInfo>)> {
        self.head.score(requests)
    }
}

/// The reranker options a cross-encoder head loads with: raw logits, because
/// the softmax is over a question's options, not per pair.
pub(crate) fn cross_options(options: EngineOptions) -> RerankOptions {
    RerankOptions {
        max_seq_length: options.max_seq_length,
        batch_size: options.batch_size,
        precision: options.precision,
        backend: options.backend,
        sigmoid: false,
        coreml_form: options.coreml_form,
    }
}
