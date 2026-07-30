//! Shared configuration value types used across modules.

/// Which converted form to download when a CoreML Hub repo ships both a
/// compiled `.mlmodelc` and a portable `.mlpackage` for a bucket. Only affects
/// [`crate::ModelSource::CoreMlHub`] downloads — a local dir loads whatever is
/// there.
///
/// Lives here rather than in the (feature-gated) `coreml` module because it is
/// an [`crate::Options`] field and must compile unconditionally; keeping it out
/// of `model` also lets the `coreml` provisioning code use it without depending
/// back on `model`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CoreMlForm {
    /// The compiled `.mlmodelc` — no per-run compile. Falls back to the
    /// `.mlpackage` for buckets that only ship one.
    #[default]
    Compiled,
    /// The portable `.mlpackage` — compiled on load, but robust across OS
    /// versions. Falls back to the `.mlmodelc` for buckets that only ship one.
    Package,
}

/// How much of the model to quantize when `--device coreml` converts a checkpoint
/// itself. Only affects [`crate::ModelSource::CoreMlConvert`]; a pre-converted
/// bundle is whatever it was converted as.
///
/// The default is fp16 even though int8 embeddings measured free on both retrieval
/// benchmarks: a quantized bundle's vectors are not
/// interchangeable with an fp16 one's, so quantizing by default would silently
/// break an index built from a published fp16 bundle.
///
/// Here rather than in the feature-gated `coreml` module for the same reason as
/// [`CoreMlForm`]: it is part of a `ModelSource` and must compile unconditionally.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CoreMlQuantize {
    /// Everything fp16.
    #[default]
    None,
    /// The embedding table int8, one scale per row. Halves a large-vocabulary
    /// bundle at no measured retrieval cost.
    Embeddings,
    /// The embedding table and every projection int8.
    All,
}

impl CoreMlQuantize {
    /// The cache-key spelling, and what a bundle's provenance records.
    pub fn tag(self) -> &'static str {
        match self {
            Self::None => "fp16",
            Self::Embeddings => "emb8",
            Self::All => "all8",
        }
    }
}
