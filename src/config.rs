//! Shared configuration value types used across modules.

/// A cross-encoder's classification head, written beside a converted CoreML
/// bundle when the checkpoint has one.
///
/// The emitted graph is the encoder alone, so a reranker's head has nowhere to
/// go inside it; this file is how a bundle stays self-contained. Named here
/// because the two ends are in different feature-gated modules — the converter
/// (`coreml_export`) writes it, `rerank` reads it — and the name has to be one
/// string.
#[cfg_attr(
    not(any(feature = "coreml", feature = "coreml-export")),
    allow(dead_code)
)]
pub const COREML_HEAD_FILE: &str = "head.safetensors";

/// The tensors a cross-encoder's classification head is made of, as
/// `ModernBertForSequenceClassification` names them.
///
/// One description because three modules act on the same list: `rerank` loads
/// these tensors, the CoreML converter copies them into [`COREML_HEAD_FILE`],
/// and the emitter uses [`head::is_head`] to know which of a checkpoint's unread
/// tensors it was right to leave out. Spelled out separately they drift, and the
/// failure is a bundle that loads and scores wrongly rather than one that fails.
pub mod head {
    /// The projection. Its `weight` is what says a checkpoint has a head at all.
    #[cfg_attr(not(feature = "coreml"), allow(dead_code))]
    pub const DENSE: &str = "head.dense";
    /// The LayerNorm between the projection and the classifier.
    #[cfg_attr(not(feature = "coreml"), allow(dead_code))]
    pub const NORM: &str = "head.norm";
    /// The linear down to one logit.
    #[cfg_attr(not(feature = "coreml"), allow(dead_code))]
    pub const CLASSIFIER: &str = "classifier";

    /// What a head cannot be assembled without.
    #[cfg_attr(not(feature = "coreml-export"), allow(dead_code))]
    pub const REQUIRED: [&str; 4] = [
        "head.dense.weight",
        "head.norm.weight",
        "classifier.weight",
        "classifier.bias",
    ];

    /// The two biases `config.json` decides on, absent from a checkpoint that
    /// does not use them — which is ModernBERT's default.
    #[cfg_attr(not(feature = "coreml-export"), allow(dead_code))]
    pub const OPTIONAL: [&str; 2] = ["head.dense.bias", "head.norm.bias"];

    /// Whether a checkpoint tensor belongs to the head rather than the encoder.
    ///
    /// By prefix family rather than by the names above: an unrecognized `head.*`
    /// is still the head's, and reporting it as an encoder tensor the converter
    /// dropped would be wrong.
    #[cfg_attr(not(feature = "coreml-export"), allow(dead_code))]
    pub fn is_head(name: &str) -> bool {
        name.starts_with("head.") || name.starts_with("classifier.")
    }
}

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
