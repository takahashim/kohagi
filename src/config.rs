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
