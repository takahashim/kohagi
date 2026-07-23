//! Cross-cutting error types.

/// A request the CoreML backend cannot serve — built without the `coreml`
/// feature, the wrong model source, or a `--max-seq-length` beyond the largest
/// converted bucket. Carried as its own type so the CLI can map it to a
/// dedicated exit code (3) instead of the generic fatal (1), letting callers
/// distinguish "retry on --device cpu" from a real failure. There is no
/// automatic fallback: backend choice is the caller's job.
#[derive(Debug)]
pub struct UnsupportedRequest(pub String);

impl UnsupportedRequest {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl std::fmt::Display for UnsupportedRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for UnsupportedRequest {}
