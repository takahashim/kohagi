//! Information about a loaded model.

/// Information written by `--print-model-info` and the run summary.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelInfo {
    /// `--device`, as its flag value.
    pub backend: &'static str,
    /// `--precision`, as its flag value.
    pub precision: &'static str,
    /// SHA-256 of the loaded weights.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Metadata for a converted CoreML bundle.
    #[serde(flatten)]
    pub bundle: Option<Bundle>,
    /// The pooling setting resolved at load time.
    pub pooling: &'static str,
    /// The model's own dimension, its `hidden_size`.
    pub dim: usize,
    pub max_seq_length: usize,
    /// What `sentence_bert_config.json` says the model can take, when it says
    /// anything. Reported rather than obeyed: `--max-seq-length` decides the
    /// run, and the two differ on checkpoints whose declared ceiling is higher
    /// than Kohagi's default (`ruri-v3-130m` declares 8192). This is where the
    /// difference is visible, since sentence-transformers reads the same field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declared_max_seq_length: Option<usize>,
    /// The value returned for each record.
    #[serde(flatten)]
    pub output: Output,
}

/// Metadata that applies only to a converted CoreML bundle.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Bundle {
    /// The checkpoint recorded by the converter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// SHA-256 of the source checkpoint, if recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_sha256: Option<String>,
    /// Supported sequence lengths.
    pub buckets: Vec<usize>,
    /// Bundle quantization: `embeddings-int8`, `all-int8`, or `none`.
    pub quantization: String,
    /// Emitted graph version, when recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_version: Option<String>,
}

/// The value returned for each record.
///
/// It is flattened and untagged to preserve the existing JSON format.
///
/// Serialize only. Do not derive `Deserialize`: the optional field in
/// [`Output::Embedding`] would make every object match that variant.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(untagged)]
pub enum Output {
    /// One vector per text.
    Embedding {
        /// Output dimension after `--dims`, when it changes the vector size.
        #[serde(skip_serializing_if = "Option::is_none")]
        output_dim: Option<usize>,
        /// Whether each vector is unit length, so that dot product is cosine:
        /// `true` unless `--no-normalize`. Reported because it changes every
        /// vector as surely as `pooling` does, and a reader of a results file
        /// cannot tell from the numbers.
        normalized: bool,
    },
    /// One score per pair, represented as `sigmoid` or `logit`.
    Score { score: &'static str },
}

impl ModelInfo {
    /// Adds metadata from a converted CoreML bundle.
    #[cfg(feature = "coreml")]
    pub(crate) fn add_bundle(&mut self, encoder: &crate::coreml::CoreMlEncoder) {
        let p = encoder.provenance();
        self.bundle = Some(Bundle {
            source: p.source,
            source_sha256: p.source_sha256,
            // The loaded models define the supported lengths.
            buckets: encoder.buckets(),
            // A missing key means fp16.
            quantization: p.quantization.unwrap_or_else(|| "none".to_string()),
            graph_version: p.graph_version,
        });
    }

    /// Returns the digest of the weights used and its field name.
    pub fn digest(&self) -> Option<(&'static str, &str)> {
        if let Some(sha) = &self.sha256 {
            return Some(("sha256", sha));
        }
        Some((
            "source_sha256",
            self.bundle.as_ref()?.source_sha256.as_deref()?,
        ))
    }

    /// Returns the actual embedding dimension.
    pub fn reported_dim(&self) -> usize {
        match self.output {
            Output::Embedding {
                output_dim: Some(n),
                ..
            } => n,
            _ => self.dim,
        }
    }

    /// Whether each vector is unit length. `false` for a reranker, whose
    /// scores are not vectors.
    pub fn normalized(&self) -> bool {
        matches!(
            self.output,
            Output::Embedding {
                normalized: true,
                ..
            }
        )
    }

    /// The facts a summary line carries about the model: which weights, and
    /// the settings that silently change every vector when they differ.
    /// `sha256=` for a checkpoint's own digest, `source_sha256=` for a CoreML
    /// bundle's report of the checkpoint behind it: not the same claim, and a
    /// line that conflated them would be worse than one with neither.
    pub fn summary_facts(&self) -> String {
        let mut out = String::new();
        if let Some((claim, sha)) = self.digest() {
            out.push_str(&format!("{claim}={} ", crate::fingerprint::short(sha)));
        }
        out.push_str(&format!(
            "pooling={} dim={} max_seq={}",
            self.pooling,
            self.reported_dim(),
            self.max_seq_length
        ));
        if let Output::Score { score } = self.output {
            out.push_str(&format!(" score={score}"));
        }
        out
    }
}
