//! Tokenization, length bucketing, and pooling.
//!
//! Batching here is *output-invariant*: rows are sorted by token length and
//! padded only to the longest row of their own batch, padding is masked out of
//! the mean pool, and every split point is invisible in the result. That
//! freedom is what lets `model.rs` re-split batches to fit its memory budget.

use anyhow::Result;
use tokenizers::{Tokenizer, TruncationParams};

/// How to reduce the encoder's `[seq, dim]` output to one vector per text.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pooling {
    /// Mask-aware mean over token embeddings — what sentence-transformers
    /// does for Ruri v3 and modernbert-embed. The right default.
    Mean,
    /// First token only. Some encoders are trained for this; Ruri is not.
    Cls,
}

/// One padded batch: `ids`/`mask` are row-major `[batch, seq]`, and `orig[i]`
/// is the caller's index for row `i` (rows are reordered by length).
pub struct BatchInput {
    pub ids: Vec<i64>,
    pub mask: Vec<i64>,
    pub batch: usize,
    pub seq: usize,
    pub orig: Vec<usize>,
}

/// Load a tokenizer.json and pin truncation to `max_seq_length`.
pub fn load_tokenizer(path: &std::path::Path, max_seq_length: usize) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("cannot load tokenizer {}: {e}", path.display()))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_seq_length,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("truncation config: {e}"))?;
    tokenizer.with_padding(None);
    Ok(tokenizer)
}

/// Tokenize all texts (no padding), sort by token length, and split into
/// padded batches of at most `batch_size` rows, each padded only to its own
/// longest row.
pub fn tokenize_bucket(
    tokenizer: &Tokenizer,
    texts: &[&str],
    batch_size: usize,
) -> Result<Vec<BatchInput>> {
    let n = texts.len();
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| encodings[i].get_ids().len());

    let mut batches = Vec::new();
    for chunk in order.chunks(batch_size.max(1)) {
        let batch = chunk.len();
        let seq = chunk
            .iter()
            .map(|&i| encodings[i].get_ids().len())
            .max()
            .unwrap_or(0);
        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        for (bi, &i) in chunk.iter().enumerate() {
            let enc = &encodings[i];
            let eids = enc.get_ids();
            let emask = enc.get_attention_mask();
            for t in 0..eids.len() {
                ids[bi * seq + t] = eids[t] as i64;
                mask[bi * seq + t] = emask[t] as i64;
            }
        }
        batches.push(BatchInput {
            ids,
            mask,
            batch,
            seq,
            orig: chunk.to_vec(),
        });
    }
    Ok(batches)
}

/// Pool one batch row (index `b`) of the flat `[batch, seq, dim]` hidden
/// states into a single vector.
pub fn pool_one(
    data: &[f32],
    mask: &[i64],
    b: usize,
    seq: usize,
    dim: usize,
    pooling: Pooling,
) -> Vec<f32> {
    match pooling {
        Pooling::Cls => {
            let start = (b * seq) * dim;
            data[start..start + dim].to_vec()
        }
        Pooling::Mean => {
            let mut acc = vec![0.0f32; dim];
            let mut count = 0.0f32;
            for t in 0..seq {
                if mask[b * seq + t] != 0 {
                    let start = (b * seq + t) * dim;
                    for d in 0..dim {
                        acc[d] += data[start + d];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for v in &mut acc {
                    *v /= count;
                }
            }
            acc
        }
    }
}

/// L2-normalize one vector in place (no-op on the zero vector).
pub fn l2_normalize(row: &mut [f32]) {
    let norm: f64 = row
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let norm = norm as f32;
        for v in row {
            *v /= norm;
        }
    }
}
