//! Tokenization, length bucketing, and pooling.
//!
//! Batching here is *output-invariant*: rows are sorted by token length and
//! padded only to the longest row of their own batch, padding is masked out of
//! the mean pool, and every split point is invisible in the result. That
//! freedom is what lets `model.rs` re-split batches to fit its memory budget.

use anyhow::Result;
use tokenizers::{Encoding, Tokenizer, TruncationParams};

/// How to reduce the encoder's `[seq, dim]` output to one vector per text.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pooling {
    /// Mask-aware mean over token embeddings — what sentence-transformers
    /// does for Ruri v3 and modernbert-embed. The right default.
    Mean,
    /// First token only. Some encoders are trained for this; Ruri is not.
    Cls,
}

impl Pooling {
    /// The name this pooling has in a `1_Pooling/config.json` and on the
    /// command line, so that what a run reports is what a caller can pass back.
    pub fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Cls => "cls",
        }
    }
}

/// Per-text tokenization facts, surfaced so a caller can tell a truncated
/// embedding — one built from only the first `--max-seq-length` tokens — from a
/// whole one. Both fields come straight off the encoding, so producing them
/// costs nothing beyond the tokenization that already happens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenInfo {
    /// Tokens actually embedded, the model's special tokens included — that is,
    /// `min(true token length, max_seq_length)`.
    pub n_tokens: usize,
    /// The text ran past `--max-seq-length`, so its tail was dropped before
    /// embedding; the vector reflects only the kept prefix.
    pub truncated: bool,
}

/// The truncation facts for one encoding. Truncation stores the dropped tail in
/// the encoding's `overflowing` chunks, so a non-empty list is the signal — no
/// second tokenization pass and no reliance on the length happening to hit the
/// cap (a text exactly `max_seq_length` long is not truncated).
pub fn token_info(enc: &tokenizers::Encoding) -> TokenInfo {
    TokenInfo {
        n_tokens: enc.len(),
        truncated: !enc.get_overflowing().is_empty(),
    }
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

/// One tokenized text, before it is grouped into a padded batch.
struct Tokenized<'a> {
    ids: &'a [u32],
    mask: &'a [u32],
}

/// Tokenize texts, no padding.
pub fn encode(tokenizer: &Tokenizer, texts: &[&str]) -> Result<Vec<Encoding>> {
    tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))
}

/// The same, for `(query, text)` pairs — what a cross-encoder scores.
///
/// The pair is joined by the tokenizer's own template rather than by string
/// concatenation here (`<s> query </s> <s> text </s>` for the Ruri and
/// japanese-reranker families), so the two sequences arrive in the shape the
/// model was trained on, with truncation trimming the longer of the two first
/// (`longest_first`, the tokenizer's default and the one CrossEncoder asks
/// for).
pub fn encode_pairs(tokenizer: &Tokenizer, pairs: &[(&str, &str)]) -> Result<Vec<Encoding>> {
    let inputs: Vec<tokenizers::EncodeInput> = pairs
        .iter()
        .map(|&(query, text)| (query, text).into())
        .collect();
    tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))
}

/// Sort encodings by token length and split into padded batches of at most
/// `batch_size` rows, each padded only to its own longest row. Also returns
/// per-row [`TokenInfo`] in the original input order, so truncation can be
/// reported alongside the results.
///
/// A pair is one row here, as a text is: what a cross-encoder truncated is the
/// longer half of one input, not one of two inputs.
pub fn bucket_encodings(
    encodings: &[Encoding],
    batch_size: usize,
) -> (Vec<BatchInput>, Vec<TokenInfo>) {
    let info: Vec<TokenInfo> = encodings.iter().map(token_info).collect();
    let rows: Vec<Tokenized> = encodings
        .iter()
        .map(|e| Tokenized {
            ids: e.get_ids(),
            mask: e.get_attention_mask(),
        })
        .collect();
    (bucket(&rows, batch_size), info)
}

/// Group tokenized rows into padded batches. Split out from tokenization so
/// the index arithmetic can be tested on its own.
fn bucket(rows: &[Tokenized], batch_size: usize) -> Vec<BatchInput> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| rows[i].ids.len());

    let mut batches = Vec::new();
    for chunk in order.chunks(batch_size.max(1)) {
        let batch = chunk.len();
        let seq = chunk.iter().map(|&i| rows[i].ids.len()).max().unwrap_or(0);
        // Zero-filled, so anything past a row's own length stays padding.
        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        for (bi, &i) in chunk.iter().enumerate() {
            for (t, (&id, &m)) in rows[i].ids.iter().zip(rows[i].mask).enumerate() {
                ids[bi * seq + t] = id as i64;
                mask[bi * seq + t] = m as i64;
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
    batches
}

/// Reduce one row's hidden states to a single vector.
///
/// `hidden` is that row's `[seq, dim]` block of the encoder output, laid out
/// token by token, and `mask` is its `[seq]` attention mask (0 = padding).
/// The sequence length is `mask.len()`, so there is no separate argument to
/// get wrong.
pub fn pool_row(hidden: &[f32], mask: &[i64], dim: usize, pooling: Pooling) -> Vec<f32> {
    debug_assert_eq!(hidden.len(), mask.len() * dim);
    match pooling {
        Pooling::Cls => hidden[..dim].to_vec(),
        Pooling::Mean => {
            let mut acc = vec![0.0f32; dim];
            let mut count = 0.0f32;
            for (token, &keep) in mask.iter().enumerate() {
                if keep != 0 {
                    let start = token * dim;
                    for d in 0..dim {
                        acc[d] += hidden[start + d];
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

/// Matryoshka truncation: keep the leading `n` dimensions and re-normalize,
/// so the shorter vector is unit-length in its own space and dot = cosine
/// holds on it. Truncating first is the order `SentenceTransformer(...,
/// truncate_dim=N)` uses. This is the one definition, for a run's `--dims`
/// and a request's `dimensions` alike, so the two cannot drift.
pub fn truncate_renormalize(row: &mut Vec<f32>, n: usize) {
    row.truncate(n);
    l2_normalize(row);
}

/// What a truncation to `n` means on vectors of `dim` dimensions: `Some(n)`
/// when it shortens them, `None` when `n == dim` and it changes no vector,
/// and an error outside `1..=dim`. Callers phrase the error for their own
/// flag or field; this only decides.
pub fn truncation(n: usize, dim: usize) -> Result<Option<usize>, OutOfRange> {
    if !(1..=dim).contains(&n) {
        return Err(OutOfRange { dim });
    }
    Ok((n < dim).then_some(n))
}

/// A truncation outside what the vectors can be cut to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfRange {
    pub dim: usize,
}

impl std::fmt::Display for OutOfRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "must be in 1..={}", self.dim)
    }
}

impl std::error::Error for OutOfRange {}

#[cfg(test)]
mod truncation_tests {
    use super::*;

    /// The truncated vector is unit-length in its own space, not a slice of a
    /// unit vector: that is the difference between dot = cosine holding on the
    /// shorter vectors and silently not.
    #[test]
    fn truncation_renormalizes_the_kept_prefix() {
        let mut v = vec![3.0, 4.0, 12.0];
        truncate_renormalize(&mut v, 2);
        assert_eq!(v, [0.6, 0.8]);
    }

    /// Equal to the dimension asks for nothing, since it changes no vector;
    /// outside `1..=dim` is refused with the dimension named.
    #[test]
    fn a_truncation_is_a_shortening_or_nothing() {
        assert_eq!(truncation(256, 512), Ok(Some(256)));
        assert_eq!(truncation(1, 512), Ok(Some(1)));
        assert_eq!(truncation(512, 512), Ok(None));
        for bad in [0, 513] {
            let e = truncation(bad, 512).unwrap_err();
            assert_eq!(e.to_string(), "must be in 1..=512");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[seq, dim]` hidden states where token `t` is `[t, t, ...]`, so a mean
    /// over kept tokens is easy to predict by hand.
    fn hidden(seq: usize, dim: usize) -> Vec<f32> {
        (0..seq)
            .flat_map(|t| std::iter::repeat_n(t as f32, dim))
            .collect()
    }

    #[test]
    fn mean_pool_ignores_padding() {
        let (seq, dim) = (4, 3);
        // Tokens 0 and 1 are real, 2 and 3 are padding -> mean of [0, 1].
        let pooled = pool_row(&hidden(seq, dim), &[1, 1, 0, 0], dim, Pooling::Mean);
        assert_eq!(pooled, vec![0.5; dim]);

        // Without the mask it would be the mean of [0, 1, 2, 3] instead.
        let all = pool_row(&hidden(seq, dim), &[1, 1, 1, 1], dim, Pooling::Mean);
        assert_eq!(all, vec![1.5; dim]);
    }

    #[test]
    fn mean_pool_of_all_padding_is_zero() {
        let dim = 3;
        assert_eq!(
            pool_row(&hidden(4, dim), &[0; 4], dim, Pooling::Mean),
            vec![0.0; dim]
        );
    }

    #[test]
    fn cls_pool_takes_the_first_token() {
        let dim = 3;
        assert_eq!(
            pool_row(&hidden(4, dim), &[1; 4], dim, Pooling::Cls),
            vec![0.0; dim]
        );
    }

    /// Rows of the given token lengths, with ids counting up from 1 so each
    /// row is recognisable and 0 always means padding.
    fn rows(lengths: &[usize]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let ids: Vec<Vec<u32>> = lengths
            .iter()
            .enumerate()
            .map(|(i, &n)| vec![i as u32 + 1; n])
            .collect();
        let masks = lengths.iter().map(|&n| vec![1u32; n]).collect();
        (ids, masks)
    }

    fn bucket_lengths(lengths: &[usize], batch_size: usize) -> Vec<BatchInput> {
        let (ids, masks) = rows(lengths);
        let rows: Vec<Tokenized> = ids
            .iter()
            .zip(&masks)
            .map(|(i, m)| Tokenized { ids: i, mask: m })
            .collect();
        bucket(&rows, batch_size)
    }

    #[test]
    fn bucketing_pads_each_batch_to_its_own_longest_row() {
        // Sorted by length, the rows group as [1, 2] and [5, 9].
        let batches = bucket_lengths(&[9, 1, 5, 2], 2);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].seq, 2);
        assert_eq!(batches[1].seq, 9);
        // Short rows sit with short rows, so little padding is wasted.
        assert_eq!(batches[0].orig, vec![1, 3]);
        assert_eq!(batches[1].orig, vec![2, 0]);
    }

    #[test]
    fn bucketing_keeps_every_row_exactly_once() {
        let batches = bucket_lengths(&[4, 7, 2, 9, 1], 2);
        let mut seen: Vec<usize> = batches.iter().flat_map(|b| b.orig.clone()).collect();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn padded_positions_are_masked_out() {
        // One batch, rows of length 1 and 3 -> the short row gets 2 pad slots.
        let b = &bucket_lengths(&[3, 1], 2)[0];
        assert_eq!((b.batch, b.seq), (2, 3));
        // Row 0 of the batch is the length-1 row (input index 1, id 2).
        assert_eq!(b.orig[0], 1);
        assert_eq!(&b.ids[0..3], &[2, 0, 0]);
        assert_eq!(&b.mask[0..3], &[1, 0, 0]);
        // Row 1 fills the whole width.
        assert_eq!(&b.ids[3..6], &[1, 1, 1]);
        assert_eq!(&b.mask[3..6], &[1, 1, 1]);
    }

    #[test]
    fn l2_normalize_gives_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.6, 0.8]);

        // A zero vector has no direction to preserve; leave it alone.
        let mut zero = vec![0.0, 0.0];
        l2_normalize(&mut zero);
        assert_eq!(zero, vec![0.0, 0.0]);
    }
}
