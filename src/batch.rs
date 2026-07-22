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

/// One tokenized text, before it is grouped into a padded batch.
struct Tokenized<'a> {
    ids: &'a [u32],
    mask: &'a [u32],
}

/// Tokenize all texts (no padding), sort by token length, and split into
/// padded batches of at most `batch_size` rows, each padded only to its own
/// longest row.
pub fn tokenize_bucket(
    tokenizer: &Tokenizer,
    texts: &[&str],
    batch_size: usize,
) -> Result<Vec<BatchInput>> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
    let rows: Vec<Tokenized> = encodings
        .iter()
        .map(|e| Tokenized {
            ids: e.get_ids(),
            mask: e.get_attention_mask(),
        })
        .collect();
    Ok(bucket(&rows, batch_size))
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
