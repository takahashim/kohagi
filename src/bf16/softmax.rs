//! Fused masked softmax over the attention scores.
//!
//! Replaces `scores.broadcast_add(mask)` followed by candle's
//! `softmax_last_dim`, which together measured 28% of a 512-token bf16
//! forward — more than either GEMM. Two things cost there. The broadcast
//! materializes a second `[rows, heads, seq, seq]` f32 tensor purely to widen
//! a mask that is shared by every head. And candle's softmax exponentiates
//! with scalar `expf`, one element at a time (`SoftmaxLastDim::cpu_fwd` in
//! candle-nn 0.11 `src/ops.rs`), which at 512 tokens is where most of the
//! time goes.
//!
//! This adds the mask, takes the row max, exponentiates and normalizes in
//! three passes over one `seq`-float row — 2 KB, so it stays in L1 — with an
//! AVX-512 `exp`. Like [`super::gemm`] it is deliberately single-threaded:
//! parallelism comes from many forwards in flight, and candle's softmax
//! nesting a rayon `par_chunks` inside kohagi's own pool is part of what this
//! avoids.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::simd::{exp512, Avx512};

/// Floor on the exponent, which disposes of the `NaN` a fully-masked query row
/// would otherwise produce: every entry is `-inf`, so `x - max` is
/// `-inf - -inf`. Clamping turns that row into a uniform distribution instead.
/// Pooling drops those rows anyway; this just keeps a `NaN` from ever entering
/// the following matmul.
///
/// `exp(-87) ≈ 1.7e-38` — the smallest exponent that still lands on a *normal*
/// f32. Going lower (the exponential is zero well before `-104`) would put a
/// fully-masked row in the denormal range, where the two implementations stop
/// agreeing: `scalef` flushes `2^-150` to zero, giving a zero normalizer and a
/// `NaN` row back, while scalar `expf` returns the smallest denormal and
/// survives. Against a maximum of `exp(0) = 1`, a term this size is far below
/// f32 resolution, so the clamp cannot perturb a row that has any real
/// contributor.
const EXP_FLOOR: f32 = -87.0;

/// Which slice of the score matrix a call covers.
///
/// A sliding-window layer only needs a band, and [`super::modernbert`] walks
/// it one query block at a time: `queries` rows starting at `q0`, against
/// `keys` columns starting at `k0`. The scores for that block are their own
/// contiguous `[rows, heads, queries, keys]` buffer, but the mask is still the
/// full `[rows, seq, seq]` one, so it is indexed with `seq` as the row stride.
///
/// The fields are read-only from outside; [`Block::full`] and [`Block::band`]
/// are the two shapes that exist, and the second is where the arithmetic that
/// makes banding correct lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub(super) seq: usize,
    pub(super) q0: usize,
    pub(super) queries: usize,
    pub(super) k0: usize,
    pub(super) keys: usize,
}

impl Block {
    /// The whole matrix, which is what a global-attention layer wants.
    pub fn full(seq: usize) -> Self {
        Self {
            seq,
            q0: 0,
            queries: seq,
            k0: 0,
            keys: seq,
        }
    }

    /// The keys a block of queries can reach through a sliding window of
    /// half-width `window`.
    ///
    /// Query `q0 + i` attends `[q0 + i - window, q0 + i + window]`, so the
    /// block as a whole needs the union of those, clamped to the sequence.
    /// Covering every open key is what lets the caller drop the rest of the
    /// row: whatever this leaves out is masked shut, and contributes `exp` of
    /// the mask floor rather than anything f32 can hold.
    pub fn band(seq: usize, q0: usize, queries: usize, window: usize) -> Self {
        debug_assert!(queries > 0 && q0 + queries <= seq);
        let k0 = q0.saturating_sub(window);
        let last = (q0 + queries - 1 + window).min(seq - 1);
        Self {
            seq,
            q0,
            queries,
            k0,
            keys: last + 1 - k0,
        }
    }

    /// How many queries this block covers, for the caller sizing its buffers.
    pub fn queries(&self) -> usize {
        self.queries
    }

    /// Where the block's keys start, and how many there are.
    pub fn keys(&self) -> (usize, usize) {
        (self.k0, self.keys)
    }

    /// Where the block's queries start.
    pub fn q0(&self) -> usize {
        self.q0
    }
}

/// Softmax `scores` in place, adding `mask` first.
///
/// `scores` is the block's own contiguous `[rows, heads, queries, keys]`
/// buffer; `mask` is the full `[rows, seq, seq]` additive bias every head
/// shares. Rows of both are the last axis, so each `(row, head, query)` triple
/// owns one contiguous span in each.
pub fn masked_softmax(scores: &mut [f32], mask: &[f32], rows: usize, heads: usize, b: Block) {
    debug_assert_eq!(scores.len(), rows * heads * b.queries * b.keys);
    debug_assert_eq!(mask.len(), rows * b.seq * b.seq);
    debug_assert!(b.q0 + b.queries <= b.seq && b.k0 + b.keys <= b.seq);

    let simd = Avx512::detect();
    for r in 0..rows {
        for h in 0..heads {
            for i in 0..b.queries {
                let s = ((r * heads + h) * b.queries + i) * b.keys;
                // Keys are the last axis of the mask too, so a block's slice of
                // any one query row is contiguous — only the row stride differs.
                let m = (r * b.seq + b.q0 + i) * b.seq + b.k0;
                row(&mut scores[s..s + b.keys], &mask[m..m + b.keys], simd);
            }
        }
    }
}

/// One query row. `simd` is threaded in rather than re-detected per row.
fn row(s: &mut [f32], m: &[f32], simd: Option<Avx512>) {
    debug_assert_eq!(s.len(), m.len());
    #[cfg(target_arch = "x86_64")]
    if simd.is_some() {
        // SAFETY: the `Avx512` token exists only where detection found the
        // instructions; the kernel reads exactly `s.len()` floats per pointer.
        unsafe { row_avx512(s.as_mut_ptr(), m.as_ptr(), s.len()) };
        return;
    }
    let _ = simd;
    row_scalar(s, m);
}

/// Reference implementation, and what runs without AVX-512.
fn row_scalar(s: &mut [f32], m: &[f32]) {
    let mut max = f32::NEG_INFINITY;
    for (s, m) in s.iter_mut().zip(m) {
        *s += *m;
        if *s > max {
            max = *s;
        }
    }
    let mut sum = 0.0f32;
    for s in s.iter_mut() {
        // `f32::max` returns the non-NaN operand, so an all-masked row lands
        // on the floor rather than propagating NaN — matching the SIMD path.
        *s = (*s - max).max(EXP_FLOOR).exp();
        sum += *s;
    }
    for s in s.iter_mut() {
        *s /= sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn row_avx512(s: *mut f32, m: *const f32, n: usize) {
    let tail = n % 16;
    let body = n - tail;
    let k: __mmask16 = if tail == 0 { 0 } else { (1u16 << tail) - 1 };

    // Add the mask in place and track the row maximum.
    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i < body {
        let t = _mm512_add_ps(_mm512_loadu_ps(s.add(i)), _mm512_loadu_ps(m.add(i)));
        _mm512_storeu_ps(s.add(i), t);
        vmax = _mm512_max_ps(vmax, t);
        i += 16;
    }
    if tail != 0 {
        let t = _mm512_add_ps(
            _mm512_maskz_loadu_ps(k, s.add(body)),
            _mm512_maskz_loadu_ps(k, m.add(body)),
        );
        _mm512_mask_storeu_ps(s.add(body), k, t);
        // Masked so the zeroed lanes past the row cannot raise the maximum.
        vmax = _mm512_mask_max_ps(vmax, k, vmax, t);
    }
    let max = _mm512_set1_ps(_mm512_reduce_max_ps(vmax));

    // Exponentiate the differences, accumulating the normalizer.
    let floor = _mm512_set1_ps(EXP_FLOOR);
    let mut vsum = _mm512_setzero_ps();
    i = 0;
    while i < body {
        // max(d, floor) with d first: MAXPS yields the second operand when the
        // first is NaN, which is how an all-masked row avoids propagating one.
        let d = _mm512_max_ps(_mm512_sub_ps(_mm512_loadu_ps(s.add(i)), max), floor);
        let e = exp512(d);
        _mm512_storeu_ps(s.add(i), e);
        vsum = _mm512_add_ps(vsum, e);
        i += 16;
    }
    if tail != 0 {
        let d = _mm512_max_ps(
            _mm512_sub_ps(_mm512_maskz_loadu_ps(k, s.add(body)), max),
            floor,
        );
        let e = exp512(d);
        _mm512_mask_storeu_ps(s.add(body), k, e);
        vsum = _mm512_mask_add_ps(vsum, k, vsum, e);
    }
    let sum = _mm512_set1_ps(_mm512_reduce_add_ps(vsum));

    i = 0;
    while i < body {
        _mm512_storeu_ps(s.add(i), _mm512_div_ps(_mm512_loadu_ps(s.add(i)), sum));
        i += 16;
    }
    if tail != 0 {
        let q = _mm512_div_ps(_mm512_maskz_loadu_ps(k, s.add(body)), sum);
        _mm512_mask_storeu_ps(s.add(body), k, q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition, in f64, to compare both implementations against.
    fn reference(s: &[f32], m: &[f32]) -> Vec<f64> {
        let t: Vec<f64> = s
            .iter()
            .zip(m)
            .map(|(s, m)| *s as f64 + *m as f64)
            .collect();
        let max = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = t.iter().map(|t| (t - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        e.iter().map(|e| e / sum).collect()
    }

    fn worst_error(seq: usize) -> f64 {
        // Spread of magnitudes wide enough that the maximum is not always the
        // first element and the tail underflows.
        let s: Vec<f32> = (0..seq).map(|i| ((i % 37) as f32 - 18.0) * 0.7).collect();
        let m: Vec<f32> = (0..seq)
            .map(|i| if i % 5 == 0 { f32::NEG_INFINITY } else { 0.0 })
            .collect();
        let want = reference(&s, &m);

        let mut got = s.clone();
        row(&mut got, &m, Avx512::detect());
        got.iter()
            .zip(&want)
            .map(|(g, w)| (*g as f64 - w).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn matches_the_definition() {
        // 512 is the shape that matters; 7 and 33 exercise the masked tail.
        for seq in [7, 33, 512] {
            let e = worst_error(seq);
            assert!(e < 1e-7, "seq {seq}: worst absolute error {e:e}");
        }
    }

    #[test]
    fn simd_and_scalar_agree() {
        let seq = 512;
        let s: Vec<f32> = (0..seq).map(|i| ((i % 37) as f32 - 18.0) * 0.7).collect();
        let m = vec![0.0f32; seq];

        let Some(avx512) = Avx512::detect() else {
            return;
        };
        let mut simd = s.clone();
        row(&mut simd, &m, Some(avx512));
        let mut scalar = s.clone();
        row(&mut scalar, &m, None);

        let worst = simd
            .iter()
            .zip(&scalar)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-7, "worst absolute difference {worst:e}");
    }

    /// A row that is entirely masked out must not produce NaN: pooling skips
    /// those rows, but a NaN would spread through the following matmul first.
    #[test]
    fn fully_masked_row_stays_finite() {
        let seq = 512;
        // Whichever paths this CPU has: the token is `None` without AVX-512,
        // so the vector arm is simply not exercised there.
        for simd in [Avx512::detect(), None] {
            let mut s = vec![0.5f32; seq];
            let m = vec![f32::NEG_INFINITY; seq];
            row(&mut s, &m, simd);
            assert!(
                s.iter().all(|v| v.is_finite()),
                "simd={}: row contains NaN or inf",
                simd.is_some()
            );
        }
    }

    /// Every `(row, head, query)` span must get its own softmax, and the mask
    /// must be shared across heads rather than indexed as if it had a head
    /// axis.
    #[test]
    fn indexes_rows_heads_and_queries() {
        let (rows, heads, seq) = (2, 3, 4);
        let mut s: Vec<f32> = (0..rows * heads * seq * seq)
            .map(|i| (i % 11) as f32 * 0.3)
            .collect();
        // Mask query 0 of row 1 down to its first key.
        let mut m = vec![0.0f32; rows * seq * seq];
        for j in 1..seq {
            m[seq * seq + j] = f32::NEG_INFINITY;
        }
        masked_softmax(&mut s, &m, rows, heads, Block::full(seq));

        for r in 0..rows {
            for h in 0..heads {
                for i in 0..seq {
                    let o = ((r * heads + h) * seq + i) * seq;
                    let sum: f32 = s[o..o + seq].iter().sum();
                    assert!((sum - 1.0).abs() < 1e-5, "row {r} head {h} query {i}");
                }
            }
        }
        // That masked query is one-hot in every head of row 1, and untouched
        // in row 0.
        for h in 0..heads {
            let o = (heads + h) * seq * seq;
            assert!((s[o] - 1.0).abs() < 1e-6);
            let o = h * seq * seq;
            assert!(s[o] < 1.0);
        }
    }
}

#[cfg(test)]
mod candle_parity {
    use super::*;
    use candle_core::{Device, Tensor};

    /// The kernel must agree with what it replaced: `broadcast_add` of a
    /// `[rows, 1, seq, seq]` mask followed by candle's `softmax_last_dim`.
    #[test]
    fn matches_candle() {
        let (rows, heads, seq) = (2usize, 8usize, 37usize);
        let dev = Device::Cpu;

        let scores: Vec<f32> = (0..rows * heads * seq * seq)
            .map(|i| ((i * 2654435761usize % 1000) as f32 / 100.0) - 5.0)
            .collect();
        // Row 0 keeps every key; row 1 pads the last 9, as a real batch would.
        let mut mask = vec![0.0f32; rows * seq * seq];
        for i in 0..seq {
            for j in (seq - 9)..seq {
                mask[(seq + i) * seq + j] = f32::MIN;
            }
        }

        let want = {
            let s = Tensor::from_vec(scores.clone(), (rows, heads, seq, seq), &dev).unwrap();
            let m = Tensor::from_vec(mask.clone(), (rows, 1, seq, seq), &dev).unwrap();
            candle_nn::ops::softmax_last_dim(&s.broadcast_add(&m).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };

        let mut got = scores;
        masked_softmax(&mut got, &mask, rows, heads, Block::full(seq));

        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-7, "worst absolute difference {worst:e}");
    }
}
#[cfg(test)]
mod block_tests {
    use super::*;

    /// A block of the score matrix must come out exactly as the same entries
    /// do when the whole matrix is softmaxed — provided the block covers every
    /// key the mask leaves open for those queries, which is the precondition
    /// the sliding-window path relies on.
    #[test]
    fn block_matches_the_full_matrix() {
        let (rows, heads, seq) = (2usize, 3usize, 64usize);
        let w = 8usize; // half-window
        let scores: Vec<f32> = (0..rows * heads * seq * seq)
            .map(|i| ((i * 2_654_435_761usize % 997) as f32 / 100.0) - 5.0)
            .collect();
        // Sliding-window mask: everything past the window is closed.
        let mut mask = vec![0.0f32; rows * seq * seq];
        for r in 0..rows {
            for i in 0..seq {
                for j in 0..seq {
                    if j.abs_diff(i) > w {
                        mask[(r * seq + i) * seq + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        let mut full = scores.clone();
        masked_softmax(&mut full, &mask, rows, heads, Block::full(seq));

        // One block of 16 queries, against exactly the keys they can reach.
        let (q0, queries) = (16usize, 16usize);
        let b = Block::band(seq, q0, queries, w);
        let (k0, keys) = b.keys();
        let mut block = vec![0.0f32; rows * heads * queries * keys];
        for r in 0..rows {
            for h in 0..heads {
                for i in 0..queries {
                    let src = ((r * heads + h) * seq + q0 + i) * seq + k0;
                    let dst = ((r * heads + h) * queries + i) * keys;
                    block[dst..dst + keys].copy_from_slice(&scores[src..src + keys]);
                }
            }
        }
        masked_softmax(&mut block, &mask, rows, heads, b);

        for r in 0..rows {
            for h in 0..heads {
                for i in 0..queries {
                    let src = ((r * heads + h) * seq + q0 + i) * seq + k0;
                    let dst = ((r * heads + h) * queries + i) * keys;
                    for j in 0..keys {
                        let (a, b) = (block[dst + j], full[src + j]);
                        assert_eq!(a, b, "row {r} head {h} query {i} key {j}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod band_tests {
    use super::*;

    /// The band must contain every key the window leaves open for every query
    /// in the block, and nothing is required beyond that. This is the whole
    /// correctness argument for computing a band instead of a full row, so it
    /// is checked exhaustively over the shapes rather than spot-checked.
    #[test]
    fn covers_every_key_the_window_opens() {
        for seq in [1usize, 5, 32, 64, 129, 512] {
            for window in [0usize, 1, 8, 64, 600] {
                for size in [1usize, 3, 32] {
                    for q0 in (0..seq).step_by(size) {
                        let queries = size.min(seq - q0);
                        let b = Block::band(seq, q0, queries, window);
                        let (k0, keys) = b.keys();

                        assert!(
                            k0 + keys <= seq,
                            "seq {seq} w {window} q0 {q0}: past the end"
                        );
                        for i in 0..queries {
                            let q = q0 + i;
                            let lo = q.saturating_sub(window);
                            let hi = (q + window).min(seq - 1);
                            assert!(
                                k0 <= lo && hi < k0 + keys,
                                "seq {seq} w {window} q0 {q0}: query {q} wants {lo}..={hi}, \
                                 block has {k0}..{}",
                                k0 + keys
                            );
                        }
                    }
                }
            }
        }
    }

    /// A window wide enough to span the sequence degenerates to the full row,
    /// which is what makes `banding_pays` in `super::modernbert` a pure
    /// optimization rather than a correctness switch.
    #[test]
    fn a_window_spanning_the_sequence_is_the_full_row() {
        let seq = 32;
        assert_eq!(Block::band(seq, 0, seq, seq), Block::full(seq));
    }

    /// The first and last blocks are the ones that clamp, and an off-by-one in
    /// either direction would silently drop a key.
    #[test]
    fn clamps_at_both_ends() {
        let (seq, w) = (100usize, 10usize);
        // At the start there is nothing to the left to reach for.
        assert_eq!(Block::band(seq, 0, 8, w).keys(), (0, 8 + w));
        // In the middle both sides are open.
        assert_eq!(Block::band(seq, 50, 8, w).keys(), (40, 8 + 2 * w));
        // At the end the right side runs out.
        assert_eq!(Block::band(seq, 92, 8, w).keys(), (82, 18));
    }
}
