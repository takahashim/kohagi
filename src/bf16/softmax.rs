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

/// Whether the fused kernel can run on this CPU.
fn avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Softmax `scores` in place, adding `mask` first.
///
/// `scores` is `[rows, heads, seq, seq]`; `mask` is `[rows, seq, seq]`, the
/// additive bias every head shares. Rows of `scores` are the last axis, so
/// each `(row, head, query)` triple owns one contiguous `seq`-float span.
pub fn masked_softmax(scores: &mut [f32], mask: &[f32], rows: usize, heads: usize, seq: usize) {
    debug_assert_eq!(scores.len(), rows * heads * seq * seq);
    masked_softmax_block(scores, mask, rows, heads, Block::full(seq));
}

/// Which slice of the score matrix a call covers.
///
/// A sliding-window layer only needs a band, and [`super::modernbert`] walks
/// it one query block at a time: `queries` rows starting at `q0`, against
/// `keys` columns starting at `k0`. The scores for that block are their own
/// contiguous `[rows, heads, queries, keys]` buffer, but the mask is still the
/// full `[rows, seq, seq]` one, so it is indexed with `seq` as the row stride.
#[derive(Clone, Copy)]
pub struct Block {
    pub seq: usize,
    pub q0: usize,
    pub queries: usize,
    pub k0: usize,
    pub keys: usize,
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
}

/// [`masked_softmax`] over one block of the score matrix.
pub fn masked_softmax_block(scores: &mut [f32], mask: &[f32], rows: usize, heads: usize, b: Block) {
    debug_assert_eq!(scores.len(), rows * heads * b.queries * b.keys);
    debug_assert_eq!(mask.len(), rows * b.seq * b.seq);
    debug_assert!(b.q0 + b.queries <= b.seq && b.k0 + b.keys <= b.seq);

    let simd = avx512();
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
fn row(s: &mut [f32], m: &[f32], simd: bool) {
    debug_assert_eq!(s.len(), m.len());
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: guarded by the caller's feature check; the kernel reads
        // exactly `s.len()` floats from each pointer.
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

/// `exp` for 16 lanes: `exp(x) = 2^n · exp(r)`, with `n = round(x·log2 e)` and
/// `r = x - n·ln2` split hi/lo (Cody-Waite) so the reduction stays exact. The
/// polynomial is Cephes' `expf`, accurate to about 1 ulp.
///
/// Callers pass `x ≤ 0`, so no overflow path is needed and `scalef` handles
/// the underflow tail. [`super::geglu`] reuses it for `exp(-z²)`, which is
/// also never positive.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn exp512(x: __m512) -> __m512 {
    // 355/512, written as the quotient because it has to be *exactly*
    // representable: the point of the split is that `n · LN2_HI` is exact, so
    // only the tiny `LN2_LO` term carries rounding.
    const LN2_HI: f32 = 355.0 / 512.0;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const P: [f32; 6] = [
        1.987_569_1e-4,
        1.398_199_9e-3,
        8.333_452e-3,
        4.166_579_6e-2,
        1.666_666_5e-1,
        0.5,
    ];

    let n = _mm512_roundscale_ps::<0>(_mm512_mul_ps(x, _mm512_set1_ps(std::f32::consts::LOG2_E)));
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_HI), x);
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_LO), r);

    let mut y = _mm512_set1_ps(P[0]);
    for p in &P[1..] {
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(*p));
    }
    // y·r² + r + 1
    y = _mm512_fmadd_ps(
        y,
        _mm512_mul_ps(r, r),
        _mm512_add_ps(r, _mm512_set1_ps(1.0)),
    );
    _mm512_scalef_ps(y, n)
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
        row(&mut got, &m, avx512());
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

        let mut simd = s.clone();
        row(&mut simd, &m, true);
        let mut scalar = s.clone();
        row(&mut scalar, &m, false);

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
        for simd in [true, false] {
            let mut s = vec![0.5f32; seq];
            let m = vec![f32::NEG_INFINITY; seq];
            row(&mut s, &m, simd);
            assert!(
                s.iter().all(|v| v.is_finite()),
                "simd={simd}: row contains NaN or inf"
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
        masked_softmax(&mut s, &m, rows, heads, seq);

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
        masked_softmax(&mut got, &mask, rows, heads, seq);

        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-7, "worst absolute difference {worst:e}");
    }
}

#[cfg(test)]
mod exp_accuracy {
    use super::*;

    /// Sweep the whole argument range the softmax can hand `exp512` and
    /// compare against libm, which is what candle's scalar path uses.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn exp512_matches_libm() {
        if !avx512() {
            return;
        }
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f32;
        let mut x = EXP_FLOOR;
        while x <= 0.0 {
            let want = x.exp() as f64;
            let mut lanes = [0.0f32; 16];
            // SAFETY: guarded by the avx512 check above.
            let got = unsafe {
                let v = exp512(_mm512_set1_ps(x));
                _mm512_storeu_ps(lanes.as_mut_ptr(), v);
                lanes[0] as f64
            };
            if want > 0.0 {
                let rel = ((got - want) / want).abs();
                if rel > worst {
                    worst = rel;
                    worst_at = x;
                }
            }
            x += 0.0009765625; // 1/1024, so the sweep hits exact binary points
        }
        assert!(
            worst < 1e-6,
            "worst relative error {worst:e} at x = {worst_at}"
        );
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
        masked_softmax(&mut full, &mask, rows, heads, seq);

        // One block of 16 queries, against exactly the keys they can reach.
        let (q0, queries) = (16usize, 16usize);
        let k0 = q0 - w;
        let keys = (q0 + queries - 1 + w + 1) - k0;
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
        masked_softmax_block(
            &mut block,
            &mask,
            rows,
            heads,
            Block {
                seq,
                q0,
                queries,
                k0,
                keys,
            },
        );

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
