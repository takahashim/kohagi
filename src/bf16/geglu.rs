//! Fused GeGLU, for every CPU engine rather than only the bf16 one.
//!
//! `Wi` emits `[tokens, 2·inter]` — the gate half followed by the up half —
//! and ModernBERT reduces it to `gelu(gate) · up`. Done with candle ops that
//! is `chunk`, `gelu_erf`, and a multiply, and the `gelu_erf` alone measured
//! 10% of a 512-token bf16 forward: candle evaluates `erf` one element at a
//! time (`crate::cpu::erf` behind `UnaryOpT for Erf`), and the MLP's
//! intermediate is four times the hidden width, so it is the widest
//! elementwise op in the model.
//!
//! This reads both halves and writes the gated result in one pass, with a
//! vectorized `erf`. Working on the `Vec<f32>` the GEMM already produces also
//! drops the two intermediate tensors the candle version allocated per layer.
//!
//! ## Who takes it
//!
//! It began beside the bf16 path and the module name still says so, but nothing
//! here changes precision: it is f32 in and f32 out. Three callers reach it now,
//! through [`crate::fused`] and `crate::burn_engine::flex`, and the reason is
//! the same each time and the size is not:
//!
//! - **x86 with AVX-512.** Worth about 2.6% end to end on the Burn engine.
//!   burn-flex's own gelu is already SIMD there, so this only fuses the pass.
//! - **aarch64.** Worth 3.45x on the operation, because neither burn-flex nor
//!   candle vectorises `erf` on ARM: measured at ruri-v3-130m's shape, 19 calls
//!   over `[460, 2048]`, 0.0176 s here against 0.0608 composed and 0.0600
//!   through candle's `gelu_erf`. It is a floor under both engines rather than
//!   a gap between them.
//! - **x86 without AVX-512.** Nothing: [`row`] falls back to [`gelu_scalar`],
//!   which loses to a SIMD library op. [`vectorised`] is what callers ask.
//!
//! ## Accuracy
//!
//! `erf` is Abramowitz & Stegun 7.1.26, whose absolute error is bounded by
//! 1.5e-7. That is weaker than libm's, which mattered less when only
//! `--precision bf16` reached it; it now decides the default f32 vectors too, so
//! it is measured rather than argued: over 16 texts of 460 tokens the embeddings
//! move by `1 - cosine` 1.1e-13, an order below the 3e-12 Kohagi already reports
//! against PyTorch. candle's `gelu_erf` uses the same A&S coefficients, so the
//! two differ only in which `exp` evaluates the decay.
//!
//! `tests::matches_candle` pins the end result against `gelu_erf` over the range
//! a well-behaved layer produces, and `tests::matches_candle_on_wide_inputs`
//! over one wider than that.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use super::simd::exp512;
#[cfg(target_arch = "aarch64")]
use super::simd::exp_neon;
use super::simd::Avx512;

/// `gelu(x) = 0.5·x·(1 + erf(x/√2))`, so this scales the argument.
const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Abramowitz & Stegun 7.1.26, for `z ≥ 0`:
/// `erf(z) = 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)·exp(-z²)`, `t = 1/(1 + pz)`.
const P: f32 = 0.327_591_1;
const A: [f32; 5] = [
    0.254_829_6,
    -0.284_496_74,
    1.421_413_7,
    -1.453_152,
    1.061_405_4,
];

/// Whether this build reaches a vector kernel rather than [`gelu_scalar`].
///
/// The question callers actually have is not "which instruction set is this"
/// but "is taking the escape hatch out of a tensor library worth it here",
/// and the answer is yes exactly when a vector path exists. On aarch64 NEON is
/// architectural, so it always does; on x86 it takes AVX-512, and without it
/// the scalar rows below lose to a SIMD library op.
pub fn vectorised() -> bool {
    cfg!(target_arch = "aarch64") || Avx512::detect().is_some()
}

/// Reduce `wide` — `[rows, 2·inter]`, gate half then up half — to
/// `[rows, inter]` holding `gelu(gate) · up`.
pub fn geglu(wide: &[f32], rows: usize, inter: usize) -> Vec<f32> {
    debug_assert_eq!(wide.len(), rows * 2 * inter);
    let mut out = vec![0.0f32; rows * inter];
    let simd = Avx512::detect();
    for r in 0..rows {
        let base = r * 2 * inter;
        row(
            &wide[base..base + inter],
            &wide[base + inter..base + 2 * inter],
            &mut out[r * inter..(r + 1) * inter],
            simd,
        );
    }
    out
}

/// The same reduction from two `[rows, inter]` buffers rather than one
/// interleaved `[rows, 2·inter]`.
///
/// Which shape the halves arrive in is the caller's `Wi` layout, and both exist:
/// one wide projection leaves them interleaved, two projections leave them
/// apart. The row kernel never cared — it has taken two pointers all along — so
/// this spares the caller a copy to fabricate a layout it does not have.
pub fn geglu_split(gate: &[f32], up: &[f32], rows: usize, inter: usize) -> Vec<f32> {
    debug_assert_eq!(gate.len(), rows * inter);
    debug_assert_eq!(up.len(), gate.len());
    let mut out = vec![0.0f32; rows * inter];
    let simd = Avx512::detect();
    for r in 0..rows {
        let span = r * inter..(r + 1) * inter;
        row(&gate[span.clone()], &up[span.clone()], &mut out[span], simd);
    }
    out
}

/// One row, vectorised where the instructions exist.
fn row(gate: &[f32], up: &[f32], dst: &mut [f32], simd: Option<Avx512>) {
    // aarch64 needs no `simd` token: NEON is architectural rather than
    // detected, so unlike the x86 path there is nothing to check and no scalar
    // fallback to keep for the machines that lack it.
    #[cfg(target_arch = "aarch64")]
    {
        let _ = simd;
        // SAFETY: all three slices are `dst.len()` long, which is what the
        // kernel reads and writes.
        unsafe { row_neon(gate.as_ptr(), up.as_ptr(), dst.as_mut_ptr(), dst.len()) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if simd.is_some() {
            // SAFETY: the `Avx512` token exists only where detection found the
            // instructions; all three slices are the same length, which is what
            // the kernel reads and writes.
            unsafe { row_avx512(gate.as_ptr(), up.as_ptr(), dst.as_mut_ptr(), dst.len()) };
            return;
        }
        let _ = simd;
        for ((d, g), u) in dst.iter_mut().zip(gate).zip(up) {
            *d = gelu_scalar(*g) * *u;
        }
    }
}

/// Reference implementation, and what runs without AVX-512. Uses the same
/// approximation as the kernel so the two agree to the bit where they can.
///
/// Written as `0.5·x·erfc(-x/√2)` rather than the textbook
/// `0.5·x·(1 + erf(x/√2))`. The two are the same identity, but the second one
/// cancels catastrophically on the negative tail, where `erf → -1` and the sum
/// loses every significant bit it had. Since A&S 7.1.26 approximates *erfc*
/// directly — the `poly·t·exp(-z²)` below is `erfc(z)` — going through it
/// avoids ever forming that difference. `erfc` of a negative argument is
/// `2 - erfc(|·|)`, which is the branch, and it subtracts from 2 a value in
/// `[0, 1]`, so that direction is stable too.
fn gelu_scalar(x: f32) -> f32 {
    let z = x.abs() * INV_SQRT2;
    let t = 1.0 / (1.0 + P * z);
    let poly = (((A[4] * t + A[3]) * t + A[2]) * t + A[1]) * t + A[0];
    let ec = poly * t * (-z * z).exp();
    0.5 * x * if x > 0.0 { 2.0 - ec } else { ec }
}

/// The NEON counterpart of [`row_avx512`], four lanes at a time.
///
/// The ragged tail goes through [`gelu_scalar`] rather than a masked load,
/// which NEON has no equivalent of. At the widths this runs on (`inter` is
/// 2048 for ruri-v3-130m) the tail is at most three elements, so a branch there
/// costs nothing worth avoiding.
///
/// # Safety
///
/// All three pointers must be valid for `n` elements.
#[cfg(target_arch = "aarch64")]
unsafe fn row_neon(gate: *const f32, up: *const f32, dst: *mut f32, n: usize) {
    let tail = n % 4;
    let body = n - tail;

    let mut i = 0;
    while i < body {
        let y = gelu4(vld1q_f32(gate.add(i)));
        vst1q_f32(dst.add(i), vmulq_f32(y, vld1q_f32(up.add(i))));
        i += 4;
    }
    for j in body..n {
        *dst.add(j) = gelu_scalar(*gate.add(j)) * *up.add(j);
    }
}

/// `gelu` over four lanes, by the identity [`gelu_scalar`] documents: the A&S
/// polynomial gives `erfc` directly, so the negative tail never forms the
/// cancelling `1 + erf` difference.
///
/// # Safety
///
/// NEON is part of the aarch64 baseline; this is `unsafe` only because the
/// intrinsics are.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn gelu4(x: float32x4_t) -> float32x4_t {
    let z = vmulq_n_f32(vabsq_f32(x), INV_SQRT2);

    let t = vdivq_f32(
        vdupq_n_f32(1.0),
        vfmaq_f32(vdupq_n_f32(1.0), vdupq_n_f32(P), z),
    );
    let mut poly = vdupq_n_f32(A[4]);
    for a in A[..4].iter().rev() {
        poly = vfmaq_f32(vdupq_n_f32(*a), poly, t);
    }
    // `exp(-z^2)` is never positive, which is the range `exp_neon` is built for.
    let decay = exp_neon(vnegq_f32(vmulq_f32(z, z)));
    // `erfc(|x|/sqrt(2))`, in [0, 1]. See `gelu_scalar` for why this and not `erf`.
    let ec = vmulq_f32(vmulq_f32(poly, t), decay);
    let positive = vcgtq_f32(x, vdupq_n_f32(0.0));
    let erfc = vbslq_f32(positive, vsubq_f32(vdupq_n_f32(2.0), ec), ec);

    vmulq_f32(vmulq_n_f32(x, 0.5), erfc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn row_avx512(gate: *const f32, up: *const f32, dst: *mut f32, n: usize) {
    let tail = n % 16;
    let body = n - tail;
    let k: __mmask16 = if tail == 0 { 0 } else { (1u16 << tail) - 1 };

    let mut i = 0;
    while i < body {
        let x = _mm512_loadu_ps(gate.add(i));
        let y = gelu16(x);
        _mm512_storeu_ps(dst.add(i), _mm512_mul_ps(y, _mm512_loadu_ps(up.add(i))));
        i += 16;
    }
    if tail != 0 {
        let x = _mm512_maskz_loadu_ps(k, gate.add(body));
        let y = gelu16(x);
        let z = _mm512_mul_ps(y, _mm512_maskz_loadu_ps(k, up.add(body)));
        _mm512_mask_storeu_ps(dst.add(body), k, z);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gelu16(x: __m512) -> __m512 {
    let ax = _mm512_abs_ps(x);
    let z = _mm512_mul_ps(ax, _mm512_set1_ps(INV_SQRT2));

    let t = _mm512_div_ps(
        _mm512_set1_ps(1.0),
        _mm512_fmadd_ps(_mm512_set1_ps(P), z, _mm512_set1_ps(1.0)),
    );
    let mut poly = _mm512_set1_ps(A[4]);
    for a in A[..4].iter().rev() {
        poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(*a));
    }
    // `exp(-z²)` is never positive, which is the range `exp512` is built for.
    let decay = exp512(_mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), z), z));
    // `erfc(|x|/√2)`, in [0, 1]. See `gelu_scalar` for why this and not `erf`.
    let ec = _mm512_mul_ps(_mm512_mul_ps(poly, t), decay);
    let positive = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(x, _mm512_setzero_ps());
    let erfc = _mm512_mask_blend_ps(positive, ec, _mm512_sub_ps(_mm512_set1_ps(2.0), ec));

    _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5), x), erfc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Realistic activations: the MLP's gate sits roughly in this range.
    fn inputs(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 2_654_435_761usize % 4001) as f32 / 200.0) - 10.0)
            .collect()
    }

    /// The same, but out to +-40, which is where a vector kernel's `exp` stops
    /// being asked for an ordinary number.
    ///
    /// This range is not decoration. The NEON kernel shipped a NaN past the
    /// +-10 sweep above, because `gelu` feeds `exp` a `-z^2` and the exponent
    /// construction only broke once `|x|` passed about 13. A real forward found
    /// it immediately; the tests did not.
    fn wide_inputs(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 2_654_435_761usize % 16001) as f32 / 200.0) - 40.0)
            .collect()
    }

    /// The kernel must agree with what it replaced: candle's `gelu_erf`
    /// followed by a multiply.
    #[test]
    fn matches_candle() {
        let (rows, inter) = (3usize, 37usize);
        let wide = inputs(rows * 2 * inter);
        let dev = Device::Cpu;

        let want: Vec<f32> = {
            let mut v = Vec::with_capacity(rows * inter);
            for r in 0..rows {
                let base = r * 2 * inter;
                let gate = Tensor::from_slice(&wide[base..base + inter], inter, &dev).unwrap();
                let up =
                    Tensor::from_slice(&wide[base + inter..base + 2 * inter], inter, &dev).unwrap();
                let g = (gate.gelu_erf().unwrap() * up).unwrap();
                v.extend(g.to_vec1::<f32>().unwrap());
            }
            v
        };

        let got = geglu(&wide, rows, inter);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0, f64::max);
        // A&S 7.1.26 bounds erf at 1.5e-7; `gelu` scales that by 0.5·|x| and
        // the `up` factor scales it again, so over |x| ≤ 10 this is the room
        // the approximation needs.
        assert!(worst < 2e-5, "worst absolute difference {worst:e}");
    }

    /// The vector kernel against the scalar reference it is supposed to
    /// reproduce. On x86 that pairing only exists where AVX-512 does; on
    /// aarch64 the kernel is architectural, so this always has something to
    /// compare and `geglu` never reaches `gelu_scalar` on its own.
    #[test]
    fn simd_and_scalar_agree() {
        if cfg!(not(target_arch = "aarch64")) && Avx512::detect().is_none() {
            return;
        }
        let (rows, inter) = (1usize, 512usize);
        let wide = inputs(rows * 2 * inter);

        let simd = geglu(&wide, rows, inter);
        let scalar: Vec<f32> = (0..inter)
            .map(|j| gelu_scalar(wide[j]) * wide[inter + j])
            .collect();

        // Relative, not absolute: the two differ only in which `exp` they call
        // (libm's on the scalar side, `exp512` or `exp_neon` on the other) and
        // the gated values here run out to ~100, where a 1e-7 relative gap is
        // microns of absolute difference that an absolute bound would flag as a
        // failure.
        let worst = simd
            .iter()
            .zip(&scalar)
            .filter(|(_, b)| b.abs() > 1e-3)
            .map(|(a, b)| ((*a as f64 - *b as f64) / *b as f64).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-6, "worst relative difference {worst:e}");
    }

    /// Every value the kernel produces must be a number, over a range wider
    /// than a well-behaved layer's. The tails saturate rather than overflow, so
    /// there is nothing here an `exp` may return `inf` or `NaN` for.
    #[test]
    fn stays_finite_on_wide_inputs() {
        let (rows, inter) = (4usize, 257usize);
        let wide = wide_inputs(rows * 2 * inter);
        for (i, v) in geglu(&wide, rows, inter).iter().enumerate() {
            assert!(v.is_finite(), "lane {i} is {v}");
        }
    }

    /// The kernel against `gelu_erf` out where the approximation is under the
    /// most strain, rather than only over the comfortable range.
    #[test]
    fn matches_candle_on_wide_inputs() {
        let (rows, inter) = (3usize, 61usize);
        let wide = wide_inputs(rows * 2 * inter);
        let dev = Device::Cpu;

        let mut want = Vec::with_capacity(rows * inter);
        for r in 0..rows {
            let base = r * 2 * inter;
            let gate = Tensor::from_slice(&wide[base..base + inter], inter, &dev).unwrap();
            let up =
                Tensor::from_slice(&wide[base + inter..base + 2 * inter], inter, &dev).unwrap();
            let g = (gate.gelu_erf().unwrap() * up).unwrap();
            want.extend(g.to_vec1::<f32>().unwrap());
        }

        let got = geglu(&wide, rows, inter);
        // Relative, because `gelu(x) -> x` on the positive tail and the `up`
        // factor takes the product out to ~1600 here, where the absolute bound
        // the narrow test uses would be meaningless.
        let worst = got
            .iter()
            .zip(&want)
            .filter(|(_, b)| b.abs() > 1e-3)
            .map(|(a, b)| ((*a as f64 - *b as f64) / *b as f64).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-5, "worst relative difference {worst:e}");
    }

    /// GELU's defining values, which a sign or `abs` slip would break.
    #[test]
    fn matches_gelu_at_known_points() {
        for (x, want) in [(0.0, 0.0), (1.0, 0.841_345), (-1.0, -0.158_655)] {
            let got = gelu_scalar(x);
            assert!(
                (got - want).abs() < 1e-5,
                "gelu({x}) = {got}, expected {want}"
            );
        }
        // Far out on the tails it saturates to the identity and to zero.
        assert!((gelu_scalar(8.0) - 8.0).abs() < 1e-5);
        assert!(gelu_scalar(-8.0).abs() < 1e-5);
    }

    /// The gate half and the up half must not be swapped: only the gate goes
    /// through GELU.
    #[test]
    fn gates_the_first_half_only() {
        let (rows, inter) = (1usize, 4usize);
        // gate = [1, 1, 1, 1], up = [1, 2, 3, 4]
        let wide = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        let got = geglu(&wide, rows, inter);
        let g1 = gelu_scalar(1.0);
        for (j, g) in got.iter().enumerate() {
            let want = g1 * (j as f32 + 1.0);
            assert!((g - want).abs() < 1e-6, "lane {j}: {g} vs {want}");
        }
    }
}
