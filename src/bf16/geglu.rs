//! Fused GeGLU for the bf16 MLP.
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
//! Accuracy: `erf` is Abramowitz & Stegun 7.1.26, whose absolute error is
//! bounded by 1.5e-7 — deliberately weaker than candle's, and chosen because
//! `--precision bf16` is opt-in and already sits three orders of magnitude
//! above that. `tests::matches_candle` pins the end result against
//! `gelu_erf`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::simd::{exp512, has_avx512f};

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

/// Reduce `wide` — `[rows, 2·inter]`, gate half then up half — to
/// `[rows, inter]` holding `gelu(gate) · up`.
pub fn geglu(wide: &[f32], rows: usize, inter: usize) -> Vec<f32> {
    debug_assert_eq!(wide.len(), rows * 2 * inter);
    let mut out = vec![0.0f32; rows * inter];
    let simd = has_avx512f();
    for r in 0..rows {
        let base = r * 2 * inter;
        let gate = &wide[base..base + inter];
        let up = &wide[base + inter..base + 2 * inter];
        let dst = &mut out[r * inter..(r + 1) * inter];

        #[cfg(target_arch = "x86_64")]
        if simd {
            // SAFETY: guarded by the feature check; all three slices are
            // `inter` long, which is what the kernel reads and writes.
            unsafe { row_avx512(gate.as_ptr(), up.as_ptr(), dst.as_mut_ptr(), inter) };
            continue;
        }
        let _ = simd;
        for ((d, g), u) in dst.iter_mut().zip(gate).zip(up) {
            *d = gelu_scalar(*g) * *u;
        }
    }
    out
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

    #[test]
    fn simd_and_scalar_agree() {
        if !has_avx512f() {
            return;
        }
        let (rows, inter) = (1usize, 512usize);
        let wide = inputs(rows * 2 * inter);

        let simd = geglu(&wide, rows, inter);
        let scalar: Vec<f32> = (0..inter)
            .map(|j| gelu_scalar(wide[j]) * wide[inter + j])
            .collect();

        // Relative, not absolute: the two differ only in which `exp` they call
        // — libm's on the scalar side, `exp512` on the other — and the gated
        // values here run out to ~100, where a 1e-7 relative gap is microns of
        // absolute difference that an absolute bound would flag as a failure.
        let worst = simd
            .iter()
            .zip(&scalar)
            .filter(|(_, b)| b.abs() > 1e-3)
            .map(|(a, b)| ((*a as f64 - *b as f64) / *b as f64).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-6, "worst relative difference {worst:e}");
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
