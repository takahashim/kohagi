//! CPU feature detection and the vector math the kernels share.
//!
//! [`super::softmax`] and [`super::geglu`] are siblings — neither is built on
//! the other — but both need the same AVX-512 `exp` and the same question
//! answered about the CPU underneath. Both live here so that neither module
//! has to export a primitive on the other's behalf.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Proof that this CPU has AVX-512F, and the only way to reach the vector
/// path of the elementwise kernels ([`super::softmax`], [`super::geglu`]).
///
/// The kernels are `unsafe` behind a runtime feature check, and the check used
/// to travel to them as a plain `bool`. A caller could then get it wrong, and
/// one did: a test passed `true` unconditionally and ran AVX-512 on whatever
/// CPU it landed on, which is fine on a developer box with the instructions
/// and an illegal instruction on CI without them. A token that only
/// [`Avx512::detect`] can produce makes asking for an unavailable path
/// unexpressible rather than merely documented.
#[derive(Clone, Copy)]
pub struct Avx512(());

impl Avx512 {
    /// `Some` on a CPU that can run the vector kernels; callers fall back to
    /// scalar rows on `None`.
    pub fn detect() -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return Some(Self(()));
            }
        }
        None
    }
}

/// Whether [`super::gemm`] can run at all — it has no scalar fallback, so
/// this gates `--precision bf16` at load time rather than per call.
pub fn has_avx512bf16() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        Avx512::detect().is_some() && std::is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `exp` for 16 lanes: `exp(x) = 2^n · exp(r)`, with `n = round(x·log2 e)` and
/// `r = x - n·ln2` split hi/lo (Cody-Waite) so the reduction stays exact. The
/// polynomial is Cephes' `expf`, accurate to about 1 ulp.
///
/// Callers pass `x ≤ 0` — softmax exponentiates differences from the row
/// maximum, `geglu` the `-z²` of an `erfc` — so there is no overflow path, and
/// `scalef` handles the underflow tail.
///
/// # Safety
///
/// The caller must hold an [`Avx512`] token, or otherwise be inside a
/// function that already enables the feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn exp512(x: __m512) -> __m512 {
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

/// `exp` for 4 lanes on aarch64, by the same recipe as [`exp512`]: the same
/// Cody-Waite reduction, the same Cephes polynomial, the same coefficients. The
/// two kernels agree to the bit wherever the hardware rounds the same way, and
/// that is deliberate: a vector `gelu` that answered differently on ARM than on
/// x86 would put the platform into the embedding.
///
/// NEON has no `scalef`, so `2^n` is built from the exponent field, and that is
/// where the care goes. Two things are needed and neither is optional:
///
/// - **Two halves.** A single shift leaves the normal range once `n` reaches
///   -127, which `geglu` hits at `|x|` around 13. Splitting `2^n` into
///   `2^(n/2) * 2^(n - n/2)` doubles the reach for two instructions.
/// - **A clamp at zero.** Past even that, a negative biased exponent shifted
///   left by 23 runs into the sign bit and reinterprets as an arbitrary float,
///   which is how this produced NaN rather than the underflow it owed. Clamping
///   the biased exponent at 0 makes each half exactly 0.0, so the product is
///   the 0 that `exp` of a large negative argument should be. `scalef` does
///   this for the AVX-512 side; here it is spelled out.
///
/// Callers pass `x <= 0`, as with [`exp512`], so there is no overflow path.
/// Unlike [`exp512`], whose callers bottom out at softmax's -87, `geglu` hands
/// this the `-z^2` of an `erfc` and so reaches several hundred negative.
///
/// # Safety
///
/// NEON is part of the aarch64 baseline, so this needs no feature detection.
/// It is `unsafe` only because the intrinsics are.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn exp_neon(x: float32x4_t) -> float32x4_t {
    // Written as the quotient for the same reason as `exp512`: `n * LN2_HI` has
    // to be exact, and 355/512 is.
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

    let n = vrndnq_f32(vmulq_n_f32(x, std::f32::consts::LOG2_E));
    // `vfmsq_f32(a, b, c)` is `a - b * c`, which is x86's `fnmadd`.
    let r = vfmsq_f32(x, n, vdupq_n_f32(LN2_HI));
    let r = vfmsq_f32(r, n, vdupq_n_f32(LN2_LO));

    let mut y = vdupq_n_f32(P[0]);
    for p in &P[1..] {
        y = vfmaq_f32(vdupq_n_f32(*p), y, r);
    }
    // y * r^2 + r + 1
    y = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0)), y, vmulq_f32(r, r));

    // 2^n, in two halves and clamped at zero. See the note above: both are
    // load-bearing, and the clamp is what turns an out-of-range exponent into
    // the underflow to 0.0 it should always have been.
    let ni = vcvtq_s32_f32(n);
    let lo = vshrq_n_s32(ni, 1);
    let hi = vsubq_s32(ni, lo);
    let pow2 = |e: int32x4_t| {
        let biased = vmaxq_s32(vaddq_s32(e, vdupq_n_s32(127)), vdupq_n_s32(0));
        vreinterpretq_f32_s32(vshlq_n_s32(biased, 23))
    };
    vmulq_f32(vmulq_f32(y, pow2(lo)), pow2(hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sweep the whole argument range the callers can hand `exp512` and
    /// compare against libm, which is what candle's scalar path uses.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn exp512_matches_libm() {
        if Avx512::detect().is_none() {
            return;
        }
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f32;
        // -87 is the softmax floor; nothing below it reaches the kernel.
        let mut x = -87.0f32;
        while x <= 0.0 {
            let want = x.exp() as f64;
            let mut lanes = [0.0f32; 16];
            // SAFETY: guarded by the feature check above.
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
            x += 0.000_976_562_5; // 1/1024, so the sweep hits exact binary points
        }
        assert!(
            worst < 1e-6,
            "worst relative error {worst:e} at x = {worst_at}"
        );
    }

    /// The same sweep for the NEON kernel, held to the same bound, so the two
    /// architectures are known to agree with libm rather than only with each
    /// other.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn exp_neon_matches_libm() {
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f32;
        // Down to -800, not softmax's -87: `geglu` passes `-z^2`, which for the
        // activations a real MLP produces runs far past where the exponent
        // construction needs its clamp. Sweeping only the softmax range is what
        // let a NaN reach the model.
        let mut x = -800.0f32;
        while x <= 0.0 {
            let want = x.exp() as f64;
            let mut lanes = [0.0f32; 4];
            // SAFETY: NEON is part of the aarch64 baseline.
            let got = unsafe {
                let v = exp_neon(vdupq_n_f32(x));
                vst1q_f32(lanes.as_mut_ptr(), v);
                lanes[0] as f64
            };
            assert!(got.is_finite(), "exp_neon({x}) = {got}");
            if want >= f32::MIN_POSITIVE as f64 {
                // Where the answer is a normal float, the same 1e-6 `exp512` is
                // held to.
                let rel = ((got - want) / want).abs();
                if rel > worst {
                    worst = rel;
                    worst_at = x;
                }
            } else {
                // Below that the answer is subnormal or zero, and the two-step
                // scaling rounds twice on the way there: measured 3.6e-4
                // relative at its worst, on values around 1e-42. Nothing is
                // spent fixing that. What still has to hold is that the result
                // is small and not garbage, which is the failure this range
                // exists to catch.
                assert!(
                    (0.0..1e-37).contains(&got),
                    "exp_neon({x}) = {got}, expected a subnormal or zero"
                );
            }
            x += 0.000_976_562_5;
        }
        assert!(
            worst < 1e-6,
            "worst relative error {worst:e} at x = {worst_at}"
        );
    }
}
