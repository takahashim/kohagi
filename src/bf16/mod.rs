//! Optional bf16 fast path for x86_64 CPUs with AVX512-BF16 (Zen 4, Sapphire
//! Rapids and newer).
//!
//! [`gemm`] is a hand-written bf16 matrix kernel, ~2× candle's f32 gemm; the
//! encoder in [`modernbert`] uses it for the four projection `Linear`s and
//! leaves the rest of the forward in f32.
//!
//! [`softmax`] and [`geglu`] are the other two kernels, and they change no
//! precision at all — they are f32 in and f32 out. They exist because candle
//! evaluates `exp` and `erf` one element at a time, which profiling put at 28%
//! and 10% of a 512-token forward respectively, and because fusing lets the
//! attention mask and the GELU gate be applied without materializing a tensor
//! for each. Vectorizing the two is worth 1.37× at 512 tokens by itself.
//!
//! Opt in with `--precision bf16` (`Precision::Bf16`) — f32 stays the default
//! everywhere so the same text yields the same vector on every machine.
//!
//! [`modernbert`] also walks the band rather than the whole score matrix in
//! the sliding-window layers, which is 12 of ruri-v3's 19 and where three
//! quarters of a 512-token score matrix is masked off. That one is exactly
//! equivalent, down to the bit.
//!
//! On short texts this is ~2.3× faster than the f32 path at cosine ≈ 0.99999,
//! and ~2.0× at 512 tokens, where the attention matmuls still run in f32.
//! Both the module and the CPU features are checked before use: non-x86_64
//! builds skip this module entirely, [`gemm::supported`] gates it at load
//! time, and both elementwise kernels fall back to scalar rows when AVX-512
//! is absent.

// [`geglu`] and [`simd`] build everywhere; the rest is x86_64 only. The split
// is not tidiness: [`geglu`] is f32 in and f32 out and changes no precision, so
// it is useful to any CPU engine, and it now carries a NEON kernel beside the
// AVX-512 one. What keeps it here is that it shares [`simd`]'s vector `exp`
// with [`softmax`], which is x86 only. See the note in [`geglu`] on why the
// name of this module undersells what is in it.
pub mod geglu;
#[cfg(target_arch = "x86_64")]
pub mod gemm;
#[cfg(target_arch = "x86_64")]
pub mod modernbert;
pub mod simd;
#[cfg(target_arch = "x86_64")]
pub mod softmax;

#[cfg(target_arch = "x86_64")]
pub use gemm::supported;
#[cfg(target_arch = "x86_64")]
pub use modernbert::Bf16ModernBert;
