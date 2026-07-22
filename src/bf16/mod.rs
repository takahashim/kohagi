//! Optional bf16 fast path for x86_64 CPUs with AVX512-BF16 (Zen 4, Sapphire
//! Rapids and newer).
//!
//! [`gemm`] is a hand-written bf16 matrix kernel, ~2× candle's f32 gemm; the
//! encoder in [`modernbert`] uses it for the four projection `Linear`s and
//! leaves the rest of the forward in f32. Opt in with `--precision bf16`
//! (`Precision::Bf16`) — f32 stays the default everywhere so the same text
//! yields the same vector on every machine.
//!
//! On short texts this is ~2× faster than the f32 path at cosine ≈ 0.99999;
//! on long ones the gap closes because f32 attention dominates. Both the
//! module and the CPU feature are checked before use: non-x86_64 builds skip
//! this module entirely, and [`gemm::supported`] gates it at load time.

pub mod gemm;
pub mod modernbert;

pub use gemm::supported;
pub use modernbert::Bf16ModernBert;
