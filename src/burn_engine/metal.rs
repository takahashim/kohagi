//! The Metal device for the Burn engine: the Apple GPU, on Burn instead of candle.
//!
//! The same wgpu runtime as [`super::vulkan`] on the other API — CubeCL emits
//! Metal Shading Language for it directly rather than going through WGSL — and
//! so the same precision recipe and the same generic kernels. What differs is
//! measured here rather than assumed: `crate::encoder` carries a hand-fused
//! GeGLU kernel for candle's Metal path, and whether CubeCL's own fusion makes
//! up for not having one is the comparison `--device metal-burn` exists for.
//!
//! Transitional, like `cpu-burn`: it is here so the two Metal paths can be
//! compared in one binary, and goes away when one of them wins.
//!
//! ## What it measures at, on an M1 Pro
//!
//! In steady state — the same process, the same shapes again — `f16` is level
//! with candle's Metal path or a little ahead: 0.124 s against 0.135 for 64
//! texts of 42 tokens, 1.57 s against 1.67 at 460, 4.23 s against 4.30 for 256
//! texts of mixed length. Worst `1 - cosine` against the CPU is 2.6e-7 for
//! `f16` and 2.3e-13 for `f32`.
//!
//! What the CLI sees is that plus two per-process costs candle does not pay:
//! about 0.45 s more to load (device set-up and the kernels' first Metal
//! compile — Apple caches those per executable, so a fresh binary's first run
//! is seconds slower still), and roughly half a second per distinct padded
//! length, which is CubeCL generating and compiling that shape's kernels. So
//! 256 mixed-length texts take 5.0 s against candle's 4.4, and 64 short ones
//! 1.3 s against 0.5. The first run ever of a shape also autotunes it, which
//! took 17 s for those 256 texts and is cached under the user's cache
//! directory (see `super::wgpu`) from then on.
//!
//! `f32` is the exact path, 10–20% slower than `f16` and kept for the reason
//! [`super::vulkan`] gives.

use anyhow::Result;

use super::wgpu::{claim_precision, configure_tune_cache};
use super::{weights, BurnEncoder, Forward, FusedOps, Shape};
use crate::encoder::Config;
use crate::model::Precision;

// Both precisions take the generic kernels, for the reasons `vulkan` gives.
impl FusedOps for Exact {}
impl FusedOps for Half {}

/// f32 throughout: what makes a disagreement diagnosable, as on Vulkan.
type Exact = burn::backend::Metal;

/// f16 storage and matmuls with f32 reductions — see `encoder`'s precision note.
type Half = burn::backend::Metal<burn::tensor::f16, i32>;

/// Rows allowed in one forward at padded length `seq`, as `budget / seq^2`.
///
/// The Vulkan figure, until measured here. Unified memory changes what a
/// row costs to hold but not the reason the bound is tight: throughput on a
/// GPU is flat in the row count once one row fills the device.
const METAL_ATTN_BUDGET: usize = 8 * 512 * 512;

/// Load a checkpoint onto the Metal device.
pub fn load(
    weights: &std::path::Path,
    config: &Config,
    precision: Precision,
) -> Result<BurnEncoder> {
    configure_tune_cache();
    let checkpoint = weights::Checkpoint::open(weights)?;
    // After the CPU-side read, for the reason `vulkan::load` gives.
    claim_precision("Metal", precision)?;
    let model: Box<dyn Forward + Send + Sync> = match precision {
        Precision::F16 => {
            let device = Default::default();
            Box::new(weights::load::<Half>(
                &checkpoint,
                config,
                true,
                METAL_ATTN_BUDGET,
                &device,
            )?)
        }
        _ => {
            let device = Default::default();
            Box::new(weights::load::<Exact>(
                &checkpoint,
                config,
                false,
                METAL_ATTN_BUDGET,
                &device,
            )?)
        }
    };
    Ok(BurnEncoder {
        model,
        dim: config.hidden_size,
        shape: Shape {
            budget: METAL_ATTN_BUDGET,
            // One GPU: wide forwards back to back, as on Vulkan.
            fan_out: false,
            max_rows: usize::MAX,
        },
    })
}
