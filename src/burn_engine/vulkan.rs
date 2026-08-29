//! The Vulkan device for the Burn engine: any Vulkan 1.2 GPU, no vendor runtime.
//!
//! CubeCL compiles the same kernels to SPIR-V, which every Mesa and every vendor
//! driver already speaks. On the Radeon 780M this was developed against, that
//! means no ROCm install, no `HSA_OVERRIDE_GFX_VERSION`, and no dependence on a
//! GPU being on AMD's support list.
//!
//! ## What it costs to start
//!
//! CubeCL autotunes its kernels on first use and caches the result. Cold that is
//! about 12 s; warm it is about 0.4 s on top of the model load, and the cache is
//! roughly 60 KB. `super::wgpu::configure_tune_cache` decides where it lands.

use anyhow::Result;

use super::wgpu::{claim_precision, configure_tune_cache};
use super::{weights, BurnEncoder, Forward, FusedOps, Shape};
use crate::encoder::Config;
use crate::model::Precision;

// Both precisions take the generic kernels. Fusing RoPE by hand here is the
// wrong shape for a GPU — CubeCL already fuses across the graph — and splitting
// `Wi` measured 7% slower than one wider matmul.
impl FusedOps for Exact {}
impl FusedOps for Half {}

/// f32 throughout: the same vectors as the CPU path (worst `1 - cosine` 8.5e-13
/// measured against it), and slower than the CPU's bf16 path. Kept because it is
/// what makes a disagreement diagnosable — if `f16` drifts, this says whether the
/// cause is the precision recipe or the implementation.
type Exact = burn::backend::Vulkan;

/// f16 storage and matmuls with f32 reductions — see `encoder`'s precision note.
type Half = burn::backend::Vulkan<burn::tensor::f16, i32>;

/// Rows allowed in one forward at padded length `seq`, as `budget / seq^2`.
///
/// The attention scores dominate: `[rows, heads, seq, seq]`, materialized in the
/// tensor's dtype and again in f32 for the softmax. Measured at 17.8 MiB per row
/// at seq 473, which is what this bounds.
///
/// Smaller than the candle path's [`crate::model::GPU_ATTN_BUDGET`] on purpose.
/// Throughput here is flat in the row count — 19.3 rows/s at one row against
/// 20.5 at sixty-four, because a single 473-token row already saturates the
/// device — so a tighter bound on memory costs nothing to buy.
const VULKAN_ATTN_BUDGET: usize = 8 * 512 * 512;

/// Load a checkpoint onto a Vulkan device.
pub fn load(
    weights: &std::path::Path,
    config: &Config,
    precision: Precision,
) -> Result<BurnEncoder> {
    configure_tune_cache();
    let checkpoint = weights::Checkpoint::open(weights)?;
    // Claimed here rather than on entry: `Checkpoint::open` reads the file on
    // the CPU and can fail without the device ever being touched, and a claim
    // made before that would refuse a later, legitimate load in the other
    // precision. The next line is what actually binds the element type.
    claim_precision("Vulkan", precision)?;
    let model: Box<dyn Forward + Send + Sync> = match precision {
        Precision::F16 => {
            let device = Default::default();
            Box::new(weights::load::<Half>(
                &checkpoint,
                config,
                true,
                VULKAN_ATTN_BUDGET,
                &device,
            )?)
        }
        _ => {
            let device = Default::default();
            Box::new(weights::load::<Exact>(
                &checkpoint,
                config,
                false,
                VULKAN_ATTN_BUDGET,
                &device,
            )?)
        }
    };
    Ok(BurnEncoder {
        model,
        dim: config.hidden_size,
        shape: Shape {
            budget: VULKAN_ATTN_BUDGET,
            // One GPU: fanning out would only make threads contend over command
            // submission and multiply scratch memory.
            fan_out: false,
            max_rows: usize::MAX,
        },
    })
}
