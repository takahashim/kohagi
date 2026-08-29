//! The burn-flex device: the CPU, on Burn instead of candle.
//!
//! burn-flex is Burn's own fast portable CPU backend — the same `gemm` crate
//! candle uses for its Linux GEMM, plus SIMD elementwise kernels through
//! `macerator`. It is not a CubeCL backend, so a `#[cube]` kernel cannot reach
//! it; what it offers instead is a documented way down to the buffer, which is
//! what the two overrides below use.
//!
//! Measured on an 8-core Zen 4 against `--device cpu` on `ruri-v3-130m`, 16
//! rows of ~450 tokens, at `1 - cosine` 1.2e-12:
//!
//! | | rows/s |
//! | --- | ---: |
//! | one wide forward, generic kernels | 0.9 |
//! | fanned out across the pool | 5.7 |
//! | + two contiguous `Wi` matmuls | 6.1 |
//! | `--device cpu` (candle) | 6.8 |
//!
//! The fan-out is the whole difference and it is not Burn's to make: Burn
//! parallelises inside an operation, and that does not substitute for running
//! independent forwards at once. `Shape::fan_out` is what asks for it.

use anyhow::Result;
use burn::backend::flex::FlexTensor;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

use super::{weights, BurnEncoder, Forward, FusedOps, Shape};
use crate::encoder::{Activation, Config};

/// f32 throughout. There is no half-precision path here: `--precision bf16` is
/// the hand-written candle kernel in `crate::bf16`, and f16 is the Vulkan
/// device's recipe.
type Cpu = burn::backend::Flex;

/// One row per forward at seq 512, which is half what the candle CPU path
/// allows itself.
///
/// Same reasoning — a forward holds `rows * heads * seq^2` scores, and narrow
/// units load-balance across the pool better than wide ones — but measured
/// again here, because the two engines do not agree on where the optimum is:
/// one row per unit reached 5.7 rows/s against 4.8 at two and 3.0 at four.
const CPU_ATTN_BUDGET: usize = 512 * 512;

impl FusedOps for Cpu {
    /// Splitting `Wi` is worth +9% here: the wide form leaves gate and up as
    /// strided views, and the elementwise path walks those badly.
    const SPLIT_WI: bool = true;

    /// `burn_nn`'s RoPE builds a sign matrix and multiplies by it, then
    /// concatenates; candle-nn has a fused kernel and this is the gap. One pass
    /// over the buffer instead of seven measured 54x on the operation and +10%
    /// end to end.
    fn rope(x: Tensor<Self, 4>, cos: Tensor<Self, 4>, sin: Tensor<Self, 4>) -> Tensor<Self, 4> {
        let [rows, heads, seq, hd] = x.dims();
        let half = hd / 2;
        let (xp, cp, sp) = (
            contiguous(x.into_primitive().tensor()),
            contiguous(cos.into_primitive().tensor()),
            contiguous(sin.into_primitive().tensor()),
        );
        let (xs, cs, ss) = (slice(&xp), slice(&cp), slice(&sp));

        let mut out = vec![0f32; xs.len()];
        for r in 0..rows * heads {
            for p in 0..seq {
                let (o, t) = ((r * seq + p) * hd, p * hd);
                for i in 0..half {
                    let (a, b) = (xs[o + i], xs[o + i + half]);
                    out[o + i] = a * cs[t + i] - b * ss[t + i];
                    out[o + i + half] = b * cs[t + i + half] + a * ss[t + i + half];
                }
            }
        }
        rebuild(out, [rows, heads, seq, hd])
    }

    /// The vectorised GeGLU this crate already owns, from the bf16 module — it
    /// is f32 in and f32 out and has nothing to do with bf16 beyond living
    /// beside the path that needed it first. Falls back to burn-flex's own
    /// gelu-then-multiply without AVX-512, which is faster than that kernel's
    /// own scalar path.
    ///
    /// Writing a *scalar* fused kernel here was tried and measured: it changes
    /// nothing, because burn-flex's gelu and multiply are already SIMD and one
    /// pass of scalar code loses more on vectorisation than it wins on memory
    /// traffic. Only a vector kernel is worth the escape hatch.
    fn geglu(gate: Tensor<Self, 3>, up: Tensor<Self, 3>, act: Activation) -> Tensor<Self, 3> {
        // Gated on the vector path actually existing: without AVX-512
        // `crate::bf16::geglu` falls back to scalar rows, and burn-flex's own
        // SIMD gelu-then-multiply beats that. The escape hatch is only worth
        // taking when it lands somewhere better.
        #[cfg(target_arch = "x86_64")]
        if act == Activation::Gelu && crate::bf16::simd::Avx512::detect().is_some() {
            let dims = gate.dims();
            let inter = dims[2];
            let rows = dims[0] * dims[1];
            let (gp, up_) = (
                contiguous(gate.into_primitive().tensor()),
                contiguous(up.into_primitive().tensor()),
            );
            return rebuild(
                crate::bf16::geglu::geglu_split(slice(&gp), slice(&up_), rows, inter),
                [dims[0], dims[1], inter],
            );
        }
        super::geglu_composed(gate, up, act)
    }
}

fn contiguous(t: FlexTensor) -> FlexTensor {
    if t.is_contiguous() {
        t
    } else {
        t.to_contiguous()
    }
}

/// `FlexTensor::as_slice` is `bytemuck::cast_slice` over the tensor's own
/// storage — a view, not a copy, which is what makes descending to a kernel
/// worth doing at all.
fn slice(t: &FlexTensor) -> &[f32] {
    t.as_slice::<f32>().expect("contiguous f32 storage")
}

fn rebuild<const D: usize>(values: Vec<f32>, shape: [usize; D]) -> Tensor<Cpu, D> {
    Tensor::from_primitive(TensorPrimitive::Float(FlexTensor::from_data(
        TensorData::new(values, shape),
    )))
}

/// Load a checkpoint for the CPU.
pub fn load(weights: &std::path::Path, config: &Config) -> Result<BurnEncoder> {
    let checkpoint = weights::Checkpoint::open(weights)?;
    let device = Default::default();
    let model: Box<dyn Forward + Send + Sync> =
        Box::new(weights::load::<Cpu>(&checkpoint, config, false, &device)?);
    Ok(BurnEncoder {
        model,
        dim: config.hidden_size,
        shape: Shape {
            budget: CPU_ATTN_BUDGET,
            fan_out: true,
        },
    })
}
