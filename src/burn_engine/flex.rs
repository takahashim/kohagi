//! The burn-flex device: the CPU, on Burn instead of candle.
//!
//! burn-flex is Burn's own fast portable CPU backend — the same `gemm` crate
//! candle uses for its Linux GEMM, plus SIMD elementwise kernels through
//! `macerator`. It is not a CubeCL backend, so a `#[cube]` kernel cannot reach
//! it; what it offers instead is a documented way down to the buffer, which is
//! what the three overrides below use.
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
//!
//! ## macOS is a different comparison
//!
//! The sentence above about "the same `gemm` crate candle uses" holds on Linux
//! and is the reason the Zen 4 numbers come out close. It does not hold on
//! macOS: there `Cargo.toml` gives candle the `accelerate` feature, so its
//! sgemm is Apple's AMX coprocessor and not `gemm` at all. Against that,
//! burn-flex's NEON kernels are running a different race — 395 GFLOP/s to
//! Accelerate's 2072 at `2048x512x2048` on an M1 Pro — and `--device cpu-burn`
//! measured 3.0x slower end to end rather than the 1.1x Zen 4 shows.
//!
//! Three things closed most of it, and the ladder at 64 texts of 460 tokens is
//! 8.24 s → 4.64 → 3.73 → 3.52, against `--device cpu`'s 2.72:
//!
//! - `gemm`'s own AMX kernels, which burn-flex offers as `apple-amx` and
//!   `Cargo.toml` now asks for. 395 GFLOP/s to 1240.
//! - [`FusedOps::add_mask`], because the mask's broadcast is over the head
//!   axis and burn-flex accelerates only innermost-axis ones.
//! - Reading RoPE's input where it lies rather than copying it contiguous.
//!
//! What is left is mostly not this crate's to fix: `gemm`'s AMX kernel reaches
//! 60% of Accelerate and packs its operands with scalar code, and burn-flex has
//! no BLAS feature to reach Accelerate through the way `burn-ndarray` does. The
//! scalar `erf` under `gelu` is the other item, and that one is a floor under
//! *both* engines on aarch64 rather than a gap between them — candle spends
//! more of its own profile there than burn does.

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
/// at 64 rows of ~460 tokens, one row per unit measured 12.5 s against 13.8 at
/// two, 13.4 at four and 16.4 at eight — and 59 s not fanned out at all.
const CPU_ATTN_BUDGET: usize = 512 * 512;

/// Rows a forward may hold whatever the budget allows.
///
/// The budget only bites on long inputs; on short ones it would let a whole
/// batch into one forward, and then nothing is fanned out and the pool sits
/// idle. This is what keeps the units coming. Measured on 64 texts of 42
/// tokens: 2.89 s uncapped, 1.92 at one row, 1.88 at two, **1.48 at four**,
/// 1.50 at eight — against 2.15 s for `--device cpu`.
///
/// `Weights::max_rows_per_forward` reaches the same 4 for the bf16 path, from
/// its own measurements against its own kernels. Two engines agreeing on a
/// constant neither derived from the other is worth noting, and worth not
/// reading too much into.
const MAX_ROWS_PER_FORWARD: usize = 4;

impl FusedOps for Cpu {
    /// Splitting `Wi` is worth +9% here: the wide form leaves gate and up as
    /// strided views, and the elementwise path walks those badly.
    const SPLIT_WI: bool = true;

    /// `burn_nn`'s RoPE builds a sign matrix and multiplies by it, then
    /// concatenates; candle-nn has a fused kernel and this is the gap. One pass
    /// over the buffer instead of seven measured 54x on the operation and +10%
    /// end to end.
    ///
    /// The head transpose comes free with it, because `x` is read where it lies.
    ///
    /// `x` reaches here as `[rows, seq, heads, head_dim]` with the middle two
    /// axes swapped, so asking for it contiguous first copies the whole tensor
    /// — and this kernel writes a fresh contiguous buffer anyway, so that copy
    /// bought nothing the loop below does not already do. Only the innermost
    /// axis has to be contiguous for the row slice, and the swap leaves it that
    /// way; anything else falls back rather than gathering element by element.
    ///
    /// Worth 3.73 s to 3.52 s on an M1 Pro at 64 texts of 460 tokens: one
    /// `[heads, seq, head_dim]` copy for each of `q` and `k`, in each of the
    /// nineteen layers.
    fn rope(x: Tensor<Self, 4>, cos: Tensor<Self, 4>, sin: Tensor<Self, 4>) -> Tensor<Self, 4> {
        let [rows, heads, seq, hd] = x.dims();
        let half = hd / 2;
        let (cp, sp) = (
            contiguous(cos.into_primitive().tensor()),
            contiguous(sin.into_primitive().tensor()),
        );
        let (cs, ss) = (slice(&cp), slice(&sp));

        let xp = x.into_primitive().tensor();
        let unit_last = xp.layout().strides()[3] == 1;
        let xp = if unit_last { xp } else { xp.to_contiguous() };
        let stride: Vec<usize> = xp.layout().strides().iter().map(|&s| s as usize).collect();
        let base = xp.layout().start_offset();
        let xs = xp.storage::<f32>();

        let mut out = vec![0f32; rows * heads * seq * hd];
        for r in 0..rows {
            for h in 0..heads {
                for p in 0..seq {
                    let src = base + r * stride[0] + h * stride[1] + p * stride[2];
                    let xrow = &xs[src..src + hd];
                    let o = ((r * heads + h) * seq + p) * hd;
                    let (dst, t) = (&mut out[o..o + hd], p * hd);
                    for i in 0..half {
                        let (a, b) = (xrow[i], xrow[i + half]);
                        dst[i] = a * cs[t + i] - b * ss[t + i];
                        dst[i + half] = b * cs[t + i + half] + a * ss[t + i + half];
                    }
                }
            }
        }
        rebuild(out, [rows, heads, seq, hd])
    }

    /// The mask add, on contiguous planes rather than through the broadcast.
    ///
    /// burn-flex accelerates exactly two broadcast shapes — a shared innermost
    /// row, and a per-row scalar — and this is neither: the mask repeats over
    /// the *head* axis, which is not the innermost one. Everything else falls to
    /// `binary_op_typed`, a scalar `StridedIter` walk over
    /// `[rows, heads, queries, keys]`, the widest tensor the model builds.
    ///
    /// The broadcast is a plane repeat rather than an element one, so it does
    /// not need burn to see it as one: each head's scores are a contiguous
    /// `queries * keys` block and the mask's own plane is contiguous beside it,
    /// so the add is a flat loop the compiler vectorises on any target. In
    /// place, since the scores are a fresh matmul result nothing else holds.
    ///
    /// Worth 4.64 s to 3.73 s on an M1 Pro at 64 texts of 460 tokens. Twelve of
    /// nineteen layers slide a window, and those are the ones whose mask carries
    /// a query axis and so misses the shared-row path even at one row.
    fn add_mask(scores: Tensor<Self, 4>, mask: Tensor<Self, 4>) -> Tensor<Self, 4> {
        let [rows, heads, queries, keys] = scores.dims();
        let [m_rows, m_heads, m_queries, m_keys] = mask.dims();
        // The encoder builds `[rows, 1, queries | 1, keys]` and nothing else;
        // anything that is not that shape goes back through burn.
        if (m_rows, m_heads, m_keys) != (rows, 1, keys) || !(m_queries == queries || m_queries == 1)
        {
            return scores + mask;
        }

        let mp = contiguous(mask.into_primitive().tensor());
        let ms = slice(&mp);
        let mut sp = contiguous(scores.into_primitive().tensor());
        let base = sp
            .layout()
            .contiguous_offsets()
            .expect("contiguous scores")
            .0;
        let ss = &mut sp.storage_mut::<f32>()[base..];

        let plane = queries * keys;
        for r in 0..rows {
            let m_plane = &ms[r * m_queries * keys..][..m_queries * keys];
            for h in 0..heads {
                let s_plane = &mut ss[(r * heads + h) * plane..][..plane];
                for (i, s_row) in s_plane.chunks_exact_mut(keys).enumerate() {
                    // `m_queries == 1` is the padding-only mask of a global
                    // layer, where every query reads the same row.
                    let m_row = &m_plane[if m_queries == 1 { 0 } else { i * keys }..][..keys];
                    for (s, m) in s_row.iter_mut().zip(m_row) {
                        *s += *m;
                    }
                }
            }
        }
        Tensor::from_primitive(TensorPrimitive::Float(sp))
    }

    /// The vectorised GeGLU this crate already owns, from the bf16 module — it
    /// is f32 in and f32 out and has nothing to do with bf16 beyond living
    /// beside the path that needed it first. Falls back to burn-flex's own
    /// gelu-then-multiply without AVX-512, which is faster than that kernel's
    /// own scalar path.
    ///
    /// Writing a *scalar* fused kernel here was tried and measured: it changes
    /// nothing, and one pass of scalar code loses more on vectorisation than it
    /// wins on memory traffic. Only a vector kernel is worth the escape hatch.
    ///
    /// The multiply it falls back to is SIMD; the `gelu` is not, on any target.
    /// `burn_flex::ops::activation::gelu` is a scalar closure through
    /// `unary_op` calling `erf` per element, and on aarch64 — where
    /// [`Avx512::detect`](crate::bf16::simd::Avx512::detect) can never succeed
    /// and this override therefore never fires — it is the largest compute item
    /// in the profile after the GEMM. It is not a *gap*, though: candle's
    /// `gelu_erf` is scalar on aarch64 too, and takes a larger share of its
    /// profile than this does of Burn's. Both engines want the same vectorised
    /// kernel, which is why fixing it does not belong here.
    fn geglu(gate: Tensor<Self, 3>, up: Tensor<Self, 3>, act: Activation) -> Tensor<Self, 3> {
        // Gated on the vector path actually existing: without AVX-512
        // `crate::bf16::geglu` falls back to scalar rows, and burn-flex's own
        // gelu-then-multiply beats that. The escape hatch is only worth taking
        // when it lands somewhere better.
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
    let model: Box<dyn Forward + Send + Sync> = Box::new(weights::load::<Cpu>(
        &checkpoint,
        config,
        false,
        CPU_ATTN_BUDGET,
        &device,
    )?);
    Ok(BurnEncoder {
        model,
        dim: config.hidden_size,
        shape: Shape {
            budget: CPU_ATTN_BUDGET,
            fan_out: true,
            max_rows: MAX_ROWS_PER_FORWARD,
        },
    })
}
