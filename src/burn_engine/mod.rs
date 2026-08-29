//! The Burn engine: ModernBERT on Burn's tensors rather than candle's.
//!
//! Like `crate::coreml`, and unlike the CPU/Metal/CUDA backends, this does not
//! go through candle: Burn has its own tensor type, so there is no `Device` for
//! [`crate::model::open_device`] to return and no way to hand a batch to
//! [`crate::model::run_batches`]. What it *does* share is everything either side
//! of the forward — the tokenizer, the length bucketing and row placement in
//! `crate::batch`, and `embed_row`'s pooling, truncation and normalization — so
//! a vector differs from the CPU path only by the arithmetic that produced the
//! hidden states.
//!
//! [`encoder`] and [`weights`] are generic over [`FusedOps`] and name no device.
//! A device module — [`vulkan`], [`flex`] — supplies the element type, the
//! memory budget, whether to fan out across cores, and any kernel it wants
//! instead of the generic one.

mod encoder;
#[cfg(feature = "cpu-burn")]
pub mod flex;
#[cfg(feature = "vulkan")]
pub mod vulkan;
mod weights;

use anyhow::Result;
use burn::tensor::{backend::Backend, Tensor};
use rayon::prelude::*;

use crate::encoder::Activation;

#[cfg(feature = "cpu-burn")]
pub use flex::load as load_flex;
#[cfg(feature = "vulkan")]
pub use vulkan::load as load_vulkan;

/// The Burn Book's "backend extension" pattern: a trait with a generic default
/// so every backend works, overridden where a device can do better.
///
/// Each operation here is a measured answer rather than a starting point.
/// burn-flex already implements softmax, gelu and layer_norm as SIMD backend
/// ops, so routing through `ActivationOps` / `ModuleOps` beats anything written
/// here; what is listed is what that did *not* cover. RoPE, because `burn_nn`
/// composes it from primitives (`matmul` against a sign matrix, then `cat`)
/// where candle-nn has a kernel. GeGLU, because this crate already owns a
/// vectorised one. The GEMMs, the head split and merge, and the mask add,
/// because on macOS burn-flex has no way to reach Accelerate, and copies —
/// one element at a time — any view its `reshape` cannot re-label. See
/// [`flex`] for each.
pub(crate) trait FusedOps: Backend {
    /// Whether to load `Wi` as two contiguous matrices rather than one wide one.
    ///
    /// Measured both ways on real weights: splitting is worth +9% on burn-flex,
    /// because the wide form leaves the gate and up halves as strided views that
    /// the elementwise path then walks badly, and **−7% on Vulkan**, where one
    /// wider matmul is the cheaper shape. `crate::encoder` records the same
    /// split for the same reason on the candle CPU path.
    const SPLIT_WI: bool = false;

    /// `[b, s, in] @ [in, out]`, the projection every layer makes six times.
    ///
    /// Here because the GEMM is the one place a backend's own choice of kernel
    /// decides the whole comparison: on macOS, candle's sgemm is Apple's AMX
    /// coprocessor through Accelerate, and burn-flex has no way to reach it. See
    /// the override in [`flex`].
    fn project(x: Tensor<Self, 3>, w: Tensor<Self, 2>) -> Tensor<Self, 3> {
        linear_composed(x, w)
    }

    /// One of q, k, v out of the fused `[rows, seq, 3 * hidden]` projection, as
    /// `[rows, heads, seq, head_dim]`.
    ///
    /// Composed, this is a narrow, a reshape and a transpose, and whether the
    /// reshape costs a copy is the backend's business: it does on burn-flex,
    /// whose `reshape` copies anything that is not contiguous from offset zero,
    /// and the copy is a scalar one. See the override in [`flex`], which builds
    /// the view directly.
    fn split_heads(qkv: Tensor<Self, 3>, part: usize, heads: usize) -> Tensor<Self, 4> {
        split_heads_composed(qkv, part, heads)
    }

    /// The attention output `[rows, heads, seq, head_dim]` back as
    /// `[rows, seq, hidden]`, for the output projection.
    ///
    /// This one is a real transpose — the head axis moves inside the sequence
    /// axis — so a copy is owed; the override in [`flex`] is about how that copy
    /// is made rather than whether.
    fn merge_heads(context: Tensor<Self, 4>) -> Tensor<Self, 3> {
        merge_heads_composed(context)
    }

    /// `x * cos + rotate_half(x) * sin`, over `[rows, heads, seq, head_dim]`.
    fn rope(x: Tensor<Self, 4>, cos: Tensor<Self, 4>, sin: Tensor<Self, 4>) -> Tensor<Self, 4> {
        rope_composed(x, cos, sin)
    }

    /// `q @ k^T`: `[rows, heads, queries, head_dim]` against
    /// `[rows, heads, keys, head_dim]`, to `[rows, heads, queries, keys]`.
    ///
    /// Batched over `rows * heads` with a 64-wide contraction, which is a shape
    /// a backend's general matmul may handle badly — see [`flex`], where it goes
    /// to Accelerate along with [`Self::project`].
    fn scores(q: Tensor<Self, 4>, k: Tensor<Self, 4>) -> Tensor<Self, 4> {
        scores_composed(q, k)
    }

    /// `probs @ v`: `[rows, heads, queries, keys]` against
    /// `[rows, heads, keys, head_dim]`, to `[rows, heads, queries, head_dim]`.
    fn context(probs: Tensor<Self, 4>, v: Tensor<Self, 4>) -> Tensor<Self, 4> {
        context_composed(probs, v)
    }

    /// `scores + pad + window`, over scores of `[rows, heads, queries, keys]`.
    ///
    /// `pad` is `[rows, 1, 1, keys]` — whether a key is padding is the key's
    /// property alone — and `window` is `[1, 1, queries, keys]` where the layer
    /// slides one, `None` where it attends globally. Both broadcast over the
    /// head axis, which a backend may have no fast path for; and summing them
    /// first would materialise `[rows, 1, queries, keys]` for nothing. See the
    /// override in [`flex`], which is both of those cases.
    fn add_mask(
        scores: Tensor<Self, 4>,
        pad: Tensor<Self, 4>,
        window: Option<Tensor<Self, 4>>,
    ) -> Tensor<Self, 4> {
        add_mask_composed(scores, pad, window)
    }

    /// The gated feed-forward's elementwise half: `act(gate) * up`.
    fn geglu(gate: Tensor<Self, 3>, up: Tensor<Self, 3>, act: Activation) -> Tensor<Self, 3> {
        geglu_composed(gate, up, act)
    }
}

/// The generic defaults, as free functions so an override can fall back to them.
///
/// One 2-D matmul rather than a batched 3-D one, which is the shape the GEMM
/// wants; the weights were transposed once at load for the same reason.
pub(crate) fn linear_composed<B: Backend>(x: Tensor<B, 3>, w: Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, seq, input] = x.dims();
    let output = w.dims()[1];
    x.reshape([batch * seq, input])
        .matmul(w)
        .reshape([batch, seq, output])
}

pub(crate) fn split_heads_composed<B: Backend>(
    qkv: Tensor<B, 3>,
    part: usize,
    heads: usize,
) -> Tensor<B, 4> {
    let [rows, seq, width] = qkv.dims();
    let hidden = width / 3;
    qkv.narrow(2, part * hidden, hidden)
        .reshape([rows, seq, heads, hidden / heads])
        .swap_dims(1, 2)
}

pub(crate) fn merge_heads_composed<B: Backend>(context: Tensor<B, 4>) -> Tensor<B, 3> {
    let [rows, heads, seq, head_dim] = context.dims();
    context
        .swap_dims(1, 2)
        .reshape([rows, seq, heads * head_dim])
}

pub(crate) fn scores_composed<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>) -> Tensor<B, 4> {
    q.matmul(k.swap_dims(2, 3))
}

pub(crate) fn context_composed<B: Backend>(probs: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
    probs.matmul(v)
}

pub(crate) fn add_mask_composed<B: Backend>(
    scores: Tensor<B, 4>,
    pad: Tensor<B, 4>,
    window: Option<Tensor<B, 4>>,
) -> Tensor<B, 4> {
    // The two masks summed first, so the wide tensor is touched once; a
    // backend that would rather not materialise the sum overrides `add_mask`.
    let mask = match window {
        Some(window) => pad + window,
        None => pad,
    };
    scores + mask
}

pub(crate) fn rope_composed<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let half = x.dims()[3] / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    x * cos + Tensor::cat(vec![-x2, x1], 3) * sin
}

pub(crate) fn geglu_composed<B: Backend>(
    gate: Tensor<B, 3>,
    up: Tensor<B, 3>,
    act: Activation,
) -> Tensor<B, 3> {
    let activated = match act {
        Activation::Gelu => burn::tensor::activation::gelu(gate),
        Activation::Silu => burn::tensor::activation::silu(gate),
    };
    activated * up
}

/// One loaded model's forward, with the element type erased.
///
/// A device may offer more than one precision, and those are separate Burn
/// backends and so separate Rust types. Erasing the parameter here keeps that
/// from turning into a `match` at every call site whose arms differ in nothing
/// but the type.
trait Forward {
    /// This unit's `[rows, seq, hidden]` states, read back to the host as f32
    /// whatever the graph ran in.
    fn hidden(&self, unit: &crate::batch::Unit) -> Result<Vec<f32>>;
}

impl<B: FusedOps> Forward for encoder::ModernBert<B> {
    fn hidden(&self, unit: &crate::batch::Unit) -> Result<Vec<f32>> {
        let device = Default::default();
        self.forward(unit.ids(), unit.mask(), unit.rows, unit.batch.seq, &device)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("cannot read hidden states back: {e:?}"))
    }
}

/// How a device wants the work shaped. Both fields are measurements, not
/// preferences; see the device module that sets them.
struct Shape {
    /// Attention scratch allowed per forward, in scores per head, divided by
    /// `seq^2` to get rows.
    budget: usize,
    /// Run the units across a thread pool rather than back to back.
    fan_out: bool,
    /// Rows a forward may hold whatever the budget allows.
    max_rows: usize,
}

/// A loaded model plus the shape its device wants its work in.
pub struct BurnEncoder {
    model: Box<dyn Forward + Send + Sync>,
    dim: usize,
    shape: Shape,
}

impl BurnEncoder {
    /// Run bucketed batches and reduce each row, in the caller's original order.
    ///
    /// Splitting and placement are [`crate::batch`]'s, the same ones
    /// [`crate::model::run_batches`] uses. What differs is the budget and the
    /// fan-out, and the second of those is not a detail: on burn-flex, one wide
    /// forward reaches 0.9 rows/s where the same work split one row at a time
    /// across the pool reaches 5.7. Burn's own intra-op parallelism does not
    /// substitute for it.
    pub fn run<T, F>(
        &self,
        batches: &[crate::batch::BatchInput],
        rows_total: usize,
        reduce: F,
    ) -> Result<Vec<T>>
    where
        T: Send,
        F: Fn(&[f32], &[i64], usize) -> Result<T> + Sync,
    {
        let units = crate::batch::split_units(batches, |seq| {
            (self.shape.budget / (seq * seq).max(1))
                .max(1)
                .min(self.shape.max_rows)
        });
        let run = |unit: &crate::batch::Unit| -> Result<Vec<(usize, T)>> {
            let hidden = self.model.hidden(unit)?;
            unit.reduce_rows(&hidden, self.dim, &reduce)
        };
        // rayon's *global* pool, where the candle path builds its own
        // physical-core one. burn-flex parallelises inside an operation on the
        // global pool, so nesting a second pool outside it would put two
        // schedulers on one machine with no shared work-stealing. Measured at
        // 64 rows the two are within noise of each other; this is the simpler
        // arrangement and the one the jig's numbers were taken with.
        let per_unit: Vec<Result<Vec<(usize, T)>>> = if self.shape.fan_out {
            units.par_iter().map(run).collect()
        } else {
            units.iter().map(run).collect()
        };
        crate::batch::place_rows(per_unit, rows_total)
    }
}
