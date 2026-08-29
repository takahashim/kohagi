//! ModernBERT's forward pass in Burn, for the Vulkan backend.
//!
//! A second implementation of the architecture `crate::encoder` already carries
//! in candle, for the same reason `crate::coreml_export::modernbert` is a third:
//! the tensor types do not meet. What keeps them from drifting is not a shared
//! forward but a shared *meaning* — [`crate::encoder::Activation`] and
//! [`crate::encoder::Config`] are the same types the candle path parses, so a
//! checkpoint cannot be read two ways, and `tests/vulkan_matches_cpu.rs` holds
//! the outputs to each other.
//!
//! Structural details that are easy to get subtly wrong, all mirroring
//! `crate::encoder`:
//!
//! - Layer 0 ships no `attn_norm`; the reference model applies identity there.
//! - Every LayerNorm is weight-only (`norm_bias = false`) and every projection
//!   is bias-free (`attention_bias` / `mlp_bias = false`).
//! - Layer `i` attends globally iff `i % global_attn_every_n_layers == 0`, and
//!   global and local layers use *different* RoPE bases.
//! - The sliding window reaches `local_attention / 2` either side.
//! - `Wi` concatenates gate and up along the output axis; the *first* half is
//!   the one the activation is applied to.
//!
//! ## Precision
//!
//! [`Precision::F16`] does not simply run the whole graph in f16: that measured
//! `1 - cosine` of 3.5e-2 against the CPU on real ruri-v3-130m weights, which is
//! not an embedding anyone should index. What is wrong there is the reductions,
//! not the matrix multiplies — LayerNorm's variance and softmax accumulate over
//! the whole last axis, and f16 runs out of mantissa doing it. Casting only
//! those two to f32 costs 16% of the throughput and buys back four orders of
//! magnitude (worst 8.9e-6 over 64 texts, median 3.4e-7).
//!
//! Note this is the opposite arrangement to the CPU's bf16 path, which keeps
//! the *activations* in f32 and lowers only the projections. That recipe was
//! measured here too and is 1.4x slower than this one: on a GPU sharing system
//! memory, round-tripping every activation through f32 costs more bandwidth
//! than the reductions cost accuracy.

use burn::tensor::{activation, backend::Backend, FloatDType, Int, Tensor, TensorData};

use super::FusedOps;
use crate::encoder::Config;

/// Where a position may not attend.
///
/// Finite rather than `f32::MIN`, for two reasons: a fully padded query row
/// would otherwise be all `-inf` and softmax of that is NaN, and anything past
/// about 65504 is already `inf` once the tensor is f16. Pooling drops padded
/// rows either way, so the only thing this constant decides is whether the
/// arithmetic stays finite while it gets there.
const BLOCKED: f32 = -1.0e4;

/// Queries per block when a sliding-window layer walks its band.
///
/// A block reads `BAND_BLOCK + 2·(window/2)` keys to serve `BAND_BLOCK`
/// queries, so narrow blocks read fewer keys per query and pay more per-call
/// overhead. Burn's operations are coarser than the candle path's hand-written
/// kernels, so this was measured again rather than assumed — and landed on the
/// same 32. At 2048 tokens, 8 rows: 7.82 s at 32, 7.80 at 64, 8.06 at 128, and
/// 16.26 s with no banding at all. The first two are a tie within the spread,
/// which is why this takes the one the candle path already uses.
///
/// `crate::attention::banding_pays` decides whether any of this applies, and
/// below about 514 tokens it does not — the window already spans too much of
/// the row for narrowing to pay. The default `--max-seq-length 512` therefore
/// never reaches this constant.
const BAND_BLOCK: usize = 32;

/// How `Wi` is stored, which differs by device — see [`FusedOps::SPLIT_WI`].
///
/// The same choice `crate::encoder` makes for the candle backends, made for the
/// same reason and measured again here.
pub(super) enum Wi<B: Backend> {
    /// One `[hidden, 2 * inter]` matrix; the halves come out as views.
    Wide(Tensor<B, 2>),
    /// Two `[hidden, inter]` matrices, so both halves are contiguous.
    Split {
        gate: Tensor<B, 2>,
        up: Tensor<B, 2>,
    },
}

/// One layer's parameters, already on the device.
pub(super) struct LayerWeights<B: Backend> {
    /// `None` for layer 0, which the reference model leaves as identity.
    pub attn_norm: Option<Tensor<B, 1>>,
    pub wqkv: Tensor<B, 2>,
    pub wo: Tensor<B, 2>,
    pub mlp_norm: Tensor<B, 1>,
    pub wi: Wi<B>,
    pub mlp_wo: Tensor<B, 2>,
    pub global: bool,
}

/// Everything a padded length decides, and nothing that depends on the input.
///
/// A bucket is split into as many forwards as the memory budget allows, so
/// these would otherwise be regenerated on the host and re-uploaded once per
/// unit — the window mask alone is `seq * seq` floats, a megabyte at seq 512.
/// [`ModernBert::tables`] keeps the last set and rebuilds only when the padded
/// length changes, which is once per bucket.
pub(super) struct Tables<B: Backend> {
    seq: usize,
    cos_global: Tensor<B, 4>,
    sin_global: Tensor<B, 4>,
    cos_local: Tensor<B, 4>,
    sin_local: Tensor<B, 4>,
    window: WindowMask<B>,
}

impl<B: Backend> Clone for WindowMask<B> {
    fn clone(&self) -> Self {
        match self {
            WindowMask::Table { table, reach } => WindowMask::Table {
                table: table.clone(),
                reach: *reach,
            },
            WindowMask::Whole(m) => WindowMask::Whole(m.clone()),
        }
    }
}

impl<B: Backend> Clone for Tables<B> {
    /// Cheap: a Burn tensor is a handle, so this shares the uploaded buffers
    /// rather than copying them.
    fn clone(&self) -> Self {
        Tables {
            seq: self.seq,
            cos_global: self.cos_global.clone(),
            sin_global: self.sin_global.clone(),
            cos_local: self.cos_local.clone(),
            sin_local: self.sin_local.clone(),
            window: self.window.clone(),
        }
    }
}

impl<B: Backend> Tables<B> {
    fn build(config: &Config, seq: usize, device: &B::Device) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let (cos_global, sin_global) =
            rope_tables::<B>(seq, head_dim, config.global_rope_theta as f32, device);
        let (cos_local, sin_local) =
            rope_tables::<B>(seq, head_dim, config.local_rope_theta as f32, device);
        Tables {
            seq,
            cos_global,
            sin_global,
            cos_local,
            sin_local,
            // Half, because `crate::attention` measures a window as its reach.
            window: WindowMask::build(seq, config.local_attention / 2, BAND_BLOCK, device),
        }
    }

    /// The RoPE base this layer wants. Global and local layers use different
    /// thetas, which is the one thing the two kinds of layer do not share.
    fn rope(&self, global: bool) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if global {
            (self.cos_global.clone(), self.sin_global.clone())
        } else {
            (self.cos_local.clone(), self.sin_local.clone())
        }
    }
}

/// The whole model, plus the last padded length's tables.
pub(super) struct ModernBert<B: Backend> {
    pub tok_embeddings: Tensor<B, 2>,
    pub embeddings_norm: Tensor<B, 1>,
    pub layers: Vec<LayerWeights<B>>,
    pub final_norm: Tensor<B, 1>,
    pub config: Config,
    pub half: bool,
    pub budget: usize,
    /// Rebuilt only when the padded length changes; see [`Tables`]. A mutex
    /// rather than a `RefCell` because the loaded model is shared behind `&`
    /// and must stay `Sync`, not because anything here runs concurrently.
    /// Starts empty — the first forward decides the length.
    pub tables: std::sync::Mutex<Option<Tables<B>>>,
}

/// RoPE's cosine and sine tables, `[1, 1, seq, head_dim]`, with the angles
/// duplicated across the two halves of the last axis — the layout
/// [`super::rope_composed`]'s `x1`/`x2` split expects.
fn rope_tables<B: Backend>(
    seq: usize,
    head_dim: usize,
    theta: f32,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = head_dim / 2;
    let (mut cos, mut sin) = (vec![0f32; seq * head_dim], vec![0f32; seq * head_dim]);
    for pos in 0..seq {
        for i in 0..half {
            let angle = pos as f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
            cos[pos * head_dim + i] = angle.cos();
            cos[pos * head_dim + i + half] = angle.cos();
            sin[pos * head_dim + i] = angle.sin();
            sin[pos * head_dim + i + half] = angle.sin();
        }
    }
    (
        Tensor::from_data(TensorData::new(cos, [1, 1, seq, head_dim]), device),
        Tensor::from_data(TensorData::new(sin, [1, 1, seq, head_dim]), device),
    )
}

/// An additive sliding-window mask, `[1, 1, queries, keys]`.
///
/// `offset` is how far the block's first query sits past its first key, which
/// is what places the band inside the rectangle.
fn window_mask<B: Backend>(
    queries: usize,
    offset: usize,
    reach: usize,
    keys: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut m = vec![0f32; queries * keys];
    for i in 0..queries {
        for j in 0..keys {
            if (i + offset).abs_diff(j) > reach {
                m[i * keys + j] = BLOCKED;
            }
        }
    }
    Tensor::from_data(TensorData::new(m, [1, 1, queries, keys]), device)
}

/// Where a banded block's slice of the sliding-window mask comes from.
///
/// A banded layer's blocks all reach the same way — `reach` keys either side of
/// their own queries — so the mask depends on the offset between a block's
/// first query and its first key, not on where in the sequence the block sits.
/// One table therefore serves every block, the ends included: they take a
/// shifted or shortened view of it.
///
/// That is the difference between 20 KB and 256 MiB. Built whole, an 8192-token
/// layer's mask is `seq * seq` floats, and the padding has to be added into a
/// copy of it; `crate::encoder` shares one table for the same reason.
enum WindowMask<B: Backend> {
    /// `[1, 1, width, width + 2 * reach]`, viewed by every banded block.
    Table { table: Tensor<B, 4>, reach: usize },
    /// The whole `[1, 1, seq, seq]`, for a layer that is windowed but not
    /// banded. There is one block in that case, so there is nothing to share.
    Whole(Tensor<B, 4>),
}

impl<B: Backend> WindowMask<B> {
    /// `reach` is how far either side a query sees — half `local_attention`,
    /// which is the sense `crate::attention` gives the word throughout.
    fn build(seq: usize, reach: usize, band: usize, device: &B::Device) -> Self {
        if crate::attention::banding_pays(seq, reach) {
            WindowMask::Table {
                table: window_mask::<B>(band, reach, reach, band + 2 * reach, device),
                reach,
            }
        } else {
            WindowMask::Whole(window_mask::<B>(seq, 0, reach, seq, device))
        }
    }

    /// This block's view of the mask.
    fn of(&self, block: &crate::attention::Block) -> Tensor<B, 4> {
        match self {
            // Entry `(i, j)` of the block is `|i - j + offset| <= reach`, and of
            // the table `|i - j + reach| <= reach`, so the block's row of the
            // table starts `reach - offset` columns in.
            WindowMask::Table { table, reach } => {
                let offset = block.q0 - block.k0;
                table
                    .clone()
                    .narrow(2, 0, block.queries)
                    .narrow(3, reach - offset, block.keys)
            }
            WindowMask::Whole(m) => m
                .clone()
                .narrow(2, block.q0, block.queries)
                .narrow(3, block.k0, block.keys),
        }
    }
}

/// The additive padding mask, `[rows, 1, 1, seq]`.
///
/// Whether a position is padding is a property of the key being attended to,
/// not of the query attending to it, so the query axis stays 1 — broadcasting
/// covers it without ever paying for the `seq^2` this shape avoids.
fn padding_mask<B: Backend>(
    mask: &[i64],
    rows: usize,
    seq: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let values: Vec<f32> = mask
        .iter()
        .map(|&m| if m == 0 { BLOCKED } else { 0.0 })
        .collect();
    Tensor::from_data(TensorData::new(values, [rows, 1, 1, seq]), device)
}

fn rope<B: FusedOps>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    B::rope(x, cos, sin)
}

/// Weight-only LayerNorm over the last axis.
///
/// Under `half`, the reduction runs in f32 and the result comes back in the
/// tensor's own dtype — see this module's precision note.
fn layer_norm<B: Backend>(
    x: Tensor<B, 3>,
    gamma: Tensor<B, 1>,
    eps: f64,
    half: bool,
) -> Tensor<B, 3> {
    let dtype = x.dtype();
    let (x, gamma) = if half {
        (x.cast(FloatDType::F32), gamma.cast(FloatDType::F32))
    } else {
        (x, gamma)
    };
    let hidden = x.dims()[2];
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    let normed = centered / (var + eps).sqrt() * gamma.reshape([1, 1, hidden]);
    if half {
        normed.cast(dtype)
    } else {
        normed
    }
}

/// `[b, s, in] @ [in, out]` as one 2-D matmul, which is the shape the GEMM
/// wants; the weights were transposed once at load for the same reason.
fn linear<B: Backend>(x: Tensor<B, 3>, w: Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, seq, input] = x.dims();
    let output = w.dims()[1];
    x.reshape([batch * seq, input])
        .matmul(w)
        .reshape([batch, seq, output])
}

impl<B: FusedOps> ModernBert<B> {
    /// This padded length's tables, built if the last forward used another one.
    fn tables(&self, seq: usize, device: &B::Device) -> Tables<B> {
        let mut slot = self.tables.lock().expect("tables mutex poisoned");
        match slot.as_ref() {
            Some(tables) if tables.seq == seq => tables.clone(),
            _ => {
                let built = Tables::build(&self.config, seq, device);
                *slot = Some(built.clone());
                built
            }
        }
    }

    /// The token embeddings, normalized — the block stack's input.
    fn embed(&self, ids: &[i64], rows: usize, seq: usize, device: &B::Device) -> Tensor<B, 3> {
        let index = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                ids.iter().map(|&i| i as i32).collect::<Vec<_>>(),
                [ids.len()],
            ),
            device,
        );
        let embedded = self.tok_embeddings.clone().select(0, index).reshape([
            rows,
            seq,
            self.config.hidden_size,
        ]);
        layer_norm(
            embedded,
            self.embeddings_norm.clone(),
            self.config.layer_norm_eps,
            self.half,
        )
    }

    /// Operations 1-16: the fused QKV, RoPE, masked attention and the output
    /// projection. `normed` is the pre-normalized input; the residual is the
    /// caller's.
    fn attention(
        &self,
        layer: &LayerWeights<B>,
        normed: Tensor<B, 3>,
        tables: &Tables<B>,
        padding: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let head_dim = hidden / heads;
        let [rows, seq, _] = normed.dims();

        let qkv = linear(normed, layer.wqkv.clone());
        // Linear weights are [out, in] and the fused Wqkv concatenates q, k and
        // v along that output axis, so after the transpose at load the three
        // live at column offsets 0, h and 2h.
        let part = |i: usize| {
            qkv.clone()
                .narrow(2, i * hidden, hidden)
                .reshape([rows, seq, heads, head_dim])
                .swap_dims(1, 2)
        };
        let (cos, sin) = tables.rope(layer.global);
        let scale = 1.0 / (head_dim as f64).sqrt();
        let q = rope(part(0), cos.clone(), sin.clone()) * scale;
        let k = rope(part(1), cos, sin);
        let v = part(2);

        // A sliding-window layer scores far less than the whole matrix: the keys
        // outside the window are masked shut and contribute an exact zero. The
        // geometry is `crate::attention`'s, the same one the candle path walks,
        // so the two engines cannot disagree about which keys a block reads.
        let window = (!layer.global).then_some(cfg.local_attention / 2);
        // Past the point where one row's scores exceed the budget, the queries
        // are split instead — the same handover `crate::encoder` makes, and the
        // reason a long sequence does not materialise `[heads, seq, seq]`.
        let tile = (self.budget / seq).max(1);
        let plan = crate::attention::plan(seq, window, tile, BAND_BLOCK);

        // One block's additive mask, in the block's own shape: the padding for
        // the keys it reads, plus the window where the layer has one. Building
        // it here rather than once per forward is what keeps a banded layer from
        // holding `[rows, seq, seq]` — the summed form has no shape to share.
        let mask_for = |b: &crate::attention::Block| {
            let pad = padding.clone().narrow(3, b.k0, b.keys);
            if layer.global {
                pad
            } else {
                pad + tables.window.of(b)
            }
        };

        let context = if plan.blocks.len() == 1 {
            self.attend(q, k, v, &mask_for(&plan.blocks[0]))
        } else {
            let parts: Vec<Tensor<B, 4>> = plan
                .blocks
                .iter()
                .map(|b| {
                    self.attend(
                        q.clone().narrow(2, b.q0, b.queries),
                        k.clone().narrow(2, b.k0, b.keys),
                        v.clone().narrow(2, b.k0, b.keys),
                        &mask_for(b),
                    )
                })
                .collect();
            Tensor::cat(parts, 2)
        };
        let context = context.swap_dims(1, 2).reshape([rows, seq, hidden]);
        linear(context, layer.wo.clone())
    }

    /// One block of queries against the keys it may reach.
    ///
    /// `mask` is additive and already carries everything that masks this block:
    /// padding, and the window where there is one. Combining them costs one pass
    /// over `[rows, 1, seq, seq]` per forward instead of a second broadcast add
    /// over `[rows, heads, seq, seq]` in each of the twelve sliding-window
    /// layers.
    fn attend(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        // Scaling the queries rather than the scores: `q` is
        // `[rows, heads, seq, head_dim]` and the scores are
        // `[rows, heads, seq, seq]`, so at 460 tokens this is one pass over a
        // seventh of the elements.
        let scores = q.matmul(k.swap_dims(2, 3));
        let dtype = scores.dtype();
        // Under `half` the softmax and the mask it consumes run in f32; see the
        // module's precision note for why this reduction in particular.
        let scores = if self.half {
            scores.cast(FloatDType::F32) + mask.clone().cast(FloatDType::F32)
        } else {
            scores + mask.clone()
        };
        let probs = activation::softmax(scores, 3);
        let probs = if self.half { probs.cast(dtype) } else { probs };
        probs.matmul(v)
    }

    /// Operations 17-23: the gated feed-forward. `Wi` concatenates gate and up
    /// along the output axis, and the activation applies to the *first* half.
    fn feed_forward(&self, layer: &LayerWeights<B>, normed: Tensor<B, 3>) -> Tensor<B, 3> {
        let inter = self.config.intermediate_size;
        let (gate, up) = match &layer.wi {
            Wi::Split { gate, up } => (
                linear(normed.clone(), gate.clone()),
                linear(normed, up.clone()),
            ),
            Wi::Wide(wi) => {
                let wide = linear(normed, wi.clone());
                (
                    wide.clone().narrow(2, 0, inter),
                    wide.narrow(2, inter, inter),
                )
            }
        };
        let gated = B::geglu(gate, up, self.config.activation);
        linear(gated, layer.mlp_wo.clone())
    }

    /// One transformer block: pre-norm, attention, residual, pre-norm, gated
    /// feed-forward, residual.
    fn block(
        &self,
        layer: &LayerWeights<B>,
        x: Tensor<B, 3>,
        tables: &Tables<B>,
        padding: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let eps = self.config.layer_norm_eps;
        let residual = x.clone();
        // Layer 0 ships no `attn_norm`; the reference model applies identity.
        let normed = match &layer.attn_norm {
            Some(gamma) => layer_norm(x, gamma.clone(), eps, self.half),
            None => x,
        };
        let attended = residual + self.attention(layer, normed, tables, padding);

        let normed = layer_norm(attended.clone(), layer.mlp_norm.clone(), eps, self.half);
        attended + self.feed_forward(layer, normed)
    }

    /// Hidden states for one padded forward: `ids` and `mask` are row-major
    /// `[rows, seq]`, and the result is `[rows, seq, hidden]` in f32.
    pub(super) fn forward(
        &self,
        ids: &[i64],
        mask: &[i64],
        rows: usize,
        seq: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let tables = self.tables(seq, device);
        let padding = padding_mask::<B>(mask, rows, seq, device);

        let mut x = self.embed(ids, rows, seq, device);
        for layer in &self.layers {
            x = self.block(layer, x, &tables, &padding);
        }
        // Read back as f32 whatever the graph ran in: `embed_row` pools and
        // normalizes in f32, the same as every other backend.
        layer_norm(
            x,
            self.final_norm.clone(),
            self.config.layer_norm_eps,
            self.half,
        )
        .cast(FloatDType::F32)
    }
}
