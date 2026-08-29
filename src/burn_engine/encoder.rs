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
    window: Tensor<B, 4>,
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
            window: window_mask::<B>(seq, config.local_attention, device),
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

/// Which positions a sliding-window layer may not reach, `[1, 1, seq, seq]`.
///
/// A constant for a given length: the window depends on the position pair
/// alone, and only the padding part of a mask depends on the input.
fn window_mask<B: Backend>(seq: usize, window: usize, device: &B::Device) -> Tensor<B, 4> {
    let reach = window / 2;
    let mut m = vec![0f32; seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            if (q as isize - k as isize).unsigned_abs() > reach {
                m[q * seq + k] = BLOCKED;
            }
        }
    }
    Tensor::from_data(TensorData::new(m, [1, 1, seq, seq]), device)
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
        let q = rope(part(0), cos.clone(), sin.clone());
        let k = rope(part(1), cos, sin);
        let v = part(2);

        let scores = q.matmul(k.swap_dims(2, 3));
        let dtype = scores.dtype();
        let scale = 1.0 / (head_dim as f64).sqrt();
        // Under `half` the softmax and the mask it consumes run in f32; see the
        // module's precision note for why this reduction in particular.
        let (scores, pad, win) = if self.half {
            (
                scores.cast(FloatDType::F32) * scale,
                padding.clone().cast(FloatDType::F32),
                tables.window.clone().cast(FloatDType::F32),
            )
        } else {
            (scores * scale, padding.clone(), tables.window.clone())
        };
        let mut scores = scores + pad;
        if !layer.global {
            scores = scores + win;
        }
        let probs = activation::softmax(scores, 3);
        let probs = if self.half { probs.cast(dtype) } else { probs };

        let context = probs.matmul(v).swap_dims(1, 2).reshape([rows, seq, hidden]);
        linear(context, layer.wo.clone())
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
