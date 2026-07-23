//! The ModernBERT encoder.
//!
//! Lifted from candle-transformers 0.11.0 (`src/models/modernbert.rs`, MIT OR
//! Apache-2.0) and modified for kohagi. The original file has no dependency on
//! the rest of candle-transformers, so carrying this one file is lighter than
//! vendoring the whole crate to patch it — and it removes the `[patch.crates-io]`
//! that would otherwise be dropped on publish, taking the Metal speedups with
//! it.
//!
//! The changes are all Metal wins that leave f32 output unchanged: the fused
//! softmax kernel, SDPA with a view mask, a fused LayerNorm, and a per-backend
//! QKV layout that keeps q/k/v as views. See git history for the reasoning and
//! measurements. The upstream candle bugs these route around (Metal sdpa
//! ignoring a non-zero start offset, and its unenforced contiguity
//! precondition) are why the QKV layout is split per backend.
//!
//! ModernBERT: <https://arxiv.org/abs/2412.13663>.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{
    embedding, linear_no_bias, ops::softmax_last_dim, Embedding, LayerNorm, Linear, Module,
    VarBuilder,
};
use serde::Deserialize;

use core::f32;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, config: &Config, rope_theta: f64, dev: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let max_seq_len = config.max_position_embeddings;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// Rotate `[b, heads, seq, dim]`, the layout the fused QKV slices arrive in.
    fn rope_bhsd(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?,
        ))
    }

    /// Rotate `[b, seq, heads, dim]`, the layout the split projections produce.
    ///
    /// Same arithmetic as [`Self::rope_bhsd`] — verified bit-identical — but it
    /// takes the tensors as they come off the projection, so the transpose into
    /// attention's layout happens afterwards and stays a view.
    fn rope_thd(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            candle_nn::rotary_emb::rope_thd(q, &self.cos, &self.sin)?,
            candle_nn::rotary_emb::rope_thd(k, &self.cos, &self.sin)?,
        ))
    }
}

#[derive(Clone)]
enum Qkv {
    Fused(Linear),
    Split { q: Linear, k: Linear, v: Linear },
}

#[derive(Clone)]
struct ModernBertAttention {
    /// How the QKV projection is stored, which differs by backend.
    ///
    /// Metal wants it split. Fused, the three slices come out of one tensor at
    /// offsets h*d apart, and candle's Metal sdpa mishandles a non-zero start
    /// offset — silently, with error proportional to the offset. Giving each of
    /// q/k/v its own allocation at offset 0 lets sdpa read strided views
    /// correctly, which removes three per-layer copies and measured 1.43x end
    /// to end.
    ///
    /// The CPU wants it fused: it has no sdpa to hand views to, so it would
    /// materialize them anyway, and three narrow matmuls run slower there than
    /// one wide one.
    qkv: Qkv,
    proj: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl ModernBertAttention {
    fn load(vb: VarBuilder, config: &Config, rotary_emb: Arc<RotaryEmbedding>) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let qkv = linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?;
        let qkv = if vb.device().is_metal() {
            // Linear weights are [out, in]; the fused Wqkv concatenates q, k
            // and v along the output axis, so the split is by rows.
            let w = qkv.weight();
            let h = config.hidden_size;
            Qkv::Split {
                q: Linear::new(w.narrow(0, 0, h)?.contiguous()?, None),
                k: Linear::new(w.narrow(0, h, h)?.contiguous()?, None),
                v: Linear::new(w.narrow(0, 2 * h, h)?.contiguous()?, None),
            }
        } else {
            Qkv::Fused(qkv)
        };
        let proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"))?;

        Ok(Self {
            qkv,
            proj,
            num_attention_heads,
            attention_head_size,
            rotary_emb,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let xs = hidden_states.clone();
        let (b, seq_len, d) = xs.dims3()?;
        let heads = (
            b,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        );
        // Both arms end at [b, heads, seq, dim], but they get there differently
        // and rotate in different layouts to avoid materializing on the way.
        let (q, k, v) = match &self.qkv {
            Qkv::Split { q, k, v } => {
                let q = xs.apply(q)?.reshape(heads)?;
                let k = xs.apply(k)?.reshape(heads)?;
                let v = xs.apply(v)?.reshape(heads)?.transpose(1, 2)?;
                let (q, k) = self.rotary_emb.rope_thd(&q, &k)?;
                (q.transpose(1, 2)?, k.transpose(1, 2)?, v)
            }
            Qkv::Fused(qkv) => {
                let t = xs
                    .apply(qkv)?
                    .reshape((
                        b,
                        seq_len,
                        3,
                        self.num_attention_heads,
                        self.attention_head_size,
                    ))?
                    .permute((2, 0, 3, 1, 4))?;
                let (q, k) = self.rotary_emb.rope_bhsd(&t.get(0)?, &t.get(1)?)?;
                (q, k, t.get(2)?)
            }
        };

        let scale = (self.attention_head_size as f64).powf(-0.5);

        // On Metal, fuse the whole attention so the [b, h, s, s] score tensor is
        // never materialized. sdpa is Metal-only, so the CPU keeps the explicit
        // path.
        let xs = if q.device().is_metal() {
            let (mb, _, ms, mk) = attention_mask.dims4()?;
            // Clamp on the small [b, 1, s, s] mask, then widen to the head count
            // as a *view*: sdpa checks dims but reads the mask through strides,
            // so a stride-0 head axis satisfies it without materializing the
            // [b, h, s, s] tensor this fusion exists to avoid.
            //
            // The floor is finite because a fully padded query row is all -inf,
            // and softmax of that is NaN. The explicit path hides it — pooling
            // skips padded positions — but the fused kernel lets the NaN reach
            // the whole row.
            let mask = attention_mask.clamp(-60f32, 0f32)?.broadcast_as((
                mb,
                self.num_attention_heads,
                ms,
                mk,
            ))?;
            candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.0)?
        } else {
            // The CPU matmul cannot consume the transposed views, and there is
            // no sdpa to hand them to, so materialize here rather than paying
            // for it on both backends.
            let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
            let q = (q * scale)?;
            let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
            let att = att.broadcast_add(attention_mask)?;
            let att = softmax_last_dim(&att)?;
            att.matmul(&v)?
        };

        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        let xs = xs.apply(&self.proj)?;
        let xs = xs.reshape((b, seq_len, d))?;

        Ok(xs)
    }
}

/// ModernBERT's norms have no bias, and `layer_norm_no_bias` routes to
/// candle's generic multi-pass implementation. Handing it an explicit zero
/// bias instead selects the fused kernel, which measured 11x faster on Metal
/// (2.95 ms -> 0.26 ms over [2048, 512]) for an arithmetically identical
/// result.
fn layer_norm_fused(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = Tensor::zeros(size, weight.dtype(), weight.device())?;
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Clone)]
pub struct ModernBertMLP {
    // Wi is stored pre-split. A single Wi followed by chunk(2, last) leaves both
    // halves as strided views, and elementwise work on those runs 6-8x slower on
    // Metal than on contiguous memory — enough that the GeGLU costs more than
    // both GEMMs. Splitting the weight once at load makes each half its own
    // tensor, so gelu and the product stay contiguous.
    wi_gate: Linear,
    wi_up: Linear,
    wo: Linear,
}

impl ModernBertMLP {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let wi = linear_no_bias(
            config.hidden_size,
            config.intermediate_size * 2,
            vb.pp("Wi"),
        )?;
        // Linear weights are [out, in]; the chunk this replaces split the
        // output axis, so the same split applies to rows here.
        let w = wi.weight();
        let inter = config.intermediate_size;
        let wi_gate = Linear::new(w.narrow(0, 0, inter)?.contiguous()?, None);
        let wi_up = Linear::new(w.narrow(0, inter, inter)?.contiguous()?, None);
        let wo = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("Wo"))?;
        Ok(Self { wi_gate, wi_up, wo })
    }
}

impl Module for ModernBertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.apply(&self.wi_gate)?;
        let up = xs.apply(&self.wi_up)?;
        (gate.gelu_erf()? * up)?.apply(&self.wo) // GeGLU
    }
}

#[derive(Clone)]
pub struct ModernBertLayer {
    attn: ModernBertAttention,
    mlp: ModernBertMLP,
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    uses_local_attention: bool,
}

impl ModernBertLayer {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        uses_local_attention: bool,
    ) -> Result<Self> {
        let attn = ModernBertAttention::load(vb.pp("attn"), config, rotary_emb)?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config)?;
        let attn_norm = layer_norm_fused(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attn_norm"),
        )
        .ok();
        let mlp_norm =
            layer_norm_fused(config.hidden_size, config.layer_norm_eps, vb.pp("mlp_norm"))?;
        Ok(Self {
            attn,
            mlp,
            attn_norm,
            mlp_norm,
            uses_local_attention,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        global_attention_mask: &Tensor,
        local_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }

        let attention_mask = if self.uses_local_attention {
            &global_attention_mask.broadcast_add(local_attention_mask)?
        } else {
            global_attention_mask
        };
        let xs = self.attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }
}

fn prepare_4d_attention_mask(
    mask: &Tensor,
    dtype: DType,
    tgt_len: Option<usize>,
) -> Result<Tensor> {
    let bsz = mask.dim(0)?;
    let src_len = mask.dim(1)?;
    let tgt_len = tgt_len.unwrap_or(src_len);

    let expanded_mask = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, tgt_len, src_len))?
        .to_dtype(dtype)?;

    let inverted_mask = (1.0 - expanded_mask)?;

    (inverted_mask * f32::MIN as f64)?.to_dtype(dtype)
}

// Attention mask caused by the sliding window
fn get_local_attention_mask(
    seq_len: usize,
    max_distance: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if (j as i32 - i as i32).abs() > max_distance as i32 {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}

// ModernBERT backbone
#[derive(Clone)]
pub struct ModernBert {
    word_embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm,
    local_attention_size: usize,
}

impl ModernBert {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embeddings.tok_embeddings"),
        )?;
        let norm = layer_norm_fused(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("model.embeddings.norm"),
        )?;
        let global_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.global_rope_theta,
            vb.device(),
        )?);
        let local_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.local_rope_theta,
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            let layer_uses_local_attention = layer_id % config.global_attn_every_n_layers != 0;
            layers.push(ModernBertLayer::load(
                vb.pp(format!("model.layers.{layer_id}")),
                config,
                if layer_uses_local_attention {
                    local_rotary_emb.clone()
                } else {
                    global_rotary_emb.clone()
                },
                layer_uses_local_attention,
            )?);
        }

        let final_norm = layer_norm_fused(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("model.final_norm"),
        )?;

        Ok(Self {
            word_embeddings,
            norm,
            layers,
            final_norm,
            local_attention_size: config.local_attention,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let seq_len = xs.shape().dims()[1];
        let global_attention_mask =
            prepare_4d_attention_mask(mask, DType::F32, None)?.to_device(xs.device())?;
        let local_attention_mask =
            get_local_attention_mask(seq_len, self.local_attention_size / 2, xs.device())?;
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, &global_attention_mask, &local_attention_mask)?;
        }
        let xs = xs.apply(&self.final_norm)?;
        Ok(xs)
    }
}
