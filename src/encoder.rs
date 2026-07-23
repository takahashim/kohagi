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

/// A ModernBERT `config.json`, in the two spellings the field names come in.
///
/// Deserialized through [`RawConfig`] so the LayerNorm epsilon can arrive as
/// `norm_eps` (HF's `ModernbertConfig`), `layer_norm_eps` (older
/// sentence-transformers checkpoints), both (ruri ships both), or neither.
#[derive(Debug, Clone, PartialEq)]
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

#[derive(Deserialize)]
struct RawConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    // A serde `alias` would reject a config carrying both names as a duplicate
    // field, and ruri-v3 carries both, so they are two optional fields merged
    // below rather than one aliased one.
    layer_norm_eps: Option<f64>,
    norm_eps: Option<f64>,
    pad_token_id: u32,
    global_attn_every_n_layers: usize,
    global_rope_theta: f64,
    local_attention: usize,
    local_rope_theta: f64,
}

impl<'de> Deserialize<'de> for Config {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let r = RawConfig::deserialize(d)?;
        Ok(Config {
            vocab_size: r.vocab_size,
            hidden_size: r.hidden_size,
            num_hidden_layers: r.num_hidden_layers,
            num_attention_heads: r.num_attention_heads,
            intermediate_size: r.intermediate_size,
            max_position_embeddings: r.max_position_embeddings,
            // Prefer the explicit `layer_norm_eps`; fall back to `norm_eps`,
            // then to HF's default. The two agree wherever both appear.
            layer_norm_eps: r.layer_norm_eps.or(r.norm_eps).unwrap_or(1e-5),
            pad_token_id: r.pad_token_id,
            global_attn_every_n_layers: r.global_attn_every_n_layers,
            global_rope_theta: r.global_rope_theta,
            local_attention: r.local_attention,
            local_rope_theta: r.local_rope_theta,
        })
    }
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

/// How the `Wi` projection is stored, which differs by backend — see
/// [`ModernBertMLP::load`].
#[derive(Clone)]
enum Wi {
    Wide(Linear),
    Split { gate: Linear, up: Linear },
}

#[derive(Clone)]
pub struct ModernBertMLP {
    wi: Wi,
    inter: usize,
    wo: Linear,
}

impl ModernBertMLP {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let wi = linear_no_bias(
            config.hidden_size,
            config.intermediate_size * 2,
            vb.pp("Wi"),
        )?;
        let inter = config.intermediate_size;
        let wi = if vb.device().is_metal() {
            // Metal keeps Wi wide: one matmul, and the fused kernel reads both
            // halves of [tokens, 2*inter] straight out. Splitting would add a
            // second matmul and a chunk copy for no gain there.
            Wi::Wide(wi)
        } else {
            // Linear weights are [out, in]; the fused Wi concatenates gate and
            // up along the output axis, so the split is by rows. The CPU has no
            // fused kernel and its elementwise path is slower on the strided
            // views a chunk would leave, so it takes two contiguous matmuls.
            let w = wi.weight();
            Wi::Split {
                gate: Linear::new(w.narrow(0, 0, inter)?.contiguous()?, None),
                up: Linear::new(w.narrow(0, inter, inter)?.contiguous()?, None),
            }
        };
        let wo = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("Wo"))?;
        Ok(Self { wi, inter, wo })
    }
}

impl Module for ModernBertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = match &self.wi {
            Wi::Wide(wi) => crate::fused::geglu(&xs.apply(wi)?, self.inter)?,
            Wi::Split { gate, up } => (xs.apply(gate)?.gelu_erf()? * xs.apply(up)?)?,
        };
        gated.apply(&self.wo)
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

        // `local_attention_mask` already holds `global + local` (combined once
        // in ModernBert::forward, since every local layer would otherwise redo
        // the same broadcast_add — 12 of 13 of them redundant).
        let attention_mask = if self.uses_local_attention {
            local_attention_mask
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
        // Combined once here rather than in each of the 13 local layers: the
        // sliding-window mask and the padding mask are identical across layers,
        // so their sum is too.
        let sliding =
            get_local_attention_mask(seq_len, self.local_attention_size / 2, xs.device())?;
        let local_attention_mask = global_attention_mask.broadcast_add(&sliding)?;
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, &global_attention_mask, &local_attention_mask)?;
        }
        let xs = xs.apply(&self.final_norm)?;
        Ok(xs)
    }
}

#[cfg(test)]
mod config_tests {
    use super::Config;

    /// A ModernBERT config with everything except the eps field, which the
    /// three cases below vary. Values are arbitrary but well-formed.
    fn config_json(eps_line: &str) -> String {
        format!(
            r#"{{
                "vocab_size": 50004,
                "hidden_size": 768,
                "num_hidden_layers": 22,
                "num_attention_heads": 12,
                "intermediate_size": 1152,
                "max_position_embeddings": 8192,
                {eps_line}
                "pad_token_id": 0,
                "global_attn_every_n_layers": 3,
                "global_rope_theta": 160000.0,
                "local_attention": 128,
                "local_rope_theta": 10000.0
            }}"#
        )
    }

    /// The older sentence-transformers spelling.
    #[test]
    fn accepts_layer_norm_eps() {
        let c: Config = serde_json::from_str(&config_json(r#""layer_norm_eps": 2e-5,"#)).unwrap();
        assert_eq!(c.layer_norm_eps, 2e-5);
    }

    /// ruri-v3 ships both spellings; an `alias` would have rejected that as a
    /// duplicate field. They agree, and `layer_norm_eps` wins by construction.
    #[test]
    fn accepts_both_spellings() {
        let c: Config = serde_json::from_str(&config_json(
            r#""layer_norm_eps": 1e-5, "norm_eps": 1e-5,"#,
        ))
        .unwrap();
        assert_eq!(c.layer_norm_eps, 1e-5);
    }

    /// HF `ModernbertConfig`'s own name — e.g. CodeSearch-ModernBERT-Crow-Plus,
    /// which kohagi rejected before this alias.
    #[test]
    fn accepts_norm_eps() {
        let c: Config = serde_json::from_str(&config_json(r#""norm_eps": 3e-5,"#)).unwrap();
        assert_eq!(c.layer_norm_eps, 3e-5);
    }

    /// Neither present: fall back to HF's default rather than failing to parse.
    #[test]
    fn defaults_when_absent() {
        let c: Config = serde_json::from_str(&config_json("")).unwrap();
        assert_eq!(c.layer_norm_eps, 1e-5);
    }
}
