//! Mixed-precision ModernBERT forward.
//!
//! Structurally a port of candle-transformers' `modernbert.rs`, with the four
//! projection `Linear`s — attention `Wqkv` and `Wo`, MLP `Wi` and `Wo` — run
//! through the bf16 kernel in [`super::gemm`]. Everything else (embeddings,
//! LayerNorm, rotary embeddings, attention scores, softmax, GeGLU, residuals)
//! stays f32 in candle.
//!
//! That split mirrors what `torch.autocast(bf16)` does, and it is why accuracy
//! holds up: the reduced-precision inputs only affect the projections, and
//! mean pooling plus L2 normalization average away most of the per-element
//! noise. End to end the embeddings land at cosine ≈ 0.99999 against the f32
//! path — well inside the noise floor for retrieval ranking.
//!
//! Since only the projections change, this is *not* a general speedup: it wins
//! on short inputs, where the projections dominate, and fades on long ones,
//! where f32 attention (quadratic in sequence length) takes over.

use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, layer_norm_no_bias, ops::softmax, Embedding, LayerNorm, VarBuilder};
use candle_transformers::models::modernbert::Config;

use super::gemm::Bf16Linear;

/// Apply a bf16 `Linear` to a tensor whose last dimension is the Linear's
/// input width, flattening the leading dimensions into GEMM rows.
///
/// Each call copies the tensor out to a `Vec<f32>` and the result back.
/// Profiling puts that at ~3% of the forward — the GEMM dominates — so it is
/// not worth the complexity of borrowing candle's storage directly.
fn apply(lin: &Bf16Linear, x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    debug_assert_eq!(*dims.last().unwrap(), lin.k);
    let m: usize = dims[..dims.len() - 1].iter().product();
    let xv = x.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let y = lin.forward(&xv, m);
    let mut out_dims = dims;
    *out_dims.last_mut().unwrap() = lin.n;
    Tensor::from_vec(y, out_dims, x.device()).map_err(Into::into)
}

/// Read a `Linear`'s `[out, in]` weight and pack it for the bf16 kernel.
fn load_linear(vb: &VarBuilder, in_features: usize, out_features: usize) -> Result<Bf16Linear> {
    let w = vb.get((out_features, in_features), "weight")?;
    let wv = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(Bf16Linear::new(&wv, out_features, in_features))
}

/// Rotary position embeddings — f32, and identical to candle's.
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, rope_theta: f64, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let max_seq_len = cfg.max_position_embeddings;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self { sin: freqs.sin()?, cos: freqs.cos()? })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?;
        Ok((q, k))
    }
}

struct Attention {
    qkv: Bf16Linear,
    proj: Bf16Linear,
    num_heads: usize,
    head_size: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &Config, rotary: Arc<RotaryEmbedding>) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            qkv: load_linear(&vb.pp("Wqkv"), h, h * 3)?,
            proj: load_linear(&vb.pp("Wo"), h, h)?,
            num_heads: cfg.num_attention_heads,
            head_size: h / cfg.num_attention_heads,
            rotary,
        })
    }

    fn forward(&self, xs: &Tensor, attn_mask: &Tensor) -> Result<Tensor> {
        let (b, seq, d) = xs.dims3()?;
        let qkv = apply(&self.qkv, xs)?
            .reshape((b, seq, 3, self.num_heads, self.head_size))?
            .permute((2, 0, 3, 1, 4))?;
        let (q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
        let (q, k) = self.rotary.apply(&q, &k)?;

        // Scores, mask, and softmax stay f32 — the precision-sensitive part.
        let q = (q * (self.head_size as f64).powf(-0.5))?;
        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let att = softmax(&att.broadcast_add(attn_mask)?, D::Minus1)?;
        let xs = att.matmul(&v)?.transpose(1, 2)?.reshape((b, seq, d))?;
        apply(&self.proj, &xs)
    }
}

struct Mlp {
    wi: Bf16Linear,
    wo: Bf16Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            wi: load_linear(&vb.pp("Wi"), cfg.hidden_size, cfg.intermediate_size * 2)?,
            wo: load_linear(&vb.pp("Wo"), cfg.intermediate_size, cfg.hidden_size)?,
        })
    }

    /// GeGLU: `Wi` produces two halves, one gated by the other's GELU.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = apply(&self.wi, xs)?;
        let chunks = xs.chunk(2, D::Minus1)?;
        let gated = (chunks[0].gelu_erf()? * &chunks[1])?;
        apply(&self.wo, &gated)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    /// Absent on the first layer, which reuses the embedding norm.
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    /// Whether this layer attends only within a sliding window.
    uses_local: bool,
}

impl Layer {
    fn load(
        vb: VarBuilder,
        cfg: &Config,
        rotary: Arc<RotaryEmbedding>,
        uses_local: bool,
    ) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(vb.pp("attn"), cfg, rotary)?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
            attn_norm: layer_norm_no_bias(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("attn_norm"))
                .ok(),
            mlp_norm: layer_norm_no_bias(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("mlp_norm"))?,
            uses_local,
        })
    }

    fn forward(&self, xs: &Tensor, global_mask: &Tensor, local_mask: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let mut h = xs.clone();
        if let Some(norm) = &self.attn_norm {
            h = h.apply(norm)?;
        }
        let mask = if self.uses_local {
            global_mask.broadcast_add(local_mask)?
        } else {
            global_mask.clone()
        };
        let xs = (residual + self.attn.forward(&h, &mask)?)?;
        let mlp_out = self.mlp.forward(&xs.apply(&self.mlp_norm)?)?;
        Ok((xs + mlp_out)?)
    }
}

/// Turn a `[batch, seq]` padding mask into additive `[batch, 1, seq, seq]`
/// attention bias (0 to keep, a large negative to drop).
fn padding_bias(mask: &Tensor) -> Result<Tensor> {
    let (bsz, src) = mask.dims2()?;
    let expanded = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, src, src))?
        .to_dtype(DType::F32)?;
    Ok(((1.0 - expanded)? * f32::MIN as f64)?)
}

/// Additive `[seq, seq]` bias restricting attention to a sliding window.
fn sliding_window_bias(seq: usize, max_dist: usize, dev: &Device) -> Result<Tensor> {
    let m: Vec<f32> = (0..seq)
        .flat_map(|i| {
            (0..seq).map(move |j| {
                if (j as i32 - i as i32).unsigned_abs() as usize > max_dist {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    Ok(Tensor::from_slice(&m, (seq, seq), dev)?)
}

/// A ModernBERT encoder whose projections run in bf16.
pub struct Bf16ModernBert {
    embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    local_attention: usize,
    device: Device,
}

impl Bf16ModernBert {
    /// `vb` must be rooted at the encoder itself, i.e. `embeddings.*`,
    /// `layers.*` and `final_norm.*` resolve directly.
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let device = vb.device().clone();
        let embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embeddings.tok_embeddings"),
        )?;
        let norm =
            layer_norm_no_bias(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("embeddings.norm"))?;

        // Local layers use a shorter rope period than global ones.
        let global_rot = Arc::new(RotaryEmbedding::new(cfg, cfg.global_rope_theta, &device)?);
        let local_rot = Arc::new(RotaryEmbedding::new(cfg, cfg.local_rope_theta, &device)?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for id in 0..cfg.num_hidden_layers {
            let uses_local = id % cfg.global_attn_every_n_layers != 0;
            let rot = if uses_local { local_rot.clone() } else { global_rot.clone() };
            layers.push(Layer::load(vb.pp(format!("layers.{id}")), cfg, rot, uses_local)?);
        }
        let final_norm =
            layer_norm_no_bias(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("final_norm"))?;

        Ok(Self {
            embeddings,
            norm,
            layers,
            final_norm,
            local_attention: cfg.local_attention,
            device,
        })
    }

    /// Run one padded batch, returning flat `[batch * seq * dim]` hidden
    /// states and the dimension — same shape contract as the f32 path.
    pub fn forward_batch(
        &self,
        ids: &[i64],
        mask: &[i64],
        batch: usize,
        seq: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let ids_u: Vec<u32> = ids.iter().map(|&v| v as u32).collect();
        let mask_u: Vec<u32> = mask.iter().map(|&v| v as u32).collect();
        let ids = Tensor::from_vec(ids_u, (batch, seq), &self.device)?;
        let mask = Tensor::from_vec(mask_u, (batch, seq), &self.device)?;

        let global_mask = padding_bias(&mask)?;
        let local_mask = sliding_window_bias(seq, self.local_attention / 2, &self.device)?;
        let mut xs = ids.apply(&self.embeddings)?.apply(&self.norm)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &global_mask, &local_mask)?;
        }
        let out = xs.apply(&self.final_norm)?;

        let dim = out.dim(2)?;
        Ok((out.flatten_all()?.to_vec1::<f32>()?, dim))
    }
}
