//! Mixed-precision ModernBERT forward.
//!
//! Structurally a port of candle-transformers' `modernbert.rs`, with the four
//! projection `Linear`s — attention `Wqkv` and `Wo`, MLP `Wi` and `Wo` — run
//! through the bf16 kernel in [`super::gemm`]. Two more stages leave candle
//! without changing precision: the attention mask and softmax fuse into
//! [`super::softmax`], and the GELU gate into [`super::geglu`], both f32 in
//! and f32 out. Everything else (embeddings, LayerNorm, rotary embeddings,
//! attention scores, residuals) is candle's.
//!
//! That split mirrors what `torch.autocast(bf16)` does, and it is why accuracy
//! holds up: the reduced-precision inputs only affect the projections, and
//! mean pooling plus L2 normalization average away most of the per-element
//! noise. End to end the embeddings land at cosine ≈ 0.99999 against the f32
//! path — well inside the noise floor for retrieval ranking.
//!
//! Since only the projections change precision, this is *not* a uniform
//! speedup: it wins most on short inputs, where the projections dominate, and
//! less on long ones, where the f32 attention matmuls — quadratic in sequence
//! length — take over. What claws some of that back is [`Attention::banded`],
//! which computes only the band a sliding-window layer can attend to.

use std::sync::Arc;

use crate::encoder::Config;
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, layer_norm_no_bias, Embedding, LayerNorm, VarBuilder};

use super::geglu;
use super::gemm::Bf16Linear;
use super::softmax::{self, Block};

/// Queries per block in the sliding-window attention path.
///
/// Each block reads `Q_BLOCK + 2w` keys to serve `Q_BLOCK` queries, so smaller
/// blocks compute less of the score matrix — the limit as the block shrinks is
/// the `2w + 1` keys a single query needs — but they mean more and narrower
/// matmuls, each with its own narrow, copy and softmax call.
///
/// That tradeoff has an interior optimum, and it is not where the arithmetic
/// alone would put it. Encode time on 240 512-token texts, median of five
/// interleaved runs against a dense baseline of 22.4 s:
///
/// | `Q_BLOCK` | share of the dense score matrix | encode |
/// |---:|---:|---:|
/// | 16 | 28% | 22.7 s |
/// | 32 | 31% | **20.3 s** |
/// | 64 | 38% | 21.5 s |
/// | 128 | 50% | 21.9 s |
///
/// 16 computes the least and finishes no faster than not banding at all.
const Q_BLOCK: usize = 32;

/// Whether walking the band beats computing the whole score matrix.
///
/// It cannot help once the window already spans the sequence, and near that
/// point the block overhead outweighs what little is masked off, so this
/// wants the band to be a real fraction of the row.
fn banding_pays(seq: usize, window: usize) -> bool {
    seq > 2 * (2 * window + 1)
}

/// Apply a bf16 `Linear` to a tensor whose last dimension is the Linear's
/// input width, flattening the leading dimensions into GEMM rows.
///
/// Each call copies the tensor out to a `Vec<f32>`; the result goes back
/// without a copy, since `from_vec` takes ownership. Profiling puts the copies
/// across every `apply` in a forward at 0.6% of it — the GEMM dominates — so
/// it is not worth the complexity of borrowing candle's storage directly.
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
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
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

    /// `window` is the half-width of a sliding-window layer, or `None` for a
    /// global one.
    fn forward(&self, xs: &Tensor, attn_mask: &[f32], window: Option<usize>) -> Result<Tensor> {
        let (b, seq, d) = xs.dims3()?;
        let qkv = apply(&self.qkv, xs)?
            .reshape((b, seq, 3, self.num_heads, self.head_size))?
            .permute((2, 0, 3, 1, 4))?;
        let (q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
        let (q, k) = self.rotary.apply(&q, &k)?;

        // Scores, mask, and softmax stay f32 — the precision-sensitive part.
        let q = (q * (self.head_size as f64).powf(-0.5))?;

        let ctx = match window {
            Some(w) if banding_pays(seq, w) => self.banded(&q, &k, &v, attn_mask, w)?,
            _ => self.attend(&q, &k, &v, attn_mask, Block::full(seq))?,
        };

        let xs = ctx.transpose(1, 2)?.reshape((b, seq, d))?;
        apply(&self.proj, &xs)
    }

    /// Attention over one block of the score matrix: the queries `block`
    /// names, against the keys it names, returning their `[b, heads, queries,
    /// dim]` context.
    ///
    /// This is the whole of attention — [`Block::full`] makes it the dense
    /// case, so the banded path is not a second implementation of it.
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &[f32],
        block: Block,
    ) -> Result<Tensor> {
        let b = q.dim(0)?;
        let (k0, keys) = block.keys();
        let queries = block.queries();

        let qb = q.narrow(2, block.q0(), queries)?;
        let kb = k.narrow(2, k0, keys)?;
        let vb = v.narrow(2, k0, keys)?;

        let att = qb.matmul(&kb.transpose(D::Minus2, D::Minus1)?)?;

        // The mask add and the softmax are fused (see `super::softmax`), which
        // is why the mask arrives as a flat `[b, seq, seq]` slice rather than a
        // tensor to broadcast against.
        let mut scores = att.flatten_all()?.to_vec1::<f32>()?;
        softmax::masked_softmax(&mut scores, mask, b, self.num_heads, block);
        let att = Tensor::from_vec(scores, (b, self.num_heads, queries, keys), q.device())?;

        att.matmul(&vb).map_err(Into::into)
    }

    /// The same result for a sliding-window layer, without computing the
    /// three quarters of the score matrix the window masks off.
    ///
    /// Queries are walked in blocks of [`Q_BLOCK`], each a small dense
    /// attention over the keys that block can reach ([`Block::band`]). The
    /// masked entries the dense path would have computed contribute `exp` of
    /// the mask floor — around 1e-38 against a normalizer of at least 1 — so
    /// dropping them moves the result by far less than f32 can represent.
    fn banded(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &[f32], w: usize) -> Result<Tensor> {
        let seq = q.dim(2)?;
        let mut blocks = Vec::with_capacity(seq.div_ceil(Q_BLOCK));
        for q0 in (0..seq).step_by(Q_BLOCK) {
            let block = Block::band(seq, q0, Q_BLOCK.min(seq - q0), w);
            blocks.push(self.attend(q, k, v, mask, block)?);
        }
        Tensor::cat(&blocks, 2).map_err(Into::into)
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
    ///
    /// Both GEMMs and the gating run on plain `Vec<f32>`, so the whole MLP
    /// costs one copy in and one out. Going through tensors instead would
    /// allocate three more `[tokens, inter]` intermediates per layer, for the
    /// chunk, the GELU and the multiply.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        debug_assert_eq!(*dims.last().unwrap(), self.wi.k);
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let xv = xs.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let wide = self.wi.forward(&xv, rows);
        let gated = geglu::geglu(&wide, rows, self.wo.k);
        let y = self.wo.forward(&gated, rows);

        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = self.wo.n;
        Tensor::from_vec(y, out_dims, xs.device()).map_err(Into::into)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    /// Absent on the first layer, which reuses the embedding norm.
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    /// The sliding window's half-width, or `None` if this layer attends
    /// globally. Fixed by the config at load time, so the attention path never
    /// has to be told which kind of layer it is running in.
    window: Option<usize>,
}

impl Layer {
    fn load(
        vb: VarBuilder,
        cfg: &Config,
        rotary: Arc<RotaryEmbedding>,
        window: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(vb.pp("attn"), cfg, rotary)?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
            attn_norm: layer_norm_no_bias(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("attn_norm"))
                .ok(),
            mlp_norm: layer_norm_no_bias(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("mlp_norm"))?,
            window,
        })
    }

    fn forward(&self, xs: &Tensor, global_mask: &[f32], local_mask: &[f32]) -> Result<Tensor> {
        let residual = xs.clone();
        let mut h = xs.clone();
        if let Some(norm) = &self.attn_norm {
            h = h.apply(norm)?;
        }
        // `local_mask` already holds `global + local`, combined once in
        // forward_batch — every local layer would otherwise redo the identical
        // broadcast_add, and all but the first are redundant.
        let mask = match self.window {
            Some(_) => local_mask,
            None => global_mask,
        };
        let xs = (residual + self.attn.forward(&h, mask, self.window)?)?;
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
    /// Half of the config's `local_attention`, which is the only form any
    /// caller wants: the distance a sliding-window query reaches either way.
    half_window: usize,
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
        let norm = layer_norm_no_bias(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("embeddings.norm"),
        )?;

        // Local layers use a shorter rope period than global ones.
        let global_rot = Arc::new(RotaryEmbedding::new(cfg, cfg.global_rope_theta, &device)?);
        let local_rot = Arc::new(RotaryEmbedding::new(cfg, cfg.local_rope_theta, &device)?);

        let half_window = cfg.local_attention / 2;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for id in 0..cfg.num_hidden_layers {
            let window = (id % cfg.global_attn_every_n_layers != 0).then_some(half_window);
            let rot = if window.is_some() {
                local_rot.clone()
            } else {
                global_rot.clone()
            };
            layers.push(Layer::load(
                vb.pp(format!("layers.{id}")),
                cfg,
                rot,
                window,
            )?);
        }
        let final_norm =
            layer_norm_no_bias(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("final_norm"))?;

        Ok(Self {
            embeddings,
            norm,
            layers,
            final_norm,
            half_window,
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

        // The sliding-window bias and the padding bias are both identical
        // across layers, so their sum is computed once here rather than in
        // each of the local layers — and both are handed to the fused softmax
        // as plain slices, so the conversion happens once per forward too.
        let global_mask = padding_bias(&mask)?;
        let sliding = sliding_window_bias(seq, self.half_window, &self.device)?;
        let local_mask = global_mask.broadcast_add(&sliding)?;
        let global_mask = global_mask.flatten_all()?.to_vec1::<f32>()?;
        let local_mask = local_mask.flatten_all()?.to_vec1::<f32>()?;
        let mut xs = ids.apply(&self.embeddings)?.apply(&self.norm)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &global_mask, &local_mask)?;
        }
        let out = xs.apply(&self.final_norm)?;

        let dim = out.dim(2)?;
        Ok((out.flatten_all()?.to_vec1::<f32>()?, dim))
    }
}
