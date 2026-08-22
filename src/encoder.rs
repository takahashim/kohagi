//! The ModernBERT encoder.
//!
//! Lifted from candle-transformers 0.11.0 (`src/models/modernbert.rs`, MIT OR
//! Apache-2.0) and modified for Kohagi. The original file has no dependency on
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
    pub activation: Activation,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
}

/// The gate's activation in ModernBERT's gated feed-forward, which is what makes
/// the block a GeGLU or a SwiGLU.
///
/// Here rather than beside either forward, because both the candle path and the
/// CoreML emitter choose from it and must agree on which names are supported and
/// on what to say about one that is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// Exact GeLU, the ModernBERT default.
    #[default]
    Gelu,
    /// SiLU, which `hidden_activation: "silu"` selects (granite-embedding-r2).
    Silu,
}

impl Activation {
    /// The `hidden_activation` spelling, for messages and for a converted
    /// bundle's provenance.
    #[cfg_attr(not(feature = "coreml-export"), allow(dead_code))]
    pub fn name(self) -> &'static str {
        match self {
            Self::Gelu => "gelu",
            Self::Silu => "silu",
        }
    }

    /// Resolve a `hidden_activation` value, or say why it cannot be.
    ///
    /// `Err` rather than a fallback to gelu: the wrong nonlinearity still
    /// produces plausible-looking vectors. The caller chooses how to report it —
    /// the runtime fails the parse, the emitter collects it with the other
    /// reasons — but not what counts as supported.
    pub fn from_name(name: Option<&str>) -> std::result::Result<Self, String> {
        match name {
            None | Some("gelu") => Ok(Self::Gelu),
            Some("silu") => Ok(Self::Silu),
            Some(other) => Err(format!(
                "hidden_activation: {other}, and Kohagi's gated feed-forward \
                 implements gelu and silu"
            )),
        }
    }
}

/// A ModernBERT `config.json` as written, before anything is decided about it.
///
/// The one place the file's field names and their alternative spellings are
/// known. Both the runtime [`Config`] and the CoreML emitter's `EncoderConfig`
/// are built from this rather than parsing the file separately: when they did,
/// they drifted, and a checkpoint saved by transformers 5.x converted fine while
/// failing to load.
#[derive(Deserialize)]
pub(crate) struct RawConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) max_position_embeddings: usize,
    // A serde `alias` would reject a config carrying both names as a duplicate
    // field, and ruri-v3 carries both, so they are two optional fields merged
    // by `eps()` rather than one aliased one.
    pub(crate) layer_norm_eps: Option<f64>,
    pub(crate) norm_eps: Option<f64>,
    pub(crate) pad_token_id: u32,
    pub(crate) global_attn_every_n_layers: usize,
    pub(crate) hidden_activation: Option<String>,
    // transformers 5.x moves the RoPE thetas into `rope_parameters` and stops
    // writing the flat keys, so both spellings are optional here and merged by
    // `theta()`.
    pub(crate) global_rope_theta: Option<f64>,
    pub(crate) local_attention: usize,
    pub(crate) local_rope_theta: Option<f64>,
    pub(crate) rope_parameters: Option<RopeParameters>,
}

/// Which attention kind a theta belongs to, naming both spellings of it.
#[derive(Clone, Copy)]
pub(crate) enum Attention {
    Global,
    Local,
}

impl RawConfig {
    /// The LayerNorm epsilon under whichever name it arrived, or HF's default.
    /// The two agree wherever both appear.
    pub(crate) fn eps(&self) -> f64 {
        self.layer_norm_eps.or(self.norm_eps).unwrap_or(1e-5)
    }

    /// The RoPE theta for one attention kind, from either spelling.
    ///
    /// `None` when the config carries neither. That is an error for every caller,
    /// not a default: assuming a theta would produce wrong embeddings for every
    /// position without saying so.
    pub(crate) fn theta(&self, kind: Attention) -> Option<f64> {
        let (nested, flat) = match kind {
            Attention::Global => (
                self.rope_parameters
                    .as_ref()
                    .and_then(|p| p.full_attention.as_ref()),
                self.global_rope_theta,
            ),
            Attention::Local => (
                self.rope_parameters
                    .as_ref()
                    .and_then(|p| p.sliding_attention.as_ref()),
                self.local_rope_theta,
            ),
        };
        nested.and_then(|t| t.rope_theta).or(flat)
    }

    /// The flat name of a kind's theta, for an error message.
    pub(crate) fn theta_name(kind: Attention) -> &'static str {
        match kind {
            Attention::Global => "global_rope_theta",
            Attention::Local => "local_rope_theta",
        }
    }

    pub(crate) fn activation(&self) -> std::result::Result<Activation, String> {
        Activation::from_name(self.hidden_activation.as_deref())
    }
}

/// The transformers 5.x spelling of the RoPE settings: one entry per attention
/// kind, each carrying its own theta.
#[derive(serde::Deserialize)]
pub(crate) struct RopeParameters {
    full_attention: Option<RopeTheta>,
    sliding_attention: Option<RopeTheta>,
}

#[derive(serde::Deserialize)]
pub(crate) struct RopeTheta {
    rope_theta: Option<f64>,
}

impl<'de> Deserialize<'de> for Config {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let r = RawConfig::deserialize(d)?;
        let theta = |kind: Attention| -> std::result::Result<f64, D::Error> {
            r.theta(kind).ok_or_else(|| {
                serde::de::Error::custom(format!(
                    "config has neither `rope_parameters` nor `{}`",
                    RawConfig::theta_name(kind)
                ))
            })
        };
        let activation = r.activation().map_err(serde::de::Error::custom)?;
        let global_rope_theta = theta(Attention::Global)?;
        let local_rope_theta = theta(Attention::Local)?;
        Ok(Config {
            vocab_size: r.vocab_size,
            hidden_size: r.hidden_size,
            num_hidden_layers: r.num_hidden_layers,
            num_attention_heads: r.num_attention_heads,
            intermediate_size: r.intermediate_size,
            max_position_embeddings: r.max_position_embeddings,
            layer_norm_eps: r.eps(),
            pad_token_id: r.pad_token_id,
            global_attn_every_n_layers: r.global_attn_every_n_layers,
            activation,
            global_rope_theta,
            local_attention: r.local_attention,
            local_rope_theta,
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

/// What one layer's attention may attend to.
///
/// The two forms are the backend dispatch. Metal's fused sdpa takes one dense
/// `[b, 1, s, s]` additive mask; everywhere else the mask arrives in pieces,
/// because summing them into that shape is itself an `s^2` tensor and the point
/// of the tiled path is that no such tensor exists. Which form is built is
/// decided once, in [`ModernBert::forward`], alongside the QKV layout
/// [`ModernBertAttention::load`] picks for the same reason.
enum Mask {
    /// Padding and window already summed, as sdpa wants them.
    Dense(Tensor),
    /// The padding mask `[b, 1, 1, s]`, plus the sliding window `[s, s]` on a
    /// local layer. Summed one query tile at a time in [`attend_tile`].
    Tiled {
        keys: Tensor,
        sliding: Option<Tensor>,
    },
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

    fn forward(&self, hidden_states: &Tensor, mask: &Mask) -> Result<Tensor> {
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

        // Neither arm materializes the [b, h, s, s] score tensor: Metal fuses
        // the whole attention into one kernel, and the other backends walk the
        // queries in tiles. sdpa is Metal-only, which is why that is two
        // implementations rather than one.
        let xs = match mask {
            Mask::Dense(dense) => {
                let (mb, _, ms, mk) = dense.dims4()?;
                // Clamp on the small [b, 1, s, s] mask, then widen to the head
                // count as a *view*: sdpa checks dims but reads the mask through
                // strides, so a stride-0 head axis satisfies it without
                // materializing the [b, h, s, s] tensor this fusion exists to
                // avoid.
                //
                // The floor is finite because a fully padded query row is all
                // -inf, and softmax of that is NaN. The explicit path hides it —
                // pooling skips padded positions — but the fused kernel lets the
                // NaN reach the whole row.
                let mask = dense.clamp(-60f32, 0f32)?.broadcast_as((
                    mb,
                    self.num_attention_heads,
                    ms,
                    mk,
                ))?;
                candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.0)?
            }
            Mask::Tiled { keys, sliding } => {
                // The matmul cannot consume the transposed views, and there is
                // no sdpa to hand them to, so materialize here rather than
                // paying for it on both backends.
                let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
                let q = (q * scale)?;
                let tile = query_tile(seq_len, q.device());
                attend_tiled(&q, &k, &v, keys, sliding.as_ref(), tile)?
            }
        };

        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        let xs = xs.apply(&self.proj)?;
        let xs = xs.reshape((b, seq_len, d))?;

        Ok(xs)
    }
}

/// Attention one query tile at a time, so the `[b, h, s, s]` score tensor is
/// never built.
///
/// This is the dense computation, not an approximation of it. Softmax runs
/// along the key axis, so query rows are independent: a row's scores, its
/// softmax and its weighted sum of V are the same numbers reduced in the same
/// order whether the row is computed with all the others or with 63 of them.
/// What changes is only how many of them are in memory at once.
fn attend_tiled(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    keys: &Tensor,
    sliding: Option<&Tensor>,
    tile: usize,
) -> Result<Tensor> {
    let (_, _, seq, _) = q.dims4()?;
    let kt = k.transpose(D::Minus2, D::Minus1)?;
    // Short inputs are one tile, which is the dense path with no narrow, no
    // copy and no concatenation: the settings measured at 512 tokens are the
    // ones still running there.
    if tile >= seq {
        return attend_tile(q, &kt, v, keys, sliding, 0, seq);
    }
    let mut tiles = Vec::with_capacity(seq.div_ceil(tile));
    for start in (0..seq).step_by(tile) {
        let rows = tile.min(seq - start);
        // Narrowing the query axis leaves a strided view, and matmul wants its
        // operands whole. The copy is `rows * heads * head_dim`, which is the
        // small side of everything here.
        let qt = q.narrow(2, start, rows)?.contiguous()?;
        tiles.push(attend_tile(&qt, &kt, v, keys, sliding, start, rows)?);
    }
    Tensor::cat(&tiles, 2)
}

/// One tile's `softmax(q·kᵀ + mask)·v`, for query rows `start .. start + rows`.
fn attend_tile(
    q: &Tensor,
    kt: &Tensor,
    v: &Tensor,
    keys: &Tensor,
    sliding: Option<&Tensor>,
    start: usize,
    rows: usize,
) -> Result<Tensor> {
    // Reassigned rather than shadowed: a shadowed binding lives to the end of
    // its block, so the pre-mask scores, the masked scores and the softmax
    // would all be alive at once. Reassigning drops each as the next is bound,
    // leaving the operation's own input and output as the peak.
    let mut att = q.matmul(kt)?;
    att = match sliding {
        // Padding and window are summed into the tile's own shape first, which
        // is one pass over the scores instead of two, and keeps the association
        // the dense path used: `scores + (padding + window)`.
        Some(w) => att.broadcast_add(&keys.broadcast_add(&w.narrow(0, start, rows)?)?)?,
        None => att.broadcast_add(keys)?,
    };
    att = softmax_last_dim(&att)?;
    att.matmul(v)
}

/// How many query rows one score tile covers.
///
/// `budget / seq`, so a tile holds at most `budget` scores per head and per
/// batch row however long the input is. Below `sqrt(budget)` (724 rows on the
/// CPU) a tile is the whole sequence.
///
/// The same budget caps the rows [`crate::model::rows_per_forward`] puts in one
/// forward, and the pair is what bounds the score memory: under the threshold
/// the rows absorb the growth (`rows * seq^2 <= budget`), over it the rows are
/// pinned at 1 and the tile absorbs it (`tile * seq <= budget`). Either way one
/// forward holds at most `budget` scores per head.
fn query_tile(seq: usize, device: &Device) -> usize {
    let budget = if device.is_cuda() {
        crate::model::GPU_ATTN_BUDGET
    } else {
        crate::model::ATTN_BUDGET
    };
    (budget / seq.max(1)).max(1)
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
    act: Activation,
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
        Ok(Self {
            wi,
            inter,
            act: config.activation,
            wo,
        })
    }
}

impl Module for ModernBertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = match &self.wi {
            Wi::Wide(wi) => crate::fused::gated(&xs.apply(wi)?, self.inter, self.act)?,
            Wi::Split { gate, up } => {
                let g = xs.apply(gate)?;
                let g = match self.act {
                    Activation::Gelu => g.gelu_erf()?,
                    Activation::Silu => g.silu()?,
                };
                (g * xs.apply(up)?)?
            }
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

    fn forward(&self, xs: &Tensor, global: &Mask, local: &Mask) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }

        let mask = if self.uses_local_attention {
            local
        } else {
            global
        };
        let xs = self.attn.forward(&xs, mask)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }
}

/// The additive padding mask, `[b, 1, 1, s]`: 0 where the position holds a
/// token, `f32::MIN` where it is padding.
///
/// Whether a position is padding is a property of the key being attended to,
/// not of the query attending to it, so the query axis is 1 and the
/// `[b, 1, s, s]` that [`prepare_4d_attention_mask`] builds is these same
/// numbers repeated `s` times.
fn padding_mask(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let (bsz, src_len) = mask.dims2()?;
    let mask = mask.reshape((bsz, 1, 1, src_len))?.to_dtype(dtype)?;
    ((1.0 - mask)? * f32::MIN as f64)?.to_dtype(dtype)
}

/// The same mask spread over the query axis as well, `[b, 1, s, s]`.
///
/// Only Metal's sdpa needs it in this shape (see [`Mask`]); everything else
/// takes [`padding_mask`] and never pays for the `s^2`.
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
        let dev = xs.device();
        let seq_len = xs.shape().dims()[1];
        let sliding = get_local_attention_mask(seq_len, self.local_attention_size / 2, dev)?;
        let (global, local) = if dev.is_metal() {
            let padding = prepare_4d_attention_mask(mask, DType::F32, None)?.to_device(dev)?;
            // Summed once here rather than in each of the 13 local layers: the
            // sliding-window mask and the padding mask are identical across
            // layers, so their sum is too.
            let local = padding.broadcast_add(&sliding)?;
            (Mask::Dense(padding), Mask::Dense(local))
        } else {
            let keys = padding_mask(mask, DType::F32)?.to_device(dev)?;
            (
                Mask::Tiled {
                    keys: keys.clone(),
                    sliding: None,
                },
                Mask::Tiled {
                    keys,
                    sliding: Some(sliding),
                },
            )
        };
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, &global, &local)?;
        }
        let xs = xs.apply(&self.final_norm)?;
        Ok(xs)
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;

    const B: usize = 2;
    const H: usize = 3;
    const S: usize = 128;
    /// Ruri v3's own head dimension, and not a smaller one on purpose: these
    /// tests assert bit equality, and the GEMM picks its reduction order partly
    /// from the shape. At `HEAD_DIM = 4` a 48-row tile and a 128-row one sum the
    /// same four products in different orders and land 1 to 3 ULP apart, which
    /// says nothing about the tiling and would only teach the test to accept a
    /// tolerance it does not need at any real size.
    const HEAD_DIM: usize = 64;
    /// Small enough that `S` needs three of them, the last one ragged.
    const TILE: usize = 48;

    /// Deterministic values spread over `[-1, 1]`, so no two positions score
    /// alike and a wrong offset cannot pass by symmetry.
    fn spread(offset: f32) -> Tensor {
        let n = B * H * S * HEAD_DIM;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 + offset) * 0.37).sin()).collect();
        Tensor::from_vec(data, (B, H, S, HEAD_DIM), &Device::Cpu).unwrap()
    }

    /// One row whole and one with two padding positions at the end.
    fn keys() -> Tensor {
        let mut m = vec![1u32; B * S];
        m[B * S - 1] = 0;
        m[B * S - 2] = 0;
        let m = Tensor::from_vec(m, (B, S), &Device::Cpu).unwrap();
        padding_mask(&m, DType::F32).unwrap()
    }

    /// What the tiled path replaced: the whole `[b, h, s, s]` score tensor,
    /// masked, softmaxed and applied in one go.
    fn attend_dense(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        keys: &Tensor,
        sliding: Option<&Tensor>,
    ) -> Tensor {
        let mask = match sliding {
            Some(w) => keys.broadcast_add(w).unwrap(),
            None => keys.clone(),
        };
        let att = q
            .matmul(&k.transpose(D::Minus2, D::Minus1).unwrap())
            .unwrap();
        let att = att.broadcast_add(&mask).unwrap();
        let att = softmax_last_dim(&att).unwrap();
        att.matmul(v).unwrap()
    }

    /// Bit patterns rather than values: this is an identity claim, and `==` on
    /// floats would let a rounding difference through.
    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }

    fn same_as_dense(sliding: Option<&Tensor>, tile: usize) {
        let (q, k, v) = (spread(0.0), spread(11.0), spread(23.0));
        let keys = keys();
        let tiled = attend_tiled(&q, &k, &v, &keys, sliding, tile).unwrap();
        let dense = attend_dense(&q, &k, &v, &keys, sliding);
        assert_eq!(tiled.dims(), dense.dims());
        assert_eq!(bits(&tiled), bits(&dense));
    }

    /// Tiling divides the queries and nothing else, so a global layer's output
    /// is the dense one to the bit — including over the ragged last tile.
    #[test]
    fn tiled_global_attention_is_the_dense_one() {
        same_as_dense(None, TILE);
    }

    /// The same for a sliding-window layer, where each tile also has to take
    /// its own rows of the window mask.
    #[test]
    fn tiled_window_attention_is_the_dense_one() {
        let sliding = get_local_attention_mask(S, 16, &Device::Cpu).unwrap();
        same_as_dense(Some(&sliding), TILE);
    }

    /// A tile at least as wide as the sequence takes the single-tile path,
    /// which skips the narrow and the concatenation.
    #[test]
    fn a_single_tile_is_the_dense_one() {
        let sliding = get_local_attention_mask(S, 16, &Device::Cpu).unwrap();
        same_as_dense(None, S);
        same_as_dense(Some(&sliding), S);
        same_as_dense(Some(&sliding), S * 4);
    }

    /// The handover with `model::rows_per_forward`: up to `sqrt(budget)` a tile
    /// is the whole sequence and the rows carry the budget, past it the tile
    /// shrinks so that `tile * seq` stays inside it.
    #[test]
    fn the_tile_holds_the_budget_at_any_length() {
        let cpu = Device::Cpu;
        assert!(query_tile(512, &cpu) >= 512);
        assert!(query_tile(724, &cpu) >= 724);
        assert!(query_tile(725, &cpu) < 725);
        for seq in [725, 1024, 2048, 8192, 100_000] {
            let tile = query_tile(seq, &cpu);
            assert!(tile >= 1, "seq {seq} tiled to nothing");
            assert!(
                tile * seq <= crate::model::ATTN_BUDGET || tile == 1,
                "seq {seq} tile {tile} holds more than the budget"
            );
        }
    }
}

#[cfg(test)]
mod config_tests {
    use super::{Activation, Config};

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
    fn reads_the_gate_activation() {
        let base = r#""vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "intermediate_size": 128,
            "max_position_embeddings": 512, "pad_token_id": 0,
            "global_attn_every_n_layers": 3, "local_attention": 128,
            "global_rope_theta": 160000.0, "local_rope_theta": 10000.0"#;
        let of = |extra: &str| serde_json::from_str::<Config>(&format!("{{{base}{extra}}}"));

        // Absent means the ModernBERT default.
        assert_eq!(of("").unwrap().activation, Activation::Gelu);
        assert_eq!(
            of(r#", "hidden_activation": "gelu""#).unwrap().activation,
            Activation::Gelu
        );
        assert_eq!(
            of(r#", "hidden_activation": "silu""#).unwrap().activation,
            Activation::Silu
        );

        // Anything else is refused rather than silently treated as gelu.
        let err = of(r#", "hidden_activation": "relu""#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("relu"), "{err}");
    }

    #[test]
    fn reads_the_transformers_5_rope_spelling() {
        // transformers 5.x writes `rope_parameters` and drops the flat keys.
        let json = r#"{
            "vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "intermediate_size": 128,
            "max_position_embeddings": 512, "pad_token_id": 0,
            "global_attn_every_n_layers": 3, "local_attention": 128,
            "rope_parameters": {
                "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }"#;
        let c: Config = serde_json::from_str(json).unwrap();
        assert_eq!(c.global_rope_theta, 160_000.0);
        assert_eq!(c.local_rope_theta, 10_000.0);

        // Carrying neither spelling is an error rather than a silent default.
        let json = r#"{
            "vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "intermediate_size": 128,
            "max_position_embeddings": 512, "pad_token_id": 0,
            "global_attn_every_n_layers": 3, "local_attention": 128
        }"#;
        let err = serde_json::from_str::<Config>(json)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rope_parameters"), "{err}");
    }

    #[test]
    fn accepts_both_spellings() {
        let c: Config =
            serde_json::from_str(&config_json(r#""layer_norm_eps": 1e-5, "norm_eps": 1e-5,"#))
                .unwrap();
        assert_eq!(c.layer_norm_eps, 1e-5);
    }

    /// HF `ModernbertConfig`'s own name — e.g. CodeSearch-ModernBERT-Crow-Plus,
    /// which Kohagi rejected before this alias.
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
