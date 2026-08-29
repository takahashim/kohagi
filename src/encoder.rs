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

/// Which spelling of the parameter names a checkpoint uses: `""` for a bare
/// ModernBERT save, `"model."` for one written from a task head that kept the
/// encoder under its own field.
///
/// Every engine that reads a checkpoint has to answer this — candle's
/// `VarBuilder` and the Vulkan backend's safetensors reader both do — and they
/// have to answer it identically, so the probe key and the polarity live here
/// rather than being written out (and inverted) at each site. `has` is the
/// caller's own lookup, because the two hold the file open differently.
pub(crate) fn name_prefix(has: impl Fn(&str) -> bool) -> &'static str {
    if has("model.embeddings.tok_embeddings.weight") {
        "model."
    } else {
        ""
    }
}

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

/// Which attention this model runs.
///
/// Resolved once, from the device, at load. Three things follow from it and
/// have to agree: how the QKV projection is stored, which form the masks take,
/// and how wide a block of scores may be. Asking the device once and keeping
/// the answer is what makes them agree — asked separately they could disagree,
/// and the failure would be a silently slow path or a mask the kernel cannot
/// read, rather than an error.
#[derive(Clone, Copy)]
enum Attn {
    /// Metal, whose sdpa fuses the whole attention and takes one dense mask.
    Fused,
    /// Everywhere else: query blocks over a mask kept in pieces. Carries the
    /// scores one block may hold, which is not the same number on the CPU as on
    /// a GPU running the same path.
    Blocked { budget: usize },
}

impl Attn {
    fn of(device: &Device) -> Self {
        if device.is_metal() {
            Self::Fused
        } else if device.is_cuda() {
            Self::Blocked {
                budget: crate::model::GPU_ATTN_BUDGET,
            }
        } else {
            Self::Blocked {
                budget: crate::model::ATTN_BUDGET,
            }
        }
    }

    /// The global and local masks, in the form this attention consumes.
    ///
    /// The only place a [`Mask`] is built, which is what ties its form to the
    /// rest of the choices above rather than to a second reading of the device.
    fn masks(self, mask: &Tensor, seq: usize, window: usize, dev: &Device) -> Result<(Mask, Mask)> {
        Ok(match self {
            Self::Fused => {
                let padding = prepare_4d_attention_mask(mask, DType::F32, None)?.to_device(dev)?;
                // Summed once here rather than in each of the 13 local layers:
                // the sliding-window mask and the padding mask are identical
                // across layers, so their sum is too.
                let sliding = get_local_attention_mask(seq, window, dev)?;
                let local = padding.broadcast_add(&sliding)?;
                (Mask::Dense(padding), Mask::Dense(local))
            }
            Self::Blocked { .. } => {
                let keys = padding_mask(mask, DType::F32)?.to_device(dev)?;
                (
                    Mask::Blocked {
                        keys: keys.clone(),
                        window: None,
                    },
                    Mask::Blocked {
                        keys,
                        window: Some(window),
                    },
                )
            }
        })
    }
}

/// What one layer's attention may attend to.
///
/// Built only by [`Attn::masks`], so the form always matches the attention that
/// will read it.
enum Mask {
    /// Padding and window already summed, as sdpa wants them.
    Dense(Tensor),
    /// The padding mask `[b, 1, 1, s]`, plus a local layer's window as a
    /// half-width rather than a matrix. Both are applied one block at a time in
    /// [`attend_block`], which is what keeps either from being an `s^2` tensor,
    /// and the half-width is also what lets that path skip the keys the window
    /// shuts out instead of masking them.
    Blocked { keys: Tensor, window: Option<usize> },
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
    attn: Attn,
}

impl ModernBertAttention {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        attn: Attn,
    ) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let qkv = linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?;
        let qkv = if matches!(attn, Attn::Fused) {
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
            attn,
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
            Mask::Blocked { keys, window } => {
                // The matmul cannot consume the transposed views, and there is
                // no sdpa to hand them to, so materialize here rather than
                // paying for it on both backends.
                let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
                let q = (q * scale)?;
                let budget = match self.attn {
                    Attn::Blocked { budget } => budget,
                    // `Mask::Blocked` is built only by `Attn::Blocked::masks`.
                    Attn::Fused => unreachable!("a fused attention has no blocks"),
                };
                let plan = crate::attention::plan(
                    seq_len,
                    *window,
                    query_tile(seq_len, budget),
                    BAND_BLOCK,
                );
                attend_blocked(&q, &k, &v, keys, *window, &plan)?
            }
        };

        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        let xs = xs.apply(&self.proj)?;
        let xs = xs.reshape((b, seq_len, d))?;

        Ok(xs)
    }
}

/// Queries per block in a sliding-window layer.
///
/// A block reads `BAND_BLOCK + 2w` keys to serve `BAND_BLOCK` queries, so
/// narrower blocks compute less of the score matrix, down to the `2w + 1` keys
/// a single query needs. What stops it there is that each block is its own
/// narrow, matmul, mask, softmax and matmul, and candle charges per call.
///
/// 120 512-token texts, fastest of five interleaved runs on an 8-core M-series:
///
/// | `BAND_BLOCK` | keys read per query block | encode |
/// |---:|---:|---:|
/// | 32 | 160 | **17.2 s** |
/// | 64 | 192 | 20.2 s |
/// | 128 | 256 | 22.0 s |
/// | 256 | 384 | 23.2 s |
///
/// The order follows the key count, so nothing here has yet reached the width
/// where per-call overhead takes over. At 8192 tokens all four land within 4%,
/// because banding has already cut the sliding-window layers to 3% of the
/// attention and the seven global ones decide the time. `bf16`'s `Q_BLOCK`
/// measured the same 32 against its own kernels, which are not these.
const BAND_BLOCK: usize = 32;

/// Attention one block of queries at a time, so the `[b, h, s, s]` score
/// tensor is never built.
///
/// This is the dense computation, not an approximation of it. Softmax runs
/// along the key axis, so query rows are independent: a row's scores, its
/// softmax and its weighted sum of V are over the same keys whether the row is
/// computed with all the others or with 31 of them. A sliding-window layer
/// narrows the keys as well, and what that drops are the ones the window masks
/// shut, which contribute `exp(-inf)` — an exact zero — to both the softmax
/// denominator and the sum over V.
///
/// Exact in arithmetic is not the same as identical in f32, though. A GEMM
/// blocks its reduction by the shape it is handed, so a 32-row block and a
/// 128-row one can round differently, and whether they do depends on the BLAS:
/// the same shapes come out bit-identical under one Accelerate and a few ULP
/// apart under another. The tests measure the distance rather than assert the
/// bits.
fn attend_blocked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    keys: &Tensor,
    window: Option<usize>,
    plan: &crate::attention::Plan,
) -> Result<Tensor> {
    let windows = Windows::new(window, plan, q.device())?;

    // One block covering everything skips the narrows and the concatenation:
    // the settings measured at 512 tokens are the ones still running there.
    if let Some(whole) = plan.whole() {
        return attend_block(q, k, v, keys, windows.of(&whole)?.as_ref(), whole);
    }
    let mut out = Vec::with_capacity(plan.blocks.len());
    for block in &plan.blocks {
        let mask = windows.of(block)?;
        out.push(attend_block(q, k, v, keys, mask.as_ref(), *block)?);
    }
    Tensor::cat(&out, 2)
}

/// Where a block's slice of the sliding-window mask comes from.
///
/// A banded layer's blocks all reach the same way — `window` keys to the left
/// and to the right of their own queries — so the mask depends on the offset
/// between a block's first query and its first key, not on where in the
/// sequence the block sits. One table therefore serves every block, the ends
/// included: they take a shifted or shortened view of it. That matters because
/// a banded 8192-token layer has 256 blocks, and building a mask for each would
/// be 256 allocations and 256 host-to-device copies per layer.
///
/// A layer that is windowed but not banded reads every key, so its blocks do
/// not repeat and there is nothing to share. It has one block in every case
/// Kohagi's models produce (a window too wide to band by is a window nearly as
/// wide as the sequence, and a sequence that short fits one tile), so building
/// per block costs one build.
enum Windows {
    None,
    /// The `[width, width + 2 * window]` table every banded block views.
    Table {
        table: Tensor,
        window: usize,
    },
    PerBlock {
        window: usize,
        device: Device,
    },
}

impl Windows {
    fn new(window: Option<usize>, plan: &crate::attention::Plan, device: &Device) -> Result<Self> {
        Ok(match window {
            None => Self::None,
            Some(window) if plan.banded => Self::Table {
                table: window_mask(plan.width, window, window, plan.width + 2 * window, device)?,
                window,
            },
            Some(window) => Self::PerBlock {
                window,
                device: device.clone(),
            },
        })
    }

    /// This block's `[queries, keys]` additive mask, or `None` where the layer
    /// has no window at all.
    fn of(&self, block: &crate::attention::Block) -> Result<Option<Tensor>> {
        let (k0, keys) = block.keys();
        let offset = block.q0() - k0;
        Ok(match self {
            Self::None => None,
            // Entry `(i, j)` of the block is `|i - j + offset| <= window`, and
            // of the table `|i - j + window| <= window`, so the block's row of
            // the table starts `window - offset` columns in.
            Self::Table { table, window } => Some(table.narrow(0, 0, block.queries())?.narrow(
                1,
                window - offset,
                keys,
            )?),
            Self::PerBlock { window, device } => {
                Some(window_mask(block.queries(), offset, *window, keys, device)?)
            }
        })
    }
}

/// An additive sliding-window mask: 0 where a query can reach a key, `-inf`
/// where the window shuts it out.
///
/// `offset` is how far the first query is past the first key, which is what
/// places the band inside the `[queries, keys]` rectangle.
fn window_mask(
    queries: usize,
    offset: usize,
    window: usize,
    keys: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<f32> = (0..queries)
        .flat_map(|i| {
            (0..keys).map(move |j| {
                if (i + offset).abs_diff(j) > window {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (queries, keys), device)
}

/// One block's `softmax(q·kᵀ + mask)·v`.
fn attend_block(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    keys: &Tensor,
    window: Option<&Tensor>,
    block: crate::attention::Block,
) -> Result<Tensor> {
    let (k0, n_keys) = block.keys();
    let (q0, n_queries) = (block.q0(), block.queries());
    // Narrowing the query axis leaves a strided view, and matmul wants its lhs
    // whole. The copy is `queries * heads * head_dim`, the small side of
    // everything here.
    let qb = if n_queries == block.seq {
        q.clone()
    } else {
        q.narrow(2, q0, n_queries)?.contiguous()?
    };
    let whole_row = n_keys == block.seq;
    let (kb, vb) = if whole_row {
        (k.clone(), v.clone())
    } else {
        (k.narrow(2, k0, n_keys)?, v.narrow(2, k0, n_keys)?)
    };
    let padding = if whole_row {
        keys.clone()
    } else {
        keys.narrow(3, k0, n_keys)?
    };

    // Reassigned rather than shadowed: a shadowed binding lives to the end of
    // its block, so the pre-mask scores, the masked scores and the softmax
    // would all be alive at once. Reassigning drops each as the next is bound,
    // leaving the operation's own input and output as the peak.
    let mut att = qb.matmul(&kb.transpose(D::Minus2, D::Minus1)?)?;
    // Padding and window are summed into the block's own shape first, which is
    // one pass over the scores instead of two.
    att = match window {
        Some(w) => att.broadcast_add(&padding.broadcast_add(w)?)?,
        None => att.broadcast_add(&padding)?,
    };
    att = softmax_last_dim(&att)?;
    att.matmul(&vb)
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
fn query_tile(seq: usize, budget: usize) -> usize {
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
        kind: Attn,
    ) -> Result<Self> {
        let attn = ModernBertAttention::load(vb.pp("attn"), config, rotary_emb, kind)?;
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
    attn: Attn,
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

        let attn = Attn::of(vb.device());
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
                attn,
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
            attn,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let dev = xs.device();
        let seq_len = xs.shape().dims()[1];
        let (global, local) = self
            .attn
            .masks(mask, seq_len, self.local_attention_size / 2, dev)?;
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
    /// Ruri v3's own head dimension, and not a smaller one on purpose. The
    /// shapes decide how the GEMM rounds, so a toy head dimension would measure
    /// a reduction no model runs: at `HEAD_DIM = 4` a 48-row tile and a 128-row
    /// one sum the same four products in different orders and land ULPs apart
    /// on a machine where the real shape lands on the same bits.
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

    /// How far apart two outputs are: the largest absolute gap, the largest
    /// value in the reference, and `1 - cos` between them.
    ///
    /// Absolute rather than in ULP. An attention output is a weighted mean of
    /// values spread over `[-1, 1]`, so many of them sit near zero by
    /// cancellation, and there a gap of one part in ten million is hundreds of
    /// ULP while meaning nothing. What the vectors downstream are sensitive to
    /// is the gap against the scale of the output, and their direction.
    fn distance(a: &Tensor, b: &Tensor) -> (f32, f32, f64) {
        let (a, b) = (
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let scale = b.iter().map(|y| y.abs()).fold(0f32, f32::max);
        let norm = |t: &[f32]| t.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        (worst, scale, 1.0 - dot / (norm(&a) * norm(&b)))
    }

    fn against_dense(window: Option<usize>, tile: usize) -> (Tensor, Tensor) {
        let (q, k, v) = (spread(0.0), spread(11.0), spread(23.0));
        let keys = keys();
        let sliding = window.map(|w| get_local_attention_mask(S, w, &Device::Cpu).unwrap());
        let plan = crate::attention::plan(S, window, tile, BAND_BLOCK);
        let blocked = attend_blocked(&q, &k, &v, &keys, window, &plan).unwrap();
        let dense = attend_dense(&q, &k, &v, &keys, sliding.as_ref());
        assert_eq!(blocked.dims(), dense.dims());
        (blocked, dense)
    }

    /// Within f32 rounding of the dense result, and pointing the same way.
    ///
    /// Not bit equality: a GEMM blocks its reduction by the shape it is handed,
    /// so a 32-row block and a 128-row one need not round alike, and neither
    /// need two versions of Accelerate. Both are correct, so an equality
    /// assertion would be testing the BLAS.
    ///
    /// The ceilings sit above rounding and below a mistake. Rounding moves a
    /// value by at most `keys * f32::EPSILON`, 1.5e-5 at these shapes, and
    /// turns the output by ~1e-12. A key the window opens but the band left out
    /// moves its row by a part in thirty and turns the output by ~1e-5.
    fn matches_dense(window: Option<usize>, tile: usize) {
        let (blocked, dense) = against_dense(window, tile);
        let (worst, scale, off_cos) = distance(&blocked, &dense);
        assert!(
            worst <= 1e-4 * scale,
            "moved a value by {worst:e} against a scale of {scale:e}"
        );
        assert!(off_cos <= 1e-9, "turned the output by {off_cos:e}");
    }

    /// Tiling divides the queries and nothing else, including over the ragged
    /// last tile.
    #[test]
    fn tiling_matches_the_dense_result() {
        matches_dense(None, TILE);
    }

    /// A window too wide to band by is still a window: every key is scored and
    /// the mask does the work.
    #[test]
    fn a_window_too_wide_to_band_matches_the_dense_result() {
        assert!(!crate::attention::banding_pays(S, 60));
        matches_dense(Some(60), TILE);
    }

    /// One block covering everything is the one case that has to be the dense
    /// path exactly: it runs the same operations on the same whole tensors,
    /// with no narrow and no concatenation, so any difference at all would mean
    /// the fast path is not the path it claims to be.
    #[test]
    fn a_single_tile_is_the_dense_one() {
        for tile in [S, S * 4] {
            let (blocked, dense) = against_dense(None, tile);
            assert_eq!(bits(&blocked), bits(&dense));
        }
    }

    /// Banding drops the keys the window shuts out rather than masking them.
    /// Those terms are `exp(-inf)`, an exact zero in both the softmax
    /// denominator and the sum over V, so nothing is lost by dropping them.
    #[test]
    fn banding_matches_the_dense_result() {
        assert!(crate::attention::banding_pays(S, 16));
        matches_dense(Some(16), TILE);
    }

    /// The handover with `model::rows_per_forward`: up to `sqrt(budget)` a tile
    /// is the whole sequence and the rows carry the budget, past it the tile
    /// shrinks so that `tile * seq` stays inside it.
    #[test]
    fn the_tile_holds_the_budget_at_any_length() {
        let budget = crate::model::ATTN_BUDGET;
        assert!(query_tile(512, budget) >= 512);
        assert!(query_tile(724, budget) >= 724);
        assert!(query_tile(725, budget) < 725);
        for seq in [725, 1024, 2048, 8192, 100_000] {
            let tile = query_tile(seq, budget);
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
