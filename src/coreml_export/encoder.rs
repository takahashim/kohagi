//! Emitting a whole ModernBERT encoder: the prologue, every block, and the final
//! norm.
//!
//! The graph is the one `scripts/convert_coreml.py` produces, transcribed from a
//! converted `cl-nagoya/ruri-v3-130m`: 12
//! prologue operations, then 19 identical 38-operation blocks, then one
//! `layer_norm`. Interface and boundary match `src/coreml.rs` exactly —
//! `input_ids` and `attention_mask` in, `hidden` out, pooling and normalization
//! left to the caller.

use anyhow::{Context, Result};

use super::blob::Writer;
use super::mil::{Builder, DType, Tensor};
use super::modernbert::{self, Activation, BlockOffsets, Config, Stored};
use super::Io;

/// A ModernBERT config, as far as emitting needs it.
#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub eps: f32,
    /// Sliding-window width for the local layers.
    pub local_attention: usize,
    /// Every n-th layer attends globally; the others use the window.
    pub global_every: usize,
    pub local_rope_theta: f32,
    pub global_rope_theta: f32,
    /// The longest position the checkpoint was trained for. A bucket past it has
    /// no trained RoPE frequencies behind it, so it would run and be wrong.
    pub max_positions: usize,
    /// The gate's activation in the feed-forward.
    pub activation: Activation,
}

/// What a `config.json` says that this emitter's fixed graph cannot honour, other
/// than the values it reads.
///
/// Separate from reading because none of it produces a value: these are the keys
/// `crate::encoder` has no opinion on, since the runtime never builds a graph from
/// them. Returned as a list so an unsupported checkpoint reports everything wrong
/// with it at once.
fn graph_assumptions(v: &serde_json::Value, global_every: usize) -> Vec<String> {
    let mut out = Vec::new();

    // Every `linear` this emitter writes takes a zero bias, and every norm has
    // gamma without beta.
    for (key, what) in [
        ("attention_bias", "attention projections"),
        ("mlp_bias", "feed-forward projections"),
        ("norm_bias", "layer normalizations"),
    ] {
        if v.get(key).and_then(serde_json::Value::as_bool) == Some(true) {
            out.push(format!(
                "{key}: true, but the emitted graph gives the {what} no bias"
            ));
        }
    }

    // The emitted RoPE has no scaling, so a `rope_type` other than default would be
    // applied by the reference and not here.
    if let Some(rope) = v.get("rope_parameters") {
        for kind in ["full_attention", "sliding_attention"] {
            match rope
                .get(kind)
                .and_then(|k| k.get("rope_type"))
                .and_then(serde_json::Value::as_str)
            {
                None | Some("default") => {}
                Some(other) => out.push(format!(
                    "rope_parameters.{kind}.rope_type: {other}, and the emitted RoPE \
                     has no scaling"
                )),
            }
        }
    }

    // Where `layer_types` is present it, not the interval, is authoritative — so a
    // disagreement means the interval this emitter follows is the wrong rule.
    if let Some(types) = v.get("layer_types").and_then(serde_json::Value::as_array) {
        let mismatch: Vec<usize> = types
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                let global = t.as_str() == Some("full_attention");
                global != (global_every != 0 && i.is_multiple_of(global_every))
            })
            .map(|(i, _)| i)
            .collect();
        if !mismatch.is_empty() {
            out.push(format!(
                "layer_types: layers {mismatch:?} disagree with \
                 global_attn_every_n_layers {global_every}, and the emitter follows \
                 the interval"
            ));
        }
    }
    out
}

impl EncoderConfig {
    /// Read a Hugging Face `config.json`, refusing anything this emitter would
    /// silently get wrong.
    ///
    /// The graph is fixed: no projection biases, RoPE over the whole head
    /// dimension, and the global layers chosen by `layer % n == 0`. The one thing a
    /// config chooses is the feed-forward gate's activation, gelu or silu. A config
    /// that disagrees with any of those would still convert, and would still
    /// produce plausible-looking numbers, so every one of them is checked here.
    /// All the problems are reported at once:
    /// someone looking at an unsupported checkpoint wants the list, not the first
    /// item.
    ///
    /// Both spellings of the LayerNorm epsilon are accepted, as `crate::encoder`
    /// does — ruri ships `norm_eps` and `layer_norm_eps`.
    pub fn from_json(text: &str) -> Result<Self> {
        // Values come from `crate::encoder`'s reader, so the two never disagree
        // about what a field is called or which spelling wins. What is checked
        // here is only what that reader has no opinion on: the keys the runtime
        // does not model, and the shapes this graph is fixed to.
        let raw: crate::encoder::RawConfig =
            serde_json::from_str(text).context("parsing config.json")?;
        let v: serde_json::Value = serde_json::from_str(text).context("parsing config.json")?;
        let mut unsupported: Vec<String> = Vec::new();

        let hidden = raw.hidden_size;
        let heads = raw.num_attention_heads;
        let global_every = raw.global_attn_every_n_layers;

        // The graph slices each head's last axis in half for RoPE, and reshapes the
        // fused QKV into exactly `heads` heads.
        if heads == 0 || !hidden.is_multiple_of(heads) {
            unsupported.push(format!(
                "head shape: hidden_size {hidden} does not divide into {heads} heads"
            ));
        } else if !(hidden / heads).is_multiple_of(2) {
            unsupported.push(format!(
                "head dimension: {} is odd, and RoPE rotates two halves of it",
                hidden / heads
            ));
        }
        if global_every == 0 {
            unsupported.push(
                "global_attn_every_n_layers: 0, which leaves no rule for choosing the \
                 globally-attending layers"
                    .to_string(),
            );
        }

        unsupported.extend(graph_assumptions(&v, global_every));

        // Collected rather than returned, so an unsupported checkpoint reports
        // everything wrong with it at once.
        let activation = raw.activation().unwrap_or_else(|why| {
            unsupported.push(why);
            Activation::Gelu
        });

        let mut theta = |kind| {
            raw.theta(kind).map(|n| n as f32).unwrap_or_else(|| {
                unsupported.push(format!(
                    "rope theta: neither `rope_parameters` nor `{}` is present, and \
                         defaulting one would be wrong without saying so",
                    crate::encoder::RawConfig::theta_name(kind)
                ));
                0.0
            })
        };
        let global_rope_theta = theta(crate::encoder::Attention::Global);
        let local_rope_theta = theta(crate::encoder::Attention::Local);

        if !unsupported.is_empty() {
            anyhow::bail!(
                "unsupported ModernBERT configuration:\n{}",
                unsupported
                    .iter()
                    .map(|u| format!("- {u}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }

        Ok(Self {
            hidden,
            heads,
            layers: raw.num_hidden_layers,
            intermediate: raw.intermediate_size,
            vocab: raw.vocab_size,
            eps: raw.eps() as f32,
            local_attention: raw.local_attention,
            global_every,
            local_rope_theta,
            global_rope_theta,
            max_positions: raw.max_position_embeddings,
            activation,
        })
    }

    /// The longest bucket this emitter will produce.
    ///
    /// The sliding-window condition is baked as a `seq * seq` bool constant, so the
    /// specification grows with the square of the length: 167 KB at 128, 17 MB at
    /// 4096, 67 MB at 8192. At 4096 CoreML compiles it in seconds; at 8192 its
    /// compiler had not finished after ten minutes on an M2, with no error and no
    /// way for a caller to tell a slow compile from a stuck one.
    ///
    /// Measured rather than derived, so this is the longest length shown to work
    /// rather than the shortest shown to fail. Raising it means measuring the value
    /// you want to raise it to.
    pub const MAX_SEQUENCE_LENGTH: usize = 4096;

    /// Refuse a bucket set this emitter cannot serve, before anything is written.
    ///
    /// Both callers check before opening the checkpoint, so a bad `--sequence-lengths`
    /// costs a config read rather than 500MB of weights.
    pub fn check_lengths(&self, lengths: &[usize]) -> Result<()> {
        anyhow::ensure!(!lengths.is_empty(), "a bundle needs at least one length");
        if let Some(&over) = lengths.iter().find(|&&s| s > self.max_positions) {
            anyhow::bail!(
                "sequence length {over} is past `max_position_embeddings` {}; the \
                 checkpoint has no RoPE frequencies trained that far, and the model \
                 would run and be wrong",
                self.max_positions
            );
        }
        if let Some(&over) = lengths.iter().find(|&&s| s > Self::MAX_SEQUENCE_LENGTH) {
            anyhow::bail!(
                "sequence length {over} is past the longest this emitter can produce \
                 ({}); the window condition is a {over}x{over} constant, which CoreML's \
                 compiler does not get through in any usable time",
                Self::MAX_SEQUENCE_LENGTH
            );
        }
        Ok(())
    }

    fn block(&self, seq: usize, global: bool) -> Config {
        Config {
            hidden: self.hidden,
            heads: self.heads,
            intermediate: self.intermediate,
            seq,
            eps: self.eps,
            rope_theta: if global {
                self.global_rope_theta
            } else {
                self.local_rope_theta
            },
            activation: self.activation,
        }
    }

    /// Whether the head dimension divides evenly and is even, which the RoPE split
    /// needs. Enforced by [`Self::from_json`]; a config built by hand can still be
    /// checked with this.
    pub fn head_dim_ok(&self) -> bool {
        self.heads != 0
            && self.hidden.is_multiple_of(self.heads)
            && (self.hidden / self.heads).is_multiple_of(2)
    }

    /// Whether layer `i` attends globally. The reference model's 19 layers come
    /// out 7 global and 12 local at `global_every == 3`.
    pub fn is_global(&self, layer: usize) -> bool {
        self.global_every == 0 || layer.is_multiple_of(self.global_every)
    }
}

/// Supplies a named weight as `f32`, whatever it is stored as.
///
/// A trait so that the emitter does not care where the checkpoint came from, and
/// so a test can drive it without a 500MB download.
///
/// Names are the checkpoint's own root layout, as `cl-nagoya/ruri-v3-130m` stores
/// them: `embeddings.tok_embeddings.weight`, `layers.0.attn.Wqkv.weight`,
/// `final_norm.weight`. A checkpoint that wraps all of those under `model.`
/// exists too, and resolving between the two belongs to the implementation of
/// this trait rather than here — the inference path draws the same line
/// (`crate::model`, where `ModernBert::load` is given either the root or a
/// prefix-stripping view).
pub trait Weights {
    /// The tensor named `name`, flattened row-major, or an error naming what is
    /// missing. Never a zero fill: a weight this cannot find is a bug in the
    /// caller's names, and zeros would convert and be silently wrong.
    fn get(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>>;

    /// Everything the source holds, for reporting what an emit did not read.
    /// A checkpoint carrying a classifier head or an LM head converts fine and
    /// the extra tensors are simply dropped, which is right — but silence about
    /// it reads as full coverage.
    fn available(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Names the source holds that the emit did not read, with the `model.` prefix
/// some checkpoints wrap the encoder in normalized away so a wrapped checkpoint
/// does not look entirely unused.
pub fn unused(weights: &dyn Weights, read: &[String]) -> Vec<String> {
    let strip = |n: &str| n.strip_prefix("model.").unwrap_or(n).to_string();
    let read: std::collections::BTreeSet<String> = read.iter().map(|n| strip(n)).collect();
    let mut out: Vec<String> = weights
        .available()
        .into_iter()
        .filter(|n| !read.contains(&strip(n)))
        .collect();
    out.sort();
    out
}

/// Where every weight of the emitted model sits in the blob file.
struct Offsets {
    embeddings: Stored,
    embeddings_norm: u64,
    final_norm: u64,
    blocks: Vec<BlockOffsets>,
}

/// Everything in the blob that does not depend on the sequence length.
///
/// Split out from the RoPE tables because that split is what makes a
/// multi-function bundle worth emitting: the lengths share all of this and differ
/// only in [`write_rope_tables`]. Writing it once per length instead was a real
/// mistake, caught by the bundle being three times the size of one length.
struct Shared {
    /// The embedding table: an fp16 blob, or an int8 one with a scale per row.
    embeddings: Stored,
    embeddings_norm: u64,
    final_norm: u64,
    /// Per layer, everything but the RoPE offsets.
    blocks: Vec<BlockOffsets>,
}

/// Quantize per row: one scale for the whole table would spend most of int8's
/// range on whichever row has the largest value.
///
/// Symmetric, so the zero point stays 0. Measured on bekko at an unchanged JaCWIR
/// MAP@10 while taking a two-bucket bundle from 247MB to 149MB.
fn quantize_rows(values: &[f32], rows: usize) -> (Vec<i8>, Vec<f32>) {
    let width = values.len() / rows;
    let mut quantized = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(rows);
    for row in values.chunks(width) {
        let max = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // A row of zeros would divide by zero; any positive scale reproduces it.
        let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
        quantized.extend(
            row.iter()
                .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8),
        );
        scales.push(scale);
    }
    (quantized, scales)
}

/// Write every length-independent weight, in the order the reference model writes
/// them: the embedding table first, then per-layer, then the final norm.
///
/// The zero biases are written once each and shared, as the converter does, rather
/// than repeated per layer.
/// Write one weight, quantized per output row or not.
fn store(out: &mut Writer, values: &[f32], rows: usize, quantize: bool) -> Stored {
    if quantize {
        let (quantized, scales) = quantize_rows(values, rows);
        Stored::Int8 {
            offset: out.write_int8(&quantized),
            scales,
        }
    } else {
        Stored::Fp16(out.write_f32_as_fp16(values))
    }
}

fn write_shared(
    weights: &dyn Weights,
    cfg: &EncoderConfig,
    quantize_embeddings: bool,
    quantize_projections: bool,
    out: &mut Writer,
    read: &mut Vec<String>,
) -> Result<Shared> {
    let mut get = |name: String, shape: &[usize]| -> Result<Vec<f32>> {
        let values = weights.get(&name, shape)?;
        read.push(name);
        Ok(values)
    };
    let h = cfg.hidden;
    let inter = cfg.intermediate;

    let table = get(
        "embeddings.tok_embeddings.weight".to_string(),
        &[cfg.vocab, h],
    )?;
    let embeddings = store(out, &table, cfg.vocab, quantize_embeddings);
    let embeddings_norm = out.write_f32_as_fp16(&get("embeddings.norm.weight".to_string(), &[h])?);

    // ModernBERT's projections have no bias, but the reference graph's `linear`
    // operations all take one; zeros stand in.
    let zeros_3h = out.write_f32_as_fp16(&vec![0.0; 3 * h]);
    let zeros_h = out.write_f32_as_fp16(&vec![0.0; h]);
    let zeros_2i = out.write_f32_as_fp16(&vec![0.0; 2 * inter]);

    let mut blocks = Vec::with_capacity(cfg.layers);
    for layer in 0..cfg.layers {
        let p = format!("layers.{layer}");
        // Layer 0's attn_norm is the identity, and the checkpoint ships no
        // weight for it.
        let attn_norm = (layer > 0)
            .then(|| get(format!("{p}.attn_norm.weight"), &[h]))
            .transpose()?
            .map(|w| out.write_f32_as_fp16(&w));
        blocks.push(BlockOffsets {
            wqkv: store(
                out,
                &get(format!("{p}.attn.Wqkv.weight"), &[3 * h, h])?,
                3 * h,
                quantize_projections,
            ),
            wqkv_bias: zeros_3h,
            wo: store(
                out,
                &get(format!("{p}.attn.Wo.weight"), &[h, h])?,
                h,
                quantize_projections,
            ),
            wo_bias: zeros_h,
            mlp_norm: out.write_f32_as_fp16(&get(format!("{p}.mlp_norm.weight"), &[h])?),
            mlp_wi: store(
                out,
                &get(format!("{p}.mlp.Wi.weight"), &[2 * inter, h])?,
                2 * inter,
                quantize_projections,
            ),
            mlp_wi_bias: zeros_2i,
            mlp_wo: store(
                out,
                &get(format!("{p}.mlp.Wo.weight"), &[h, inter])?,
                h,
                quantize_projections,
            ),
            mlp_wo_bias: zeros_h,
            // Filled in per length by `with_rope`.
            rope_cos: 0,
            rope_sin: 0,
            attn_norm,
        });
    }

    let final_norm = out.write_f32_as_fp16(&get("final_norm.weight".to_string(), &[h])?);
    Ok(Shared {
        embeddings,
        embeddings_norm,
        final_norm,
        blocks,
    })
}

/// The two RoPE table pairs one length needs, one per theta. The reference model
/// has exactly two, shared by the layers that use each.
fn write_rope_tables(
    cfg: &EncoderConfig,
    seq: usize,
    out: &mut Writer,
) -> ((u64, u64), (u64, u64)) {
    (
        modernbert::write_rope(out, &cfg.block(seq, false)),
        modernbert::write_rope(out, &cfg.block(seq, true)),
    )
}

impl Shared {
    /// This length's view of the shared weights: the same offsets, with each
    /// block pointed at the RoPE tables for its theta.
    fn with_rope(&self, cfg: &EncoderConfig, local: (u64, u64), global: (u64, u64)) -> Offsets {
        Offsets {
            embeddings: self.embeddings.clone(),
            embeddings_norm: self.embeddings_norm,
            final_norm: self.final_norm,
            blocks: self
                .blocks
                .iter()
                .enumerate()
                .map(|(layer, b)| {
                    let (cos, sin) = if cfg.is_global(layer) { global } else { local };
                    BlockOffsets {
                        rope_cos: cos,
                        rope_sin: sin,
                        ..b.clone()
                    }
                })
                .collect(),
        }
    }
}

/// The 12 prologue operations: build both attention masks from `attention_mask`,
/// make the token ids non-negative, and look up the embeddings.
///
/// Returns `(embeddings, global_mask, local_mask)`.
fn prologue(
    b: &mut Builder,
    cfg: &EncoderConfig,
    seq: usize,
    ids: &Tensor,
    mask_input: &Tensor,
    off: &Offsets,
) -> (Tensor, Tensor, Tensor) {
    let s = seq as i32;
    let square = [1, 1, seq, seq];

    // attention_mask [1, seq] -> [1, 1, seq, seq], 1 where a position is kept.
    let ax1 = b.const_i32(Tensor::new("m_ax1", DType::Int32, &[1]), &[1]);
    let m3 = b.op(
        "expand_dims",
        Tensor::new("m3", DType::Int32, &[1, 1, seq]),
        &[("x", mask_input), ("axes", &ax1)],
    );
    let ax2 = b.const_i32(Tensor::new("m_ax2", DType::Int32, &[1]), &[2]);
    let m4 = b.op(
        "expand_dims",
        Tensor::new("m4", DType::Int32, &[1, 1, 1, seq]),
        &[("x", &m3), ("axes", &ax2)],
    );
    let reps = b.const_i32(Tensor::new("m_reps", DType::Int32, &[4]), &[1, 1, s, 1]);
    let tiled = b.op(
        "tile",
        Tensor::new("m_tiled", DType::Int32, &square),
        &[("x", &m4), ("reps", &reps)],
    );
    let to_fp16 = b.const_str("m_dtype_fp16", "fp16");
    let as_fp16 = b.op(
        "cast",
        Tensor::new("m_fp16", DType::Fp16, &square),
        &[("x", &tiled), ("dtype", &to_fp16)],
    );
    // 1 - kept: 1 where padding, 0 where kept.
    let one = b.const_fp16(Tensor::new("m_one", DType::Fp16, &[]), &[1.0]);
    let inverted = b.op(
        "sub",
        Tensor::new("m_inverted", DType::Fp16, &square),
        &[("x", &one), ("y", &as_fp16)],
    );
    let to_bool = b.const_str("m_dtype_bool", "bool");
    let is_padding = b.op(
        "cast",
        Tensor::new("m_is_padding", DType::Bool, &square),
        &[("x", &inverted), ("dtype", &to_bool)],
    );
    // -inf where padding, else the 0 that `inverted` already holds. The reference
    // model uses fp16 -inf rather than a finite sentinel.
    let neg_inf = b.const_fp16_bits(
        Tensor::new("m_neg_inf", DType::Fp16, &[]),
        half::f16::NEG_INFINITY,
    );
    let global_mask = b.op(
        "select",
        Tensor::new("global_mask", DType::Fp16, &square),
        &[("cond", &is_padding), ("a", &neg_inf), ("b", &inverted)],
    );
    // The window itself is a constant: only the padding part depends on input.
    let outside = modernbert::window_condition(seq, cfg.local_attention);
    let outside_c = b.const_bools_shaped(Tensor::new("m_outside", DType::Bool, &square), &outside);
    let local_mask = b.op(
        "select",
        Tensor::new("local_mask", DType::Fp16, &square),
        &[("cond", &outside_c), ("a", &neg_inf), ("b", &global_mask)],
    );

    // Negative ids wrap into the table, as the traced Python does.
    let zero = b.const_i32(Tensor::new("id_zero", DType::Int32, &[]), &[0]);
    let non_negative = b.op(
        "greater_equal",
        Tensor::new("id_ok", DType::Bool, &[1, seq]),
        &[("x", ids), ("y", &zero)],
    );
    let vocab = b.const_i32(
        Tensor::new("id_vocab", DType::Int32, &[]),
        &[cfg.vocab as i32],
    );
    let wrapped = b.op(
        "add",
        Tensor::new("id_wrapped", DType::Int32, &[1, seq]),
        &[("x", ids), ("y", &vocab)],
    );
    let safe_ids = b.op(
        "select",
        Tensor::new("id_safe", DType::Int32, &[1, seq]),
        &[("cond", &non_negative), ("a", ids), ("b", &wrapped)],
    );

    let declared = Tensor::new("tok_embeddings", DType::Fp16, &[cfg.vocab, cfg.hidden]);
    let table = off.embeddings.bind_in(b, declared);
    let axis = b.const_i32(Tensor::new("emb_axis", DType::Int32, &[]), &[0]);
    let batch_dims = b.const_i32(Tensor::new("emb_batch_dims", DType::Int32, &[]), &[0]);
    let validate = b.const_bool("emb_validate", false);
    let embedded = b.op(
        "gather",
        Tensor::new("embedded", DType::Fp16, &[1, seq, cfg.hidden]),
        &[
            ("x", &table),
            ("indices", &safe_ids),
            ("axis", &axis),
            ("batch_dims", &batch_dims),
            ("validate_indices", &validate),
        ],
    );
    (embedded, global_mask, local_mask)
}

/// The name of the function serving `seq` in a multi-function bundle. One place so
/// that this and `crate::coreml`'s reader agree.
pub fn function_name(seq: usize) -> String {
    format!("seq_{seq}")
}

/// Build one length's graph over an already-written blob.
fn build(cfg: &EncoderConfig, off: &Offsets, seq: usize) -> (Builder, Tensor, Tensor, Tensor) {
    let ids = Tensor::new("input_ids", DType::Int32, &[1, seq]);
    let mask_input = Tensor::new("attention_mask", DType::Int32, &[1, seq]);
    let mut b = Builder::new(&[ids.clone(), mask_input.clone()]);
    let out = graph(&mut b, cfg, off, seq, &ids, &mask_input);
    (b, ids, mask_input, out)
}

/// Emit a fixed-length encoder: the `.mlmodel` and the `weight.bin` beside it.
/// Emit one length as a plain single-function model, whose only function is
/// `main`.
///
/// [`emit_all`] is what a converted directory should carry, even for one length:
/// a `seq_<N>` function reads the same way whether the bundle serves one bucket or
/// four. This exists because a single-function model is the simplest thing to
/// compare against the Python conversion, which emits exactly that per length.
pub fn emit(
    cfg: &EncoderConfig,
    weights: &dyn Weights,
    seq: usize,
) -> Result<(crate::coreml_proto::Model, Vec<u8>)> {
    let mut blob = Writer::new();
    let mut read = Vec::new();
    let shared = write_shared(weights, cfg, false, false, &mut blob, &mut read)
        .context("collecting the weights for a CoreML package")?;
    let (local, global) = write_rope_tables(cfg, seq, &mut blob);
    let off = shared.with_rope(cfg, local, global);
    let (b, ids, mask_input, out) = build(cfg, &off, seq);
    let model = super::model(
        b.finish(),
        Io {
            inputs: vec![ids, mask_input],
            outputs: vec![out],
        },
    );
    Ok((model, blob.finish()))
}

/// Emit one bundle serving several lengths, as `seq_<N>` functions over a single
/// copy of the weights.
///
/// This is the form worth publishing. Only the RoPE tables depend on the length,
/// so for a large-vocabulary encoder the lengths share nearly every byte: the
/// Python route reaches the same place by converting each length separately and
/// deduplicating afterwards (`save_multifunction`), and three buckets of
/// bekko-a25m go from 740MB to 248MB that way.
/// Emitting it directly means the duplicate copies never exist.
pub fn emit_multi(
    cfg: &EncoderConfig,
    weights: &dyn Weights,
    lengths: &[usize],
) -> Result<(crate::coreml_proto::Model, Vec<u8>)> {
    emit_all(cfg, weights, lengths, &super::Provenance::default())
}

/// One bundle for `lengths`, recording what produced it.
pub fn emit_all(
    cfg: &EncoderConfig,
    weights: &dyn Weights,
    lengths: &[usize],
    provenance: &super::Provenance,
) -> Result<(crate::coreml_proto::Model, Vec<u8>)> {
    emit_with(cfg, weights, lengths, provenance, &Options::default())
}

/// What to vary about an emit beyond the model itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Store the embedding table as int8 with a scale per row, dequantized in the
    /// graph. For a large vocabulary that is most of the bytes: on bekko it took a
    /// bundle from 247MB to 149MB at an unchanged JaCWIR MAP@10.
    /// Vectors shift by ~7e-4, so a quantized
    /// bundle and an fp16 one should not share an index.
    pub quantize_embeddings: bool,
    /// The same for every projection weight, with a scale per output channel. The
    /// norms and the biases stay fp16.
    pub quantize_projections: bool,
}

/// One bundle for `lengths`, with options.
pub fn emit_with(
    cfg: &EncoderConfig,
    weights: &dyn Weights,
    lengths: &[usize],
    provenance: &super::Provenance,
    opts: &Options,
) -> Result<(crate::coreml_proto::Model, Vec<u8>)> {
    cfg.check_lengths(lengths)?;
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut blob = Writer::new();
    // The length-independent weights go in once, and only the RoPE tables repeat.
    let mut read = Vec::new();
    let shared = write_shared(
        weights,
        cfg,
        opts.quantize_embeddings,
        opts.quantize_projections,
        &mut blob,
        &mut read,
    )
    .context("collecting the weights for a CoreML bundle")?;
    for name in unused(weights, &read) {
        eprintln!("kohagi: the checkpoint's `{name}` was not read; the emitted encoder has no place for it");
    }
    let mut per_length = Vec::with_capacity(sorted.len());
    for &seq in &sorted {
        let (local, global) = write_rope_tables(cfg, seq, &mut blob);
        per_length.push((seq, shared.with_rope(cfg, local, global)));
    }

    let mut program: Option<crate::coreml_proto::mil_spec::Program> = None;
    let mut functions = Vec::new();
    for (seq, off) in &per_length {
        let (b, ids, mask_input, out) = build(cfg, off, *seq);
        let one = b.finish();
        let name = function_name(*seq);
        let f = one
            .functions
            .get("main")
            .expect("build() emits a main function")
            .clone();
        let program = program.get_or_insert_with(|| crate::coreml_proto::mil_spec::Program {
            version: one.version,
            functions: std::collections::BTreeMap::new(),
            doc_string: String::new(),
            attributes: std::collections::BTreeMap::new(),
        });
        program.functions.insert(name.clone(), f);
        functions.push((name, vec![ids, mask_input], vec![out]));
    }

    let model = super::multi_function_model(
        program.expect("at least one length"),
        &functions,
        &function_name(sorted[0]),
        provenance,
    );
    Ok((model, blob.finish()))
}

/// The graph itself, shared by both entry points.
fn graph(
    b: &mut Builder,
    cfg: &EncoderConfig,
    off: &Offsets,
    seq: usize,
    ids: &Tensor,
    mask_input: &Tensor,
) -> Tensor {
    let (embedded, global_mask, local_mask) = prologue(b, cfg, seq, ids, mask_input, off);
    let epsilon = b.const_fp16(Tensor::new("eps", DType::Fp16, &[]), &[cfg.eps]);

    // The embedding norm, which stands in for layer 0's identity attn_norm.
    let gamma = b.const_blob(
        Tensor::new("embeddings_norm", DType::Fp16, &[cfg.hidden]),
        off.embeddings_norm,
    );
    let axes = b.const_i32(Tensor::new("emb_norm_axes", DType::Int32, &[1]), &[-1]);
    let mut hidden = b.op(
        "layer_norm",
        Tensor::new("embeddings_normed", DType::Fp16, &[1, seq, cfg.hidden]),
        &[
            ("x", &embedded),
            ("axes", &axes),
            ("epsilon", &epsilon),
            ("gamma", &gamma),
        ],
    );

    for layer in 0..cfg.layers {
        let global = cfg.is_global(layer);
        let mask = if global { &global_mask } else { &local_mask };
        hidden = modernbert::block(
            b,
            &cfg.block(seq, global),
            layer,
            &hidden,
            &off.blocks[layer],
            mask,
            &epsilon,
        );
    }

    let final_gamma = b.const_blob(
        Tensor::new("final_norm", DType::Fp16, &[cfg.hidden]),
        off.final_norm,
    );
    let final_axes = b.const_i32(Tensor::new("final_axes", DType::Int32, &[1]), &[-1]);
    let out = b.op(
        "layer_norm",
        Tensor::new("hidden", DType::Fp16, &[1, seq, cfg.hidden]),
        &[
            ("x", &hidden),
            ("axes", &final_axes),
            ("epsilon", &epsilon),
            ("gamma", &final_gamma),
        ],
    );
    b.returns(&out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EncoderConfig {
        EncoderConfig {
            max_positions: 8192,
            activation: Activation::Gelu,
            hidden: 512,
            heads: 8,
            layers: 19,
            intermediate: 2048,
            vocab: 102_400,
            eps: 1e-5,
            local_attention: 128,
            global_every: 3,
            local_rope_theta: 10_000.0,
            global_rope_theta: 160_000.0,
        }
    }

    /// A config with everything this emitter needs and nothing it refuses, with
    /// `extra` merged over it.
    ///
    /// Merged rather than appended: a repeated key is a duplicate field, which the
    /// reader rejects outright, and a case here means "this checkpoint says
    /// something else", not "this file is malformed".
    fn json(extra: &str) -> String {
        let base = r#"{"hidden_size": 512, "num_attention_heads": 8, "num_hidden_layers": 19,
            "intermediate_size": 2048, "vocab_size": 102400, "norm_eps": 1e-5,
            "local_attention": 128, "global_attn_every_n_layers": 3,
            "max_position_embeddings": 8192, "pad_token_id": 1,
            "local_rope_theta": 10000.0, "global_rope_theta": 160000.0}"#;
        let mut v: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(base).expect("the base config parses");
        if !extra.is_empty() {
            let over: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&format!("{{{extra}}}")).expect("the override parses");
            v.extend(over);
        }
        serde_json::to_string(&v).expect("serializes")
    }

    #[test]
    fn a_supported_config_reads_the_values_it_needs() {
        let cfg = EncoderConfig::from_json(&json("")).expect("supported");
        assert_eq!((cfg.hidden, cfg.heads, cfg.layers), (512, 8, 19));
        assert!(cfg.head_dim_ok());
        assert_eq!(cfg.local_rope_theta, 10_000.0);
        assert_eq!(cfg.global_rope_theta, 160_000.0);
        assert_eq!(cfg.max_positions, 8192);
    }

    #[test]
    fn the_new_rope_spelling_is_read_rather_than_defaulted() {
        // transformers 4.57 moves the thetas into `rope_parameters`. A config with
        // only those must not silently come out at the legacy default.
        let text = json(
            r#""rope_parameters": {
                "full_attention": {"rope_theta": 160000, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 160000, "rope_type": "default"}},
               "local_rope_theta": null, "global_rope_theta": null"#,
        );
        let cfg = EncoderConfig::from_json(&text).expect("supported");
        assert_eq!(cfg.local_rope_theta, 160_000.0);
        assert_eq!(cfg.global_rope_theta, 160_000.0);

        // And neither spelling present is an error, not a default.
        let text = json(r#""local_rope_theta": null, "global_rope_theta": null"#);
        let err = format!(
            "{:#}",
            EncoderConfig::from_json(&text).expect_err("no theta")
        );
        assert!(err.contains("rope theta"), "{err}");
    }

    #[test]
    fn every_assumption_the_graph_makes_is_refused_when_broken() {
        let cases = [
            (r#""attention_bias": true"#, "attention_bias"),
            (r#""mlp_bias": true"#, "mlp_bias"),
            (r#""norm_bias": true"#, "norm_bias"),
            (r#""hidden_activation": "relu""#, "hidden_activation"),
            (r#""num_attention_heads": 7"#, "head shape"),
            (
                r#""global_attn_every_n_layers": 0"#,
                "global_attn_every_n_layers",
            ),
            (
                r#""rope_parameters": {"full_attention": {"rope_theta": 1, "rope_type": "llama3"}}"#,
                "rope_type",
            ),
        ];
        for (extra, expected) in cases {
            let err = format!(
                "{:#}",
                EncoderConfig::from_json(&json(extra)).expect_err(extra)
            );
            assert!(
                err.contains("unsupported ModernBERT configuration"),
                "{err}"
            );
            assert!(err.contains(expected), "{extra} -> {err}");
        }
    }

    #[test]
    fn a_bucket_the_emitter_cannot_serve_is_refused() {
        let cfg = EncoderConfig::from_json(&json("")).expect("supported");
        assert_eq!(cfg.max_positions, 8192);

        cfg.check_lengths(&[128, 256, EncoderConfig::MAX_SEQUENCE_LENGTH])
            .expect("up to the emitter's limit");

        // Past what CoreML's compiler gets through, but still inside the positions
        // the checkpoint was trained for — so the two limits are separate reasons and
        // each says which one it is.
        let err = format!(
            "{:#}",
            cfg.check_lengths(&[EncoderConfig::MAX_SEQUENCE_LENGTH + 1])
                .expect_err("past the emitter's limit")
        );
        assert!(err.contains("longest this emitter can produce"), "{err}");

        let err = format!(
            "{:#}",
            cfg.check_lengths(&[8193])
                .expect_err("past the trained positions")
        );
        assert!(err.contains("max_position_embeddings"), "{err}");

        let err = format!("{:#}", cfg.check_lengths(&[]).expect_err("no lengths"));
        assert!(err.contains("at least one length"), "{err}");
    }

    #[test]
    fn layer_types_that_disagree_with_the_interval_are_refused() {
        // The interval is what the emitter follows, so a config whose explicit
        // per-layer kinds differ would convert into the wrong model.
        let mut types = vec!["sliding_attention"; 19];
        types[0] = "full_attention";
        types[1] = "full_attention"; // 1 % 3 != 0, so this disagrees
        let extra = format!(
            r#""layer_types": [{}]"#,
            types
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let err = format!(
            "{:#}",
            EncoderConfig::from_json(&json(&extra)).expect_err("disagreement")
        );
        assert!(err.contains("layer_types"), "{err}");
        assert!(err.contains('1'), "{err}");

        // The interval's own assignment is accepted.
        let agreeing: Vec<String> = (0..19)
            .map(|i| {
                let t = if i % 3 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                };
                format!("\"{t}\"")
            })
            .collect();
        let extra = format!(r#""layer_types": [{}]"#, agreeing.join(", "));
        EncoderConfig::from_json(&json(&extra)).expect("agreeing layer_types");
    }

    #[test]
    fn the_global_layers_are_the_ones_the_reference_model_has() {
        let cfg = cfg();
        let global: Vec<usize> = (0..cfg.layers).filter(|&l| cfg.is_global(l)).collect();
        // 7 global and 12 local, which is what the reference model's two RoPE
        // tables are used by (14 and 24 multiplications, two per layer).
        assert_eq!(global, [0, 3, 6, 9, 12, 15, 18]);
        assert_eq!(cfg.layers - global.len(), 12);
    }

    #[test]
    fn each_theta_goes_to_the_layers_that_use_it() {
        let cfg = cfg();
        assert_eq!(cfg.block(128, true).rope_theta, 160_000.0);
        assert_eq!(cfg.block(128, false).rope_theta, 10_000.0);
    }

    /// A missing weight must be reported, not filled in.
    /// A checkpoint of the right shapes with deterministic values, so an emit can
    /// be exercised without a 500MB download. The shapes come from whatever the
    /// emitter asks for, so it needs no config of its own.
    struct Synthetic;

    impl Weights for Synthetic {
        fn get(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
            let n: usize = expected.iter().product();
            // Seeded by the name, so every tensor differs and none is zero.
            let mut state = name.bytes().fold(0x811c_9dc5u32, |h, b| {
                (h ^ u32::from(b)).wrapping_mul(16_777_619)
            }) | 1;
            Ok((0..n)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 16) as f32 / 32_768.0 - 1.0
                })
                .collect())
        }
    }

    fn tiny() -> EncoderConfig {
        EncoderConfig {
            hidden: 16,
            heads: 2,
            layers: 3,
            intermediate: 8,
            vocab: 32,
            eps: 1e-5,
            local_attention: 4,
            global_every: 3,
            local_rope_theta: 10_000.0,
            global_rope_theta: 160_000.0,
            max_positions: 64,
            activation: Activation::Gelu,
        }
    }

    /// A frozen hash of both artifacts.
    ///
    /// Generation is reproducible by design, so any change to the graph, the
    /// constants or the blob layout moves these numbers. That is the point: a
    /// change that was meant to be a refactor shows up here, and a change that was
    /// meant to alter the output has to be acknowledged by updating them — and by
    /// bumping `GRAPH_VERSION`, so that no cached conversion outlives the graph it
    /// was emitted from.
    ///
    /// The provenance metadata is cleared first, because it carries
    /// `CARGO_PKG_VERSION`: leaving it in would move these hashes on every release
    /// and, through that convention, invalidate every user's conversion cache for a
    /// graph that had not changed.
    ///
    /// If this fails and the change was intended, `mil-inventory --diff` and
    /// `milblob diff` (`tools/coreml-jigs`) say what moved.
    #[test]
    fn the_emitted_artifacts_match_their_golden_hashes() {
        use prost::Message;

        let cfg = tiny();
        let (mut model, blob) = emit_all(
            &cfg,
            &Synthetic,
            &[8, 16],
            &super::super::Provenance::default(),
        )
        .expect("emit");
        if let Some(description) = model.description.as_mut() {
            description.metadata = None;
        }
        let encoded = model.encode_to_vec();

        assert_eq!(encoded.len(), 55_944, "model.mlmodel size");
        assert_eq!(
            crate::fnv::hash(&encoded),
            0x77c6_381d_d954_3ad5,
            "model.mlmodel digest"
        );
        assert_eq!(blob.len(), 13_760, "weight.bin size");
        assert_eq!(
            crate::fnv::hash(&blob),
            0x979d_60e1_49a0_dad6,
            "weight.bin digest"
        );
    }

    /// The operation count is fixed by the configuration alone: 12 for the prologue,
    /// one for the embedding norm, 37 for layer 0 (whose `attn_norm` is the
    /// identity), 38 for each layer after it, and one final norm.
    ///
    /// Checked over a range of shapes rather than one, because the arithmetic that
    /// would break it — a head split, a GeGLU half, a reshape — depends on them.
    #[test]
    fn the_operation_count_follows_from_the_config_alone() {
        let mut state = 0x5eed_1234u32;
        let mut next = |lo: usize, hi: usize| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            lo + (state >> 16) as usize % (hi - lo + 1)
        };
        for _ in 0..24 {
            let heads = next(1, 6);
            let cfg = EncoderConfig {
                // Even head dimensions only; `from_json` refuses the rest.
                hidden: heads * 2 * next(1, 8),
                heads,
                layers: next(1, 6),
                intermediate: next(1, 8) * 2,
                vocab: next(4, 64),
                eps: 1e-5,
                local_attention: next(1, 4) * 2,
                global_every: next(1, 4),
                local_rope_theta: 10_000.0,
                global_rope_theta: 160_000.0,
                max_positions: 8192,
                activation: Activation::Gelu,
            };
            assert!(cfg.head_dim_ok(), "{cfg:?}");
            let seq = next(1, 4) * 4;
            let (model, blob) = emit_all(
                &cfg,
                &Synthetic,
                &[seq],
                &super::super::Provenance::default(),
            )
            .unwrap_or_else(|e| panic!("{cfg:?}: {e:#}"));

            let Some(crate::coreml_proto::model::Type::MlProgram(program)) = &model.r#type else {
                panic!("not an ML Program");
            };
            let block = program.functions[&function_name(seq)]
                .block_specializations
                .values()
                .next()
                .expect("a block");
            let compute = block
                .operations
                .iter()
                .filter(|op| op.r#type != "const")
                .count();
            assert_eq!(
                compute,
                38 * cfg.layers + 13,
                "{cfg:?} at seq {seq} produced {compute} operations"
            );

            // And the output is what `src/coreml.rs` reads, at any shape.
            let out = &block.outputs;
            assert_eq!(out, &["hidden".to_string()]);
            assert!(!blob.is_empty());
        }
    }

    /// Two emits of the same inputs produce the same bytes, whatever the shape.
    #[test]
    fn emitting_is_reproducible_across_shapes() {
        for (hidden, heads, layers) in [(16, 2, 1), (32, 4, 3), (24, 3, 2)] {
            let cfg = EncoderConfig {
                hidden,
                heads,
                layers,
                ..tiny()
            };
            let once = emit_all(&cfg, &Synthetic, &[8], &super::super::Provenance::default());
            let twice = emit_all(&cfg, &Synthetic, &[8], &super::super::Provenance::default());
            let (a, b) = (once.expect("first"), twice.expect("second"));
            use prost::Message;
            assert_eq!(a.0.encode_to_vec(), b.0.encode_to_vec(), "{cfg:?} model");
            assert_eq!(a.1, b.1, "{cfg:?} blob");
        }
    }

    #[test]
    fn weights_the_emit_never_reads_are_reported() {
        struct Head;
        impl Weights for Head {
            fn available(&self) -> Vec<String> {
                vec![
                    // Under `model.`, as a wrapped checkpoint stores the encoder.
                    "model.embeddings.norm.weight".to_string(),
                    // And two the emitter has no place for.
                    "head.dense.weight".to_string(),
                    "decoder.bias".to_string(),
                ]
            }
            fn get(&self, _: &str, _: &[usize]) -> Result<Vec<f32>> {
                unreachable!("not called by this test")
            }
        }
        // The prefix is normalized away, so a wrapped checkpoint does not read as
        // entirely unused.
        let read = vec!["embeddings.norm.weight".to_string()];
        assert_eq!(
            unused(&Head, &read),
            ["decoder.bias".to_string(), "head.dense.weight".to_string()]
        );
    }

    #[test]
    fn a_missing_weight_stops_the_emit() {
        struct Empty;
        impl Weights for Empty {
            fn get(&self, name: &str, _: &[usize]) -> Result<Vec<f32>> {
                anyhow::bail!("no tensor named {name}")
            }
        }
        let err = emit(&cfg(), &Empty, 128).expect_err("cannot emit without weights");
        let text = format!("{err:#}");
        assert!(text.contains("tok_embeddings"), "{text}");
    }
}
