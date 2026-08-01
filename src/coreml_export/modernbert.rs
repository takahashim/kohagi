//! Emitting a ModernBERT transformer block as MIL.
//!
//! Transcribed from the block a converted `cl-nagoya/ruri-v3-130m` contains: 38
//! operations repeated identically for all 19 layers. Only one input differs between layers: which of
//! the two attention masks the pre-softmax `add` consumes.
//!
//! The order and the parameters are the reference model's, not a fresh design:
//! matching a graph that is known to load, compute correctly and land on the ANE
//! is worth more than a tidier one that has to establish all three from scratch.
//!
//! Two details of the reference are worth stating because they look like mistakes:
//!
//! - **Every `linear` carries a bias**, although ModernBERT's projections have
//!   none. The tracer materializes zero biases, and the converter shares one
//!   zero-filled blob between the two 512-wide ones. Reproduced rather than
//!   dropped, so the operation shapes match.
//! - **A block's leading `layer_norm` is the previous residual's norm.** For
//!   layer 0 that is the embedding norm, because ModernBERT's first layer has an
//!   identity `attn_norm`. That is what keeps all 19 blocks the same length.

use half::f16;

use super::blob::Writer;
use super::mil::{Builder, DType, Tensor};

/// The parts of a ModernBERT config this emitter needs.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub seq: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub activation: Activation,
}

/// The gate's activation, shared with the candle forward so that a config means
/// the same thing on both. Each is one MIL operation of the same shape, so the
/// rest of the graph does not change with it.
pub use crate::encoder::Activation;

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// `1 / sqrt(head_dim)`, the reference model's `mul` after the QK matmul.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim() as f32).sqrt()
    }
}

/// How one weight is stored in the blob file.
///
/// A projection is either fp16 or int8 with a scale per output channel. The norms,
/// the biases and the RoPE tables stay fp16: they are small, and a norm's gamma is
/// where precision shows.
#[derive(Debug, Clone)]
pub enum Stored {
    Fp16(u64),
    Int8 { offset: u64, scales: Vec<f32> },
}

impl Stored {
    /// Place this weight in the graph, dequantizing it if that is how it is
    /// stored. `constexpr_affine_dequantize` is a constant expression, so the
    /// operation consuming it sees a constant either way.
    pub fn bind_in(&self, b: &mut Builder, out: Tensor) -> Tensor {
        match self {
            Self::Fp16(offset) => b.const_blob(out, *offset),
            // Axis 0: one scale per output channel of an `[out, in]` weight.
            Self::Int8 { offset, scales } => b.dequantize_int8(out, *offset, 0, scales, 0),
        }
    }
}

/// Where one block's weights sit in `weights/weight.bin`.
#[derive(Debug, Clone)]
pub struct BlockOffsets {
    /// `[3 * hidden, hidden]`, the fused Q/K/V projection.
    pub wqkv: Stored,
    /// `[3 * hidden]`, zero in a real checkpoint.
    pub wqkv_bias: u64,
    /// `[hidden, hidden]`.
    pub wo: Stored,
    /// `[hidden]`, shared with `mlp_wo_bias` in the reference model.
    pub wo_bias: u64,
    /// `[hidden]`.
    pub mlp_norm: u64,
    /// `[2 * intermediate, hidden]`, the gated feed-forward's input projection.
    pub mlp_wi: Stored,
    /// `[2 * intermediate]`.
    pub mlp_wi_bias: u64,
    /// `[hidden, intermediate]`.
    pub mlp_wo: Stored,
    pub mlp_wo_bias: u64,
    /// `[1, 1, seq, head_dim]` each.
    pub rope_cos: u64,
    pub rope_sin: u64,
    /// `[hidden]`, or `None` for a block whose norm is applied by its caller
    /// (layer 0, whose `attn_norm` is the identity).
    pub attn_norm: Option<u64>,
}

/// The RoPE tables a fixed sequence length lets us precompute.
///
/// Each is `[1, 1, seq, head_dim]` with the angles duplicated across the two
/// halves of the last axis, which is the layout the `x1`/`x2` split in
/// `rotate_half` expects.
pub fn rope_tables(cfg: &Config) -> (Vec<f32>, Vec<f32>) {
    let d = cfg.head_dim();
    let half = d / 2;
    let mut cos = Vec::with_capacity(cfg.seq * d);
    let mut sin = Vec::with_capacity(cfg.seq * d);
    for pos in 0..cfg.seq {
        let angles: Vec<f32> = (0..half)
            .map(|i| {
                let exponent = -((2 * i) as f32) / d as f32;
                let freq = cfg.rope_theta.powf(exponent);
                pos as f32 * freq
            })
            .collect();
        // Duplicated, not interleaved: the graph rotates the first half against
        // the second, so position i and i + half share an angle.
        cos.extend(angles.iter().map(|a| a.cos()));
        cos.extend(angles.iter().map(|a| a.cos()));
        sin.extend(angles.iter().map(|a| a.sin()));
        sin.extend(angles.iter().map(|a| a.sin()));
    }
    (cos, sin)
}

/// Write both RoPE tables and return their offsets.
pub fn write_rope(weights: &mut Writer, cfg: &Config) -> (u64, u64) {
    let (cos, sin) = rope_tables(cfg);
    (
        weights.write_f32_as_fp16(&cos),
        weights.write_f32_as_fp16(&sin),
    )
}

/// `concat(-x2, x1)` over the last axis, where `x1` and `x2` are its halves.
/// Seven operations, emitted once for the query and once for the key.
fn rotate_half(
    b: &mut Builder,
    cfg: &Config,
    tag: &str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Tensor {
    let d = cfg.head_dim();
    let half = d / 2;
    let heads = cfg.heads;
    let seq = cfg.seq;
    let full = [1, heads, seq, d];
    let halved = [1, heads, seq, half];

    let straight = b.op(
        "mul",
        Tensor::new(format!("{tag}_cos"), DType::Fp16, &full),
        &[("x", x), ("y", cos)],
    );

    let begin_lo = b.const_i32(
        Tensor::new(format!("{tag}_lo_begin"), DType::Int32, &[4]),
        &[0, 0, 0, 0],
    );
    let end_lo = b.const_i32(
        Tensor::new(format!("{tag}_lo_end"), DType::Int32, &[4]),
        &[1, heads as i32, seq as i32, half as i32],
    );
    // `end_mask` says which axes ignore `end` and run to the end of the axis. The
    // reference masks all but the axis being cut, so only the last one is honoured.
    let mask_lo = b.const_bools(
        Tensor::new(format!("{tag}_lo_mask"), DType::Bool, &[4]),
        &[true, true, true, false],
    );
    let x1 = b.op(
        "slice_by_index",
        Tensor::new(format!("{tag}_x1"), DType::Fp16, &halved),
        &[
            ("x", x),
            ("begin", &begin_lo),
            ("end", &end_lo),
            ("end_mask", &mask_lo),
        ],
    );

    let begin_hi = b.const_i32(
        Tensor::new(format!("{tag}_hi_begin"), DType::Int32, &[4]),
        &[0, 0, 0, half as i32],
    );
    let end_hi = b.const_i32(
        Tensor::new(format!("{tag}_hi_end"), DType::Int32, &[4]),
        &[1, heads as i32, seq as i32, d as i32],
    );
    let mask_hi = b.const_bools(
        Tensor::new(format!("{tag}_hi_mask"), DType::Bool, &[4]),
        &[true, true, true, true],
    );
    let x2 = b.op(
        "slice_by_index",
        Tensor::new(format!("{tag}_x2"), DType::Fp16, &halved),
        &[
            ("x", x),
            ("begin", &begin_hi),
            ("end", &end_hi),
            ("end_mask", &mask_hi),
        ],
    );

    let minus_one = b.const_fp16(
        Tensor::new(format!("{tag}_neg1"), DType::Fp16, &[]),
        &[-1.0],
    );
    let negated = b.op(
        "mul",
        Tensor::new(format!("{tag}_negx2"), DType::Fp16, &halved),
        &[("x", &x2), ("y", &minus_one)],
    );

    let axis = b.const_i32(
        Tensor::new(format!("{tag}_cat_axis"), DType::Int32, &[]),
        &[-1],
    );
    let interleave = b.const_bool(&format!("{tag}_interleave"), false);
    let rotated = b.op_variadic(
        "concat",
        Tensor::new(format!("{tag}_rot"), DType::Fp16, &full),
        &[("axis", &axis), ("interleave", &interleave)],
        ("values", &[&negated, &x1]),
    );
    let crossed = b.op(
        "mul",
        Tensor::new(format!("{tag}_sin"), DType::Fp16, &full),
        &[("x", &rotated), ("y", sin)],
    );
    b.op(
        "add",
        Tensor::new(format!("{tag}_rope"), DType::Fp16, &full),
        &[("x", &straight), ("y", &crossed)],
    )
}

/// One transformer block. `input` is `[1, seq, hidden]`; the result is the next
/// block's input.
///
/// `mask` is `[1, 1, seq, seq]` additive: the local one for a sliding-window layer
/// and the global one every `global_attn_every_n_layers`. That single argument is
/// the only difference between layers.
#[allow(clippy::too_many_lines)] // A 38-operation block reads better in one piece.
pub fn block(
    b: &mut Builder,
    cfg: &Config,
    layer: usize,
    input: &Tensor,
    w: &BlockOffsets,
    mask: &Tensor,
    epsilon: &Tensor,
) -> Tensor {
    let attended = attention(b, cfg, layer, input, w, mask, epsilon);
    feed_forward(b, cfg, layer, &attended, w, epsilon)
}

/// Operations 1-16: the pre-norm, the fused QKV, RoPE, masked attention, the
/// output projection and the residual. The result is the feed-forward's input.
fn attention(
    b: &mut Builder,
    cfg: &Config,
    layer: usize,
    input: &Tensor,
    w: &BlockOffsets,
    mask: &Tensor,
    epsilon: &Tensor,
) -> Tensor {
    let (h, heads, seq) = (cfg.hidden, cfg.heads, cfg.seq);
    let d = cfg.head_dim();
    let tag = |s: &str| format!("l{layer}_{s}");
    let hidden_shape = [1, seq, h];
    let per_head = [1, heads, seq, d];

    // 1. attn_norm, or the caller's norm for layer 0.
    let normed = match w.attn_norm {
        None => input.clone(),
        Some(offset) => {
            let gamma = b.const_blob(Tensor::new(tag("attn_norm"), DType::Fp16, &[h]), offset);
            let axes = b.const_i32(
                Tensor::new(tag("attn_norm_axes"), DType::Int32, &[1]),
                &[-1],
            );
            b.op(
                "layer_norm",
                Tensor::new(tag("attn_normed"), DType::Fp16, &hidden_shape),
                &[
                    ("x", input),
                    ("axes", &axes),
                    ("epsilon", epsilon),
                    ("gamma", &gamma),
                ],
            )
        }
    };

    // 2-5. Fused QKV, then split into three [1, heads, seq, d].
    let wqkv = w
        .wqkv
        .bind_in(b, Tensor::new(tag("wqkv"), DType::Fp16, &[3 * h, h]));
    let wqkv_bias = b.const_blob(
        Tensor::new(tag("wqkv_bias"), DType::Fp16, &[3 * h]),
        w.wqkv_bias,
    );
    let qkv = b.op(
        "linear",
        Tensor::new(tag("qkv"), DType::Fp16, &[1, seq, 3 * h]),
        &[("x", &normed), ("weight", &wqkv), ("bias", &wqkv_bias)],
    );
    let shape5 = b.const_i32(
        Tensor::new(tag("qkv_shape"), DType::Int32, &[5]),
        &[1, -1, 3, heads as i32, d as i32],
    );
    let qkv5 = b.op(
        "reshape",
        Tensor::new(tag("qkv5"), DType::Fp16, &[1, seq, 3, heads, d]),
        &[("x", &qkv), ("shape", &shape5)],
    );
    let perm = b.const_i32(
        Tensor::new(tag("qkv_perm"), DType::Int32, &[5]),
        &[0, 3, 2, 1, 4],
    );
    let qkvt = b.op(
        "transpose",
        Tensor::new(tag("qkvt"), DType::Fp16, &[1, heads, 3, seq, d]),
        &[("x", &qkv5), ("perm", &perm)],
    );
    let split_axis = b.const_i32(Tensor::new(tag("qkv_split_axis"), DType::Int32, &[]), &[2]);
    let split_sizes = b.const_i32(
        Tensor::new(tag("qkv_split_sizes"), DType::Int32, &[3]),
        &[1, 1, 1],
    );
    let parts = b.op_multi_output(
        "split",
        &[
            Tensor::new(tag("qkv_q"), DType::Fp16, &[1, heads, 1, seq, d]),
            Tensor::new(tag("qkv_k"), DType::Fp16, &[1, heads, 1, seq, d]),
            Tensor::new(tag("qkv_v"), DType::Fp16, &[1, heads, 1, seq, d]),
        ],
        &[
            ("x", &qkvt),
            ("axis", &split_axis),
            ("split_sizes", &split_sizes),
        ],
    );
    let squeezed: Vec<Tensor> = ["q", "k", "v"]
        .iter()
        .zip(&parts)
        .map(|(name, part)| {
            let axes = b.const_i32(
                Tensor::new(tag(&format!("{name}_sq_axes")), DType::Int32, &[1]),
                &[2],
            );
            b.op(
                "squeeze",
                Tensor::new(tag(name), DType::Fp16, &per_head),
                &[("x", part), ("axes", &axes)],
            )
        })
        .collect();

    // 6-7. RoPE on the query and the key. Value is left alone.
    let cos = b.const_blob(
        Tensor::new(tag("rope_cos"), DType::Fp16, &[1, 1, seq, d]),
        w.rope_cos,
    );
    let sin = b.const_blob(
        Tensor::new(tag("rope_sin"), DType::Fp16, &[1, 1, seq, d]),
        w.rope_sin,
    );
    let query = rotate_half(b, cfg, &tag("q"), &squeezed[0], &cos, &sin);
    let key = rotate_half(b, cfg, &tag("k"), &squeezed[1], &cos, &sin);

    // 8-12. Scaled dot-product attention with the additive mask.
    let no = b.const_bool(&tag("false"), false);
    let yes = b.const_bool(&tag("true"), true);
    let scores = b.op(
        "matmul",
        Tensor::new(tag("scores"), DType::Fp16, &[1, heads, seq, seq]),
        &[
            ("x", &query),
            ("y", &key),
            ("transpose_x", &no),
            ("transpose_y", &yes),
        ],
    );
    let scale = b.const_fp16(Tensor::new(tag("scale"), DType::Fp16, &[]), &[cfg.scale()]);
    let scaled = b.op(
        "mul",
        Tensor::new(tag("scaled"), DType::Fp16, &[1, heads, seq, seq]),
        &[("x", &scores), ("y", &scale)],
    );
    let masked = b.op(
        "add",
        Tensor::new(tag("masked"), DType::Fp16, &[1, heads, seq, seq]),
        &[("x", &scaled), ("y", mask)],
    );
    let softmax_axis = b.const_i32(Tensor::new(tag("sm_axis"), DType::Int32, &[]), &[-1]);
    let probs = b.op(
        "softmax",
        Tensor::new(tag("probs"), DType::Fp16, &[1, heads, seq, seq]),
        &[("x", &masked), ("axis", &softmax_axis)],
    );
    let context = b.op(
        "matmul",
        Tensor::new(tag("context"), DType::Fp16, &per_head),
        &[
            ("x", &probs),
            ("y", &squeezed[2]),
            ("transpose_x", &no),
            ("transpose_y", &no),
        ],
    );

    // 13-16. Merge the heads and project out, then the residual.
    let out_perm = b.const_i32(
        Tensor::new(tag("out_perm"), DType::Int32, &[4]),
        &[0, 2, 1, 3],
    );
    let merged = b.op(
        "transpose",
        Tensor::new(tag("merged"), DType::Fp16, &[1, seq, heads, d]),
        &[("x", &context), ("perm", &out_perm)],
    );
    let flat_shape = b.const_i32(
        Tensor::new(tag("flat_shape"), DType::Int32, &[3]),
        &[1, -1, h as i32],
    );
    let flat = b.op(
        "reshape",
        Tensor::new(tag("flat"), DType::Fp16, &hidden_shape),
        &[("x", &merged), ("shape", &flat_shape)],
    );
    let wo =
        w.wo.bind_in(b, Tensor::new(tag("wo"), DType::Fp16, &[h, h]));
    let wo_bias = b.const_blob(Tensor::new(tag("wo_bias"), DType::Fp16, &[h]), w.wo_bias);
    let projected = b.op(
        "linear",
        Tensor::new(tag("attn_out"), DType::Fp16, &hidden_shape),
        &[("x", &flat), ("weight", &wo), ("bias", &wo_bias)],
    );
    b.op(
        "add",
        Tensor::new(tag("attn_residual"), DType::Fp16, &hidden_shape),
        &[("x", input), ("y", &projected)],
    )
}

/// Operations 17-23: the pre-norm, the gated feed-forward and its residual.
fn feed_forward(
    b: &mut Builder,
    cfg: &Config,
    layer: usize,
    input: &Tensor,
    w: &BlockOffsets,
    epsilon: &Tensor,
) -> Tensor {
    let (h, seq, inter) = (cfg.hidden, cfg.seq, cfg.intermediate);
    let tag = |s: &str| format!("l{layer}_{s}");
    let hidden_shape = [1, seq, h];

    let mlp_gamma = b.const_blob(Tensor::new(tag("mlp_norm"), DType::Fp16, &[h]), w.mlp_norm);
    let mlp_axes = b.const_i32(Tensor::new(tag("mlp_norm_axes"), DType::Int32, &[1]), &[-1]);
    let mlp_normed = b.op(
        "layer_norm",
        Tensor::new(tag("mlp_normed"), DType::Fp16, &hidden_shape),
        &[
            ("x", input),
            ("axes", &mlp_axes),
            ("epsilon", epsilon),
            ("gamma", &mlp_gamma),
        ],
    );
    let wi = w
        .mlp_wi
        .bind_in(b, Tensor::new(tag("mlp_wi"), DType::Fp16, &[2 * inter, h]));
    let wi_bias = b.const_blob(
        Tensor::new(tag("mlp_wi_bias"), DType::Fp16, &[2 * inter]),
        w.mlp_wi_bias,
    );
    let wide = b.op(
        "linear",
        Tensor::new(tag("mlp_wide"), DType::Fp16, &[1, seq, 2 * inter]),
        &[("x", &mlp_normed), ("weight", &wi), ("bias", &wi_bias)],
    );
    let geglu_axis = b.const_i32(Tensor::new(tag("geglu_axis"), DType::Int32, &[]), &[-1]);
    let geglu_sizes = b.const_i32(
        Tensor::new(tag("geglu_sizes"), DType::Int32, &[2]),
        &[inter as i32, inter as i32],
    );
    let halves = b.op_multi_output(
        "split",
        &[
            Tensor::new(tag("gate_in"), DType::Fp16, &[1, seq, inter]),
            Tensor::new(tag("up"), DType::Fp16, &[1, seq, inter]),
        ],
        &[
            ("x", &wide),
            ("axis", &geglu_axis),
            ("split_sizes", &geglu_sizes),
        ],
    );
    // GeLU takes a mode; SiLU takes only its input. Everything downstream is
    // shape-identical, so the two differ by this one op.
    let out = Tensor::new(tag("gate"), DType::Fp16, &[1, seq, inter]);
    let gate = match cfg.activation {
        Activation::Gelu => {
            let mode = b.const_str(&tag("gelu_mode"), "EXACT");
            b.op("gelu", out, &[("x", &halves[0]), ("mode", &mode)])
        }
        Activation::Silu => b.op("silu", out, &[("x", &halves[0])]),
    };
    let gated = b.op(
        "mul",
        Tensor::new(tag("gated"), DType::Fp16, &[1, seq, inter]),
        &[("x", &gate), ("y", &halves[1])],
    );
    let mlp_wo = w
        .mlp_wo
        .bind_in(b, Tensor::new(tag("mlp_wo"), DType::Fp16, &[h, inter]));
    let mlp_wo_bias = b.const_blob(
        Tensor::new(tag("mlp_wo_bias"), DType::Fp16, &[h]),
        w.mlp_wo_bias,
    );
    let mlp_out = b.op(
        "linear",
        Tensor::new(tag("mlp_out"), DType::Fp16, &hidden_shape),
        &[("x", &gated), ("weight", &mlp_wo), ("bias", &mlp_wo_bias)],
    );
    b.op(
        "add",
        Tensor::new(tag("out"), DType::Fp16, &hidden_shape),
        &[("x", input), ("y", &mlp_out)],
    )
}

/// Which positions a sliding-window layer may **not** attend to, `[1, 1, seq, seq]`
/// row-major.
///
/// A constant, because the window depends only on the sequence length; only the
/// padding part of a mask depends on the input. The reference model bakes exactly
/// this as a `bool` const and applies it with a `select`.
///
/// `window` is the total width, so a position reaches `window / 2` either side,
/// which is how ModernBERT defines `local_attention`.
pub fn window_condition(seq: usize, window: usize) -> Vec<bool> {
    let reach = window / 2;
    let mut out = vec![false; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            out[i * seq + j] = i.abs_diff(j) > reach;
        }
    }
    out
}

/// A sliding-window additive mask, `[1, 1, seq, seq]`: zero where a position may
/// attend, a large negative elsewhere.
///
/// `window` is the total width, so a position sees `window / 2` either side, which
/// is how ModernBERT's `local_attention` is defined. `None` gives the global mask,
/// which is all zeros for an unpadded input.
pub fn attention_mask(seq: usize, window: Option<usize>) -> Vec<f32> {
    // -10_000 rather than -inf: fp16 saturates, and softmax of an all -inf row is
    // NaN. The reference model uses a finite sentinel for the same reason.
    const BLOCKED: f32 = -10_000.0;
    let mut mask = vec![0.0f32; seq * seq];
    if let Some(window) = window {
        let reach = window / 2;
        for i in 0..seq {
            for j in 0..seq {
                if i.abs_diff(j) > reach {
                    mask[i * seq + j] = BLOCKED;
                }
            }
        }
    }
    mask
}

/// fp16 rounding applied to a slice, for computing a reference the same way the
/// graph will see its inputs.
pub fn to_fp16(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&v| f16::from_f32(v).to_f32()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_duplicate_the_angles_across_the_halves() {
        let cfg = Config {
            hidden: 8,
            heads: 2,
            intermediate: 8,
            seq: 3,
            eps: 1e-5,
            rope_theta: 10_000.0,
            activation: Activation::Gelu,
        };
        let d = cfg.head_dim();
        assert_eq!(d, 4);
        let (cos, sin) = rope_tables(&cfg);
        assert_eq!(cos.len(), cfg.seq * d);

        for pos in 0..cfg.seq {
            let row = &cos[pos * d..(pos + 1) * d];
            // The two halves carry the same angles, which is what lets the graph
            // rotate the first half against the second.
            assert_eq!(row[0], row[d / 2]);
            assert_eq!(row[1], row[d / 2 + 1]);
        }
        // Position 0 has angle 0 everywhere.
        assert!(cos[..d].iter().all(|&c| (c - 1.0).abs() < 1e-6));
        assert!(sin[..d].iter().all(|&s| s.abs() < 1e-6));
        // And position 1's first frequency is 1 radian.
        assert!((cos[d] - 1.0f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn the_window_condition_marks_what_cannot_be_attended() {
        let outside = window_condition(5, 4);
        let blocked = |i: usize, j: usize| outside[i * 5 + j];
        assert!(!blocked(2, 0) && !blocked(2, 4));
        assert!(blocked(0, 3));
        for i in 0..5 {
            assert!(!blocked(i, i), "a position must attend to itself");
        }
        // The additive mask and the condition must agree on what is blocked.
        let additive = attention_mask(5, Some(4));
        for (k, &out) in outside.iter().enumerate() {
            assert_eq!(out, additive[k] < 0.0, "disagreement at {k}");
        }
    }

    #[test]
    fn a_sliding_window_mask_reaches_half_the_window_each_way() {
        let mask = attention_mask(5, Some(4));
        let blocked = |i: usize, j: usize| mask[i * 5 + j] < 0.0;
        // Window 4 means two either side.
        assert!(!blocked(2, 0) && !blocked(2, 4));
        assert!(!blocked(0, 2));
        assert!(blocked(0, 3));
        assert!(blocked(4, 1));
        // Every position attends to itself.
        for i in 0..5 {
            assert!(!blocked(i, i));
        }
        // The global mask blocks nothing.
        assert!(attention_mask(5, None).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn the_scale_is_one_over_root_head_dim() {
        let cfg = Config {
            hidden: 512,
            heads: 8,
            intermediate: 2048,
            seq: 128,
            eps: 1e-5,
            rope_theta: 10_000.0,
            activation: Activation::Gelu,
        };
        assert_eq!(cfg.head_dim(), 64);
        // 0x1p-3 in the reference model.
        assert!((cfg.scale() - 0.125).abs() < 1e-9);
    }
}
