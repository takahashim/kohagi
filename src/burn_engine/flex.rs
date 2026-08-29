//! The burn-flex device: the CPU, on Burn instead of candle.
//!
//! burn-flex is Burn's own fast portable CPU backend — the same `gemm` crate
//! candle uses for its Linux GEMM, plus SIMD elementwise kernels through
//! `macerator`. It is not a CubeCL backend, so a `#[cube]` kernel cannot reach
//! it; what it offers instead is a documented way down to the buffer, which is
//! what the overrides below use.
//!
//! Measured on an 8-core Zen 4 against `--device cpu` on `ruri-v3-130m`, 16
//! rows of ~450 tokens, at `1 - cosine` 1.2e-12:
//!
//! | | rows/s |
//! | --- | ---: |
//! | one wide forward, generic kernels | 0.9 |
//! | fanned out across the pool | 5.7 |
//! | + two contiguous `Wi` matmuls | 6.1 |
//! | `--device cpu` (candle) | 6.8 |
//!
//! The fan-out is the whole difference and it is not Burn's to make: Burn
//! parallelises inside an operation, and that does not substitute for running
//! independent forwards at once. `Shape::fan_out` is what asks for it.
//!
//! ## macOS is a different comparison
//!
//! The sentence above about "the same `gemm` crate candle uses" holds on Linux
//! and is the reason the Zen 4 numbers come out close. It does not hold on
//! macOS: there `Cargo.toml` gives candle the `accelerate` feature, so its
//! sgemm is Apple's AMX coprocessor and not `gemm` at all. Against that,
//! burn-flex's NEON kernels are running a different race — 395 GFLOP/s to
//! Accelerate's 2072 at `2048x512x2048` on an M1 Pro — and `--device cpu-burn`
//! measured 3.0x slower end to end rather than the 1.1x Zen 4 shows.
//!
//! So on macOS this module does what candle's `accelerate` feature does, and
//! no more: it hands the GEMMs to Accelerate — [`FusedOps::project`],
//! [`FusedOps::scores`], [`FusedOps::context`]. The other overrides are about
//! not copying. burn-flex's `reshape` copies any view that is not contiguous
//! from offset zero, and does it one element at a time, so the head split is
//! built as a view ([`FusedOps::split_heads`]) and the merge as `memcpy`s
//! ([`FusedOps::merge_heads`]); RoPE reads through strides; and the mask add
//! takes its two masks separately ([`FusedOps::add_mask`]), burn-flex having no
//! fast path for a broadcast over the head axis.
//!
//! The ladder at 64 texts of 460 tokens, against `--device cpu`'s 2.73 s:
//! 8.24 s as found; 4.64 with `gemm`'s AMX kernels (since dropped — see
//! `Cargo.toml`); 3.73 with the mask add; 3.52 with RoPE reading strided; 3.13
//! with the projections on Accelerate; 3.06 with the head views; 2.86 with the
//! attention products on Accelerate, the masks unsummed and four rows per
//! unit. Over the range that is 0.74 s to candle's 0.81 at 64 × 42 tokens,
//! 2.86 to 2.73 at 460, 2.39 to 2.12 at 8 × 2048 and 9.42 to 8.83 at 4 × 8192
//! — ahead at the default `--max-seq-length`, 4 to 13% behind on long inputs.
//!
//! What is left is not in the kernels. The two engines' profiles agree to
//! within a few percent on every compute symbol, gelu's scalar `erf` included
//! (a floor under both on aarch64, and the largest item in each). The
//! difference is user time inside Accelerate's own GCD threading — 18.2 s to
//! 16.0 at 460 tokens — and neither the BLAS entry point (`sgemm_` as candle
//! binds it, against `cblas_sgemm`), the unit shape, nor a dedicated fan-out
//! pool moved it beyond noise.

use anyhow::Result;
use burn::backend::flex::{FlexTensor, Layout};
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

use super::{weights, BurnEncoder, Forward, FusedOps, Shape};
use crate::encoder::{Activation, Config};

/// f32 throughout. There is no half-precision path here: `--precision bf16` is
/// the hand-written candle kernel in `crate::bf16`, and f16 is the Vulkan
/// device's recipe.
type Cpu = burn::backend::Flex;

/// Four rows per forward at seq 512, twice what the candle CPU path allows
/// itself.
///
/// A forward holds `rows * heads * seq^2` scores, and narrow units load-balance
/// across the pool better than wide ones; but a unit is also one GEMM call per
/// projection, and on Accelerate each call brings its own threads up. Measured
/// at 64 rows of ~460 tokens with the projections on Accelerate: one row per
/// unit 3.12 s, two 3.02, **four 2.93**. With `gemm` doing the GEMMs it had
/// been the other way — 12.5 s at one row against 13.8 at two — which is the
/// per-call overhead moving from the kernel to the runtime underneath it.
const CPU_ATTN_BUDGET: usize = 4 * 512 * 512;

/// Rows a forward may hold whatever the budget allows.
///
/// The budget only bites on long inputs; on short ones it would let a whole
/// batch into one forward, and then nothing is fanned out and the pool sits
/// idle. This is what keeps the units coming. Measured on 64 texts of 42
/// tokens on `gemm`: 2.89 s uncapped, 1.92 at one row, 1.88 at two, 1.48 at
/// four, 1.50 at eight; and again on Accelerate, where eight edges four by a
/// hundredth of a second at 42 tokens and by 0.09 s at 460 — the same reason
/// as the budget above, larger calls being cheaper calls.
const MAX_ROWS_PER_FORWARD: usize = 8;

impl FusedOps for Cpu {
    /// Splitting `Wi` is worth +9% here: the wide form leaves gate and up as
    /// strided views, and the elementwise path walks those badly.
    const SPLIT_WI: bool = true;

    /// The projections on Accelerate, which is where candle's already are.
    ///
    /// This is the whole macOS gap. `Cargo.toml` gives candle the `accelerate`
    /// feature there, and Accelerate's sgemm runs on the AMX coprocessor at
    /// 2072 GFLOP/s where `gemm`'s own AMX kernels reach 1240 and its NEON ones
    /// 395 (all at 2048x512x2048 on an M1 Pro). burn-flex has no BLAS feature
    /// to reach it through, the way `burn-ndarray` does — so this reaches it
    /// directly, which is all candle's `accelerate` path does too.
    ///
    /// [`Self::scores`] and [`Self::context`] send the attention products the
    /// same way.
    #[cfg(target_os = "macos")]
    fn project(x: Tensor<Self, 3>, w: Tensor<Self, 2>) -> Tensor<Self, 3> {
        let [batch, seq, input] = x.dims();
        let output = w.dims()[1];
        let (xp, wp) = (
            contiguous(x.into_primitive().tensor()),
            contiguous(w.into_primitive().tensor()),
        );
        let (xs, ws) = (slice(&xp), slice(&wp));
        let m = batch * seq;
        let mut out = vec![0f32; m * output];
        // SAFETY: all three buffers are contiguous row-major at the leading
        // dimensions passed, and `out` is exactly `m * output` long.
        unsafe {
            accelerate::sgemm(
                m, output, input, xs, input, ws, output, false, &mut out, output,
            )
        };
        rebuild(out, [batch, seq, output])
    }

    /// The attention products on Accelerate too, read through their strides.
    ///
    /// These are `rows * heads` small matmuls with a 64-wide contraction, and
    /// `gemm`'s AMX kernel is not at its best there: measured one head at 460
    /// tokens, Accelerate is 1.9x faster on `q @ k^T` and 3.5x on `probs @ v`
    /// (208 and 735 GFLOP/s against 111 and 208). The operands are also all
    /// views — `q` and `k` of a banded block, `v` of the fused projection — and
    /// a leading dimension is what a view *is* to BLAS, so none of them is
    /// copied first.
    #[cfg(target_os = "macos")]
    fn scores(q: Tensor<Self, 4>, k: Tensor<Self, 4>) -> Tensor<Self, 4> {
        let [rows, heads, queries, head_dim] = q.dims();
        let keys = k.dims()[2];
        let (qp, kp) = (q.into_primitive().tensor(), k.into_primitive().tensor());
        let (Some(qa), Some(ka)) = (Planes::of(&qp), Planes::of(&kp)) else {
            return super::scores_composed(tensor(qp), tensor(kp));
        };
        let plane = queries * keys;
        let mut out = vec![0f32; rows * heads * plane];
        for r in 0..rows {
            for h in 0..heads {
                let c = &mut out[(r * heads + h) * plane..][..plane];
                // SAFETY: `Planes::of` checked both operands hold their planes
                // at the leading dimensions passed; `c` is `queries * keys`.
                unsafe {
                    accelerate::sgemm(
                        queries,
                        keys,
                        head_dim,
                        qa.plane(r, h),
                        qa.ld,
                        ka.plane(r, h),
                        ka.ld,
                        true,
                        c,
                        keys,
                    )
                };
            }
        }
        rebuild(out, [rows, heads, queries, keys])
    }

    /// See [`Self::scores`].
    #[cfg(target_os = "macos")]
    fn context(probs: Tensor<Self, 4>, v: Tensor<Self, 4>) -> Tensor<Self, 4> {
        let [rows, heads, queries, keys] = probs.dims();
        let head_dim = v.dims()[3];
        let (pp, vp) = (probs.into_primitive().tensor(), v.into_primitive().tensor());
        let (Some(pa), Some(va)) = (Planes::of(&pp), Planes::of(&vp)) else {
            return super::context_composed(tensor(pp), tensor(vp));
        };
        let plane = queries * head_dim;
        let mut out = vec![0f32; rows * heads * plane];
        for r in 0..rows {
            for h in 0..heads {
                let c = &mut out[(r * heads + h) * plane..][..plane];
                // SAFETY: as in `scores`.
                unsafe {
                    accelerate::sgemm(
                        queries,
                        head_dim,
                        keys,
                        pa.plane(r, h),
                        pa.ld,
                        va.plane(r, h),
                        va.ld,
                        false,
                        c,
                        head_dim,
                    )
                };
            }
        }
        rebuild(out, [rows, heads, queries, head_dim])
    }

    /// The view, built directly, so nothing is copied.
    ///
    /// `Layout::reshape` on burn-flex gives up on anything that is not
    /// contiguous from offset zero, and the narrowed slice of `qkv` is neither;
    /// composed, each of q, k and v therefore cost a scalar `StridedIter` copy
    /// of `[seq, hidden]` per layer. But the shape wanted here is only a
    /// re-labelling of the same storage — `[rows, heads, seq, head_dim]` at
    /// strides `[seq * width, head_dim, width, 1]` from column `part * hidden`
    /// — and everything downstream reads it through its strides: [`Self::rope`]
    /// walks them, and burn-flex's matmul takes strided operands.
    fn split_heads(qkv: Tensor<Self, 3>, part: usize, heads: usize) -> Tensor<Self, 4> {
        let [rows, seq, width] = qkv.dims();
        let hidden = width / 3;
        let head_dim = hidden / heads;
        let p = qkv.into_primitive().tensor();
        let (s, offset) = (p.layout().strides().to_vec(), p.layout().start_offset());
        if s[2] != 1 {
            let qkv = Tensor::from_primitive(TensorPrimitive::Float(p));
            return super::split_heads_composed(qkv, part, heads);
        }
        let view = p.with_layout(Layout::new(
            [rows, heads, seq, head_dim].into(),
            vec![s[0], head_dim as isize, s[1], 1],
            offset + part * hidden,
        ));
        Tensor::from_primitive(TensorPrimitive::Float(view))
    }

    /// The transpose as `head_dim`-wide `memcpy`s rather than element by element.
    ///
    /// Composed, this is `swap_dims` then `reshape`, and the reshape is the
    /// same scalar copy [`Self::split_heads`] avoids. The copy is owed here —
    /// the head axis really does move inside the sequence axis — but each
    /// `(row, head, position)` is a contiguous run of `head_dim` on both sides,
    /// so it is `rows * heads * seq` copies of 256 bytes, not `rows * hidden *
    /// seq` index computations.
    fn merge_heads(context: Tensor<Self, 4>) -> Tensor<Self, 3> {
        let [rows, heads, seq, head_dim] = context.dims();
        let p = context.into_primitive().tensor();
        let s: Vec<usize> = p.layout().strides().iter().map(|&s| s as usize).collect();
        if s[3] != 1 {
            let context = Tensor::from_primitive(TensorPrimitive::Float(p));
            return super::merge_heads_composed(context);
        }
        let base = p.layout().start_offset();
        let xs = p.storage::<f32>();
        let hidden = heads * head_dim;
        let mut out = vec![0f32; rows * seq * hidden];
        for r in 0..rows {
            for h in 0..heads {
                for q in 0..seq {
                    let src = base + r * s[0] + h * s[1] + q * s[2];
                    let dst = (r * seq + q) * hidden + h * head_dim;
                    out[dst..dst + head_dim].copy_from_slice(&xs[src..src + head_dim]);
                }
            }
        }
        rebuild(out, [rows, seq, hidden])
    }

    /// `burn_nn`'s RoPE builds a sign matrix and multiplies by it, then
    /// concatenates; candle-nn has a fused kernel and this is the gap. One pass
    /// over the buffer instead of seven measured 54x on the operation and +10%
    /// end to end.
    ///
    /// The head transpose comes free with it, because `x` is read where it lies.
    ///
    /// `x` reaches here as `[rows, seq, heads, head_dim]` with the middle two
    /// axes swapped, so asking for it contiguous first copies the whole tensor
    /// — and this kernel writes a fresh contiguous buffer anyway, so that copy
    /// bought nothing the loop below does not already do. Only the innermost
    /// axis has to be contiguous for the row slice, and the swap leaves it that
    /// way; anything else falls back rather than gathering element by element.
    ///
    /// Worth 3.73 s to 3.52 s on an M1 Pro at 64 texts of 460 tokens: one
    /// `[heads, seq, head_dim]` copy for each of `q` and `k`, in each of the
    /// nineteen layers.
    fn rope(x: Tensor<Self, 4>, cos: Tensor<Self, 4>, sin: Tensor<Self, 4>) -> Tensor<Self, 4> {
        let [rows, heads, seq, hd] = x.dims();
        let half = hd / 2;
        let (cp, sp) = (
            contiguous(cos.into_primitive().tensor()),
            contiguous(sin.into_primitive().tensor()),
        );
        let (cs, ss) = (slice(&cp), slice(&sp));

        let xp = x.into_primitive().tensor();
        let unit_last = xp.layout().strides()[3] == 1;
        let xp = if unit_last { xp } else { xp.to_contiguous() };
        let stride: Vec<usize> = xp.layout().strides().iter().map(|&s| s as usize).collect();
        let base = xp.layout().start_offset();
        let xs = xp.storage::<f32>();

        let mut out = vec![0f32; rows * heads * seq * hd];
        for r in 0..rows {
            for h in 0..heads {
                for p in 0..seq {
                    let src = base + r * stride[0] + h * stride[1] + p * stride[2];
                    let xrow = &xs[src..src + hd];
                    let o = ((r * heads + h) * seq + p) * hd;
                    let (dst, t) = (&mut out[o..o + hd], p * hd);
                    for i in 0..half {
                        let (a, b) = (xrow[i], xrow[i + half]);
                        dst[i] = a * cs[t + i] - b * ss[t + i];
                        dst[i + half] = b * cs[t + i + half] + a * ss[t + i + half];
                    }
                }
            }
        }
        rebuild(out, [rows, heads, seq, hd])
    }

    /// The mask add, on contiguous planes rather than through the broadcast.
    ///
    /// burn-flex accelerates exactly two broadcast shapes — a shared innermost
    /// row, and a per-row scalar — and this is neither: the masks repeat over
    /// the *head* axis, which is not the innermost one. Everything else falls to
    /// `binary_op_typed`, a scalar `StridedIter` walk over
    /// `[rows, heads, queries, keys]`, the widest tensor the model builds.
    ///
    /// The broadcast is a plane repeat rather than an element one, so it does
    /// not need burn to see it as one: each head's scores are a contiguous
    /// `queries * keys` block, and each mask contributes a contiguous row of
    /// `keys` to every row of it. So the add is a flat loop the compiler
    /// vectorises on any target, in place, since the scores are a fresh matmul
    /// result nothing else holds — and it takes the two masks separately, so
    /// their sum is never materialised. Composed, that sum was its own scalar
    /// walk over `[rows, 1, queries, keys]` in each sliding-window layer.
    ///
    /// Worth 4.64 s to 3.73 s on an M1 Pro at 64 texts of 460 tokens, as one
    /// mask. Twelve of nineteen layers slide a window, and those are the ones
    /// whose mask carries a query axis and so misses the shared-row path even
    /// at one row.
    fn add_mask(
        scores: Tensor<Self, 4>,
        pad: Tensor<Self, 4>,
        window: Option<Tensor<Self, 4>>,
    ) -> Tensor<Self, 4> {
        let [rows, heads, queries, keys] = scores.dims();
        // The encoder builds `[rows, 1, 1, keys]` and `[1, 1, queries, keys]`
        // and nothing else; anything that is not that goes back through burn.
        let shapes_ok = pad.dims() == [rows, 1, 1, keys]
            && window
                .as_ref()
                .is_none_or(|w| w.dims() == [1, 1, queries, keys]);
        if !shapes_ok {
            return super::add_mask_composed(scores, pad, window);
        }
        let pp = pad.into_primitive().tensor();
        let wp = window.map(|w| w.into_primitive().tensor());
        let unit_inner = |t: &FlexTensor| t.layout().strides()[3] == 1;
        if !unit_inner(&pp) || wp.as_ref().is_some_and(|w| !unit_inner(w)) {
            return super::add_mask_composed(scores, tensor(pp), wp.map(tensor));
        }
        let (ps, pbase, pv) = (
            strides(&pp),
            pp.layout().start_offset(),
            pp.storage::<f32>(),
        );
        let win = wp
            .as_ref()
            .map(|w| (strides(w), w.layout().start_offset(), w.storage::<f32>()));

        let mut sp = contiguous(scores.into_primitive().tensor());
        let base = sp
            .layout()
            .contiguous_offsets()
            .expect("contiguous scores")
            .0;
        let ss = &mut sp.storage_mut::<f32>()[base..];

        let plane = queries * keys;
        for r in 0..rows {
            let prow = &pv[pbase + r * ps[0]..][..keys];
            for h in 0..heads {
                let s_plane = &mut ss[(r * heads + h) * plane..][..plane];
                for (i, s_row) in s_plane.chunks_exact_mut(keys).enumerate() {
                    match &win {
                        Some((ws, wbase, wv)) => {
                            let wrow = &wv[wbase + i * ws[2]..][..keys];
                            for ((s, p), w) in s_row.iter_mut().zip(prow).zip(wrow) {
                                *s += *p + *w;
                            }
                        }
                        None => {
                            for (s, p) in s_row.iter_mut().zip(prow) {
                                *s += *p;
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_primitive(TensorPrimitive::Float(sp))
    }

    /// The vectorised GeGLU this crate already owns, from the bf16 module — it
    /// is f32 in and f32 out and has nothing to do with bf16 beyond living
    /// beside the path that needed it first. Falls back to burn-flex's own
    /// gelu-then-multiply without AVX-512, which is faster than that kernel's
    /// own scalar path.
    ///
    /// Writing a *scalar* fused kernel here was tried and measured: it changes
    /// nothing, and one pass of scalar code loses more on vectorisation than it
    /// wins on memory traffic. Only a vector kernel is worth the escape hatch.
    ///
    /// The multiply it falls back to is SIMD; the `gelu` is not, on any target.
    /// `burn_flex::ops::activation::gelu` is a scalar closure through
    /// `unary_op` calling `erf` per element, and on aarch64 — where
    /// [`Avx512::detect`](crate::bf16::simd::Avx512::detect) can never succeed
    /// and this override therefore never fires — it is the largest compute item
    /// in the profile, ahead of any one GEMM. It is not a *gap*, though: candle's
    /// `gelu_erf` is scalar on aarch64 too, and takes a larger share of its
    /// profile than this does of Burn's. Both engines want the same vectorised
    /// kernel, which is why fixing it does not belong here.
    fn geglu(gate: Tensor<Self, 3>, up: Tensor<Self, 3>, act: Activation) -> Tensor<Self, 3> {
        // Gated on the vector path actually existing: without AVX-512
        // `crate::bf16::geglu` falls back to scalar rows, and burn-flex's own
        // gelu-then-multiply beats that. The escape hatch is only worth taking
        // when it lands somewhere better.
        #[cfg(target_arch = "x86_64")]
        if act == Activation::Gelu && crate::bf16::simd::Avx512::detect().is_some() {
            let dims = gate.dims();
            let inter = dims[2];
            let rows = dims[0] * dims[1];
            let (gp, up_) = (
                contiguous(gate.into_primitive().tensor()),
                contiguous(up.into_primitive().tensor()),
            );
            return rebuild(
                crate::bf16::geglu::geglu_split(slice(&gp), slice(&up_), rows, inter),
                [dims[0], dims[1], inter],
            );
        }
        super::geglu_composed(gate, up, act)
    }
}

fn contiguous(t: FlexTensor) -> FlexTensor {
    if t.is_contiguous() {
        t
    } else {
        t.to_contiguous()
    }
}

/// `FlexTensor::as_slice` is `bytemuck::cast_slice` over the tensor's own
/// storage — a view, not a copy, which is what makes descending to a kernel
/// worth doing at all.
fn slice(t: &FlexTensor) -> &[f32] {
    t.as_slice::<f32>().expect("contiguous f32 storage")
}

/// Back to a tensor, for handing a primitive to a composed fallback.
fn tensor<const D: usize>(p: FlexTensor) -> Tensor<Cpu, D> {
    Tensor::from_primitive(TensorPrimitive::Float(p))
}

/// The layout's strides as element counts. Nothing here flips an axis, so a
/// negative stride is a bug rather than a case.
fn strides(t: &FlexTensor) -> Vec<usize> {
    t.layout()
        .strides()
        .iter()
        .map(|&s| usize::try_from(s).expect("no flipped axes"))
        .collect()
}

/// A `[rows, heads, m, n]` tensor as `rows * heads` row-major matrices, which
/// is what BLAS reads: a base offset per plane and one leading dimension.
///
/// Any view whose innermost axis is contiguous qualifies, whatever the other
/// strides do — a narrowed block, or the q/k/v slice of the fused projection.
#[cfg(target_os = "macos")]
struct Planes<'a> {
    data: &'a [f32],
    base: usize,
    row_stride: usize,
    head_stride: usize,
    /// The leading dimension: elements between one row of a plane and the next.
    ld: usize,
    m: usize,
    n: usize,
}

#[cfg(target_os = "macos")]
impl<'a> Planes<'a> {
    fn of(t: &'a FlexTensor) -> Option<Self> {
        let s = strides(t);
        if s[3] != 1 {
            return None;
        }
        let dims: [usize; 4] = t.layout().shape().dims();
        Some(Planes {
            data: t.storage::<f32>(),
            base: t.layout().start_offset(),
            row_stride: s[0],
            head_stride: s[1],
            ld: s[2],
            m: dims[2],
            n: dims[3],
        })
    }

    fn plane(&self, r: usize, h: usize) -> &'a [f32] {
        let start = self.base + r * self.row_stride + h * self.head_stride;
        &self.data[start..start + (self.m - 1) * self.ld + self.n]
    }
}

fn rebuild<const D: usize>(values: Vec<f32>, shape: [usize; D]) -> Tensor<Cpu, D> {
    Tensor::from_primitive(TensorPrimitive::Float(FlexTensor::from_data(
        TensorData::new(values, shape),
    )))
}

/// Apple's BLAS, for [`FusedOps::project`].
///
/// The framework is already on the link line: candle's `accelerate-src` puts it
/// there for the candle CPU path. The `#[link]` here is so this module still
/// builds the day that dependency goes, which is the point of the Burn engine.
///
/// `cblas_sgemm` rather than the Fortran `sgemm_` candle binds, because it takes
/// a row-major flag and so needs no transposition argument juggling; both are
/// plain exports of `libBLAS` and neither is the `$NEWLAPACK` alias that candle
/// notes as failing to link.
#[cfg(target_os = "macos")]
mod accelerate {
    use std::os::raw::{c_float, c_int};

    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
    }

    /// `c[m, n] = a[m, k] @ b[k, n]`, all row-major with the given leading
    /// dimensions — or, with `b_transposed`, `a[m, k] @ b[n, k]^T`, which is
    /// how `q @ k^T` reads `k` as it lies.
    ///
    /// # Safety
    ///
    /// `a` must hold `m` rows of `k` at stride `lda`; `b` `k` rows of `n` (or
    /// `n` rows of `k` when transposed) at stride `ldb`; and `c` `m` rows of
    /// `n` at stride `ldc`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        b_transposed: bool,
        c: &mut [f32],
        ldc: usize,
    ) {
        debug_assert!(a.len() >= (m - 1) * lda + k);
        debug_assert!(if b_transposed {
            b.len() >= (n - 1) * ldb + k
        } else {
            b.len() >= (k - 1) * ldb + n
        });
        debug_assert!(c.len() >= (m - 1) * ldc + n);
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            if b_transposed { TRANS } else { NO_TRANS },
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a.as_ptr(),
            lda as c_int,
            b.as_ptr(),
            ldb as c_int,
            0.0,
            c.as_mut_ptr(),
            ldc as c_int,
        )
    }
}

/// Load a checkpoint for the CPU.
pub fn load(weights: &std::path::Path, config: &Config) -> Result<BurnEncoder> {
    let checkpoint = weights::Checkpoint::open(weights)?;
    let device = Default::default();
    let model: Box<dyn Forward + Send + Sync> = Box::new(weights::load::<Cpu>(
        &checkpoint,
        config,
        false,
        CPU_ATTN_BUDGET,
        &device,
    )?);
    Ok(BurnEncoder {
        model,
        dim: config.hidden_size,
        shape: Shape {
            budget: CPU_ATTN_BUDGET,
            fan_out: true,
            max_rows: MAX_ROWS_PER_FORWARD,
        },
    })
}
