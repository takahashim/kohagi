//! Does a ModernBERT encoder forward run on this machine through Burn/CubeCL/Vulkan,
//! does it produce the same numbers as the CPU, and is it faster?
//!
//! Shapes are ruri-v3-130m's own config: hidden 512, 19 layers, 8 heads,
//! intermediate 2048, local_attention 128, global every 3rd layer, GeGLU.

use std::time::Instant;

use burn::tensor::{activation, backend::Backend, FloatDType, Tensor, TensorData};

#[derive(Clone, Copy)]
struct Config {
    hidden: usize,
    layers: usize,
    heads: usize,
    inter: usize,
    local_attention: usize,
    global_every: usize,
    eps: f64,
    /// Projections in the backend's low precision, everything else forced to f32 —
    /// kohagi's own bf16 recipe (docs/devices.md).
    mixed: u8,
}

impl Config {
    fn ruri130m() -> Self {
        Config { hidden: 512, layers: 19, heads: 8, inter: 2048, local_attention: 128, global_every: 3, eps: 1e-5, mixed: 0 }
    }
    fn head_dim(&self) -> usize { self.hidden / self.heads }
}

/// Deterministic weights, so both backends see bit-identical inputs.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((self.0 >> 33) as f32) / (u32::MAX >> 1) as f32 - 1.0) * 0.05
    }
    fn vec(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.next_f32()).collect() }
}

struct Weights {
    // per layer: attn_norm gamma/beta, Wqkv, Wo, mlp_norm gamma/beta, Wi, Wo2
    layers: Vec<LayerW>,
    final_gamma: Vec<f32>,
    final_beta: Vec<f32>,
}
struct LayerW {
    an_g: Vec<f32>, an_b: Vec<f32>,
    wqkv: Vec<f32>, wo: Vec<f32>,
    mn_g: Vec<f32>, mn_b: Vec<f32>,
    wi: Vec<f32>, wo2: Vec<f32>,
}

fn make_weights(cfg: &Config, seed: u64) -> Weights {
    let mut r = Lcg(seed);
    let h = cfg.hidden;
    let layers = (0..cfg.layers).map(|_| LayerW {
        an_g: vec![1.0; h], an_b: vec![0.0; h],
        wqkv: r.vec(h * 3 * h), wo: r.vec(h * h),
        mn_g: vec![1.0; h], mn_b: vec![0.0; h],
        wi: r.vec(h * 2 * cfg.inter), wo2: r.vec(cfg.inter * h),
    }).collect();
    Weights { layers, final_gamma: vec![1.0; h], final_beta: vec![0.0; h] }
}

fn t2<B: Backend>(v: &[f32], r: usize, c: usize, d: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), d)
}
fn t1<B: Backend>(v: &[f32], d: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(v.to_vec(), [v.len()]), d)
}

/// LayerNorm over the last axis.
fn layer_norm<B: Backend>(x: Tensor<B, 3>, g: Tensor<B, 1>, b: Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let [_bs, _s, h] = x.dims();
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    let normed = centered / (var + eps).sqrt();
    normed * g.reshape([1, 1, h]) + b.reshape([1, 1, h])
}

/// Level 2: the reduction runs in f32, the tensor it returns is back in `dt`.
fn layer_norm_f32<B: Backend>(x: Tensor<B, 3>, g: Tensor<B, 1>, b: Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let dt = x.dtype();
    let y = layer_norm(x.cast(FloatDType::F32), g.cast(FloatDType::F32), b.cast(FloatDType::F32), eps);
    y.cast(dt)
}

/// `concat(-x2, x1)` over the last axis.
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, _, hd] = x.dims();
    let half = hd / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);
    Tensor::cat(vec![-x2, x1], 3)
}

fn rope<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    x.clone() * cos + rotate_half(x) * sin
}

/// `[1, 1, seq, head_dim]`, angles duplicated across the two halves.
fn rope_tables<B: Backend>(seq: usize, hd: usize, theta: f32, d: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = hd / 2;
    let mut cos = vec![0f32; seq * hd];
    let mut sin = vec![0f32; seq * hd];
    for p in 0..seq {
        for i in 0..half {
            let inv = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
            let a = p as f32 * inv;
            cos[p * hd + i] = a.cos();
            cos[p * hd + i + half] = a.cos();
            sin[p * hd + i] = a.sin();
            sin[p * hd + i + half] = a.sin();
        }
    }
    (
        Tensor::from_data(TensorData::new(cos, [1, 1, seq, hd]), d),
        Tensor::from_data(TensorData::new(sin, [1, 1, seq, hd]), d),
    )
}

/// Additive mask `[1, 1, seq, seq]`: 0 where a position may attend, BLOCKED where not.
/// Finite, not -inf: a fully-masked row must not make softmax produce NaN.
const BLOCKED: f32 = -1.0e4;
fn attn_mask<B: Backend>(seq: usize, window: Option<usize>, d: &B::Device) -> Tensor<B, 4> {
    let mut m = vec![0f32; seq * seq];
    if let Some(w) = window {
        let reach = w / 2;
        for q in 0..seq {
            for k in 0..seq {
                if (q as isize - k as isize).unsigned_abs() > reach {
                    m[q * seq + k] = BLOCKED;
                }
            }
        }
    }
    Tensor::from_data(TensorData::new(m, [1, 1, seq, seq]), d)
}

/// Everything that lives on the device across forwards: weights, RoPE tables, masks.
struct Resident<B: Backend> {
    layers: Vec<LayerT<B>>,
    final_g: Tensor<B, 1>,
    final_b: Tensor<B, 1>,
    cos_g: Tensor<B, 4>, sin_g: Tensor<B, 4>,
    cos_l: Tensor<B, 4>, sin_l: Tensor<B, 4>,
    mask_g: Tensor<B, 4>, mask_l: Tensor<B, 4>,
}
struct LayerT<B: Backend> {
    an_g: Tensor<B, 1>, an_b: Tensor<B, 1>,
    wqkv: Tensor<B, 2>, wo: Tensor<B, 2>,
    mn_g: Tensor<B, 1>, mn_b: Tensor<B, 1>,
    wi: Tensor<B, 2>, wo2: Tensor<B, 2>,
}

fn upload<B: Backend>(cfg: &Config, w: &Weights, seq: usize, dev: &B::Device) -> Resident<B> {
    let h = cfg.hidden;
    let hd = cfg.head_dim();
    let (cos_g, sin_g) = rope_tables::<B>(seq, hd, 160_000.0, dev);
    let (cos_l, sin_l) = rope_tables::<B>(seq, hd, 10_000.0, dev);
    let f32ify1 = |t: Tensor<B, 1>| if cfg.mixed == 1 { t.cast(FloatDType::F32) } else { t };
    let f32ify4 = |t: Tensor<B, 4>| if cfg.mixed == 1 { t.cast(FloatDType::F32) } else { t };
    Resident {
        layers: w.layers.iter().map(|lw| LayerT {
            an_g: f32ify1(t1::<B>(&lw.an_g, dev)), an_b: f32ify1(t1::<B>(&lw.an_b, dev)),
            wqkv: t2::<B>(&lw.wqkv, h, 3 * h, dev), wo: t2::<B>(&lw.wo, h, h, dev),
            mn_g: f32ify1(t1::<B>(&lw.mn_g, dev)), mn_b: f32ify1(t1::<B>(&lw.mn_b, dev)),
            wi: t2::<B>(&lw.wi, h, 2 * cfg.inter, dev), wo2: t2::<B>(&lw.wo2, cfg.inter, h, dev),
        }).collect(),
        final_g: f32ify1(t1::<B>(&w.final_gamma, dev)), final_b: f32ify1(t1::<B>(&w.final_beta, dev)),
        cos_g: f32ify4(cos_g), sin_g: f32ify4(sin_g), cos_l: f32ify4(cos_l), sin_l: f32ify4(sin_l),
        mask_g: f32ify4(attn_mask::<B>(seq, None, dev)),
        mask_l: f32ify4(attn_mask::<B>(seq, Some(cfg.local_attention), dev)),
    }
}

/// `[b, s, in] @ [in, out]` as one 2-D matmul, which is what the GEMM wants.
fn linear<B: Backend>(cfg: &Config, x: Tensor<B, 3>, w: Tensor<B, 2>) -> Tensor<B, 3> {
    let [bs, seq, h] = x.dims();
    let [_, out] = w.dims();
    let x = x.reshape([bs * seq, h]);
    if cfg.mixed == 1 {
        // Down to the weight's dtype for the GEMM, back to f32 for everything after it.
        let x = x.cast(w.dtype());
        x.matmul(w).cast(FloatDType::F32).reshape([bs, seq, out])
    } else {
        x.matmul(w).reshape([bs, seq, out])
    }
}

fn encoder<B: Backend>(cfg: &Config, r: &Resident<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let x = if cfg.mixed == 1 { x.cast(FloatDType::F32) } else { x };
    let [bs, seq, h] = x.dims();
    let hd = cfg.head_dim();
    let scale = 1.0 / (hd as f64).sqrt();

    let mut x = x;
    for (i, lw) in r.layers.iter().enumerate() {
        let global = i % cfg.global_every == 0;
        let (cos, sin, mask) = if global {
            (r.cos_g.clone(), r.sin_g.clone(), r.mask_g.clone())
        } else {
            (r.cos_l.clone(), r.sin_l.clone(), r.mask_l.clone())
        };

        // --- attention ---
        let n = if cfg.mixed == 2 { layer_norm_f32(x.clone(), lw.an_g.clone(), lw.an_b.clone(), cfg.eps) }
                else { layer_norm(x.clone(), lw.an_g.clone(), lw.an_b.clone(), cfg.eps) };
        let qkv = linear(cfg, n, lw.wqkv.clone());
        let split = |o: usize| {
            qkv.clone().narrow(2, o * h, h).reshape([bs, seq, cfg.heads, hd]).swap_dims(1, 2)
        };
        let q = rope(split(0), cos.clone(), sin.clone());
        let k = rope(split(1), cos, sin);
        let v = split(2);

        let raw = q.matmul(k.swap_dims(2, 3));
        let probs = if cfg.mixed == 2 {
            let dt = raw.dtype();
            let s32 = raw.cast(FloatDType::F32) * scale + mask.cast(FloatDType::F32);
            activation::softmax(s32, 3).cast(dt)
        } else {
            activation::softmax(raw * scale + mask, 3)
        };
        let ctx = probs.matmul(v).swap_dims(1, 2).reshape([bs, seq, h]);
        let attn_out = linear(cfg, ctx, lw.wo.clone());
        x = x + attn_out;

        // --- gated feed-forward (GeGLU) ---
        let n = if cfg.mixed == 2 { layer_norm_f32(x.clone(), lw.mn_g.clone(), lw.mn_b.clone(), cfg.eps) }
                else { layer_norm(x.clone(), lw.mn_g.clone(), lw.mn_b.clone(), cfg.eps) };
        let gu = linear(cfg, n, lw.wi.clone());
        let gate = gu.clone().narrow(2, 0, cfg.inter);
        let up = gu.narrow(2, cfg.inter, cfg.inter);
        let ff = activation::gelu(gate) * up;
        x = x + linear(cfg, ff, lw.wo2.clone());
    }
    if cfg.mixed == 2 { layer_norm_f32(x, r.final_g.clone(), r.final_b.clone(), cfg.eps) }
    else { layer_norm(x, r.final_g.clone(), r.final_b.clone(), cfg.eps) }
}

fn input_data(bs: usize, seq: usize, h: usize, seed: u64) -> TensorData {
    let mut r = Lcg(seed);
    TensorData::new(r.vec(bs * seq * h), [bs, seq, h])
}

/// Mean-pool then compare, which is what kohagi actually emits.
fn pooled(t: TensorData) -> Vec<f32> {
    let v: Vec<f32> = t.to_vec().unwrap();
    v
}

fn compare(a: &[f32], b: &[f32]) -> (f32, f64) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    (max, dot / (na.sqrt() * nb.sqrt()))
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let cfg = Config::ruri130m();

    if mode == "all" || mode.starts_with("correctness") {
        let small = Config { layers: 3, ..cfg };
        let (bs, seq) = (1usize, 128usize);
        let w = make_weights(&small, 42);
        let data = input_data(bs, seq, small.hidden, 7);

        fn run<B: Backend>(cfg: &Config, w: &Weights, data: TensorData, seq: usize) -> (Vec<f32>, String) {
            let d = B::Device::default();
            let r = upload::<B>(cfg, w, seq, &d);
            let out = encoder::<B>(cfg, &r, Tensor::from_data(data, &d)).into_data();
            let dt = format!("{:?}", out.dtype);
            (out.convert::<f32>().to_vec().unwrap(), dt)
        }

        // One GPU backend per process: CubeCL keeps a single client per device, so a
        // second backend on the same device inherits the first one's element type.
        let (cpu, _) = run::<burn::backend::NdArray>(&small, &w, data.clone(), seq);
        let (gpu, dg) = match mode.as_str() {
            "correctness16" => run::<burn::backend::Vulkan<burn::tensor::f16, i32>>(&small, &w, data, seq),
            "correctnessmixed2" => {
                let m = Config { mixed: 2, ..small };
                run::<burn::backend::Vulkan<burn::tensor::f16, i32>>(&m, &w, data, seq)
            }
            "correctnessmixed" => {
                let m = Config { mixed: 1, ..small };
                run::<burn::backend::Vulkan<burn::tensor::f16, i32>>(&m, &w, data, seq)
            }
            "correctnessbf16" => run::<burn::backend::Vulkan<burn::tensor::bf16, i32>>(&small, &w, data, seq),
            _ => run::<burn::backend::Vulkan>(&small, &w, data, seq),
        };
        let (max, cos) = compare(&cpu, &gpu);
        println!("vulkan[{dg}] vs cpu f32  layers={} seq={seq} : max|diff| = {:.3e}  cosine = {:.9}  1-cos = {:.3e}",
                 small.layers, max, cos, 1.0 - cos);
    }

    if mode == "all" || mode == "bench" {
        let bs: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(8);
        let prec = std::env::args().nth(3).unwrap_or_else(|| "f32".into());
        let seq = 512usize;
        let w = make_weights(&cfg, 42);
        let data = input_data(bs, seq, cfg.hidden, 7);

        fn go<B: Backend>(cfg: &Config, w: &Weights, data: TensorData, bs: usize, seq: usize, tag: &str) {
            let d = B::Device::default();
            let tu = Instant::now();
            let res = upload::<B>(cfg, w, seq, &d);
            let up = tu.elapsed();

            let t0 = Instant::now();
            let out = encoder::<B>(cfg, &res, Tensor::from_data(data.clone(), &d));
            let dt = out.into_data();
            let first = t0.elapsed();

            let mut best = std::time::Duration::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                let out = encoder::<B>(cfg, &res, Tensor::from_data(data.clone(), &d));
                let _ = out.into_data();
                best = best.min(t.elapsed());
            }
            println!("{tag:10} bs={bs} seq={seq}  out_dtype={:?}  upload={:?}  first={:?}  warm_best={:?}  ({:.1} rows/s)",
                     dt.dtype, up, first, best, bs as f64 / best.as_secs_f64());
        }

        match prec.as_str() {
            "f16" => go::<burn::backend::Vulkan<burn::tensor::f16, i32>>(&cfg, &w, data, bs, seq, "vulkan f16"),
            "bf16" => go::<burn::backend::Vulkan<burn::tensor::bf16, i32>>(&cfg, &w, data, bs, seq, "vulkan bf16"),
            "mixed2" => {
                let m = Config { mixed: 2, ..cfg };
                go::<burn::backend::Vulkan<burn::tensor::f16, i32>>(&m, &w, data, bs, seq, "vulkan mix2")
            }
            "mixed" => {
                let m = Config { mixed: 1, ..cfg };
                go::<burn::backend::Vulkan<burn::tensor::f16, i32>>(&m, &w, data, bs, seq, "vulkan mix")
            }
            _ => go::<burn::backend::Vulkan>(&cfg, &w, data, bs, seq, "vulkan f32"),
        }
    }
}
