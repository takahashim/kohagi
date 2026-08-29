//! Phase 0/1: run the real ruri-v3-130m weights through Burn/Vulkan and compare
//! the resulting embeddings against kohagi's own CPU f32 output.
//!
//! Architecture mirrors src/encoder.rs exactly: layer 0 has no attn_norm, every
//! LayerNorm is weight-only (norm_bias=false), every projection is bias-free,
//! layer i attends globally iff i % global_attn_every_n_layers == 0, the sliding
//! window reaches local_attention/2 either side, and the gated MLP activates the
//! *first* half of Wi with erf-GELU.

use std::collections::HashMap;
use std::time::Instant;

use burn::tensor::{activation, backend::Backend, FloatDType, Int, Tensor, TensorData};

// ---------------------------------------------------------------- config

#[derive(Clone, Copy)]
struct Cfg {
    hidden: usize,
    layers: usize,
    heads: usize,
    inter: usize,
    local_attention: usize,
    global_every: usize,
    global_theta: f32,
    local_theta: f32,
    eps: f64,
    /// 0 = backend dtype throughout, 2 = f16 stream with f32 reductions.
    mixed: u8,
}

impl Cfg {
    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
    fn read(path: &str) -> Cfg {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let g = |k: &str| v[k].as_u64().unwrap() as usize;
        let f = |k: &str| v[k].as_f64().unwrap();
        Cfg {
            hidden: g("hidden_size"),
            layers: g("num_hidden_layers"),
            heads: g("num_attention_heads"),
            inter: g("intermediate_size"),
            local_attention: g("local_attention"),
            global_every: g("global_attn_every_n_layers"),
            global_theta: f("global_rope_theta") as f32,
            local_theta: f("local_rope_theta") as f32,
            eps: v["norm_eps"].as_f64().or_else(|| v["layer_norm_eps"].as_f64()).unwrap(),
            mixed: 0,
        }
    }
}

// ---------------------------------------------------------------- safetensors

/// Minimal reader. burn-store is the real answer for kohagi; this keeps Phase 0
/// about the numerics rather than about a name-mapping layer.
struct Safetensors {
    bytes: Vec<u8>,
    index: HashMap<String, (Vec<usize>, usize, usize)>,
}

impl Safetensors {
    fn open(path: &str) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        let mut index = HashMap::new();
        for (k, v) in hdr.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            assert_eq!(v["dtype"], "F32", "{k} is not F32");
            let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|s| s.as_u64().unwrap() as usize).collect();
            let off = v["data_offsets"].as_array().unwrap();
            let a = off[0].as_u64().unwrap() as usize + 8 + n;
            let b = off[1].as_u64().unwrap() as usize + 8 + n;
            index.insert(k.clone(), (shape, a, b));
        }
        Safetensors { bytes, index }
    }

    fn raw(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (shape, a, b) = self.index.get(name).unwrap_or_else(|| panic!("missing {name}"));
        let v = self.bytes[*a..*b].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        (shape.clone(), v)
    }

    /// A `[out, in]` Linear weight, transposed on the host to `[in, out]` so the
    /// GEMM sees a contiguous right-hand side.
    fn linear<B: Backend>(&self, name: &str, dev: &B::Device) -> Tensor<B, 2> {
        let (shape, v) = self.raw(name);
        let (o, i) = (shape[0], shape[1]);
        let mut t = vec![0f32; v.len()];
        for r in 0..o {
            for c in 0..i {
                t[c * o + r] = v[r * i + c];
            }
        }
        Tensor::from_data(TensorData::new(t, [i, o]), dev)
    }

    /// `[out, in]` の行 `start..start+len` だけを取り、`[in, len]` に転置する。
    fn linear_rows<B: Backend>(
        &self,
        name: &str,
        start: usize,
        len: usize,
        dev: &B::Device,
    ) -> Tensor<B, 2> {
        let (shape, v) = self.raw(name);
        let i = shape[1];
        let mut t = vec![0f32; len * i];
        for r in 0..len {
            for c in 0..i {
                t[c * len + r] = v[(start + r) * i + c];
            }
        }
        Tensor::from_data(TensorData::new(t, [i, len]), dev)
    }

    fn vec1<B: Backend>(&self, name: &str, dev: &B::Device) -> Tensor<B, 1> {
        let (shape, v) = self.raw(name);
        Tensor::from_data(TensorData::new(v, [shape[0]]), dev)
    }

    fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}

// ---------------------------------------------------------------- weights

struct LayerW<B: Backend> {
    attn_norm: Option<Tensor<B, 1>>,
    wqkv: Tensor<B, 2>,
    wo: Tensor<B, 2>,
    mlp_norm: Tensor<B, 1>,
    wi: Tensor<B, 2>,
    /// kohagi の `Wi::Split` と同じ配置: gate と up を独立した行列に割り、
    /// それぞれ contiguous な出力を作る。幅広 Wi + narrow は strided view を
    /// 残すので、CPU の要素演算がそこで遅くなる（src/encoder.rs の Wi::Split）。
    wi_gate: Tensor<B, 2>,
    wi_up: Tensor<B, 2>,
    mlp_wo: Tensor<B, 2>,
    global: bool,
}

struct Model<B: Backend> {
    tok: Tensor<B, 2>,
    emb_norm: Tensor<B, 1>,
    layers: Vec<LayerW<B>>,
    final_norm: Tensor<B, 1>,
    cos_g: Tensor<B, 4>,
    sin_g: Tensor<B, 4>,
    cos_l: Tensor<B, 4>,
    sin_l: Tensor<B, 4>,
    window: Tensor<B, 4>,
}

fn rope_tables<B: Backend>(seq: usize, hd: usize, theta: f32, d: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = hd / 2;
    let (mut cos, mut sin) = (vec![0f32; seq * hd], vec![0f32; seq * hd]);
    for p in 0..seq {
        for i in 0..half {
            let a = p as f32 / theta.powf(2.0 * i as f32 / hd as f32);
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

/// Finite rather than -inf: a fully padded query row would otherwise make
/// softmax return NaN, and in f16 anything past ~65504 is already inf.
const BLOCKED: f32 = -1.0e4;

fn window_mask<B: Backend>(seq: usize, window: usize, d: &B::Device) -> Tensor<B, 4> {
    let reach = window / 2;
    let mut m = vec![0f32; seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            if (q as isize - k as isize).unsigned_abs() > reach {
                m[q * seq + k] = BLOCKED;
            }
        }
    }
    Tensor::from_data(TensorData::new(m, [1, 1, seq, seq]), d)
}

fn load<B: FusedOps>(st: &Safetensors, cfg: &Cfg, seq: usize, dev: &B::Device) -> Model<B> {
    let hd = cfg.head_dim();
    let (cos_g, sin_g) = rope_tables::<B>(seq, hd, cfg.global_theta, dev);
    let (cos_l, sin_l) = rope_tables::<B>(seq, hd, cfg.local_theta, dev);
    let layers = (0..cfg.layers)
        .map(|i| {
            let p = format!("layers.{i}");
            LayerW {
                // Layer 0 ships no attn_norm; kohagi's `layer_norm_fused(..).ok()`
                // turns that into None rather than an error.
                attn_norm: st.has(&format!("{p}.attn_norm.weight")).then(|| st.vec1(&format!("{p}.attn_norm.weight"), dev)),
                wqkv: st.linear(&format!("{p}.attn.Wqkv.weight"), dev),
                wo: st.linear(&format!("{p}.attn.Wo.weight"), dev),
                mlp_norm: st.vec1(&format!("{p}.mlp_norm.weight"), dev),
                wi: st.linear(&format!("{p}.mlp.Wi.weight"), dev),
                wi_gate: st.linear_rows(&format!("{p}.mlp.Wi.weight"), 0, cfg.inter, dev),
                wi_up: st.linear_rows(&format!("{p}.mlp.Wi.weight"), cfg.inter, cfg.inter, dev),
                mlp_wo: st.linear(&format!("{p}.mlp.Wo.weight"), dev),
                global: i % cfg.global_every == 0,
            }
        })
        .collect();
    Model {
        tok: {
            let (shape, v) = st.raw("embeddings.tok_embeddings.weight");
            Tensor::from_data(TensorData::new(v, [shape[0], shape[1]]), dev)
        },
        emb_norm: st.vec1("embeddings.norm.weight", dev),
        layers,
        final_norm: st.vec1("final_norm.weight", dev),
        cos_g,
        sin_g,
        cos_l,
        sin_l,
        window: window_mask::<B>(seq, cfg.local_attention, dev),
    }
}

// ---------------------------------------------------------------- backend extension

/// Burn Book の "Backend Extension" パターン。
///
/// 汎用デフォルトを素の演算で書いておけば全バックエンドが動き、性能が要る
/// バックエンドだけ差し替える。エンコーダは `impl<B: FusedOps>` のままなので
/// genericity は壊れない。
///
/// 対象は RoPE だけ。調べた結果、softmax・gelu・layer_norm は burn-flex が
/// 既にバックエンド op として SIMD 融合実装を持っており、`ModuleOps` /
/// `ActivationOps` 経由で降りれば手を出す必要がない。burn-nn の RoPE だけが
/// 素の演算からの合成（`matmul(sign_tensor)` と `cat`）で、candle-nn の
/// `rotary_emb::rope` に相当する融合カーネルがない。
pub trait FusedOps: Backend {
    /// `x * cos + rotate_half(x) * sin`、`[rows, heads, seq, head_dim]`。
    fn rope(x: Tensor<Self, 4>, cos: Tensor<Self, 4>, sin: Tensor<Self, 4>) -> Tensor<Self, 4> {
        rope_composed(x, cos, sin)
    }

    /// ModernBERT の gated feed-forward。`wide` は `[.., 2 * inter]` で、前半に
    /// 活性化がかかる。kohagi の `crate::fused::gated` と同じ役割。
    fn geglu(wide: Tensor<Self, 3>, inter: usize) -> Tensor<Self, 3> {
        geglu_composed(wide, inter)
    }
}

fn geglu_composed<B: Backend>(wide: Tensor<B, 3>, inter: usize) -> Tensor<B, 3> {
    let gate = wide.clone().narrow(2, 0, inter);
    let up = wide.narrow(2, inter, inter);
    activation::gelu(gate) * up
}

/// 汎用デフォルトの本体。融合実装からも A/B 用に呼べるよう自由関数にしてある。
fn rope_composed<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let half = x.dims()[3] / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    x * cos + Tensor::cat(vec![-x2, x1], 3) * sin
}

/// `SPLIT=1` で kohagi の `Wi::Split` 配置に切り替える。
fn split_wi() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SPLIT").map(|v| v != "0").unwrap_or(false))
}

/// 診断用スイッチ（`GELU=tanh`）。
fn tanh_gelu() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GELU").map(|v| v == "tanh").unwrap_or(false))
}

/// `FUSED=0` で融合実装を切り、同じバイナリで A/B できるようにする。
fn fused_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FUSED").map(|v| v != "0").unwrap_or(true))
}

impl FusedOps for burn::backend::NdArray {}
impl FusedOps for burn::backend::Cpu {}
impl FusedOps for burn::backend::Vulkan {}
impl FusedOps for burn::backend::Vulkan<burn::tensor::f16, i32> {}

/// burn-flex 向けの融合実装。
///
/// `FlexTensor::as_slice()` はゼロコピービュー（`bytemuck::cast_slice`）なので、
/// 降りるのに転送コストがない。Burn Book の例が `TchTensor` に降りるのと同じ形。
impl FusedOps for burn::backend::Flex {
    fn rope(x: Tensor<Self, 4>, cos: Tensor<Self, 4>, sin: Tensor<Self, 4>) -> Tensor<Self, 4> {
        if !fused_enabled() {
            return rope_composed(x, cos, sin);
        }
        let [rows, heads, seq, hd] = x.dims();
        let half = hd / 2;
        let xp = contiguous(x.into_primitive().tensor());
        let cp = contiguous(cos.into_primitive().tensor());
        let sp = contiguous(sin.into_primitive().tensor());
        let (xs, cs, ss) = (slice(&xp), slice(&cp), slice(&sp));

        let mut out = vec![0f32; xs.len()];
        for r in 0..rows * heads {
            for p in 0..seq {
                let o = (r * seq + p) * hd;
                let t = p * hd;
                for i in 0..half {
                    let (a, b) = (xs[o + i], xs[o + i + half]);
                    out[o + i] = a * cs[t + i] - b * ss[t + i];
                    out[o + i + half] = b * cs[t + i + half] + a * ss[t + i + half];
                }
            }
        }
        rebuild(out, [rows, heads, seq, hd])
    }

}

fn contiguous(t: burn::backend::flex::FlexTensor) -> burn::backend::flex::FlexTensor {
    if t.is_contiguous() {
        t
    } else {
        t.to_contiguous()
    }
}

fn slice(t: &burn::backend::flex::FlexTensor) -> &[f32] {
    t.as_slice::<f32>().expect("contiguous f32 storage")
}

fn rebuild<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
) -> Tensor<burn::backend::Flex, D> {
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(
        burn::backend::flex::FlexTensor::from_data(TensorData::new(values, shape)),
    ))
}


// ---------------------------------------------------------------- profiling

/// 演算ごとの累積時間。burn-flex は同期実行なので `Instant` で素直に取れる。
mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub const LABELS: [&str; 11] = [
        "embed", "attn_norm", "qkv", "rope", "scores+mask", "softmax", "ctx+wo", "mlp_norm",
        "wi", "geglu", "wo2",
    ];
    pub static NS: [AtomicU64; 11] = [
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    ];

    pub fn on() -> bool {
        std::env::var("PROFILE").map(|v| v != "0").unwrap_or(false)
    }

    pub fn time<T>(idx: usize, f: impl FnOnce() -> T) -> T {
        if !on() {
            return f();
        }
        let t = Instant::now();
        let r = f();
        NS[idx].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }

    pub fn report() {
        if !on() {
            return;
        }
        let total: u64 = NS.iter().map(|a| a.load(Ordering::Relaxed)).sum();
        println!("--- 演算別（合計 {:.2} s、全ワーカースレッドの和）---", total as f64 / 1e9);
        let mut rows: Vec<(f64, &str)> = LABELS
            .iter()
            .enumerate()
            .map(|(i, l)| (NS[i].load(Ordering::Relaxed) as f64 / 1e9, *l))
            .collect();
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (s, l) in rows {
            println!("  {l:<12} {s:7.3} s  {:5.1}%", s / total as f64 * 1e9 * 100.0);
        }
    }
}

// ---------------------------------------------------------------- forward

/// Weight-only LayerNorm (`norm_bias = false`).
fn norm<B: FusedOps>(x: Tensor<B, 3>, g: Tensor<B, 1>, eps: f64, mixed: u8) -> Tensor<B, 3> {
    let dt = x.dtype();
    let (x, g) = if mixed == 2 {
        (x.cast(FloatDType::F32), g.cast(FloatDType::F32))
    } else {
        (x, g)
    };
    // Burn 自身のバックエンド op。汎用デフォルトが burn-backend にあり、
    // burn-flex はそれを SIMD 融合実装で上書きしている（ops/activation.rs）。
    // 契約は kohagi の LayerNorm と同じ: gamma は [d_model]、beta なし、
    // 偏りのある分散、eps は sqrt の前。
    let out = B::layer_norm(
        x.into_primitive().tensor(),
        g.into_primitive().tensor(),
        None,
        eps,
    );
    let y = Tensor::<B, 3>::from_primitive(burn::tensor::TensorPrimitive::Float(out));
    if mixed == 2 { y.cast(dt) } else { y }
}

fn rope<B: FusedOps>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    B::rope(x, cos, sin)
}

fn linear<B: Backend>(x: Tensor<B, 3>, w: Tensor<B, 2>) -> Tensor<B, 3> {
    let [b, s, i] = x.dims();
    let o = w.dims()[1];
    x.reshape([b * s, i]).matmul(w).reshape([b, s, o])
}

fn forward<B: FusedOps>(cfg: &Cfg, m: &Model<B>, ids: &[u32], pad: &[f32], bs: usize, seq: usize, dev: &B::Device) -> Tensor<B, 3> {
    let hd = cfg.head_dim();
    let h = cfg.hidden;
    let scale = 1.0 / (hd as f64).sqrt();

    let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(ids.iter().map(|&i| i as i32).collect::<Vec<_>>(), [ids.len()]), dev);
    let x = prof::time(0, || m.tok.clone().select(0, idx).reshape([bs, seq, h]));
    let mut x = prof::time(0, || norm(x, m.emb_norm.clone(), cfg.eps, cfg.mixed));

    // [b, 1, 1, seq]: padding is a property of the key, not the query.
    let padm = Tensor::<B, 4>::from_data(TensorData::new(pad.to_vec(), [bs, 1, 1, seq]), dev);

    for lw in m.layers.iter() {
        let residual = x.clone();
        let n = prof::time(1, || match &lw.attn_norm {
            Some(g) => norm(x.clone(), g.clone(), cfg.eps, cfg.mixed),
            None => x.clone(),
        });
        let qkv = prof::time(2, || linear(n, lw.wqkv.clone()));
        let split = |o: usize| qkv.clone().narrow(2, o * h, h).reshape([bs, seq, cfg.heads, hd]).swap_dims(1, 2);
        let (cos, sin) = if lw.global {
            (m.cos_g.clone(), m.sin_g.clone())
        } else {
            (m.cos_l.clone(), m.sin_l.clone())
        };
        let q = prof::time(3, || rope(split(0), cos.clone(), sin.clone()));
        let k = prof::time(3, || rope(split(1), cos, sin));
        let v = split(2);

        let raw = prof::time(4, || q.matmul(k.swap_dims(2, 3)));
        let dt = raw.dtype();
        let mut s = prof::time(4, || if cfg.mixed == 2 { raw.cast(FloatDType::F32) * scale } else { raw * scale });
        let (pm, wm) = if cfg.mixed == 2 {
            (padm.clone().cast(FloatDType::F32), m.window.clone().cast(FloatDType::F32))
        } else {
            (padm.clone(), m.window.clone())
        };
        s = s + pm;
        if !lw.global {
            s = s + wm;
        }
        let probs = prof::time(5, || activation::softmax(s, 3));
        let probs = if cfg.mixed == 2 { probs.cast(dt) } else { probs };

        let x2 = prof::time(6, || {
            let ctx = probs.matmul(v).swap_dims(1, 2).reshape([bs, seq, h]);
            residual + linear(ctx, lw.wo.clone())
        });

        let n = prof::time(7, || norm(x2.clone(), lw.mlp_norm.clone(), cfg.eps, cfg.mixed));
        let gated = if split_wi() {
            let g = prof::time(8, || linear(n.clone(), lw.wi_gate.clone()));
            let u = prof::time(8, || linear(n, lw.wi_up.clone()));
            prof::time(9, || activation::gelu(g) * u)
        } else {
            let gu = prof::time(8, || linear(n, lw.wi.clone()));
            prof::time(9, || B::geglu(gu, cfg.inter))
        };
        x = prof::time(10, || x2 + linear(gated, lw.mlp_wo.clone()));
    }
    norm(x, m.final_norm.clone(), cfg.eps, cfg.mixed)
}

/// Mask-aware mean pooling, then L2 normalization — kohagi's default.
fn pool<B: Backend>(h: Tensor<B, 3>, pad_ok: &[f32], bs: usize, seq: usize, dev: &B::Device) -> Vec<Vec<f32>> {
    let dim = h.dims()[2];
    let h = h.cast(FloatDType::F32);
    // Pooling always runs in f32; the mask must follow, or a f16 backend's
    // default dtype meets the cast hidden states and burn-ir rejects the pair.
    let mask = Tensor::<B, 3>::from_data(TensorData::new(pad_ok.to_vec(), [bs, seq, 1]), dev).cast(FloatDType::F32);
    let summed = (h * mask.clone()).sum_dim(1);
    let counts = mask.sum_dim(1);
    let mean = summed / counts;
    let v: Vec<f32> = mean.into_data().convert::<f32>().to_vec().unwrap();
    v.chunks(dim)
        .map(|r| {
            let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
            r.iter().map(|x| x / n).collect()
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

// ---------------------------------------------------------------- main

fn run<B: FusedOps>(cfg: &Cfg, st: &Safetensors, ids: &[u32], pad_add: &[f32], pad_ok: &[f32], bs: usize, seq: usize, tag: &str, reference: &[Vec<f32>]) {
    let dev = B::Device::default();
    let t = Instant::now();
    let m = load::<B>(st, cfg, seq, &dev);
    let load_ms = t.elapsed();

    let t = Instant::now();
    let out = forward::<B>(cfg, &m, ids, pad_add, bs, seq, &dev);
    let dtype = format!("{:?}", out.dtype());
    let got = pool::<B>(out, pad_ok, bs, seq, &dev);
    let first = t.elapsed();

    let mut best = std::time::Duration::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let o = forward::<B>(cfg, &m, ids, pad_add, bs, seq, &dev);
        let _ = o.into_data();
        best = best.min(t.elapsed());
    }

    let mut cs: Vec<f64> = reference.iter().zip(&got).map(|(a, b)| cosine(a, b)).collect();
    let worst = cs.iter().cloned().fold(1.0f64, f64::min);
    cs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = cs.len();
    let med = cs[n / 2];
    let p95 = cs[(n * 5 / 100).min(n - 1)];
    println!(
        "  1-cos  中央値 {:.3e}  p5 {:.3e}  最悪 {:.3e}   (n={n})",
        1.0 - med, 1.0 - p95, 1.0 - worst
    );
    println!(
        "{tag:12} dtype={dtype:4} load={:?} first={:?} warm={:?} ({:.1} rows/s)  worst cosine vs kohagi = {:.9}  (1-cos = {:.3e})",
        load_ms, first, best, bs as f64 / best.as_secs_f64(), worst, 1.0 - worst
    );
}

/// kohagi の CPU 経路と同じ形: バッチを `per` 行のユニットに割り、rayon で撒く。
/// フレームワーク内部の並列性ではなく、行方向の並列性で埋めるやり方。
fn run_fanout<B: FusedOps>(
    cfg: &Cfg,
    st: &Safetensors,
    ids: &[u32],
    pad_add: &[f32],
    pad_ok: &[f32],
    bs: usize,
    seq: usize,
    per: usize,
    tag: &str,
    reference: &[Vec<f32>],
) {
    use rayon::prelude::*;
    let dev = B::Device::default();
    let m = load::<B>(st, cfg, seq, &dev);

    let units: Vec<(usize, usize)> = (0..bs)
        .step_by(per)
        .map(|s| (s, per.min(bs - s)))
        .collect();

    let once = || {
        let parts: Vec<(usize, Vec<Vec<f32>>)> = units
            .par_iter()
            .map(|&(start, rows)| {
                let d = B::Device::default();
                let from = start * seq;
                let to = (start + rows) * seq;
                let out = forward::<B>(cfg, &m, &ids[from..to], &pad_add[from..to], rows, seq, &d);
                (start, pool::<B>(out, &pad_ok[from..to], rows, seq, &d))
            })
            .collect();
        let mut got: Vec<Vec<f32>> = vec![Vec::new(); bs];
        for (start, rows) in parts {
            for (i, v) in rows.into_iter().enumerate() {
                got[start + i] = v;
            }
        }
        got
    };

    let got = once();
    let mut best = std::time::Duration::MAX;
    for _ in 0..2 {
        let t = Instant::now();
        std::hint::black_box(once());
        best = best.min(t.elapsed());
    }
    prof::report();
    let worst = reference
        .iter()
        .zip(&got)
        .map(|(a, b)| cosine(a, b))
        .fold(1.0f64, f64::min);
    println!(
        "{tag:14} bs={bs} per={per}  best={:?}  ({:.1} rows/s)  worst 1-cos = {:.3e}",
        best,
        bs as f64 / best.as_secs_f64(),
        1.0 - worst
    );
}

fn main() {
    let root = std::env::var("MODEL").unwrap();
    let rows: usize = std::env::var("ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let mode = std::env::args().nth(1).unwrap_or_else(|| "mixed2".into());

    let cfg0 = Cfg::read(&format!("{root}/config.json"));
    let st = Safetensors::open(&format!("{root}/model.safetensors"));

    // Same texts kohagi was given, tokenized with the model's own tokenizer.
    let texts: Vec<String> = std::fs::read_to_string(std::env::var("TEXTS").unwrap())
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["text"].as_str().unwrap().to_string())
        .collect();
    let reference: Vec<Vec<f32>> = std::fs::read_to_string(std::env::var("REF").unwrap())
        .unwrap()
        .lines()
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["embedding"]
                .as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
        })
        .collect();

    let tk = tokenizers::Tokenizer::from_file(format!("{root}/tokenizer.json")).unwrap();
    let encs: Vec<_> = texts.iter().take(rows).map(|t| tk.encode(t.as_str(), true).unwrap()).collect();
    let seq = encs.iter().map(|e| e.get_ids().len().min(512)).max().unwrap();
    let bs = encs.len();

    let mut ids = vec![0u32; bs * seq];
    let mut pad_add = vec![0f32; bs * seq];
    let mut pad_ok = vec![0f32; bs * seq];
    for (r, e) in encs.iter().enumerate() {
        let n = e.get_ids().len().min(seq);
        for c in 0..seq {
            if c < n {
                ids[r * seq + c] = e.get_ids()[c];
                pad_ok[r * seq + c] = 1.0;
            } else {
                pad_add[r * seq + c] = BLOCKED;
            }
        }
    }
    println!("rows={bs} seq={seq} layers={} hidden={}", cfg0.layers, cfg0.hidden);

    let reference = &reference[..bs];
    match mode.as_str() {
        "ndarray" => run::<burn::backend::NdArray>(&cfg0, &st, &ids, &pad_add, &pad_ok, bs, seq, "burn ndarray", reference),
        "flexpar" => {
            let per: usize = std::env::var("PER").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            run_fanout::<burn::backend::Flex>(&cfg0, &st, &ids, &pad_add, &pad_ok, bs, seq, per, "flex fanout", reference)
        }
        "flex" => run::<burn::backend::Flex>(&cfg0, &st, &ids, &pad_add, &pad_ok, bs, seq, "burn flex", reference),
        "burncpu" => run::<burn::backend::Cpu>(&cfg0, &st, &ids, &pad_add, &pad_ok, bs, seq, "burn cpu", reference),
        "f32" => run::<burn::backend::Vulkan>(&cfg0, &st, &ids, &pad_add, &pad_ok, bs, seq, "vulkan f32", reference),
        "f16" => run::<burn::backend::Vulkan<burn::tensor::f16, i32>>(&cfg0, &st, &ids, &pad_add, &pad_ok, bs, seq, "vulkan f16", reference),
        _ => {
            let c = Cfg { mixed: 2, ..cfg0 };
            run::<burn::backend::Vulkan<burn::tensor::f16, i32>>(&c, &st, &ids, &pad_add, &pad_ok, bs, seq, "vulkan mix2", reference)
        }
    }
}
