//! Does `--device cpu-burn` produce the same numbers as `--device cpu`?
//!
//! Both engines run the CPU in f32, so this is not a precision question: the
//! only thing between them is float addition order, and anything larger is the
//! graph. That makes this the sharpest test of the Burn ModernBERT there is —
//! a wrong mask, RoPE base, gate half or layer-0 `attn_norm` has nowhere to
//! hide. `vulkan_matches_cpu` asks the same of a device that also changes
//! precision, and so cannot separate the two causes on its own.
//!
//! Unlike the Vulkan pair, both engines fit in one process: burn-flex is not a
//! CubeCL backend and binds no per-device element type.
//!
//! ```console
//! cargo test --features cpu-burn --test cpu_burn_matches_cpu
//! ```

#![cfg(feature = "cpu-burn")]

use kohagi::{Backend, Embedder, ModelSource, Options, Precision};

/// What float addition order may cost between two f32 engines, and nothing
/// else. Measured worst over these texts is 1.0e-12.
const TOLERANCE: f64 = 1.0e-9;

fn source() -> ModelSource {
    ModelSource::Hub {
        repo: "cl-nagoya/ruri-v3-130m".to_string(),
    }
}

/// Mixed lengths on purpose: a short row beside a long one is what exercises
/// the padding mask, and enough rows that the budget splits them into more
/// units than the pool has threads.
fn texts() -> Vec<String> {
    let parts = [
        "日本語の文埋め込みモデルの評価においては、検索タスクと意味的類似度タスクの双方を考慮する必要がある。",
        "ModernBERTはローカル注意とグローバル注意を交互に用いることで長い系列を効率的に扱う。",
        "回転位置埋め込みは相対位置の情報を注意スコアに直接与えるため、外挿性能に優れるとされる。",
        "The quick brown fox jumps over the lazy dog while the model computes its vectors.",
    ];
    [1usize, 12, 3, 20, 7, 2, 15, 9]
        .iter()
        .enumerate()
        .map(|(i, &n)| (0..n).map(|j| parts[(i + j) % parts.len()]).collect())
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// `None` when the model is not cached, which is not a failure — this suite is
/// skipped on such a machine rather than red on it.
fn embed(backend: Backend, texts: &[&str]) -> Option<Vec<Vec<f32>>> {
    let opts = Options {
        backend,
        precision: Precision::F32,
        ..Default::default()
    };
    Embedder::load(&source(), opts).ok()?.embed(texts).ok()
}

#[test]
fn the_two_cpu_engines_agree() {
    let owned = texts();
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    let (Some(candle), Some(burn)) = (embed(Backend::Cpu, &texts), embed(Backend::CpuBurn, &texts))
    else {
        eprintln!("skipped: no model available");
        return;
    };
    for (i, (a, b)) in candle.iter().zip(&burn).enumerate() {
        let off = 1.0 - cosine(a, b);
        assert!(
            off < TOLERANCE,
            "row {i}: cpu-burn is 1-cos {off:.3e} from candle, over {TOLERANCE:.0e}. \
             Both are f32, so the cause is the graph, not the arithmetic."
        );
    }
}

/// The precisions this device does not have, refused at load rather than
/// silently substituted.
#[test]
fn unsupported_precisions_are_refused() {
    for precision in [Precision::Bf16, Precision::F16] {
        let refused = Embedder::load(
            &source(),
            Options {
                backend: Backend::CpuBurn,
                precision,
                ..Default::default()
            },
        );
        assert!(
            refused.is_err(),
            "{} on cpu-burn should be refused",
            precision.name()
        );
    }
}
