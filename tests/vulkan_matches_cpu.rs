//! Does `--device vulkan --precision f16` stay close enough to the CPU?
//!
//! The companion to `vulkan_exact_matches_cpu`, which asks the same question of
//! f32 and has to be a separate binary to do it (one float element type per
//! device per process). The pair separates two failures that look alike from
//! here: if f32 also drifts, the graph is wrong — a mask, a RoPE base, a gate
//! half, layer 0's missing `attn_norm`. If only this one does, it is the
//! precision recipe.
//!
//! Needs the feature, a Vulkan device, and the model, so it skips elsewhere:
//!
//! ```console
//! cargo test --features vulkan --test vulkan_matches_cpu
//! ```

#![cfg(feature = "vulkan")]

use kohagi::{Backend, Embedder, ModelSource, Options, Precision};

/// What `--precision f16` may cost, and the same figure `docs/devices.md`
/// publishes for the CoreML path. Measured worst over these texts on a Radeon
/// 780M is 8.9e-6; this leaves room for another device to round differently
/// without leaving room for a wrong kernel.
const F16_TOLERANCE: f64 = 2.0e-5;

/// Long enough to reach the sliding window (`local_attention` is 128) and to
/// put several rows in one padded batch, short enough not to need truncation.
fn texts() -> Vec<String> {
    let parts = [
        "日本語の文埋め込みモデルの評価においては、検索タスクと意味的類似度タスクの双方を考慮する必要がある。",
        "ModernBERTはローカル注意とグローバル注意を交互に用いることで長い系列を効率的に扱う。",
        "回転位置埋め込みは相対位置の情報を注意スコアに直接与えるため、外挿性能に優れるとされる。",
        "The quick brown fox jumps over the lazy dog while the model computes its vectors.",
    ];
    // Mixed lengths on purpose: a short row beside a long one is what exercises
    // the padding mask, and a one-row batch is what exercises the budget split.
    [1usize, 12, 3, 20, 7]
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

/// `None` when there is no GPU or no cached model, which is not a failure —
/// this suite is skipped on such a machine rather than red on it.
fn embed(backend: Backend, precision: Precision, texts: &[&str]) -> Option<Vec<Vec<f32>>> {
    let source = ModelSource::Hub {
        repo: "cl-nagoya/ruri-v3-130m".to_string(),
    };
    let opts = Options {
        backend,
        precision,
        ..Default::default()
    };
    let embedder = Embedder::load(&source, opts).ok()?;
    embedder.embed(texts).ok()
}

#[test]
fn f16_stays_within_the_published_tolerance() {
    let owned = texts();
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    let Some(cpu) = embed(Backend::Cpu, Precision::F32, &texts) else {
        eprintln!("skipped: no model available");
        return;
    };
    let Some(gpu) = embed(Backend::Vulkan, Precision::F16, &texts) else {
        eprintln!("skipped: no Vulkan device available");
        return;
    };
    for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
        let off = 1.0 - cosine(a, b);
        assert!(
            off < F16_TOLERANCE,
            "row {i}: vulkan f16 is 1-cos {off:.3e} from the CPU, over {F16_TOLERANCE:.0e}"
        );
    }

    // Asserted here rather than as its own `#[test]` so the order is decided:
    // this device is now bound to f16, and asking it for f32 must fail loudly.
    // Left unguarded it does not fail at all — it returns plain-f16 vectors,
    // 1-cos 2.5e-2 from the CPU, with nothing to say so.
    let after = Embedder::load(
        &ModelSource::Hub {
            repo: "cl-nagoya/ruri-v3-130m".to_string(),
        },
        Options {
            backend: Backend::Vulkan,
            precision: Precision::F32,
            ..Default::default()
        },
    );
    assert!(
        after.is_err(),
        "a second Vulkan precision in one process must be refused, not silently served"
    );
}

/// The two precisions the Vulkan backend does not have, refused at load rather
/// than silently substituted.
#[test]
fn unsupported_precisions_are_refused() {
    let source = ModelSource::Hub {
        repo: "cl-nagoya/ruri-v3-130m".to_string(),
    };
    let bf16 = Embedder::load(
        &source,
        Options {
            backend: Backend::Vulkan,
            precision: Precision::Bf16,
            ..Default::default()
        },
    );
    assert!(bf16.is_err(), "bf16 on Vulkan should be refused");

    let f16_on_cpu = Embedder::load(
        &source,
        Options {
            backend: Backend::Cpu,
            precision: Precision::F16,
            ..Default::default()
        },
    );
    assert!(f16_on_cpu.is_err(), "f16 on the CPU should be refused");
}
