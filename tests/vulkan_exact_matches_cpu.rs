//! Does `--device vulkan --precision f32` reproduce the CPU path exactly?
//!
//! Its own file, and so its own process, because CubeCL binds one float element
//! type per device: a test binary that loaded f16 first would run this in f16
//! and fail for a reason that has nothing to do with what it checks.
//! `crate::vulkan::claim_precision` refuses that rather than letting it happen
//! quietly, which is why this cannot simply live beside the f16 test.
//!
//! ```console
//! cargo test --features vulkan --test vulkan_exact_matches_cpu
//! ```

#![cfg(feature = "vulkan")]

use kohagi::{Backend, Embedder, ModelSource, Options, Precision};

/// What float addition order may cost, and nothing else. Measured worst is 1.2e-12.
const F32_TOLERANCE: f64 = 1.0e-9;

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
fn f32_reproduces_the_cpu_path() {
    let owned = texts();
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    let Some(cpu) = embed(Backend::Cpu, Precision::F32, &texts) else {
        eprintln!("skipped: no model available");
        return;
    };
    let Some(gpu) = embed(Backend::Vulkan, Precision::F32, &texts) else {
        eprintln!("skipped: no Vulkan device available");
        return;
    };
    for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
        let off = 1.0 - cosine(a, b);
        assert!(
            off < F32_TOLERANCE,
            "row {i}: vulkan f32 is 1-cos {off:.3e} from the CPU, over {F32_TOLERANCE:.0e}. \
             At this precision the cause is the graph, not the arithmetic."
        );
    }
}
