//! Does the reranker agree between the two CPU engines?
//!
//! It is the same encoder with a different reduction, and the head stays on the
//! CPU in candle whichever engine ran the layers — so a disagreement here that
//! `cpu_burn_matches_cpu` did not catch is in `classifier_pooling` or the
//! plumbing around it, not in ModernBERT. The Vulkan device reaches the
//! reranker through the same code and differs only in what runs the layers.
//!
//! ```console
//! cargo test --features cpu-burn --test rerank_on_burn
//! ```

#![cfg(feature = "cpu-burn")]

use kohagi::{Backend, ModelSource, Precision};

/// `japanese-reranker-xsmall-v2` rather than the default 310m: this asks a
/// question about wiring, and the smallest cross-encoder answers it.
#[test]
fn the_reranker_agrees_too() {
    use kohagi::rerank::{Options as RerankOptions, Reranker};

    let pairs: Vec<(&str, &str)> = vec![
        (
            "日本語の埋め込みモデルの評価",
            "検索タスクと意味的類似度タスクの双方を考慮する必要がある。",
        ),
        (
            "ModernBERTの注意機構",
            "ローカル注意とグローバル注意を交互に用いる。",
        ),
        ("回転位置埋め込み", "犬が公園で遊んでいる。"),
    ];
    let source = ModelSource::Hub {
        repo: "hotchpotch/japanese-reranker-xsmall-v2".to_string(),
    };
    let load = |backend| {
        Reranker::load(
            &source,
            RerankOptions {
                backend,
                precision: Precision::F32,
                ..Default::default()
            },
        )
        .ok()
    };
    let (Some(candle), Some(burn)) = (load(Backend::Cpu), load(Backend::CpuBurn)) else {
        eprintln!("skipped: no reranker available");
        return;
    };
    let (a, _) = candle.score(&pairs).expect("candle scores");
    let (b, _) = burn.score(&pairs).expect("cpu-burn scores");
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        // Absolute, not relative: a sigmoid output is already in 0..1, and the
        // scores that matter most here are the ones near zero.
        assert!(
            (x - y).abs() < 1e-5,
            "pair {i}: candle {x} against cpu-burn {y}"
        );
    }
}
