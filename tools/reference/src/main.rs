//! Kohagi's answer, written down so another implementation can be held to it.
//!
//! Torobi (the training framework) describes ModernBERT in its own IR and
//! runs it on MLX. Whether that description is right is a question about
//! numbers, and the useful comparison is against a second implementation of
//! the same architecture rather than against itself. Kohagi is that: candle
//! on the CPU, the same published weights, an encoder whose output is
//! already the thing people rely on.
//!
//! What this writes is everything the other side needs to reproduce the
//! same forward and nothing about how Kohagi got there: the token ids (so
//! no tokenizer is needed on the other side), the digest of the weights, the
//! pooled and normalized vector, and the settings both sides must agree on.
//! Those settings are read off the loaded model rather than written down
//! here, so a run with other options cannot describe itself as this one.
//!
//! Kohagi gained one item for this, `VERSION`, so the artifact can name what
//! produced it; nothing else. The tokenizer is loaded the same way rather
//! than exported (Kohagi's `batch::load_tokenizer` is four lines and
//! private), and the ids emitted are checked against what Kohagi embedded,
//! by count, before anything is written.
//!
//!     cargo run --release --manifest-path tools/reference/Cargo.toml -- \
//!       --model-path <snapshot>/model.safetensors \
//!       --tokenizer-path <snapshot>/tokenizer.json \
//!       --out reference.json "瑠璃も玻璃も照らせば光る"

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use kohagi::{Embedder, ModelSource, Options};
use tokenizers::{Tokenizer, TruncationParams};

/// Write Kohagi's forward pass down as a reference artifact.
///
/// Emits the token ids, the pooled and normalized vector, and the settings
/// both sides must agree on, so another implementation of the same
/// architecture can be held to these numbers.
#[derive(Parser)]
#[command(name = "kohagi-reference", version)]
struct Cli {
    /// The safetensors weights to embed with (config.json must sit next to it).
    #[arg(long)]
    model_path: PathBuf,
    /// The tokenizer.json for those weights.
    #[arg(long)]
    tokenizer_path: PathBuf,
    /// Where to write the artifact.
    #[arg(long)]
    out: PathBuf,
    /// Token-level truncation length.
    #[arg(long, default_value_t = 512)]
    max_seq_length: usize,
    /// The texts to embed, one artifact case each.
    #[arg(required = true, value_name = "TEXT")]
    texts: Vec<String>,
}

fn main() -> Result<()> {
    let Cli {
        model_path,
        tokenizer_path,
        out,
        max_seq_length,
        texts,
    } = Cli::parse();

    let tokenizer = load_tokenizer(&tokenizer_path, max_seq_length)?;
    let borrowed: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encodings = tokenizer
        .encode_batch(borrowed.clone(), true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    let embedder = Embedder::load(
        &ModelSource::Files {
            model: model_path.clone(),
            tokenizer: tokenizer_path.clone(),
        },
        Options {
            max_seq_length,
            ..Default::default()
        },
    )?;
    let (vectors, tokens) = embedder.embed_with_tokens(&borrowed)?;
    let info = embedder.info();

    // The ids written here have to be the ones Kohagi embedded, or the
    // comparison is between two different inputs. Kohagi reports how many
    // tokens each text became; if that disagrees with what this tokenized,
    // the two are not configured the same and nothing is written.
    for ((text, encoding), token) in texts.iter().zip(&encodings).zip(&tokens) {
        anyhow::ensure!(
            encoding.get_ids().len() == token.n_tokens,
            "this tokenized {text:?} to {} tokens and Kohagi embedded {}; \
             the two are not configured the same",
            encoding.get_ids().len(),
            token.n_tokens
        );
    }

    let cases: Vec<serde_json::Value> = texts
        .iter()
        .zip(&encodings)
        .zip(&vectors)
        .map(|((text, encoding), vector)| {
            serde_json::json!({
                "text": text,
                "input_ids": encoding.get_ids(),
                "attention_mask": encoding.get_attention_mask(),
                "embedding": vector,
            })
        })
        .collect();

    let artifact = serde_json::json!({
        "schema_version": 1,
        // kohagi's version, not this jig's: the numbers below are kohagi's.
        "produced_by": format!("kohagi {}", kohagi::VERSION),
        // Which weights answered. The other side is being held to these
        // numbers, and `pooling` and `dim` agreeing means nothing if the
        // checkpoint behind them was a different one.
        "sha256": info.sha256,
        // What both sides have to agree on, or the comparison means nothing.
        // Read off the loaded model rather than written down here, so a run
        // with other options cannot describe itself as this one.
        "settings": {
            "max_seq_length": info.max_seq_length,
            "pooling": info.pooling,
            "normalized": info.normalized(),
            "precision": info.precision,
        },
        "dim": embedder.dim(),
        "cases": cases,
    });
    std::fs::write(
        &out,
        format!("{}\n", serde_json::to_string_pretty(&artifact)?),
    )?;
    eprintln!("wrote {} ({} cases)", out.display(), texts.len());
    Ok(())
}

/// The same four lines Kohagi's own loader runs: truncation pinned to the
/// sequence limit, no padding. Repeated rather than exported, so this tool
/// does not widen what Kohagi promises.
fn load_tokenizer(path: &std::path::Path, max_seq_length: usize) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("cannot load tokenizer {}: {e}", path.display()))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_seq_length,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("truncation config: {e}"))?;
    tokenizer.with_padding(None);
    Ok(tokenizer)
}
