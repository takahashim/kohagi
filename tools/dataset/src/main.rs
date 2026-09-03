//! A distillation dataset, made once and trained on many times.
//!
//! Distilling a cross-encoder means teaching a small model to answer what a
//! large one answers. The student needs two things per example: the token
//! ids it will read, and the teacher's score to be pulled towards. Both
//! come from here, in one pass, because Kohagi is the one place that holds
//! both the tokenizer and the teacher.
//!
//! Made once because it is preprocessing, not training. A student is run
//! many times, with different rates and schedules; recomputing a 310M
//! forward on every step of every one of those runs would be paying for
//! the same answer over and over. The teacher's score is one float.
//!
//! Reading it back needs no tokenizer, which is the point: Torobi is handed
//! ids and never has to agree with anyone about how text becomes them.
//!
//!     cargo run --release --manifest-path tools/dataset/Cargo.toml -- \
//!       --model-path <reranker>/model.safetensors \
//!       --tokenizer-path <reranker>/tokenizer.json \
//!       --pairs pairs.jsonl --out train.jsonl
//!
//! `pairs.jsonl` is one `{"query": ..., "text": ...}` per line; anything
//! else on the line is carried through, so an id or a label survives the
//! trip. `train.jsonl` is the same with `input_ids` and `teacher` added.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use kohagi::rerank::{Options, Reranker};
use kohagi::ModelSource;
use tokenizers::{EncodeInput, Tokenizer, TruncationParams};

/// How many pairs to score at once. The reranker batches internally; this
/// is only how much is held in memory here at a time.
const CHUNK: usize = 256;

fn main() -> Result<()> {
    let options = Cli::parse();
    let tokenizer = load_tokenizer(&options.tokenizer_path, options.max_seq_length)?;
    let reranker = Reranker::load(
        &ModelSource::Files {
            model: options.model_path.clone(),
            tokenizer: options.tokenizer_path.clone(),
        },
        Options {
            max_seq_length: options.max_seq_length,
            sigmoid: options.sigmoid,
            ..Default::default()
        },
    )?;

    let rows = read_pairs(&options.pairs)?;
    anyhow::ensure!(
        !rows.is_empty(),
        "{} holds no pairs",
        options.pairs.display()
    );

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&options.out)
            .with_context(|| format!("writing {}", options.out.display()))?,
    );
    let mut written = 0usize;
    for chunk in rows.chunks(CHUNK) {
        for line in score_chunk(&reranker, &tokenizer, chunk)? {
            writeln!(out, "{line}")?;
            written += 1;
        }
        eprint!("\r{written}/{} pairs", rows.len());
    }
    out.flush()?;
    eprintln!("\rwrote {} ({written} rows)", options.out.display());
    Ok(())
}

/// One chunk of pairs: tokenized here, scored by the teacher, and checked
/// that the two saw the same thing before either is written.
fn score_chunk(reranker: &Reranker, tokenizer: &Tokenizer, rows: &[Row]) -> Result<Vec<String>> {
    let pairs: Vec<(&str, &str)> = rows
        .iter()
        .map(|row| (row.query.as_str(), row.text.as_str()))
        .collect();
    let inputs: Vec<EncodeInput> = pairs.iter().map(|&pair| pair.into()).collect();
    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
    let (scores, tokens) = reranker.score(&pairs)?;

    rows.iter()
        .zip(&encodings)
        .zip(&scores)
        .zip(&tokens)
        .map(|(((row, encoding), score), token)| {
            // The ids written here have to be the ones the teacher scored,
            // or the student is being taught an answer to a different
            // question. Kohagi reports how many tokens each pair became.
            anyhow::ensure!(
                encoding.get_ids().len() == token.n_tokens,
                "this tokenized a pair to {} tokens and the teacher scored {}; \
                 the two are not configured the same",
                encoding.get_ids().len(),
                token.n_tokens
            );
            let mut record = row.rest.clone();
            record.insert("input_ids".into(), serde_json::json!(encoding.get_ids()));
            record.insert("teacher".into(), serde_json::json!(score));
            record.insert("truncated".into(), serde_json::json!(token.truncated));
            Ok(serde_json::to_string(&record)?)
        })
        .collect()
}

/// One input line: the pair, and whatever else it carried.
struct Row {
    query: String,
    text: String,
    rest: serde_json::Map<String, serde_json::Value>,
}

fn read_pairs(path: &std::path::Path) -> Result<Vec<Row>> {
    let file = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(i, line)| match line {
            Err(e) => Some(Err(e.into())),
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(parse_row(&line, i + 1)),
        })
        .collect()
}

fn parse_row(line: &str, number: usize) -> Result<Row> {
    let mut rest: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(line).with_context(|| format!("line {number} is not JSON"))?;
    let take = |rest: &mut serde_json::Map<_, _>, key: &str| -> Result<String> {
        rest.remove(key)
            .and_then(|v| v.as_str().map(str::to_string))
            .with_context(|| format!("line {number} has no string {key:?}"))
    };
    let query = take(&mut rest, "query")?;
    let text = take(&mut rest, "text")?;
    Ok(Row { query, text, rest })
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

/// Score query/document pairs with Kohagi's cross-encoder and write the token
/// ids beside the score, so a student can be trained on them many times.
///
/// `--pairs` is one {"query", "text"} per line; anything else on the line is
/// carried through. The output is the same with `input_ids`, `teacher` and
/// `truncated` added.
#[derive(Parser)]
#[command(name = "kohagi-dataset", version)]
struct Cli {
    /// The teacher's safetensors weights (config.json must sit next to it).
    #[arg(long)]
    model_path: PathBuf,
    /// The teacher's tokenizer.json, which is also what tokenizes the pairs.
    #[arg(long)]
    tokenizer_path: PathBuf,
    /// The pairs to score, one JSON object per line.
    #[arg(long)]
    pairs: PathBuf,
    /// Where to write the scored rows.
    #[arg(long)]
    out: PathBuf,
    /// Token-level truncation length for a pair.
    #[arg(long, default_value_t = 512)]
    max_seq_length: usize,
    /// Distil against the raw logit instead of its sigmoid. Which one to
    /// distil against is the recipe's business: a squashed target loses
    /// resolution at the ends, and an unsquashed one is what a distillation
    /// loss usually wants.
    #[arg(long = "logit", action = clap::ArgAction::SetFalse)]
    sigmoid: bool,
}
