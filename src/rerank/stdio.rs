//! The reranker's stdin/stdout JSONL protocol (see PROTOCOL-rerank.md).
//!
//! `{"id": …, "query": …, "text": …}` in, `{"id": …, "score": …}` out. Every
//! rule the embedding protocol has holds here too: `id` is opaque and echoed,
//! a malformed line is skipped rather than fatal, a blank line means "score
//! what I have sent and answer now", and the answer to a batch ends with a
//! blank line of its own. Only the record shape differs.

use std::io::{BufRead, BufWriter, Write};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;

use super::Reranker;
use crate::TokenInfo;

/// Pairs scored per chunk. As in the embedding path, this bounds resident
/// memory to one chunk rather than the whole input.
const CHUNK_ROWS: usize = 1024;

#[derive(Serialize)]
struct OutRecord<'a> {
    id: &'a Value,
    score: f32,
    /// `--report-tokens` only, as in the embedding protocol.
    #[serde(skip_serializing_if = "Option::is_none")]
    n_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncated: Option<bool>,
}

fn write_record(
    out: &mut impl Write,
    id: &Value,
    score: f32,
    tokens: Option<&TokenInfo>,
) -> Result<()> {
    serde_json::to_writer(
        &mut *out,
        &OutRecord {
            id,
            score,
            n_tokens: tokens.map(|t| t.n_tokens),
            truncated: tokens.map(|t| t.truncated),
        },
    )?;
    out.write_all(b"\n")?;
    Ok(())
}

/// One accepted input line.
struct InRecord {
    id: Value,
    query: String,
    text: String,
}

enum Line {
    Record(InRecord),
    Flush,
}

/// Parse one physical line. `Err` = skip with a warning.
fn parse_line(line: &str) -> Result<Line, String> {
    if line.trim().is_empty() {
        return Ok(Line::Flush);
    }
    let v: Value = serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = v.as_object().ok_or("not a JSON object")?;
    let id = obj.get("id").ok_or("missing \"id\"")?.clone();
    let field = |name: &str| -> Result<String, String> {
        let s = obj
            .get(name)
            .and_then(Value::as_str)
            .ok_or(format!("missing or non-string \"{name}\""))?;
        if s.is_empty() {
            return Err(format!("empty \"{name}\""));
        }
        Ok(s.to_string())
    };
    let query = field("query")?;
    let text = field("text")?;
    Ok(Line::Record(InRecord { id, query, text }))
}

struct Flushed {
    written: usize,
    truncated: usize,
}

/// Score the buffered chunk, write its lines, and empty the buffer. The model
/// is loaded here on first use, so input with no valid records never loads it.
fn flush_chunk(
    reranker: &mut Option<Reranker>,
    load: &impl Fn() -> Result<Reranker>,
    report_tokens: bool,
    chunk: &mut Vec<InRecord>,
    out: &mut impl Write,
) -> Result<Flushed> {
    if chunk.is_empty() {
        return Ok(Flushed {
            written: 0,
            truncated: 0,
        });
    }
    let reranker = match reranker {
        Some(r) => r,
        None => reranker.insert(load()?),
    };

    let pairs: Vec<(&str, &str)> = chunk
        .iter()
        .map(|r| (r.query.as_str(), r.text.as_str()))
        .collect();
    let (scores, tokens) = reranker.score(&pairs)?;

    let mut truncated = 0usize;
    for ((record, &score), info) in chunk.iter().zip(&scores).zip(&tokens) {
        truncated += info.truncated as usize;
        write_record(out, &record.id, score, report_tokens.then_some(info))?;
    }
    out.flush()?;

    let written = chunk.len();
    chunk.clear();
    Ok(Flushed { written, truncated })
}

/// Run the protocol over stdin/stdout. Returns the number of skipped lines,
/// which the caller maps to exit code 2.
pub fn run(
    load: impl Fn() -> Result<Reranker>,
    report_tokens: bool,
    model_label: &str,
) -> Result<usize> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let mut reranker: Option<Reranker> = None;
    let mut chunk: Vec<InRecord> = Vec::new();
    let mut n_out = 0usize;
    let mut skipped = 0usize;
    let mut truncated = 0usize;

    for (lineno, line) in stdin.lock().lines().enumerate() {
        let line = line.context("reading stdin")?;
        match parse_line(&line) {
            Ok(Line::Record(record)) => {
                chunk.push(record);
                if chunk.len() >= CHUNK_ROWS {
                    let f = flush_chunk(&mut reranker, &load, report_tokens, &mut chunk, &mut out)?;
                    n_out += f.written;
                    truncated += f.truncated;
                }
            }
            Ok(Line::Flush) => {
                let f = flush_chunk(&mut reranker, &load, report_tokens, &mut chunk, &mut out)?;
                n_out += f.written;
                truncated += f.truncated;
                // The reply marker, so a long-lived caller can read until the
                // blank line instead of counting records a skip would spoil.
                out.write_all(b"\n")?;
                out.flush()?;
            }
            Err(why) => {
                skipped += 1;
                eprintln!("kohagi-rerank: skip line {}: {why}", lineno + 1);
            }
        }
    }
    let f = flush_chunk(&mut reranker, &load, report_tokens, &mut chunk, &mut out)?;
    out.flush()?;
    n_out += f.written;
    truncated += f.truncated;

    let facts = reranker.as_ref().map_or_else(
        || "dim=0".to_string(),
        |r| crate::stdio::summary_facts(&r.info()),
    );
    let n_in = n_out + skipped;
    eprintln!(
        "kohagi-rerank: model={model_label} {facts} in={n_in} out={n_out} \
         skipped={skipped} truncated={truncated}"
    );
    Ok(skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(line: &str) -> InRecord {
        match parse_line(line).expect("a record") {
            Line::Record(r) => r,
            Line::Flush => panic!("expected a record, got a batch boundary"),
        }
    }

    #[test]
    fn parse_takes_a_pair_and_echoes_any_id() {
        let r = record(r#"{"id": "a-1", "query": "配列の並べ替え", "text": "sort は…"}"#);
        assert_eq!(r.id, Value::from("a-1"));
        assert_eq!(r.query, "配列の並べ替え");
        assert_eq!(r.text, "sort は…");
    }

    /// A pair needs both halves. Scoring a query against nothing would produce
    /// a number, which is worse than skipping the line.
    #[test]
    fn parse_skips_what_it_cannot_score() {
        assert!(matches!(parse_line(""), Ok(Line::Flush)));
        assert!(parse_line("not json").is_err());
        assert!(parse_line(r#"{"query": "q", "text": "t"}"#).is_err());
        assert!(parse_line(r#"{"id": 1, "text": "t"}"#).is_err());
        assert!(parse_line(r#"{"id": 1, "query": "q"}"#).is_err());
        assert!(parse_line(r#"{"id": 1, "query": "", "text": "t"}"#).is_err());
        assert!(parse_line(r#"{"id": 1, "query": "q", "text": 5}"#).is_err());
    }

    #[test]
    fn the_output_record_is_id_and_score() {
        let id = Value::from(7);
        let mut plain = Vec::new();
        write_record(&mut plain, &id, 0.5, None).unwrap();
        assert_eq!(plain, b"{\"id\":7,\"score\":0.5}\n");

        let info = TokenInfo {
            n_tokens: 512,
            truncated: true,
        };
        let mut full = Vec::new();
        write_record(&mut full, &id, 0.5, Some(&info)).unwrap();
        assert_eq!(
            full,
            b"{\"id\":7,\"score\":0.5,\"n_tokens\":512,\"truncated\":true}\n"
        );
    }
}
