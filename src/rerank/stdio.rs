//! The reranker's stdin/stdout JSONL protocol (see PROTOCOL-rerank.md).
//!
//! `{"id": …, "query": …, "text": …}` in, `{"id": …, "score": …}` out. Every
//! rule the embedding protocol has holds here too: `id` is opaque and echoed,
//! a malformed line is skipped rather than fatal, a blank line means "score
//! what I have sent and answer now", and the answer to a batch ends with a
//! blank line of its own. Only the record shape differs.

use std::io::{BufWriter, Write};

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use super::Reranker;
use crate::protocol::{
    drive, parse_object, summarize, summary_facts, take_id, take_nonempty_str, Lazy, Records,
};
use crate::TokenInfo;

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

/// Writes one output record as a JSONL line. Both the stdio loop and `--pair`
/// use this output format. `tokens` is present only with `--report-tokens`,
/// which adds the `n_tokens` and `truncated` fields.
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

/// Parse one physical line. `Ok(None)` is a blank line — the batch boundary.
/// `Err` means the line is skipped with a warning. The envelope rules match
/// the embedding protocol; only the fields differ.
fn parse_line(line: &str) -> Result<Option<InRecord>, String> {
    let Some(obj) = parse_object(line)? else {
        return Ok(None);
    };
    Ok(Some(InRecord {
        id: take_id(&obj)?,
        query: take_nonempty_str(&obj, "query")?,
        text: take_nonempty_str(&obj, "text")?,
    }))
}

/// The scoring end of the protocol.
struct Score<W: Write, F> {
    model: Lazy<Reranker, F>,
    report_tokens: bool,
    out: W,
}

impl<W: Write, F: Fn() -> Result<Reranker>> Records for Score<W, F> {
    type Record = InRecord;
    type Answer = f32;

    fn parse(line: &str) -> Result<Option<InRecord>, String> {
        parse_line(line)
    }

    fn answer(&mut self, chunk: &[InRecord]) -> Result<(Vec<f32>, Vec<TokenInfo>)> {
        let pairs: Vec<(&str, &str)> = chunk
            .iter()
            .map(|r| (r.query.as_str(), r.text.as_str()))
            .collect();
        self.model.get()?.score(&pairs)
    }

    fn write(
        &mut self,
        record: &InRecord,
        answer: &Self::Answer,
        tokens: &TokenInfo,
    ) -> Result<()> {
        write_record(
            &mut self.out,
            &record.id,
            *answer,
            self.report_tokens.then_some(tokens),
        )
    }

    fn flush_output(&mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }

    /// A blank line, so a long-lived caller can read until it instead of
    /// counting records a skip would spoil.
    fn boundary(&mut self) -> Result<()> {
        self.out.write_all(b"\n")?;
        self.out.flush()?;
        Ok(())
    }
}

/// Run the protocol over stdin/stdout. Returns the number of skipped lines,
/// which the caller maps to exit code 2.
pub fn run(
    load: impl Fn() -> Result<Reranker>,
    report_tokens: bool,
    model_label: &str,
) -> Result<usize> {
    let stdout = std::io::stdout();
    let mut records = Score {
        model: Lazy::new(load),
        report_tokens,
        out: BufWriter::new(stdout.lock()),
    };

    let counts = drive(&mut records)?;
    records.out.flush()?;

    let facts = records
        .model
        .loaded()
        .map_or_else(|| "dim=0".to_string(), |r| summary_facts(&r.info()));
    summarize(model_label, &facts, &counts);
    Ok(counts.skipped)
}

/// Scores command-line pairs (`--pair`) instead of stdin input and writes
/// records as [`run`] does, using the pair positions as IDs.
///
/// Keeping the output code here ensures both paths use the same format. The
/// binary would otherwise use `serde_json::json!`, which converts `f32` scores
/// to `f64` and can print them differently.
///
/// Like `kohagi --text`, this prints no summary line.
pub fn run_pairs(reranker: &Reranker, pairs: &[(&str, &str)], report_tokens: bool) -> Result<()> {
    let (scores, tokens) = reranker.score(pairs)?;
    anyhow::ensure!(
        scores.len() == pairs.len(),
        "model returned {} scores for {} pairs",
        scores.len(),
        pairs.len()
    );
    anyhow::ensure!(
        tokens.len() == pairs.len(),
        "model returned token information for {} scores and {} pairs",
        tokens.len(),
        pairs.len()
    );

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for (id, (&score, info)) in scores.iter().zip(&tokens).enumerate() {
        write_record(
            &mut out,
            &Value::from(id),
            score,
            report_tokens.then_some(info),
        )?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(line: &str) -> InRecord {
        parse_line(line)
            .expect("a record")
            .expect("a record, not a batch boundary")
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
        assert!(matches!(parse_line(""), Ok(None)));
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
