//! The stdin/stdout JSONL protocol (see PROTOCOL.md).
//!
//! One record per line: `{"id": …, "text": "…"}` in, `{"id": …, "embedding":
//! […]}` out. `id` is opaque and echoed verbatim — callers map results by id,
//! not by order. Kohagi only prepends the configured prefix and embeds; text
//! shaping (trimming, truncation by characters, dedup) is the caller's job,
//! so an id's embedding always corresponds to exactly the text that was sent.
//! stdout carries records only; warnings and the final summary go to stderr.

use std::io::{BufRead, BufWriter, Write};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;

use crate::{Embedder, TokenInfo};

/// Records encoded (and written out) per chunk. Bounds resident memory to one
/// chunk's texts + embeddings instead of the whole input, while leaving
/// plenty of rows for length bucketing and the parallel fan-out. Output is
/// flushed after each chunk, so callers can consume it incrementally.
const CHUNK_ROWS: usize = 1024;

#[derive(Serialize)]
struct OutRecord<'a> {
    id: &'a Value,
    embedding: &'a [f32],
    /// Present only under `--report-tokens`; omitted entirely otherwise, so the
    /// default output is byte-for-byte the protocol-1 shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    n_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncated: Option<bool>,
}

/// Write one output record as a JSONL line — the single definition of the
/// protocol's output shape, shared by the stdio loop and the `--text` one-shot
/// path. `tokens` is `Some` only under `--report-tokens`, which adds the
/// `n_tokens` / `truncated` fields; `None` keeps the plain protocol-1 shape.
pub fn write_record(
    out: &mut impl Write,
    id: &Value,
    embedding: &[f32],
    tokens: Option<&TokenInfo>,
) -> Result<()> {
    serde_json::to_writer(
        &mut *out,
        &OutRecord {
            id,
            embedding,
            n_tokens: tokens.map(|t| t.n_tokens),
            truncated: tokens.map(|t| t.truncated),
        },
    )?;
    out.write_all(b"\n")?;
    Ok(())
}

/// The shape stdout takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Format {
    /// One `{"id", "embedding"}` object per line — PROTOCOL.md.
    #[default]
    Jsonl,
    /// One OpenAI `/v1/embeddings` response object for the whole run, so that
    /// code written against that API can read Kohagi's output unchanged.
    OpenAi,
}

/// One item of an OpenAI response's `data` array.
///
/// `index` is the position among the records that were embedded, which is what
/// the API means by it. Input ids have no place in this shape: a caller that
/// needs them wants [`Format::Jsonl`].
#[derive(Serialize)]
struct OpenAiItem<'a> {
    object: &'static str,
    index: usize,
    embedding: &'a [f32],
}

/// Writes records in whichever [`Format`] was asked for.
///
/// The OpenAI shape is a single object holding every embedding, which would
/// ordinarily mean buffering the whole run. It does not here: the envelope is
/// written in pieces — head, then one item per record, then `model` and `usage`
/// — so resident memory stays one chunk's worth either way. The cost is that an
/// aborted run leaves an incomplete JSON document, where JSONL would have left
/// a shorter but valid one.
pub struct Writer<W: Write> {
    out: W,
    format: Format,
    model: String,
    report_tokens: bool,
    written: usize,
    prompt_tokens: usize,
}

impl<W: Write> Writer<W> {
    pub fn new(out: W, format: Format, model: &str, report_tokens: bool) -> Self {
        Self {
            out,
            format,
            model: model.to_string(),
            report_tokens,
            written: 0,
            prompt_tokens: 0,
        }
    }

    /// Write one embedding. `tokens` is always supplied — the OpenAI shape needs
    /// the count for `usage` whether or not `--report-tokens` asked to see it.
    pub fn record(&mut self, id: &Value, embedding: &[f32], tokens: &TokenInfo) -> Result<()> {
        self.prompt_tokens += tokens.n_tokens;
        match self.format {
            Format::Jsonl => {
                write_record(
                    &mut self.out,
                    id,
                    embedding,
                    self.report_tokens.then_some(tokens),
                )?;
            }
            Format::OpenAi => {
                self.out.write_all(if self.written == 0 {
                    b"{\"object\":\"list\",\"data\":["
                } else {
                    b","
                })?;
                serde_json::to_writer(
                    &mut self.out,
                    &OpenAiItem {
                        object: "embedding",
                        index: self.written,
                        embedding,
                    },
                )?;
            }
        }
        self.written += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }

    /// Close the document. For JSONL there is nothing to close; for OpenAI this
    /// is where `model` and `usage` go, since the token total is only known now.
    pub fn finish(mut self) -> Result<()> {
        if self.format == Format::OpenAi {
            if self.written == 0 {
                self.out.write_all(b"{\"object\":\"list\",\"data\":[")?;
            }
            self.out.write_all(b"],\"model\":")?;
            serde_json::to_writer(&mut self.out, &self.model)?;
            writeln!(
                self.out,
                ",\"usage\":{{\"prompt_tokens\":{n},\"total_tokens\":{n}}}}}",
                n = self.prompt_tokens
            )?;
        }
        self.out.flush()?;
        Ok(())
    }
}

/// One accepted input line: the opaque id and the raw text.
struct InRecord {
    id: Value,
    text: String,
}

/// Parse one physical line. `Ok(None)` = blank line (ignored, not counted);
/// `Err` = skip with a warning (malformed JSON, missing id, empty/missing text).
fn parse_line(line: &str) -> Result<Option<InRecord>, String> {
    if line.trim().is_empty() {
        return Ok(None);
    }
    let v: Value = serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = v.as_object().ok_or("not a JSON object")?;
    let id = obj.get("id").ok_or("missing \"id\"")?.clone();
    let text = obj
        .get("text")
        .and_then(Value::as_str)
        .ok_or("missing or non-string \"text\"")?;
    if text.is_empty() {
        return Err("empty \"text\"".to_string());
    }
    Ok(Some(InRecord {
        id,
        text: text.to_string(),
    }))
}

/// How many records a chunk produced, and how many of them were truncated.
struct Flushed {
    written: usize,
    truncated: usize,
}

/// Embed the buffered chunk, write its output lines, and empty the buffer.
///
/// The model is loaded here on first use, so input with no valid records
/// never loads it at all. Each record is written as one complete line, so an
/// abort can never leave a half-written line for the caller to misread.
/// `report_tokens` adds the `n_tokens` / `truncated` fields to each record; the
/// truncated count is tallied for the summary regardless.
fn flush_chunk(
    embedder: &mut Option<Embedder>,
    load: &impl Fn() -> Result<Embedder>,
    prefix: &str,
    chunk: &mut Vec<InRecord>,
    out: &mut Writer<impl Write>,
) -> Result<Flushed> {
    if chunk.is_empty() {
        return Ok(Flushed {
            written: 0,
            truncated: 0,
        });
    }
    let embedder = match embedder {
        Some(e) => e,
        None => embedder.insert(load()?),
    };

    let prefixed: Vec<String> = chunk
        .iter()
        .map(|r| format!("{prefix}{}", r.text))
        .collect();
    let texts: Vec<&str> = prefixed.iter().map(String::as_str).collect();
    let (vectors, tokens) = embedder.embed_with_tokens(&texts)?;

    let mut truncated = 0usize;
    for ((record, vector), info) in chunk.iter().zip(&vectors).zip(&tokens) {
        truncated += info.truncated as usize;
        out.record(&record.id, vector, info)?;
    }
    // Flush per chunk so the caller can consume output as it is produced.
    out.flush()?;

    let written = chunk.len();
    chunk.clear();
    Ok(Flushed { written, truncated })
}

/// Run the protocol over stdin/stdout. Returns the number of skipped lines —
/// the caller maps >0 to exit code 2; fatal errors (model load, I/O) return
/// `Err` (exit 1).
pub fn run(
    load: impl Fn() -> Result<Embedder>,
    prefix: &str,
    report_tokens: bool,
    model_label: &str,
    format: Format,
) -> Result<usize> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = Writer::new(
        BufWriter::new(stdout.lock()),
        format,
        model_label,
        report_tokens,
    );

    let mut embedder: Option<Embedder> = None;
    let mut chunk: Vec<InRecord> = Vec::new();
    let mut n_out = 0usize;
    let mut skipped = 0usize;
    let mut truncated = 0usize;

    for (lineno, line) in stdin.lock().lines().enumerate() {
        let line = line.context("reading stdin")?;
        match parse_line(&line) {
            Ok(Some(record)) => {
                chunk.push(record);
                if chunk.len() >= CHUNK_ROWS {
                    let f = flush_chunk(&mut embedder, &load, prefix, &mut chunk, &mut out)?;
                    n_out += f.written;
                    truncated += f.truncated;
                }
            }
            Ok(None) => {}
            Err(why) => {
                skipped += 1;
                eprintln!("kohagi: skip line {}: {why}", lineno + 1);
            }
        }
    }
    let f = flush_chunk(&mut embedder, &load, prefix, &mut chunk, &mut out)?;
    out.finish()?;
    n_out += f.written;
    truncated += f.truncated;

    // `in` counts record lines (blank lines are ignored entirely); with no
    // valid input the model was never loaded and dim is unknown (0).
    let dim = embedder.as_ref().map_or(0, Embedder::dim);
    let n_in = n_out + skipped;
    eprintln!(
        "kohagi: model={model_label} dim={dim} in={n_in} out={n_out} \
         skipped={skipped} truncated={truncated}"
    );
    Ok(skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_int_and_string_ids() {
        let r = parse_line(r#"{"id": 123, "text": "hello"}"#)
            .unwrap()
            .unwrap();
        assert_eq!(r.id, Value::from(123));
        assert_eq!(r.text, "hello");
        let r = parse_line(r#"{"id": "b-9", "text": "改行\nあり"}"#)
            .unwrap()
            .unwrap();
        assert_eq!(r.id, Value::from("b-9"));
        assert_eq!(r.text, "改行\nあり");
    }

    #[test]
    fn write_record_omits_token_fields_unless_reported() {
        let id = Value::from(1);
        let v = [0.5f32, 0.5];

        // Default: the plain protocol-1 shape, byte for byte plus the newline.
        let mut plain = Vec::new();
        write_record(&mut plain, &id, &v, None).unwrap();
        assert_eq!(plain, b"{\"id\":1,\"embedding\":[0.5,0.5]}\n");

        // --report-tokens: both fields appear, `truncated: false` included.
        let mut full = Vec::new();
        let info = TokenInfo {
            n_tokens: 7,
            truncated: false,
        };
        write_record(&mut full, &id, &v, Some(&info)).unwrap();
        assert_eq!(
            full,
            b"{\"id\":1,\"embedding\":[0.5,0.5],\"n_tokens\":7,\"truncated\":false}\n"
        );
    }

    /// The OpenAI shape, including the two things that make it awkward to
    /// stream: the envelope has to open before the first record and close after
    /// the last, and `usage` is only known at the end.
    #[test]
    fn the_openai_shape_is_one_document_written_in_pieces() {
        fn run(n: usize) -> serde_json::Value {
            let mut buf = Vec::new();
            let mut w = Writer::new(&mut buf, Format::OpenAi, "some/model", false);
            for i in 0..n {
                let info = TokenInfo {
                    n_tokens: 3 + i,
                    truncated: false,
                };
                w.record(&Value::from(format!("id-{i}")), &[0.5, -0.5], &info)
                    .unwrap();
            }
            w.finish().unwrap();
            serde_json::from_slice(&buf).expect("one valid JSON document")
        }

        // Empty input is still a document, not an empty file.
        let none = run(0);
        assert_eq!(none["object"], "list");
        assert_eq!(none["data"].as_array().unwrap().len(), 0);
        assert_eq!(none["usage"]["prompt_tokens"], 0);

        let two = run(2);
        assert_eq!(two["model"], "some/model");
        let data = two["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        // Position, not the input id — that is what the API means by `index`.
        assert_eq!(data[0]["index"], 0);
        assert_eq!(data[1]["index"], 1);
        assert_eq!(data[1]["object"], "embedding");
        assert!(data[0].get("id").is_none());
        // usage totals every record, whatever `--report-tokens` asked for.
        assert_eq!(two["usage"]["prompt_tokens"], 7);
        assert_eq!(two["usage"]["total_tokens"], 7);
    }

    /// The default shape is unchanged by the writer sitting in front of it.
    #[test]
    fn the_jsonl_writer_still_emits_protocol_1_lines() {
        let info = TokenInfo {
            n_tokens: 7,
            truncated: true,
        };
        let mut plain = Vec::new();
        let mut w = Writer::new(&mut plain, Format::Jsonl, "some/model", false);
        w.record(&Value::from(1), &[0.5, 0.5], &info).unwrap();
        w.finish().unwrap();
        assert_eq!(plain, b"{\"id\":1,\"embedding\":[0.5,0.5]}\n");

        let mut full = Vec::new();
        let mut w = Writer::new(&mut full, Format::Jsonl, "some/model", true);
        w.record(&Value::from(1), &[0.5, 0.5], &info).unwrap();
        w.finish().unwrap();
        assert_eq!(
            full,
            b"{\"id\":1,\"embedding\":[0.5,0.5],\"n_tokens\":7,\"truncated\":true}\n"
        );
    }

    #[test]
    fn parse_skips_bad_lines_and_ignores_blank() {
        assert!(parse_line("").unwrap().is_none());
        assert!(parse_line("   ").unwrap().is_none());
        assert!(parse_line("not json").is_err());
        assert!(parse_line(r#"[1,2]"#).is_err());
        assert!(parse_line(r#"{"text": "no id"}"#).is_err());
        assert!(parse_line(r#"{"id": 1}"#).is_err());
        assert!(parse_line(r#"{"id": 1, "text": ""}"#).is_err());
        assert!(parse_line(r#"{"id": 1, "text": 5}"#).is_err());
    }
}
