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
    /// Records in the batch being written now, which is also the next `index`.
    written: usize,
    prompt_tokens: usize,
    /// An OpenAI envelope has been opened and not yet closed.
    open: bool,
    /// Batches completed, so that a run with no records at all still produces
    /// one (empty) response rather than nothing.
    batches: usize,
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
            open: false,
            batches: 0,
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
                if !self.open {
                    self.out.write_all(b"{\"object\":\"list\",\"data\":[")?;
                    self.open = true;
                } else {
                    self.out.write_all(b",")?;
                }
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

    /// End the current batch and say so, so that a caller waiting on a reply
    /// knows it has everything.
    ///
    /// Without this a long-lived reader has to count records, and a record that
    /// was skipped for being malformed would leave it waiting for one that is
    /// never coming. For JSONL the marker is a blank line, which no record can
    /// be; for the OpenAI shape it is the close of that batch's response, since
    /// one flush is one request's worth.
    pub fn boundary(&mut self) -> Result<()> {
        match self.format {
            Format::Jsonl => self.out.write_all(b"\n")?,
            Format::OpenAi => self.close_document()?,
        }
        self.out.flush()?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }

    /// Close the document. For JSONL there is nothing to close; for OpenAI this
    /// is where `model` and `usage` go, since the token total is only known now.
    pub fn finish(mut self) -> Result<()> {
        // Close what is open; if nothing ever was, a run that embedded nothing
        // should still answer with an empty response rather than an empty file.
        if self.format == Format::OpenAi && (self.open || self.batches == 0) {
            self.close_document()?;
        }
        self.out.flush()?;
        Ok(())
    }

    /// Write `]`, `model` and `usage`, and start counting the next batch.
    fn close_document(&mut self) -> Result<()> {
        if !self.open {
            self.out.write_all(b"{\"object\":\"list\",\"data\":[")?;
        }
        self.out.write_all(b"],\"model\":")?;
        serde_json::to_writer(&mut self.out, &self.model)?;
        writeln!(
            self.out,
            ",\"usage\":{{\"prompt_tokens\":{n},\"total_tokens\":{n}}}}}",
            n = self.prompt_tokens
        )?;
        self.open = false;
        self.batches += 1;
        self.written = 0;
        self.prompt_tokens = 0;
        Ok(())
    }
}

/// One accepted input line: the opaque id and the raw text.
struct InRecord {
    id: Value,
    text: String,
}

/// Parse one physical line. `Ok(None)` is a blank line — embed whatever is
/// buffered now rather than waiting for a full chunk. `Err` = skip with a
/// warning (malformed JSON, missing id, empty/missing text).
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
pub(crate) struct Flushed {
    pub written: usize,
    pub truncated: usize,
}

/// What a whole run produced, for the summary line.
pub(crate) struct Counts {
    pub written: usize,
    pub skipped: usize,
    pub truncated: usize,
}

/// One end of the protocol: what a line means here, and what to do with a batch
/// of them. [`drive`] supplies everything the two ends share.
///
/// `kohagi` reads `{"id","text"}` and answers with embeddings; `kohagi-rerank`
/// reads `{"id","query","text"}` and answers with scores. Every other rule is
/// the same rule — an opaque echoed `id`, a malformed line skipped rather than
/// fatal, a blank line meaning "answer what I have sent", a blank line back to
/// mark the reply, output flushed per chunk, and the model loaded on first use
/// so input with no valid records never loads it. Stated once here, because two
/// statements of it are two things to keep in step with PROTOCOL.md.
pub(crate) trait Records {
    type Record;

    /// `Ok(Some)` a record, `Ok(None)` a batch boundary, `Err` a skip reason.
    fn parse(line: &str) -> Result<Option<Self::Record>, String>;

    /// Answer the buffered records, write them out, and empty the buffer.
    fn flush(&mut self, chunk: &mut Vec<Self::Record>) -> Result<Flushed>;

    /// Say that a batch is complete, so a reader knows not to wait for more.
    fn boundary(&mut self) -> Result<()>;
}

/// Read the protocol off stdin and hand it to `records`, chunk by chunk.
///
/// Does not write the summary or close the output: both need what the caller
/// knows and this does not — which model answered, and whether the output shape
/// has a document to close.
pub(crate) fn drive<R: Records>(records: &mut R, program: &str) -> Result<Counts> {
    let stdin = std::io::stdin();
    let mut chunk: Vec<R::Record> = Vec::new();
    let mut counts = Counts {
        written: 0,
        skipped: 0,
        truncated: 0,
    };
    let take = |f: Flushed, counts: &mut Counts| {
        counts.written += f.written;
        counts.truncated += f.truncated;
    };

    for (lineno, line) in stdin.lock().lines().enumerate() {
        let line = line.context("reading stdin")?;
        match R::parse(&line) {
            Ok(Some(record)) => {
                chunk.push(record);
                if chunk.len() >= CHUNK_ROWS {
                    take(records.flush(&mut chunk)?, &mut counts);
                }
            }
            // A blank line is the caller saying "that is one batch": answer what
            // is buffered even though the chunk is short, and mark the boundary
            // so a reader knows not to wait for more.
            Ok(None) => {
                take(records.flush(&mut chunk)?, &mut counts);
                records.boundary()?;
            }
            Err(why) => {
                counts.skipped += 1;
                eprintln!("{program}: skip line {}: {why}", lineno + 1);
            }
        }
    }
    take(records.flush(&mut chunk)?, &mut counts);
    Ok(counts)
}

/// The embedding end of the protocol: prefix each text, embed the chunk, write
/// the vectors.
struct Embed<'a, W: Write, F> {
    embedder: Option<Embedder>,
    load: F,
    prefix: &'a str,
    out: Writer<W>,
}

impl<W: Write, F: Fn() -> Result<Embedder>> Records for Embed<'_, W, F> {
    type Record = InRecord;

    fn parse(line: &str) -> Result<Option<InRecord>, String> {
        parse_line(line)
    }

    fn flush(&mut self, chunk: &mut Vec<InRecord>) -> Result<Flushed> {
        if chunk.is_empty() {
            return Ok(Flushed {
                written: 0,
                truncated: 0,
            });
        }
        let embedder = match &mut self.embedder {
            Some(e) => e,
            None => self.embedder.insert((self.load)()?),
        };

        let prefixed: Vec<String> = chunk
            .iter()
            .map(|r| format!("{}{}", self.prefix, r.text))
            .collect();
        let texts: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        let (vectors, tokens) = embedder.embed_with_tokens(&texts)?;

        let mut truncated = 0usize;
        // Each record is written as one complete line, so an abort can never
        // leave a half-written line for the caller to misread.
        for ((record, vector), info) in chunk.iter().zip(&vectors).zip(&tokens) {
            truncated += info.truncated as usize;
            self.out.record(&record.id, vector, info)?;
        }
        // Flush per chunk so the caller can consume output as it is produced.
        self.out.flush()?;

        let written = chunk.len();
        chunk.clear();
        Ok(Flushed { written, truncated })
    }

    fn boundary(&mut self) -> Result<()> {
        self.out.boundary()
    }
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
    let stdout = std::io::stdout();
    let mut records = Embed {
        embedder: None,
        load,
        prefix,
        out: Writer::new(
            BufWriter::new(stdout.lock()),
            format,
            model_label,
            report_tokens,
        ),
    };

    let counts = drive(&mut records, "kohagi")?;
    records.out.finish()?;

    // `in` counts record lines (blank lines are ignored entirely); with no
    // valid input the model was never loaded, and there is nothing to say
    // about it beyond the name that would have been used.
    let facts = records
        .embedder
        .as_ref()
        .map_or_else(|| "dim=0".to_string(), |e| summary_facts(&e.info()));
    summarize("kohagi", model_label, &facts, &counts);
    Ok(counts.skipped)
}

/// The stderr line every run ends with, in the one shape a log parser has to
/// know.
pub(crate) fn summarize(program: &str, model_label: &str, facts: &str, counts: &Counts) {
    let n_in = counts.written + counts.skipped;
    eprintln!(
        "{program}: model={model_label} {facts} in={n_in} out={} skipped={} truncated={}",
        counts.written, counts.skipped, counts.truncated
    );
}

/// The model's part of the summary line: which weights, resolved how.
///
/// Enough to reconstruct what produced the vectors from a captured log alone —
/// the fingerprint answers *which* checkpoint, and pooling and max_seq answer
/// the two settings that silently change every vector when they differ.
pub(crate) fn summary_facts(info: &crate::ModelInfo) -> String {
    let mut out = String::new();
    if let Some(sha) = &info.sha256 {
        out.push_str(&format!("sha256={} ", crate::fingerprint::short(sha)));
    }
    // A CoreML bundle has no weights of its own to hash, so it reports the
    // checkpoint it was converted from — under a different key, because it is
    // a different claim.
    if let Some(sha) = &info.source_sha256 {
        out.push_str(&format!(
            "source_sha256={} ",
            crate::fingerprint::short(sha)
        ));
    }
    // The dimension the vectors actually have: `--dims` when it truncated
    // them, the model's own otherwise. A log line saying 512 about a 256-dim
    // run would defeat the line's purpose.
    out.push_str(&format!(
        "pooling={} dim={} max_seq={}",
        info.pooling,
        info.output_dim.unwrap_or(info.dim),
        info.max_seq_length
    ));
    // A reranker's numbers mean different things either side of the sigmoid,
    // so a log of scores has to say which it holds.
    if let Some(score) = info.score {
        out.push_str(&format!(" score={score}"));
    }
    out
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
    fn parse_accepts_int_and_string_ids() {
        let r = record(r#"{"id": 123, "text": "hello"}"#);
        assert_eq!(r.id, Value::from(123));
        assert_eq!(r.text, "hello");
        let r = record(r#"{"id": "b-9", "text": "改行\nあり"}"#);
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

    /// A blank line ends a batch, and the boundary is visible on the way out —
    /// which is what lets a long-lived caller read a reply without counting
    /// records that a skipped line would have made it wait for.
    #[test]
    fn a_blank_line_ends_a_batch_in_both_formats() {
        fn batches(format: Format) -> Vec<u8> {
            let mut buf = Vec::new();
            let mut w = Writer::new(&mut buf, format, "some/model", false);
            let info = TokenInfo {
                n_tokens: 4,
                truncated: false,
            };
            w.record(&Value::from(0), &[1.0], &info).unwrap();
            w.boundary().unwrap();
            w.record(&Value::from(1), &[2.0], &info).unwrap();
            w.finish().unwrap();
            buf
        }

        assert_eq!(
            String::from_utf8(batches(Format::Jsonl)).unwrap(),
            "{\"id\":0,\"embedding\":[1.0]}\n\n{\"id\":1,\"embedding\":[2.0]}\n"
        );

        // One flush is one request's worth, so each is a whole response with its
        // own `index` from zero and its own `usage`.
        let text = String::from_utf8(batches(Format::OpenAi)).unwrap();
        let docs: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("each line is a document"))
            .collect();
        assert_eq!(docs.len(), 2);
        for d in &docs {
            assert_eq!(d["data"].as_array().unwrap().len(), 1);
            assert_eq!(d["data"][0]["index"], 0);
            assert_eq!(d["usage"]["prompt_tokens"], 4);
        }
    }

    /// An empty flush still answers, so a caller cannot hang waiting for a reply
    /// to a request that happened to have nothing in it.
    #[test]
    fn an_empty_batch_still_answers() {
        for format in [Format::Jsonl, Format::OpenAi] {
            let mut buf = Vec::new();
            let mut w = Writer::new(&mut buf, format, "some/model", false);
            w.boundary().unwrap();
            w.finish().unwrap();
            assert_eq!(
                String::from_utf8(buf).unwrap().lines().count(),
                1,
                "{format:?}"
            );
        }
    }

    fn info() -> crate::ModelInfo {
        crate::ModelInfo {
            backend: "cpu",
            precision: "f32",
            sha256: Some("0123456789abcdef0123456789abcdef".to_string()),
            source: None,
            source_sha256: None,
            buckets: None,
            quantization: None,
            graph_version: None,
            pooling: "mean",
            dim: 512,
            output_dim: None,
            max_seq_length: 512,
            score: None,
        }
    }

    /// The summary is where a captured log says which weights answered, so the
    /// two paths spell their fingerprints differently: a checkpoint's own hash
    /// and a bundle's report of the checkpoint behind it are not the same
    /// claim, and a log that conflated them would be worse than one with
    /// neither.
    #[test]
    fn the_summary_says_which_weights_it_used() {
        assert_eq!(
            summary_facts(&info()),
            "sha256=0123456789ab pooling=mean dim=512 max_seq=512"
        );

        let coreml = crate::ModelInfo {
            backend: "coreml",
            sha256: None,
            source_sha256: Some("fedcba9876543210fedcba9876543210".to_string()),
            ..info()
        };
        assert_eq!(
            summary_facts(&coreml),
            "source_sha256=fedcba987654 pooling=mean dim=512 max_seq=512"
        );

        // A bundle that records no provenance says nothing about one, rather
        // than reporting its own identity as the checkpoint's.
        let unknown = crate::ModelInfo {
            source_sha256: None,
            ..coreml
        };
        assert_eq!(summary_facts(&unknown), "pooling=mean dim=512 max_seq=512");
    }

    /// `--dims` changes what every vector is, so the summary's `dim=` reports
    /// what came out, not what the model would have produced.
    #[test]
    fn the_summary_reports_the_truncated_dimension() {
        let truncated = crate::ModelInfo {
            output_dim: Some(256),
            ..info()
        };
        assert_eq!(
            summary_facts(&truncated),
            "sha256=0123456789ab pooling=mean dim=256 max_seq=512"
        );
    }

    /// What `--print-model-info` writes, which evaluation scripts read into
    /// their results files: every key present, and the ones that do not apply
    /// to this path absent rather than null.
    #[test]
    fn the_model_info_json_omits_what_does_not_apply() {
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&info()).unwrap()).unwrap();
        assert_eq!(json["backend"], "cpu");
        assert_eq!(json["precision"], "f32");
        assert_eq!(json["pooling"], "mean");
        assert_eq!(json["dim"], 512);
        assert_eq!(json["max_seq_length"], 512);
        assert_eq!(json["sha256"], "0123456789abcdef0123456789abcdef");
        for absent in [
            "source",
            "source_sha256",
            "buckets",
            "quantization",
            "output_dim",
        ] {
            assert!(json.get(absent).is_none(), "{absent} should be omitted");
        }

        // With `--dims`, the JSON says both what the model is and what was
        // produced: `dim` stays the model's own, `output_dim` is the flag's.
        let truncated = crate::ModelInfo {
            output_dim: Some(256),
            ..info()
        };
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&truncated).unwrap()).unwrap();
        assert_eq!(json["dim"], 512);
        assert_eq!(json["output_dim"], 256);
    }

    #[test]
    fn parse_skips_bad_lines_and_ignores_blank() {
        assert!(matches!(parse_line(""), Ok(None)));
        assert!(matches!(parse_line("   "), Ok(None)));
        assert!(parse_line("not json").is_err());
        assert!(parse_line(r#"[1,2]"#).is_err());
        assert!(parse_line(r#"{"text": "no id"}"#).is_err());
        assert!(parse_line(r#"{"id": 1}"#).is_err());
        assert!(parse_line(r#"{"id": 1, "text": ""}"#).is_err());
        assert!(parse_line(r#"{"id": 1, "text": 5}"#).is_err());
    }
}
