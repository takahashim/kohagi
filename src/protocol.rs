//! Shared JSONL protocol support.

use std::io::BufRead;

use anyhow::{Context, Result};
use serde_json::{Map, Value};

use crate::program::remark;
use crate::TokenInfo;

/// Records handled in one chunk.
const CHUNK_ROWS: usize = 1024;

/// Parses one JSON object input line. Blank lines mark batch boundaries.
pub(crate) fn parse_object(line: &str) -> Result<Option<Map<String, Value>>, String> {
    if line.trim().is_empty() {
        return Ok(None);
    }
    let v: Value = serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    match v {
        Value::Object(obj) => Ok(Some(obj)),
        _ => Err("not a JSON object".to_string()),
    }
}

/// Returns the required opaque record ID.
pub(crate) fn take_id(obj: &Map<String, Value>) -> Result<Value, String> {
    Ok(obj.get("id").ok_or("missing \"id\"")?.clone())
}

/// Returns a required, nonempty string field.
pub(crate) fn take_nonempty_str(obj: &Map<String, Value>, name: &str) -> Result<String, String> {
    let s = obj
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string \"{name}\""))?;
    if s.is_empty() {
        return Err(format!("empty \"{name}\""));
    }
    Ok(s.to_string())
}

/// How many records a chunk produced, and how many were truncated.
pub(crate) struct Flushed {
    pub written: usize,
    pub truncated: usize,
}

/// Counts for one protocol run.
pub(crate) struct Counts {
    pub written: usize,
    pub skipped: usize,
    pub truncated: usize,
}

/// Loads a model only when the first record requires it.
pub(crate) struct Lazy<M, F> {
    model: Option<M>,
    load: F,
}

impl<M, F: Fn() -> Result<M>> Lazy<M, F> {
    pub(crate) fn new(load: F) -> Self {
        Self { model: None, load }
    }

    /// Returns the model, loading it if needed.
    pub(crate) fn get(&mut self) -> Result<&M> {
        if self.model.is_none() {
            self.model = Some((self.load)()?);
        }
        Ok(self.model.as_ref().expect("loaded just above"))
    }

    /// Returns the model if it has been loaded.
    pub(crate) fn loaded(&self) -> Option<&M> {
        self.model.as_ref()
    }
}

/// One end of the shared protocol. [`drive`] handles the input loop.
pub(crate) trait Records {
    type Record;
    type Answer;

    /// `Ok(Some)` is a record, `Ok(None)` is a batch boundary, and `Err` skips a line.
    fn parse(line: &str) -> Result<Option<Self::Record>, String>;

    /// Answers a chunk and returns matching token information.
    fn answer(&mut self, chunk: &[Self::Record]) -> Result<(Vec<Self::Answer>, Vec<TokenInfo>)>;

    /// Writes one answered record.
    fn write(
        &mut self,
        record: &Self::Record,
        answer: &Self::Answer,
        tokens: &TokenInfo,
    ) -> Result<()>;

    /// Flushes buffered output.
    fn flush_output(&mut self) -> Result<()>;

    /// Marks the end of a batch.
    fn boundary(&mut self) -> Result<()>;

    /// Answers, writes, and clears a chunk.
    fn flush(&mut self, chunk: &mut Vec<Self::Record>) -> Result<Flushed> {
        if chunk.is_empty() {
            return Ok(Flushed {
                written: 0,
                truncated: 0,
            });
        }
        let (answers, tokens) = self.answer(chunk)?;
        anyhow::ensure!(
            answers.len() == chunk.len(),
            "model returned {} answers for {} records",
            answers.len(),
            chunk.len()
        );
        anyhow::ensure!(
            tokens.len() == chunk.len(),
            "model returned token information for {} answers and {} records",
            tokens.len(),
            chunk.len()
        );

        let mut truncated = 0usize;
        for ((record, answer), info) in chunk.iter().zip(&answers).zip(&tokens) {
            truncated += info.truncated as usize;
            self.write(record, answer, info)?;
        }
        self.flush_output()?;

        let written = chunk.len();
        chunk.clear();
        Ok(Flushed { written, truncated })
    }
}

/// Reads stdin and processes records in chunks.
pub(crate) fn drive<R: Records>(records: &mut R) -> Result<Counts> {
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
            Ok(None) => {
                take(records.flush(&mut chunk)?, &mut counts);
                records.boundary()?;
            }
            Err(why) => {
                counts.skipped += 1;
                remark!("skip line {}: {why}", lineno + 1);
            }
        }
    }
    take(records.flush(&mut chunk)?, &mut counts);
    Ok(counts)
}

/// Writes the final summary line.
pub(crate) fn summarize(model_label: &str, facts: &str, counts: &Counts) {
    let n_in = counts.written + counts.skipped;
    remark!(
        "model={model_label} {facts} in={n_in} out={} skipped={} truncated={}",
        counts.written,
        counts.skipped,
        counts.truncated
    );
}

/// Returns the model-specific part of the summary line.
pub(crate) fn summary_facts(info: &crate::ModelInfo) -> String {
    let mut out = String::new();
    if let Some((claim, sha)) = info.digest() {
        out.push_str(&format!("{claim}={} ", crate::fingerprint::short(sha)));
    }
    out.push_str(&format!(
        "pooling={} dim={} max_seq={}",
        info.pooling,
        info.reported_dim(),
        info.max_seq_length
    ));
    if let crate::Output::Score { score } = info.output {
        out.push_str(&format!(" score={score}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BrokenRecords {
        answers: usize,
        tokens: usize,
    }

    impl Records for BrokenRecords {
        type Record = ();
        type Answer = ();

        fn parse(_line: &str) -> Result<Option<Self::Record>, String> {
            Ok(None)
        }

        fn answer(&mut self, _chunk: &[Self::Record]) -> Result<(Vec<()>, Vec<TokenInfo>)> {
            Ok((
                vec![(); self.answers],
                vec![
                    TokenInfo {
                        n_tokens: 0,
                        truncated: false,
                    };
                    self.tokens
                ],
            ))
        }

        fn write(
            &mut self,
            _record: &Self::Record,
            _answer: &Self::Answer,
            _tokens: &TokenInfo,
        ) -> Result<()> {
            Ok(())
        }

        fn flush_output(&mut self) -> Result<()> {
            Ok(())
        }

        fn boundary(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn flush_rejects_mismatched_answer_data() {
        for records in [
            BrokenRecords {
                answers: 0,
                tokens: 1,
            },
            BrokenRecords {
                answers: 1,
                tokens: 0,
            },
        ] {
            let mut records = records;
            let mut chunk = vec![()];
            assert!(records.flush(&mut chunk).is_err());
            assert_eq!(chunk.len(), 1);
        }
    }
}
