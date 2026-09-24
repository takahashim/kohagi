//! `kohagi-decide`'s stdin/stdout protocol (see PROTOCOL-decide.md).
//!
//! `{"id": …, "state": …, "questions": {name: …}}` in,
//! `{"id": …, "answers": {name: …}}` out. Every envelope rule is the rerank
//! protocol's: `id` is opaque and echoed, a malformed line is skipped rather
//! than fatal, a blank line means "answer what I have sent and reply now", and
//! a reply ends with a blank line of its own. Only the records differ.

use std::io::{BufWriter, Write};

use anyhow::Result;
use serde::Serialize;
use serde_json::{Map, Value};

use super::engine::Engine;
use super::{parse_request, typed_answer, Request};
use crate::protocol::{drive, summarize, Lazy, Records};
use crate::TokenInfo;

#[derive(Serialize)]
struct OutRecord<'a> {
    id: &'a Value,
    answers: &'a Value,
}

/// The answering end of the protocol.
struct Decide<W: Write, F> {
    engine: Lazy<Engine, F>,
    out: W,
}

impl<W: Write, F: Fn() -> Result<Engine>> Records for Decide<W, F> {
    type Record = Request;
    type Answer = Value;

    fn parse(line: &str) -> Result<Option<Request>, String> {
        parse_request(line)
    }

    fn answer(&mut self, chunk: &[Request]) -> Result<(Vec<Value>, Vec<TokenInfo>)> {
        let (scored, info) = self.engine.get()?.score(chunk)?;
        anyhow::ensure!(
            scored.len() == chunk.len(),
            "head returned {} requests for {} requests",
            scored.len(),
            chunk.len(),
        );
        let answers = chunk
            .iter()
            .zip(scored)
            .map(|(request, questions)| {
                anyhow::ensure!(
                    questions.len() == request.questions.len(),
                    "request {:?}: head returned {} questions; expected {}",
                    request.id,
                    questions.len(),
                    request.questions.len(),
                );
                let mut by_name = Map::new();
                for (question, labelled) in request.questions.iter().zip(questions) {
                    by_name.insert(question.name.clone(), typed_answer(question, &labelled)?);
                }
                Ok(Value::Object(by_name))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((answers, info))
    }

    fn write(&mut self, record: &Request, answer: &Value, _tokens: &TokenInfo) -> Result<()> {
        serde_json::to_writer(
            &mut self.out,
            &OutRecord {
                id: &record.id,
                answers: answer,
            },
        )?;
        self.out.write_all(b"\n")?;
        Ok(())
    }

    fn flush_output(&mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }

    fn boundary(&mut self) -> Result<()> {
        self.out.write_all(b"\n")?;
        self.out.flush()?;
        Ok(())
    }
}

/// Run the protocol over stdin/stdout. Returns the number of skipped lines,
/// which the caller maps to exit code 2.
pub fn run(load: impl Fn() -> Result<Engine>, model_label: &str) -> Result<usize> {
    let stdout = std::io::stdout();
    let mut records = Decide {
        engine: Lazy::new(load),
        out: BufWriter::new(stdout.lock()),
    };

    let counts = drive(&mut records)?;
    records.out.flush()?;

    let facts = records
        .engine
        .loaded()
        .map_or_else(|| "dim=0".to_string(), |e| e.info().summary_facts());
    summarize(model_label, &facts, &counts);
    Ok(counts.skipped)
}
