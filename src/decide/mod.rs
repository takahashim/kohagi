//! Typed decisions: a state plus typed questions in, `choice` / `score` /
//! `noul` out.
//!
//! The request and the answer are model-independent. What varies between models
//! is the *head*: how a question and its options become something the encoder
//! scores. [`engine::Engine`] is that choice, selected with `--head`, and the
//! checkpoint is selected independently with `--model-id`. Heads register in
//! [`engine::factories`], so one is added or removed without touching the rest:
//!
//! - `cross-encoder`: a one-label cross-encoder scores `(context, candidate)`
//!   pairs independently (the `argos1111` Jev-local shape, Japanese);
//! - `laya`: a packed encoder reads state, instructions and every option in one
//!   pass, and a small decision head scores the marked options (the
//!   `convaiinnovations/laya` shape, English and multilingual). Behind the
//!   `head-laya` feature, which is on by default.
//!
//! Both answer the same three shapes and are softmaxed the same way, per
//! question. v1's contract is the decision: the probabilities and the
//! `choice`/`score`/`noul` they imply. The escalation head (`act_probability`)
//! is deliberately absent; a later version adds a field. See PROTOCOL-decide.md.

pub mod cross;
pub mod engine;
#[cfg(feature = "head-laya")]
pub mod laya;
pub mod stdio;

use anyhow::Result;
use serde_json::{Map, Value};

use crate::protocol::{parse_object, take_id, take_nonempty_str};

/// One option's canonical label and calibrated, pre-softmax logit.
pub type OptionScore = (String, f32);
/// One question's scored options, in the question's canonical order. The
/// softmax runs here.
pub type QuestionScore = Vec<OptionScore>;
/// One request's questions, in the request's order.
pub type RequestScore = Vec<QuestionScore>;

/// The three typed question shapes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    fn parse(name: &str) -> Result<Self> {
        match name {
            "choice" => Ok(Self::Choice),
            "score" => Ok(Self::Score),
            "noul" => Ok(Self::Noul),
            other => {
                anyhow::bail!("unknown question type `{other}`; expected choice, score or noul")
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

/// A question's options, before any head renders them.
pub enum Criteria {
    /// `choice`: labels in criteria order, each with an optional description.
    Choice(Vec<(String, Option<String>)>),
    /// `score`: ordered level descriptions.
    Score(Vec<String>),
    /// `noul`: no criteria; a head supplies its own true/false pair.
    Noul,
}

/// One typed question. The label set the answer is keyed by is derived from the
/// criteria; the head decides how those labels become scored text.
pub struct Question {
    pub name: String,
    pub qtype: QType,
    pub instructions: String,
    pub criteria: Criteria,
}

/// One decide request: an opaque id, a state, and named typed questions.
pub struct Request {
    pub id: Value,
    pub state: Value,
    pub questions: Vec<Question>,
}

/// The canonical candidate labels a head must return for this question.
///
/// `noul` deliberately has a fixed logical order even though individual heads
/// may render its two texts in another order. A head must associate each score
/// with this label before returning it.
pub fn expected_labels(question: &Question) -> Vec<String> {
    match &question.criteria {
        Criteria::Choice(options) => options.iter().map(|(label, _)| label.clone()).collect(),
        Criteria::Score(levels) => levels.clone(),
        Criteria::Noul => vec!["false".to_string(), "true".to_string()],
    }
}

/// A JSON value as text: strings verbatim, anything else serialized. Matches
/// the reference's `content()`, so a state that is an object renders the same.
pub fn content(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Softmax a slice of raw logits into probabilities. Used only over a closed
/// set: the candidates of one question.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// A probability, rounded to six places so a run's output is stable and short.
pub fn number(x: f32) -> Value {
    Value::from((f64::from(x) * 1e6).round() / 1e6)
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in xs.iter().enumerate() {
        if *x > xs[best] {
            best = i;
        }
    }
    best
}

/// The typed answer for one question, given its canonical `(label, logit)`
/// candidates. Softmaxing happens here, once, so both heads share it.
pub fn typed_answer(question: &Question, labelled: &[(String, f32)]) -> Result<Value> {
    let expected = expected_labels(question);
    anyhow::ensure!(
        labelled.len() == expected.len(),
        "question `{}`: head returned {} candidate scores; expected {}",
        question.name,
        labelled.len(),
        expected.len(),
    );
    for (index, ((label, _), expected_label)) in labelled.iter().zip(&expected).enumerate() {
        anyhow::ensure!(
            label == expected_label,
            "question `{}`: head candidate {index} is `{label}`; expected `{expected_label}`",
            question.name,
        );
    }

    let logits: Vec<f32> = labelled.iter().map(|(_, logit)| *logit).collect();
    let probs = softmax(&logits);
    let mut out = Map::new();
    out.insert("type".to_string(), Value::from(question.qtype.name()));
    match question.qtype {
        QType::Choice => {
            let best = argmax(&probs);
            let dist: Map<String, Value> = labelled
                .iter()
                .map(|(label, _)| label.clone())
                .zip(&probs)
                .map(|(label, p)| (label, number(*p)))
                .collect();
            out.insert("choice".to_string(), Value::from(labelled[best].0.clone()));
            out.insert("probability".to_string(), number(probs[best]));
            out.insert("probabilities".to_string(), Value::Object(dist));
        }
        QType::Score => {
            // The expected zero-based level, the scalar a rubric returns; the
            // whole distribution stays alongside.
            let expected: f64 = probs
                .iter()
                .enumerate()
                .map(|(level, p)| level as f64 * f64::from(*p))
                .sum();
            out.insert(
                "score".to_string(),
                Value::from((expected * 1e6).round() / 1e6),
            );
            out.insert(
                "probabilities".to_string(),
                Value::Array(probs.iter().map(|p| number(*p)).collect()),
            );
        }
        QType::Noul => {
            let dist: Map<String, Value> = labelled
                .iter()
                .map(|(label, _)| label.clone())
                .zip(&probs)
                .map(|(label, p)| (label, number(*p)))
                .collect();
            let p_true = probs[1]; // `expected_labels` has verified false, true.
            out.insert("noul".to_string(), number(p_true));
            out.insert("probabilities".to_string(), Value::Object(dist));
        }
    }
    Ok(Value::Object(out))
}

/// Parse one input line. `Ok(None)` is a blank line (batch boundary); `Err`
/// means the line is skipped, as everywhere in the protocol.
fn parse_request(line: &str) -> Result<Option<Request>, String> {
    let Some(obj) = parse_object(line)? else {
        return Ok(None);
    };
    let id = take_id(&obj)?;
    let state = obj.get("state").cloned().ok_or("missing \"state\"")?;
    let questions = obj
        .get("questions")
        .and_then(Value::as_object)
        .ok_or("\"questions\" must be an object")?;
    if questions.is_empty() {
        return Err("\"questions\" is empty".to_string());
    }
    let parsed = questions
        .iter()
        .map(|(name, spec)| parse_question(name, spec))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(Request {
        id,
        state,
        questions: parsed,
    }))
}

fn parse_question(name: &str, spec: &Value) -> Result<Question, String> {
    let spec = spec
        .as_object()
        .ok_or_else(|| format!("question `{name}` must be an object"))?;
    let kind = spec
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("question `{name}` has no string \"type\""))?;
    let qtype = QType::parse(kind).map_err(|e| format!("question `{name}`: {e}"))?;
    let instructions =
        take_nonempty_str(spec, "instructions").map_err(|e| format!("question `{name}`: {e}"))?;
    let criteria = spec.get("criteria");

    let criteria = match qtype {
        QType::Choice => {
            let criteria = criteria.and_then(Value::as_object).ok_or_else(|| {
                format!("question `{name}`: a choice needs a \"criteria\" object")
            })?;
            if criteria.is_empty() {
                return Err(format!("question `{name}`: \"criteria\" is empty"));
            }
            Criteria::Choice(
                criteria
                    .iter()
                    .map(|(label, description)| {
                        (label.clone(), description.as_str().map(str::to_string))
                    })
                    .collect(),
            )
        }
        QType::Score => {
            let criteria = criteria
                .and_then(Value::as_array)
                .ok_or_else(|| format!("question `{name}`: a score needs a \"criteria\" array"))?;
            let levels: Result<Vec<String>, String> = criteria
                .iter()
                .map(|level| {
                    level
                        .as_str()
                        .map(str::to_string)
                        .ok_or_else(|| format!("question `{name}`: score criteria must be strings"))
                })
                .collect();
            let levels = levels?;
            if levels.len() < 2 {
                return Err(format!(
                    "question `{name}`: a score needs at least two levels"
                ));
            }
            Criteria::Score(levels)
        }
        QType::Noul => Criteria::Noul,
    };
    Ok(Question {
        name: name.to_string(),
        qtype,
        instructions,
        criteria,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Request {
        parse_request(line).expect("a request").expect("a record")
    }

    fn q<'a>(r: &'a Request, name: &str) -> &'a Question {
        r.questions.iter().find(|q| q.name == name).expect("named")
    }

    #[test]
    fn parse_reads_the_three_shapes() {
        let r = parse(
            r#"{"id": 1, "state": "二重請求です。", "questions": {
                "route": {"type": "choice", "instructions": "担当部署は？",
                          "criteria": {"billing": "請求・返金", "sales": null}},
                "urgency": {"type": "score", "instructions": "緊急度は？",
                            "criteria": ["急がない", "即時対応"]},
                "refund": {"type": "noul", "instructions": "返金を要求しているか？"}}}"#,
        );
        match &q(&r, "route").criteria {
            Criteria::Choice(o) => {
                assert_eq!(
                    o,
                    &[
                        ("billing".to_string(), Some("請求・返金".to_string())),
                        ("sales".to_string(), None),
                    ]
                );
            }
            _ => panic!("a choice"),
        }
        match &q(&r, "urgency").criteria {
            Criteria::Score(levels) => assert_eq!(levels, &["急がない", "即時対応"]),
            _ => panic!("a score"),
        }
        assert!(matches!(q(&r, "refund").criteria, Criteria::Noul));
    }

    #[test]
    fn a_choice_answer_names_the_label_and_keeps_the_distribution() {
        let request = parse(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"choice","instructions":"?","criteria":{"a":null,"b":null}}}}"#,
        );
        let answer = typed_answer(
            q(&request, "q"),
            &[("a".to_string(), 0.0), ("b".to_string(), 1.0)],
        )
        .unwrap();
        assert_eq!(answer["type"], "choice");
        assert_eq!(answer["choice"], "b");
        assert!(answer["probability"].as_f64().unwrap() > 0.7);
        assert_eq!(
            answer["probabilities"]["a"].as_f64().unwrap(),
            (1.0 / (1.0 + 1.0f64.exp()) * 1e6).round() / 1e6
        );
    }

    #[test]
    fn a_score_answer_is_the_expected_level() {
        let request = parse(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"score","instructions":"?","criteria":["low","medium","high"]}}}"#,
        );
        let answer = typed_answer(
            q(&request, "q"),
            &[
                ("low".to_string(), 0.0),
                ("medium".to_string(), 0.0),
                ("high".to_string(), 2.0),
            ],
        )
        .unwrap();
        assert_eq!(answer["type"], "score");
        assert!(answer["score"].as_f64().unwrap() > 1.5);
        assert_eq!(answer["probabilities"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn a_noul_answer_is_p_true_in_the_canonical_order() {
        let request =
            parse(r#"{"id":1,"state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}"#);
        let answer = typed_answer(
            q(&request, "q"),
            &[("false".to_string(), 0.0), ("true".to_string(), 3.0)],
        )
        .unwrap();
        assert_eq!(answer["type"], "noul");
        assert!(answer["noul"].as_f64().unwrap() > 0.9);
    }

    #[test]
    fn a_mismatched_head_score_is_an_error_not_a_guessed_answer() {
        let request =
            parse(r#"{"id":1,"state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}"#);
        let error = typed_answer(
            q(&request, "q"),
            &[("true".to_string(), 0.0), ("false".to_string(), 0.0)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("candidate 0"));
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_logits() {
        let p = softmax(&[1000.0, 1000.0]);
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6);
    }

    /// A malformed request is a skipped line, never a guessed answer.
    #[test]
    fn parse_skips_what_it_cannot_answer() {
        assert!(matches!(parse_request(""), Ok(None)));
        assert!(parse_request(r#"{"state": "s", "questions": {}}"#).is_err());
        assert!(parse_request(r#"{"id": 1, "questions": {}}"#).is_err());
        assert!(parse_request(r#"{"id": 1, "state": "s"}"#).is_err());
        assert!(parse_request(r#"{"id": 1, "state": "s", "questions": {}}"#).is_err());
        assert!(parse_request(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"choice","instructions":"?"}}}"#
        )
        .is_err());
        assert!(parse_request(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"score","instructions":"?","criteria":"x"}}}"#
        )
        .is_err());
        assert!(parse_request(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"mood","instructions":"?"}}}"#
        )
        .is_err());
    }
}
