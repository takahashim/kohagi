//! The `cross-encoder` head: score each `(context, candidate)` pair on its own
//! and let the caller softmax a question's options.
//!
//! This is the `argos1111/modernbert-ja-310m-jev` shape. The context and the
//! candidate text follow that model's prompt (`質問: …\n状況: …`,
//! `{label} — {description}`, and `["true", "false"]` for truth), because a
//! cross-encoder reads the pair as text and was trained on that rendering.

use anyhow::Result;

use super::engine::{cross_options, DecisionHead, EngineOptions, HeadFactory};
use crate::decide::{content, Criteria, Question, Request};
use crate::rerank::Reranker;
use crate::{ModelInfo, ModelSource, TokenInfo};

/// The `--head` name and loader for the cross-encoder head.
pub static FACTORY: &dyn HeadFactory = &Factory;

struct Factory;

impl HeadFactory for Factory {
    fn name(&self) -> &'static str {
        "cross-encoder"
    }

    fn load(
        &self,
        source: &ModelSource,
        options: EngineOptions,
        _subfolder: Option<&str>,
    ) -> Result<Box<dyn DecisionHead>> {
        Ok(Box::new(CrossEncoder::load(source, options)?))
    }
}

pub struct CrossEncoder {
    reranker: Reranker,
}

impl CrossEncoder {
    pub fn load(source: &ModelSource, options: EngineOptions) -> Result<Self> {
        let reranker = Reranker::load(source, cross_options(options))?;
        Ok(Self { reranker })
    }
}

impl DecisionHead for CrossEncoder {
    fn info(&self) -> ModelInfo {
        self.reranker.info()
    }

    fn score(&self, requests: &[Request]) -> Result<(Vec<super::RequestScore>, Vec<TokenInfo>)> {
        // Render every question to a context plus labelled candidates, keeping
        // all the text owned before borrowing it, because the pairs must outlive
        // the `score` call and a pair borrows both halves.
        let mut contexts: Vec<Vec<String>> = Vec::with_capacity(requests.len());
        let mut labels: Vec<Vec<Vec<String>>> = Vec::with_capacity(requests.len());
        let mut candidates: Vec<Vec<Vec<String>>> = Vec::with_capacity(requests.len());
        for request in requests {
            let mut request_contexts = Vec::with_capacity(request.questions.len());
            let mut request_labels = Vec::with_capacity(request.questions.len());
            let mut request_candidates = Vec::with_capacity(request.questions.len());
            for question in &request.questions {
                let (question_labels, question_candidates) = render_candidates(question);
                request_contexts.push(render_context(question, request));
                request_labels.push(question_labels);
                request_candidates.push(question_candidates);
            }
            contexts.push(request_contexts);
            labels.push(request_labels);
            candidates.push(request_candidates);
        }
        let mut pairs: Vec<(&str, &str)> = Vec::new();
        for (request_contexts, request_candidates) in contexts.iter().zip(&candidates) {
            for (context, question_candidates) in request_contexts.iter().zip(request_candidates) {
                for candidate in question_candidates {
                    pairs.push((context.as_str(), candidate.as_str()));
                }
            }
        }

        let (logits, tokens) = self.reranker.score(&pairs)?;

        let mut out = Vec::with_capacity(requests.len());
        let mut info = Vec::with_capacity(requests.len());
        let mut at = 0usize;
        for (request, (request_contexts, request_labels)) in
            requests.iter().zip(contexts.into_iter().zip(labels))
        {
            let _ = request_contexts;
            let mut questions = Vec::with_capacity(request.questions.len());
            let mut n_tokens = 0usize;
            let mut truncated = false;
            for (question_labels, question) in request_labels.into_iter().zip(&request.questions) {
                let n = question_labels.len();
                let mut scored: Vec<_> = question_labels
                    .into_iter()
                    .zip(&logits[at..at + n])
                    .map(|(label, logit)| (label, *logit))
                    .collect();
                // Jev renders true then false, while decide's canonical logical
                // candidates are false then true. Keep the trained prompt but
                // return scores with the labels and order every head promises.
                if matches!(question.criteria, Criteria::Noul) {
                    scored.reverse();
                }
                questions.push(scored);
                for pair in &tokens[at..at + n] {
                    n_tokens = n_tokens.max(pair.n_tokens);
                    truncated |= pair.truncated;
                }
                at += n;
            }
            out.push(questions);
            info.push(TokenInfo {
                n_tokens,
                truncated,
            });
        }
        Ok((out, info))
    }
}

/// The context a question's candidates are scored against.
fn render_context(question: &Question, request: &Request) -> String {
    format!(
        "質問: {}\n状況: {}",
        question.instructions,
        content(&request.state)
    )
}

/// The candidate text per option, and the label each probability is keyed by.
fn render_candidates(question: &Question) -> (Vec<String>, Vec<String>) {
    match &question.criteria {
        Criteria::Choice(options) => options
            .iter()
            .map(|(label, description)| {
                let text = match description {
                    Some(text) => format!("{label} — {text}"),
                    None => label.clone(),
                };
                (label.clone(), text)
            })
            .unzip(),
        Criteria::Score(levels) => (levels.clone(), levels.clone()),
        // The reference treats truth as the pair "true", "false"; the build
        // head keeps that order too.
        Criteria::Noul => (
            vec!["true".to_string(), "false".to_string()],
            vec!["true".to_string(), "false".to_string()],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decide::{QType, Question};

    fn request(state: &str) -> Request {
        Request {
            id: serde_json::Value::from(1),
            state: serde_json::Value::from(state),
            questions: Vec::new(),
        }
    }

    #[test]
    fn a_choice_renders_label_and_description_with_the_jev_dash() {
        let q = Question {
            name: "route".to_string(),
            qtype: QType::Choice,
            instructions: "担当部署は？".to_string(),
            criteria: Criteria::Choice(vec![
                ("billing".to_string(), Some("請求・返金".to_string())),
                ("sales".to_string(), None),
            ]),
        };
        let (labels, candidates) = render_candidates(&q);
        assert_eq!(labels, ["billing", "sales"]);
        assert_eq!(candidates, ["billing — 請求・返金", "sales"]);
        assert_eq!(
            render_context(&q, &request("二重請求です。")),
            "質問: 担当部署は？\n状況: 二重請求です。"
        );
    }

    #[test]
    fn noul_is_true_then_false() {
        let q = Question {
            name: "refund".to_string(),
            qtype: QType::Noul,
            instructions: "?".to_string(),
            criteria: Criteria::Noul,
        };
        assert_eq!(
            render_candidates(&q).0,
            ["true".to_string(), "false".to_string()]
        );
    }
}
