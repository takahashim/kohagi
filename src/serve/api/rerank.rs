//! `POST /v1/rerank`, in the shape Cohere and Jina share (and TEI and vLLM
//! follow): the checks that turn a request into the pairs the cross-encoder
//! takes, and the reply that hands the scores back, best first.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::{Pairs, Scores};
use super::Refusal;

/// The request, as sent. `query` and `documents` are taken as JSON values so
/// that a refusal can say what was sent.
#[derive(Deserialize)]
pub(crate) struct Request {
    pub query: Value,
    pub documents: Value,
    #[serde(default)]
    pub top_n: Option<u64>,
    #[serde(default)]
    pub return_documents: bool,
}

/// A request that passed every check.
pub(crate) struct Validated {
    query: String,
    documents: Vec<String>,
    top_n: Option<usize>,
    return_documents: bool,
}

impl Validated {
    /// Split into the model's question and what the reply keeps: `top_n`, and
    /// a copy of the documents exactly when `return_documents` asked for them
    /// back. That decision lives here and nowhere else.
    pub(crate) fn into_parts(self) -> (Pairs, Reply) {
        let documents = self.return_documents.then(|| self.documents.clone());
        (
            Pairs {
                query: self.query,
                documents: self.documents,
            },
            Reply {
                top_n: self.top_n,
                documents,
            },
        )
    }
}

/// Check a rerank request. `documents` may be strings or `{"text": …}`
/// objects, as Cohere accepts both; `max_inputs` bounds their number as it
/// bounds `input` for embeddings.
pub(crate) fn validate(req: Request, max_inputs: usize) -> Result<Validated, Refusal> {
    let query = match req.query {
        Value::String(q) if !q.is_empty() => q,
        Value::String(_) => return Err(Refusal::of("query", "`query` is an empty string")),
        _ => return Err(Refusal::of("query", "`query` must be a string")),
    };
    let items = match req.documents {
        Value::Array(items) => items,
        _ => {
            return Err(Refusal::of(
                "documents",
                "`documents` must be an array of strings or of {\"text\": …} objects",
            ))
        }
    };
    if items.is_empty() {
        return Err(Refusal::of(
            "documents",
            "`documents` must contain at least one document",
        ));
    }
    if items.len() > max_inputs {
        return Err(Refusal::of(
            "documents",
            format!(
                "`documents` has {} items; this server takes at most {max_inputs} per request \
                 (--max-inputs)",
                items.len()
            ),
        ));
    }
    let mut documents = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let text = match item {
            Value::String(s) => s,
            Value::Object(mut fields) => match fields.remove("text") {
                Some(Value::String(s)) => s,
                _ => {
                    return Err(Refusal::of(
                        "documents",
                        format!("`documents[{i}]` is an object without a string `text`"),
                    ))
                }
            },
            _ => {
                return Err(Refusal::of(
                    "documents",
                    format!("`documents[{i}]` is neither a string nor a {{\"text\": …}} object"),
                ))
            }
        };
        if text.is_empty() {
            return Err(Refusal::of(
                "documents",
                format!("`documents[{i}]` is empty; there is nothing to score"),
            ));
        }
        documents.push(text);
    }
    let top_n = match req.top_n {
        None => None,
        Some(0) => return Err(Refusal::of("top_n", "`top_n` must be at least 1")),
        Some(n) => Some(usize::try_from(n).unwrap_or(usize::MAX)),
    };
    Ok(Validated {
        query,
        documents,
        top_n,
        return_documents: req.return_documents,
    })
}

/// What the reply needs beyond the scores: how many results to keep, and the
/// documents to hand back when the caller asked for them.
pub(crate) struct Reply {
    top_n: Option<usize>,
    documents: Option<Vec<String>>,
}

#[derive(Serialize)]
struct ReplyBody<'a> {
    model: &'a str,
    results: Vec<RankedDocument<'a>>,
    usage: TotalTokens,
}

#[derive(Serialize)]
struct RankedDocument<'a> {
    index: usize,
    relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<Document<'a>>,
}

#[derive(Serialize)]
struct Document<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct TotalTokens {
    total_tokens: usize,
}

impl Reply {
    /// The body for one request's scores: every document's index and score,
    /// best first, cut to `top_n`. `usage` counts every pair the model saw,
    /// the ones `top_n` dropped included.
    pub(crate) fn body(&self, label: &str, scores: &Scores) -> Vec<u8> {
        let mut ranked: Vec<(usize, f32)> = scores.scores.iter().copied().enumerate().collect();
        // Stable, so equal scores keep their input order.
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(self.top_n.unwrap_or(usize::MAX));
        let results = ranked
            .into_iter()
            .map(|(index, relevance_score)| RankedDocument {
                index,
                relevance_score,
                document: self.documents.as_ref().map(|d| Document {
                    text: d[index].as_str(),
                }),
            })
            .collect();
        serde_json::to_vec(&ReplyBody {
            model: label,
            results,
            usage: TotalTokens {
                total_tokens: scores.tokens.iter().map(|t| t.n_tokens).sum(),
            },
        })
        .expect("a reply of strings and numbers serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenInfo;

    fn rerank(body: &str) -> Result<Validated, Refusal> {
        validate(serde_json::from_str(body).expect("test body parses"), 4)
    }

    #[test]
    fn documents_are_strings_or_text_objects() {
        let r = rerank(
            r#"{"query": "q", "documents": ["a", {"text": "b"}], "top_n": 1, "return_documents": true}"#,
        )
        .unwrap();
        assert_eq!(r.query, "q");
        assert_eq!(r.documents, ["a", "b"]);
        assert_eq!(r.top_n, Some(1));
        assert!(r.return_documents);

        let plain = rerank(r#"{"query": "q", "documents": ["a"]}"#).unwrap();
        assert_eq!(plain.top_n, None);
        assert!(!plain.return_documents);

        for (body, param, says) in [
            (r#"{"query": "", "documents": ["a"]}"#, "query", "empty"),
            (
                r#"{"query": 1, "documents": ["a"]}"#,
                "query",
                "must be a string",
            ),
            (r#"{"query": "q", "documents": "a"}"#, "documents", "array"),
            (
                r#"{"query": "q", "documents": []}"#,
                "documents",
                "at least one",
            ),
            (
                r#"{"query": "q", "documents": ["a", ""]}"#,
                "documents",
                "`documents[1]` is empty",
            ),
            (
                r#"{"query": "q", "documents": [{"body": "a"}]}"#,
                "documents",
                "without a string `text`",
            ),
            (
                r#"{"query": "q", "documents": [7]}"#,
                "documents",
                "neither",
            ),
            (
                r#"{"query": "q", "documents": ["a", "b", "c", "d", "e"]}"#,
                "documents",
                "at most 4",
            ),
            (
                r#"{"query": "q", "documents": ["a"], "top_n": 0}"#,
                "top_n",
                "at least 1",
            ),
        ] {
            let e = rerank(body).err().expect(body);
            assert_eq!(e.param, Some(param), "{body}");
            assert!(e.message.contains(says), "{body}: {}", e.message);
        }
    }

    /// The one place that decides whether documents travel back.
    #[test]
    fn into_parts_keeps_the_documents_only_when_asked_back() {
        let (pairs, reply) =
            rerank(r#"{"query": "q", "documents": ["a", "b"], "return_documents": true}"#)
                .unwrap()
                .into_parts();
        assert_eq!(pairs.query, "q");
        assert_eq!(pairs.documents, ["a", "b"]);
        assert_eq!(
            reply.documents.as_deref(),
            Some(["a".to_string(), "b".to_string()].as_slice())
        );

        let (pairs, reply) = rerank(r#"{"query": "q", "documents": ["a"]}"#)
            .unwrap()
            .into_parts();
        assert_eq!(pairs.documents, ["a"]);
        assert!(reply.documents.is_none());
    }

    #[test]
    fn the_reply_is_best_first_and_cut_to_top_n() {
        use serde_json::Value;

        let scores = Scores {
            scores: vec![0.2, 0.9, 0.5, 0.9],
            tokens: (0..4)
                .map(|n| TokenInfo {
                    n_tokens: n + 1,
                    truncated: false,
                })
                .collect(),
        };
        let documents = ["a", "b", "c", "d"].map(String::from);
        let kept = Reply {
            top_n: Some(3),
            documents: Some(documents.to_vec()),
        };
        let v: Value = serde_json::from_slice(&kept.body("r", &scores)).unwrap();
        assert_eq!(v["model"], "r");
        let results = v["results"].as_array().unwrap();
        // Ties keep their input order.
        let order: Vec<u64> = results
            .iter()
            .map(|r| r["index"].as_u64().unwrap())
            .collect();
        assert_eq!(order, [1, 3, 2]);
        assert_eq!(results[0]["relevance_score"], 0.9);
        assert_eq!(results[2]["document"]["text"], "c");
        // Every pair counts, the one top_n dropped included.
        assert_eq!(v["usage"]["total_tokens"], 10);

        let bare = Reply {
            top_n: None,
            documents: None,
        };
        let v: Value = serde_json::from_slice(&bare.body("r", &scores)).unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 4);
        assert!(v["results"][0].get("document").is_none());
    }
}
