//! The requests and replies as clients expect them: `/v1/embeddings` as
//! OpenAI shaped it, `/v1/rerank` as Cohere and Jina shaped theirs (and TEI
//! and vLLM follow), the checks that turn a request into what a model can
//! take, and the error object every refusal is written as. JSON shapes only:
//! which status a refusal gets, and which headers, is `http`'s business.
//!
//! Compatibility here means a client written for those APIs works by swapping
//! its base URL. It does not mean every field means something: `model` is
//! accepted and ignored, because which checkpoint runs is decided by the flags
//! the server was started with, and the reply names that one.

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Scores;
use crate::batch::{truncate_renormalize, truncation};
use crate::{ModelInfo, TokenInfo};

/// `POST /v1/embeddings`, as sent. `input` is taken as a JSON value so that
/// the refusal can say what was sent instead of that nothing matched.
#[derive(Deserialize)]
pub(crate) struct EmbeddingsRequest {
    pub input: Value,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<u64>,
}

/// How each vector is written: a JSON array of numbers, or the float32
/// little-endian bytes in base64, which is a third the size and a tenth the
/// parsing cost on the client.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Encoding {
    Float,
    Base64,
}

/// A request that failed a check: which field, and what was wrong. The HTTP
/// layer makes a 400 of it; this module only knows what a request may say.
#[derive(Debug)]
pub(crate) struct Refusal {
    pub param: Option<&'static str>,
    pub message: String,
}

impl Refusal {
    fn of(param: &'static str, message: impl Into<String>) -> Self {
        Self {
            param: Some(param),
            message: message.into(),
        }
    }
}

/// A request that passed every check: the texts to embed, how to write the
/// vectors, and a truncation to apply if it changes anything.
pub(crate) struct Validated {
    pub texts: Vec<String>,
    pub encoding: Encoding,
    pub dimensions: Option<usize>,
}

impl Validated {
    /// The same request with `prefix` on every text, which is the form the
    /// model sees; `--prefix` on the CLI does the same to every record.
    pub(crate) fn prefixed(mut self, prefix: &str) -> Self {
        if !prefix.is_empty() {
            for text in &mut self.texts {
                text.insert_str(0, prefix);
            }
        }
        self
    }
}

/// Check a request against the server's limit and the model's output.
/// Refusals name the field and say what was wrong, since a client sees
/// nothing else.
pub(crate) fn validate(
    req: EmbeddingsRequest,
    max_inputs: usize,
    info: &ModelInfo,
) -> Result<Validated, Refusal> {
    let texts = texts_of(req.input, max_inputs)?;

    let encoding = match req.encoding_format.as_deref() {
        None | Some("float") => Encoding::Float,
        Some("base64") => Encoding::Base64,
        Some(other) => {
            return Err(Refusal::of(
                "encoding_format",
                format!("`encoding_format` must be \"float\" or \"base64\", not \"{other}\""),
            ))
        }
    };

    let dimensions = match req.dimensions {
        None => None,
        Some(_) if !info.normalized() => {
            return Err(Refusal::of(
                "dimensions",
                "`dimensions` truncates and re-normalizes, and this server runs with \
                 --no-normalize; slice the full vectors yourself",
            ))
        }
        Some(n) => {
            let dim = info.reported_dim();
            let n = usize::try_from(n).unwrap_or(usize::MAX);
            truncation(n, dim).map_err(|e| {
                Refusal::of(
                    "dimensions",
                    format!("`dimensions` {e}; this server's vectors have {dim} dimensions"),
                )
            })?
        }
    };

    Ok(Validated {
        texts,
        encoding,
        dimensions,
    })
}

/// `input` as the texts to embed: one string, or an array of them. Empty
/// strings are refused rather than skipped: the stdio protocol skips a record
/// and reports it in the summary, but a request is answered whole or not at
/// all, and a vector missing from `data` would be found by its absence.
fn texts_of(input: Value, max_inputs: usize) -> Result<Vec<String>, Refusal> {
    let items = match input {
        Value::String(s) => vec![Value::String(s)],
        Value::Array(items) => items,
        _ => {
            return Err(Refusal::of(
                "input",
                "`input` must be a string or an array of strings",
            ))
        }
    };
    if items.is_empty() {
        return Err(Refusal::of(
            "input",
            "`input` must contain at least one string",
        ));
    }
    if items.len() > max_inputs {
        return Err(Refusal::of(
            "input",
            format!(
                "`input` has {} items; this server takes at most {max_inputs} per request \
                 (--max-inputs)",
                items.len()
            ),
        ));
    }
    let mut texts = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        match item {
            Value::String(s) if s.is_empty() => {
                return Err(Refusal::of(
                    "input",
                    format!("`input[{i}]` is an empty string; there is nothing to embed"),
                ))
            }
            Value::String(s) => texts.push(s),
            // OpenAI also accepts token ids, as integers or arrays of them.
            // Kohagi tokenizes for itself, and ids from another tokenizer
            // would be embedded as nonsense, so they are refused by name.
            Value::Number(_) | Value::Array(_) => {
                return Err(Refusal::of(
                    "input",
                    "`input` as token ids is not supported; send the text as strings",
                ))
            }
            _ => {
                return Err(Refusal::of(
                    "input",
                    format!("`input[{i}]` is not a string"),
                ))
            }
        }
    }
    Ok(texts)
}

/// Matryoshka truncation for one request, by the same definition `--dims`
/// uses for a run.
pub(crate) fn truncate(vectors: &mut [Vec<f32>], n: usize) {
    for v in vectors {
        truncate_renormalize(v, n);
    }
}

/// float32, little-endian, base64: the bytes OpenAI's `encoding_format:
/// "base64"` carries, which Ruby reads with `unpack("e*")` and Python with
/// `np.frombuffer(..., dtype="<f4")`.
pub(crate) fn base64_f32(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[derive(Serialize)]
struct EmbeddingsReply<'a> {
    object: &'static str,
    data: Vec<Item<'a>>,
    model: &'a str,
    usage: Usage,
}

#[derive(Serialize)]
struct Item<'a> {
    object: &'static str,
    index: usize,
    embedding: Embedding<'a>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Embedding<'a> {
    Float(&'a [f32]),
    Base64(String),
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// The reply body for one request's vectors. `usage` counts every token the
/// model saw, special tokens included, as `--format openai` does.
pub(crate) fn embeddings_body(
    label: &str,
    vectors: &[Vec<f32>],
    tokens: &[TokenInfo],
    encoding: Encoding,
) -> Vec<u8> {
    let data = vectors
        .iter()
        .enumerate()
        .map(|(index, v)| Item {
            object: "embedding",
            index,
            embedding: match encoding {
                Encoding::Float => Embedding::Float(v),
                Encoding::Base64 => Embedding::Base64(base64_f32(v)),
            },
        })
        .collect();
    let n = tokens.iter().map(|t| t.n_tokens).sum();
    serde_json::to_vec(&EmbeddingsReply {
        object: "list",
        data,
        model: label,
        usage: Usage {
            prompt_tokens: n,
            total_tokens: n,
        },
    })
    .expect("a reply of strings and numbers serializes")
}

/// `POST /v1/rerank`, as sent. `query` and `documents` are taken as JSON
/// values so that a refusal can say what was sent.
#[derive(Deserialize)]
pub(crate) struct RerankRequest {
    pub query: Value,
    pub documents: Value,
    #[serde(default)]
    pub top_n: Option<u64>,
    #[serde(default)]
    pub return_documents: bool,
}

/// A rerank request that passed every check.
pub(crate) struct ValidatedRerank {
    pub query: String,
    pub documents: Vec<String>,
    /// How many results to return, at most; `None` returns them all.
    pub top_n: Option<usize>,
    pub return_documents: bool,
}

/// Check a rerank request. `documents` may be strings or `{"text": …}`
/// objects, as Cohere accepts both; `max_inputs` bounds their number as it
/// bounds `input` for embeddings.
pub(crate) fn validate_rerank(
    req: RerankRequest,
    max_inputs: usize,
) -> Result<ValidatedRerank, Refusal> {
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
    Ok(ValidatedRerank {
        query,
        documents,
        top_n,
        return_documents: req.return_documents,
    })
}

#[derive(Serialize)]
struct RerankReply<'a> {
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

/// The reply for one rerank request: every document's index and score,
/// best first, cut to `top_n`, with the document's text beside it when the
/// caller asked (`documents` is `Some` exactly then). `usage` counts every
/// pair the model saw, the ones `top_n` dropped included.
pub(crate) fn rerank_body(
    label: &str,
    scores: &Scores,
    top_n: Option<usize>,
    documents: Option<&[String]>,
) -> Vec<u8> {
    let mut ranked: Vec<(usize, f32)> = scores.scores.iter().copied().enumerate().collect();
    // Stable, so equal scores keep their input order.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(top_n.unwrap_or(usize::MAX));
    let results = ranked
        .into_iter()
        .map(|(index, relevance_score)| RankedDocument {
            index,
            relevance_score,
            document: documents.map(|d| Document {
                text: d[index].as_str(),
            }),
        })
        .collect();
    serde_json::to_vec(&RerankReply {
        model: label,
        results,
        usage: TotalTokens {
            total_tokens: scores.tokens.iter().map(|t| t.n_tokens).sum(),
        },
    })
    .expect("a reply of strings and numbers serializes")
}

/// One entry of `GET /v1/models`: the OpenAI fields, plus everything
/// `--print-model-info` says under `kohagi`, so a client can check `dim` and
/// `sha256` against its index before it embeds anything.
#[derive(Serialize)]
struct Model<'a> {
    id: &'a str,
    object: &'static str,
    owned_by: &'static str,
    kohagi: &'a ModelInfo,
}

#[derive(Serialize)]
struct ModelList<'a> {
    object: &'static str,
    data: Vec<Model<'a>>,
}

fn model<'a>(label: &'a str, info: &'a ModelInfo) -> Model<'a> {
    Model {
        id: label,
        object: "model",
        owned_by: "kohagi",
        kohagi: info,
    }
}

/// Every loaded model, the embedder first.
pub(crate) fn models_body(models: &[(&str, &ModelInfo)]) -> Vec<u8> {
    serde_json::to_vec(&ModelList {
        object: "list",
        data: models
            .iter()
            .map(|(label, info)| model(label, info))
            .collect(),
    })
    .expect("model facts serialize")
}

pub(crate) fn model_body(label: &str, info: &ModelInfo) -> Vec<u8> {
    serde_json::to_vec(&model(label, info)).expect("model facts serialize")
}

#[derive(Serialize)]
struct Health<'a> {
    status: &'static str,
    model: &'a str,
}

pub(crate) fn health_body(label: &str) -> Vec<u8> {
    serde_json::to_vec(&Health {
        status: "ok",
        model: label,
    })
    .expect("a status serializes")
}

/// The object OpenAI clients read their exceptions from:
/// `{"error": {"message", "type", "param", "code"}}`.
pub(crate) fn error_body(
    message: &str,
    kind: &'static str,
    param: Option<&'static str>,
) -> Vec<u8> {
    #[derive(Serialize)]
    struct Envelope<'a> {
        error: ErrorObject<'a>,
    }
    #[derive(Serialize)]
    struct ErrorObject<'a> {
        message: &'a str,
        #[serde(rename = "type")]
        kind: &'static str,
        param: Option<&'static str>,
        code: Option<&'static str>,
    }
    serde_json::to_vec(&Envelope {
        error: ErrorObject {
            message,
            kind,
            param,
            code: None,
        },
    })
    .expect("an error object serializes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Output;

    /// A four-dimensional, normalized model.
    fn info() -> ModelInfo {
        super::super::testing::embedding_info(4)
    }

    fn request(body: &str) -> EmbeddingsRequest {
        serde_json::from_str(body).expect("test body parses")
    }

    fn accepted(body: &str) -> Validated {
        validate(request(body), 4, &info()).expect("accepted")
    }

    fn refused(body: &str) -> Refusal {
        validate(request(body), 4, &info()).err().expect("refused")
    }

    #[test]
    fn a_string_or_an_array_of_strings_is_the_input() {
        let one = accepted(r#"{"input": "瑠璃"}"#);
        assert_eq!(one.texts, ["瑠璃"]);
        assert_eq!(one.encoding, Encoding::Float);
        assert_eq!(one.dimensions, None);

        let two = accepted(r#"{"input": ["a", "b"], "model": "x"}"#);
        assert_eq!(two.texts, ["a", "b"]);
    }

    #[test]
    fn what_is_not_text_is_refused_by_name() {
        for (body, says) in [
            (r#"{"input": []}"#, "at least one"),
            (r#"{"input": ""}"#, "empty string"),
            (r#"{"input": ["a", ""]}"#, "`input[1]` is an empty string"),
            (r#"{"input": [1, 2, 3]}"#, "token ids"),
            (r#"{"input": [[1, 2], [3]]}"#, "token ids"),
            (r#"{"input": ["a", null]}"#, "`input[1]` is not a string"),
            (r#"{"input": 7}"#, "string or an array"),
            (r#"{"input": {"text": "a"}}"#, "string or an array"),
            (r#"{"input": ["a", "b", "c", "d", "e"]}"#, "at most 4"),
        ] {
            let e = refused(body);
            assert_eq!(e.param, Some("input"), "{body}");
            assert!(e.message.contains(says), "{body}: {}", e.message);
        }
    }

    #[test]
    fn the_encoding_is_float_or_base64() {
        let b = accepted(r#"{"input": "a", "encoding_format": "base64"}"#);
        assert_eq!(b.encoding, Encoding::Base64);
        let f = accepted(r#"{"input": "a", "encoding_format": "float"}"#);
        assert_eq!(f.encoding, Encoding::Float);
        let e = refused(r#"{"input": "a", "encoding_format": "hex"}"#);
        assert_eq!(e.param, Some("encoding_format"));
    }

    #[test]
    fn dimensions_is_a_truncation_within_the_model_and_only_when_it_shortens() {
        assert_eq!(
            accepted(r#"{"input": "a", "dimensions": 2}"#).dimensions,
            Some(2)
        );
        // The model's own dimension changes no vector, so it asks for nothing.
        assert_eq!(
            accepted(r#"{"input": "a", "dimensions": 4}"#).dimensions,
            None
        );
        for body in [
            r#"{"input": "a", "dimensions": 0}"#,
            r#"{"input": "a", "dimensions": 5}"#,
        ] {
            let e = refused(body);
            assert_eq!(e.param, Some("dimensions"), "{body}");
            assert!(e.message.contains("1..=4"), "{body}: {}", e.message);
        }
        // A negative number never parses as a dimension count.
        assert!(
            serde_json::from_str::<EmbeddingsRequest>(r#"{"input": "a", "dimensions": -1}"#)
                .is_err()
        );

        // Truncation re-normalizes, so a server that does not normalize
        // cannot honour it.
        let raw = ModelInfo {
            output: Output::Embedding {
                output_dim: None,
                normalized: false,
            },
            ..info()
        };
        let e = validate(request(r#"{"input": "a", "dimensions": 2}"#), 4, &raw)
            .err()
            .expect("refused");
        assert!(e.message.contains("--no-normalize"), "{}", e.message);
    }

    #[test]
    fn the_prefix_goes_on_every_text() {
        let v = accepted(r#"{"input": ["a", "b"]}"#).prefixed("検索クエリ: ");
        assert_eq!(v.texts, ["検索クエリ: a", "検索クエリ: b"]);
        let same = accepted(r#"{"input": "a"}"#).prefixed("");
        assert_eq!(same.texts, ["a"]);
    }

    #[test]
    fn truncation_keeps_the_prefix_at_unit_length() {
        let mut vectors = vec![vec![3.0, 4.0, 5.0, 6.0]];
        truncate(&mut vectors, 2);
        assert_eq!(vectors[0], [0.6, 0.8]);
    }

    #[test]
    fn base64_is_the_little_endian_float32_bytes() {
        // 1.0f32 = 00 00 80 3f, -2.5f32 = 00 00 20 c0
        assert_eq!(base64_f32(&[1.0, -2.5]), "AACAPwAAIMA=");
        assert_eq!(base64_f32(&[]), "");
    }

    #[test]
    fn the_reply_is_the_openai_shape_in_either_encoding() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let tokens = vec![
            TokenInfo {
                n_tokens: 3,
                truncated: false,
            },
            TokenInfo {
                n_tokens: 4,
                truncated: true,
            },
        ];
        let float: Value =
            serde_json::from_slice(&embeddings_body("m", &vectors, &tokens, Encoding::Float))
                .unwrap();
        assert_eq!(float["object"], "list");
        assert_eq!(float["model"], "m");
        assert_eq!(float["data"][1]["index"], 1);
        assert_eq!(float["data"][1]["object"], "embedding");
        assert_eq!(float["data"][1]["embedding"], serde_json::json!([0.0, 1.0]));
        assert_eq!(float["usage"]["prompt_tokens"], 7);
        assert_eq!(float["usage"]["total_tokens"], 7);

        let b64: Value =
            serde_json::from_slice(&embeddings_body("m", &vectors, &tokens, Encoding::Base64))
                .unwrap();
        assert_eq!(b64["data"][0]["embedding"], "AACAPwAAAAA=");
    }

    fn rerank(body: &str) -> Result<ValidatedRerank, Refusal> {
        validate_rerank(serde_json::from_str(body).expect("test body parses"), 4)
    }

    #[test]
    fn documents_are_strings_or_text_objects() {
        let r = rerank(r#"{"query": "q", "documents": ["a", {"text": "b"}], "top_n": 1, "return_documents": true}"#)
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

    #[test]
    fn the_rerank_reply_is_best_first_and_cut_to_top_n() {
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
        let v: Value =
            serde_json::from_slice(&rerank_body("r", &scores, Some(3), Some(&documents))).unwrap();
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

        let bare: Value = serde_json::from_slice(&rerank_body("r", &scores, None, None)).unwrap();
        assert_eq!(bare["results"].as_array().unwrap().len(), 4);
        assert!(bare["results"][0].get("document").is_none());
    }

    #[test]
    fn an_error_is_the_object_clients_read_their_message_from() {
        let v: Value =
            serde_json::from_slice(&error_body("no", "invalid_request_error", Some("input")))
                .unwrap();
        assert_eq!(v["error"]["message"], "no");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "input");
        assert_eq!(v["error"]["code"], Value::Null);
    }
}
