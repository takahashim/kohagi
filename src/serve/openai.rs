//! The `/v1/embeddings` request and reply as OpenAI shaped them, the checks
//! that turn a request into texts the model can take, and the error object
//! every refusal is written as.
//!
//! Compatibility here means a client written for that API works by swapping
//! its base URL. It does not mean every field means something: `model` is
//! accepted and ignored, because which checkpoint runs is decided by the flags
//! the server was started with, and the reply names that one.

use base64::Engine as _;
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::batch::l2_normalize;
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

/// What a request is checked against: the server's limits and the model's
/// output.
pub(crate) struct Limits {
    pub max_inputs: usize,
    /// The dimension vectors leave the model with (`--dims` included).
    pub output_dim: usize,
    /// Whether the model normalizes; `dimensions` needs it to.
    pub normalize: bool,
}

/// A request that passed every check: the texts to embed, how to write the
/// vectors, and a truncation to apply if it changes anything.
pub(crate) struct Validated {
    pub texts: Vec<String>,
    pub encoding: Encoding,
    pub dimensions: Option<usize>,
}

/// Check a request. Refusals name the field and say what was wrong, since a
/// client sees nothing else.
pub(crate) fn validate(req: EmbeddingsRequest, limits: &Limits) -> Result<Validated, ApiError> {
    let texts = texts_of(req.input, limits.max_inputs)?;

    let encoding = match req.encoding_format.as_deref() {
        None | Some("float") => Encoding::Float,
        Some("base64") => Encoding::Base64,
        Some(other) => {
            return Err(ApiError::invalid(
                Some("encoding_format"),
                format!("`encoding_format` must be \"float\" or \"base64\", not \"{other}\""),
            ))
        }
    };

    let dimensions = match req.dimensions {
        None => None,
        Some(_) if !limits.normalize => {
            return Err(ApiError::invalid(
                Some("dimensions"),
                "`dimensions` truncates and re-normalizes, and this server runs with \
                 --no-normalize; slice the full vectors yourself",
            ))
        }
        Some(n) => {
            let n = usize::try_from(n).unwrap_or(usize::MAX);
            if n == 0 || n > limits.output_dim {
                return Err(ApiError::invalid(
                    Some("dimensions"),
                    format!(
                        "`dimensions` must be in 1..={}; this server's vectors have {} dimensions",
                        limits.output_dim, limits.output_dim
                    ),
                ));
            }
            // Equal to the model's own changes no vector, so it is no request.
            (n < limits.output_dim).then_some(n)
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
fn texts_of(input: Value, max_inputs: usize) -> Result<Vec<String>, ApiError> {
    let items = match input {
        Value::String(s) => vec![Value::String(s)],
        Value::Array(items) => items,
        _ => {
            return Err(ApiError::invalid(
                Some("input"),
                "`input` must be a string or an array of strings",
            ))
        }
    };
    if items.is_empty() {
        return Err(ApiError::invalid(
            Some("input"),
            "`input` must contain at least one string",
        ));
    }
    if items.len() > max_inputs {
        return Err(ApiError::invalid(
            Some("input"),
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
                return Err(ApiError::invalid(
                    Some("input"),
                    format!("`input[{i}]` is an empty string; there is nothing to embed"),
                ))
            }
            Value::String(s) => texts.push(s),
            // OpenAI also accepts token ids, as integers or arrays of them.
            // Kohagi tokenizes for itself, and ids from another tokenizer
            // would be embedded as nonsense, so they are refused by name.
            Value::Number(_) | Value::Array(_) => {
                return Err(ApiError::invalid(
                    Some("input"),
                    "`input` as token ids is not supported; send the text as strings",
                ))
            }
            _ => {
                return Err(ApiError::invalid(
                    Some("input"),
                    format!("`input[{i}]` is not a string"),
                ))
            }
        }
    }
    Ok(texts)
}

/// Matryoshka truncation for one request, as `--dims` does it for a run: keep
/// the leading `n` and re-normalize, so dot = cosine holds on the shorter
/// vectors too.
pub(crate) fn truncate(vectors: &mut [Vec<f32>], n: usize) {
    for v in vectors {
        v.truncate(n);
        l2_normalize(v);
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
    data: [Model<'a>; 1],
}

fn model<'a>(label: &'a str, info: &'a ModelInfo) -> Model<'a> {
    Model {
        id: label,
        object: "model",
        owned_by: "kohagi",
        kohagi: info,
    }
}

pub(crate) fn models_body(label: &str, info: &ModelInfo) -> Vec<u8> {
    serde_json::to_vec(&ModelList {
        object: "list",
        data: [model(label, info)],
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

/// A refusal, in the shape OpenAI clients read their exceptions from:
/// `{"error": {"message", "type", "param", "code"}}`.
#[derive(Debug)]
pub(crate) struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub param: Option<&'static str>,
    pub message: String,
    /// `Allow` for a 405, `Retry-After` for a 503: the one header the status
    /// calls for.
    pub header: Option<(&'static str, &'static str)>,
}

impl ApiError {
    pub(crate) fn invalid(param: Option<&'static str>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            param,
            message: message.into(),
            header: None,
        }
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            param: None,
            message: message.into(),
            header: None,
        }
    }

    pub(crate) fn method_not_allowed(allow: &'static str) -> Self {
        Self {
            status: StatusCode::METHOD_NOT_ALLOWED,
            kind: "invalid_request_error",
            param: None,
            message: format!("this path takes {allow}"),
            header: Some(("allow", allow)),
        }
    }

    pub(crate) fn too_large(limit: usize) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            kind: "invalid_request_error",
            param: None,
            message: format!(
                "the request body is longer than this server reads ({limit} bytes, \
                 --max-body-bytes); send fewer texts per request"
            ),
            header: None,
        }
    }

    pub(crate) fn busy() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "server_error",
            param: None,
            message: "the model's queue is full (--max-queue); retry shortly".to_string(),
            header: Some(("retry-after", "1")),
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            param: None,
            message: message.into(),
            header: None,
        }
    }

    pub(crate) fn body(&self) -> Vec<u8> {
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
                message: &self.message,
                kind: self.kind,
                param: self.param,
                code: None,
            },
        })
        .expect("an error object serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> Limits {
        Limits {
            max_inputs: 4,
            output_dim: 4,
            normalize: true,
        }
    }

    fn request(body: &str) -> EmbeddingsRequest {
        serde_json::from_str(body).expect("test body parses")
    }

    fn refusal(body: &str, limits: &Limits) -> ApiError {
        validate(request(body), limits).err().expect("refused")
    }

    #[test]
    fn a_string_or_an_array_of_strings_is_the_input() {
        let one = validate(request(r#"{"input": "瑠璃"}"#), &limits()).unwrap();
        assert_eq!(one.texts, ["瑠璃"]);
        assert_eq!(one.encoding, Encoding::Float);
        assert_eq!(one.dimensions, None);

        let two = validate(request(r#"{"input": ["a", "b"], "model": "x"}"#), &limits()).unwrap();
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
            let e = refusal(body, &limits());
            assert_eq!(e.status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(e.param, Some("input"), "{body}");
            assert!(e.message.contains(says), "{body}: {}", e.message);
        }
    }

    #[test]
    fn the_encoding_is_float_or_base64() {
        let b = validate(
            request(r#"{"input": "a", "encoding_format": "base64"}"#),
            &limits(),
        )
        .unwrap();
        assert_eq!(b.encoding, Encoding::Base64);
        let f = validate(
            request(r#"{"input": "a", "encoding_format": "float"}"#),
            &limits(),
        )
        .unwrap();
        assert_eq!(f.encoding, Encoding::Float);
        let e = refusal(r#"{"input": "a", "encoding_format": "hex"}"#, &limits());
        assert_eq!(e.param, Some("encoding_format"));
    }

    #[test]
    fn dimensions_is_a_truncation_within_the_model_and_only_when_it_shortens() {
        let two = validate(request(r#"{"input": "a", "dimensions": 2}"#), &limits()).unwrap();
        assert_eq!(two.dimensions, Some(2));
        // The model's own dimension changes no vector, so it asks for nothing.
        let four = validate(request(r#"{"input": "a", "dimensions": 4}"#), &limits()).unwrap();
        assert_eq!(four.dimensions, None);
        for body in [
            r#"{"input": "a", "dimensions": 0}"#,
            r#"{"input": "a", "dimensions": 5}"#,
        ] {
            let e = refusal(body, &limits());
            assert_eq!(e.param, Some("dimensions"), "{body}");
            assert!(e.message.contains("1..=4"), "{body}: {}", e.message);
        }
        // A negative number never parses as a dimension count.
        assert!(
            serde_json::from_str::<EmbeddingsRequest>(r#"{"input": "a", "dimensions": -1}"#)
                .is_err()
        );

        let raw = Limits {
            normalize: false,
            ..limits()
        };
        let e = refusal(r#"{"input": "a", "dimensions": 2}"#, &raw);
        assert!(e.message.contains("--no-normalize"), "{}", e.message);
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

    #[test]
    fn an_error_is_the_object_clients_read_their_message_from() {
        let e = ApiError::invalid(Some("input"), "no");
        let v: Value = serde_json::from_slice(&e.body()).unwrap();
        assert_eq!(v["error"]["message"], "no");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "input");
        assert_eq!(v["error"]["code"], Value::Null);

        let busy = ApiError::busy();
        assert_eq!(busy.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(busy.header, Some(("retry-after", "1")));
    }
}
