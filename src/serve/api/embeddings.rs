//! `POST /v1/embeddings`, as OpenAI shaped it: the checks that turn a request
//! into texts the model can take, and the reply the vectors are written as.

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Refusal;
use crate::batch::{truncate_renormalize, truncation};
use crate::{ModelInfo, TokenInfo};

/// The request, as sent. `input` is taken as a JSON value so that the refusal
/// can say what was sent instead of that nothing matched.
#[derive(Deserialize)]
pub(crate) struct Request {
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
    req: Request,
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
/// "base64"` carries, which Ruby Base64-decodes before `unpack("e*")` and
/// Python reads with `np.frombuffer(..., dtype="<f4")`.
fn base64_f32(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[derive(Serialize)]
struct ReplyBody<'a> {
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
pub(crate) fn reply(
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
    serde_json::to_vec(&ReplyBody {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Output;

    /// A four-dimensional, normalized model.
    fn info() -> ModelInfo {
        crate::serve::testing::embedding_info(4)
    }

    fn request(body: &str) -> Request {
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
        assert!(serde_json::from_str::<Request>(r#"{"input": "a", "dimensions": -1}"#).is_err());

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
        use serde_json::Value;

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
            serde_json::from_slice(&reply("m", &vectors, &tokens, Encoding::Float)).unwrap();
        assert_eq!(float["object"], "list");
        assert_eq!(float["model"], "m");
        assert_eq!(float["data"][1]["index"], 1);
        assert_eq!(float["data"][1]["object"], "embedding");
        assert_eq!(float["data"][1]["embedding"], serde_json::json!([0.0, 1.0]));
        assert_eq!(float["usage"]["prompt_tokens"], 7);
        assert_eq!(float["usage"]["total_tokens"], 7);

        let b64: Value =
            serde_json::from_slice(&reply("m", &vectors, &tokens, Encoding::Base64)).unwrap();
        assert_eq!(b64["data"][0]["embedding"], "AACAPwAAAAA=");
    }
}
