//! The requests and replies as clients expect them, and nothing about how
//! they travel: which status a refusal gets, and which headers, is `http`'s
//! business. [`embeddings`] speaks OpenAI's `/v1/embeddings`; [`rerank`]
//! speaks the `/v1/rerank` that Cohere and Jina share (and TEI and vLLM
//! follow). This module holds what both need: the refusal, the error object
//! every refusal is written as, and the model listings.
//!
//! Compatibility here means a client written for those APIs works by swapping
//! its base URL. It does not mean every field means something: `model` is
//! accepted and ignored, because which checkpoint runs is decided by the
//! flags the server was started with, and the reply names that one.

pub(crate) mod embeddings;
pub(crate) mod rerank;

use serde::Serialize;

use crate::ModelInfo;

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
    use serde_json::Value;

    use super::*;

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
