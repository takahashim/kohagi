//! What a handler hands back: a reply on the wire.
//!
//! A refusal is one too, which is the whole of this module's claim. `api`
//! decides what a request may say and what an answer looks like; here that
//! becomes a status, the one header the status calls for, and a body. Nothing
//! upstream of this file knows a status code, and nothing in it knows what a
//! valid request is.

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::header::CONTENT_TYPE;
use hyper::{Response, StatusCode};

use super::api::{self, Refusal};
use super::worker::WorkerError;

/// One reply, body and all. Every handler returns this, or an [`ApiError`]
/// that becomes one.
pub(crate) type Reply = Response<Full<Bytes>>;

/// A JSON body under `status`, which is every answer this server gives.
pub(crate) fn json(status: StatusCode, body: Vec<u8>) -> Reply {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(body)))
        .expect("a status and one header make a valid response")
}

/// A refusal on the wire: the status, the one header the status calls for
/// (`Allow` for 405, `Retry-After` for 503), and OpenAI's error object.
#[derive(Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    header: Option<(&'static str, &'static str)>,
    kind: &'static str,
    param: Option<&'static str>,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            header: None,
            kind,
            param: None,
            message: message.into(),
        }
    }

    pub(crate) fn invalid(param: Option<&'static str>, message: impl Into<String>) -> Self {
        Self {
            param,
            ..Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
        }
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "invalid_request_error", message)
    }

    pub(crate) fn method_not_allowed(allow: &'static str) -> Self {
        Self {
            header: Some(("allow", allow)),
            ..Self::new(
                StatusCode::METHOD_NOT_ALLOWED,
                "invalid_request_error",
                format!("this path takes {allow}"),
            )
        }
    }

    pub(crate) fn too_large(limit: usize) -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            format!(
                "the request body is longer than this server reads ({limit} bytes, \
                 --max-body-bytes); send fewer texts per request"
            ),
        )
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
    }

    /// A worker's refusal or failure, under the model's name: two models can
    /// be loaded, and the operator should not have to guess which one this
    /// was.
    pub(crate) fn worker(e: WorkerError, label: &str) -> Self {
        match e {
            WorkerError::Busy => Self {
                header: Some(("retry-after", "1")),
                ..Self::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "server_error",
                    format!("`{label}`'s queue is full (--max-queue); retry shortly"),
                )
            },
            WorkerError::Gone => Self::internal(format!(
                "`{label}`'s thread is gone; this server is shutting down"
            )),
            WorkerError::Failed(e) => Self::internal(format!("`{label}` failed: {e:#}")),
        }
    }

    pub(crate) fn reply(&self) -> Reply {
        let mut builder = Response::builder()
            .status(self.status)
            .header(CONTENT_TYPE, "application/json");
        if let Some((name, value)) = self.header {
            builder = builder.header(name, value);
        }
        builder
            .body(Full::new(Bytes::from(api::error_body(
                &self.message,
                self.kind,
                self.param,
            ))))
            .expect("a status and fixed headers make a valid response")
    }
}

impl From<Refusal> for ApiError {
    fn from(r: Refusal) -> Self {
        Self::invalid(r.param, r.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_carries_the_header_its_status_calls_for() {
        let busy = ApiError::worker(WorkerError::Busy, "m");
        assert!(busy.message.contains("`m`"), "{}", busy.message);
        let busy = busy.reply();
        assert_eq!(busy.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(busy.headers()["retry-after"], "1");
        let wrong = ApiError::method_not_allowed("POST").reply();
        assert_eq!(wrong.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(wrong.headers()["allow"], "POST");
        assert_eq!(wrong.headers()[CONTENT_TYPE], "application/json");
    }
}
