//! Routes and handlers, and what a refusal is on the wire: which status,
//! which headers. `handle` never fails: every outcome, a refusal included, is
//! a reply, and the connection stays usable for the next one.

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

use http_body_util::{BodyExt, Full, LengthLimitError, Limited};
use hyper::body::{Body, Bytes, Incoming};
use hyper::header::{CONTENT_LENGTH, CONTENT_TYPE};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::sync::watch;

use super::listen::Io;
use super::openai::{self, Refusal};
use super::worker::{Loaded, WorkerError};
use super::{Batch, Config, Pairs, Scores};
use crate::program::remark;
use crate::ModelInfo;

/// Everything a handler reads: the models, the limits, and the counters the
/// summary is written from.
pub(crate) struct State {
    embedder: Loaded<Vec<String>, Batch>,
    reranker: Option<Loaded<Pairs, Scores>>,
    prefix: String,
    max_inputs: usize,
    max_body_bytes: usize,
    counts: Counts,
}

impl State {
    pub(crate) fn new(
        config: &Config,
        embedder: Loaded<Vec<String>, Batch>,
        reranker: Option<Loaded<Pairs, Scores>>,
    ) -> Self {
        Self {
            embedder,
            reranker,
            prefix: config.prefix.clone(),
            max_inputs: config.max_inputs,
            max_body_bytes: config.max_body_bytes,
            counts: Counts::default(),
        }
    }

    pub(crate) fn embedder(&self) -> &Loaded<Vec<String>, Batch> {
        &self.embedder
    }

    pub(crate) fn reranker(&self) -> Option<&Loaded<Pairs, Scores>> {
        self.reranker.as_ref()
    }

    /// Every loaded model, as `/v1/models` lists them.
    fn models(&self) -> Vec<(&str, &ModelInfo)> {
        let mut models = vec![(self.embedder.label.as_str(), &self.embedder.info)];
        if let Some(reranker) = &self.reranker {
            models.push((reranker.label.as_str(), &reranker.info));
        }
        models
    }

    /// The run's one summary line, on stderr at shutdown, reading like the
    /// CLI's: which weights answered, and how much they answered.
    pub(crate) fn summarize(&self) {
        let c = &self.counts;
        remark!(
            "model={} {} requests={} in={} truncated={} scored={} rejected={} failed={}",
            self.embedder.label,
            self.embedder.info.summary_facts(),
            c.requests.load(Relaxed),
            c.inputs.load(Relaxed),
            c.truncated.load(Relaxed),
            c.scored.load(Relaxed),
            c.rejected.load(Relaxed),
            c.failed.load(Relaxed)
        );
    }
}

/// The summary's numbers. `requests` counts everything that arrived;
/// `rejected` the 4xx among them (the client's mistake) and `failed` the 5xx
/// (this side's). `inputs` and `truncated` are the stdio summary's `in` and
/// `truncated`, over the requests that were answered: texts embedded, and
/// texts or pairs that ran past a model's length. `scored` is the documents
/// reranked. A request is answered whole or not at all, so there is no
/// separate `out`.
#[derive(Default)]
struct Counts {
    requests: AtomicUsize,
    rejected: AtomicUsize,
    failed: AtomicUsize,
    inputs: AtomicUsize,
    truncated: AtomicUsize,
    scored: AtomicUsize,
}

impl Counts {
    fn saw(&self, status: StatusCode) {
        self.requests.fetch_add(1, Relaxed);
        if status.is_client_error() {
            self.rejected.fetch_add(1, Relaxed);
        } else if status.is_server_error() {
            self.failed.fetch_add(1, Relaxed);
        }
    }

    fn embedded(&self, batch: &Batch) {
        self.inputs.fetch_add(batch.vectors.len(), Relaxed);
        self.truncated
            .fetch_add(batch.tokens.iter().filter(|t| t.truncated).count(), Relaxed);
    }

    fn reranked(&self, scores: &Scores) {
        self.scored.fetch_add(scores.scores.len(), Relaxed);
        self.truncated.fetch_add(
            scores.tokens.iter().filter(|t| t.truncated).count(),
            Relaxed,
        );
    }
}

type Reply = Response<Full<Bytes>>;

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

    fn invalid(param: Option<&'static str>, message: impl Into<String>) -> Self {
        Self {
            param,
            ..Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "invalid_request_error", message)
    }

    fn method_not_allowed(allow: &'static str) -> Self {
        Self {
            header: Some(("allow", allow)),
            ..Self::new(
                StatusCode::METHOD_NOT_ALLOWED,
                "invalid_request_error",
                format!("this path takes {allow}"),
            )
        }
    }

    fn too_large(limit: usize) -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            format!(
                "the request body is longer than this server reads ({limit} bytes, \
                 --max-body-bytes); send fewer texts per request"
            ),
        )
    }

    fn busy() -> Self {
        Self {
            header: Some(("retry-after", "1")),
            ..Self::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "the model's queue is full (--max-queue); retry shortly",
            )
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
    }

    fn reply(&self) -> Reply {
        let mut builder = Response::builder()
            .status(self.status)
            .header(CONTENT_TYPE, "application/json");
        if let Some((name, value)) = self.header {
            builder = builder.header(name, value);
        }
        builder
            .body(Full::new(Bytes::from(openai::error_body(
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

impl From<WorkerError> for ApiError {
    fn from(e: WorkerError) -> Self {
        match e {
            WorkerError::Busy => Self::busy(),
            WorkerError::Gone => {
                Self::internal("the model's thread is gone; this server is shutting down")
            }
            WorkerError::Failed(e) => Self::internal(format!("the model failed: {e:#}")),
        }
    }
}

/// Answer one request. Generic over the body so tests can hand it one they
/// built; the server hands it hyper's.
pub(crate) async fn handle<B>(req: Request<B>, state: Arc<State>) -> Reply
where
    B: Body,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let reply = match route(req, &state).await {
        Ok(reply) => reply,
        Err(e) => e.reply(),
    };
    state.counts.saw(reply.status());
    reply
}

async fn route<B>(req: Request<B>, state: &State) -> Result<Reply, ApiError>
where
    B: Body,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    if let Some(id) = path.strip_prefix("/v1/models/") {
        return match method {
            Method::GET | Method::HEAD => {
                match state.models().into_iter().find(|(label, _)| *label == id) {
                    Some((label, info)) => {
                        Ok(json(StatusCode::OK, openai::model_body(label, info)))
                    }
                    None => Err(ApiError::not_found(format!(
                        "model `{id}` is not loaded; this server runs `{}`",
                        state.embedder.label
                    ))),
                }
            }
            _ => Err(ApiError::method_not_allowed("GET, HEAD")),
        };
    }

    match (method, path.as_str()) {
        (Method::POST, "/v1/embeddings") => embeddings(req, state).await,
        (Method::POST, "/v1/rerank") => rerank(req, state).await,
        (Method::GET | Method::HEAD, "/v1/models") => {
            Ok(json(StatusCode::OK, openai::models_body(&state.models())))
        }
        (Method::GET | Method::HEAD, "/health") => Ok(json(
            StatusCode::OK,
            openai::health_body(&state.embedder.label),
        )),
        (_, "/v1/embeddings" | "/v1/rerank") => Err(ApiError::method_not_allowed("POST")),
        (_, "/v1/models" | "/health") => Err(ApiError::method_not_allowed("GET, HEAD")),
        (method, path) => Err(ApiError::not_found(format!("no route for {method} {path}"))),
    }
}

async fn embeddings<B>(req: Request<B>, state: &State) -> Result<Reply, ApiError>
where
    B: Body,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let body = read_body(req, state.max_body_bytes).await?;
    let request: openai::EmbeddingsRequest = serde_json::from_slice(&body).map_err(|e| {
        ApiError::invalid(None, format!("the body is not an embeddings request: {e}"))
    })?;
    let request =
        openai::validate(request, state.max_inputs, &state.embedder.info)?.prefixed(&state.prefix);

    let mut batch = state.embedder.handle.ask(request.texts).await?;
    if let Some(dims) = request.dimensions {
        openai::truncate(&mut batch.vectors, dims);
    }
    state.counts.embedded(&batch);

    Ok(json(
        StatusCode::OK,
        openai::embeddings_body(
            &state.embedder.label,
            &batch.vectors,
            &batch.tokens,
            request.encoding,
        ),
    ))
}

async fn rerank<B>(req: Request<B>, state: &State) -> Result<Reply, ApiError>
where
    B: Body,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let Some(reranker) = &state.reranker else {
        return Err(ApiError::not_found(
            "this server has no reranker; start kohagi-serve with --rerank-model-id \
             (or --rerank-model-path) to answer /v1/rerank",
        ));
    };
    let body = read_body(req, state.max_body_bytes).await?;
    let request: openai::RerankRequest = serde_json::from_slice(&body)
        .map_err(|e| ApiError::invalid(None, format!("the body is not a rerank request: {e}")))?;
    let request = openai::validate_rerank(request, state.max_inputs)?;

    // The reply carries the documents back only when asked to; the model
    // needs them either way, so they are cloned rather than kept.
    let documents = request.return_documents.then(|| request.documents.clone());
    let scores = reranker
        .handle
        .ask(Pairs {
            query: request.query,
            documents: request.documents,
        })
        .await?;
    state.counts.reranked(&scores);

    Ok(json(
        StatusCode::OK,
        openai::rerank_body(
            &reranker.label,
            &scores,
            request.top_n,
            documents.as_deref(),
        ),
    ))
}

/// The whole body, or 413 once it is known to be longer than `limit`: from
/// `Content-Length` before reading a byte, or from the count while reading.
async fn read_body<B>(req: Request<B>, limit: usize) -> Result<Bytes, ApiError>
where
    B: Body,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let declared = req
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|len| len > limit) {
        return Err(ApiError::too_large(limit));
    }
    match Limited::new(req.into_body(), limit).collect().await {
        Ok(collected) => Ok(collected.to_bytes()),
        Err(e) if e.downcast_ref::<LengthLimitError>().is_some() => Err(ApiError::too_large(limit)),
        Err(e) => Err(ApiError::invalid(
            None,
            format!("reading the request body failed: {e}"),
        )),
    }
}

fn json(status: StatusCode, body: Vec<u8>) -> Reply {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(body)))
        .expect("a status and one header make a valid response")
}

/// One connection, for as many requests as the client sends on it. Told to
/// stop, it finishes the reply in flight and then closes, which is what hyper
/// calls a graceful shutdown; an idle connection closes at once.
pub(crate) async fn serve_connection(
    io: Io,
    state: Arc<State>,
    mut shutdown: watch::Receiver<bool>,
) {
    let service = service_fn(move |req: Request<Incoming>| {
        let state = state.clone();
        async move { Ok::<_, Infallible>(handle(req, state).await) }
    });
    let conn = http1::Builder::new()
        .timer(TokioTimer::new())
        .serve_connection(TokioIo::new(io), service);
    tokio::pin!(conn);

    let mut signalled = *shutdown.borrow();
    if signalled {
        conn.as_mut().graceful_shutdown();
    }
    loop {
        tokio::select! {
            outcome = conn.as_mut() => {
                if let Err(e) = outcome {
                    if worth_a_line(&e) {
                        remark!("connection: {e}");
                    }
                }
                return;
            }
            _ = shutdown.changed(), if !signalled => {
                signalled = true;
                conn.as_mut().graceful_shutdown();
            }
        }
    }
}

/// A client that hung up mid-request, or a socket already closed by the peer
/// when a stop signal shuts it down, is its own business: neither is anything
/// this side can act on, and a supervisor's log is not the place for them.
fn worth_a_line(e: &hyper::Error) -> bool {
    let io_underneath = std::error::Error::source(e)
        .is_some_and(|source| source.downcast_ref::<std::io::Error>().is_some());
    !e.is_incomplete_message() && !io_underneath
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use base64::Engine as _;
    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::super::worker;
    use super::super::{Engine, Listen, Load};
    use super::*;
    use crate::{Output, TokenInfo};

    fn info(dim: usize, output: Output) -> ModelInfo {
        ModelInfo {
            backend: "cpu",
            precision: "f32",
            sha256: Some("ab".repeat(32)),
            bundle: None,
            pooling: "mean",
            dim,
            max_seq_length: 8,
            declared_max_seq_length: None,
            output,
        }
    }

    /// Four dimensions, one-hot on the text's position, and a token count of
    /// the text's characters, so a test can see what the model was handed.
    struct Stub {
        /// Received texts, for the prefix test.
        seen: Arc<Mutex<Vec<String>>>,
        /// Waits here before answering, when a test needs the queue to fill.
        gate: Option<std::sync::mpsc::Receiver<()>>,
        started: Option<std::sync::mpsc::Sender<()>>,
    }

    impl Engine for Stub {
        type Input = Vec<String>;
        type Output = Batch;

        fn info(&self) -> ModelInfo {
            info(
                4,
                Output::Embedding {
                    output_dim: None,
                    normalized: true,
                },
            )
        }

        fn answer(&self, texts: Vec<String>) -> anyhow::Result<Batch> {
            if let Some(started) = &self.started {
                let _ = started.send(());
            }
            if let Some(gate) = &self.gate {
                gate.recv().expect("the test releases the gate");
            }
            self.seen.lock().unwrap().extend(texts.iter().cloned());
            let vectors = texts
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut v = vec![0.0; 4];
                    v[i % 4] = 1.0;
                    v
                })
                .collect();
            let tokens = texts
                .iter()
                .map(|t| TokenInfo {
                    n_tokens: t.chars().count(),
                    truncated: t.chars().count() > 8,
                })
                .collect();
            Ok(Batch { vectors, tokens })
        }
    }

    /// Scores a document by its length, so the order is decided by the test.
    struct RerankStub;

    impl Engine for RerankStub {
        type Input = Pairs;
        type Output = Scores;

        fn info(&self) -> ModelInfo {
            info(768, Output::Score { score: "sigmoid" })
        }

        fn answer(&self, pairs: Pairs) -> anyhow::Result<Scores> {
            let scores = pairs
                .documents
                .iter()
                .map(|d| d.chars().count() as f32 / 10.0)
                .collect();
            let tokens = pairs
                .documents
                .iter()
                .map(|d| TokenInfo {
                    n_tokens: pairs.query.chars().count() + d.chars().count(),
                    truncated: d.chars().count() > 8,
                })
                .collect();
            Ok(Scores { scores, tokens })
        }
    }

    fn config() -> Config {
        Config {
            listen: Listen::default(),
            prefix: String::new(),
            max_inputs: 4,
            max_body_bytes: 1024,
            max_queue: 1,
            shutdown_timeout: std::time::Duration::from_secs(1),
        }
    }

    fn state_with(config: Config, stub: Stub, reranker: bool) -> Arc<State> {
        let embedder = Load {
            label: "stub/model".to_string(),
            load: move || Ok(stub),
        };
        let embedder = worker::spawn("test-model", embedder, config.max_queue)
            .unwrap()
            .loaded;
        let reranker = reranker.then(|| {
            let reranker = Load {
                label: "stub/reranker".to_string(),
                load: || Ok(RerankStub),
            };
            worker::spawn("test-reranker", reranker, config.max_queue)
                .unwrap()
                .loaded
        });
        Arc::new(State::new(&config, embedder, reranker))
    }

    fn plain_stub() -> Stub {
        Stub {
            seen: Arc::default(),
            gate: None,
            started: None,
        }
    }

    fn state() -> Arc<State> {
        state_with(config(), plain_stub(), false)
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    fn request(method: Method, path: &str, body: &str) -> Request<Full<Bytes>> {
        Request::builder()
            .method(method)
            .uri(path)
            .body(Full::new(Bytes::from(body.to_string())))
            .unwrap()
    }

    async fn call(
        state: &Arc<State>,
        method: Method,
        path: &str,
        body: &str,
    ) -> (StatusCode, Value) {
        let reply = handle(request(method, path, body), state.clone()).await;
        let status = reply.status();
        let bytes = reply.into_body().collect().await.unwrap().to_bytes();
        let value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, value)
    }

    #[test]
    fn embeddings_answers_in_the_openai_shape() {
        block_on(async {
            let state = state();
            let (status, v) = call(
                &state,
                Method::POST,
                "/v1/embeddings",
                r#"{"input": ["ab", "cde"]}"#,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(v["object"], "list");
            assert_eq!(v["model"], "stub/model");
            assert_eq!(v["data"][0]["index"], 0);
            assert_eq!(
                v["data"][0]["embedding"],
                serde_json::json!([1.0, 0.0, 0.0, 0.0])
            );
            assert_eq!(
                v["data"][1]["embedding"],
                serde_json::json!([0.0, 1.0, 0.0, 0.0])
            );
            assert_eq!(v["usage"]["prompt_tokens"], 5);
            assert_eq!(state.counts.inputs.load(Relaxed), 2);
        });
    }

    #[test]
    fn base64_and_float_carry_the_same_vector() {
        block_on(async {
            let state = state();
            let (_, f) = call(&state, Method::POST, "/v1/embeddings", r#"{"input": "x"}"#).await;
            let (_, b) = call(
                &state,
                Method::POST,
                "/v1/embeddings",
                r#"{"input": "x", "encoding_format": "base64"}"#,
            )
            .await;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b["data"][0]["embedding"].as_str().unwrap())
                .unwrap();
            let decoded: Vec<f32> = bytes
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let floats: Vec<f32> = f["data"][0]["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(decoded, floats);
        });
    }

    #[test]
    fn dimensions_truncates_and_renormalizes() {
        block_on(async {
            let state = state();
            let (status, v) = call(
                &state,
                Method::POST,
                "/v1/embeddings",
                r#"{"input": ["a", "b"], "dimensions": 2}"#,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(v["data"][0]["embedding"], serde_json::json!([1.0, 0.0]));
            assert_eq!(v["data"][1]["embedding"], serde_json::json!([0.0, 1.0]));
        });
    }

    #[test]
    fn the_prefix_is_prepended_before_the_model_sees_the_text() {
        block_on(async {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let state = state_with(
                Config {
                    prefix: "検索クエリ: ".to_string(),
                    ..config()
                },
                Stub {
                    seen: seen.clone(),
                    gate: None,
                    started: None,
                },
                false,
            );
            let (status, v) = call(
                &state,
                Method::POST,
                "/v1/embeddings",
                r#"{"input": "瑠璃"}"#,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(seen.lock().unwrap().as_slice(), ["検索クエリ: 瑠璃"]);
            // `usage` counts what the model saw, prefix included.
            assert_eq!(v["usage"]["prompt_tokens"], 9);
        });
    }

    #[test]
    fn refusals_are_status_codes_with_an_error_object() {
        block_on(async {
            let state = state();
            for (method, path, body, status, says) in [
                (
                    Method::POST,
                    "/v1/embeddings",
                    "not json",
                    StatusCode::BAD_REQUEST,
                    "not an embeddings request",
                ),
                (
                    Method::POST,
                    "/v1/embeddings",
                    r#"{"model": "x"}"#,
                    StatusCode::BAD_REQUEST,
                    "missing field `input`",
                ),
                (
                    Method::POST,
                    "/v1/embeddings",
                    r#"{"input": []}"#,
                    StatusCode::BAD_REQUEST,
                    "at least one",
                ),
                (
                    Method::POST,
                    "/v1/embeddings",
                    r#"{"input": "a", "dimensions": 9}"#,
                    StatusCode::BAD_REQUEST,
                    "1..=4",
                ),
                (
                    Method::GET,
                    "/v1/embeddings",
                    "",
                    StatusCode::METHOD_NOT_ALLOWED,
                    "POST",
                ),
                (
                    Method::POST,
                    "/health",
                    "",
                    StatusCode::METHOD_NOT_ALLOWED,
                    "GET, HEAD",
                ),
                (
                    Method::GET,
                    "/v1/models/other",
                    "",
                    StatusCode::NOT_FOUND,
                    "not loaded",
                ),
                (
                    Method::POST,
                    "/v1/models/stub/model",
                    "",
                    StatusCode::METHOD_NOT_ALLOWED,
                    "GET, HEAD",
                ),
                (
                    Method::GET,
                    "/nope",
                    "",
                    StatusCode::NOT_FOUND,
                    "no route for GET /nope",
                ),
                // No reranker was loaded, and the refusal says how to get one.
                (
                    Method::POST,
                    "/v1/rerank",
                    r#"{"query": "q", "documents": ["a"]}"#,
                    StatusCode::NOT_FOUND,
                    "--rerank-model-id",
                ),
            ] {
                let (got, v) = call(&state, method.clone(), path, body).await;
                assert_eq!(got, status, "{method} {path} {body}");
                let message = v["error"]["message"].as_str().unwrap();
                assert!(message.contains(says), "{method} {path}: {message}");
                assert_eq!(v["error"]["type"], "invalid_request_error");
            }
            // Every one of those was the client's mistake, and none embedded
            // anything.
            assert_eq!(state.counts.requests.load(Relaxed), 10);
            assert_eq!(state.counts.rejected.load(Relaxed), 10);
            assert_eq!(state.counts.failed.load(Relaxed), 0);
            assert_eq!(state.counts.inputs.load(Relaxed), 0);
        });
    }

    #[test]
    fn a_refusal_carries_the_header_its_status_calls_for() {
        let busy = ApiError::busy().reply();
        assert_eq!(busy.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(busy.headers()["retry-after"], "1");
        let wrong = ApiError::method_not_allowed("POST").reply();
        assert_eq!(wrong.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(wrong.headers()["allow"], "POST");
        assert_eq!(wrong.headers()[CONTENT_TYPE], "application/json");
    }

    #[test]
    fn a_body_past_the_limit_is_413_whether_declared_or_discovered() {
        block_on(async {
            let state = state();
            let long = format!(r#"{{"input": "{}"}}"#, "x".repeat(2000));
            let (status, _) = call(&state, Method::POST, "/v1/embeddings", &long).await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);

            // Declared longer than the limit, before any of it is read.
            let req = Request::builder()
                .method(Method::POST)
                .uri("/v1/embeddings")
                .header(CONTENT_LENGTH, "999999")
                .body(Full::new(Bytes::from_static(b"{}")))
                .unwrap();
            assert_eq!(
                handle(req, state.clone()).await.status(),
                StatusCode::PAYLOAD_TOO_LARGE
            );
        });
    }

    #[test]
    fn models_and_health_describe_the_loaded_models() {
        block_on(async {
            let state = state_with(config(), plain_stub(), true);
            let (status, v) = call(&state, Method::GET, "/v1/models", "").await;
            assert_eq!(status, StatusCode::OK);
            let ids: Vec<&str> = v["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m["id"].as_str().unwrap())
                .collect();
            assert_eq!(ids, ["stub/model", "stub/reranker"]);
            assert_eq!(v["data"][0]["kohagi"]["dim"], 4);
            assert_eq!(v["data"][0]["kohagi"]["normalized"], true);
            assert_eq!(v["data"][0]["kohagi"]["sha256"], "ab".repeat(32));
            assert_eq!(v["data"][1]["kohagi"]["score"], "sigmoid");

            for id in ["stub/model", "stub/reranker"] {
                let (status, v) = call(&state, Method::GET, &format!("/v1/models/{id}"), "").await;
                assert_eq!(status, StatusCode::OK, "{id}");
                assert_eq!(v["id"], id);
            }

            let (status, v) = call(&state, Method::GET, "/health", "").await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(v["status"], "ok");

            // Without a reranker, the list has one entry.
            let alone = state_with(config(), plain_stub(), false);
            let (_, v) = call(&alone, Method::GET, "/v1/models", "").await;
            assert_eq!(v["data"].as_array().unwrap().len(), 1);
        });
    }

    #[test]
    fn rerank_orders_the_documents_by_score() {
        block_on(async {
            let state = state_with(config(), plain_stub(), true);
            let (status, v) = call(
                &state,
                Method::POST,
                "/v1/rerank",
                r#"{"query": "q", "documents": ["aa", "aaaaaa", {"text": "aaaa"}], "top_n": 2, "return_documents": true}"#,
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            assert_eq!(v["model"], "stub/reranker");
            let results = v["results"].as_array().unwrap();
            assert_eq!(results.len(), 2, "top_n");
            assert_eq!(results[0]["index"], 1);
            assert_eq!(results[0]["relevance_score"], 0.6);
            assert_eq!(results[0]["document"]["text"], "aaaaaa");
            assert_eq!(results[1]["index"], 2);
            assert_eq!(results[1]["document"]["text"], "aaaa");
            // Tokens for every pair, including the one top_n dropped.
            assert_eq!(v["usage"]["total_tokens"], 1 + 2 + 1 + 6 + 1 + 4);
            assert_eq!(state.counts.scored.load(Relaxed), 3);

            // Without `return_documents`, no document comes back.
            let (_, v) = call(
                &state,
                Method::POST,
                "/v1/rerank",
                r#"{"query": "q", "documents": ["a", "bb"]}"#,
            )
            .await;
            assert!(v["results"][0].get("document").is_none(), "{v}");
            assert_eq!(v["results"][0]["index"], 1);
        });
    }

    #[test]
    fn a_full_queue_is_refused_at_once_with_503() {
        block_on(async {
            let (gate_tx, gate_rx) = std::sync::mpsc::channel();
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let state = state_with(
                config(), // max_queue: 1
                Stub {
                    seen: Arc::default(),
                    gate: Some(gate_rx),
                    started: Some(started_tx),
                },
                false,
            );
            let post = |s: &Arc<State>| {
                let s = s.clone();
                tokio::spawn(async move {
                    handle(
                        request(Method::POST, "/v1/embeddings", r#"{"input": "a"}"#),
                        s,
                    )
                    .await
                    .status()
                })
            };

            // A is taken by the model thread and held at the gate...
            let a = post(&state);
            tokio::task::yield_now().await;
            started_rx.recv().unwrap();
            // ...B waits in the queue's one slot, and C finds it full.
            let b = post(&state);
            tokio::task::yield_now().await;
            let c = post(&state);
            assert_eq!(c.await.unwrap(), StatusCode::SERVICE_UNAVAILABLE);

            gate_tx.send(()).unwrap();
            gate_tx.send(()).unwrap();
            assert_eq!(a.await.unwrap(), StatusCode::OK);
            assert_eq!(b.await.unwrap(), StatusCode::OK);
            // A 503 is this side's, not the client's.
            assert_eq!(state.counts.failed.load(Relaxed), 1);
            assert_eq!(state.counts.rejected.load(Relaxed), 0);
        });
    }

    /// The wiring under hyper: a request on a byte stream gets its reply, the
    /// connection stays open for the next, and a stop signal closes it after
    /// the reply in flight.
    #[test]
    fn a_connection_serves_requests_until_told_to_stop() {
        block_on(async {
            let state = state();
            let (shutdown_tx, shutdown_rx) = watch::channel(false);
            let (mut client, server) = tokio::io::duplex(64 * 1024);
            let served = tokio::spawn(serve_connection(Box::pin(server), state, shutdown_rx));

            async fn exchange(client: &mut tokio::io::DuplexStream, request: &str) -> String {
                client.write_all(request.as_bytes()).await.unwrap();
                let mut buf = vec![0u8; 8192];
                let n = client.read(&mut buf).await.unwrap();
                String::from_utf8_lossy(&buf[..n]).into_owned()
            }

            let first = exchange(&mut client, "GET /health HTTP/1.1\r\nhost: k\r\n\r\n").await;
            assert!(first.starts_with("HTTP/1.1 200 OK\r\n"), "{first}");
            assert!(
                first.ends_with(r#"{"status":"ok","model":"stub/model"}"#),
                "{first}"
            );

            let body = r#"{"input":"ab"}"#;
            let second = exchange(
                &mut client,
                &format!(
                    "POST /v1/embeddings HTTP/1.1\r\nhost: k\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                ),
            )
            .await;
            assert!(second.starts_with("HTTP/1.1 200 OK\r\n"), "{second}");
            assert!(
                second.contains(r#""usage":{"prompt_tokens":2,"total_tokens":2}"#),
                "{second}"
            );

            shutdown_tx.send(true).unwrap();
            served.await.unwrap();
            // Closed from the server's side: nothing more to read.
            let mut rest = Vec::new();
            assert_eq!(client.read_to_end(&mut rest).await.unwrap(), 0);
        });
    }
}
