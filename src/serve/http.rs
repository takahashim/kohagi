//! Routes and handlers: which path is answered by which model, and what one
//! request's answer is made of. `handle` never fails: every outcome, a refusal
//! included, is a reply (`reply`), and the connection stays usable for the
//! next one.

use std::sync::Arc;

use http_body_util::{BodyExt, LengthLimitError, Limited};
use hyper::body::{Body, Bytes};
use hyper::header::CONTENT_LENGTH;
use hyper::{Method, Request, StatusCode};

use super::api::{self, embeddings, rerank};
use super::counts::Counts;
use super::reply::{json, ApiError, Reply};
use super::worker::Loaded;
use super::{Batch, Config, Pairs, Scores};
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

    /// The loaded models' names, for a refusal that has to say what this
    /// server does run instead: one name, or both joined by "and".
    fn loaded_models(&self) -> String {
        self.models()
            .iter()
            .map(|(label, _)| format!("`{label}`"))
            .collect::<Vec<_>>()
            .join(" and ")
    }

    /// Every loaded model, as `/v1/models` lists them.
    fn models(&self) -> Vec<(&str, &ModelInfo)> {
        let mut models = vec![(self.embedder.label.as_str(), &self.embedder.info)];
        if let Some(reranker) = &self.reranker {
            models.push((reranker.label.as_str(), &reranker.info));
        }
        models
    }

    /// The run's one summary line: the numbers are [`Counts`]', and which
    /// model answered is this state's.
    pub(crate) fn summarize(&self) {
        self.counts
            .summarize(&self.embedder.label, &self.embedder.info.summary_facts());
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
                    Some((label, info)) => Ok(json(StatusCode::OK, api::model_body(label, info))),
                    // Every loaded model is named, not just the embedder: a
                    // server with a reranker has two, and the one the caller
                    // meant is likelier to be in the list than in a guess.
                    None => Err(ApiError::not_found(format!(
                        "model `{id}` is not loaded; this server runs {}",
                        state.loaded_models()
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
            Ok(json(StatusCode::OK, api::models_body(&state.models())))
        }
        (Method::GET | Method::HEAD, "/health") => Ok(json(
            StatusCode::OK,
            api::health_body(&state.embedder.label),
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
    let request: embeddings::Request = serde_json::from_slice(&body).map_err(|e| {
        ApiError::invalid(None, format!("the body is not an embeddings request: {e}"))
    })?;
    let request = embeddings::validate(request, state.max_inputs, &state.embedder.info)?
        .prefixed(&state.prefix);

    let mut batch = state
        .embedder
        .handle
        .ask(request.texts)
        .await
        .map_err(|e| ApiError::worker(e, &state.embedder.label))?;
    if let Some(dims) = request.dimensions {
        embeddings::truncate(&mut batch.vectors, dims);
    }
    state.counts.embedded(&batch);

    Ok(json(
        StatusCode::OK,
        embeddings::reply(
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
    let request: rerank::Request = serde_json::from_slice(&body)
        .map_err(|e| ApiError::invalid(None, format!("the body is not a rerank request: {e}")))?;
    let (pairs, reply) = rerank::validate(request, state.max_inputs)?.into_parts();

    let scores = reranker
        .handle
        .ask(pairs)
        .await
        .map_err(|e| ApiError::worker(e, &reranker.label))?;
    state.counts.reranked(&scores);

    Ok(json(StatusCode::OK, reply.body(&reranker.label, &scores)))
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::Mutex;

    use base64::Engine as _;
    use http_body_util::Full;
    use serde_json::Value;

    use super::super::{testing, worker};
    use super::super::{Engine, Listen, Load};
    use super::*;
    use crate::TokenInfo;

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
            testing::embedding_info(4)
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
            testing::reranker_info()
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
        let embedder = Load::new("stub/model", move || Ok(stub));
        let embedder = worker::spawn("test-model", embedder, config.max_queue)
            .unwrap()
            .loaded;
        let reranker = reranker.then(|| {
            let reranker = Load::new("stub/reranker", || Ok(RerankStub));
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

            // A model that is not here is refused by naming the ones that are,
            // both of them: the caller's next guess should not have to be one.
            let (status, v) = call(&state, Method::GET, "/v1/models/other", "").await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            let message = v["error"]["message"].as_str().unwrap();
            assert!(
                message.contains("`stub/model` and `stub/reranker`"),
                "{message}"
            );

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
}
