//! The listening side: accepting connections for as long as it is asked to,
//! serving each until its client leaves or a stop asks it to finish, and the
//! ordered stop itself (stop accepting, let the replies in flight finish,
//! close what is left).

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::Request;
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::sync::watch;
use tokio::task::JoinSet;

use super::http::{self, State};
use super::listen::{Bound, Io, Listen};
use crate::program::remark;

/// A bound listener and the connections it has accepted.
pub(crate) struct Server {
    bound: Bound,
    state: Arc<State>,
    /// Flipped to `true` at stop; every connection holds a receiver.
    stop: watch::Sender<bool>,
    connections: JoinSet<()>,
}

impl Server {
    pub(crate) async fn start(listen: &Listen, state: Arc<State>) -> Result<Self> {
        let bound = Bound::bind(listen).await?;
        let (stop, _) = watch::channel(false);
        Ok(Self {
            bound,
            state,
            stop,
            connections: JoinSet::new(),
        })
    }

    /// Where it is listening, as bound: `http://127.0.0.1:8080`, or the
    /// socket path.
    pub(crate) fn describe(&self) -> String {
        self.bound.describe()
    }

    /// Accept and serve connections until `until` completes, and return what
    /// it completed with. Connections outlive this call; `stop` closes them.
    pub(crate) async fn serve_until<T>(&mut self, until: impl Future<Output = T>) -> T {
        let mut until = std::pin::pin!(until);
        loop {
            self.reap_finished_connections();
            tokio::select! {
                biased;
                outcome = &mut until => return outcome,
                accepted = self.bound.accept() => match accepted {
                    Ok(io) => {
                        self.connections.spawn(serve_connection(
                            io,
                            self.state.clone(),
                            self.stop.subscribe(),
                        ));
                    }
                    Err(e) => {
                        // Out of descriptors, or a connection that went away
                        // between arriving and being accepted. Neither is a
                        // reason to stop; a pause keeps a persistent one from
                        // spinning.
                        remark!("accept failed: {e}");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                },
            }
        }
    }

    /// A [`JoinSet`] keeps a completed task until it is joined. Reap those
    /// connections during the run as well as at shutdown: a long-lived server
    /// must use memory for its open connections, not every connection it has
    /// ever answered.
    fn reap_finished_connections(&mut self) {
        while let Some(outcome) = self.connections.try_join_next() {
            if let Err(e) = outcome {
                remark!("connection task: {e}");
            }
        }
    }

    /// Tell every open connection to finish the reply in flight and close,
    /// wait up to `timeout` for them, and abandon the rest. Consumes the
    /// server, and with it the listener: a Unix socket's file goes too.
    pub(crate) async fn stop(mut self, timeout: Duration) {
        // An error means no connection is left to tell, which is the goal.
        let _ = self.stop.send(true);
        let drained = tokio::time::timeout(timeout, async {
            while self.connections.join_next().await.is_some() {}
        })
        .await;
        if drained.is_err() {
            remark!(
                "{} connections still open after {:?}; closing them",
                self.connections.len(),
                timeout
            );
            self.connections.abort_all();
        }
    }
}

/// One connection, for as many requests as the client sends on it. Told to
/// stop, it finishes the reply in flight and then closes, which is what hyper
/// calls a graceful shutdown; an idle connection closes at once.
async fn serve_connection(io: Io, state: Arc<State>, mut shutdown: watch::Receiver<bool>) {
    let service = service_fn(move |req: Request<Incoming>| {
        let state = state.clone();
        async move { Ok::<_, Infallible>(http::handle(req, state).await) }
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
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::super::{testing, worker, Batch, Config, Engine, Listen, Load};
    use super::*;
    use crate::{ModelInfo, TokenInfo};

    struct One;

    impl Engine for One {
        type Input = Vec<String>;
        type Output = Batch;

        fn info(&self) -> ModelInfo {
            testing::embedding_info(1)
        }

        fn answer(&self, texts: Vec<String>) -> anyhow::Result<Batch> {
            Ok(Batch {
                vectors: vec![vec![1.0]; texts.len()],
                tokens: vec![
                    TokenInfo {
                        n_tokens: 1,
                        truncated: false
                    };
                    texts.len()
                ],
            })
        }
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    fn state() -> Arc<State> {
        let config = Config {
            listen: Listen::default(),
            prefix: String::new(),
            max_inputs: 4,
            max_body_bytes: 1024,
            max_queue: 1,
            shutdown_timeout: Duration::from_secs(1),
        };
        let loaded = worker::spawn("test-serve", Load::new("one", || Ok(One)), config.max_queue)
            .unwrap()
            .loaded;
        Arc::new(State::new(&config, loaded, None))
    }

    /// The loop under the manual checks: a real TCP listener answers, the
    /// stop finishes the connection in flight, and the port closes with it.
    #[test]
    fn the_server_answers_on_tcp_and_stops_cleanly() {
        block_on(async {
            let listen = "127.0.0.1:0".parse::<Listen>().unwrap();
            let mut server = Server::start(&listen, state()).await.unwrap();
            let addr = server
                .describe()
                .strip_prefix("http://")
                .expect("a TCP listener describes a URL")
                .to_string();

            let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
            let running = tokio::spawn(async move {
                server
                    .serve_until(async {
                        let _ = stop_rx.await;
                    })
                    .await;
                server.stop(Duration::from_secs(1)).await;
            });

            let mut stream = tokio::net::TcpStream::connect(&addr).await.unwrap();
            stream
                .write_all(b"GET /health HTTP/1.1\r\nhost: k\r\n\r\n")
                .await
                .unwrap();
            let mut buf = vec![0u8; 4096];
            let n = stream.read(&mut buf).await.unwrap();
            let reply = String::from_utf8_lossy(&buf[..n]).into_owned();
            assert!(reply.starts_with("HTTP/1.1 200 OK\r\n"), "{reply}");

            stop_tx.send(()).unwrap();
            running.await.unwrap();
            // The connection was closed from the server's side...
            let mut rest = Vec::new();
            assert_eq!(stream.read_to_end(&mut rest).await.unwrap(), 0);
            // ...and the port answers no one else.
            assert!(tokio::net::TcpStream::connect(&addr).await.is_err());
        });
    }

    #[test]
    fn completed_connections_are_reaped_while_the_server_runs() {
        block_on(async {
            let listen = "127.0.0.1:0".parse::<Listen>().unwrap();
            let mut server = Server::start(&listen, state()).await.unwrap();
            server.connections.spawn(async {});
            tokio::task::yield_now().await;

            assert_eq!(server.connections.len(), 1);
            server.reap_finished_connections();
            assert!(server.connections.is_empty());
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
                first.ends_with(r#"{"status":"ok","model":"one"}"#),
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
                second.contains(r#""usage":{"prompt_tokens":1,"total_tokens":1}"#),
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
