//! The listening side: accepting connections for as long as it is asked to,
//! and the ordered stop (stop accepting, let the replies in flight finish,
//! close what is left).

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::watch;
use tokio::task::JoinSet;

use super::http::{self, State};
use super::listen::{Bound, Listen};
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
            tokio::select! {
                biased;
                outcome = &mut until => return outcome,
                accepted = self.bound.accept() => match accepted {
                    Ok(io) => {
                        self.connections.spawn(http::serve_connection(
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::super::{worker, Batch, Config, Engine, Listen, Load};
    use super::*;
    use crate::{ModelInfo, Output, TokenInfo};

    struct One;

    impl Engine for One {
        type Input = Vec<String>;
        type Output = Batch;

        fn info(&self) -> ModelInfo {
            ModelInfo {
                backend: "cpu",
                precision: "f32",
                sha256: None,
                bundle: None,
                pooling: "mean",
                dim: 1,
                max_seq_length: 8,
                declared_max_seq_length: None,
                output: Output::Embedding {
                    output_dim: None,
                    normalized: true,
                },
            }
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

    fn state() -> Arc<State> {
        let config = Config {
            listen: Listen::default(),
            prefix: String::new(),
            max_inputs: 4,
            max_body_bytes: 1024,
            max_queue: 1,
            shutdown_timeout: Duration::from_secs(1),
        };
        let loaded = worker::spawn(
            "test-serve",
            Load {
                label: "one".to_string(),
                load: || Ok(One),
            },
            config.max_queue,
        )
        .unwrap()
        .loaded;
        Arc::new(State::new(&config, loaded, None))
    }

    /// The loop under the manual checks: a real TCP listener answers, the
    /// stop finishes the connection in flight, and the port closes with it.
    #[test]
    fn the_server_answers_on_tcp_and_stops_cleanly() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
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
}
