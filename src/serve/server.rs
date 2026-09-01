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
