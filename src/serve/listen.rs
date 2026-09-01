//! Where the server listens: a TCP address, or a Unix socket path spelled the
//! way Redis and Docker spell theirs (`unix:///run/kohagi.sock`). Both carry
//! the same HTTP; the socket is for a server that shares a host with its
//! callers, where a file's permissions are the whole access control and no
//! port needs choosing.

use std::fmt;
use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::str::FromStr;

use anyhow::{Context, Result};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::net::UnixListener;

/// `--listen`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Listen {
    /// `host:port`. Loopback unless told otherwise: this server has no
    /// authentication, like a database's socket, and is not for the open
    /// network.
    Tcp(String),
    /// `unix:///path` (also `unix:/path`). Unix only.
    Unix(PathBuf),
}

impl Default for Listen {
    fn default() -> Self {
        Listen::Tcp("127.0.0.1:8080".to_string())
    }
}

impl FromStr for Listen {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        if let Some(rest) = s.strip_prefix("unix:") {
            let path = rest.strip_prefix("//").unwrap_or(rest);
            if path.is_empty() {
                return Err(
                    "`unix:` needs a socket path, as in unix:///run/kohagi.sock".to_string()
                );
            }
            #[cfg(not(unix))]
            return Err(
                "Unix sockets are not available on this platform; listen on host:port".to_string(),
            );
            #[cfg(unix)]
            return Ok(Listen::Unix(PathBuf::from(path)));
        }
        match s.rsplit_once(':') {
            Some((host, port)) if !host.is_empty() && port.parse::<u16>().is_ok() => {
                Ok(Listen::Tcp(s.to_string()))
            }
            _ => Err(format!("expected host:port or unix:///path, not `{s}`")),
        }
    }
}

impl fmt::Display for Listen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Listen::Tcp(addr) => f.write_str(addr),
            Listen::Unix(path) => write!(f, "unix://{}", path.display()),
        }
    }
}

/// One accepted connection, whichever listener it came from.
pub(crate) trait AsyncStream: AsyncRead + AsyncWrite + Send {}
impl<T: AsyncRead + AsyncWrite + Send> AsyncStream for T {}
pub(crate) type Io = Pin<Box<dyn AsyncStream>>;

/// A listener that is up. Dropping a Unix one removes its socket file.
pub(crate) enum Bound {
    Tcp(TcpListener),
    #[cfg(unix)]
    Unix(UnixListener, PathBuf),
}

impl Bound {
    pub(crate) async fn bind(listen: &Listen) -> Result<Self> {
        match listen {
            Listen::Tcp(addr) => TcpListener::bind(addr)
                .await
                .with_context(|| format!("listening on {addr}"))
                .map(Bound::Tcp),
            #[cfg(unix)]
            Listen::Unix(path) => {
                remove_stale_socket(path)?;
                let listener = UnixListener::bind(path)
                    .with_context(|| format!("listening on unix://{}", path.display()))?;
                // Owner only, as a database's socket is. The file exists for a
                // moment before this with the umask's mode; a connection in that
                // moment still waits for accept, which starts after.
                std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o600))
                    .with_context(|| format!("setting the mode of {}", path.display()))?;
                Ok(Bound::Unix(listener, path.clone()))
            }
            #[cfg(not(unix))]
            Listen::Unix(_) => unreachable!("refused when the flag was parsed"),
        }
    }

    /// What the log says was bound: the address as the kernel chose it (port
    /// 0 becomes a real port), or the socket path.
    pub(crate) fn describe(&self) -> String {
        match self {
            Bound::Tcp(listener) => match listener.local_addr() {
                Ok(addr) => format!("http://{addr}"),
                Err(_) => "http://?".to_string(),
            },
            #[cfg(unix)]
            Bound::Unix(_, path) => format!("unix://{}", path.display()),
        }
    }

    pub(crate) async fn accept(&self) -> io::Result<Io> {
        match self {
            Bound::Tcp(listener) => {
                let (stream, _) = listener.accept().await?;
                // Replies are one write each; nothing to gain from Nagle.
                stream.set_nodelay(true)?;
                Ok(Box::pin(stream))
            }
            #[cfg(unix)]
            Bound::Unix(listener, _) => {
                let (stream, _) = listener.accept().await?;
                Ok(Box::pin(stream))
            }
        }
    }
}

#[cfg(unix)]
impl Drop for Bound {
    fn drop(&mut self) {
        if let Bound::Unix(_, path) = self {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// What is at the path is a previous run's socket, left by a crash or a kill:
/// nothing else has business there, so it is removed. Anything that is not a
/// socket is refused rather than removed, since that would be someone's file.
#[cfg(unix)]
fn remove_stale_socket(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::FileTypeExt;
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_socket() => std::fs::remove_file(path)
            .with_context(|| format!("removing the stale socket {}", path.display())),
        Ok(_) => anyhow::bail!(
            "{} exists and is not a socket; refusing to replace it",
            path.display()
        ),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("checking {}", path.display())),
    }
}

/// Completes on SIGTERM or SIGINT (Ctrl-C), which is how a supervisor and a
/// terminal each ask for a stop. On Windows, on Ctrl-C, Ctrl-Break, or the
/// console closing.
pub(crate) async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("registering a SIGTERM handler");
        let mut int = signal(SignalKind::interrupt()).expect("registering a SIGINT handler");
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
    }
    #[cfg(windows)]
    {
        use tokio::signal::windows::{ctrl_break, ctrl_c, ctrl_close, ctrl_shutdown};
        let mut c = ctrl_c().expect("registering a Ctrl-C handler");
        let mut b = ctrl_break().expect("registering a Ctrl-Break handler");
        let mut close = ctrl_close().expect("registering a close handler");
        let mut down = ctrl_shutdown().expect("registering a shutdown handler");
        tokio::select! {
            _ = c.recv() => {}
            _ = b.recv() => {}
            _ = close.recv() => {}
            _ = down.recv() => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_listen_flag_is_an_address_or_a_socket_path() {
        assert_eq!(
            "127.0.0.1:8080".parse::<Listen>().unwrap(),
            Listen::Tcp("127.0.0.1:8080".to_string())
        );
        assert_eq!(
            "[::1]:8080".parse::<Listen>().unwrap(),
            Listen::Tcp("[::1]:8080".to_string())
        );
        assert_eq!(Listen::default(), "127.0.0.1:8080".parse().unwrap());
        for bad in [
            "8080",
            ":8080",
            "localhost",
            "localhost:http",
            "unix:",
            "unix://",
        ] {
            assert!(bad.parse::<Listen>().is_err(), "{bad}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_socket_path_is_spelled_like_redis_spells_it() {
        for spelled in ["unix:///run/kohagi.sock", "unix:/run/kohagi.sock"] {
            let listen = spelled.parse::<Listen>().unwrap();
            assert_eq!(
                listen,
                Listen::Unix(PathBuf::from("/run/kohagi.sock")),
                "{spelled}"
            );
            assert_eq!(listen.to_string(), "unix:///run/kohagi.sock");
        }
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    #[test]
    fn a_tcp_listener_reports_the_port_the_kernel_chose() {
        block_on(async {
            let bound = Bound::bind(&"127.0.0.1:0".parse().unwrap()).await.unwrap();
            let described = bound.describe();
            assert!(described.starts_with("http://127.0.0.1:"), "{described}");
            assert!(!described.ends_with(":0"), "{described}");
        });
    }

    #[cfg(unix)]
    #[test]
    fn a_unix_listener_owns_its_file_for_its_life() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("kohagi-serve-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("k.sock");
        let listen = Listen::Unix(path.clone());

        block_on(async {
            let bound = Bound::bind(&listen).await.unwrap();
            assert_eq!(bound.describe(), format!("unix://{}", path.display()));
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
            drop(bound);
            assert!(!path.exists(), "the socket file outlived the listener");
        });

        // A socket left by a killed run is replaced; a file that is not a
        // socket is not.
        block_on(async {
            let stale = Bound::bind(&listen).await.unwrap();
            std::mem::forget(stale);
            assert!(path.exists());
            let again = Bound::bind(&listen).await.unwrap();
            drop(again);

            std::fs::write(&path, "not a socket").unwrap();
            let refused = Bound::bind(&listen).await.err().expect("refused");
            assert!(refused.to_string().contains("not a socket"), "{refused}");
            assert!(path.exists(), "the file was removed");
        });

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
