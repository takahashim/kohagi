//! Where the server listens: a TCP address, or a Unix socket path spelled the
//! way Redis and Docker spell theirs (`unix:///run/kohagi.sock`). Both carry
//! the same HTTP; the socket is for a server that shares a host with its
//! callers, where a file's permissions are the whole access control and no
//! port needs choosing.

use std::fmt;
use std::io;
#[cfg(unix)]
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::str::FromStr;

use anyhow::{Context, Result};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::net::UnixListener;

#[cfg(unix)]
pub(crate) struct SocketFile {
    path: PathBuf,
    /// Held for the listener's whole life. This is an advisory lock, released
    /// on a crash, that serializes stale-socket cleanup and binding.
    _lock: std::fs::File,
    dev: u64,
    ino: u64,
}

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
    Unix(UnixListener, SocketFile),
}

impl Bound {
    pub(crate) async fn bind(listen: &Listen) -> Result<Self> {
        match listen {
            Listen::Tcp(addr) => TcpListener::bind(addr)
                .await
                .with_context(|| format!("listening on {addr}"))
                .map(Bound::Tcp),
            #[cfg(unix)]
            Listen::Unix(path) => bind_unix(path),
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
            Bound::Unix(_, socket) => format!("unix://{}", socket.path.display()),
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
        if let Bound::Unix(_, socket) = self {
            // Do not remove a replacement socket. The lock prevents another
            // kohagi-serve process from making one, but an operator may still
            // have replaced the path while this listener was alive.
            use std::os::unix::fs::{FileTypeExt, MetadataExt};

            let mine = std::fs::symlink_metadata(&socket.path).is_ok_and(|meta| {
                meta.file_type().is_socket() && meta.dev() == socket.dev && meta.ino() == socket.ino
            });
            if mine {
                let _ = std::fs::remove_file(&socket.path);
            }
        }
    }
}

/// Take over a socket path: hold the per-socket lock, clear a previous run's
/// socket if that is what is there, bind, and shut the file to its owner. The
/// listener's own file is recorded last, and is what [`Bound`]'s `Drop`
/// unlinks.
#[cfg(unix)]
fn bind_unix(path: &Path) -> Result<Bound> {
    let lock = lock_socket(path)?;
    remove_stale_socket(path)?;
    let listener = UnixListener::bind(path)
        .with_context(|| format!("listening on unix://{}", path.display()))?;
    // Owner only, as a database's socket is. The file exists for a moment
    // before this with the umask's mode; a connection in that moment still
    // waits for accept, which starts after.
    std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o600))
        .with_context(|| format!("setting the mode of {}", path.display()))?;
    Ok(Bound::Unix(listener, SocketFile::at(path, lock)?))
}

#[cfg(unix)]
impl SocketFile {
    fn at(path: &Path, lock: std::fs::File) -> Result<Self> {
        use std::os::unix::fs::MetadataExt;

        let meta = std::fs::symlink_metadata(path)
            .with_context(|| format!("checking the new socket {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            _lock: lock,
            dev: meta.dev(),
            ino: meta.ino(),
        })
    }
}

/// Take the per-socket lock before deciding whether a socket is stale. Without
/// it two starts can both unlink the old path and each bind a listener; the
/// second then steals new clients and the first may remove its path at exit.
#[cfg(unix)]
fn lock_socket(path: &Path) -> Result<std::fs::File> {
    use std::fs::{OpenOptions, TryLockError};

    let lock_path = PathBuf::from(format!("{}.lock", path.display()));
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("opening socket lock {}", lock_path.display()))?;
    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(TryLockError::WouldBlock) => anyhow::bail!(
            "{} is already owned by another kohagi-serve process",
            path.display()
        ),
        Err(TryLockError::Error(e)) => {
            Err(e).with_context(|| format!("locking socket {}", path.display()))
        }
    }
}

/// What is at the path is a previous run's socket, left by a crash or a kill:
/// nothing else has business there, so it is removed. Anything that is not a
/// socket is refused rather than removed, since that would be someone's file.
#[cfg(unix)]
fn remove_stale_socket(path: &Path) -> Result<()> {
    use std::os::unix::fs::FileTypeExt;
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_socket() => {
            match std::os::unix::net::UnixStream::connect(path) {
                Ok(_) => anyhow::bail!(
                    "{} has a listening server; refusing to replace it",
                    path.display()
                ),
                Err(e) if e.kind() == io::ErrorKind::ConnectionRefused => {
                    std::fs::remove_file(path)
                        .with_context(|| format!("removing the stale socket {}", path.display()))
                }
                Err(e) => Err(e).with_context(|| format!("checking socket {}", path.display())),
            }
        }
        Ok(_) => anyhow::bail!(
            "{} exists and is not a socket; refusing to replace it",
            path.display()
        ),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("checking {}", path.display())),
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

        // A socket left by a killed run is replaced; a live one is not, and a
        // file that is not a socket is not either.
        block_on(async {
            let live = Bound::bind(&listen).await.unwrap();
            let refused = Bound::bind(&listen).await.err().expect("refused");
            assert!(refused.to_string().contains("already owned"), "{refused}");
            drop(live);

            let stale = std::os::unix::net::UnixListener::bind(&path).unwrap();
            drop(stale);
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
