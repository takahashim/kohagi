//! The content hash that says which weights a run used.
//!
//! Fine-tuning a checkpoint produces directories that differ only in their
//! weights, and a blend of two checkpoints differs from its neighbours only in
//! one scalar. A results file that records the *path* it loaded records what
//! someone meant to run rather than what ran, which is how an artifact mix-up
//! survives long enough to be believed. A digest of the bytes closes that gap:
//! same bytes, same digest; one byte different, different digest.
//!
//! It is the file's bytes rather than a normalized hash over tensor values,
//! because the question is only ever "are these two artifacts the same one".
//! Hashing the file needs no agreement about tensor order, dtype or metadata
//! to be comparable between two machines.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::program::remark;
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// Hex digits carried in the stderr summary, where the whole digest would
/// crowd out the counts. 48 bits is far past what telling apart a handful of
/// checkpoints needs, and the full digest is a `--print-model-info` away.
const SHORT: usize = 12;

/// The sha256 of every byte in `path`, hex-encoded.
///
/// Streamed through a fixed buffer: a 500 MB checkpoint costs one sequential
/// read and a megabyte of memory. 0.36 s over ruri-v3-130m's 528 MB on an M2,
/// or ~1.5 GB/s — sha2 0.11 finds the CPU's SHA-256 instructions by runtime
/// detection, where 0.10 reached them only through a feature that builds C.
/// Fast enough not to matter, and still run off the load path by
/// [`Fingerprint`], because a checkpoint on a network share is read at the
/// share's speed rather than the CPU's.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("opening {} to hash it", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("reading {} to hash it", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// The prefix of a digest that goes in the summary line.
pub fn short(digest: &str) -> &str {
    &digest[..SHORT.min(digest.len())]
}

/// A weights digest being computed on its own thread.
///
/// Reading half a gigabyte takes as long as the disk takes, which on a network
/// share is seconds rather than the CPU's third of one. Paying that *before*
/// the first forward pass would make every run start late for a value only the
/// summary needs, so it runs beside the work and is collected at the end,
/// where a real run has long since covered it. A run too short to cover it is
/// one that prints no summary at all (`--text`), and never asks.
///
/// The one caller that waits is `--print-model-info`, whose whole output this
/// is.
pub struct Fingerprint(Mutex<State>);

enum State {
    Running(std::thread::JoinHandle<Result<String>>),
    Settled(Option<String>),
}

impl Fingerprint {
    /// Start hashing `path` now; the caller carries on loading it.
    pub fn spawn(path: PathBuf) -> Self {
        Self(Mutex::new(State::Running(std::thread::spawn(move || {
            sha256_file(&path)
        }))))
    }

    /// The digest, waiting for it if it is not ready.
    ///
    /// `None` when the file could not be read after all — a checkpoint
    /// replaced mid-run, say. That is reported and dropped rather than made
    /// fatal: the vectors this run produced are still good, and an unknown
    /// fingerprint that says so is better than a run that fails at the finish
    /// line. It is never a *wrong* fingerprint, which is the failure that
    /// would matter.
    pub fn get(&self) -> Option<String> {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let State::Running(_) = &*state {
            let State::Running(handle) = std::mem::replace(&mut *state, State::Settled(None))
            else {
                unreachable!("just matched Running")
            };
            let digest = match handle.join() {
                Ok(Ok(digest)) => Some(digest),
                Ok(Err(e)) => {
                    remark!("could not hash the weights ({e:#}); fingerprint unknown");
                    None
                }
                Err(_) => {
                    remark!("the thread hashing the weights panicked; fingerprint unknown");
                    None
                }
            };
            *state = State::Settled(digest);
        }
        match &*state {
            State::Settled(digest) => digest.clone(),
            State::Running(_) => unreachable!("settled just above"),
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The known digest of the empty input, which pins the hex encoding as well
    /// as the algorithm.
    #[test]
    fn hashes_the_bytes_and_not_the_path() {
        let dir = std::env::temp_dir().join(format!("kohagi-fp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let empty = dir.join("empty.bin");
        std::fs::write(&empty, b"").unwrap();
        assert_eq!(
            sha256_file(&empty).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        // Two names, one content: the digest is of the bytes.
        let a = dir.join("a.bin");
        let b = dir.join("b.bin");
        std::fs::write(&a, b"the same bytes").unwrap();
        std::fs::write(&b, b"the same bytes").unwrap();
        assert_eq!(sha256_file(&a).unwrap(), sha256_file(&b).unwrap());

        // One byte apart — the interpolated-checkpoint case in miniature.
        std::fs::write(&b, b"the same byteS").unwrap();
        assert_ne!(sha256_file(&a).unwrap(), sha256_file(&b).unwrap());

        // Longer than the read buffer, so the streaming loop is exercised
        // rather than a single read.
        let big = dir.join("big.bin");
        let mut bytes = vec![7u8; (1 << 20) + 1234];
        std::fs::write(&big, &bytes).unwrap();
        let before = sha256_file(&big).unwrap();
        bytes[1 << 20] ^= 1;
        std::fs::write(&big, &bytes).unwrap();
        assert_ne!(before, sha256_file(&big).unwrap());
        assert_eq!(short(&before).len(), 12);
        assert!(before.starts_with(short(&before)));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
