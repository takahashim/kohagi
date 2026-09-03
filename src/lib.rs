//! Kohagi — local sentence embeddings for [Ruri v3] and other ModernBERT
//! encoders, in pure Rust (via [candle]).
//!
//! The library is one type: [`Embedder`]. Load a model (from the Hugging Face
//! Hub or local files), hand it a batch of texts, get back one `Vec<f32>` per
//! text. The binary in `main.rs` wraps it in a stdin/stdout JSONL protocol
//! (see `stdio.rs` and PROTOCOL.md) so any language that can spawn a process
//! can embed text without an HTTP server; `kohagi-serve` (see `serve`) puts
//! the same type behind an OpenAI-compatible `/v1/embeddings`, for callers
//! that want one model per host instead of one per process.
//!
//! ```no_run
//! use kohagi::{Embedder, ModelSource, Options};
//!
//! let embedder = Embedder::load(
//!     &ModelSource::Hub { repo: "cl-nagoya/ruri-v3-130m".into() },
//!     Options::default(),
//! )?;
//! let vecs = embedder.embed(&["検索文書: 瑠璃も玻璃も照らせば光る"])?;
//! assert_eq!(vecs[0].len(), embedder.dim());
//! # anyhow::Ok(())
//! ```
//!
//! Memory is bounded by design: rows per forward pass are capped by an
//! attention budget and the batch fan-out runs on a physical-core thread pool,
//! so peak memory depends on core count, not input size. See `model.rs`.
//!
//! [Ruri v3]: https://huggingface.co/cl-nagoya/ruri-v3-130m
//! [candle]: https://github.com/huggingface/candle

mod attention;
mod batch;
#[cfg(target_arch = "x86_64")]
pub mod bf16;
pub mod cli;
mod config;
#[cfg(feature = "coreml")]
mod coreml;
#[cfg(feature = "coreml-export")]
pub mod coreml_export;
#[cfg(feature = "coreml-export")]
mod coreml_proto;
mod encoder;
mod errors;
mod fingerprint;
// Only the CoreML caches and the emitter's golden test need a stable hash.
#[cfg(any(feature = "coreml", test))]
mod fnv;
mod fused;
mod info;
mod model;
pub mod program;
mod protocol;
pub mod rerank;
pub mod serve;
mod source;
pub mod stdio;

/// This crate's version, for a caller that records which Kohagi produced a
/// file. A tool of its own has its own `CARGO_PKG_VERSION`, which is not this
/// one, and an artifact that names the wrong version is worse than one that
/// names none.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use batch::{Pooling, TokenInfo};
pub use config::{CoreMlForm, CoreMlQuantize};
pub use errors::UnsupportedRequest;
pub use info::{Bundle, ModelInfo, Output};
pub use model::{Backend, Embedder, Options, Precision};
pub use source::ModelSource;
