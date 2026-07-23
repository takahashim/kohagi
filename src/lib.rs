//! kohagi — local sentence embeddings for [Ruri v3] and other ModernBERT
//! encoders, in pure Rust (via [candle]).
//!
//! The library is one type: [`Embedder`]. Load a model (from the Hugging Face
//! Hub or local files), hand it a batch of texts, get back one `Vec<f32>` per
//! text. The binary in `main.rs` wraps it in a stdin/stdout JSONL protocol
//! (see `stdio.rs` and PROTOCOL.md) so any language that can spawn a process
//! can embed text without an HTTP server.
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

mod batch;
#[cfg(target_arch = "x86_64")]
pub mod bf16;
#[cfg(feature = "coreml")]
mod coreml;
mod encoder;
mod errors;
mod fused;
mod model;
pub mod stdio;

pub use batch::Pooling;
pub use errors::UnsupportedRequest;
pub use model::{Backend, CoreMlForm, Embedder, ModelSource, Options, Precision};
