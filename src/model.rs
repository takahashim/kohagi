//! Model loading and the memory-bounded parallel encoder.
//!
//! A single candle ModernBERT forward is effectively single-core on CPU, so
//! [`Embedder::embed`] fans length-bucketed batches across a rayon pool — the
//! weights are shared behind an `Arc`, each worker runs an independent
//! forward, and the result is identical to serial execution.
//!
//! Two guardrails keep peak memory flat no matter what the caller passes:
//!
//! 1. Rows per forward are capped by [`ATTN_BUDGET`]: candle's ModernBERT
//!    materializes ~`batch * heads * seq^2` f32 of attention scratch per
//!    layer, so a 64-row batch of seq-512 inputs would hold ~2 GB per worker.
//!    2 rows at seq 512 (~67 MB) measured both fastest and smallest on an
//!    8-core Zen4 — finer units also load-balance better across the pool.
//! 2. The pool defaults to *physical* cores (`RAYON_NUM_THREADS` overrides):
//!    worker count is a direct memory multiplier, and SMT siblings only add
//!    contention to these GEMM-bound forwards.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::{Config, ModernBert};
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::batch::{l2_normalize, load_tokenizer, pool_one, tokenize_bucket, BatchInput, Pooling};

/// Attention-scratch budget per forward, in `rows * seq^2` elements.
const ATTN_BUDGET: usize = 2 * 512 * 512;

/// Rows allowed in one forward of padded length `seq`.
fn rows_per_forward(seq: usize) -> usize {
    (ATTN_BUDGET / (seq * seq).max(1)).max(1)
}

/// Pooled vectors from one forward, tagged with their caller-side row index.
type PooledRows = Vec<(usize, Vec<f32>)>;

/// Where the model weights come from.
pub enum ModelSource {
    /// A Hugging Face Hub repo, e.g. `cl-nagoya/ruri-v3-130m`. Downloads
    /// `model.safetensors`, `config.json`, and `tokenizer.json` into the
    /// standard HF cache (`~/.cache/huggingface`, `HF_HOME` respected) on
    /// first use; later runs are offline.
    Hub { repo: String },
    /// Local files: the safetensors weights and tokenizer.json, with
    /// `config.json` expected next to the weights. No network access.
    Files { model: PathBuf, tokenizer: PathBuf },
}

/// Knobs for [`Embedder::load`]. `Default` matches Ruri v3.
pub struct Options {
    pub pooling: Pooling,
    /// L2-normalize each embedding (unit length, so dot = cosine).
    pub normalize: bool,
    /// Token-level truncation length. Ruri v3 accepts up to 8192 but was
    /// trained for retrieval at ~512; longer costs seq^2 attention compute.
    pub max_seq_length: usize,
    /// Bucketing granularity (rows per padded batch before the memory cap).
    pub batch_size: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { pooling: Pooling::Mean, normalize: true, max_seq_length: 512, batch_size: 64 }
    }
}

/// A loaded ModernBERT sentence encoder. Cheap to share by reference; one
/// instance can serve any number of `embed` calls.
pub struct Embedder {
    model: Arc<ModernBert>,
    device: Device,
    tokenizer: Tokenizer,
    opts: Options,
    dim: usize,
}

impl Embedder {
    pub fn load(source: &ModelSource, opts: Options) -> Result<Self> {
        let (model_path, tokenizer_path) = match source {
            ModelSource::Files { model, tokenizer } => (model.clone(), tokenizer.clone()),
            ModelSource::Hub { repo } => fetch_from_hub(repo)?,
        };

        let config_path = model_path
            .parent()
            .map(|d| d.join("config.json"))
            .context("model path has no parent dir for config.json")?;
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("cannot read {}", config_path.display()))?;
        let config: Config = serde_json::from_str(&config_str)
            .with_context(|| format!("cannot parse {}", config_path.display()))?;
        let dim = config.hidden_size;

        let device = Device::Cpu;
        let model = load_modernbert(&model_path, &config, &device)?;
        let tokenizer = load_tokenizer(&tokenizer_path, opts.max_seq_length)?;
        Ok(Self { model: Arc::new(model), device, tokenizer, opts, dim })
    }

    /// The embedding dimension (`hidden_size` — 512 for ruri-v3-130m).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embed a batch of texts, one vector per text, in input order. Prefixes
    /// (e.g. Ruri's `"検索文書: "`) are the caller's job — pass prefixed text.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batches = tokenize_bucket(&self.tokenizer, texts, self.opts.batch_size)?;

        // Split each bucketed batch into forwards that fit the attention
        // budget (row ranges over the batch's padded layout).
        let mut units: Vec<(&BatchInput, usize, usize)> = Vec::new();
        for b in &batches {
            let cap = rows_per_forward(b.seq);
            let mut start = 0;
            while start < b.batch {
                let rows = cap.min(b.batch - start);
                units.push((b, start, rows));
                start += rows;
            }
        }

        let model = &self.model;
        let device = &self.device;
        let pooling = self.opts.pooling;
        let per_unit: Vec<Result<PooledRows>> = worker_pool()?.install(|| {
            units
                .par_iter()
                .map(|&(b, start, rows)| {
                    let (lo, hi) = (start * b.seq, (start + rows) * b.seq);
                    let (ids, mask) = (&b.ids[lo..hi], &b.mask[lo..hi]);
                    let (data, dim) = forward(model, device, ids, mask, rows, b.seq)?;
                    let mut out = Vec::with_capacity(rows);
                    for bi in 0..rows {
                        let orig = b.orig[start + bi];
                        out.push((orig, pool_one(&data, mask, bi, b.seq, dim, pooling)));
                    }
                    Ok(out)
                })
                .collect()
        });

        let mut rows_out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        for unit in per_unit {
            for (orig, mut vec) in unit? {
                if self.opts.normalize {
                    l2_normalize(&mut vec);
                }
                rows_out[orig] = vec;
            }
        }
        Ok(rows_out)
    }
}

/// Download (or reuse from the HF cache) the three files a model needs.
fn fetch_from_hub(repo: &str) -> Result<(PathBuf, PathBuf)> {
    let api = hf_hub::api::sync::Api::new().context("initializing Hugging Face Hub client")?;
    let repo = api.model(repo.to_string());
    let get = |f: &str| {
        repo.get(f)
            .with_context(|| format!("cannot fetch {f} (network down? try local --model-path)"))
    };
    let model = get("model.safetensors")?;
    get("config.json")?; // lands next to the weights in the cache
    let tokenizer = get("tokenizer.json")?;
    Ok((model, tokenizer))
}

fn load_modernbert(weights: &Path, config: &Config, device: &Device) -> Result<ModernBert> {
    // Bare encoder checkpoints (ruri, modernbert-embed) store weights at the
    // root (embeddings.*, layers.*, final_norm.*). Try that first; if this
    // candle expects the "model." prefix (used by its MLM/classification
    // wrappers), remap the keys and retry.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
    if let Ok(m) = ModernBert::load(vb, config) {
        return Ok(m);
    }
    let tensors = candle_core::safetensors::load(weights, device)?;
    let remapped: std::collections::HashMap<String, Tensor> =
        tensors.into_iter().map(|(k, v)| (format!("model.{k}"), v)).collect();
    let vb = VarBuilder::from_tensors(remapped, DType::F32, device);
    ModernBert::load(vb, config)
        .context("candle ModernBert::load failed (tried both root and 'model.' prefix)")
}

/// Run one forward pass, returning flat `[batch * seq * dim]` hidden states.
fn forward(
    model: &ModernBert,
    device: &Device,
    ids: &[i64],
    mask: &[i64],
    batch: usize,
    seq: usize,
) -> Result<(Vec<f32>, usize)> {
    let ids_u: Vec<u32> = ids.iter().map(|&v| v as u32).collect();
    let mask_u: Vec<u32> = mask.iter().map(|&v| v as u32).collect();
    let xs = Tensor::from_vec(ids_u, (batch, seq), device)?;
    let m = Tensor::from_vec(mask_u, (batch, seq), device)?;
    let out = model.forward(&xs, &m)?; // [batch, seq, dim]
    let dim = out.dim(2)?;
    let data = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok((data, dim))
}

/// Physical-core rayon pool (see module docs); `RAYON_NUM_THREADS` overrides.
fn worker_pool() -> Result<rayon::ThreadPool> {
    let n = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(num_cpus::get_physical);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .context("building rayon pool")
}
