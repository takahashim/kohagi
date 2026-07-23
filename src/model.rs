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

use crate::encoder::{Config, ModernBert};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::batch::{l2_normalize, load_tokenizer, pool_row, tokenize_bucket, BatchInput, Pooling};

/// Attention-scratch budget per forward, in `rows * seq^2` elements.
const ATTN_BUDGET: usize = 2 * 512 * 512;

/// Same budget for the GPU, which runs one stream of wide forwards rather than
/// many narrow ones.
///
/// The width barely matters now: with the vendored candle's SDPA the attention
/// scores are never materialized, so 4 rows measured 16.80s against 17.59s at
/// 64 on a 240-text run. It mattered a great deal before that, in the opposite
/// direction, which is why the constant exists at all.
const METAL_ATTN_BUDGET: usize = 16 * 512 * 512;

/// Rows allowed in one forward of padded length `seq`.
fn rows_per_forward(seq: usize, backend: Backend) -> usize {
    let budget = match backend {
        Backend::Cpu => ATTN_BUDGET,
        Backend::Metal => METAL_ATTN_BUDGET,
        // CoreML runs its own fixed-shape, batch=1 path (see embed_coreml) and
        // never reaches the candle memory-budget splitter.
        Backend::CoreML => unreachable!("CoreML does not use the candle attention budget"),
    };
    (budget / (seq * seq).max(1)).max(1)
}

/// Pooled vectors from one forward, tagged with their caller-side row index.
type PooledRows = Vec<(usize, Vec<f32>)>;

/// One forward pass: rows `start .. start + rows` of `batch`. A bucketed
/// batch is split into as many of these as the memory budget requires.
struct Unit<'a> {
    batch: &'a BatchInput,
    start: usize,
    rows: usize,
}

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
    /// A directory of pre-converted CoreML models for [`Backend::CoreML`]:
    /// one `seq-<N>.mlmodelc` per bucket length, plus `tokenizer.json` and
    /// `config.json`. Only valid with `--device coreml`.
    CoreMl { dir: PathBuf },
}

/// Numeric precision of the forward pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Precision {
    /// Full f32 — matches the PyTorch reference, works everywhere.
    #[default]
    F32,
    /// The four projection `Linear`s in bf16 (see [`crate::bf16`]). Measured
    /// on an 8-core Zen 4: 1.9× faster on short texts, 1.5× on 512-token
    /// ones, at cosine ≈ 0.99999 against f32 — and it halves the memory the
    /// weights occupy. Requires x86_64 with AVX512-BF16 (Zen 4, Sapphire
    /// Rapids or newer); [`Embedder::load`] fails clearly elsewhere.
    Bf16,
}

/// Which device runs the forward pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Backend {
    /// CPU, via Apple Accelerate on macOS and candle's own gemm elsewhere.
    #[default]
    Cpu,
    /// Apple GPU via candle's Metal backend. Requires the `metal` cargo
    /// feature; [`Embedder::load`] fails clearly when it is absent.
    ///
    /// About 1.2x faster than Accelerate on an M2 at 512 tokens, with f32
    /// output unchanged. Still not the default: the margin depends on the
    /// patched candle in `vendor/`, so a build against stock candle would be
    /// markedly slower here than on the CPU. The two also use opposite
    /// execution strategies (see [`Embedder::embed`]), so this is a fork of
    /// the pipeline rather than a drop-in swap.
    Metal,
    /// Apple Neural Engine via CoreML. Requires the `coreml` cargo feature and
    /// a [`ModelSource::CoreMl`] directory of pre-converted fixed-shape models.
    /// Runs batch=1 per bucket length; unsupported requests fail fast with
    /// [`UnsupportedRequest`] rather than falling back. See [`crate::coreml`].
    CoreML,
}

/// A request the CoreML backend cannot serve — built without the `coreml`
/// feature, wrong model source, or a `--max-seq-length` beyond the largest
/// converted bucket. Carried as its own type so the CLI can map it to a
/// dedicated exit code (3) instead of the generic fatal (1), letting callers
/// distinguish "retry on --device cpu" from a real failure. There is no
/// automatic fallback: backend choice is the caller's job.
#[derive(Debug)]
pub struct UnsupportedRequest(pub String);

impl UnsupportedRequest {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl std::fmt::Display for UnsupportedRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for UnsupportedRequest {}

/// Knobs for [`Embedder::load`]. `Default` matches Ruri v3.
#[derive(Clone, Copy)]
pub struct Options {
    pub pooling: Pooling,
    /// L2-normalize each embedding (unit length, so dot = cosine).
    pub normalize: bool,
    /// Token-level truncation length. Ruri v3 accepts up to 8192 but was
    /// trained for retrieval at ~512; longer costs seq^2 attention compute.
    pub max_seq_length: usize,
    /// Bucketing granularity (rows per padded batch before the memory cap).
    pub batch_size: usize,
    pub precision: Precision,
    pub backend: Backend,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            pooling: Pooling::Mean,
            normalize: true,
            max_seq_length: 512,
            batch_size: 64,
            precision: Precision::F32,
            backend: Backend::Cpu,
        }
    }
}

/// The loaded weights, in whichever precision was requested.
enum Weights {
    F32(Arc<ModernBert>),
    #[cfg(target_arch = "x86_64")]
    Bf16(Arc<crate::bf16::Bf16ModernBert>),
}

/// The loaded forward-pass engine. The candle path (CPU/Metal) and the CoreML
/// path use opposite execution strategies, so they are separate arms rather
/// than a shared abstraction.
enum Engine {
    Candle {
        weights: Weights,
        device: Device,
    },
    #[cfg(feature = "coreml")]
    CoreMl(crate::coreml::CoreMlEncoder),
}

/// A loaded ModernBERT sentence encoder. Cheap to share by reference; one
/// instance can serve any number of `embed` calls.
pub struct Embedder {
    engine: Engine,
    tokenizer: Tokenizer,
    opts: Options,
    dim: usize,
}

impl Embedder {
    pub fn load(source: &ModelSource, opts: Options) -> Result<Self> {
        if opts.backend == Backend::CoreML {
            return Self::load_coreml(source, opts);
        }
        // The candle path serves the Hub/Files sources; a CoreMl directory has
        // no safetensors to load.
        let (model_path, tokenizer_path) = match source {
            ModelSource::Files { model, tokenizer } => (model.clone(), tokenizer.clone()),
            ModelSource::Hub { repo } => fetch_from_hub(repo)?,
            ModelSource::CoreMl { .. } => {
                return Err(UnsupportedRequest::new(
                    "a CoreML model directory needs `--device coreml`",
                )
                .into())
            }
        };

        let config_path = model_path
            .parent()
            .map(|d| d.join("config.json"))
            .context("model path has no parent dir for config.json")?;
        let config: Config = read_config(&config_path)?;
        let dim = config.hidden_size;

        // The bf16 path is a hand-written CPU GEMM (see `crate::bf16`), so it
        // has nothing to run on a GPU.
        anyhow::ensure!(
            !(opts.backend == Backend::Metal && opts.precision == Precision::Bf16),
            "bf16 is a CPU-only fast path and cannot run on Metal; pick one"
        );

        let device = open_device(opts.backend)?;
        let weights = load_weights(&model_path, &config, &device, opts.precision)?;
        let tokenizer = load_tokenizer(&tokenizer_path, opts.max_seq_length)?;
        Ok(Self {
            engine: Engine::Candle { weights, device },
            tokenizer,
            opts,
            dim,
        })
    }

    /// Load the CoreML/ANE backend from a directory of fixed-shape models.
    /// Every unsupported condition is caught here, before any input is read.
    #[cfg(feature = "coreml")]
    fn load_coreml(source: &ModelSource, opts: Options) -> Result<Self> {
        let dir = match source {
            ModelSource::CoreMl { dir } => dir,
            _ => {
                return Err(UnsupportedRequest::new(
                    "`--device coreml` needs a CoreML model directory (`--coreml-dir`)",
                )
                .into())
            }
        };

        let config: Config = read_config(&dir.join("config.json"))?;
        let dim = config.hidden_size;
        let encoder = crate::coreml::CoreMlEncoder::load(dir, dim)?;

        // The ANE only has the bucket lengths that were converted. Every input
        // is truncated to max_seq_length, so if that fits the largest bucket
        // no individual row can overflow — one check covers the whole run.
        if opts.max_seq_length > encoder.max_bucket() {
            return Err(UnsupportedRequest::new(format!(
                "--max-seq-length {} exceeds the largest converted CoreML bucket ({}); \
                 lower it or convert a longer model",
                opts.max_seq_length,
                encoder.max_bucket()
            ))
            .into());
        }

        let tokenizer = load_tokenizer(&dir.join("tokenizer.json"), opts.max_seq_length)?;
        Ok(Self {
            engine: Engine::CoreMl(encoder),
            tokenizer,
            opts,
            dim,
        })
    }

    #[cfg(not(feature = "coreml"))]
    fn load_coreml(_source: &ModelSource, _opts: Options) -> Result<Self> {
        Err(UnsupportedRequest::new(
            "this binary was built without CoreML support; rebuild with \
             `cargo build --release --features coreml`",
        )
        .into())
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
        match &self.engine {
            Engine::Candle { weights, device } => self.embed_candle(texts, weights, device),
            #[cfg(feature = "coreml")]
            Engine::CoreMl(encoder) => self.embed_coreml(texts, encoder),
        }
    }

    /// The candle (CPU/Metal) path: length-bucketed, padded batches split to a
    /// memory budget and fanned across a thread pool (CPU) or run wide back to
    /// back (Metal).
    fn embed_candle(
        &self,
        texts: &[&str],
        weights: &Weights,
        device: &Device,
    ) -> Result<Vec<Vec<f32>>> {
        let batches = tokenize_bucket(&self.tokenizer, texts, self.opts.batch_size)?;

        // Split each bucketed batch into forwards that fit the memory budget.
        let limit = weights.max_rows_per_forward();
        let mut units: Vec<Unit> = Vec::new();
        for batch in &batches {
            let cap = rows_per_forward(batch.seq, self.opts.backend).min(limit);
            let mut start = 0;
            while start < batch.batch {
                let rows = cap.min(batch.batch - start);
                units.push(Unit { batch, start, rows });
                start += rows;
            }
        }

        let pooling = self.opts.pooling;
        let run = |unit: &Unit| -> Result<PooledRows> {
            let (batch, seq) = (unit.batch, unit.batch.seq);
            // This unit's slice of the batch's `[batch, seq]` layout.
            let range = unit.start * seq..(unit.start + unit.rows) * seq;
            let ids = &batch.ids[range.clone()];
            let mask = &batch.mask[range];
            let (hidden, dim) = weights.forward(device, ids, mask, unit.rows, seq)?;

            let mut pooled = Vec::with_capacity(unit.rows);
            for row in 0..unit.rows {
                let vector = pool_row(
                    &hidden[row * seq * dim..(row + 1) * seq * dim],
                    &mask[row * seq..(row + 1) * seq],
                    dim,
                    pooling,
                );
                pooled.push((batch.orig[unit.start + row], vector));
            }
            Ok(pooled)
        };

        // The two backends want opposite shapes. On the CPU, parallelism comes
        // from running many narrow forwards at once. There is only one GPU, so
        // fanning out just makes threads contend over command submission and
        // multiplies scratch memory; Metal runs wide forwards back to back
        // instead, and gets its parallelism inside each one.
        let per_unit: Vec<Result<PooledRows>> = match self.opts.backend {
            Backend::Metal => units.iter().map(run).collect(),
            // CoreML never reaches embed_candle; treat anything non-Metal as
            // the CPU fan-out.
            _ => worker_pool()?.install(|| units.par_iter().map(run).collect()),
        };

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

    /// The CoreML/ANE path: one fixed-shape, batch=1 forward per text, routed
    /// to the smallest bucket that fits. Serial by design — the ANE is a single
    /// shared engine, so a thread pool would only add contention.
    #[cfg(feature = "coreml")]
    fn embed_coreml(
        &self,
        texts: &[&str],
        encoder: &crate::coreml::CoreMlEncoder,
    ) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let dim = encoder.dim();
        let pooling = self.opts.pooling;

        let mut rows_out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for enc in &encodings {
            let ids = enc.get_ids();
            let seq = encoder.bucket_for(ids.len()).ok_or_else(|| {
                // Unreachable given the load-time max_seq_length check, but
                // never silently truncate past what the model can do.
                UnsupportedRequest::new(format!(
                    "{} tokens exceed the largest CoreML bucket ({})",
                    ids.len(),
                    encoder.max_bucket()
                ))
            })?;

            // Pad this row to the exact bucket length; zeros stay masked out.
            let mut ids_pad = vec![0i64; seq];
            let mut mask_pad = vec![0i64; seq];
            for (t, (&id, &m)) in ids.iter().zip(enc.get_attention_mask()).enumerate() {
                ids_pad[t] = id as i64;
                mask_pad[t] = m as i64;
            }

            let hidden = encoder.forward(&ids_pad, &mask_pad, seq)?;
            let mut vector = pool_row(&hidden, &mask_pad, dim, pooling);
            if self.opts.normalize {
                l2_normalize(&mut vector);
            }
            rows_out.push(vector);
        }
        Ok(rows_out)
    }
}

/// Open the requested device, failing with a fixable message rather than a
/// silent fallback — a run that quietly lands on the CPU looks like a Metal
/// benchmark result.
fn open_device(backend: Backend) -> Result<Device> {
    match backend {
        Backend::Cpu => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        Backend::Metal => {
            Device::new_metal(0).context("cannot open Metal device 0 (no Apple GPU available?)")
        }
        #[cfg(not(feature = "metal"))]
        Backend::Metal => anyhow::bail!(
            "this binary was built without Metal support; rebuild with \
             `cargo build --release --features metal`"
        ),
        // CoreML is routed to its own loader before open_device is reached.
        Backend::CoreML => unreachable!("CoreML backend does not use a candle Device"),
    }
}

/// Read and parse a `config.json`.
fn read_config(path: &Path) -> Result<Config> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
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

fn load_weights(
    path: &Path,
    config: &Config,
    device: &Device,
    precision: Precision,
) -> Result<Weights> {
    // Two views of the same memory-mapped file, because the two loaders ask
    // for different names. candle's `ModernBert::load` prefixes every weight
    // with `model.` — right for checkpoints saved from a wrapper class (MLM,
    // classification), wrong for the bare sentence encoders we target (ruri,
    // modernbert-embed), which store `embeddings.*`, `layers.*` and
    // `final_norm.*` at the root. Our own bf16 loader reads them at the root.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    let (wrapped, encoder) = if vb.contains_tensor("model.embeddings.tok_embeddings.weight") {
        (vb.clone(), vb.pp("model"))
    } else {
        let strip = vb
            .clone()
            .rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_string());
        (strip, vb)
    };

    match precision {
        Precision::F32 => {
            let model = ModernBert::load(wrapped, config).context("loading ModernBERT weights")?;
            Ok(Weights::F32(Arc::new(model)))
        }
        Precision::Bf16 => {
            #[cfg(target_arch = "x86_64")]
            {
                anyhow::ensure!(
                    crate::bf16::supported(),
                    "bf16 needs an x86_64 CPU with AVX512-BF16 (Zen 4, Sapphire Rapids or newer); \
                     this CPU lacks it — use the default f32 precision"
                );
                let model = crate::bf16::Bf16ModernBert::load(encoder, config)
                    .context("loading ModernBERT weights for the bf16 path")?;
                Ok(Weights::Bf16(Arc::new(model)))
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let _ = encoder;
                anyhow::bail!(
                    "bf16 is an x86_64-only fast path (AVX512-BF16); use the default f32 precision"
                );
            }
        }
    }
}

impl Weights {
    /// Run one forward pass, returning flat `[batch * seq * dim]` hidden
    /// states and the dimension.
    fn forward(
        &self,
        device: &Device,
        ids: &[i64],
        mask: &[i64],
        batch: usize,
        seq: usize,
    ) -> Result<(Vec<f32>, usize)> {
        match self {
            Self::F32(model) => {
                let ids_u: Vec<u32> = ids.iter().map(|&v| v as u32).collect();
                let mask_u: Vec<u32> = mask.iter().map(|&v| v as u32).collect();
                let xs = Tensor::from_vec(ids_u, (batch, seq), device)?;
                let m = Tensor::from_vec(mask_u, (batch, seq), device)?;
                let out = model.forward(&xs, &m)?; // [batch, seq, dim]
                let dim = out.dim(2)?;
                let data = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                Ok((data, dim))
            }
            #[cfg(target_arch = "x86_64")]
            Self::Bf16(model) => model.forward_batch(ids, mask, batch, seq),
        }
    }

    /// Upper bound on rows per forward, on top of the memory budget.
    ///
    /// The bf16 GEMM is single-threaded by design, so all parallelism comes
    /// from having many forwards in flight; coarse ones load-balance badly
    /// because the last wave leaves cores idle, and they waste more padding.
    /// Measured on 1200 short texts, 8-core Zen 4: 4 rows 5.3s, 8 → 5.6s,
    /// 16 → 6.3s, 64 → 8.4s (2 rows is 5.5s — past the sweet spot, per-call
    /// overhead starts to show). The budget already caps long inputs below
    /// this, so it only bites on short ones.
    ///
    /// The f32 path needs no such limit: candle's gemm is internally
    /// efficient on wider batches.
    fn max_rows_per_forward(&self) -> usize {
        match self {
            Self::F32(_) => usize::MAX,
            #[cfg(target_arch = "x86_64")]
            Self::Bf16(_) => 4,
        }
    }
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
