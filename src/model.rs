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

use crate::batch::{l2_normalize, load_tokenizer, pool_row, BatchInput, Pooling, TokenInfo};
use crate::config::CoreMlForm;
use crate::errors::UnsupportedRequest;

/// Attention-scratch budget per forward, in `rows * seq^2` elements.
///
/// It bounds memory, but it also lands on the fastest setting: at 512 tokens
/// it allows 2 rows, and 240 512-token texts encode in 20.0s there against
/// 22.4s at 1 row, 25.3s at 4 and 27.5s at 8 (bf16, median of five
/// interleaved runs). Wider forwards spill the score matrix out of cache
/// without buying any parallelism, since the rows already run in parallel.
const ATTN_BUDGET: usize = 2 * 512 * 512;

/// Same budget for a GPU, which runs one stream of wide forwards rather than
/// many narrow ones.
///
/// The width barely matters now: with the vendored candle's SDPA the attention
/// scores are never materialized, so 4 rows measured 16.80s against 17.59s at
/// 64 on a 240-text run. It mattered a great deal before that, in the opposite
/// direction, which is why the constant exists at all.
const GPU_ATTN_BUDGET: usize = 16 * 512 * 512;

/// Rows allowed in one forward of padded length `seq`.
fn rows_per_forward(seq: usize, backend: Backend) -> usize {
    let budget = match backend {
        Backend::Cpu => ATTN_BUDGET,
        Backend::Metal | Backend::Cuda => GPU_ATTN_BUDGET,
        // CoreML runs its own fixed-shape, batch=1 path (see embed_coreml) and
        // never reaches the candle memory-budget splitter.
        Backend::CoreML => unreachable!("CoreML does not use the candle attention budget"),
    };
    (budget / (seq * seq).max(1)).max(1)
}

/// One forward pass: rows `start .. start + rows` of `batch`. A bucketed
/// batch is split into as many of these as the memory budget requires.
struct Unit<'a> {
    batch: &'a BatchInput,
    start: usize,
    rows: usize,
}

/// Run bucketed batches through the encoder and reduce each row's hidden
/// states to one result, in the caller's original order.
///
/// The shared middle of every candle-backed task: split each batch into
/// forwards that fit the memory budget, run them the way this backend wants,
/// and put the rows back where they came from. What varies is only the last
/// step — mean-pool and normalize for an embedding, take the CLS token and
/// push it through a classifier head for a reranker — so that is the closure.
///
/// `reduce` receives one row's `[seq, dim]` hidden states, that row's mask,
/// and `dim`. It runs on a worker thread, so it must not assume an order.
pub(crate) fn run_batches<T: Send>(
    weights: &Weights,
    device: &Device,
    backend: Backend,
    batches: &[BatchInput],
    rows_total: usize,
    reduce: impl Fn(&[f32], &[i64], usize) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    // Split each bucketed batch into forwards that fit the memory budget.
    let limit = weights.max_rows_per_forward();
    let mut units: Vec<Unit> = Vec::new();
    for batch in batches {
        let cap = rows_per_forward(batch.seq, backend).min(limit);
        let mut start = 0;
        while start < batch.batch {
            let rows = cap.min(batch.batch - start);
            units.push(Unit { batch, start, rows });
            start += rows;
        }
    }

    let run = |unit: &Unit| -> Result<Vec<(usize, T)>> {
        let (batch, seq) = (unit.batch, unit.batch.seq);
        // This unit's slice of the batch's `[batch, seq]` layout.
        let range = unit.start * seq..(unit.start + unit.rows) * seq;
        let ids = &batch.ids[range.clone()];
        let mask = &batch.mask[range];
        let (hidden, dim) = weights.forward(device, ids, mask, unit.rows, seq)?;

        let mut done = Vec::with_capacity(unit.rows);
        for row in 0..unit.rows {
            let reduced = reduce(
                &hidden[row * seq * dim..(row + 1) * seq * dim],
                &mask[row * seq..(row + 1) * seq],
                dim,
            )?;
            done.push((batch.orig[unit.start + row], reduced));
        }
        Ok(done)
    };

    // The two backends want opposite shapes. On the CPU, parallelism comes
    // from running many narrow forwards at once. There is only one GPU, so
    // fanning out just makes threads contend over command submission and
    // multiplies scratch memory; a GPU runs wide forwards back to back
    // instead, and gets its parallelism inside each one.
    let per_unit: Vec<Result<Vec<(usize, T)>>> = match backend {
        Backend::Cpu => worker_pool()?.install(|| units.par_iter().map(run).collect()),
        Backend::Metal | Backend::Cuda => units.iter().map(run).collect(),
        // The CoreML backend runs its own fixed-shape path and never arrives here.
        Backend::CoreML => unreachable!("CoreML does not use the candle batch runner"),
    };

    let mut out: Vec<Option<T>> = (0..rows_total).map(|_| None).collect();
    for unit in per_unit {
        for (orig, value) in unit? {
            out[orig] = Some(value);
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(i, v)| v.with_context(|| format!("row {i} came back from no batch")))
        .collect()
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
    /// one `seq-<N>.mlpackage` per bucket length, plus `tokenizer.json` and
    /// `config.json`. Only valid with `--device coreml`.
    CoreMl { dir: PathBuf },
    /// A Hugging Face Hub repo holding the same CoreML layout (the
    /// `seq-<N>.mlpackage` buckets plus `tokenizer.json` / `config.json`),
    /// downloaded into the standard HF cache. Only valid with `--device
    /// coreml`.
    CoreMlHub { repo: String },
    /// A plain checkpoint to convert for [`Backend::CoreML`] on first use, caching
    /// the bundle so later runs load it directly. What `--device coreml` does when
    /// given neither `--coreml-dir` nor `--coreml-model-id`.
    ///
    /// `checkpoint` is the [`Self::Hub`] or [`Self::Files`] source the CPU path
    /// would have taken, so the same `--model-id` serves both devices. Needs a
    /// build with the `coreml-export` feature.
    CoreMlConvert {
        checkpoint: Box<ModelSource>,
        /// Fixed sequence lengths to emit, one CoreML function each.
        buckets: Vec<usize>,
        quantize: crate::CoreMlQuantize,
    },
}

/// Numeric precision of the forward pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Precision {
    /// Full f32 — matches the PyTorch reference, works everywhere.
    #[default]
    F32,
    /// The four projection `Linear`s in bf16, plus the vectorized softmax,
    /// GeGLU and sliding-window attention that come with them (the `bf16`
    /// module, which exists on x86_64 only). Measured on an 8-core Zen 4: 2.3× faster on short
    /// texts, 2.0× on 512-token ones, at cosine ≈ 0.99999 against f32 — and it
    /// halves the memory the weights occupy. Requires x86_64 with AVX512-BF16
    /// (Zen 4, Sapphire Rapids or newer); [`Embedder::load`] fails clearly
    /// elsewhere.
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
    /// NVIDIA GPU via Candle's CUDA backend. Requires the `cuda` cargo feature
    /// and an NVIDIA driver with a compatible CUDA runtime.
    Cuda,
    /// Apple Neural Engine via CoreML. Requires the `coreml` cargo feature and a
    /// converted model: a [`ModelSource::CoreMl`] directory, a
    /// [`ModelSource::CoreMlHub`] repo, or a [`ModelSource::CoreMlConvert`]
    /// checkpoint this converts itself.
    /// Runs batch=1 per bucket length; unsupported requests fail fast with
    /// [`UnsupportedRequest`] rather than falling back.
    CoreML,
}

impl Backend {
    /// The name `--device` takes, so a report can be pasted back as a flag.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::CoreML => "coreml",
        }
    }
}

impl Precision {
    /// The name `--precision` takes.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }
}

/// What a loaded model is, in the terms a results file should record.
///
/// The point of it is [`Self::sha256`]: everything else here can be inferred
/// from the command line, but which weights actually answered cannot. Written
/// by `--print-model-info` as one JSON line, and abbreviated into the stderr
/// summary of every run.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelInfo {
    /// `--device`, as its flag value.
    pub backend: &'static str,
    /// `--precision`, as its flag value.
    pub precision: &'static str,
    /// sha256 of the `model.safetensors` these weights were loaded from.
    /// Absent on the CoreML path, which loads a converted bundle instead and
    /// reports [`Self::source_sha256`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// The checkpoint a CoreML bundle was converted from, as its converter
    /// recorded it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// sha256 of that checkpoint's weights. `None` for a bundle converted
    /// before Kohagi recorded it — an unknown provenance says so rather than
    /// borrowing the bundle's own identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_sha256: Option<String>,
    /// The fixed sequence lengths a CoreML bundle serves.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buckets: Option<Vec<usize>>,
    /// `embeddings-int8` / `all-int8` for a quantized CoreML bundle, `none`
    /// for an fp16 one. A quantized bundle's vectors are not interchangeable
    /// with an fp16 one's, so the number a run produced needs it recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// The emitted graph's version (`GRAPH_VERSION`), for a CoreML bundle that
    /// records one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_version: Option<String>,
    /// The pooling that was resolved at load — the checkpoint's own choice
    /// unless `--pooling` overrode it.
    pub pooling: &'static str,
    /// Output dimension, the model's `hidden_size`.
    pub dim: usize,
    pub max_seq_length: usize,
    /// `sigmoid` or `logit` for a reranker: which of the two a score is. Absent
    /// for an embedding model, which has no score to shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<&'static str>,
}

impl ModelInfo {
    /// Fill in what only a converted bundle can say about itself.
    ///
    /// Shared by the embedder and the reranker because it is one claim either
    /// way: a bundle's provenance is a property of the bundle, not of what is
    /// being asked of it.
    #[cfg(feature = "coreml")]
    pub(crate) fn add_bundle(&mut self, encoder: &crate::coreml::CoreMlEncoder) {
        let p = encoder.provenance();
        self.source = p.source;
        self.source_sha256 = p.source_sha256;
        self.graph_version = p.graph_version;
        // Every bundle serves the lengths it was compiled for, whether or not
        // its metadata says so, so this reads the loaded models.
        self.buckets = Some(encoder.buckets());
        // An fp16 bundle carries no quantization key; saying "none" is what
        // makes the two cases distinguishable in a results file.
        self.quantization = Some(p.quantization.unwrap_or_else(|| "none".to_string()));
    }
}

/// Reject a precision the requested device cannot run.
///
/// The bf16 path is a hand-written CPU GEMM (see `crate::bf16`), so it has
/// nothing to run on a GPU. Checked before the device is opened, and in one
/// place so that the embedder and the reranker cannot drift on which
/// combinations they accept.
pub(crate) fn check_precision(backend: Backend, precision: Precision) -> Result<()> {
    anyhow::ensure!(
        !matches!(backend, Backend::Metal | Backend::Cuda) || precision != Precision::Bf16,
        "bf16 is a CPU-only fast path and cannot run on a GPU; pick f32"
    );
    Ok(())
}

/// Knobs for [`Embedder::load`]. `Default` matches Ruri v3.
#[derive(Clone, Copy)]
pub struct Options {
    /// `None` (the default) takes the pooling from the checkpoint's
    /// `1_Pooling/config.json`, falling back to mean with a warning when the
    /// model publishes none. `Some(p)` forces `p`, warning if it disagrees
    /// with what the checkpoint declares.
    pub pooling: Option<Pooling>,
    /// L2-normalize each embedding (unit length, so dot = cosine).
    pub normalize: bool,
    /// Token-level truncation length. Ruri v3 accepts up to 8192 but was
    /// trained for retrieval at ~512; longer costs seq^2 attention compute.
    pub max_seq_length: usize,
    /// Bucketing granularity (rows per padded batch before the memory cap).
    pub batch_size: usize,
    pub precision: Precision,
    pub backend: Backend,
    /// Which form to download from a CoreML Hub repo (see [`CoreMlForm`]).
    pub coreml_form: CoreMlForm,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            pooling: None,
            normalize: true,
            max_seq_length: 512,
            batch_size: 64,
            precision: Precision::F32,
            backend: Backend::Cpu,
            coreml_form: CoreMlForm::Compiled,
        }
    }
}

/// The loaded weights, in whichever precision was requested.
pub(crate) enum Weights {
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
    /// Resolved from `opts.pooling` and the checkpoint's `1_Pooling` at load
    /// time (see [`resolve_pooling`]), so `embed` never re-decides it.
    pooling: Pooling,
    dim: usize,
    /// sha256 of the weights file this was loaded from, started at load and
    /// collected when something asks. `None` on the CoreML path, whose
    /// provenance is read from the bundle instead (see [`Embedder::info`]).
    fingerprint: Option<crate::fingerprint::Fingerprint>,
}

impl Embedder {
    pub fn load(source: &ModelSource, opts: Options) -> Result<Self> {
        if opts.backend == Backend::CoreML {
            return Self::load_coreml(source, opts);
        }
        // The candle path serves the Hub/Files sources; a CoreMl directory has
        // no safetensors to load.
        let (model_path, tokenizer_path, detected_pooling) = match source {
            ModelSource::Files { model, tokenizer } => {
                (model.clone(), tokenizer.clone(), local_pooling(model))
            }
            ModelSource::Hub { repo } => {
                let f = fetch_checkpoint(repo)?;
                (f.weights, f.tokenizer, f.pooling)
            }
            ModelSource::CoreMl { .. }
            | ModelSource::CoreMlHub { .. }
            | ModelSource::CoreMlConvert { .. } => {
                return Err(UnsupportedRequest::new(
                    "a CoreML model source needs `--device coreml`",
                )
                .into())
            }
        };

        let pooling = resolve_pooling_warned(opts.pooling, detected_pooling);

        let config_path = model_path
            .parent()
            .map(|d| d.join("config.json"))
            .context("model path has no parent dir for config.json")?;
        let config: Config = read_config(&config_path)?;
        let dim = config.hidden_size;

        check_precision(opts.backend, opts.precision)?;

        let device = open_device(opts.backend)?;
        // Started here and collected at the end of the run: the weights are
        // about to be memory-mapped, so this reads the same file the forward
        // pass will fault in, and doing it on another thread keeps half a
        // gigabyte of hashing out of the caller's first result.
        let fingerprint = crate::fingerprint::Fingerprint::spawn(model_path.clone());
        let weights = load_weights(&model_path, &config, &device, opts.precision)?;
        let tokenizer = load_tokenizer(&tokenizer_path, opts.max_seq_length)?;
        Ok(Self {
            engine: Engine::Candle { weights, device },
            tokenizer,
            opts,
            pooling,
            dim,
            fingerprint: Some(fingerprint),
        })
    }

    /// Load the CoreML/ANE backend from a directory of fixed-shape models.
    /// Every unsupported condition is caught here, before any input is read.
    #[cfg(feature = "coreml")]
    fn load_coreml(source: &ModelSource, opts: Options) -> Result<Self> {
        // `converted` says whether this run did the conversion, which is when the
        // self-check below is worth its few seconds.
        let (dir, converted) = match source {
            ModelSource::CoreMl { dir } => (dir.clone(), false),
            ModelSource::CoreMlHub { repo } => (
                crate::coreml::fetch_from_hub(repo, opts.coreml_form)?,
                false,
            ),
            ModelSource::CoreMlConvert {
                checkpoint,
                buckets,
                quantize,
            } => convert_for_coreml(checkpoint, buckets, *quantize)?,
            _ => {
                return Err(UnsupportedRequest::new(
                    "`--device coreml` needs a CoreML model directory (`--coreml-dir`) \
                     or Hub repo (`--coreml-model-id`)",
                )
                .into())
            }
        };

        let config: Config = read_config(&dir.join("config.json"))?;
        let dim = config.hidden_size;
        let encoder = crate::coreml::CoreMlEncoder::load(&dir, dim)?;

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

        let pooling = resolve_pooling_warned(opts.pooling, pooling_in_dir(&dir));

        let embedder = Self {
            engine: Engine::CoreMl(encoder),
            tokenizer,
            opts,
            pooling,
            dim,
            // A bundle has no safetensors to hash; what it can say about the
            // checkpoint behind it is in its own metadata, read by `info`.
            fingerprint: None,
        };
        #[cfg(feature = "coreml-export")]
        if converted {
            if let ModelSource::CoreMlConvert { checkpoint, .. } = source {
                embedder.self_check(checkpoint);
            }
        }
        // Only a conversion sets it, and only this feature can convert.
        #[cfg(not(feature = "coreml-export"))]
        let _ = converted;
        Ok(embedder)
    }

    /// Compare a few sentences against the checkpoint's own f32 forward, once,
    /// right after converting it.
    ///
    /// The emitter itself is verified against the Python conversion, but what no
    /// converter can know in advance is whether a *checkpoint* is sensitive to fp16:
    /// `nomic-ai/modernbert-embed-base` drifts far past fp16 rounding under both
    /// this converter and coremltools. That is a property worth knowing and not worth failing on,
    /// so this warns and continues. Any error here is ignored for the same reason —
    /// a check that cannot run must not stop a working model from loading.
    #[cfg(all(feature = "coreml", feature = "coreml-export"))]
    fn self_check(&self, checkpoint: &ModelSource) {
        /// Mixed scripts, so the comparison covers more of the embedding table
        /// than one language would, and one long passage — fp16 error accumulates
        /// with sequence length, and short probes miss it. On
        /// `nomic-ai/modernbert-embed-base` the short ones diverge by 5e-5, under
        /// the threshold, and adding the long one takes the worst to 7e-3.
        const PROBES: [&str; 4] = [
            "瑠璃も玻璃も照らせば光る",
            "Local inference keeps data on the device.",
            "近くの喫茶店で二時間ほど本を読んでいた。",
            "Running a sentence encoder on the neural engine requires fixed shapes, one \
             model per sequence length, and padding every row to that exact length, which \
             is why the converter emits one function per bucket and shares a single copy \
             of the weights between them; the embedding table dominates the bytes for a \
             large vocabulary, so quantizing it to eight bits with one scale per row \
             halves that part of the file while leaving retrieval quality unchanged on \
             both benchmarks that were measured.",
        ];

        // Every failure here is reported and then dropped: a check that cannot run
        // must not stop a working model from loading, but a silent skip would leave
        // no way to tell "checked and fine" from "never checked".
        let skip = |e: &anyhow::Error| {
            eprintln!("kohagi: could not compare the converted model against float32 ({e:#})");
        };
        let reference = match Self::load(
            checkpoint,
            Options {
                backend: Backend::Cpu,
                ..self.opts
            },
        ) {
            Ok(r) => r,
            Err(e) => return skip(&e),
        };
        let (theirs, ours) = match (reference.embed(&PROBES), self.embed(&PROBES)) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => return skip(&e),
        };
        let worst = theirs
            .iter()
            .zip(&ours)
            .map(|(a, b)| {
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let norm = |v: &Vec<f32>| v.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (norm(a) * norm(b)).max(f32::MIN_POSITIVE)
            })
            .fold(0.0f32, f32::max);

        // fp16 rounding through a 19-layer encoder lands around 1e-5; anything past
        // 1e-4 is the checkpoint's own sensitivity rather than rounding.
        if worst > 1e-4 {
            eprintln!(
                "kohagi: warning: this model's Neural Engine output differs from its \
                 float32 output by 1 - cosine = {worst:.1e}, more than fp16 rounding \
                 explains; the checkpoint is sensitive to fp16, so its ANE vectors are \
                 not interchangeable with its CPU ones"
            );
        }
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

    /// What this model is, for a summary line or a results file. Cheap: every
    /// field was resolved at load, except a CoreML bundle's provenance, which
    /// is a metadata lookup on an already-open model.
    pub fn info(&self) -> ModelInfo {
        // Mutated only under the `coreml` feature, which is the only path with
        // bundle fields to fill in.
        #[allow(unused_mut)]
        let mut info = ModelInfo {
            backend: self.opts.backend.name(),
            precision: self.opts.precision.name(),
            sha256: self.fingerprint.as_ref().and_then(|f| f.get()),
            source: None,
            source_sha256: None,
            buckets: None,
            quantization: None,
            graph_version: None,
            pooling: self.pooling.name(),
            dim: self.dim,
            max_seq_length: self.opts.max_seq_length,
            score: None,
        };
        #[cfg(feature = "coreml")]
        if let Engine::CoreMl(encoder) = &self.engine {
            info.add_bundle(encoder);
        }
        info
    }

    /// Embed a batch of texts, one vector per text, in input order. Prefixes
    /// (e.g. Ruri's `"検索文書: "`) are the caller's job — pass prefixed text.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(self.embed_with_tokens(texts)?.0)
    }

    /// Like [`Embedder::embed`], but also returns a [`TokenInfo`] per text (same
    /// order as the vectors), so a caller can tell which embeddings were built
    /// from truncated input. The vectors are identical to [`Embedder::embed`]'s.
    pub fn embed_with_tokens(&self, texts: &[&str]) -> Result<(Vec<Vec<f32>>, Vec<TokenInfo>)> {
        if texts.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        match &self.engine {
            Engine::Candle { weights, device } => self.embed_candle(texts, weights, device),
            #[cfg(feature = "coreml")]
            Engine::CoreMl(encoder) => self.embed_coreml(texts, encoder),
        }
    }

    /// The candle (CPU/GPU) path: length-bucketed, padded batches split to a
    /// memory budget and fanned across a thread pool (CPU) or run wide back to
    /// back (GPU).
    fn embed_candle(
        &self,
        texts: &[&str],
        weights: &Weights,
        device: &Device,
    ) -> Result<(Vec<Vec<f32>>, Vec<TokenInfo>)> {
        let encodings = crate::batch::encode(&self.tokenizer, texts)?;
        let (batches, info) = crate::batch::bucket_encodings(&encodings, self.opts.batch_size);
        let (pooling, normalize) = (self.pooling, self.opts.normalize);
        let rows_out = run_batches(
            weights,
            device,
            self.opts.backend,
            &batches,
            texts.len(),
            |hidden, mask, dim| Ok(embed_row(hidden, mask, dim, pooling, normalize)),
        )?;
        Ok((rows_out, info))
    }

    /// The CoreML/ANE path: one fixed-shape, batch=1 forward per text, routed
    /// to the smallest bucket that fits.
    #[cfg(feature = "coreml")]
    fn embed_coreml(
        &self,
        texts: &[&str],
        encoder: &crate::coreml::CoreMlEncoder,
    ) -> Result<(Vec<Vec<f32>>, Vec<TokenInfo>)> {
        let encodings = crate::batch::encode(&self.tokenizer, texts)?;
        let info: Vec<TokenInfo> = encodings.iter().map(crate::batch::token_info).collect();
        let (pooling, normalize) = (self.pooling, self.opts.normalize);
        let rows_out = encoder.run_rows(&encodings, |hidden, mask, dim| {
            Ok(embed_row(hidden, mask, dim, pooling, normalize))
        })?;
        Ok((rows_out, info))
    }
}

/// One row's hidden states to its embedding — the step that differs between an
/// embedder and a reranker, and the only one either batch runner does not do.
///
/// A free function rather than a method so the closures handing it to
/// [`run_batches`] capture two `Copy` scalars instead of `&self`: the CPU
/// fan-out needs a `Sync` closure, and an [`Embedder`] may hold a CoreML model
/// that is not shareable between threads.
fn embed_row(
    hidden: &[f32],
    mask: &[i64],
    dim: usize,
    pooling: Pooling,
    normalize: bool,
) -> Vec<f32> {
    let mut vector = pool_row(hidden, mask, dim, pooling);
    if normalize {
        l2_normalize(&mut vector);
    }
    vector
}

/// Open the requested device, failing with a fixable message rather than a
/// silent fallback — a run that quietly lands on the CPU looks like a Metal
/// benchmark result.
pub(crate) fn open_device(backend: Backend) -> Result<Device> {
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
        #[cfg(feature = "cuda")]
        Backend::Cuda => Device::new_cuda(0)
            .context("cannot open CUDA device 0 (is an NVIDIA driver and CUDA runtime installed?)"),
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => anyhow::bail!(
            "this binary was built without CUDA support; rebuild with \
             `cargo build --release --features cuda`"
        ),
        // CoreML is routed to its own loader before open_device is reached.
        Backend::CoreML => unreachable!("CoreML backend does not use a candle Device"),
    }
}

/// What a checkpoint's `1_Pooling/config.json` says its pooling is, or `None`
/// when it publishes neither a cls nor a mean flag (or has no such file).
///
/// Two spellings are in circulation. sentence-transformers 5 writes a single
/// `pooling_mode` string; every earlier version wrote one `pooling_mode_*` bool
/// per mode, which is still what most checkpoints on the Hub carry. Both are
/// read, because a 5.x checkpoint judged by the older rule alone looks like a
/// checkpoint with no pooling config at all — and that reads as mean, which is
/// silently wrong for a cls model.
///
/// Only the two Kohagi supports are read; a checkpoint pooling some other way
/// (max, weighted mean) reads as `None`, the same as no file at all, which is
/// the honest answer since Kohagi cannot reproduce it. A checkpoint combining
/// several modes writes an array there, which is not a string and so takes the
/// same path.
fn pooling_from_st_config(json: &str) -> Option<Pooling> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    if let Some(mode) = v.get("pooling_mode").and_then(|m| m.as_str()) {
        return match mode {
            "cls" => Some(Pooling::Cls),
            "mean" => Some(Pooling::Mean),
            _ => None,
        };
    }
    if v.get("pooling_mode_cls_token")? == &serde_json::Value::Bool(true) {
        Some(Pooling::Cls)
    } else if v.get("pooling_mode_mean_tokens")? == &serde_json::Value::Bool(true) {
        Some(Pooling::Mean)
    } else {
        None
    }
}

/// Something worth telling the user about the pooling choice, from comparing
/// what they asked for against what the checkpoint declares.
#[derive(Debug, PartialEq, Eq)]
enum PoolingNote {
    /// The request disagrees with the checkpoint. The request still wins — the
    /// user may know something — but a `--pooling cls` model run at the mean
    /// default is silently wrong, so it is worth saying.
    Mismatch { used: Pooling, declared: Pooling },
    /// No `1_Pooling/config.json`: a reranker or a base LM rather than a
    /// sentence encoder, most likely. The vectors will still be produced, and
    /// may be near-useless.
    NoConfig { used: Pooling },
}

impl PoolingNote {
    fn message(&self) -> String {
        let name = |p: Pooling| match p {
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
        };
        match self {
            PoolingNote::Mismatch { used, declared } => format!(
                "warning: using --pooling {} but the model declares {} pooling; \
                 embeddings will be wrong if this is not deliberate",
                name(*used),
                name(*declared),
            ),
            PoolingNote::NoConfig { used } => format!(
                "warning: model publishes no 1_Pooling/config.json, so it may not be a \
                 sentence-embedding model (a reranker or base LM produces near-degenerate \
                 vectors); pooling with {}",
                name(*used),
            ),
        }
    }
}

/// Decide the pooling to use, and whether to warn.
///
/// `requested` is `Some` only when the caller passed `--pooling` explicitly;
/// `detected` is what the checkpoint's `1_Pooling` declares. A request always
/// wins so a user can override a mislabeled checkpoint, but the mismatch is
/// surfaced. With no request, the declared pooling is used silently — that is
/// the point, so a cls model just works — and only a missing config warns.
fn resolve_pooling(
    requested: Option<Pooling>,
    detected: Option<Pooling>,
) -> (Pooling, Option<PoolingNote>) {
    match (requested, detected) {
        (Some(used), Some(declared)) if used != declared => {
            (used, Some(PoolingNote::Mismatch { used, declared }))
        }
        (Some(used), _) => (used, None),
        (None, Some(declared)) => (declared, None),
        (None, None) => (
            Pooling::Mean,
            Some(PoolingNote::NoConfig {
                used: Pooling::Mean,
            }),
        ),
    }
}

/// [`resolve_pooling`], with its warning (if any) emitted to stderr. The two
/// load paths share this so the decision and its message live in one place;
/// the pure `resolve_pooling` stays separate for testing.
fn resolve_pooling_warned(requested: Option<Pooling>, detected: Option<Pooling>) -> Pooling {
    let (pooling, note) = resolve_pooling(requested, detected);
    if let Some(note) = note {
        eprintln!("kohagi: {}", note.message());
    }
    pooling
}

/// Read and parse a `config.json`.
pub(crate) fn read_config(path: &Path) -> Result<Config> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

/// Convert a checkpoint for the ANE, or reuse the cached bundle, and return the
/// directory to load from.
///
/// The result has the layout `--coreml-dir` expects, so the caller carries on
/// through the ordinary directory loader — which is what keeps `check_io` and the
/// compile cache in the path.
#[cfg(all(feature = "coreml", feature = "coreml-export"))]
pub(crate) fn convert_for_coreml(
    checkpoint: &ModelSource,
    buckets: &[usize],
    quantize: crate::CoreMlQuantize,
) -> Result<(PathBuf, bool)> {
    use crate::coreml::autoconvert;

    let resolved = match checkpoint {
        ModelSource::Hub { repo } => {
            // The download dominates a first run on a machine that has never used
            // the CPU path, and hf-hub is silent about it, so say so before
            // starting. Nothing is printed when the cache already has the files.
            if !hub_checkpoint_is_cached(repo) {
                eprintln!("kohagi: downloading {repo} (safetensors; first run only) ...");
            }
            let files = fetch_checkpoint(repo)?;
            crate::coreml_export::Checkpoint {
                config: beside(&files.weights, "config.json")?,
                weights: files.weights,
                tokenizer: files.tokenizer,
                pooling: files.pooling_file,
                source: repo.clone(),
            }
        }
        ModelSource::Files { model, tokenizer } => crate::coreml_export::Checkpoint {
            config: beside(model, "config.json")?,
            weights: model.clone(),
            tokenizer: tokenizer.clone(),
            pooling: model
                .parent()
                .map(|d| d.join("1_Pooling").join("config.json"))
                .filter(|p| p.is_file()),
            source: model.display().to_string(),
        },
        // `Args::source` only ever wraps a checkpoint source.
        _ => {
            return Err(UnsupportedRequest::new(
                "`--device coreml` can convert a checkpoint (`--model-id` or \
                 `--model-path`), not another CoreML model",
            )
            .into())
        }
    };
    let provisioned = autoconvert::provision(&resolved, buckets, quantize)?;
    Ok((
        provisioned.path().to_path_buf(),
        matches!(provisioned, autoconvert::Provisioned::Converted(_)),
    ))
}

#[cfg(all(feature = "coreml", not(feature = "coreml-export")))]
pub(crate) fn convert_for_coreml(
    _checkpoint: &ModelSource,
    _buckets: &[usize],
    _quantize: crate::CoreMlQuantize,
) -> Result<(PathBuf, bool)> {
    Err(UnsupportedRequest::new(
        "this binary cannot convert checkpoints for CoreML; pass an already \
         converted model with `--coreml-dir` or `--coreml-model-id`, or rebuild \
         with `--features coreml,coreml-export`",
    )
    .into())
}

/// Whether the Hub cache already holds this checkpoint's weights, so that the
/// "downloading" notice is only printed when there is a download to wait for.
///
/// Asks hf-hub for the file without hitting the network; a miss (or a cache layout
/// this cannot read) errs towards printing the notice, which costs a line rather
/// than a wrong wait.
#[cfg(all(feature = "coreml", feature = "coreml-export"))]
fn hub_checkpoint_is_cached(repo: &str) -> bool {
    hf_hub::Cache::default()
        .model(repo.to_string())
        .get("model.safetensors")
        .is_some()
}

/// A sibling of `path`, for the `config.json` that has to sit beside the weights.
#[cfg(all(feature = "coreml", feature = "coreml-export"))]
fn beside(path: &Path, name: &str) -> Result<PathBuf> {
    path.parent()
        .map(|d| d.join(name))
        .with_context(|| format!("{} has no parent directory for {name}", path.display()))
}

/// What a Hub checkpoint download produced.
pub(crate) struct Fetched {
    pub(crate) weights: PathBuf,
    pub(crate) tokenizer: PathBuf,
    /// `1_Pooling/config.json` in the cache, when the checkpoint ships one. Kept as
    /// a path as well as parsed, because the CoreML converter copies the file into
    /// the bundle it writes.
    #[cfg_attr(
        not(all(feature = "coreml", feature = "coreml-export")),
        allow(dead_code)
    )]
    pooling_file: Option<PathBuf>,
    pooling: Option<Pooling>,
}

/// Download (or reuse from the HF cache) the files a model needs, plus its
/// declared pooling if the checkpoint ships one.
pub(crate) fn fetch_checkpoint(repo: &str) -> Result<Fetched> {
    let api = hf_hub::api::sync::Api::new().context("initializing Hugging Face Hub client")?;
    let repo = api.model(repo.to_string());
    let get = |f: &str| {
        repo.get(f)
            .with_context(|| format!("cannot fetch {f} (network down? try local --model-path)"))
    };
    let weights = get("model.safetensors")?;
    get("config.json")?; // lands next to the weights in the cache
    let tokenizer = get("tokenizer.json")?;
    // Optional: many checkpoints ship it, rerankers and base LMs do not, and a
    // 404 here is information rather than an error.
    let pooling_file = repo.get("1_Pooling/config.json").ok();
    let pooling = pooling_file
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| pooling_from_st_config(&s));
    Ok(Fetched {
        weights,
        tokenizer,
        pooling_file,
        pooling,
    })
}

/// The declared pooling in a checkpoint directory, read from its
/// `1_Pooling/config.json` if present. The one place the `1_Pooling/config.json`
/// path convention lives for on-disk checkpoints (the Hub path fetches it
/// through the API instead).
fn pooling_in_dir(dir: &Path) -> Option<Pooling> {
    let text = std::fs::read_to_string(dir.join("1_Pooling").join("config.json")).ok()?;
    pooling_from_st_config(&text)
}

/// The declared pooling of a local model, read from `1_Pooling/config.json`
/// beside the weights if present.
fn local_pooling(model_path: &Path) -> Option<Pooling> {
    pooling_in_dir(model_path.parent()?)
}

pub(crate) fn load_weights(
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
    pub(crate) fn forward(
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
    /// Too fine and per-call overhead takes over instead. Measured on 1200
    /// short texts, 8-core Zen 4, median of five interleaved runs: 2 rows
    /// 4.64s, 4 → 4.30s, 8 → 4.26s, 16 → 4.55s. 4 and 8 are a tie within the
    /// run-to-run spread; either side of them is not. The budget already caps
    /// long inputs below this, so it only bites on short ones.
    ///
    /// The f32 path needs no such limit: candle's gemm is internally
    /// efficient on wider batches.
    pub(crate) fn max_rows_per_forward(&self) -> usize {
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

#[cfg(test)]
mod pooling_tests {
    use super::*;

    #[test]
    fn gpu_backends_use_the_wide_forward_budget() {
        assert_eq!(
            rows_per_forward(512, Backend::Metal),
            rows_per_forward(512, Backend::Cuda)
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_request_without_feature_explains_how_to_enable_it() {
        let err = open_device(Backend::Cuda).unwrap_err().to_string();
        assert!(err.contains("--features cuda"), "unexpected error: {err}");
    }

    #[test]
    fn reads_cls_and_mean_from_st_config() {
        let cls = r#"{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}"#;
        let mean = r#"{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}"#;
        assert_eq!(pooling_from_st_config(cls), Some(Pooling::Cls));
        assert_eq!(pooling_from_st_config(mean), Some(Pooling::Mean));
    }

    #[test]
    fn reads_cls_and_mean_from_the_5x_st_config() {
        // What sentence-transformers 5 saves, and so what a model fine-tuned
        // today ships: one string in place of the per-mode bools.
        let cls = r#"{"embedding_dimension": 512, "pooling_mode": "cls"}"#;
        let mean = r#"{"embedding_dimension": 512, "pooling_mode": "mean",
                       "include_prompt": true}"#;
        assert_eq!(pooling_from_st_config(cls), Some(Pooling::Cls));
        assert_eq!(pooling_from_st_config(mean), Some(Pooling::Mean));
    }

    #[test]
    fn unsupported_or_absent_pooling_reads_as_none() {
        // A mode Kohagi cannot reproduce, and junk, both decline rather than guess.
        let other = r#"{"pooling_mode_max_tokens": true}"#;
        assert_eq!(pooling_from_st_config(other), None);
        assert_eq!(pooling_from_st_config("not json"), None);
    }

    #[test]
    fn unsupported_or_combined_pooling_reads_as_none_in_the_5x_config() {
        let weighted = r#"{"pooling_mode": "weightedmean"}"#;
        // Several modes at once are concatenated into one vector; the array is
        // not a string, and Kohagi has no way to reproduce the result anyway.
        let combined = r#"{"pooling_mode": ["mean", "cls"]}"#;
        assert_eq!(pooling_from_st_config(weighted), None);
        assert_eq!(pooling_from_st_config(combined), None);
    }

    #[test]
    fn no_request_takes_the_declared_pooling_silently() {
        // The point of the feature: a cls model just works, no flag, no noise.
        assert_eq!(
            resolve_pooling(None, Some(Pooling::Cls)),
            (Pooling::Cls, None)
        );
        assert_eq!(
            resolve_pooling(None, Some(Pooling::Mean)),
            (Pooling::Mean, None)
        );
    }

    #[test]
    fn no_request_and_no_config_defaults_to_mean_and_warns() {
        // The reranker / base-LM case.
        assert_eq!(
            resolve_pooling(None, None),
            (
                Pooling::Mean,
                Some(PoolingNote::NoConfig {
                    used: Pooling::Mean
                })
            ),
        );
    }

    #[test]
    fn a_request_wins_but_a_disagreement_warns() {
        // Forcing mean on a cls model (the gte footgun) is honored but flagged.
        assert_eq!(
            resolve_pooling(Some(Pooling::Mean), Some(Pooling::Cls)),
            (
                Pooling::Mean,
                Some(PoolingNote::Mismatch {
                    used: Pooling::Mean,
                    declared: Pooling::Cls,
                }),
            ),
        );
        // Forcing the pooling the model already declares is silent.
        assert_eq!(
            resolve_pooling(Some(Pooling::Cls), Some(Pooling::Cls)),
            (Pooling::Cls, None),
        );
    }

    #[test]
    fn a_request_with_no_config_is_honored_without_the_reranker_warning() {
        // The user said what they want; do not second-guess with the NoConfig note.
        assert_eq!(
            resolve_pooling(Some(Pooling::Cls), None),
            (Pooling::Cls, None),
        );
    }
}
