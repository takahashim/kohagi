//! Model loading and the memory-bounded parallel encoder.
//!
//! A single candle ModernBERT forward is effectively single-core on CPU, so
//! [`Embedder::embed`] fans length-bucketed batches across a rayon pool — the
//! weights are shared behind an `Arc`, each worker runs an independent
//! forward, and the result is identical to serial execution.
//!
//! Two guardrails keep the attention scratch of one forward under
//! [`ATTN_BUDGET`] scores per head, whatever the caller passes:
//!
//! 1. Rows per forward are capped by [`ATTN_BUDGET`]: a forward holds
//!    ~`rows * heads * seq^2` f32 of scores, so a 64-row batch of seq-512
//!    inputs would hold ~2 GB per worker. 2 rows at seq 512 (~67 MB) measured
//!    both fastest and smallest on an 8-core Zen4 — finer units also
//!    load-balance better across the pool.
//! 2. Past seq 724 one row already exceeds the budget and this cap can go no
//!    lower, so the encoder splits the *queries* instead: `attend_tiled` in
//!    `encoder.rs` divides the same budget by `seq` to size a tile. The two
//!    hand over at exactly that length, and neither regime holds more than
//!    `ATTN_BUDGET` scores per head.
//!
//! What is not flat is everything linear in the input: hidden states, Q/K/V,
//! the MLP activations and the output all grow with `seq * hidden_size`, and
//! the pool is a multiplier on all of it. It defaults to *physical* cores
//! (`RAYON_NUM_THREADS` overrides), since SMT siblings only add contention to
//! these GEMM-bound forwards.

use std::path::Path;
use std::sync::Arc;

use crate::encoder::{Config, ModernBert};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::batch::{
    l2_normalize, load_tokenizer, pool_row, truncate_renormalize, truncation, BatchInput, Pooling,
    TokenInfo,
};
use crate::config::CoreMlForm;
use crate::errors::UnsupportedRequest;
use crate::info::{ModelInfo, Output};
use crate::program::remark;
use crate::source::ModelSource;

/// Attention-scratch budget per forward, in scores per head: `rows * seq^2`
/// while a row still fits, `tile * seq` once one does not (see the module
/// note, and `encoder::query_tile` for the other half).
///
/// It bounds memory, but it also lands on the fastest setting: at 512 tokens
/// it allows 2 rows, and 240 512-token texts encode in 20.0s there against
/// 22.4s at 1 row, 25.3s at 4 and 27.5s at 8 (bf16, median of five
/// interleaved runs). Wider forwards spill the score matrix out of cache
/// without buying any parallelism, since the rows already run in parallel.
pub(crate) const ATTN_BUDGET: usize = 2 * 512 * 512;

/// Same budget for a GPU, which runs one stream of wide forwards rather than
/// many narrow ones.
///
/// The width barely matters on Metal: its SDPA never materializes the attention
/// scores, so 4 rows measured 16.80s against 17.59s at 64 on a 240-text run. It
/// mattered a great deal before that, in the opposite direction, which is why
/// the constant exists at all. CUDA has no such kernel here and runs the tiled
/// path, where this is also the tile size.
pub(crate) const GPU_ATTN_BUDGET: usize = 16 * 512 * 512;

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
    /// Keep only the first N dimensions of each pooled vector and re-normalize
    /// (Matryoshka truncation). `None` outputs the model's full dimension.
    ///
    /// The re-normalization is not optional — it is what keeps the protocol's
    /// "dot = cosine" promise on the truncated vectors — so this refuses to
    /// combine with `normalize: false` at load. A caller wanting raw truncated
    /// vectors can slice the full output itself.
    pub dims: Option<usize>,
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
            dims: None,
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
    /// What the checkpoint declares it can take, for [`Embedder::info`] to
    /// report beside what this run actually took.
    declared_max_seq: Option<usize>,
}

impl Embedder {
    pub fn load(source: &ModelSource, opts: Options) -> Result<Self> {
        // Before any file is touched: this combination is wrong whatever the
        // model turns out to be.
        anyhow::ensure!(
            opts.dims.is_none() || opts.normalize,
            "--dims re-normalizes after truncating, which is what keeps dot = cosine on \
             the shorter vectors; it cannot be combined with --no-normalize. For raw \
             truncation, slice the full --no-normalize output yourself"
        );
        if opts.backend == Backend::CoreML {
            return Self::load_coreml(source, opts);
        }
        // The candle path serves the Hub/Files sources; a CoreMl directory has
        // no safetensors to load.
        let files = source.checkpoint_files()?;

        let pooling = resolve_pooling_warned(opts.pooling, files.pooling);

        let config_path = files
            .weights
            .parent()
            .map(|d| d.join("config.json"))
            .context("model path has no parent dir for config.json")?;
        let config: Config = read_config(&config_path)?;
        let dim = config.hidden_size;
        let opts = Options {
            dims: check_dims(opts.dims, dim)?,
            ..opts
        };
        check_max_seq(opts.max_seq_length, config.max_position_embeddings)?;

        check_precision(opts.backend, opts.precision)?;

        let device = open_device(opts.backend)?;
        // Started here and collected at the end of the run: the weights are
        // about to be memory-mapped, so this reads the same file the forward
        // pass will fault in, and doing it on another thread keeps half a
        // gigabyte of hashing out of the caller's first result.
        let fingerprint = crate::fingerprint::Fingerprint::spawn(&files.weights);
        let weights = load_weights(&files.weights, &config, &device, opts.precision)?;
        fingerprint.confirm(&files.weights);
        let tokenizer = load_tokenizer(&files.tokenizer, opts.max_seq_length)?;
        Ok(Self {
            engine: Engine::Candle { weights, device },
            tokenizer,
            opts,
            pooling,
            dim,
            fingerprint: Some(fingerprint),
            declared_max_seq: files.declared_max_seq,
        })
    }

    /// Load the CoreML/ANE backend from a directory of fixed-shape models.
    /// Every unsupported condition is caught here, before any input is read.
    #[cfg(feature = "coreml")]
    fn load_coreml(source: &ModelSource, opts: Options) -> Result<Self> {
        // `converted` says whether this run did the conversion, which is when the
        // self-check below is worth its few seconds.
        let resolved = source.resolve_coreml(opts.coreml_form)?;
        let dir = resolved.dir;
        let converted = resolved.converted;

        let config: Config = read_config(&dir.join("config.json"))?;
        let dim = config.hidden_size;
        let opts = Options {
            dims: check_dims(opts.dims, dim)?,
            ..opts
        };
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

        let pooling = resolve_pooling_warned(opts.pooling, crate::source::pooling_in_dir(&dir));

        let embedder = Self {
            engine: Engine::CoreMl(encoder),
            tokenizer,
            opts,
            pooling,
            dim,
            // A bundle has no safetensors to hash; what it can say about the
            // checkpoint behind it is in its own metadata, read by `info`.
            fingerprint: None,
            declared_max_seq: crate::source::declared_max_seq_in_dir(&dir),
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
            remark!("could not compare the converted model against float32 ({e:#})");
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
            remark!(
                "warning: this model's Neural Engine output differs from its \
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

    /// The dimension of the vectors this embedder produces: `Options::dims`
    /// when it shortens them, otherwise the model's `hidden_size` (512 for
    /// ruri-v3-130m). `load` already reduced a `dims` equal to the model's own
    /// to `None`, since it changes no vector.
    pub fn dim(&self) -> usize {
        self.opts.dims.unwrap_or(self.dim)
    }

    /// The reduction both engine paths apply to every row, stated once.
    fn reduce(&self) -> Reduce {
        Reduce {
            pooling: self.pooling,
            normalize: self.opts.normalize,
            dims: self.opts.dims,
        }
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
            bundle: None,
            pooling: self.pooling.name(),
            dim: self.dim,
            max_seq_length: self.opts.max_seq_length,
            declared_max_seq_length: self.declared_max_seq,
            output: Output::Embedding {
                output_dim: self.opts.dims,
                normalized: self.opts.normalize,
            },
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
        let reduce = self.reduce();
        let rows_out = run_batches(
            weights,
            device,
            self.opts.backend,
            &batches,
            texts.len(),
            |hidden, mask, dim| Ok(embed_row(hidden, mask, dim, reduce)),
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
        let reduce = self.reduce();
        let rows_out = encoder.run_rows(&encodings, |hidden, mask, dim| {
            Ok(embed_row(hidden, mask, dim, reduce))
        })?;
        Ok((rows_out, info))
    }
}

/// How one row's hidden states become the caller's vector: the pooling, the
/// optional Matryoshka truncation, and whether to normalize.
///
/// One `Copy` value so the closures handing rows to the batch runners capture
/// it instead of `&self` (see [`embed_row`]), and so the two engine paths
/// cannot drift on which settings the reduction takes.
#[derive(Clone, Copy)]
struct Reduce {
    pooling: Pooling,
    normalize: bool,
    dims: Option<usize>,
}

/// One row's hidden states to its embedding — the step that differs between an
/// embedder and a reranker, and the only one either batch runner does not do.
///
/// A free function rather than a method so the closures handing it to
/// [`run_batches`] capture one `Copy` value instead of `&self`: the CPU
/// fan-out needs a `Sync` closure, and an [`Embedder`] may hold a CoreML model
/// that is not shareable between threads.
fn embed_row(hidden: &[f32], mask: &[i64], dim: usize, reduce: Reduce) -> Vec<f32> {
    let mut vector = pool_row(hidden, mask, dim, reduce.pooling);
    match (reduce.dims, reduce.normalize) {
        // `load` refused `dims` without `normalize`, so a truncation always
        // re-normalizes; the order is the shared function's business.
        (Some(n), _) => truncate_renormalize(&mut vector, n),
        (None, true) => l2_normalize(&mut vector),
        (None, false) => {}
    }
    vector
}

/// `--dims` as the truncation it asks for: `None` when it asks for nothing
/// (not given, or equal to the model's own dimension, which changes no
/// vector), refused outside `1..=dim`. Decided at load rather than at the
/// first embed, so a bad value stops the run before any input is read, and
/// the refusal names the model's dimension, which is the number the caller
/// was guessing at.
fn check_dims(dims: Option<usize>, dim: usize) -> Result<Option<usize>> {
    match dims {
        None => Ok(None),
        Some(n) => truncation(n, dim).map_err(|e| {
            anyhow::anyhow!(
                "--dims {n} is outside this model's dimensions; it produces {dim}, so \
                 --dims {e}"
            )
        }),
    }
}

/// Refuse a `--max-seq-length` the model has no positions for.
///
/// The rotary tables are built at load with one entry per position the config
/// declares, so a longer input reaches past them and candle reports a shape
/// mismatch from inside the attention (`inconsistent last dim size in rope`).
/// That is an error either way, but only after the model is loaded and the
/// input tokenized, and it names candle's tensors rather than the flag.
///
/// The CoreML path does not need this. A bundle cannot be converted for a
/// length past its positions in the first place (`EncoderConfig::check_lengths`
/// refuses it: "the model would run and be wrong"), and a `--max-seq-length`
/// past the largest bucket is already refused at load with the bucket named,
/// which is the more useful of the two messages.
pub(crate) fn check_max_seq(max_seq_length: usize, positions: usize) -> Result<()> {
    anyhow::ensure!(
        max_seq_length <= positions,
        "--max-seq-length {max_seq_length} is longer than this model has positions for \
         ({positions}); lower it, and split the text if all of it has to be embedded"
    );
    Ok(())
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
        remark!("{}", note.message());
    }
    pooling
}

/// Read a `config.json` once, for callers that want more than one view of it.
///
/// A reranker's `config.json` describes both the encoder and the head, and two
/// reads could land either side of a file being replaced — a checkpoint swapped
/// mid-load would then be half one model and half another.
pub(crate) fn read_config_json(path: &Path) -> Result<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

/// Read and parse a `config.json`.
pub(crate) fn read_config(path: &Path) -> Result<Config> {
    parse_config(read_config_json(path)?, path)
}

/// One view of an already-read `config.json`. `path` only names the file in an
/// error.
pub(crate) fn parse_config<T: serde::de::DeserializeOwned>(
    json: serde_json::Value,
    path: &Path,
) -> Result<T> {
    serde_json::from_value(json).with_context(|| format!("cannot parse {}", path.display()))
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
mod dims_tests {
    use super::*;

    /// `--dims` keeps the leading dimensions and re-normalizes them, so the
    /// truncated vector is unit-length in its own space rather than a slice of
    /// a unit vector — which is the difference between dot = cosine holding
    /// and silently not.
    #[test]
    fn truncation_renormalizes_the_kept_prefix() {
        let reduce = |dims| Reduce {
            pooling: Pooling::Cls,
            normalize: true,
            dims,
        };
        // One token, dim 2, CLS pooling: the row *is* the hidden state [3, 4].
        let hidden = [3.0f32, 4.0];
        let full = embed_row(&hidden, &[1], 2, reduce(None));
        assert_eq!(full, vec![0.6, 0.8]);

        // Truncated to 1 dim, the survivor is re-normalized to unit length,
        // not left at its sliced value 0.6.
        let cut = embed_row(&hidden, &[1], 2, reduce(Some(1)));
        assert_eq!(cut, vec![1.0]);

        // dims == dim is allowed and changes nothing.
        let same = embed_row(&hidden, &[1], 2, reduce(Some(2)));
        assert_eq!(same, full);
    }

    /// A `--dims` that shortens nothing is no truncation: the vectors are a
    /// full run's, so the metadata must say so too, and `output_dim` marks
    /// only vectors that are not interchangeable with a full run's.
    #[test]
    fn dims_equal_to_the_model_claim_nothing() {
        assert_eq!(check_dims(Some(256), 512).unwrap(), Some(256));
        assert_eq!(check_dims(Some(512), 512).unwrap(), None);
        assert_eq!(check_dims(None, 512).unwrap(), None);
    }

    #[test]
    fn dims_outside_the_model_are_refused_with_the_model_dimension_named() {
        assert!(check_dims(Some(1), 512).is_ok());
        for bad in [0, 513] {
            let e = check_dims(Some(bad), 512).unwrap_err().to_string();
            assert!(e.contains("512"), "should name the model dim: {e}");
        }
    }

    /// Ruri v3 has 8192 positions, and one token past them used to fail inside
    /// candle's rotary embedding, after the model had loaded and the input had
    /// been tokenized.
    #[test]
    fn a_length_past_the_models_positions_is_refused_with_the_limit_named() {
        assert!(check_max_seq(512, 8192).is_ok());
        assert!(check_max_seq(8192, 8192).is_ok());
        let e = check_max_seq(8193, 8192).unwrap_err().to_string();
        assert!(e.contains("8192"), "should name the model's limit: {e}");
        assert!(e.contains("8193"), "should name what was asked for: {e}");
    }

    /// The combination is refused before any file is opened, so the error is
    /// about the flags rather than a missing path.
    #[test]
    fn dims_without_normalization_is_refused_at_load() {
        let result = Embedder::load(
            &ModelSource::Files {
                model: "/nonexistent/model.safetensors".into(),
                tokenizer: "/nonexistent/tokenizer.json".into(),
            },
            Options {
                dims: Some(256),
                normalize: false,
                ..Options::default()
            },
        );
        let e = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("loading should have been refused"),
        };
        assert!(e.contains("--no-normalize"), "unexpected error: {e}");
    }
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
