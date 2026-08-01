# Changelog

## [0.5.0] - 2026-08-01

### Added

- **Compiled CoreML models are cached between runs.** `--device coreml` used to
  compile every `.mlpackage` bucket into a throwaway temporary directory, so a
  converted model that ships only the portable form paid ~20 s per bucket on
  every run. The compile now lands in `~/Library/Caches/kohagi/coreml`
  (`$KOHAGI_COREML_CACHE` to relocate it), keyed by the package's path and the
  size and mtime of its contents, so only the first run pays — 8.1 s to 0.2 s on
  an M2. A repository can therefore ship the `.mlpackage` alone instead of
  doubling its size with a `compiled/` copy. Re-converting a model supersedes its
  entry rather than adding one, so the cache holds at most one compile per bucket
  per directory. The cache is not load-bearing: if it cannot be read or written,
  or a cached bundle no longer loads after an OS update, Kohagi compiles as
  before.
- **CoreML development jigs** under `tools/coreml-jigs`, for checking a converted
  directory before publishing it: `coreml-inspect` (declared inputs, outputs and
  converter provenance, read without compiling), `milblob` (validate, round-trip
  and diff a `weight.bin`), `computeplan` (per-operation ANE placement, with a
  recorded baseline to diff against), `mil-inventory` (operation inventory and
  order, from either MIL form), `bucket-latency` and `parity`. They sit outside
  the workspace and are not part of the published crate.
- **Windows NVIDIA GPU support.** The new `cuda` feature and `--device cuda`
  run Candle's CUDA backend on NVIDIA GPUs. Windows x64 release builds now
  bundle that backend, and CI compiles both the default Windows build and the
  CUDA release build. AMD and Intel GPUs remain unsupported.
- **Truncation visibility.** Text longer than `--max-seq-length` is truncated
  before embedding; that used to be silent. The stderr summary now always ends
  with `truncated=N`, and `--report-tokens` adds `n_tokens` and `truncated` to
  each output record so a caller can route truncated documents to a chunking
  pass. Without the flag the output is byte-for-byte the previous protocol-1
  shape. New library method `Embedder::embed_with_tokens` returns the same
  vectors plus a `TokenInfo` per text.

- **`coreml-export` feature.** Generates a CoreML `.mlpackage` for a
  ModernBERT encoder from Rust, reading the checkpoint's safetensors directly
  instead of going through `scripts/convert_coreml.py`. For `cl-nagoya/ruri-v3-130m`
  at sequence length 128 the result is bit-identical to the Python conversion's
  output, with the same 735 operations in the same order and the same Neural
  Engine placement. The protobuf bindings are
  committed under `src/coreml_proto/generated/`, so building Kohagi needs neither
  `protoc` nor a build script. A `coreml-convert` binary drives it:

  ```console
  cargo run --release --bin coreml-convert --features coreml-export -- \
      --model-id cl-nagoya/ruri-v3-130m --out-dir ./coreml \
      --sequence-lengths 128,256,512
  ```

  Verified against ten ModernBERT checkpoints from 256 to 1024 wide; nine match
  Kohagi's CPU path to fp16 rounding, and the tenth
  (`nomic-ai/modernbert-embed-base`) diverges identically under the Python
  conversion, so that one is the checkpoint's own fp16 sensitivity rather than a
  converter difference. `--compiled` also emits `compiled/<name>.mlmodelc`, which
  needs a build with `--features coreml,coreml-export`.

  Every length is one CoreML function over a single copy of the weights, so the
  three buckets come to 264.8 MB against 794 MB for the published per-length
  packages. `required-features` keeps the binary out of the default build and the
  release archives, though the macOS build enables the feature so that
  `--device coreml` can convert for itself; the emitter is also reachable as the
  `kohagi::coreml_export` module. `--quantize-embeddings` stores the embedding table as int8 with
  a scale per row, dequantized inside the graph: 264.8 MB to 212.3 MB for
  ruri-v3-130m, at 1.7e-5 cosine distance from the CPU path against 3.6e-6 for
  fp16, and `--quantize-all` extends it to the projections for 132.6 MB at 1.6e-4.
  A quantized bundle's vectors are not interchangeable with an fp16 one's, so the
  model records which it is. Measured on JaCWIR (750 queries, 68,078 documents),
  quantizing the embeddings costs nothing — MAP@10 0.8592 to 0.8599 and JQaRA
  nDCG@10 0.7112 to 0.7122 — while quantizing everything costs 0.001 to 0.002 on
  both for half the size.

- **`--device coreml` converts a checkpoint itself.** The Neural Engine backend
  used to need a bundle someone had already converted (`--coreml-dir` or
  `--coreml-model-id`). Given neither, it now emits one from the same
  `--model-id` the CPU path would take, caches it, and loads it — so
  `kohagi --device coreml` works on any supported ModernBERT checkpoint with no
  Python and nothing published. The first run reports each slow step
  (download, convert, compile) and takes about 20 s to convert; later runs load
  from the cache in about 0.3 s. `--coreml-buckets` sets the sequence lengths and
  `--coreml-quantize {embeddings,all}` the quantization; fp16 stays the default
  because a quantized bundle's vectors are not interchangeable with an fp16
  one's. The cache key covers the checkpoint revision, the buckets, the
  quantization and a graph version, and superseding only removes the entries a
  new one replaces. macOS release binaries now include the converter.

- **A converted model is checked against its own float32 output, once.** Right
  after converting, Kohagi embeds four probes on both the ANE and the CPU and
  warns if they differ by more than fp16 rounding explains. This catches what no
  converter can know in advance: `nomic-ai/modernbert-embed-base` drifts by 7e-3
  under this converter *and* under coremltools, because the checkpoint itself is
  sensitive to fp16. Loading a quantized bundle also says so, since its vectors
  cannot be mixed with an fp16 bundle's in one index.

- **`examples/eval_retrieval.py` measures JaCWIR and JQaRA.** The retrieval
  quality behind the quantization numbers is now reproducible from the
  repository: it fetches the datasets, runs any Kohagi configuration passed after
  `--`, and reports MAP@10/HIT@10 or nDCG@10.

- **`hidden_activation: "silu"` is supported.** ModernBERT's MLP activates the
  gate of a gated feed-forward, so this choice is what makes the block a GeGLU or
  a SwiGLU. Both paths now read it from the config: Candle's `gelu_erf` or `silu`
  on CPU, a second fused Metal kernel, and the `gelu` or `silu` MIL operation in
  the CoreML emitter. That opens `ibm-granite/granite-embedding-*-r2`, the most
  downloaded ModernBERT embedding family Kohagi previously refused. Verified at
  every level: the emitted block matches an independent f32 reference on both
  activations, the fused Metal kernel matches the split path to 2.4e-13, and
  `granite-embedding-97m-multilingual-r2`'s CoreML bundle matches the CPU path to
  1.0e-5 with 97.4% of operations on the Neural Engine. An activation Kohagi does
  not implement is still an error rather than a silent fall back to gelu.

### Changed

- **A `config.json` Kohagi cannot honour is refused rather than assumed.** An
  unknown `hidden_activation` fails the parse instead of silently running gelu,
  and a config carrying neither `rope_parameters` nor the flat RoPE thetas fails
  instead of defaulting one. Both would otherwise produce plausible-looking
  vectors. The converter additionally requires `max_position_embeddings` and
  `pad_token_id`, which it previously ignored — converting a checkpoint Kohagi
  cannot then load is worse than failing at conversion — and rejects a config
  with a duplicate key rather than taking the last value.

- **Loading a quantized CoreML bundle says so.** Its vectors are close enough to
  an fp16 bundle's to score the same on a retrieval benchmark but are not the
  same vectors, so mixing them in one index degrades quietly.

### Fixed

- **transformers 5.x configs load again.** transformers 5.x writes the RoPE
  thetas into `rope_parameters` and stops writing `global_rope_theta` /
  `local_rope_theta`, which Kohagi's config reader required — so every ModernBERT
  checkpoint saved with transformers 5.x (94 of the 690 surveyed on the Hub)
  failed to load on any device. Both spellings are now read, and a config
  carrying neither is an error rather than a silent default. Found by surveying
  every ModernBERT config on the Hub against the converter
  (`scripts/survey_modernbert.py`).

- **CoreML output is read through its strides.** `--device coreml` copied the
  hidden states straight off `MLMultiArray`'s `dataPointer`, which is only correct
  when the array is densely packed. CoreML may pad an axis, and reading a padded
  array that way interleaves values with padding. No shipped model was affected —
  512 and 768 are already aligned — so this was silent and would have stayed
  silent until a model whose hidden size is not. Found while checking a
  Rust-generated model's output against a reference.

- **A mismatched CoreML directory now fails at load instead of returning wrong
  vectors.** `--device coreml` took the embedding width from the directory's
  `config.json` and the bucket lengths from the bundles' file names, without
  checking either against the models. A `config.json` from a different
  checkpoint would panic if its `hidden_size` was larger and silently return
  garbage vectors if it was smaller, and a bundle named for the wrong length was
  padded and pooled at that wrong length. Each bucket's input names, output name
  and output shape `[1, seq, dim]` are now verified against the model's own
  description before any input is read.

## [0.4.0] - 2026-07-24

### Added

- **Pooling is taken from the checkpoint.** Kohagi reads the model's
  `1_Pooling/config.json` and uses the mode it declares, so a CLS-pooled model
  such as `Alibaba-NLP/gte-modernbert-base` works without a flag. `--pooling`
  now only overrides, and warns if the choice disagrees with the checkpoint, or
  if the model ships no pooling config at all (usually a reranker or a base LM
  rather than a sentence encoder).
- **Broader `config.json` compatibility.** The LayerNorm epsilon is accepted
  under HF's `norm_eps` spelling as well as `layer_norm_eps` (ruri ships both),
  and a config carrying neither falls back to the default rather than failing to
  load. Lets more ModernBERT checkpoints run unchanged.
- `examples/model_check.py` — smoke-test Kohagi against any ModernBERT sentence
  encoder on the Hub, checking retrieval and paraphrase structure rather than
  just that the process exited 0.

### Changed

- The prebuilt **macOS** release binary now bundles the Metal and CoreML
  backends, so `--device metal` and `--device coreml` work without building from
  source. The Linux binary stays CPU-only (both backends are macOS-only), and
  `cargo install kohagi` still needs `--features metal` / `--features coreml`.
- **`--pooling` no longer defaults to `mean`.** With the flag omitted, the
  pooling now comes from the checkpoint (see above) instead of always being
  mean. Mean-pooled models are unaffected; CLS models now pick CLS on their own.

### Notes

- `Options::pooling` changed from `Pooling` to `Option<Pooling>` (`None` =
  detect from the checkpoint). A breaking change for library code that sets the
  field.
- The project name is now written **Kohagi** as a proper noun in prose; the
  command, crate, and repository stay lowercase `kohagi`. Documentation only —
  nothing to change in your setup.

## [0.3.0] - 2026-07-23

### Added

- **CoreML / Apple Neural Engine backend** (`--device coreml`, macOS-only,
  behind the `coreml` cargo feature). Runs the ModernBERT encoder on the ANE
  from pre-converted, fixed-shape models — about 4× the Metal path at 512
  tokens, at cosine ≈ 0.99999 against the CPU output. Short texts still favour
  the multicore CPU path.
  - `--coreml-dir <DIR>` loads a local model directory;
    `--coreml-model-id <REPO>` downloads one from the Hugging Face Hub.
  - `--coreml-prefer {compiled,package}` chooses which form to download when a
    repo ships both `.mlmodelc` and `.mlpackage` buckets.
  - `scripts/convert_coreml.py` converts a model to the expected layout.
- **Exit code 3** for a CoreML request the backend cannot serve (built without
  the feature, no model given, or `--max-seq-length` past the largest bucket).
  Detected before any input is read, so no output is produced and a caller can
  retry on `--device cpu`. Only ever returned with `--device coreml`.
- Public API: `Backend::CoreML`, `ModelSource::CoreMl` / `CoreMlHub`,
  `CoreMlForm`, `UnsupportedRequest`, and an `Options::coreml_form` field.

### Changed

- **`--precision bf16` is faster.** The softmax and GeGLU are now vectorized
  (AVX-512), and the sliding-window attention layers walk only the band they
  attend to. Measured on an 8-core Zen 4: ~2.3× the f32 path on short texts and
  ~2.0× at 512 tokens (was ~1.9× and ~1.5×), at unchanged cosine ≈ 0.99999 to
  f32. Without AVX-512 the elementwise kernels fall back to scalar rows. The
  default f32 path is unchanged.

### Notes

- Adding the `Options::coreml_form` field and the new `Backend` / `ModelSource`
  variants is a breaking change for library code that builds those by struct
  literal or matches them non-exhaustively.

## [0.2.0] - 2026-07-23

### Added

- **Metal backend** (`--device metal`, behind the `metal` cargo feature): an
  opt-in Apple GPU path, ~1.8× the Accelerate CPU path at 512 tokens with
  unchanged f32 output.
- Benchmark and parity tooling under `examples/` — timing against Sentence
  Transformers, plus a reproducibility check.

### Changed

- Moved to candle 0.11.
- Hardened CI: `cargo fmt` / `--locked` checks, per-target release builds, and a
  Metal lint.

[0.5.0]: https://github.com/takahashim/kohagi/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/takahashim/kohagi/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/takahashim/kohagi/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/takahashim/kohagi/compare/v0.1.0...v0.2.0
