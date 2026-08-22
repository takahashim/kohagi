# Changelog

## [Unreleased]

### Added

- **`--print-model-info` reports `declared_max_seq_length`.** The token limit
  the checkpoint's `sentence_bert_config.json` declares, which
  sentence-transformers obeys and Kohagi does not: `--max-seq-length` still
  decides the run, and still defaults to 512. `ruri-v3-130m` declares 8192, so
  the same directory embeds a long text differently under the two libraries;
  this is where that is visible. Converted CoreML bundles carry the file too.

### Changed

- **Long inputs cost far less time and memory.** Peak memory is now flat from
  512 to 8192 tokens (1.0 GB, was up to 6.8 GB), and an 8192-token document
  takes 16 s instead of 100 s on one worker. The eight-worker default can embed
  8192-token documents at all, in 2.4 GB.

  Vectors are unchanged to f32 rounding (worst 1.8e-6 against values reaching
  1.0), so an index built with 0.6.0 does not need rebuilding.

## [0.6.0] - 2026-08-18

### Added

- **`--dims N`: Matryoshka truncation.** Keeps the first N dimensions of each
  embedding and re-normalizes, matching `SentenceTransformer(model,
  truncate_dim=N)` (worst `1 - cos` 1.8e-11), on every device. The summary's
  `dim=` reports the output dimension; `--print-model-info` keeps `dim` as the
  model's own and adds `output_dim`. N outside `1..=dim` and combining with
  `--no-normalize` are refused at load. Truncated and full vectors must not
  share an index.

- **`--expect-sha256 <hex>`.** Pass a prefix of the expected digest (the
  summary's 12 digits or the full 64) and either binary refuses weights whose
  digest does not start with it. The run exits 1 at load, before any output. On
  `--device coreml` the bundle's recorded `source_sha256` is checked; a bundle
  that recorded none (converted before 0.6) is refused. Combine with
  `--print-model-info` for a standalone check.

- **`kohagi-rerank`: a cross-encoder for reordering search results.** Reads
  `{"id","query","text"}` JSONL and writes `{"id","score"}`, under the same
  protocol rules as `kohagi`; see [PROTOCOL-rerank.md](PROTOCOL-rerank.md).
  Retrieval finds candidates; this reorders them, at a forward pass per pair.

  It runs any one-label ModernBERT sequence-classification checkpoint, which
  `cl-nagoya/ruri-v3-reranker-310m` and the `hotchpotch/japanese-reranker-*-v2`
  family all are. The score is the sigmoid `sentence_transformers.CrossEncoder`
  returns for such a model, matched to f32 rounding (worst |diff| 5.1e-07), so
  **thresholds tuned against that library carry over unchanged**; `--raw-logits`
  gives the logit instead.

  `--device coreml` runs it on the Neural Engine at 18.5 pairs/s against 3.3 on
  the CPU (`ruri-v3-reranker-310m`, M2).

- **Every run says which weights answered.** The stderr summary now carries the
  sha256 of the loaded `model.safetensors`, plus the resolved pooling, dimension
  and `max_seq_length`. Fine-tuned checkpoints differ only in their bytes, so a
  results file that records a path records what someone meant to run rather than
  what ran. The digest matches `sha256sum` on the same file, and costs nothing:
  it is computed alongside the encoding.

- **`--print-model-info`** writes those same facts as one JSON line on stdout and
  exits without reading stdin, for an evaluation script to record beside its
  numbers. On both binaries.

- **A converted CoreML bundle records the checkpoint it came from**, as a Hub id
  or path and as that checkpoint's sha256, reported as `source` and
  `source_sha256`. `scripts/convert_coreml.py` records the same.

- **New library API:** `kohagi::rerank`, `ModelInfo` with `Embedder::info` and
  `Reranker::info`, and `kohagi::cli`. `ModelInfo` carries a converted bundle's
  facts as a `Bundle` and a run's answer shape as an `Output` (`Embedding` with
  its `output_dim`, or `Score`), so a model cannot claim both; the JSON
  `--print-model-info` writes is unchanged. `kohagi::program` names the running
  binary, which is the prefix every stderr line carries.

### Changed

- **The stderr summary line has more fields, and `dim=` moved.** From
  `model=… dim=512 in=…` to
  `model=… sha256=1c342581efc2 pooling=mean dim=512 max_seq=512 in=…`.
  **Anything parsing that line by position needs updating**; by key it does not.
  A CoreML bundle reports `source_sha256=` in place of `sha256=`.

- **A checkpoint is labelled by its directory** when its weights file is
  `model.safetensors`, which every fine-tune's is: `model=alpha05` rather than
  `model=model.safetensors` for every model on the machine.

- **`Options` has a new `dims` field**, and `Embedder::dim` returns the output
  dimension (`dims` when set, `hidden_size` otherwise). A breaking change for
  library code building `Options` by struct literal; `..Options::default()`
  spellings are unaffected.

- **Release archives contain both binaries**, `kohagi` and `kohagi-rerank`. The
  archive name is unchanged.

### Fixed

- **A head bias the config does not declare is reported rather than dropped in
  silence.** `classifier_bias` and `norm_bias` decide whether the head's biases
  load, as they do in Hugging Face; a checkpoint carrying one its config does not
  declare used to lose it without a word, and a head missing a term does not
  fail, it scores slightly wrong.

- **`tools/rerank_parity.py` and `tools/rerank_fp16_bands.py` stop at any nonzero
  exit.** They checked only for exit 1, so a run that finished with skipped lines
  reported a number computed over a different sample than the one asked for.

- **`tools/eval_retrieval.py` takes its passthrough arguments from an explicit
  `--` split**, whose survival through argparse differs by Python version.

### Compatibility

- **A CoreML bundle converted before 0.6 cannot be used for reranking.** It has
  no `head.safetensors`, so `kohagi-rerank --device coreml` says so at load and
  exits 3. Reconvert the checkpoint, or score on `--device cpu`. **Embedding with
  an old bundle is unaffected**; it simply reports no `source_sha256`.

- **A reranker on `--device coreml` is not interchangeable with one on the CPU at
  a fixed threshold**, because the encoder is fp16 there. How much of a corpus
  can cross a given threshold, and how to work it out for a threshold of your
  own, is in PROTOCOL-rerank.md.

## [0.5.1] - 2026-08-08

### Fixed

- **A converted CoreML model no longer returns NaN on the CPU.** The emitted
  attention mask blocked with fp16 `-inf`, which leaves a deeply padded input with
  query positions that have nothing left to attend to; softmax over such a row is
  NaN, and one of them takes the rest of the output with it, so **the whole
  embedding came back NaN** rather than part of it. It needed a text padded that
  far — the 256 bucket and up — and the CPU compute unit, since the Neural Engine's
  fp16 saturates instead. The ANE path was never wrong.

  Cached conversions rebuild themselves. **A bundle converted by an earlier version
  carries this and needs reconverting**, including one downloaded from the Hub;
  `takahashim/ruri-v3-130m-coreml` has been republished.

- **`coreml-convert --compiled` fails before converting rather than after.** It
  needs a build with `--features coreml,coreml-export`; without it the run
  downloaded the checkpoint and wrote a 260 MB bundle before saying so. The
  documented command passed the wrong features to match.

## [0.5.0] - 2026-08-01

### Added

- **`--device coreml` converts a checkpoint itself.** The Neural Engine backend
  used to need a bundle someone had already converted (`--coreml-dir` or
  `--coreml-model-id`). Given neither, `kohagi --device coreml` now converts the
  same `--model-id` the CPU path would take, caches it, and loads it — no Python,
  nothing to publish first. The first run reports each slow step and takes about
  20 s to convert; later runs load from the cache in about 0.3 s. Caches live in
  `~/Library/Caches/kohagi/coreml` (`$KOHAGI_COREML_CACHE` to relocate) and are
  safe to delete.

  `--coreml-buckets` sets the sequence lengths, `64,128,256,512` by default and
  up to 4096; past that CoreML's compiler stops finishing in any usable time, so
  it is refused rather than left to hang. The lengths share one copy of the
  weights, so the set costs no disk, but each is a model to open — match it to
  the lengths your texts actually are, because a bucket nothing lands in is pure
  overhead. `--coreml-quantize {embeddings,all}` stores the weights as int8 —
  264.8 MB to 212.3 MB, or to 132.6 MB for all of them. fp16 stays the default
  because **a quantized bundle's vectors are not interchangeable with an fp16
  one's**, so the two must not share an index; loading a quantized bundle says so.

- **`coreml-export` feature: a CoreML converter in Rust.** Reads a ModernBERT
  checkpoint's safetensors directly instead of going through
  `scripts/convert_coreml.py` and its PyTorch install. For `cl-nagoya/ruri-v3-130m`
  the output is bit-identical to the Python conversion, and it is verified against
  ten ModernBERT checkpoints from 256 to 1024 wide. Every sequence length is one
  CoreML function over a single copy of the weights, so three buckets come to
  264.8 MB against 794 MB for separate packages.

  ```console
  cargo run --release --bin coreml-convert --features coreml-export -- \
      --model-id cl-nagoya/ruri-v3-130m --out-dir ./coreml \
      --sequence-lengths 64,128,256,512
  ```

  A checkpoint the converter cannot honour is refused before anything is written,
  naming every reason at once. The emitter is also reachable as the
  `kohagi::coreml_export` module.

- **A converted model is checked against its own float32 output, once.** After
  converting, Kohagi compares four probes on the ANE and the CPU and warns if they
  differ by more than fp16 rounding explains. Some checkpoints are themselves
  sensitive to fp16 — `nomic-ai/modernbert-embed-base` drifts by 7e-3 under this
  converter and under coremltools alike — which no converter can know in advance.

- **Compiled CoreML models are cached between runs.** `--device coreml` used to
  compile every `.mlpackage` bucket into a temporary directory, paying ~20 s per
  bucket on every run; now only the first run pays (8.1 s to 0.2 s on an M2). A
  repository can therefore ship the `.mlpackage` alone instead of doubling its
  size with a `compiled/` copy. If the cache cannot be used, Kohagi compiles as
  before.

- **`hidden_activation: "silu"` is supported** on every device, which opens the
  `ibm-granite/granite-embedding-*-r2` family. An activation Kohagi does not
  implement is still an error rather than a silent fall back to gelu.

- **Windows NVIDIA GPU support.** The new `cuda` feature and `--device cuda` run
  Candle's CUDA backend on NVIDIA GPUs, and Windows x64 release builds bundle it.
  AMD and Intel GPUs remain unsupported.

- **Truncation visibility.** Text longer than `--max-seq-length` is truncated
  before embedding; that used to be silent. The stderr summary now always ends
  with `truncated=N`, and `--report-tokens` adds `n_tokens` and `truncated` to
  each output record so a caller can route truncated documents to a chunking
  pass. Without the flag the output is byte-for-byte the previous protocol-1
  shape. New library method `Embedder::embed_with_tokens` returns the same
  vectors plus a `TokenInfo` per text.

- **`tools/eval_retrieval.py`** measures JaCWIR and JQaRA for any Kohagi
  configuration, and **`tools/coreml-jigs`** inspects a converted directory
  before publishing it (declared I/O and provenance, `weight.bin` validation,
  per-operation Neural Engine placement, per-bucket latency, and output parity
  between two configurations).

- **An OpenAI-compatible endpoint, in `examples/`.** `openai_proxy.py`,
  `.rb` and `.ts` each serve `/v1/embeddings` in front of Kohagi, so an existing
  OpenAI client works by swapping `base_url` and nothing else — which is what
  that compatibility is actually worth. They stay examples rather than a
  `--serve` flag: the HTTP layer is the caller's to choose, and Kohagi's
  contract stays "spawn a process, write JSONL".

- **A blank line ends a batch.** Kohagi embeds in chunks of 1024 records and at
  end of input, so a caller wanting an answer sooner had no way to ask for one:
  a long-lived process fed two records returned nothing, and closing stdin —
  the only signal available — ended the process. A blank line on stdin now means
  "embed what you have and reply now", and Kohagi answers with a blank line of
  its own so the caller knows it has everything. Counting records instead would
  hang as soon as one was skipped for being malformed. Nothing changes for
  `cat texts.jsonl | kohagi`, and with `--format openai` each batch is its own
  complete response object.

- **`--format openai`** writes one OpenAI `/v1/embeddings` response object for
  the whole run instead of the JSONL stream, so code already written against
  that API can read Kohagi's output unchanged. Embeddings are identified by
  `index` (the input position) rather than by id, because that is what the API
  means; `usage.prompt_tokens` carries the token total. The document is written
  in pieces rather than buffered, so memory stays flat — but an aborted run
  leaves it incomplete, where JSONL would have left a shorter valid file. The
  default is unchanged.

### Changed

- **A `config.json` Kohagi cannot honour is refused rather than assumed.** An
  unknown `hidden_activation` fails the parse instead of silently running gelu,
  and a config carrying neither `rope_parameters` nor the flat RoPE thetas fails
  instead of defaulting one; both would otherwise produce plausible-looking
  vectors. The converter additionally requires `max_position_embeddings` and
  `pad_token_id`, and rejects a config with a duplicate key rather than taking the
  last value.

### Fixed

- **transformers 5.x configs load again.** transformers 5.x writes the RoPE
  thetas into `rope_parameters` and stops writing `global_rope_theta` /
  `local_rope_theta`, which Kohagi's config reader required — so every ModernBERT
  checkpoint saved with it (94 of the 690 published on the Hub, and growing)
  failed to load on any device.

- **CoreML output is read through its strides.** `--device coreml` copied the
  hidden states straight off `MLMultiArray`'s `dataPointer`, which is only correct
  when the array is densely packed. CoreML may pad an axis, and reading a padded
  array that way builds the embedding partly out of padding. No shipped model was
  affected — 512 and 768 are already aligned — so it would have stayed silent
  until a model whose hidden size is not.

- **A mismatched CoreML directory now fails at load instead of returning wrong
  vectors.** `--device coreml` took the embedding width from the directory's
  `config.json` and the bucket lengths from file names without checking either
  against the models, so a `config.json` from a different checkpoint returned
  vectors pooled over the wrong stride. Each bucket's input names, output name and
  output shape are now verified against the model's own description before any
  input is read.

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
- `tools/model_check.py` — smoke-test Kohagi against any ModernBERT sentence
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
