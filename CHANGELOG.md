# Changelog

## [0.3.1] - 2026-07-23

### Changed

- The prebuilt **macOS** release binary now bundles the Metal and CoreML
  backends, so `--device metal` and `--device coreml` work without building from
  source. The Linux binary stays CPU-only (both backends are macOS-only), and
  `cargo install kohagi` still needs `--features metal` / `--features coreml`.

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

[0.3.1]: https://github.com/takahashim/kohagi/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/takahashim/kohagi/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/takahashim/kohagi/compare/v0.1.0...v0.2.0
