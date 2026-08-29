# burnprobe

The measurement jig for the candle → Burn migration. Outside the workspace, like
`coreml-jigs`, so its backends (`flex`, `cpu`, `ndarray`, `vulkan`) stay out of
Kohagi's `Cargo.lock`.

It carries a second ModernBERT forward written against Burn's tensors, generic
over the backend, so one binary can put every candidate backend on the same
weights and the same texts and compare against `kohagi --device cpu`.

## Binaries

| | what it answers |
| --- | --- |
| `real` | the whole encoder on real `ruri-v3-130m` weights, against kohagi's own output |
| `gemm` | matrix multiply alone, candle against burn-flex and burn-ndarray |
| `fuse` | one fused kernel against the same thing composed from primitives |
| `burnprobe` | the same encoder on random weights, for shapes without a checkpoint |

## Running `real`

```console
$ export MODEL=~/.cache/huggingface/hub/models--cl-nagoya--ruri-v3-130m/snapshots/<sha>
$ kohagi --max-seq-length 512 < texts.jsonl > ref.jsonl     # the baseline
$ export TEXTS=texts.jsonl REF=ref.jsonl ROWS=16
$ cargo run --release --bin real -- flexpar
```

`texts.jsonl` is Kohagi's own input format; the reference is whatever
`--device cpu` produced for it. Modes: `flexpar` (burn-flex with the row fan-out
Kohagi's CPU path uses), `flex`, `ndarray`, `burncpu`, `f32`, `f16`, `mixed2`.

Every knob is an environment variable so an A/B runs in one binary — the only
way to compare two paths on a machine whose noise floor can invert the result:

| | |
| --- | --- |
| `FUSED=0` | fall back to the composed default instead of the fused kernel |
| `SPLIT=1` | two contiguous `Wi` matmuls instead of one wide one plus a narrow |
| `PER=n` | rows per fanned-out unit |
| `PROFILE=1` | per-operation totals |
| `GELU=tanh` | diagnostic only: does the transcendental cost anything (it did not) |

## What it has established

On a Radeon 780M and an 8-core Zen 4, against `kohagi --device cpu`
(6.8 rows/s at 16 rows of ~450 tokens):

- burn-flex reaches **90%** of that, at `1 - cosine` 1.2e-12.
- Three things got it there, in order of size: the row fan-out (6.3x — Burn's
  own parallelism does not substitute for it), two contiguous `Wi` matmuls
  instead of one wide one (+9%, which is the arrangement `Wi::Split` in
  `src/encoder.rs` already records for the CPU), and a fused RoPE (+10%).
- Fusing GeGLU by hand did **not** help: burn-flex's gelu and multiply are
  already SIMD, and a scalar single-pass kernel loses more on vectorisation
  than it wins on memory traffic. Beating it needs a `macerator` kernel.
- Burn's own backend ops already cover softmax, gelu and layer_norm.
  `burn_nn`'s RoPE is the one composed from primitives, which is why that is
  the only kernel this carries.
- `burn-cpu` (the MLIR backend) is 24x off and compute-bound; restricting cores
  does not help it, so cubecl#1566 is not the cause. cubecl#1527 explains the
  117 s that its first run costs on every process start.
