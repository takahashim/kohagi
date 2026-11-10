# Choosing a device

- `--device` picks the backend for the forward pass.
- The default, `cpu`, works everywhere and needs no special build.
- The others trade something for speed, and what they trade differs:

| `--device` | Build | Speed | Output |
| --- | --- | --- | --- |
| `cpu` (default) | any | baseline | f32, matches PyTorch to rounding |
| `cpu --precision bf16` | x86_64 with AVX512-BF16 | ~2× | cosine ≈ 0.99999 vs f32 |
| `metal` | `--features metal` | ~1.8× on an M2 | f32, unchanged (worst `1 - cosine` 9e-13) |
| `cuda` | `--features cuda` | GPU-dependent | f32 |
| `coreml` | `--features coreml` | one text per forward, padded to a fixed bucket; see below | fp16 encoder, cosine ≈ 0.99999 |

- An unsupported request is refused at startup rather than falling back
  silently. A run that quietly landed on the CPU would look like a Metal
  benchmark result.

## CPU

- Apple Accelerate on macOS, which performs within about 20% of PyTorch with
  equivalent output.
- Linux links no BLAS at all (candle's pure-Rust `gemm` does the matrix
  multiplies), so on Linux `--precision bf16` is where the throughput is.
- Batches run in parallel across physical CPU cores.
- `RAYON_NUM_THREADS` overrides the count; more threads may raise throughput at
  the cost of memory, since worker count is a direct memory multiplier.

## `--precision bf16` on AVX512-BF16 CPUs

- On Zen 4, Sapphire Rapids and newer, `bf16` is used for the projection layers
  while normalization, softmax and attention scores stay `f32`.
- What stays f32 and unfused is the `q·kᵀ` and `att·v` matmuls, which is why the
  long row gains less than the short one.
- bf16 also pays about a second more at load, converting the weights, which
  matters if you spawn a process per small batch.
- Vectors stay very close to f32 (cosine ≈ 0.99999) but are **not
  bit-identical**, which is why it is opt-in: f32 produces the same vectors on
  every machine, and that matters when embeddings generated on different hosts
  share one index.
- Unsupported CPUs, including Apple Silicon, reject `--precision bf16` at
  startup.

## `--device metal` on Apple Silicon

- Build with `--features metal`.
- On an M2 it runs about 1.8× faster than the Accelerate CPU path, measured on
  512-token batches.
- Output is f32 and unchanged (worst `1 - cosine` 9e-13 against CPU).

## `--device cuda` on NVIDIA GPUs

- Build with `--features cuda`, or use the Windows release binary, which bundles
  it.
- It needs a compatible NVIDIA driver and CUDA runtime at run time.
- AMD and Intel GPUs are not supported.

## `--device coreml` on the Apple Neural Engine

- Build with `--features coreml`. Cosine ≈ 0.99999 against the CPU output.
- How it compares to the CPU and GPU paths depends on the chip generation, the
  text lengths and the machine's cooling more than on anything Kohagi does,
  so no ratio is published here; measure on your own machine with
  [`tools/benchmark.py`](../tools/benchmark.py). Two things hold across
  machines: the Neural Engine is most efficient per token in the middle
  buckets and least in the largest, where attention's quadratic share grows;
  and for short inputs the multicore CPU backend may still be faster, because
  what the ANE gains per token it gives back padding each row to a fixed bucket
  length and running one text per forward.
- The ANE needs fixed input shapes, so this device runs a converted bundle
  rather than the safetensors the others read. See
  [CoreML bundles](coreml.md).

## Reranking

- `kohagi-rerank` takes the same `--device`.
- **A score threshold does not carry unchanged from `cpu` to `coreml`**. See
  [Retrieval and reranking](reranking.md).

## Measuring it yourself

- These numbers moved by a factor of two between runs on the same machine while
  other work was going on.
- Treat them as an order of magnitude rather than a ranking, and compare against
  a run of your own.
- [`tools/benchmark.py`](../tools/benchmark.py) times Kohagi against Sentence
  Transformers on the same corpus and settings.

## Accuracy

- Kohagi's f32 output matches the Sentence Transformers and PyTorch reference to
  within f32 rounding error: on 512-token inputs, `1 - cosine ≈ 3e-12`.
- Verify it on your own texts with
  [`tools/parity_check.py`](../tools/parity_check.py) for embeddings and
  [`tools/rerank_parity.py`](../tools/rerank_parity.py) for scores.
- Three settings must match on both sides for such a comparison to mean
  anything: the sequence length, the pooling, and the prefix. The scripts set
  them explicitly.
- To record which weights produced a given result, use `--print-model-info`,
  whose `sha256` names the checkpoint by content rather than by path.
