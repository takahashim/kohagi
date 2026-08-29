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
| `coreml` | `--features coreml` | ~4× Metal at 512 tokens | fp16 encoder, cosine ≈ 0.99999 |
| `vulkan --precision f16` | `--features vulkan` | ~1.9× the bf16 CPU path | f16 encoder, cosine ≈ 0.99999 |
| `vulkan` (f32) | `--features vulkan` | slower than bf16 on CPU | f32, matches the CPU (worst `1 - cosine` 1.2e-12) |
| `cpu-burn` | `--features cpu-burn` | 1.5× `cpu` short, 0.85× long | f32, matches it (worst `1 - cosine` 1.0e-12) |

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

## `--device vulkan` on AMD, Intel and NVIDIA GPUs

- Build with `--features vulkan`, and pass `--precision f16`: the f32 path here
  is slower than `--device cpu --precision bf16` and exists to verify the other.
- Needs a Vulkan 1.2 driver and **nothing else** — no ROCm, no CUDA, no
  `HSA_OVERRIDE_GFX_VERSION`. Burn/CubeCL compiles the same kernels to SPIR-V, so
  a GPU that is not on a vendor's compute support list still runs. This is what
  makes AMD and Intel reachable at all: candle has neither a ROCm nor a Vulkan
  device, and `Device` is a closed enum, so no backend can be added from outside it.
- Measured on a Radeon 780M (gfx1103, Mesa RADV) with `ruri-v3-130m`, 64 texts of
  about 430 tokens, best of three interleaved runs, wall clock including load:

| | wall | rows/s (encode) |
| --- | ---: | ---: |
| `cpu` | 11.1 s | 6.4 |
| `cpu --precision bf16` | 6.7 s | 10.6 |
| `vulkan --precision f16` | **5.0 s** | **18.6** |

- **Startup costs about 0.8 s more than the CPU path**, so it pays from roughly
  **20 rows of that length**; below that `--device cpu --precision bf16` wins.
  CubeCL autotunes its kernels on first use — about 12 s once, then ~0.4 s per
  run from a 60 KB cache under `~/.cache/kohagi/autotune`.
- Accuracy against this crate's own CPU f32 output, over those 64 texts: median
  `1 - cosine` 3.4e-7, worst 8.9e-6 — the same order the CoreML path publishes.
- `--precision f16` is not a plain f16 forward. LayerNorm's variance and softmax
  run in f32 while everything else stays f16; lowering those two as well is 16%
  faster and lands at `1 - cosine` 3.5e-2, which is not an embedding worth
  indexing.
- `--precision bf16` is refused here. CubeCL emulates it rather than reaching the
  hardware: measured 12× slower than f16, and it produced NaN.
- **One precision per process.** CubeCL binds one float element type per device,
  so a second `Embedder` asking for the other precision is refused at load rather
  than silently served in the first one's arithmetic.
- Memory stays bounded by the same kind of budget the candle GPU path uses, and a
  tighter one: throughput here is flat in the row count (19.3 rows/s at one row
  against 20.5 at sixty-four), so bounding memory harder costs nothing. 512 rows
  of 430 tokens peaked at 1.1 GiB of GPU memory.
- Reranking runs here too: the encoder moves and the classifier head stays in
  f32 on the CPU, the same split the CoreML path makes.

## `--device cpu-burn`: the CPU, on Burn instead of candle

- **Transitional.** It exists so the two CPU engines can be compared in one
  process, which is the only way to compare anything on a machine whose noise
  floor can invert a result. It goes away when one of them wins.
- Build with `--features cpu-burn`. f32 only: `bf16` is the hand-written candle
  kernel in `src/bf16`, and `f16` is the Vulkan device's recipe.
- **Which is faster depends on the length**, which is the interesting part.
  Measured on an 8-core Zen 4 with `ruri-v3-130m`, best of three interleaved
  runs, wall clock:

| texts | `cpu` | `cpu-burn` | | peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 64 × 42 tokens | 2.21 s | **1.44 s** | 1.53× | 1027 / 1022 MiB |
| 64 × 460 tokens | 11.55 s | **9.56 s** | 1.21× | 1028 / 1166 MiB |
| 8 × 2048 tokens | 7.60 s | **7.26 s** | 1.05× | 1385 / 1135 MiB |
| 4 × 8192 tokens | **30.2 s** | 33.5 s | 0.90× | 2005 / 1573 MiB |

- Worst `1 - cosine` against `--device cpu` is 1.0e-12 at any of those lengths,
  which is float addition order and nothing else: both engines run f32.
- Loading costs about 0.26 s more (0.82 s against 0.56 s). candle memory-maps
  and builds its tensors lazily; this borrows f32 out of the same mapping and
  transposes it once so the GEMM's right-hand side is contiguous.
- What got it there, in order of what each was worth:
  - **Fanning out** one forward at a time across the pool rather than one wide
    forward: 0.9 rows/s to 5.7. Burn parallelises inside an operation, and that
    does not substitute for running independent forwards at once.
  - **A row cap** of 4 on top of the budget, which is what keeps short inputs
    fanned out at all — uncapped, a whole batch of 42-token texts lands in one
    forward and the pool sits idle. Worth 2.89 s to 1.48 s on its own, and it is
    the reason the short rows above come out ahead.
  - **Two contiguous `Wi` matmuls** rather than one wide one and a strided
    split, +9% — the same choice `crate::encoder` records for the candle CPU
    path, and the opposite of what Vulkan wants.
  - **A fused RoPE**, +10%. `burn_nn` composes it from a matmul against a sign
    matrix; candle-nn has a kernel. It is the only operation where Burn composes
    what candle fuses.
  - **Banding the sliding-window layers**, which is 1.9× at 2048 tokens (8.57 s
    against 16.26 s) and nothing below about 258, where
    `crate::attention::banding_pays` declines to try.
  - **`gemm`'s AVX-512 kernels**, which burn-flex offers as `x86-v4` and candle
    does not enable for its own copy of the same crate. Dispatched at run time,
    so a CPU without the instructions takes the path it always took; worth 3% on
    short texts and 9% at 460 tokens for 330 KB of binary.
  - **Splitting the queries** once one row's scores pass the budget, which is
    the handover `crate::encoder` already makes at about 724 tokens. Without it
    a long sequence materialises `[heads, seq, seq]` per layer: 4 texts of 8192
    tokens peaked at **16.8 GiB** and took 40.5 s, against 2.8 GiB and 34.3 s
    with it.
  - **One shared window table** rather than a `[seq, seq]` one. A banded layer's
    blocks all reach the same way, so the mask depends on the offset between a
    block's first query and its first key and not on where the block sits; one
    `[width, width + 2·reach]` table serves every block through a view. 20 KB
    instead of 256 MiB at 8192 tokens, which took the peak to 1.6 GiB — below
    the candle path's 2.0.
  - **Measuring the window as its reach.** `crate::attention` counts a window as
    how far either side a query sees, and this was handing it the whole
    `local_attention`. The band came out twice as wide as it needed to be — the
    extra keys were masked, so the answer was right and the work was not — and
    `banding_pays` declined below 514 tokens instead of 258. Worth 11.19 s to
    9.56 s at 460 tokens, where banding now applies at all.
  - **One mask instead of two, and the scale on the queries.** A sliding-window
    layer was adding padding and window separately, two broadcast adds over
    `[rows, heads, seq, seq]` in each of twelve layers; they are summed into the
    block's own shape first, which is one. Scaling `q` rather than the scores
    touches a seventh of the elements at 460 tokens.
  - The vectorised GeGLU this crate already owned, worth a further 2.6%.
- Past 2048 tokens the candle path pulls ahead again — 30.2 s against 33.5 s at
  8192, though on less memory rather than more — and that is where the remaining
  work is. The GEMM is not it: at the shape a fanned-out forward actually runs,
  burn-flex measured 138 GFLOP/s against candle's 133 single-threaded, and 627
  against 325 across all cores.

## `--device coreml` on the Apple Neural Engine

- Build with `--features coreml`.
- About 4× faster than Metal at 512 tokens, with cosine ≈ 0.99999 against the
  CPU output.
- For short inputs the multicore CPU backend may still be faster, because what
  the ANE gains per token it gives back padding each row to a fixed bucket
  length.
- Measured against PyTorch on the same machine (M2, `ruri-v3-130m`, the default
  buckets, median of three runs), from
  [`tools/benchmark.py`](../tools/benchmark.py):

| Input | kohagi (CPU) | kohagi (`--device coreml`) | torch (MPS) |
| --- | ---: | ---: | ---: |
| 1200 short (~30 tokens) | 7.1 s / 7.4 s | **4.0 s / 4.7 s** | 4.3 s / 13.7 s |
| 240 long (512 tokens) | 30.8 s / 31.5 s | **5.9 s / 6.6 s** | 15.2 s / 24.9 s |

- Encode / total, where total adds startup and model load.
- At 512 tokens the ANE encodes 2.6× faster than torch/MPS and 5.2× faster than
  Kohagi's own CPU path.
- The totals go further: torch spends 9 to 10 s importing and loading per process
  against Kohagi's under a second, so a rake task or a per-batch subprocess sees
  2.9× (short) and 3.8× (long).
- The ANE needs fixed input shapes, so this device runs a converted bundle
  rather than the safetensors the others read. See
  [CoreML bundles](coreml.md).

## Reranking

- `kohagi-rerank` takes the same `--device`.
- On the Neural Engine it is 3.4× the fastest PyTorch path.
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
