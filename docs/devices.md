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
| `vulkan --precision f16` | `--features vulkan` | ~1.9× the bf16 CPU path | f16 encoder, cosine ≈ 0.99999 |
| `vulkan` (f32) | `--features vulkan` | slower than bf16 on CPU | f32, matches the CPU (worst `1 - cosine` 1.2e-12) |
| `cpu-burn` | `--features cpu-burn` | 1.5× `cpu` short and 0.85× long on x86_64; 1.1× short and 0.9× long on aarch64 | f32, matches it (worst `1 - cosine` 1.0e-12) |
| `metal-burn --precision f16` | `--features metal-burn` | level with `metal` in steady state; 0.5–0.8 s more per process | f16 encoder, cosine ≈ 0.9999997 |

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

### On aarch64 the GEMM *was* it

Everything above was measured on x86_64, where both engines put their sgemm
through the same `gemm` crate. macOS is not that comparison: `Cargo.toml` gives
candle the `accelerate` feature there, so `--device cpu` runs on Apple's AMX
coprocessor while burn-flex runs NEON. At `2048x512x2048` on an M1 Pro that is
2072 GFLOP/s against 395 — and candle without `accelerate` measures 521, which
is what says the gap is Accelerate and not burn-flex.

So on macOS the Burn engine now does what candle's `accelerate` feature does:
`src/burn_engine/flex.rs` hands its GEMMs to Accelerate directly, through the
same `extern "C"` and link line. Measured on a 6+2-core M1 Pro with
`ruri-v3-130m`, best of interleaved runs, wall clock, at worst `1 - cosine`
2.5e-11 against `--device cpu`:

| texts | `cpu` | `cpu-burn` as found | now |
| --- | ---: | ---: | ---: |
| 64 × 42 tokens | 0.81 s | 1.19 s | **0.74 s** |
| 64 × 460 tokens | **2.74 s** | 8.23 s | 2.86 s |
| 8 × 2048 tokens | **2.12 s** | 5.71 s | 2.39 s |
| 4 × 8192 tokens | **8.83 s** | 19.01 s | 9.42 s |

In order of what each was worth at 460 tokens:

- **`gemm`'s AMX kernels** (`apple-amx`), 8.24 s to 4.64 — the first step, and
  since removed: nothing on the macOS hot path goes through `gemm` any more.
- **The mask add on contiguous planes** (`FusedOps::add_mask`), 4.64 s to 3.73.
  burn-flex accelerates two broadcast shapes, both over the innermost axis, and
  the attention mask broadcasts over the *head* axis; everything else walks a
  scalar `StridedIter` over the widest tensor in the model. It now also takes
  the padding and window masks separately, so their sum is never built.
- **The projections on Accelerate** (`FusedOps::project`), 3.52 s to 3.13.
  The reason this is not a Burn feature is that burn-flex has none —
  `burn-ndarray` offers `blas-accelerate`, burn-flex does not — and the reason
  it is fair is that it is exactly what candle's `accelerate` feature is.
- **The attention products on Accelerate** (`FusedOps::scores`, `::context`),
  with the masks unsummed and four rows per unit instead of one, 3.06 s to
  2.86. On a 64-wide contraction Accelerate is 1.9× to 3.5× `gemm`'s AMX
  kernel, and a BLAS leading dimension reads the q/k/v views where they lie.
- **RoPE reading its input where it lies**, 3.73 s to 3.52.
- **Head split as a view, merge as `memcpy`s** (`FusedOps::split_heads`,
  `::merge_heads`), 3.13 s to 3.06. burn-flex's `reshape` copies any view that
  is not contiguous from offset zero, one element at a time; the split needs
  no copy at all, and the merge needs `head_dim`-wide ones.

What remains is not in the kernels. The two engines' profiles agree to within
a few percent on every compute symbol — the scalar `erf` under `gelu`
included, which is a floor under both on aarch64 and the largest item in
each — and the difference is user time inside Accelerate's own GCD threading
(18.2 s to 16.0 at 460 tokens). Binding `sgemm_` as candle does instead of
`cblas_sgemm`, the unit shape, and a dedicated fan-out pool were each tried
and measured within noise.

So on aarch64 `--device cpu-burn` is ahead of `--device cpu` at the default
`--max-seq-length` and 4% to 13% behind on long inputs; on x86_64 it is ahead
at every length below 8192.

## `--device metal-burn`: the Apple GPU, on Burn instead of candle

- **Transitional**, like `cpu-burn`: the two Metal paths compared in one
  binary. Build with `--features metal-burn` (and `metal` beside it to compare)
  and pair with `--precision f16`; `f32` is the exact path and 10–20% slower.
- The same engine as `vulkan` on the other GPU API — CubeCL emits Metal Shading
  Language for it directly — so the same precision recipe: f16 storage and
  matmuls, f32 reductions. Worst `1 - cosine` against `--device cpu` is 2.6e-7
  for f16 over 256 mixed-length texts and 2.3e-13 for f32.
- Measured on an M1 Pro, best of interleaved runs, wall clock:

| texts | `cpu` | `metal` (candle) | `metal-burn f16` | in-process, steady |
| --- | ---: | ---: | ---: | ---: |
| 64 × 42 tokens | 0.81 s | **0.51 s** | 1.3 s | 0.124 s vs 0.135 |
| 64 × 460 tokens | 2.71 s | **2.04 s** | 2.72 s | 1.57 s vs 1.67 |
| 256 × 460 tokens | 9.32 s | 7.07 s | **7.30 s** | — |
| 256 × mixed lengths | 6.0 s | **4.4 s** | 5.0 s | 4.23 s vs 4.30 |
| 8 × 2048 tokens | 2.14 s | **1.72 s** | 3.13 s | — |

- The last column is the library called twice in one process on the same
  texts, and it is the finding: **in steady state the Burn path is level with
  candle's or slightly ahead**, without the hand-fused GeGLU kernel candle's
  Metal path carries. Everything the CLI loses is per-process: about 0.45 s
  more to load, and about half a second per distinct padded length while
  CubeCL generates and compiles that shape's kernels. Apple caches compiled
  Metal per executable, so a freshly built binary's first run is seconds
  slower again, for candle too (4.5 s to load, 7.7 s for its first forward).
- The first time a shape is ever seen it is also autotuned — 17 s for the 256
  mixed-length texts — and the result is cached under the user's cache
  directory (`~/Library/Caches/kohagi` on macOS), so that is paid once per
  machine, not per run.
- `burn/fusion` is off here, unlike on Vulkan: measured, it changed nothing in
  steady state and added per-shape compile time.
- One precision per process, as on Vulkan; see `src/burn_engine/wgpu.rs`.

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
