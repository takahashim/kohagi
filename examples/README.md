# Examples

- [`parity_check.py`](parity_check.py) — verify Kohagi against the
  sentence-transformers / PyTorch reference.
- [`benchmark.py`](benchmark.py) — time Kohagi against that same reference.
- [`model_check.py`](model_check.py) — smoke-test Kohagi against any other
  ModernBERT sentence encoder on the Hub.
- [`rails_open3.rb`](rails_open3.rb) — drive Kohagi's stdio protocol from
  Ruby/Rails, the pattern any language can copy.

---

## `parity_check.py` — matching PyTorch

Kohagi's f32 output *is* the sentence-transformers output, to f32 rounding.
That is the whole premise of using it instead of a Python service, so it is
worth checking on your own machine and your own texts rather than taking it
on faith.

```bash
pip install sentence-transformers
python examples/parity_check.py --kohagi ./target/release/kohagi
```

```console
model      : cl-nagoya/ruri-v3-130m (512 dims, f32)
texts      : 5
mean 1-cos : 1.387e-12
worst 1-cos: 5.760e-12
max |diff| : 5.993e-07

OK: worst 1-cos 5.760e-12 < 1e-09
```

It exits non-zero on failure (threshold `1e-9` for f32, `1e-3` for bf16), so
it works as a regression test. Useful flags: `--model-id`, `--prefix`,
`--max-seq-length`, `--precision`, `--pooling`, and `--texts FILE` to use your
own corpus (one text per line).

`--pooling` defaults to `model`, meaning each side uses the checkpoint's own
`1_Pooling` config. Passing `mean` or `cls` forces that mode on both sides,
which is the only way to exercise a mode the model was not published with —
ruri-v3 ships as mean, so `--pooling cls` is what covers Kohagi's CLS path.
Both were checked on ruri-v3-130m over 400 mixed-length texts:

| pooling | mean `1 - cosine` | worst |
|---|---:|---:|
| mean | 2.9e-13 | 1.7e-12 |
| cls | 6.6e-13 | 2.1e-12 |

Kohagi does the same by default: with no `--pooling`, it reads the model's
`1_Pooling/config.json` and uses what the checkpoint declares, so a CLS model
such as `Alibaba-NLP/gte-modernbert-base` works without a flag. Passing
`--pooling` forces a mode and warns if it disagrees with the checkpoint; a
model that ships no `1_Pooling` (a reranker, a base LM) falls back to mean
with a warning that it may not be a sentence encoder.

### What "matching" means

Not bit-identical, and it never will be: candle uses a pure-Rust `gemm` (or
Apple's Accelerate on macOS) while PyTorch uses MKL. Different matrix-multiply
implementations sum in different orders, so the last bits differ. What holds
is that the difference stays at f32 rounding level.

Measured on ruri-v3-130m against `sentence-transformers` on CPU (Linux,
8-core Zen 4):

| texts | mean `1 - cosine` | worst |
|---|---:|---:|
| 1200 short (~60 tokens) | 2e-13 | 9e-12 |
| 240 long (512 tokens) | 3e-12 | 2e-11 |
| 12 texts swept from 15 to 645 tokens | — | below f32 resolution at every length |

For scale, a `vector(512)` column in pgvector stores float4, which cannot
represent a difference below ~1e-7 in the first place. The disagreement
between Kohagi and PyTorch is roughly five orders of magnitude smaller than
what the storage format can hold — it disappears the moment you save it.

`--precision bf16` is a deliberate tradeoff and sits well above this, at
`1 - cosine ≈ 2e-6` (worst 2e-5). Still negligible for ranking, but it is a
real difference rather than rounding.

Measured against Kohagi's own f32 output over 120 short and 120 long texts,
which is the same comparison to within the `1e-12` that separates Kohagi's f32
from PyTorch. The figure used to be an order of magnitude larger; fusing the
mask into the softmax removed a rounding step that the separate `broadcast_add`
had been introducing.

### Three settings that must match

Nearly every reported "mismatch" is one of these, not an implementation
difference. The script pins all three; if you compare by hand, do the same.

- **prefix**, exactly, trailing space included. `"検索文書:"` instead of
  `"検索文書: "` moves `1 - cosine` to ~`3e-3` — ten orders of magnitude above
  the real difference, and easy to cause with an unquoted shell variable.
- **max_seq_length**. Kohagi defaults to 512; sentence-transformers uses
  whatever `sentence_bert_config.json` says, which is 8192 for ruri-v3. Any
  text between those limits gets truncated on one side only.
- **pooling**, against the model's `1_Pooling/config.json` — mean for ruri-v3
  and modernbert-embed. Note ruri-v3 sets `include_prompt: true`, meaning the
  prefix tokens participate in the mean, which is what Kohagi does by
  prepending the prefix as ordinary text.

Also worth knowing: ruri-v3 defines no built-in prompts
(`config_sentence_transformers.json` has `"prompts": {}`), so the task prefix
is the caller's job on both sides. There is no hidden prompt handling to
account for.

### Accumulate the cosine in float64

Two f32 vectors this close saturate f32 arithmetic. The same pair of outputs
scores:

| accumulator | `1 - cosine` |
|---|---:|
| float32 | 1.19e-07 |
| float64 | 8.71e-12 |

The float32 figure measures the accumulator, not the vectors — it is roughly
`f32::EPSILON` and you will get it for *any* sufficiently close pair. A
comparison that reports `0.9999998` has usually hit this floor rather than
found a discrepancy. Cast to float64 before the dot product (the script
does).

The elementwise maximum difference does not have this problem — subtracting
two nearby floats is exact — so it is the more informative number when you
only have f32 tooling.

---

## `benchmark.py` — how fast, and against what

```bash
pip install sentence-transformers
python examples/benchmark.py --kohagi ./target/release/kohagi --kind long
```

```console
                   load   encode    total
kohagi            1.73s   12.00s   13.73s
torch/mps         9.74s    7.65s   17.51s
```

Both sides are pinned to Kohagi's defaults — mean pooling, L2 normalize,
`max_seq_length` 512, batch size 64 — and torch runs in a fresh subprocess so
it pays interpreter startup the same way a real batch job would. Timing it
in-process would let Python's module cache serve the second
`import sentence_transformers` for free and hide about 3 seconds.

Useful flags: `--kind short|long`, `--count`, `--runs`, `--device cpu|mps|cuda`,
`--texts FILE`, `--skip-torch`.

### Measured on an Apple M2

8 cores, 16 GB, macOS 26.3, `ruri-v3-130m` f32 on the CPU, median of three
runs. 1200 short texts (~30 tokens) or 240 long ones that fill the 512-token
window. Kohagi here is the default CPU backend, not `--device metal`.

| | kohagi | torch/cpu | torch/mps |
|---|---:|---:|---:|
| startup + model load | 0.3–0.9 s | 3–4 s | 3–5 s |
| encode, short | 7.2 s | 8.9 s | 4.2 s |
| encode, long | 31.5 s | 36.9 s | 24.3 s |
| **total, short** | 7.5 s | 11.7 s | 7.4 s |
| **total, long** | 32.3 s | 40.7 s | 28.6 s |

Absolute figures drift substantially across sessions on this laptop as it
warms up, and torch's load time depends on whether the model is already in the
OS file cache. Treat the ratios as the result and the seconds as indicative,
and run it yourself before making a decision on it.

Two things the table says:

- **On CPU, compute is a wash** — Kohagi is within about 15% of torch/cpu
  either way. Accelerate and PyTorch call comparable sgemm, which is the
  expected outcome, not a surprising one.
- **The end-to-end win is startup.** torch spends several seconds loading
  before embedding anything; Kohagi spends well under one. That gap is what
  makes a per-invocation subprocess practical, and it is why Kohagi ties
  torch/mps on short totals despite the GPU being twice as fast at the encode.

### Where Kohagi loses

At the encode itself, torch/mps is roughly twice as fast — it runs on the
Apple GPU while the table's `kohagi` column is the CPU. On long totals it already
comes out ahead (28.6 s vs 32.3 s), because at 512 tokens the compute gap
outgrows Kohagi's startup lead. A `--features metal` build closes most of that
(see the top-level README), but a warm, long-lived torch/mps service still
out-throughputs a process spawned per batch once the corpus is large enough.

So the honest framing is not "Kohagi is faster than PyTorch". It is that
Kohagi has nothing to amortize. If you spawn a process per batch — a rake
task, a cron job, a queue worker handling a few hundred records — that is the
number that matters. If you run a long-lived Python service that loads the
model once and embeds continuously, torch on MPS will out-throughput it.

---

## `model_check.py` — does another model work?

Kohagi runs any ModernBERT encoder that ships a fast `tokenizer.json` and a
`1_Pooling/config.json`, not just ruri-v3. This script points it at one and
checks the embeddings are usable — a retrieval model returns plausible floats
no matter what, so "it exited 0" proves nothing.

```bash
python examples/model_check.py --kohagi ./target/release/kohagi \
    Alibaba-NLP/gte-modernbert-base
```

```console
model    : Alibaba-NLP/gte-modernbert-base
pooling  : cls   (Kohagi autodetects; no flag needed)
dims     : 768
retrieval: 4/4 correct, smallest margin over runner-up +0.240
paraphr. : within 0.881-0.917  across 0.425-0.444  [OK]
bf16     : vs f32  worst 1-cos 6.71e-05

OK: retrieval and paraphrase structure both hold
```

It reads the checkpoint's own `1_Pooling/config.json` to report the pooling
Kohagi will autodetect — and to flag a model that ships none, which is usually
a reranker or a base LM rather than a sentence encoder. Requires no Python
packages; it shells out to the binary and does the arithmetic in the standard
library. The built-in corpus is English, so for a non-English model pass
`--prefix-doc` / `--prefix-query` if it expects task prefixes and treat the
retrieval line as a smoke test, not a benchmark.

---

## `rails_open3.rb` — calling Kohagi from Ruby

Spawn the process, write `{"id","text"}` JSONL to stdin, read
`{"id","embedding"}` JSONL from stdout, and map results back by `id`.

The one structural requirement: **read stdout from a separate thread while
writing stdin.** Kohagi emits results in chunks as it goes, so writing an
entire corpus before reading anything fills the pipe buffer and deadlocks
both processes. The example uses a writer thread plus a reader loop.

Exit codes matter too — `0` clean, `2` finished with skipped lines (the
output you did receive is valid; investigate stderr and resend those
records), `1` fatal. See [PROTOCOL.md](../PROTOCOL.md) for the full contract.
