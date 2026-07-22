# Examples

- [`parity_check.py`](parity_check.py) — verify kohagi against the
  sentence-transformers / PyTorch reference.
- [`benchmark.py`](benchmark.py) — time kohagi against that same reference.
- [`rails_open3.rb`](rails_open3.rb) — drive kohagi's stdio protocol from
  Ruby/Rails, the pattern any language can copy.

---

## `parity_check.py` — matching PyTorch

kohagi's f32 output *is* the sentence-transformers output, to f32 rounding.
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
ruri-v3 ships as mean, so `--pooling cls` is what covers kohagi's CLS path.
Both were checked on ruri-v3-130m over 400 mixed-length texts:

| pooling | mean `1 - cosine` | worst |
|---|---:|---:|
| mean | 2.9e-13 | 1.7e-12 |
| cls | 6.6e-13 | 2.1e-12 |

Note that kohagi's own default is `--pooling mean` regardless of what the
checkpoint says, so a CLS model such as `Alibaba-NLP/gte-modernbert-base`
needs `--pooling cls` passed explicitly or the vectors will be wrong in a way
nothing warns you about.

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
between kohagi and PyTorch is roughly five orders of magnitude smaller than
what the storage format can hold — it disappears the moment you save it.

`--precision bf16` is a deliberate tradeoff and sits well above this, at
`1 - cosine ≈ 2e-5` (worst 9e-5). Still negligible for ranking, but it is a
real difference rather than rounding.

### Three settings that must match

Nearly every reported "mismatch" is one of these, not an implementation
difference. The script pins all three; if you compare by hand, do the same.

- **prefix**, exactly, trailing space included. `"検索文書:"` instead of
  `"検索文書: "` moves `1 - cosine` to ~`3e-3` — ten orders of magnitude above
  the real difference, and easy to cause with an unquoted shell variable.
- **max_seq_length**. kohagi defaults to 512; sentence-transformers uses
  whatever `sentence_bert_config.json` says, which is 8192 for ruri-v3. Any
  text between those limits gets truncated on one side only.
- **pooling**, against the model's `1_Pooling/config.json` — mean for ruri-v3
  and modernbert-embed. Note ruri-v3 sets `include_prompt: true`, meaning the
  prefix tokens participate in the mean, which is what kohagi does by
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

Both sides are pinned to kohagi's defaults — mean pooling, L2 normalize,
`max_seq_length` 512, batch size 64 — and torch runs in a fresh subprocess so
it pays interpreter startup the same way a real batch job would. Timing it
in-process would let Python's module cache serve the second
`import sentence_transformers` for free and hide about 3 seconds.

Useful flags: `--kind short|long`, `--count`, `--runs`, `--device cpu|mps|cuda`,
`--texts FILE`, `--skip-torch`.

### Measured on an Apple M2

8 cores, 16 GB, macOS 26.3, `ruri-v3-130m` f32, median of three runs. 1200
short texts (~60 tokens) or 240 long ones (truncated to 512).

| | kohagi | torch/cpu | torch/mps |
|---|---:|---:|---:|
| startup + model load | 0.3–2 s | 8.9 s | 9.2–9.7 s |
| encode, short | 10.2 s | 8.5 s | 4.4 s |
| encode, long | 12.0 s | 15.4 s | 7.7 s |
| **total, short** | 10.5 s | 17.4 s | 13.7 s |
| **total, long** | 13.7 s | 24.7 s | 17.5 s |

Absolute figures drifted by up to 30% across sessions on this laptop as it
warmed up, so treat the ratios as the result and the seconds as indicative.
Run it yourself before making a decision on it.

Two things the table says:

- **On CPU, compute is a wash** — within about 15% either way, direction
  depending on length. Accelerate and PyTorch are calling comparable sgemm,
  which is the expected outcome, not a surprising one.
- **The end-to-end win is startup, not throughput.** Torch spends ~9 seconds
  before embedding anything; kohagi spends well under one. That gap is what
  makes a per-invocation subprocess practical.

### Where kohagi loses

On MPS, torch computes long texts roughly 1.6× faster (7.7 s vs 12.0 s) — an
Apple GPU against candle's CPU path, since candle 0.10 has no Metal kernels
for ModernBERT's rotary embeddings. kohagi still wins end to end here, but
only because the 9-second load swamps it, and that lead is finite: at this
per-text gap the two cross over somewhere around 400–500 long texts, after
which a warm torch/mps process is simply faster.

So the honest framing is not "kohagi is faster than PyTorch". It is that
kohagi has nothing to amortize. If you spawn a process per batch — a rake
task, a cron job, a queue worker handling a few hundred records — that is the
number that matters. If you run a long-lived Python service that loads the
model once and embeds continuously, torch on MPS will out-throughput it.

---

## `rails_open3.rb` — calling kohagi from Ruby

Spawn the process, write `{"id","text"}` JSONL to stdin, read
`{"id","embedding"}` JSONL from stdout, and map results back by `id`.

The one structural requirement: **read stdout from a separate thread while
writing stdin.** kohagi emits results in chunks as it goes, so writing an
entire corpus before reading anything fills the pipe buffer and deadlocks
both processes. The example uses a writer thread plus a reader loop.

Exit codes matter too — `0` clean, `2` finished with skipped lines (the
output you did receive is valid; investigate stderr and resend those
records), `1` fatal. See [PROTOCOL.md](../PROTOCOL.md) for the full contract.
