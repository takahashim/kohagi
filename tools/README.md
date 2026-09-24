# Tools

For measuring and verifying Kohagi, rather than demonstrating how to call it
(that is [`examples/`](../examples/)). Outside the Cargo workspace and not part
of the published crate.

- [`parity_check.py`](parity_check.py) verifies Kohagi against the
  sentence-transformers / PyTorch reference.
- [`benchmark.py`](benchmark.py) times Kohagi against that same reference.
- [`model_check.py`](model_check.py) smoke-tests Kohagi against any other
  ModernBERT sentence encoder on the Hub.
- [`scaling_check.py`](scaling_check.py) measures how throughput and memory
  move with input length and batch size.
- [`eval_retrieval.py`](eval_retrieval.py) measures JaCWIR and JQaRA retrieval
  quality for any Kohagi configuration, which is what the quantization numbers
  rest on.
- [`dataset/`](dataset/) turns `(query, text)` pairs into a distillation
  dataset: the token ids the student will read, and the teacher's score to be
  pulled towards. One pass, because Kohagi is the one place holding both the
  tokenizer and the teacher; made once, because a student is trained many
  times and a 310M forward per step per run is paying for the same answer
  repeatedly.
- [`reference/`](reference/) writes Kohagi's forward pass down as a JSON
  artifact — token ids and the pooled, normalized vector — so another
  implementation of the same architecture can be held to it. Torobi (the MLX
  training framework) uses it to check that the ModernBERT it describes in its
  own IR computes what Kohagi computes. Rust, no Python, and the settings it
  records are read off the loaded model rather than written down in the jig.
- [`decide_eval.py`](decide_eval.py) measures a typed-decision task the way
  `notes/plan-decide.md` §6 designs it: Top-1 / MRR / NLL / ECE per split, for
  the Jev candidate-softmax approach against the embedding zero-shot baseline
  and the majority floor. It needs an evaluation set, which the same note says
  is the open prerequisite — the script is the shape the decision fixed, made
  runnable.
- [`decide_latency.py`](decide_latency.py) measures the two costs of the Jev
  candidate path — the model load a process pays once, and the per-pair forward
  it pays per candidate — reporting wall and child CPU side by side, so a loaded
  machine's contention is visible rather than folded into the seconds. CPU and
  Neural Engine (`--device coreml`).
- [`coreml-jigs/`](coreml-jigs/) holds Rust jigs for the CoreML backend, which
  read a converted bundle, check Neural Engine placement, measure per-bucket
  latency, and compare two configurations. See its own README.

The Python here needs `pip install sentence-transformers torch` (and `pyarrow`
for `eval_retrieval.py`); Kohagi itself never does.

---

## Matching PyTorch (`parity_check.py`)

Kohagi's f32 output *is* the sentence-transformers output, to f32 rounding.
That is the whole premise of using it instead of a Python service, so it is
worth checking on your own machine and your own texts rather than taking it
on faith.

```bash
pip install sentence-transformers
python tools/parity_check.py --kohagi ./target/release/kohagi
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
which is the only way to exercise a mode the model was not published with.
Since ruri-v3 ships as mean, `--pooling cls` is what covers Kohagi's CLS path.
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
| 12 texts swept from 15 to 645 tokens | n/a | below f32 resolution at every length |

For scale, a `vector(512)` column in pgvector stores float4, which cannot
represent a difference below ~1e-7 in the first place. The disagreement
between Kohagi and PyTorch is roughly five orders of magnitude smaller than
what the storage format can hold. It disappears the moment you save it.

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
  `"検索文書: "` moves `1 - cosine` to ~`3e-3`, ten orders of magnitude above
  the real difference, and an unquoted shell variable makes the mistake easy.
- **max_seq_length**. Kohagi defaults to 512; sentence-transformers uses
  whatever `sentence_bert_config.json` says, which is 8192 for ruri-v3. Any
  text between those limits gets truncated on one side only.
- **pooling**, against the model's `1_Pooling/config.json` (mean for ruri-v3
  and modernbert-embed). Note ruri-v3 sets `include_prompt: true`, meaning the
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

The float32 figure measures the accumulator, not the vectors. It is roughly
`f32::EPSILON`, and you will get it for *any* sufficiently close pair. A
comparison that reports `0.9999998` has usually hit this floor rather than
found a discrepancy. Cast to float64 before the dot product (the script
does).

The elementwise maximum difference does not have this problem (subtracting
two nearby floats is exact), so it is the more informative number when you
only have f32 tooling.

---

## How fast, and against what (`benchmark.py`)

```bash
pip install sentence-transformers
python tools/benchmark.py --kohagi ./target/release/kohagi --kind long
```

```console
                   load   encode    total
kohagi            1.73s   12.00s   13.73s
torch/mps         9.74s    7.65s   17.51s
```

Both sides are pinned to Kohagi's defaults (mean pooling, L2 normalize,
`max_seq_length` 512, batch size 64), and torch runs in a fresh subprocess so
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

- **On CPU, compute is a wash.** Kohagi is within about 15% of torch/cpu
  either way. Accelerate and PyTorch call comparable sgemm, which is the
  expected outcome, not a surprising one.
- **The end-to-end win is startup.** torch spends several seconds loading
  before embedding anything; Kohagi spends well under one. That gap is what
  makes a per-invocation subprocess practical, and it is why Kohagi ties
  torch/mps on short totals despite the GPU being twice as fast at the encode.

### Where Kohagi loses

At the encode itself, torch/mps is roughly twice as fast, because it runs on
the Apple GPU while the table's `kohagi` column is the CPU. On long totals it already
comes out ahead (28.6 s vs 32.3 s), because at 512 tokens the compute gap
outgrows Kohagi's startup lead. A `--features metal` build closes most of that
(see the top-level README), but a warm, long-lived torch/mps service still
out-throughputs a process spawned per batch once the corpus is large enough.

So the honest framing is not "Kohagi is faster than PyTorch". It is that
Kohagi has nothing to amortize. If you spawn a process per batch (a rake
task, a cron job, a queue worker handling a few hundred records), that is the
number that matters. If you run a long-lived Python service that loads the
model once and embeds continuously, torch on MPS will out-throughput it.

---

## Does another model work? (`model_check.py`)

Kohagi runs any ModernBERT encoder that ships a fast `tokenizer.json` and a
`1_Pooling/config.json`, not just ruri-v3. This script points it at one and
checks the embeddings are usable. A retrieval model returns plausible floats
no matter what, so "it exited 0" proves nothing.

```bash
python tools/model_check.py --kohagi ./target/release/kohagi \
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
Kohagi will autodetect, and to flag a model that ships none, which is usually
a reranker or a base LM rather than a sentence encoder. Requires no Python
packages; it shells out to the binary and does the arithmetic in the standard
library. The built-in corpus is English, so for a non-English model pass
`--prefix-doc` / `--prefix-query` if it expects task prefixes and treat the
retrieval line as a smoke test, not a benchmark.

---

## Where the CPU path stops scaling (`scaling_check.py`)

Kohagi fans length-bucketed batches across a rayon pool sized to physical
cores, and on an 8-core M2 that measured only ~2× over serial. This sweeps
worker count and batch size independently, since they are different mechanisms,
and checks the machine is quiet enough to resolve the answer before believing
it.

```bash
python tools/scaling_check.py --kohagi ./target/release/kohagi
```

---

## Does a change cost retrieval quality? (`eval_retrieval.py`)

MAP@10 and HIT@10 on JaCWIR, nDCG@10 on JQaRA, for any Kohagi configuration.
Everything after `--` is passed through, so a device, a model or a quantization
level can be measured without editing the script. This is what the claim that
an int8 embedding table costs nothing rests on, kept here so a later change can
be checked against the same metric rather than against a remembered figure.

```bash
python3 tools/eval_retrieval.py --benchmark jacwir -- --device cpu
python3 tools/eval_retrieval.py --benchmark jacwir -- --device coreml --coreml-quantize embeddings
```

Recorded for `cl-nagoya/ruri-v3-130m` at `--max-seq-length 512`, float32 on CPU:
JaCWIR MAP@10 0.8584, HIT@10 0.9696; JQaRA nDCG@10 0.7106. A full run is long
(JaCWIR's 5000 queries name about 497k distinct documents), so `--limit` takes
a prefix of the queries, which is enough to tell the method matches.

Needs `pip install pyarrow` and network access on the first run; the datasets
are cached and not committed.

---

## Is a typed decision good enough? (`decide_eval.py`)

Whether `kohagi` should grow a typed-decision binary is decided by measurement,
and `notes/plan-decide.md` §6 fixes what that measurement is before the numbers
exist. This script is that design made runnable: it reads a JSONL evaluation
set, scores it, and reports Top-1, MRR, NLL and ECE per split for three systems
— the majority floor, the embedding zero-shot baseline, and the Jev
candidate-softmax approach (`argos1111/modernbert-ja-310m-jev` via
`kohagi-rerank --raw-logits`, softmaxed outside kohagi).

```bash
python3 tools/decide_eval.py --data eval.jsonl \
    --kohagi ./target/release/kohagi \
    --kohagi-rerank ./target/release/kohagi-rerank \
    --embed-model cl-nagoya/ruri-v3-310m \
    --embed-prefix-query "検索クエリ: " --embed-prefix-doc "検索文書: "
```

An input record is `{"id", "split", "question", "state", "candidates", "gold"}`
(see the module docstring for `query`, index golds and the `long` / `unknown`
splits). `--dump` writes every case's per-candidate scores and the gold rank,
which separates a "learned the wrong label system" failure from a "read the
state wrongly" one. `--self-test` runs the model card's own example through the
real binaries, so the plumbing can be checked before any data exists.

Two controls run by default (`--no-controls` to skip), both borrowed from the
open-implementation survey in plan-decide §6. The **wrong-context** control
scores each case against the next case's state: on the 16-item customer-support
set the Jev Top-1 falls from 0.9375 to 0.3125, which is what "it reads the
state" looks like. The **option-order** controls reverse and permute the menu
and compare by label: they report a flip rate, the mean total-variation
distance, and whether flips sit at low confidence. For the Jev
candidate-softmax path the flip rate is 0 by construction — each candidate is
scored independently, so the menu's order cannot matter; the control is there to
catch a packed-sequence model (Laya, GLiClass), where it can.

The embedding baseline has no NLL/ECE: a cosine is not a probability. The
hand-tuned LLM row §6 lists as an upper bound is not here either — it belongs to
whatever LLM you own, not to kohagi.

For a provisional set, the reference implementation `Argos1111/jev_local` ships
held-out tasks in `tools/evaluate_heldout_tasks.py`: a hand-written 16-item
customer-support set (noul / choice / score, with gold) and generators for
livedoor 9-way news and JMMLU. Run through this script, the 16 items reproduce
the model card's 93.8% for the Jev backend (15/16), against 0.75 for the
ruri-v3-130m embedding zero-shot and 0.25 for majority. **That repository has no
license** (`license: null`), so its data is not vendored here; use it locally, or
substitute a clearly-licensed set (JGLUE test, CC BY-SA 4.0; MASSIVE, CC BY 4.0).

`build_decide_eval_data.py` supplies a reproducible, local-only provisional
set from MASSIVE 1.1 `ja-JP` test data. It does **not** commit or redistribute
the examples: it downloads the official archive into `.cache/`, writes JSONL
and a SHA-256 manifest there, and fixes the seed (`122`). The `main` split uses
the 18 scenario labels that the Jev checkpoint learned from MASSIVE train;
`unknown` uses its finer intent taxonomy (59 present in ja-JP test), which it did
not learn to choose;
and `long` appends a neutral audit tail past the 512-token state window. The
last is a truncation stress test, not evidence about naturally long workflows.

```bash
python3 tools/build_decide_eval_data.py \
    --out .cache/decide-eval/massive-ja-test.jsonl
python3 tools/decide_eval.py --data .cache/decide-eval/massive-ja-test.jsonl \
    --kohagi ./target/release/kohagi \
    --kohagi-rerank ./target/release/kohagi-rerank
```

Do not tune a threshold, prompt, or candidate wording on this test set. A
permitted, anonymized real routing set is still needed before treating the
long split as product evidence.

`--rerank-arg` / `--embed-arg` pass flags straight through to the two binaries,
so the same evaluation runs on the Neural Engine:

```bash
python3 tools/decide_eval.py --data .cache/decide-eval/massive-ja-test.jsonl \
    --kohagi ./target/release/kohagi --kohagi-rerank ./target/release/kohagi-rerank \
    --embed-arg=--device --embed-arg=coreml --embed-arg=--coreml-dir --embed-arg="$RURI" \
    --rerank-arg=--device --rerank-arg=coreml --rerank-arg=--coreml-dir --rerank-arg="$ARGOS"
```

Measured on an M2 (60 / 60 / 30 cases, Neural Engine for both binaries,
`ruri-v3-130m` embedding baseline with `検索クエリ:` / `検索文書:` prefixes):
the Jev model beats the embedding zero-shot on all three splits, and the
wrong-context control collapses everywhere, so it is reading the state.

| split | majority | embed | Jev | Jev ECE | shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| main (18 scenarios) | 0.233 | 0.350 | **0.967** | 0.041 | 0.133 |
| unknown (59 intents) | 0.083 | 0.333 | **0.667** | 0.150 | 0.017 |
| long (audit tail) | 0.167 | 0.233 | **0.733** | 0.380 | 0.000 |

Two caveats the table makes visible: the unknown split is a proxy (MASSIVE text,
a novel label system), and the long split's calibration (ECE 0.380) is far worse
than its ranking (0.733) — a long irrelevant tail distracts the probabilities
even though the task signal itself is not truncated away.

---

## How fast is a typed decision? (`decide_latency.py`)

The Jev candidate path pays two costs: a model load once per process, and a
per-pair forward per candidate. This measures both, on the CPU and on the
Neural Engine, and reports wall and child CPU side by side. The ratio is the
point: on a quiet machine `cpu/wall ≈ 1`, and on a loaded one it falls, which is
how a machine's own noise shows up as spread instead of as a wrong number.

```bash
python3 tools/decide_latency.py --device cpu \
    --kohagi-rerank ./target/release/kohagi-rerank

cargo build --release --features coreml,coreml-export --bin kohagi-rerank
python3 tools/decide_latency.py --device coreml \
    --kohagi-rerank ./target/release/kohagi-rerank \
    --coreml-dir ~/Library/Caches/kohagi/coreml/converted/.../<bundle>
```

`--ks` sweeps the candidate count, `--length short|long` picks a ~50-token or
512-token state, and `--reps` interleaves repeats so a transient load spike hits
every K alike. Per-question is per-pair × K: the CoreML path is batch=1 per pair,
so its cost is linear in candidates. Measured on a loaded M2, short state:
CPU 60–140 ms/pair, CoreML 10–15 ms/pair; 512 tokens: CPU ~2.2 s, CoreML ~75 ms.
The load is once per process (CPU ~2 s, CoreML ~0.9 s), so a long-lived caller
avoids it.
