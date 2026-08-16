# Kohagi rerank stdio protocol (v1)

`kohagi-rerank` is a pure function from id-tagged `(query, text)` pairs to
id-tagged scores, spoken over stdin/stdout as JSONL. It is a separate binary
from `kohagi` because it is a different function — pairs in, numbers out — but
every rule around the records is the one [PROTOCOL.md](PROTOCOL.md) states, and
this document only says where the two differ.

Where an embedding model reads the query and the document apart and compares
them with a dot product, a **cross-encoder** reads them together, so every
layer can attend from one to the other. That costs a forward pass per pair
rather than per document, which is why it reorders a retrieved shortlist
instead of searching a corpus.

## Input (stdin, JSONL, UTF-8, one record per line)

```json
{"id": 123, "query": "配列を並べ替えるには", "text": "sort は…"}
```

| field | type | notes |
|---|---|---|
| `id` | any JSON value | **Opaque.** Echoed verbatim in the output; never interpreted. |
| `query` | string | What was asked. Raw text, no prefix — a reranker takes none, unlike Ruri's embedding prefixes. |
| `text` | string | The candidate being scored against it. |

- **Order matters.** `(query, text)`, not `(text, query)`: a cross-encoder is
  not symmetric, and swapping them returns a different number rather than an
  error.
- **Skips (non-fatal).** A line is skipped — with `kohagi-rerank: skip line N:
  <reason>` on stderr and a count in the summary — when it is not valid JSON,
  not a JSON object, has no `id`, or has a missing / empty / non-string `query`
  or `text`. A pair missing half of itself is skipped rather than scored
  against nothing, because scoring it would produce a plausible number.
- **Blank lines** end a batch, exactly as in the embedding protocol: everything
  buffered is scored and answered, and the reply ends with a blank line of its
  own.

## Output (stdout, JSONL, one record per line)

```json
{"id": 123, "score": 0.9283465}
```

- `id` is the input value, unchanged. **Map by id, not by order.**
- `score` is the sigmoid of the model's logit — a number in 0..1, higher means
  more relevant. This matches `sentence_transformers.CrossEncoder.predict` for
  a one-label model, which is what these rerankers are, so a threshold tuned
  against that library carries over unchanged.
- **`--raw-logits`** reports the logit instead, unsquashed. The ranking is
  identical (sigmoid is monotonic); what changes is what a threshold means.
- **`--report-tokens`** adds `n_tokens` and `truncated`, as in the embedding
  protocol. Here they describe the pair as a whole: `n_tokens` counts both
  halves plus the four special tokens joining them, and `truncated` says the
  pair ran past `--max-seq-length` so the longer half lost its tail.

## Scores are comparable within a model, not across models

A score is a number this model assigns this pair; it is not a probability of
anything and not a quantity two models agree on. Ranking candidates for one
query is what it is for. Comparing 0.7 from one reranker with 0.7 from another
is not meaningful, and neither is comparing scores across queries unless you
have checked that they behave that way on your own data.

## Which models this speaks to

Any ModernBERT sequence-classification checkpoint with one label:
`cl-nagoya/ruri-v3-reranker-310m` (the default) and the
`hotchpotch/japanese-reranker-{tiny,xsmall,small,base}-v2` family are all this
shape. The head is `norm(gelu(dense(h_cls)))` into a single linear output, and
the pair is joined by the tokenizer's own template — `<s> query </s> <s> text
</s>` for these checkpoints — rather than by string concatenation here.

Kohagi refuses rather than guesses when a checkpoint is something else: a model
with more than one label is an error (one number per pair is the whole
contract), and an encoder with no classification head fails at load, naming
what it is missing.

## Summary and exit codes

```
kohagi-rerank: model=ruri-v3-reranker-310m sha256=1c342581efc2 pooling=cls dim=768 max_seq=512 score=sigmoid in=2141 out=2141 skipped=0 truncated=3
```

`score=sigmoid` or `score=logit` says which of the two the run produced, since
a log of numbers cannot be read without it. `--print-model-info` prints the
same facts as one JSON line, as in [PROTOCOL.md](PROTOCOL.md), with `score`
added. Exit codes are the same: 0 every pair scored, 2 finished with skipped
lines, 1 fatal.

## Devices

`--device cpu`, `--device metal` (`--features metal`), `--device cuda`
(`--features cuda`), and `--device coreml` (`--features coreml`), which is the
fastest by a wide margin — 18.5 pairs/s against 3.3 on the CPU for
`ruri-v3-reranker-310m` on an M2.

The Neural Engine runs fixed shapes and fp16, so it needs a converted bundle.
`kohagi-rerank --device coreml` converts `--model-id` itself on first use and
caches the result, or takes one with `--coreml-dir` / `--coreml-model-id`. A
bundle for reranking carries `head.safetensors` beside the buckets: the graph
is the encoder, so the four small head tensors are written next to it and
loaded in f32 on the CPU. **Only the encoder is fp16.**

That still moves the scores, and by more than any other backend does.

### What fp16 costs a threshold

A score is `sigmoid(logit)`, and the backend's error is in the logit. The
sigmoid then compresses that error by `s(1-s)`: the same logit error `d` moves
a score near 0.02 by `d * 0.0196` and a score near 0.5 by `d * 0.25`, **12.7
times more**. So a worst-case score difference cannot be quoted on its own —
without saying where in the range it happened, it says nothing about a
threshold anywhere else.

Measuring `d` instead does transfer. For any threshold `t`, a pair can only
cross it if its score lies within

```
±  d · t · (1 - t)
```

which is derived rather than sampled, and so survives a change of threshold.
`tools/rerank_parity.py --device coreml --against cpu` reports `d` and the
bands; `tools/rerank_fp16_bands.py` samples from them and counts what actually
crosses.

**The logit error**, over 600 real training pairs stratified across the score
range (`--device coreml` against `--device cpu`, M2, 512 tokens):

| model | layers | mean | p99 | max |
|---|---|---|---|---|
| `japanese-reranker-xsmall-v2` | 10 | 0.0075 | 0.0267 | 0.0578 |
| `ruri-v3-reranker-310m` | 25 | 0.0131 | 0.0424 | 0.0558 |

`d` is a property of the encoder, not of where the score landed: split by score
region it stays between 0.0059 and 0.0109 for the first model and between
0.0122 and 0.0147 for the second. That is what lets one measurement cover every
threshold. Depth costs mean error (25 layers accumulate 1.7× what 10 do) but
not worst case.

**The bands that follow**, from the worst case `d = 0.058`, with how much of a
116,640-pair corpus scored by `xsmall-v2` sits inside each:

| threshold | flip band | pairs in it |
|---|---|---|
| 0.02 | ±0.00113 | 523 (0.45%) |
| 0.1 | ±0.00520 | 1,220 (1.05%) |
| 0.5 | ±0.01444 | 924 (0.79%) |
| 0.6 | ±0.01386 | 915 (0.78%) |

**What actually crossed**, sampling 200 pairs from each ±0.01 band of that
corpus and scoring them on both backends:

| threshold | band population | sampled | crossed | rate | implied over the band |
|---|---|---|---|---|---|
| 0.02 | 5,108 | 200 | 0 | 0% | 0 (≤77 at 95%, by the rule of three) |
| 0.1 | 2,317 | 200 | 4 | 2.0% | 46 |
| 0.5 | 636 | 200 | 24 | 12.0% | 76 |
| 0.6 | 651 | 200 | 26 | 13.0% | 85 |

The rates line up with `t(1-t)` — 0.0196, 0.09, 0.25, 0.24 against 0%, 2%, 12%,
13% — which is the compression argument showing up in the data rather than
being assumed.

**So the exposure is smallest where the stakes are highest.** A low threshold
sits in the flat tail of the sigmoid, and low thresholds are the consequential
ones: a cutoff that decides whether a training pair survives lives at 0.02,
where nothing crossed. The thresholds that do move are near 0.5, where a
threshold is usually a reporting band rather than a gate, and where a pair
landing on either side of 0.501 was never a meaningful distinction.

This is a statement about *these* backends and *this* corpus, not a licence.
For a new threshold, take `d = 0.058`, compute `d · t · (1 - t)`, and count how
much of your own data is inside it — that is the upper bound on what can move,
before any measurement. **A threshold tuned to three decimal places is still
not automatically safe**, and scores from two backends should not be mixed in
one comparison: treat them as different scales, as with a quantized embedding
bundle.

Ranking is a separate question and a smaller one: over 120 pairs across 8
queries, 4 of 840 within-query orderings changed and no query's top-1 did.

A bundle converted before Kohagi 0.6 has no `head.safetensors`. Loading one for
reranking fails at load saying so, rather than scoring with something else.
