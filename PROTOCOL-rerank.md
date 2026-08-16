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

That still moves the scores, and by more than any other backend does. Measured
on `ruri-v3-reranker-310m` over 120 pairs spanning 0.0001 to 0.9995:

| | mean | worst |
|---|---|---|
| `metal` against `cpu` | 3.9e-08 | 1.8e-06 |
| `coreml` against `cpu` | 1.1e-04 | 4.9e-03 |

No pair crossed 0.02, 0.1, 0.5 or 0.6, and 4 of 840 within-query orderings
changed with no top-1 changing. So ranking survives, and a threshold in the
middle of the range survives; a threshold you have tuned to three decimals does
not automatically. If scores from the two backends are being compared with each
other — one index scored on the ANE, another on the CPU — treat them as
different scales and re-check the cutoff, exactly as with a quantized
embedding bundle.

A bundle converted before Kohagi 0.6 has no `head.safetensors`. Loading one for
reranking fails at load saying so, rather than scoring with something else.
