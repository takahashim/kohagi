# Retrieval and reranking

- `kohagi-rerank` reorders candidates; it does not find them.
- It reads the query and the document together rather than apart, which orders
  better than comparing two vectors but costs one forward pass per pair, so it
  runs over tens of candidates, not over a corpus.

```
query ──> kohagi (検索クエリ: …) ──> vector ──┐
                                             ├──> your vector store ──> top 100 candidates
corpus ─> kohagi (検索文書: …) ──> vectors ───┘                              │
                                                                            v
                                          kohagi-rerank ──> scores ──> top 10
```

- Kohagi supplies the vectors and the scores. The search between them is your
  store's job: pgvector, Qdrant, an in-process index, whatever you already have.

## Scoring the candidates

- `kohagi-rerank` reads `{"id","query","text"}` and writes `{"id","score"}`.
- The `id` is echoed verbatim, so it is the candidate's id from your store:

```console
$ echo '{"id":"doc-41","query":"Rubyで配列を並べ替えるには","text":"配列の並べ替えには sort と sort_by がある。"}' | kohagi-rerank
{"id":"doc-41","score":0.9283465}
```

- One process can score any number of queries' candidates: repeat the query
  text on each record.
- A blank line means "score what I have sent and answer now", so a long-lived
  process can serve one query's candidates per request.
- See [PROTOCOL-rerank.md](../PROTOCOL-rerank.md) for the full contract.

## How many candidates

- Cost is linear in pairs, so this is arithmetic rather than tuning.
- At the measured 18.5 pairs/s for `ruri-v3-reranker-310m` on the Neural Engine,
  100 candidates is about 5 s per query; `japanese-reranker-xsmall-v2` does the
  same 100 in under a second.
- Truncation applies to the pair as a whole: `--max-seq-length` counts the
  query, the document and the special tokens between them, and the longer of the
  two halves is trimmed first.
- A long document scored against a short query therefore loses its tail.
  `--report-tokens` says which pairs that happened to.

## Thresholds

- The default score is `sigmoid(logit)`, the same number
  `sentence_transformers.CrossEncoder.predict` returns for a one-label model, so
  a threshold tuned against that library carries over unchanged.
- `--raw-logits` reports the logit instead.

### Three things to know before putting a number in a config file

- **A score is comparable within a model, not across models.** 0.7 from one
  reranker means nothing about 0.7 from another, and comparing across queries
  needs checking on your own data. PROTOCOL-rerank.md says this more precisely.
- **A threshold does not carry from `--device cpu` to `--device coreml`.** The
  ANE runs the encoder in fp16, and the error that introduces is compressed by
  the sigmoid: hardest near 0 and 1, least near 0.5. A cutoff at 0.02 moved
  nothing out of 200 sampled boundary pairs; one at 0.5 moved 12% of them.
  PROTOCOL-rerank.md derives the band for any threshold, so you can compute it
  rather than re-measure; `tools/rerank_fp16_bands.py` re-measures it if you
  would rather.
- **A threshold belongs to the weights, not to the run.** Record what produced
  the scores next to them:

```console
$ kohagi-rerank --print-model-info
{"model":"cl-nagoya/ruri-v3-reranker-310m","backend":"cpu","sha256":"93a48c41…","score":"sigmoid",…}
```

- `score` says which side of the sigmoid the numbers are on.
- `sha256` names the checkpoint by content. Two fine-tunes differ only in their
  bytes, and a threshold tuned for one is not a threshold for the other.

## Which checkpoints work

- Any ModernBERT sequence-classification checkpoint with one label:
  `cl-nagoya/ruri-v3-reranker-310m` (the default) and the
  `hotchpotch/japanese-reranker-{tiny,xsmall,small,base}-v2` family.
