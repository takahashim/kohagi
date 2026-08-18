# Choosing a model

- Kohagi runs ModernBERT-based sentence encoders from the Hugging Face Hub.
- The default is `cl-nagoya/ruri-v3-130m`, a Japanese model producing
  512-dimensional vectors.
- It is downloaded on first use and cached under `~/.cache/huggingface`.

## Other models

- Any ModernBERT sentence encoder works the same way:

```bash
# English retrieval
kohagi --model-id nomic-ai/modernbert-embed-base \
       --prefix "search_document: " < texts.jsonl

# Japanese, 768 dimensions
kohagi --model-id cl-nagoya/ruri-v3-310m --prefix "検索文書: " < texts.jsonl
```

- The prefix belongs to the model, not to Kohagi. Ruri v3 wants `"検索文書: "` /
  `"検索クエリ: "`; `modernbert-embed-base` wants `"search_document: "` /
  `"search_query: "`; many models want none at all.
- Kohagi prepends `--prefix` verbatim and does nothing else to the text, so the
  caller passes its documents unchanged.
- To check whether a model you are considering produces usable embeddings under
  Kohagi, run [`tools/model_check.py`](../tools/model_check.py) against it.
- For reranker checkpoints, see [Retrieval and reranking](reranking.md).

## Pooling is taken from the model

- Kohagi reads the checkpoint's `1_Pooling/config.json` and uses the mode it
  declares, so a CLS-pooled model such as `Alibaba-NLP/gte-modernbert-base`
  needs no flag.
- Pass `--pooling` only to override it.
- Kohagi warns if your choice disagrees with what the checkpoint declares.
- It also warns if the model ships no pooling config at all, which usually
  means it is a reranker or a base language model rather than a sentence
  encoder.
- Pooling changes every vector, so it is reported in the summary line and by
  `--print-model-info`. Two runs of the same weights under different pooling
  produce embeddings that must not share an index.

## Truncated dimensions (`--dims`)

- `--dims N` keeps the first N dimensions of each embedding and re-normalizes,
  matching `SentenceTransformer(model, truncate_dim=N)` — Matryoshka
  truncation, for models trained so a prefix of the vector is itself a usable
  embedding.
- Whether the shorter vectors retrieve well is the model's property, not
  Kohagi's: a checkpoint trained with Matryoshka loss keeps most of its
  quality at half dimension, one that was not may lose a great deal. Measure
  on your own data before committing an index to it.
- Truncated and full vectors must not share an index. The summary's `dim=` and
  `--print-model-info`'s `output_dim` record which one a run produced.
- N outside `1..=dim` is refused at load, as is combining with
  `--no-normalize` (re-normalization is what keeps dot product = cosine on the
  shorter vectors).

## Offline files

- Point Kohagi at local files and it makes no network requests:

```bash
kohagi --model-path models/ruri-v3-130m/model.safetensors \
       --tokenizer-path models/ruri-v3-130m/tokenizer.json
```

- `config.json` must sit in the same directory as the weights.
- A `1_Pooling/config.json` beside them is read too if present.
- Without one, pass `--pooling` for a CLS model, since there is nothing to
  detect from.

## Long inputs

- Text longer than `--max-seq-length` (512 tokens by default) is truncated
  before embedding, so its vector reflects only the beginning.
- This is not silent: the summary line always ends with `truncated=N`.
- `--report-tokens` adds the counts per record, so a caller can route the
  truncated ones to a chunking pass:

```console
$ echo '{"id": 1, "text": "…a very long document…"}' | kohagi --report-tokens
{"id":1,"embedding":[…],"n_tokens":512,"truncated":true}
```

- Raising `--max-seq-length` embeds more of each text at a quadratic cost in
  attention compute. It is the single flag with the largest effect on
  throughput.
- On `--device coreml` the largest converted bucket caps `--max-seq-length`; see
  [CoreML bundles](coreml.md).
