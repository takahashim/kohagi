# Kohagi stdio protocol (v1)

Kohagi is a pure function from id-tagged texts to id-tagged vectors, spoken
over stdin/stdout as JSONL. It holds no state and knows nothing about your
schema; the caller owns the data and maps results back by `id`.

A design principle follows from that: **text shaping is the caller's job.**
Kohagi never trims, truncates (by characters), deduplicates, or otherwise
edits the text it receives. It only prepends the configured `--prefix`,
tokenizes (with token-level truncation to `--max-seq-length`), and embeds. If
you store a digest of what you sent, it corresponds to exactly what was
embedded.

## Input (stdin, JSONL, UTF-8, one record per line)

```json
{"id": 123, "text": "タイトル\n紹介文\n本文…"}
```

| field | type | notes |
|---|---|---|
| `id` | any JSON value | **Opaque.** Echoed verbatim in the output; never interpreted. |
| `text` | string | Raw text, without the task prefix. Keep newlines JSON-escaped (`\n`) so each record is one physical line. |

- **Skips (non-fatal).** A line is skipped (with `kohagi: skip line N:
  <reason>` on stderr and a count in the summary) when it is not valid JSON,
  not a JSON object, has no `id`, or has a missing / empty / non-string
  `text`. Processing continues; resend skipped records in a later run.
- **Blank lines** (empty or whitespace-only) are silently ignored and not
  counted, so a trailing newline is always safe.

## Batches (a blank line)

A blank line on stdin means **embed what you have sent so far and reply now**,
instead of waiting for an internal chunk to fill. Kohagi answers with the
records for that batch followed by a blank line of its own, so a long-lived
caller can send a request and read until the blank line:

```
{"id": 1, "text": "…"}      →   {"id": 1, "embedding": […]}
{"id": 2, "text": "…"}          {"id": 2, "embedding": […]}
                     ↵          ↵
```

The reply marker is what makes this safe to build on. Counting records instead
would hang the moment one of them was skipped for being malformed, since the
count would never be reached.

Without a blank line nothing is lost: Kohagi still embeds in chunks of 1024 and
at end of input, which is what `cat texts.jsonl | kohagi` relies on. A blank
line is only needed when the caller wants an answer before then. An empty batch
(a blank line with nothing buffered) is answered too, so a stray blank line
costs a marker rather than a deadlock.

## Output (stdout, JSONL, one record per line)

```json
{"id": 123, "embedding": [0.0123, -0.0456, …]}
```

- `id` is the input value, unchanged. **Map by id, not by order** (current
  output order matches input order, but the contract doesn't promise it).
- `embedding` has the model's dimension (512 for ruri-v3-130m) and is
  L2-normalized unless `--no-normalize` is set.
- **`--dims N`** keeps only the first N dimensions and re-normalizes them
  (Matryoshka truncation), so dot = cosine still holds on the shorter vectors.
  That is why the flag refuses to combine with `--no-normalize`, and why a
  truncated run's vectors must not share an index with a full run's. N outside
  `1..=dim` is refused at load, before any input is read. The result matches
  `SentenceTransformer(model, truncate_dim=N)`; whether the shorter vectors are
  any good is the model's property (one trained with Matryoshka loss keeps
  most of its quality, others may not).
- **`--report-tokens`** adds two fields per record; without the flag they are
  omitted entirely, so the default output is unchanged.

  ```json
  {"id": 123, "embedding": [0.0123, …], "n_tokens": 512, "truncated": true}
  ```

  | field | type | notes |
  |---|---|---|
  | `n_tokens` | integer | The number of tokens actually embedded, special tokens included, which is `min(true length, --max-seq-length)`. |
  | `truncated` | bool | The text ran past `--max-seq-length`, so its tail was dropped; the vector reflects only the kept prefix. Route these to a chunking pass if whole-document coverage matters. |
- stdout carries records only; every line is written whole (one `write` per
  record), so a reader never sees a partial line. Logs, warnings, and the
  summary go to stderr.
- Internally Kohagi encodes in chunks (1024 records) against a single model
  load and flushes output after each chunk, so resident memory stays flat on
  arbitrarily large input and the caller can consume results incrementally.
  **Read stdout concurrently while writing stdin** (e.g. a reader thread);
  writing everything before reading anything can deadlock both processes on
  the pipe buffer.

## Summary and exit codes (stderr / process exit)

On completion, one summary line on stderr:

```
kohagi: model=cl-nagoya/ruri-v3-130m sha256=1c342581efc2 pooling=mean dim=512 max_seq=512 in=2141 out=2141 skipped=0 truncated=3
```

`in` = lines parsed as records = `out + skipped`; blank lines are not counted.
`truncated` counts how many of the `out` records ran past `--max-seq-length`
(always reported, with or without `--report-tokens`); a nonzero value is a hint
to raise `--max-seq-length` or chunk those documents, not an error.

The fields before `in` describe the model rather than the run: `sha256` is the
first 12 hex digits of the weights file's digest, and `pooling` / `dim` /
`max_seq` are the three settings that silently change every vector when they
differ. `dim` is the dimension the vectors actually have, which is `--dims N`
when that flag truncated them and the model's own otherwise. Together they make a captured log say *which* model produced the
vectors, not just which one was asked for. Two fine-tunes of one checkpoint,
or two blends of one pair, are otherwise indistinguishable in a log.

Input with no valid records never loads the model, and the line then carries
`dim=0` and none of the other model fields: nothing was loaded, so there is
nothing to report about it.

Read the line as `key=value` pairs rather than by position: it is meant for
people and for grep, and later versions may add fields.

## `--print-model-info`

The same facts, machine-readable, without embedding anything. Kohagi loads the
model, writes one JSON object on stdout, and exits 0 without reading stdin:

```console
$ kohagi --print-model-info
{"model":"cl-nagoya/ruri-v3-130m","backend":"cpu","precision":"f32","sha256":"1c342581efc23d0b50b92fb11ac1eeb02719691bcc59bdc0dc0b09a36b4fc6d1","pooling":"mean","dim":512,"max_seq_length":512,"normalized":true}
```

Call it once at the start of an evaluation and record the object beside the
results: renaming a directory, or copying the wrong one into it, then cannot
change what the numbers say they came from.

| field | notes |
|---|---|
| `model` | The name this run used: the `--model-id` repo, or the directory of a `--model-path` checkpoint. |
| `backend` / `precision` | `--device` and `--precision`, as their flag values. |
| `sha256` | Of every byte of `model.safetensors`. Identical weights always give an identical digest, and one byte's difference gives a different one. Absent on `--device coreml`, which has no safetensors to hash. |
| `pooling` | The pooling resolved at load, taken from the checkpoint's own `1_Pooling/config.json` unless `--pooling` overrode it. |
| `dim` | The model's own dimension. |
| `output_dim` | `--dims N`, when it truncated the output below `dim`. Absent when nothing was truncated, whether because the flag was not given or because `--dims` equalled `dim` and changed no vector. |
| `normalized` | Whether each vector is unit length, so that dot product is cosine: `true` unless `--no-normalize`. Changes every vector as surely as `pooling` does, and the numbers alone do not say. |
| `max_seq_length` | Token-level truncation length for this run. |
| `declared_max_seq_length` | What the checkpoint's `sentence_bert_config.json` says it can take, when it ships one. Reported, not obeyed: `--max-seq-length` decides the run. sentence-transformers reads this field, so a value above `max_seq_length` is where the two libraries embed the same long text differently (`ruri-v3-130m` declares 8192). |
| `source`, `source_sha256` | `--device coreml` only: the checkpoint the bundle was converted from, and the digest of *its* weights. Absent for a bundle whose converter recorded none; an unknown provenance is reported as unknown rather than guessed. |
| `buckets`, `quantization` | `--device coreml` only: the sequence lengths the bundle serves, and `none` / `embeddings-int8` / `all-int8`. A quantized bundle's vectors are not interchangeable with an fp16 one's, so a result that came from one should say so. |

Fields that do not apply to a run are omitted rather than set to null.

## `--expect-sha256`

The digest above makes a run recordable; this flag makes the record
enforceable. Pass a hex prefix of the expected digest (the summary's 12
digits, or the full 64 from `--print-model-info`) and Kohagi refuses to embed
anything with weights whose digest does not start with it:

```console
$ kohagi --model-path models/alpha05/model.safetensors … --expect-sha256 1c342581efc2 < texts.jsonl
kohagi: error: these are not the expected weights: --expect-sha256 1c342581efc2, but the loaded model's sha256 is e831a463bddb…
$ echo $?
1
```

- The check runs when the model loads, so a mismatch produces **no output at
  all**. The run exits 1 before any record is answered. A pipeline that recorded the
  digest beside its index can paste it back and be certain the wrong
  checkpoint (a renamed directory, a mixed-up interpolation, a stale download)
  never adds a vector to that index.
- On `--device coreml` the check reads the bundle's recorded `source_sha256`,
  the digest of the checkpoint it was converted from, which is the value a
  caller has. A bundle that recorded
  none (converted before Kohagi 0.6) cannot be verified and is refused.
- Empty input still exits 0 without loading the model, as ever; nothing was
  embedded, so nothing needed verifying.
- `--print-model-info --expect-sha256 <hex>` is a standalone check. It exits 0
  with the model's facts when the digest matches, and 1 when it does not.

| exit | meaning |
|---|---|
| 0 | every record embedded (`skipped=0`). Empty input is also 0. Nothing to do is success, and the model is not even loaded. |
| 2 | finished, but ≥1 line was skipped. Received output lines are all valid. Consume them, then investigate stderr and resend the skipped records. |
| 1 | fatal: model load failure, bad flags, I/O error. Output may be truncated at a line boundary (never mid-line). |
| 3 | the requested CoreML backend (`--device coreml`) cannot serve this request: built without the `coreml` feature, no `--coreml-dir`, no converted bucket for `--max-seq-length`, or a missing model. Detected before any input is read, so no output is produced and the caller can retry on `--device cpu`. Only ever returned when `--device coreml` is passed. |

## `--format openai`

For code already written against OpenAI's `/v1/embeddings`, `--format openai`
replaces the JSONL stream with one response object for the whole run:

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.0123, …]}],
 "model": "cl-nagoya/ruri-v3-130m",
 "usage": {"prompt_tokens": 15, "total_tokens": 15}}
```

This is also the reply `kohagi-serve` gives over HTTP, where it is the
protocol rather than an alternate format; see
[PROTOCOL-http.md](PROTOCOL-http.md).

Everything on stdin is unchanged; the input is still this protocol's JSONL.
What changes on stdout:

- **Embeddings are identified by `index`, not by id.** `index` is the position
  among the records that were embedded, which is what the API means by it, so a
  skipped input line shifts every index after it. Input ids are not carried; if
  you need them, use the default format.
- **`--report-tokens` is refused**, because the item shape has nowhere to put
  per-record counts. `usage.prompt_tokens` carries the total instead, and the
  summary line still reports `truncated=N`.
- **One document per batch.** A blank line closes the current response and
  starts the next, with `index` counting from zero again and `usage` covering
  that batch alone; one flush is one request's worth. With no blank lines the
  whole run is one document, which is what `kohagi --format openai < in.jsonl >
  out.json` produces.
- **An aborted run leaves an incomplete JSON document.** JSONL degrades to a
  shorter but valid file; a single object does not. Kohagi still writes the
  document in pieces rather than buffering, so memory stays flat either way.

## Versioning

The protocol is backward compatible; a breaking change would come with an
explicit `--protocol N` flag. This document describes protocol 1.
