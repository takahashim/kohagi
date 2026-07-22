# kohagi stdio protocol (v1)

kohagi is a pure function from id-tagged texts to id-tagged vectors, spoken
over stdin/stdout as JSONL. It holds no state and knows nothing about your
schema; the caller owns the data and maps results back by `id`.

A design principle follows from that: **text shaping is the caller's job.**
kohagi never trims, truncates (by characters), deduplicates, or otherwise
edits the text it receives — it only prepends the configured `--prefix`,
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

- **Skips (non-fatal).** A line is skipped — with `kohagi: skip line N:
  <reason>` on stderr and a count in the summary — when it is not valid JSON,
  not a JSON object, has no `id`, or has a missing / empty / non-string
  `text`. Processing continues; resend skipped records in a later run.
- **Blank lines** (empty or whitespace-only) are silently ignored and not
  counted, so a trailing newline is always safe.

## Output (stdout, JSONL, one record per line)

```json
{"id": 123, "embedding": [0.0123, -0.0456, …]}
```

- `id` is the input value, unchanged. **Map by id, not by order** (current
  output order matches input order, but the contract doesn't promise it).
- `embedding` has the model's dimension (512 for ruri-v3-130m) and is
  L2-normalized unless `--no-normalize` is set.
- stdout carries records only; every line is written whole (one `write` per
  record), so a reader never sees a partial line. Logs, warnings, and the
  summary go to stderr.
- Internally kohagi encodes in chunks (1024 records) against a single model
  load and flushes output after each chunk — resident memory stays flat on
  arbitrarily large input, and the caller can consume results incrementally.
  **Read stdout concurrently while writing stdin** (e.g. a reader thread);
  writing everything before reading anything can deadlock both processes on
  the pipe buffer.

## Summary and exit codes (stderr / process exit)

On completion, one summary line on stderr:

```
kohagi: model=cl-nagoya/ruri-v3-130m dim=512 in=2141 out=2141 skipped=0
```

`in` = lines parsed as records = `out + skipped`; blank lines are not counted.

| exit | meaning |
|---|---|
| 0 | every record embedded (`skipped=0`). Empty input is also 0 — nothing to do is success, and the model is not even loaded. |
| 2 | finished, but ≥1 line was skipped. Received output lines are all valid — consume them, then investigate stderr and resend the skipped records. |
| 1 | fatal: model load failure, bad flags, I/O error. Output may be truncated at a line boundary (never mid-line). |

## Versioning

The protocol is backward compatible; a breaking change would come with an
explicit `--protocol N` flag. This document describes protocol 1.
