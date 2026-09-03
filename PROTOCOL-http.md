# Kohagi HTTP protocol (v1): `kohagi-serve`

`kohagi-serve` is the same model as `kohagi`, loaded once and kept, answering
OpenAI's `POST /v1/embeddings` instead of stdin; started with
`--rerank-model-id`, it is the same cross-encoder as `kohagi-rerank` too,
behind `POST /v1/rerank` in the shape Cohere and Jina gave that call. It
exists for the caller that wants one model per host rather than one per
process: a Rails cluster whose every Puma worker would otherwise hold its own
copy, or a sidecar next to an application in another container.

The stdio protocol ([PROTOCOL.md](PROTOCOL.md)) stays the contract for
batches. It streams a corpus with flat memory, which a request cannot, and it
is what `kohagi-ruby` speaks. This document is the contract for everything
else, and what differs is only how a request arrives: the model, its flags,
its vectors and the exit codes at load are the CLI's.

## Starting

```bash
kohagi-serve --prefix "検索クエリ: "                            # http://127.0.0.1:8080
kohagi-serve --listen unix:///run/kohagi.sock --device metal
kohagi-serve --listen 0.0.0.0:8080 --model-path m/model.safetensors --tokenizer-path m/tokenizer.json
kohagi-serve --rerank-model-id cl-nagoya/ruri-v3-reranker-310m       # /v1/rerank as well
```

Every model flag of `kohagi` is taken as it is (`--model-id`, `--model-path`,
`--device`, `--precision`, `--pooling`, `--max-seq-length`, `--dims`,
`--no-normalize`, `--expect-sha256`, the CoreML flags). `--prefix` is
prepended to every input text, as it is on the CLI; one server has one prefix.

The server **loads before it listens**. A missing checkpoint, a bad flag or a
failed `--expect-sha256` (or `--rerank-expect-sha256`) is a failed start with
the CLI's exit code (1 fatal, 3 the CoreML backend cannot serve this request),
and no port is ever open for a model that is not there. Once listening it
writes one line to stderr:

```
kohagi-serve: listening on http://127.0.0.1:8080 model=cl-nagoya/ruri-v3-130m sha256=1c342581efc2 pooling=mean dim=512 max_seq=512
kohagi-serve: reranker=cl-nagoya/ruri-v3-reranker-310m sha256=9f747a085a0a pooling=cls dim=768 max_seq=512 score=sigmoid
```

The second line appears when a reranker was loaded.

It then runs until SIGTERM or SIGINT, stops accepting, lets open connections
finish the reply in flight (up to `--shutdown-timeout`, 30s), writes one
summary line, and exits 0:

```
kohagi-serve: model=cl-nagoya/ruri-v3-130m sha256=1c342581efc2 pooling=mean dim=512 max_seq=512 requests=1204 in=1310 truncated=2 scored=480 rejected=3 failed=0
kohagi-serve: reranker=cl-nagoya/ruri-v3-reranker-310m sha256=9f747a085a0a pooling=cls dim=768 max_seq=512 score=sigmoid
```

The second line appears when a reranker was loaded, as at the start, so the
weights behind `scored=` are in the same place as the numbers.

`requests` counts everything that arrived; `rejected` the 4xx among them
(the client's mistake) and `failed` the 5xx (this side's, a full queue
included). `in` is the texts embedded and `scored` the documents reranked,
over the requests that were answered; `truncated` counts the texts and the
pairs that ran past their model's length. A request is answered whole or not
at all, so there is no `out`. Read the line as `key=value` pairs; later
versions may add fields.

| flag | default | meaning |
|---|---|---|
| `--listen` | `127.0.0.1:8080` | `host:port`, or `unix:///path` (see below) |
| `--rerank-model-id` | none | a reranker repo to load as well, which turns `/v1/rerank` on: `cl-nagoya/ruri-v3-reranker-310m`, or any ModernBERT sequence-classification checkpoint with one label |
| `--rerank-model-path`, `--rerank-tokenizer-path` | none | the reranker from local files instead; also turns `/v1/rerank` on |
| `--rerank-max-seq-length` | 512 | token-level truncation for a query/document pair; the longer half is trimmed first |
| `--rerank-coreml-dir`, `--rerank-coreml-model-id` | none | the reranker's converted bundle under `--device coreml`; omitted, the checkpoint is converted on first use |
| `--rerank-coreml-buckets` | 128,256,512 | what to convert when `--device coreml` converts the reranker itself; `kohagi-rerank`'s defaults, so the two produce the same bundle |
| `--rerank-expect-sha256` | none | the reranker's digest, checked as `--expect-sha256` checks the embedder's; a mismatch is a failed start |
| `--max-inputs` | 2048 | the most `input` items (or `documents`) one request may carry (OpenAI's own limit) |
| `--max-body-bytes` | 32 MiB | the longest request body read; longer is 413 |
| `--max-queue` | 64 | requests allowed to wait for the model (1 or more); one more is 503 |
| `--shutdown-timeout` | 30 | seconds to let open connections finish after a stop signal |

## `POST /v1/embeddings`

```json
{"input": ["瑠璃も玻璃も照らせば光る", "犬も歩けば棒に当たる"],
 "model": "ruri-v3-130m",
 "encoding_format": "base64",
 "dimensions": 256}
```

| field | type | notes |
|---|---|---|
| `input` | string, or array of strings | The texts, without the prefix. Empty strings and empty arrays are refused (400): the stdio protocol skips a bad record and says so in the summary, but a request is answered whole or not at all. Token ids, which OpenAI also accepts, are refused by name. |
| `model` | string | **Ignored.** The flags the server started with decide which checkpoint runs, and the reply names it. |
| `encoding_format` | `"float"` (default) or `"base64"` | `float` writes each vector as a JSON array of numbers. `base64` writes its float32 little-endian bytes in base64: a third the size, and a tenth the parsing cost on the client. |
| `dimensions` | integer | Keep the first N dimensions and re-normalize, as `--dims N` does for a run (`SentenceTransformer(..., truncate_dim=N)`). `1..=dim`; equal to the model's dimension changes nothing. Refused when the server runs `--no-normalize`, since the truncation re-normalizes. With `--dims N` on the server, a request may only go lower. |

Other fields (`user`, and anything else) are accepted and ignored.

The reply is the object `kohagi --format openai` writes:

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.0123, …]},
          {"object": "embedding", "index": 1, "embedding": [-0.0456, …]}],
 "model": "cl-nagoya/ruri-v3-130m",
 "usage": {"prompt_tokens": 31, "total_tokens": 31}}
```

- `index` is the position in `input`; there are no ids here.
- `model` is the server's label for what it loaded, not the request's string.
- `usage.prompt_tokens` counts every token the model saw, prefix and special
  tokens included, which is `min(true length, --max-seq-length)` per text. A
  text that ran past `--max-seq-length` was embedded from its kept prefix; the
  reply does not mark it, but the summary line counts it.
- Vectors are L2-normalized unless the server runs `--no-normalize`, and are
  the same numbers `kohagi` writes for the same text and flags.

Reading `base64` back:

```ruby
require "base64"
encoded = reply.dig("data", 0, "embedding")
vec = Base64.strict_decode64(encoded).unpack("e*")
```

```python
vec = np.frombuffer(base64.b64decode(item["embedding"]), dtype="<f4")  # Python
```

## `POST /v1/rerank`

Answered only when the server was started with `--rerank-model-id` (or
`--rerank-model-path`); otherwise 404, with the flag named. The reranker runs
on the same `--device` at the same `--precision` as the embedder, on a thread
of its own, so embedding and reranking do not wait for each other. On
`--device coreml` that means two models driving the Neural Engine from two
threads, which has not been measured; the CPU and the GPUs are where this
server is expected to run.

```json
{"query": "Rubyで配列を並べ替えるには",
 "documents": ["配列の並べ替えには sort と sort_by がある。", "今日の天気は晴れです。", {"text": "Ruby の Array#sort は比較ブロックを取る。"}],
 "top_n": 2,
 "return_documents": true}
```

| field | type | notes |
|---|---|---|
| `query` | string | Not empty. Taken raw: `--prefix` is for embedding, and a cross-encoder takes the pair as it is. |
| `documents` | array of strings, or of `{"text": …}` objects | Not empty, none of them empty; at most `--max-inputs`. Both spellings are accepted, as Cohere accepts both. |
| `top_n` | integer | Return only the best `top_n`; omitted returns them all. `0` is refused. |
| `return_documents` | bool, default `false` | Put each document's text in its result, so the caller need not look it up by index. |
| `model` | string | **Ignored**, as for embeddings; the reply names what was loaded. |

The reply is the shape Cohere's and Jina's `/v1/rerank` share, which TEI and
vLLM follow as well:

```json
{"model": "cl-nagoya/ruri-v3-reranker-310m",
 "results": [{"index": 2, "relevance_score": 0.9495417, "document": {"text": "Ruby の Array#sort は比較ブロックを取る。"}},
             {"index": 0, "relevance_score": 0.6369629, "document": {"text": "配列の並べ替えには sort と sort_by がある。"}}],
 "usage": {"total_tokens": 60}}
```

- `results` is best first, cut to `top_n`; equal scores keep their input
  order. `index` is the position in `documents`.
- `relevance_score` is the sigmoid of the model's logit, the number
  `sentence_transformers.CrossEncoder.predict` returns and the one
  `kohagi-rerank` writes by default, so thresholds carry over unchanged. The
  same pair gets the same score on either.
- `document` is present only with `return_documents`.
- `usage.total_tokens` counts every pair the model saw, the ones `top_n`
  dropped included.

A cross-encoder scores every pair with a forward pass, so this costs what
`kohagi-rerank` costs: about 2 seconds for 20 documents of 40 tokens on an M2
CPU. Keep the candidate list short (see docs/reranking.md), and put the
reranker on the ANE or a GPU when there is one.

## `GET /v1/models` and `GET /v1/models/{id}`

```json
{"object": "list",
 "data": [{"id": "cl-nagoya/ruri-v3-130m", "object": "model", "owned_by": "kohagi",
           "kohagi": {"backend": "cpu", "precision": "f32", "sha256": "1c342581…",
                      "pooling": "mean", "dim": 512, "max_seq_length": 512, "declared_max_seq_length": 8192,
                      "normalized": true}}]}
```

`kohagi` holds what `kohagi --print-model-info` prints (`output_dim` too, when
`--dims` shortened the vectors). A client can check `dim` against its index's
column and `sha256` against the digest it recorded, before it embeds anything.
With a reranker loaded, `data` has a second entry for it, whose `kohagi`
carries `score` (`sigmoid`) in place of `output_dim` and `normalized`, as
`kohagi-rerank --print-model-info` does. `/v1/models/{id}` returns the one
object whose `id` matches, and 404 otherwise.

## `GET /health`

`200 {"status":"ok","model":"cl-nagoya/ruri-v3-130m"}`. Loading precedes
listening, so a reply means ready; `HEAD` works too.

## Errors

Every refusal is a status code and the object OpenAI clients read their
exception from:

```json
{"error": {"message": "`input` must contain at least one string",
           "type": "invalid_request_error", "param": "input", "code": null}}
```

| status | when |
|---|---|
| 400 | the body is not JSON, or not an embeddings (or rerank) request; `input`, `encoding_format`, `dimensions`, `query`, `documents` or `top_n` failed a check above (`param` names which) |
| 404 | no such path; `/v1/models/{id}` for a model that is not loaded; `/v1/rerank` when no reranker was loaded |
| 405 | wrong method for a known path (`Allow` says which) |
| 413 | the body is longer than `--max-body-bytes`, by `Content-Length` or as read |
| 503 | `--max-queue` requests are already waiting; `Retry-After: 1` |
| 500 | the forward pass failed; the message says how, and stderr has it too |

## One request at a time

A model answers requests one after another. A forward pass already uses
every physical core (or the whole GPU), so nothing is gained by running two at
once, and a request that arrives while another is answered waits in a queue
of `--max-queue`. A request that finds the queue full is refused at once with
503 rather than queued behind it; the client decides whether to retry. The
embedder and the reranker each have a thread and a queue of their own, so a
long rerank does not hold up a query's embedding; on a CPU the two do share
the cores.

One request's cost is the forward pass for its texts; the HTTP around it is
small. For a 20-token query on an M2 CPU, one request takes 36 ms median over
HTTP against 32 ms for one blank-line batch on the pipe. Keep the connection
open between requests (every OpenAI client does), and send several texts per
request when there are several.

## Unix sockets

`--listen unix:///run/kohagi.sock` (or `unix:/run/kohagi.sock`) listens on a
Unix domain socket instead of TCP, for a server that shares a host with its
callers: the file's permissions are the access control and no port needs
choosing. The socket is created with mode 0600. A socket left at the path by a
previous run is replaced; a running `kohagi-serve` at that path, or anything
that is not a socket, is refused rather than replaced. The file is removed at
shutdown.

shutdown. Unix only; on Windows
`--listen` takes `host:port`.

```bash
curl --unix-socket /run/kohagi.sock http://localhost/v1/embeddings \
     -d '{"input": "瑠璃"}'
```

Ruby's `Net::HTTP` does not speak to a Unix socket; `excon` and `httpx` do.

## What this is not

- **Not for the open network.** There is no TLS and no authentication, as
  there is none on a database's socket. The default bind is loopback; put a
  reverse proxy in front before `--listen 0.0.0.0:…`.
- **Not a daemon.** It runs in the foreground and stops on a signal; systemd,
  launchd, a Procfile or a Kubernetes pod supervises it.
- **HTTP/1.1.** That is the contract; whether HTTP/2 is also spoken is not.
- **Not a batch path.** A corpus goes through the stdio protocol, one process
  and flat memory; this answers requests.

## Versioning

Fields may be added to replies and to the summary line; none will be removed
or renamed without a new version of this document. This describes version 1.
