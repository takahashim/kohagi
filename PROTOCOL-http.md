# Kohagi HTTP protocol (v1): `kohagi-serve`

`kohagi-serve` is the same model as `kohagi`, loaded once and kept, answering
OpenAI's `POST /v1/embeddings` instead of stdin. It exists for the caller that
wants one model per host rather than one per process: a Rails cluster whose
every Puma worker would otherwise hold its own copy, or a sidecar next to an
application in another container.

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
```

Every model flag of `kohagi` is taken as it is (`--model-id`, `--model-path`,
`--device`, `--precision`, `--pooling`, `--max-seq-length`, `--dims`,
`--no-normalize`, `--expect-sha256`, the CoreML flags). `--prefix` is
prepended to every input text, as it is on the CLI; one server has one prefix.

The server **loads before it listens**. A missing checkpoint, a bad flag or a
failed `--expect-sha256` is a failed start with the CLI's exit code (1 fatal,
3 the CoreML backend cannot serve this request), and no port is ever open for
a model that is not there. Once listening it writes one line to stderr:

```
kohagi-serve: listening on http://127.0.0.1:8080 model=cl-nagoya/ruri-v3-130m sha256=1c342581efc2 pooling=mean dim=512 max_seq=512
```

It then runs until SIGTERM or SIGINT, stops accepting, lets open connections
finish the reply in flight (up to `--shutdown-timeout`, 30s), writes one
summary line, and exits 0:

```
kohagi-serve: model=cl-nagoya/ruri-v3-130m sha256=1c342581efc2 pooling=mean dim=512 max_seq=512 requests=1204 in=1310 out=1310 truncated=2 rejected=3
```

`in` and `out` are the stdio summary's: input texts received and vectors
returned. `requests` counts everything that arrived, `rejected` the 4xx and
5xx among them. Read the line as `key=value` pairs; later versions may add
fields.

| flag | default | meaning |
|---|---|---|
| `--listen` | `127.0.0.1:8080` | `host:port`, or `unix:///path` (see below) |
| `--max-inputs` | 2048 | the most `input` items one request may carry (OpenAI's own limit) |
| `--max-body-bytes` | 32 MiB | the longest request body read; longer is 413 |
| `--max-queue` | 64 | requests allowed to wait for the model; one more is 503 |
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
vec = reply.dig("data", 0, "embedding").unpack("e*")          # Ruby
```

```python
vec = np.frombuffer(base64.b64decode(item["embedding"]), dtype="<f4")  # Python
```

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
`/v1/models/{id}` returns that one object when `id` is the loaded model's
label, and 404 otherwise.

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
| 400 | the body is not JSON, or not an embeddings request; `input`, `encoding_format` or `dimensions` failed a check above (`param` names which) |
| 404 | no such path; `/v1/models/{id}` for a model that is not the loaded one |
| 405 | wrong method for a known path (`Allow` says which) |
| 413 | the body is longer than `--max-body-bytes`, by `Content-Length` or as read |
| 503 | `--max-queue` requests are already waiting; `Retry-After: 1` |
| 500 | the forward pass failed; the message says how, and stderr has it too |

## One request at a time

The model answers requests one after another. A forward pass already uses
every physical core (or the whole GPU), so nothing is gained by running two at
once, and a request that arrives while another is answered waits in a queue
of `--max-queue`. A request that finds the queue full is refused at once with
503 rather than queued behind it; the client decides whether to retry.

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
previous run is replaced; anything at the path that is not a socket is refused
rather than removed. The file is removed at shutdown. Unix only; on Windows
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
