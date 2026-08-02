# An OpenAI-compatible `/v1/embeddings` endpoint

Three implementations of the same ~150-line proxy, so you can lift whichever
matches the stack you already run:

| | run it |
|---|---|
| [`proxy.py`](proxy.py) | `python3 examples/openai_proxy/proxy.py --kohagi ./target/release/kohagi` |
| [`proxy.rb`](proxy.rb) | `ruby examples/openai_proxy/proxy.rb --kohagi ./target/release/kohagi` |
| [`proxy.ts`](proxy.ts) | `node --experimental-strip-types examples/openai_proxy/proxy.ts --kohagi ./target/release/kohagi` |

`proxy.rb` needs `gem install puma`; the other two are standard library only
(`proxy.ts` also runs under `deno run -A` and `bun`). Flags they do not
recognize are passed through to Kohagi, so
`--device coreml` or `--max-seq-length 256` work without being redeclared.

Then point any OpenAI client at it and nothing else in your code changes:

```python
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
client.embeddings.create(model="ruri-v3-130m", input=["…", "…"])
```

## Why this exists

That `base_url` swap is the whole value of OpenAI compatibility — it is what
lets LangChain, LlamaIndex and everything else built against that API use a
local model without a rewrite. The JSON shape alone does not get you there; the
HTTP transport does.

Kohagi has no HTTP mode and should not grow one. It speaks JSONL over a pipe,
which is a smaller contract and works from any language that can spawn a
process. These files are the bridge, so declining to build a server into Kohagi
does not cost anyone the OpenAI ecosystem — and if nobody wants them, they are
examples, so they can go.

## How they work

One long-lived Kohagi, so the model is loaded once. Each request writes its
records and then a blank line, which is Kohagi's *embed what you have and reply
now* signal (see [PROTOCOL.md](../../PROTOCOL.md)); with `--format openai` the
reply is that batch's complete response object, so there is no envelope to
assemble here.

A request costs about 40 ms warm, against 300 ms when each spawned its own
Kohagi.

**One request at a time.** All three serialize — a lock in Python, a mutex in
Ruby, a promise chain in TypeScript. Kohagi's stdout carries batches in the
order the batches were asked for and nothing ties a reply to a requester, so two
overlapping requests would each read the other's response. This is not
theoretical for `proxy.rb`: Puma serves on a thread pool, so the threads queue
on the mutex rather than racing.

## Before pointing production at one

- **Dimensions differ.** `ruri-v3-130m` returns 512 where
  `text-embedding-3-small` returns 1536, so an existing index has to be rebuilt.
  The compatibility is in the protocol, not in the vectors.
- **`model` in the request is ignored.** Which checkpoint runs is decided by the
  flags the proxy passes to Kohagi. The response says which one actually ran.
- **These are examples, not servers.** No auth, no TLS, no request limits, no
  restart if Kohagi dies. Bind to localhost, or take the ~30 lines that matter
  and put them behind whatever you already run.
