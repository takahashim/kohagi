# Examples

How to call Kohagi from your own code. Both work the same way underneath.
They spawn the process, write JSONL, and read JSONL.

- [`rails_open3.rb`](rails_open3.rb) shows the stdio protocol from
  Ruby/Rails, the pattern any language can copy.
- [`openai_proxy/`](openai_proxy/) serves an OpenAI-compatible
  `/v1/embeddings` endpoint in Python, Ruby and TypeScript, so existing
  OpenAI code works by swapping `base_url`.

The scripts that measure and verify Kohagi rather than demonstrate it
(`parity_check.py`, `benchmark.py`, `model_check.py`, `scaling_check.py`,
`eval_retrieval.py`) live in [`tools/`](../tools/) with the CoreML jigs.

---

## Calling Kohagi from Ruby (`rails_open3.rb`)

Spawn the process, write `{"id","text"}` JSONL to stdin, read
`{"id","embedding"}` JSONL from stdout, and map results back by `id`.

The one structural requirement: **read stdout from a separate thread while
writing stdin.** Kohagi emits results in chunks as it goes, so writing an
entire corpus before reading anything fills the pipe buffer and deadlocks
both processes. The example uses a writer thread plus a reader loop.

Exit codes matter too. `0` is clean, `2` finished with skipped lines (the
output you did receive is valid; investigate stderr and resend those
records), and `1` is fatal. See [PROTOCOL.md](../PROTOCOL.md) for the full
contract.
