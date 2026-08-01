"""An OpenAI-compatible /v1/embeddings endpoint in front of Kohagi.

    python3 examples/openai_proxy.py --kohagi ./target/release/kohagi

Then point any OpenAI client at it and nothing else in your code changes:

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
    r = client.embeddings.create(model="ruri-v3-130m", input=["…", "…"])

That `base_url` swap is the point. Kohagi has no HTTP mode — it speaks JSONL
over a pipe, which is a smaller contract and works from any language that can
spawn a process. This file is the bridge, so declining to build a server into
Kohagi does not cost anyone the OpenAI ecosystem.

## How it works, and why this way

One Kohagi process per request, with `--format openai`, and its stdout returned
verbatim. The response is already the right shape, `usage` included, so there is
no envelope to assemble here.

The obvious alternative — one long-lived Kohagi, fed request by request — does
not work, and it is worth knowing why before you try it. Kohagi's protocol is a
batch protocol: it embeds in chunks of 1024 records and flushes when a chunk
fills or when stdin closes. A request of two texts produces nothing until one of
those happens, so a server holding the pipe open waits forever. Closing stdin is
the only end-of-request signal there is, and closing it ends the process.

The cost is a model load per request: about 0.3 s warm on CPU for
`ruri-v3-130m`, more on the first call and more for a larger checkpoint. If that
matters more than simplicity, batch your texts into fewer, larger requests —
which is what the OpenAI API's array `input` is for anyway.

## Before swapping a production base_url

- **Dimensions differ.** `ruri-v3-130m` returns 512 where `text-embedding-3-small`
  returns 1536, so an existing index has to be rebuilt. The compatibility is in
  the protocol, not in the vectors.
- **`model` in the request is ignored.** Which checkpoint runs is decided by the
  flags this proxy passes to Kohagi. The response says which one actually ran.

Standard library only.
"""

import argparse
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGV = []
MODEL = "kohagi"


def embed(texts):
    """Kohagi's own OpenAI response for `texts`, as bytes."""
    payload = "\n".join(
        json.dumps({"id": i, "text": t}, ensure_ascii=False) for i, t in enumerate(texts)
    )
    done = subprocess.run(
        ARGV, input=payload.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Exit 2 means some lines were skipped; the records that did come back are
    # still valid, but a proxy cannot return a short array as if nothing
    # happened. See PROTOCOL.md for the exit codes.
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode(errors="replace").strip() or "kohagi failed")
    return done.stdout


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/embeddings":
            return self.fail(404, "only /v1/embeddings is served")
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or "{}")
        except json.JSONDecodeError as e:
            return self.fail(400, f"invalid JSON: {e}")

        # The API takes a string or an array of them. Arrays of tokens are also
        # legal there and not supported here; say so rather than embedding the
        # digits.
        given = body.get("input")
        texts = [given] if isinstance(given, str) else given
        if not texts or not all(isinstance(t, str) for t in texts):
            return self.fail(400, "`input` must be a string or an array of strings")

        try:
            self.send_bytes(200, embed(texts))
        except RuntimeError as e:
            self.fail(500, str(e))

    def do_GET(self):
        # Some clients list models before their first call.
        if self.path.rstrip("/") != "/v1/models":
            return self.fail(404, "only /v1/models is served")
        self.reply(
            200,
            {
                "object": "list",
                "data": [{"id": MODEL, "object": "model", "owned_by": "kohagi"}],
            },
        )

    def send_bytes(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def reply(self, status, payload):
        self.send_bytes(status, json.dumps(payload).encode())

    def fail(self, status, message):
        # The error shape OpenAI clients expect, so their exceptions carry the
        # message rather than "unknown error".
        self.reply(
            status, {"error": {"message": message, "type": "invalid_request_error"}}
        )

    def log_message(self, fmt, *args):
        pass  # Kohagi's own stderr is the interesting log here.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kohagi", default="kohagi")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model-id", default="cl-nagoya/ruri-v3-130m")
    p.add_argument("--prefix", default="", help='e.g. "検索文書: " for Ruri v3')
    p.add_argument("--device", default="cpu")
    args, extra = p.parse_known_args()

    global ARGV, MODEL
    ARGV = [
        args.kohagi,
        "--model-id", args.model_id,
        "--device", args.device,
        "--prefix", args.prefix,
        "--format", "openai",
        *extra,
    ]
    MODEL = args.model_id

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"kohagi-openai-proxy: http://{args.host}:{args.port}/v1  ({args.model_id})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
