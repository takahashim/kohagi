"""An OpenAI-compatible /v1/embeddings endpoint in front of Kohagi.

    python3 examples/openai_proxy/proxy.py --kohagi ./target/release/kohagi

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
    client.embeddings.create(model="ruri-v3-130m", input=["…", "…"])

See README.md in this directory for what this is for, how it works, and what to
know before pointing production at it. Standard library only.
"""

import argparse
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KOHAGI = None
MODEL = "kohagi"


class Kohagi:
    """One long-lived Kohagi process, one request at a time.

    The lock is not optional. Kohagi's stdout carries batches in the order the
    batches were asked for, with nothing tying a reply to a requester, so two
    overlapping requests would each read the other's response. Serializing here
    keeps the model loaded once without needing a pool.
    """

    def __init__(self, argv):
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        self.lock = threading.Lock()

    def embed(self, texts):
        """Kohagi's own OpenAI response for `texts`, as bytes."""
        payload = "".join(
            json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n"
            for i, t in enumerate(texts)
        )
        with self.lock:
            # The blank line ends the batch; without it Kohagi waits for 1024
            # records before embedding anything.
            self.proc.stdin.write(payload.encode() + b"\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("kohagi exited; see its stderr")
        return line


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
            self.send_bytes(200, KOHAGI.embed(texts))
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

    global KOHAGI, MODEL
    KOHAGI = Kohagi([
        args.kohagi,
        "--model-id", args.model_id,
        "--device", args.device,
        "--prefix", args.prefix,
        "--format", "openai",
        *extra,
    ])
    MODEL = args.model_id

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"kohagi-openai-proxy: http://{args.host}:{args.port}/v1  ({args.model_id})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
