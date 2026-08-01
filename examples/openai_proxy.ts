/**
 * An OpenAI-compatible /v1/embeddings endpoint in front of kohagi.
 *
 *     node --experimental-strip-types examples/openai_proxy.ts --kohagi ./target/release/kohagi
 *     deno run -A examples/openai_proxy.ts --kohagi ./target/release/kohagi
 *     bun examples/openai_proxy.ts --kohagi ./target/release/kohagi
 *
 * Then point any OpenAI client at it and nothing else in your code changes:
 *
 *     const client = new OpenAI({ baseURL: "http://127.0.0.1:8080/v1", apiKey: "unused" });
 *     const r = await client.embeddings.create({ model: "ruri-v3-130m", input: ["…", "…"] });
 *
 * That baseURL swap is the point. kohagi has no HTTP mode — it speaks JSONL
 * over a pipe, which is a smaller contract and works from any language that can
 * spawn a process. This file is the bridge.
 *
 * ## How it works, and why this way
 *
 * One kohagi process per request, with `--format openai`, and its stdout
 * returned verbatim. The response is already the right shape, `usage` included,
 * so there is no envelope to assemble here.
 *
 * The obvious alternative — one long-lived kohagi, fed request by request —
 * does not work, and it is worth knowing why before you try it. kohagi's
 * protocol is a batch protocol: it embeds in chunks of 1024 records and flushes
 * when a chunk fills or when stdin closes. A request of two texts produces
 * nothing until one of those happens, so a server holding the pipe open waits
 * forever. Closing stdin is the only end-of-request signal there is, and
 * closing it ends the process.
 *
 * The cost is a model load per request: about 0.3 s warm on CPU for
 * ruri-v3-130m. If that matters more than simplicity, batch texts into fewer,
 * larger requests — which is what the API's array `input` is for anyway.
 *
 * ## Before swapping a production baseURL
 *
 * - Dimensions differ. ruri-v3-130m returns 512 where text-embedding-3-small
 *   returns 1536, so an existing index has to be rebuilt. The compatibility is
 *   in the protocol, not in the vectors.
 * - `model` in the request is ignored. Which checkpoint runs is decided by the
 *   flags this proxy passes to kohagi. The response says which one actually ran.
 *
 * No dependencies; `node:http` and `node:child_process` only.
 */

import { spawn } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";

/** Flags this proxy understands; everything else goes through to kohagi. */
const OWN = ["--kohagi", "--host", "--port", "--model-id", "--prefix", "--device"];

function parseArgs(argv: string[]) {
  const own: Record<string, string> = {
    "--kohagi": "kohagi",
    "--host": "127.0.0.1",
    "--port": "8080",
    "--model-id": "cl-nagoya/ruri-v3-130m",
    "--prefix": "",
    "--device": "cpu",
  };
  const extra: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (OWN.includes(arg)) own[arg] = argv[++i] ?? "";
    else extra.push(arg);
  }
  return { own, extra };
}

const { own, extra } = parseArgs(process.argv.slice(2));
const KOHAGI = [
  "--model-id", own["--model-id"]!,
  "--device", own["--device"]!,
  "--prefix", own["--prefix"]!,
  "--format", "openai",
  ...extra,
];

/** kohagi's own OpenAI response for `texts`. */
function embed(texts: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(own["--kohagi"]!, KOHAGI);
    let out = "";
    let err = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (c: string) => (out += c));
    child.stderr.on("data", (c: string) => (err += c));
    child.on("error", reject);
    child.on("close", (code) =>
      // Exit 2 means some lines were skipped; the records that did come back
      // are still valid, but a proxy cannot return a short array as if nothing
      // happened. See PROTOCOL.md for the exit codes.
      code === 0 ? resolve(out) : reject(new Error(err.trim() || "kohagi failed")),
    );
    child.stdin.end(
      texts.map((text, id) => JSON.stringify({ id, text })).join("\n"),
      "utf8",
    );
  });
}

function send(res: ServerResponse, status: number, body: string) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

/** The error shape OpenAI clients expect, so their exceptions carry the message. */
function fail(res: ServerResponse, status: number, message: string) {
  send(res, status, JSON.stringify({ error: { message, type: "invalid_request_error" } }));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (c: string) => (body += c));
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

const server = createServer(async (req, res) => {
  const path = (req.url ?? "").split("?")[0]!.replace(/\/$/, "");

  // Some clients list models before their first call.
  if (req.method === "GET" && path === "/v1/models") {
    return send(
      res,
      200,
      JSON.stringify({
        object: "list",
        data: [{ id: own["--model-id"], object: "model", owned_by: "kohagi" }],
      }),
    );
  }
  if (path !== "/v1/embeddings") return fail(res, 404, "only /v1/embeddings is served");
  if (req.method !== "POST") return fail(res, 405, "POST only");

  let body: { input?: unknown };
  try {
    body = JSON.parse((await readBody(req)) || "{}");
  } catch (e) {
    return fail(res, 400, `invalid JSON: ${(e as Error).message}`);
  }

  // The API takes a string or an array of them. Arrays of tokens are also legal
  // there and not supported here; say so rather than embedding the digits.
  const given = body.input;
  const texts = typeof given === "string" ? [given] : given;
  if (!Array.isArray(texts) || texts.length === 0 || !texts.every((t) => typeof t === "string")) {
    return fail(res, 400, "`input` must be a string or an array of strings");
  }

  try {
    send(res, 200, await embed(texts as string[]));
  } catch (e) {
    fail(res, 500, (e as Error).message);
  }
});

server.listen(Number(own["--port"]), own["--host"], () => {
  console.log(
    `kohagi-openai-proxy: http://${own["--host"]}:${own["--port"]}/v1  (${own["--model-id"]})`,
  );
});
