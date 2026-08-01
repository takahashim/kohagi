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
 * ## How it works
 *
 * One long-lived kohagi, so the model is loaded once. Each request writes its
 * records, then a blank line — kohagi's "that is one batch" signal — and reads
 * back exactly one line, which with `--format openai` is that batch's complete
 * response. There is no envelope to assemble here and no counting of records:
 * the blank line is what makes a batch a request. Serving a request costs about
 * 0.03 s warm, against 0.3 s if the process were spawned each time.
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

/**
 * One long-lived kohagi process, one request at a time.
 *
 * The queue is not optional. kohagi's stdout carries batches in the order the
 * batches were asked for, with nothing tying a reply to a requester, so two
 * overlapping requests would each read the other's response. Node serves
 * requests concurrently, so they are chained onto one promise instead.
 */
const child = spawn(own["--kohagi"]!, KOHAGI, { stdio: ["pipe", "pipe", "inherit"] });
child.stdout.setEncoding("utf8");

/** Resolvers waiting for a reply, in the order their batches were sent. */
const waiting: Array<(line: string) => void> = [];
let pending = "";
child.stdout.on("data", (chunk: string) => {
  pending += chunk;
  let nl: number;
  while ((nl = pending.indexOf("\n")) >= 0) {
    const line = pending.slice(0, nl);
    pending = pending.slice(nl + 1);
    waiting.shift()?.(line);
  }
});

let queue: Promise<unknown> = Promise.resolve();

/** kohagi's own OpenAI response for `texts`. */
function embed(texts: string[]): Promise<string> {
  const run = queue.then(
    () =>
      new Promise<string>((resolve) => {
        waiting.push(resolve);
        const payload = texts.map((text, id) => JSON.stringify({ id, text }) + "\n").join("");
        // The blank line ends the batch; without it kohagi waits for 1024
        // records before embedding anything.
        child.stdin.write(payload + "\n", "utf8");
      }),
  );
  queue = run.catch(() => {});
  return run;
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
