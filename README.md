# kohagi

Local sentence embeddings for [Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-130m)
and other ModernBERT encoders. JSONL in, vectors out — no server, no Python,
one binary.

```console
$ echo '{"id": 1, "text": "瑠璃も玻璃も照らせば光る"}' | kohagi
{"id":1,"embedding":[0.006987,-0.032139, …]}
```

kohagi is a small [candle](https://github.com/huggingface/candle)-based CLI
(and Rust library) built for one job: batch-embedding text from *any*
language that can spawn a process — a Rails rake task, a Node script, a shell
pipeline — without running an embedding server.

**Why you might want it:**

- **Pure Rust, single binary.** No PyTorch, no ONNX Runtime, no Python
  environment. macOS (Apple Accelerate) and Linux.
- **Accurate.** f32 inference matching the PyTorch / sentence-transformers
  reference (cosine ≈ 1.0).
- **Bounded memory.** Attention scratch per forward pass is capped and input
  is processed in chunks, so peak memory depends on your core count — not on
  how much you pipe in. Embedding an entire book corpus holds a few GB, flat.
- **A boring, stable contract.** `{"id","text"}` → `{"id","embedding"}`,
  exit codes 0/2/1. See [PROTOCOL.md](PROTOCOL.md).

Not a server, not a vector database, not a framework. If you want an HTTP
embedding service, use
[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference).

## Install

Prebuilt binaries for macOS (Apple Silicon) and Linux (x86_64) are on the
[releases page](https://github.com/takahashim/kohagi/releases):

```bash
tar -xzf kohagi-<target>.tar.gz && mv kohagi ~/.local/bin/
```

Or with cargo:

```bash
cargo install kohagi
```

## Quick start

The default model is `cl-nagoya/ruri-v3-130m` (Japanese, 512 dimensions),
downloaded from the Hugging Face Hub on first run and cached in
`~/.cache/huggingface`.

```bash
# Sanity check without composing JSONL:
kohagi --text "瑠璃も玻璃も照らせば光る" --text "犬も歩けば棒に当たる"

# The real interface — pipe JSONL through it:
kohagi --prefix "検索文書: " < texts.jsonl > embeddings.jsonl
```

### Ruri v3 prefixes

Ruri v3 is trained with task prefixes; kohagi prepends `--prefix` to every
text so your caller can send raw text:

| task | `--prefix` |
|---|---|
| plain sentence similarity | *(none — the default)* |
| document to be searched | `"検索文書: "` |
| search query | `"検索クエリ: "` |
| topic / keyword | `"トピック: "` |

### Other models

Any ModernBERT sentence encoder on the Hub works, e.g. English retrieval with
[nomic-ai/modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base):

```bash
kohagi --model-id nomic-ai/modernbert-embed-base \
       --prefix "search_document: " < texts.jsonl
```

`cl-nagoya/ruri-v3-310m` (768 dimensions) works the same way. For offline /
air-gapped use, point at local files instead — no network is touched:

```bash
kohagi --model-path models/ruri-v3-130m/model.safetensors \
       --tokenizer-path models/ruri-v3-130m/tokenizer.json
```

(`config.json` is expected next to the weights.)

## Calling from another language

Spawn the process, write JSONL to stdin, read JSONL from stdout — **from a
separate thread**, so the pipe never fills up. Match results by `id`. A
complete Ruby example is in
[`examples/rails_open3.rb`](examples/rails_open3.rb); exit code semantics are
in [PROTOCOL.md](PROTOCOL.md).

## Using the library

```rust
use kohagi::{Embedder, ModelSource, Options};

let embedder = Embedder::load(
    &ModelSource::Hub { repo: "cl-nagoya/ruri-v3-130m".into() },
    Options::default(),
)?;
let vecs = embedder.embed(&["検索クエリ: 瑠璃とは何ですか"])?;
```

`Options` covers pooling (mean/cls), L2 normalization, truncation length, and
batch granularity. One `Embedder` serves any number of `embed` calls and is
what the CLI itself is built on — `main.rs` is ~100 lines.

## Performance notes

- CPU only, and deliberately so: candle 0.10 has no Metal kernels for
  ModernBERT's rotary embeddings, and on Apple Silicon the Accelerate/AMX
  path is already within ~20% of PyTorch at identical output.
- Parallelism is across batches on physical cores; set `RAYON_NUM_THREADS`
  to override (more threads ≈ slightly faster, proportionally more memory).
- `--max-seq-length` is the main throughput lever: attention cost grows with
  the square of sequence length.

### `--precision bf16` (x86_64 with AVX512-BF16)

On Zen 4, Sapphire Rapids and newer, kohagi can run the four projection
`Linear`s through a bf16 kernel while keeping norms, softmax and attention
scores in f32 — the same split `torch.autocast` uses. Measured on an 8-core
Zen 4 (ruri-v3-130m, 1200 short texts / 240 texts at the 512-token limit):

| input | f32 | `--precision bf16` | peak RSS (f32 → bf16) |
|---|---:|---:|---|
| short (~60 tokens) | 10.2 s | **5.5 s** (1.9×) | 1.5 GB → 0.9 GB |
| long (512 tokens) | 54.1 s | **37.1 s** (1.5×) | 1.8 GB → 1.6 GB |

Embeddings agree with the f32 path at cosine ≈ 0.99999 (worst case 0.9996 on
long inputs) — far below the noise floor for retrieval ranking, but *not*
bit-identical. It stays opt-in for that reason: with the default f32, the
same text yields the same vector on every machine, which matters when
embeddings from different hosts land in the same index.

Other CPUs (including Apple Silicon) reject `--precision bf16` at startup with
a clear message rather than silently falling back.

---

## 日本語

kohagi は [Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-130m) などの
ModernBERT 系文埋め込みモデルをローカルで動かす CLI / Rust ライブラリです。
標準入力に `{"id","text"}` の JSONL を流すと、標準出力に
`{"id","embedding"}` が返ります。サーバも Python も不要、バイナリひとつ。

```bash
# インストール(リリースのバイナリ、または cargo install kohagi)
kohagi --text "瑠璃も玻璃も照らせば光る"          # 動作確認
kohagi --prefix "検索文書: " < in.jsonl > out.jsonl  # 本番はこちら
```

- モデルは初回に Hugging Face Hub から自動ダウンロード
  (`--model-path`/`--tokenizer-path` でオフライン運用も可)
- x86_64 (AVX512-BF16 搭載の Zen 4 / Sapphire Rapids 以降)では
  `--precision bf16` で 1.5〜1.9 倍高速化(cosine ≈ 0.99999、既定は f32)
- 出力は f32 で PyTorch / sentence-transformers と一致(cosine ≈ 1.0)
- メモリ使用量は入力サイズによらず一定(チャンク処理+attention 予算キャップ)
- 入出力の契約・exit code(0/2/1)は [PROTOCOL.md](PROTOCOL.md) を参照。
  Rails からの呼び出し例は [`examples/rails_open3.rb`](examples/rails_open3.rb)

## License

MIT
