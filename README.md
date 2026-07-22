# kohagi

A local sentence-embeddings CLI for [Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-130m) and other ModernBERT encoders.

kohagi reads JSONL records from standard input and writes the corresponding embedding vectors as JSONL.
It runs as a single executable and requires no cloud services or embedding server.

```console
$ echo '{"id": 1, "text": "瑠璃も玻璃も照らせば光る"}' | kohagi
{"id":1,"embedding":[0.006987,-0.032139, …]}
```

kohagi is a small CLI and Rust library built with [Candle](https://github.com/huggingface/candle).
It is designed for one job: embedding text in batches from any environment that can launch a subprocess, such as a Rails rake task, a Node.js script, or a shell pipeline.

### Why use kohagi?

- **Pure Rust and a single executable.** No PyTorch, LibTorch, ONNX Runtime, or Python environment required. Supports macOS with Apple Accelerate and Linux.
- **Accurate.** Uses f32 inference and closely matches the PyTorch and Sentence Transformers reference implementations  (cosine ≈ 1.0).
- **Bounded memory usage.** Attention scratch space is capped for each forward pass, and input is processed in chunks. Peak memory usage therefore depends primarily on the number of CPU cores, not on the total amount of input.
- **A deliberately simple, stable interface.** `{"id","text"}` in, `{"id","embedding"}` out, with exit codes `0`, `2`, and `1`. See [PROTOCOL.md](PROTOCOL.md).

## Install

Prebuilt binaries for macOS (Apple Silicon) and Linux (x86_64) are on the
[releases page](https://github.com/takahashim/kohagi/releases):

```bash
tar -xzf kohagi-<target>.tar.gz && mv kohagi ~/.local/bin/
```

The binaries are unsigned, so unpacking from Finder on macOS leaves a quarantine
attribute that Gatekeeper blocks. Extracting with `tar` as above avoids it;
otherwise run `xattr -dr com.apple.quarantine ~/.local/bin/kohagi`.

Or with cargo:

```bash
cargo install kohagi
```

## Quick start

By default, kohagi uses `cl-nagoya/ruri-v3-130m` (a Japanese sentence-embedding model, 512-dimensions).
The model is downloaded from the Hugging Face Hub the first time kohagi runs and is cached under `~/.cache/huggingface`.

```bash
# Run a quick sanity check without constructing JSONL:
kohagi --text "瑠璃も玻璃も照らせば光る" --text "犬も歩けば棒に当たる"

# For normal use, stream JSONL through standard input and output:
kohagi --prefix "検索文書: " < texts.jsonl > embeddings.jsonl
```

### Ruri v3 prefixes

Ruri v3 is trained to use task-specific prefixes.
kohagi prepends the value of `--prefix` to every input text, allowing callers to pass the original text unchanged.

| Task                              | `--prefix`                    |
| --------------------------------- | ----------------------------- |
| General sentence similarity       | *(none; this is the default)* |
| Document to be indexed for search | `"検索文書: "`                    |
| Search query                      | `"検索クエリ: "`                   |
| Topic or keyword                  | `"トピック: "`                    |

### Other models

kohagi can also run other ModernBERT-based sentence encoders available on the Hugging Face Hub.
For example, you can use [nomic-ai/modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base) for English-language retrieval:

```bash
kohagi --model-id nomic-ai/modernbert-embed-base \
       --prefix "search_document: " < texts.jsonl
```

`cl-nagoya/ruri-v3-310m`, which produces 768-dimensional vectors, works in the same way.
For offline environments, specify local model files instead. In this mode, kohagi does not make any network requests:

```bash
kohagi --model-path models/ruri-v3-130m/model.safetensors \
       --tokenizer-path models/ruri-v3-130m/tokenizer.json
```

kohagi expects `config.json` to be located in the same directory as the model weights.

## Calling kohagi from another language

Launch kohagi as a subprocess, write JSONL records to its standard input, and read JSONL results from its standard output.

Read the output concurrently, such as from a separate thread, to prevent the pipe buffer from filling up and blocking the process.
Use the `id` field to match each result with its input record.

A complete Ruby example is available in [`examples/rails_open3.rb`](examples/rails_open3.rb).
See [PROTOCOL.md](PROTOCOL.md) for the exit-code semantics.

## Using the Rust library

```rust
use kohagi::{Embedder, ModelSource, Options};

let embedder = Embedder::load(
    &ModelSource::Hub { repo: "cl-nagoya/ruri-v3-130m".into() },
    Options::default(),
)?;

let embeddings = embedder.embed(&["検索クエリ: 瑠璃とは何ですか"])?;
```

`Options` controls the pooling strategy (`mean` or `cls`), L2 normalization, maximum sequence length, and batch granularity.

A single `Embedder` instance can be reused for any number of `embed` calls.
The CLI is built on the same API, and its `main.rs` is ~100 lines.

## Performance notes

* CPU by default, via Apple Accelerate on macOS, which performs within about 20% of PyTorch with equivalent output.
* Batches run in parallel across physical CPU cores. Set `RAYON_NUM_THREADS` to override the default; additional threads may improve throughput at the cost of memory.
* `--max-seq-length` has the largest effect on throughput because attention cost grows quadratically with sequence length.

Throughput is worth measuring on your own machine and texts rather than taking
numbers on faith. [`examples/benchmark.py`](examples/benchmark.py) times kohagi against
Sentence Transformers on the same corpus and settings; see
[`examples/README.md`](examples/README.md) for measured results on Apple Silicon.

### `--device metal` on Apple Silicon

Building with `--features metal` adds an Apple GPU backend. On an M2 it runs
512-token batches about 1.2× faster than the Accelerate CPU path, with f32
output unchanged (worst `1 - cosine` 9e-13 against CPU).

This needs the patched candle in [`vendor/`](vendor/README.md), so it is off by
default and only applies when building from this repository.

### `--precision bf16` on AVX512-BF16 CPUs

On Zen 4 (Sapphire Rapids) and newer CPUs, `--precision bf16` uses `bf16` for projection layers while keeping normalization, softmax, and attention scores in `f32`.

Measured on an 8-core Zen 4 CPU using `ruri-v3-130m`:

| Input                  |    f32 |              bf16 |        Peak RSS |
| ---------------------- | -----: | ----------------: | --------------: |
| Short, about 60 tokens | 10.2 s |  **5.5 s** (1.9×) | 1.5 GB → 0.9 GB |
| Long, 512 tokens       | 54.1 s | **37.1 s** (1.5×) | 1.8 GB → 1.6 GB |

The resulting embeddings remain very close to f32 output, with cosine similarity around 0.99999, but they are not bit-identical.

bf16 therefore remains opt-in. The default f32 mode produces consistent vectors across machines, which is useful when embeddings generated on different hosts share the same index.

Unsupported CPUs, including Apple Silicon, reject `--precision bf16` at startup rather than silently falling back to f32.

## Accuracy and reproducibility

kohagi's f32 output matches the Sentence Transformers and PyTorch reference implementation to within f32 rounding error.
On 512-token inputs, `1 - cosine ≈ 3e-12`.

You can verify this on your own texts using [`examples/parity_check.py`](examples/parity_check.py).
See [`examples/README.md`](examples/README.md) for the measured results and the three settings that must match for the comparison to be meaningful.

## The name

In Saeko Himuro’s Heian-era novel series *Nante Suteki ni Japonésque* (『なんて素敵にジャパネスク』), the heroine, Ruri-hime (瑠璃姫), has a lady-in-waiting named Kohagi (小萩).

---

(in Japanese)

## kohagi (小萩)

kohagiは[Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-130m) などのModernBERT系文埋め込みモデルをローカル環境で動かすためのCLI/Rustライブラリです。
使い方はシンプルで、標準入力に`{"id","text"}`のJSONLを流すと、標準出力に`{"id","embedding"}`を返します。
外部サービス等を使用せず、バイナリひとつで動作します。

```bash
# インストール(リリースのバイナリ、または cargo install kohagi)
kohagi --text "瑠璃も玻璃も照らせば光る"          # 動作確認
kohagi --prefix "検索文書: " < in.jsonl > out.jsonl  # 本番はこちら
```

- モデルは初回には Hugging Face Hub から自動ダウンロードします (`--model-path`/`--tokenizer-path` でオフライン運用も可)
- x86_64 (AVX512-BF16 搭載の Zen 4 / Sapphire Rapids 以降)では `--precision bf16` で 1.5〜1.9 倍高速化します(cosine ≈ 0.99999、既定は f32。精度は若干落ちます)
- Apple Silicon では `--features metal` でビルドすると `--device metal` が使え、CPU の約1.2倍で動きます(出力は f32 のまま変わりません)
- 出力は f32 で PyTorch / sentence-transformers と一致するのを確認しています (cosine ≈ 1.0)
- メモリ使用量は入力サイズによらず一定になるようにしました (チャンク処理+attention 予算キャップ)
- 入出力の契約・exit code(0/2/1)は [PROTOCOL.md](PROTOCOL.md) を参照してください。
  Rails からの呼び出し例は [`examples/rails_open3.rb`](examples/rails_open3.rb) にあります。

なおmacOSで隔離属性のせいで起動がブロックされた場合は以下を実行して解除してください。

```bash
xattr -dr com.apple.quarantine ~/.local/bin/kohagi
```


kohagiの名前は氷室冴子『なんて素敵にジャパネスク』に登場する、瑠璃姫の女房である小萩に由来します。

## License

MIT
