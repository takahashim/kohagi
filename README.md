# Kohagi

A local sentence-embeddings CLI for [Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-130m) and other ModernBERT encoders.

Kohagi reads JSONL records from standard input and writes the corresponding embedding vectors as JSONL.
It runs as a single executable and requires no cloud services or embedding server.

```console
$ echo '{"id": 1, "text": "瑠璃も玻璃も照らせば光る"}' | kohagi
{"id":1,"embedding":[0.006987,-0.032139, …]}
```

Kohagi is a small CLI and Rust library built with [Candle](https://github.com/huggingface/candle).
It is designed for one job: embedding text in batches from any environment that can launch a subprocess, such as a Rails rake task, a Node.js script, or a shell pipeline.

### Why use Kohagi?

- **Pure Rust and a single executable.** No PyTorch, LibTorch, ONNX Runtime, or Python environment required. Supports macOS, Linux, and Windows x64 (CUDA on NVIDIA GPUs).
- **Accurate.** Uses f32 inference and closely matches the PyTorch and Sentence Transformers reference implementations  (cosine ≈ 1.0).
- **Bounded memory usage.** Attention scratch space is capped for each forward pass, and input is processed in chunks. Peak memory usage therefore depends primarily on the number of CPU cores, not on the total amount of input.
- **A deliberately simple, stable interface.** `{"id","text"}` in, `{"id","embedding"}` out, with exit codes `0`, `2`, and `1`. See [PROTOCOL.md](PROTOCOL.md).

## Install

Prebuilt binaries for macOS (Apple Silicon), Linux (x86_64), and Windows x64
(NVIDIA CUDA) are on the [releases page](https://github.com/takahashim/kohagi/releases).
On macOS and Linux:

```bash
tar -xzf kohagi-<target>.tar.gz && mv kohagi ~/.local/bin/
```

On Windows, extract `kohagi-x86_64-pc-windows-msvc.zip` and run `kohagi.exe`.

The binaries are unsigned, so unpacking from Finder on macOS leaves a quarantine
attribute that Gatekeeper blocks. Extracting with `tar` as above avoids it;
otherwise run `xattr -dr com.apple.quarantine ~/.local/bin/kohagi`.

Or with cargo:

```bash
cargo install kohagi
```

### NVIDIA GPU on Windows

Kohagi can use an NVIDIA GPU through CUDA on Windows x64. Install a compatible
NVIDIA driver and CUDA runtime, then use the Windows release binary with
`--device cuda`:

```powershell
kohagi.exe --device cuda --prefix "検索文書: " < texts.jsonl > embeddings.jsonl
```

## Quick start

By default, Kohagi uses `cl-nagoya/ruri-v3-130m` (a Japanese sentence-embedding model, 512-dimensions).
The model is downloaded from the Hugging Face Hub the first time Kohagi runs and is cached under `~/.cache/huggingface`.

```bash
# Run a quick sanity check without constructing JSONL:
kohagi --text "瑠璃も玻璃も照らせば光る" --text "犬も歩けば棒に当たる"

# For normal use, stream JSONL through standard input and output:
kohagi --prefix "検索文書: " < texts.jsonl > embeddings.jsonl
```

### Ruri v3 prefixes

Ruri v3 is trained to use task-specific prefixes.
Kohagi prepends the value of `--prefix` to every input text, allowing callers to pass the original text unchanged.

| Task                              | `--prefix`                    |
| --------------------------------- | ----------------------------- |
| General sentence similarity       | *(none; this is the default)* |
| Document to be indexed for search | `"検索文書: "`                    |
| Search query                      | `"検索クエリ: "`                   |
| Topic or keyword                  | `"トピック: "`                    |

### Which model produced these vectors

Every run's summary line names the weights by content, not just by path:

```console
$ kohagi --model-path models/alpha05/model.safetensors --tokenizer-path models/alpha05/tokenizer.json < texts.jsonl > out.jsonl
kohagi: model=alpha05 sha256=1c342581efc2 pooling=mean dim=512 max_seq=512 in=2141 out=2141 skipped=0 truncated=3
```

`--print-model-info` gives the same facts as one line of JSON on stdout and
exits without embedding anything, for a script to record beside its results:

```console
$ kohagi --print-model-info
{"model":"cl-nagoya/ruri-v3-130m","backend":"cpu","precision":"f32","sha256":"1c342581efc2…","pooling":"mean","dim":512,"max_seq_length":512}
```

This matters as soon as there is more than one checkpoint: fine-tunes of one
model, or interpolations between two, differ only in bytes, and a results file
that records a directory name records what someone meant to load. The digest is
of the whole `model.safetensors`, so identical weights always agree and one
byte's difference always shows, and `sha256sum` on the same file gives the same
value. Hashing runs on its own thread beside the embedding and is collected at
the end, so a run doing real work pays nothing for it. The hash itself takes
0.36s of CPU for ruri-v3-130m's 528MB, and however long the disk takes for a
model on a network share.

The recorded digest can also be enforced. `--expect-sha256 1c342581efc2`
(paste the summary's 12 digits, or the full 64) refuses to embed anything with
weights whose digest does not start with it. The run exits 1 before any
output, so the wrong checkpoint cannot add a single vector to an index. Both
binaries take it; see PROTOCOL.md.

## Reranking with `kohagi-rerank`

Embedding search finds candidates; a cross-encoder reorders them. `kohagi-rerank`
is a second binary from the same crate that reads `{"id","query","text"}` and
writes `{"id","score"}`:

```console
$ echo '{"id":1,"query":"Rubyで配列を並べ替えるには","text":"配列の並べ替えには sort と sort_by がある。"}' | kohagi-rerank
{"id":1,"score":0.9283465}
```

It defaults to `cl-nagoya/ruri-v3-reranker-310m` and runs any ModernBERT
sequence-classification checkpoint with one label, including the
`hotchpotch/japanese-reranker-*-v2` family. The score is the sigmoid of the
model's logit, the same number `sentence_transformers.CrossEncoder.predict`
returns, so thresholds carry over; `--raw-logits` reports the logit instead.

A separate binary rather than a flag on `kohagi`, because it is a different
function: pairs in, numbers out. Everything around the records (opaque ids,
skipped lines, blank-line batches, exit codes) is the same protocol. See
[PROTOCOL-rerank.md](PROTOCOL-rerank.md).

```bash
# Converts the checkpoint for the ANE on first use and caches it.
kohagi-rerank --device coreml < pairs.jsonl > scores.jsonl
```

## Calling Kohagi from another language

Launch Kohagi as a subprocess, write JSONL records to its standard input, and read JSONL results from its standard output.

Read the output concurrently, such as from a separate thread, to prevent the pipe buffer from filling up and blocking the process.
Use the `id` field to match each result with its input record.

A complete Ruby example is available in [`examples/rails_open3.rb`](examples/rails_open3.rb).
See [PROTOCOL.md](PROTOCOL.md) for the exit-code semantics.

### An OpenAI-compatible endpoint

If the calling code is already written against OpenAI's `/v1/embeddings`, the
value of that compatibility is swapping `base_url` and changing nothing else.
Kohagi has no HTTP mode, so the examples supply one, the same ~150-line proxy in
[Python](examples/openai_proxy/proxy.py), [Ruby](examples/openai_proxy/proxy.rb) and
[TypeScript](examples/openai_proxy/proxy.ts):

```bash
python3 examples/openai_proxy/proxy.py --kohagi ./target/release/kohagi
```

```python
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
client.embeddings.create(model="ruri-v3-130m", input=["…", "…"])
```

Each keeps one Kohagi loaded and ends every request with a blank line, which is
Kohagi's "embed what you have and reply now" signal; with `--format openai` the
reply is that batch's complete response object, so there is nothing to assemble.
A request costs about 40 ms warm. Two caveats before pointing production at it: an
existing index has to be rebuilt, since `ruri-v3-130m` returns 512 dimensions
where `text-embedding-3-small` returns 1536, and the request's `model` is
ignored; the flags passed to Kohagi decide which checkpoint runs.

### Ruby

In Ruby, [kohagi-ruby](https://github.com/takahashim/kohagi-ruby)
packages that plumbing: it builds the command line from the CLI flags, spawns
the process without deadlocking on the pipe buffer, and turns the exit codes
into outcomes.

```ruby
require "kohagi"

client = Kohagi::Client.new(prefix: "検索文書: ")

records = [
  { id: 1, text: "瑠璃も玻璃も照らせば光る" },
  { id: 2, text: "犬も歩けば棒に当たる" },
]

# Each result is yielded as it arrives, so memory stays flat on any corpus.
summary = client.embed(records) do |result|
  store(result.id, result.embedding)   # e.g. a pgvector column
end

summary.dim        # => 512
summary.out        # => 2
summary.truncated  # => 0
```

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

## The name

In Saeko Himuro’s Heian-era novel series *Nante Suteki ni Japonésque* (『なんて素敵にジャパネスク』), the heroine, Ruri-hime (瑠璃姫), has a lady-in-waiting named Kohagi (小萩).

---

(in Japanese)

## Kohagi (小萩)

Kohagiは[Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-130m) などのModernBERT系文埋め込みモデルをローカル環境で動かすためのCLI / Rustライブラリです。
使い方はシンプルで、標準入力に`{"id","text"}`のJSONLを流すと、標準出力に`{"id","embedding"}`を返します。
外部サービス等を使用せず、バイナリ単体で動作します。CPUのみでも動作しますし、Apple Neural EngineやCUDAもサポートしています。

```bash
# インストール(リリースのバイナリ、または cargo install kohagi)
kohagi --text "瑠璃も玻璃も照らせば光る"          # 動作確認
kohagi --prefix "検索文書: " < in.jsonl > out.jsonl  # 本番はこちら
```

- モデルは初回のみ Hugging Face Hub から自動ダウンロードします (`--model-path`/`--tokenizer-path` でオフライン運用も可)
- `--dims N` で先頭 N 次元への切り詰め + 再正規化ができます(Matryoshka 学習済みモデル向け。sentence-transformers の `truncate_dim` と一致します)
- `--expect-sha256 <hex>` で「期待する重みかどうか」をロード時に検証できます(不一致なら出力を一切出さずに exit 1。チェックポイントの取り違え防止)
- x86_64 (AVX512-BF16 搭載の Zen 4 / Sapphire Rapids 以降)では `--precision bf16` で約 2 倍高速化します(cosine ≈ 0.99999、既定は f32。精度は若干落ちます)
- Apple Silicon では `--features metal` でビルドすると `--device metal` が使えます。CPUより高速です(出力は f32 のまま変わりません)
- 同様に `--features coreml` でビルドすると `--device coreml` が使え、Apple Neural Engine (ANE) 上で動かせます。長い入力ほど高速で、CPU 出力に対し cosine ≈ 0.99999 です(短い入力では ANE が固定長にパディングする分、PyTorch (MPS) やマルチコア CPU の方が速いこともあります)。ローカルの変換済みモデルは `--coreml-dir`、Hugging Face Hub 上のものは `--coreml-model-id` で指定します
- 出力は f32 で PyTorch / sentence-transformers と一致するのを確認しています (cosine ≈ 1.0)
- 入出力の契約・exit code(0/1/2/3)は [PROTOCOL.md](PROTOCOL.md) を参照してください。
  Rails からの呼び出し例は [`examples/rails_open3.rb`](examples/rails_open3.rb) に、OpenAI Embeddings API互換サーバのサンプルは[`examples/openai_proxy/proxy.py`](examples/openai_proxy/proxy.py)(Python)、[`examples/openai_proxy/proxy.rb`](examples/openai_proxy/proxy.rb)(Ruby)、[`examples/openai_proxy/proxy.ts`](examples/openai_proxy/proxy.ts)(TypeScript)にあります。

また、Ruby では Kohagi 専用の gem である [kohagi-ruby](https://github.com/takahashim/kohagi-ruby)が使えます。

なおmacOSで隔離属性のせいで起動がブロックされた場合は以下を実行して解除してください。

```bash
xattr -dr com.apple.quarantine ~/.local/bin/kohagi
```


Kohagiの名前は氷室冴子『なんて素敵にジャパネスク』に登場する、瑠璃姫の女房である小萩に由来します。

## License

MIT
