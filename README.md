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

### Other models

Kohagi can also run other ModernBERT-based sentence encoders available on the Hugging Face Hub.
For example, you can use [nomic-ai/modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base) for English-language retrieval:

```bash
kohagi --model-id nomic-ai/modernbert-embed-base \
       --prefix "search_document: " < texts.jsonl
```

`cl-nagoya/ruri-v3-310m`, which produces 768-dimensional vectors, works in the
same way. To check whether a given model produces usable embeddings under
Kohagi, run [`examples/model_check.py`](examples/model_check.py) against it.

Pooling is taken from the model. Kohagi reads the checkpoint's
`1_Pooling/config.json` and uses the mode it declares, so a CLS-pooled model
such as `Alibaba-NLP/gte-modernbert-base` needs no flag. Pass `--pooling` only
to override, and Kohagi warns if your choice disagrees with the checkpoint —
or if the model ships no pooling config at all, which usually means it is a
reranker or a base LM rather than a sentence encoder.

For offline environments, specify local model files instead. In this mode, Kohagi does not make any network requests:

```bash
kohagi --model-path models/ruri-v3-130m/model.safetensors \
       --tokenizer-path models/ruri-v3-130m/tokenizer.json
```

Kohagi expects `config.json` to be located in the same directory as the model
weights. A `1_Pooling/config.json` beside them is read too if present; without
it, pass `--pooling` for a CLS model, since there is nothing to detect from.

### Long inputs and truncation

Text longer than `--max-seq-length` (512 tokens by default) is truncated before
embedding, so its vector reflects only the beginning. This is silent by design —
the summary line on stderr always ends with `truncated=N`, and `--report-tokens`
adds `n_tokens` and `truncated` to each output record so a caller can route the
truncated ones to a chunking pass:

```console
$ echo '{"id": 1, "text": "…a very long document…"}' | kohagi --report-tokens
{"id":1,"embedding":[…],"n_tokens":512,"truncated":true}
```

Raising `--max-seq-length` embeds more of each text at a quadratic cost in
attention compute. See [PROTOCOL.md](PROTOCOL.md) for the field definitions.

## Calling Kohagi from another language

Launch Kohagi as a subprocess, write JSONL records to its standard input, and read JSONL results from its standard output.

Read the output concurrently, such as from a separate thread, to prevent the pipe buffer from filling up and blocking the process.
Use the `id` field to match each result with its input record.

A complete Ruby example is available in [`examples/rails_open3.rb`](examples/rails_open3.rb).
See [PROTOCOL.md](PROTOCOL.md) for the exit-code semantics.

### An OpenAI-compatible endpoint

If the calling code is already written against OpenAI's `/v1/embeddings`, the
value of that compatibility is swapping `base_url` and changing nothing else.
Kohagi has no HTTP mode, so the examples supply one — the same ~150-line proxy in
[Python](examples/openai_proxy.py), [Ruby](examples/openai_proxy.rb) and
[TypeScript](examples/openai_proxy.ts):

```bash
python3 examples/openai_proxy.py --kohagi ./target/release/kohagi
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
ignored — the flags passed to Kohagi decide which checkpoint runs.

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

## Performance notes

* CPU by default, via Apple Accelerate on macOS, which performs within about 20% of PyTorch with equivalent output. Linux links no BLAS at all — candle's pure-Rust `gemm` does the matrix multiplies — so `--precision bf16` is where the Linux throughput is.
* Batches run in parallel across physical CPU cores. Set `RAYON_NUM_THREADS` to override the default; additional threads may improve throughput at the cost of memory.
* `--max-seq-length` has the largest effect on throughput because attention cost grows quadratically with sequence length.

Throughput is worth measuring on your own machine and texts rather than taking
numbers on faith. [`examples/benchmark.py`](examples/benchmark.py) times Kohagi against
Sentence Transformers on the same corpus and settings; see
[`examples/README.md`](examples/README.md) for measured results on Apple Silicon.

### `--device metal` on Apple Silicon

Building with `--features metal` adds an Apple GPU backend. On an M2 it runs
about 1.8× faster than the Accelerate CPU path — measured on 512-token
batches — with f32 output unchanged (worst `1 - cosine` 9e-13 against CPU).

The changes live in Kohagi's own copy of the ModernBERT encoder
([`src/encoder.rs`](src/encoder.rs)), so any build carries them, including
`cargo install`. They are what makes the Metal path win rather than a CPU
speedup: on CPU the difference measured smaller than the run-to-run noise,
though peak RSS does drop.

### `--device coreml` on the Apple Neural Engine

Build with `--features coreml`. On an M2 it is about 4× faster than Metal at 512
tokens, with cosine similarity of approximately 0.99999 against the CPU output.
For short inputs the multicore CPU backend may still be faster.

The ANE needs fixed input shapes, so it runs a converted model rather than the
safetensors the other devices read. A release build converts one for itself:

```bash
kohagi --device coreml < texts.jsonl
kohagi --device coreml --model-id answerdotai/ModernBERT-large < texts.jsonl
```

The first run downloads the checkpoint, converts it (~20 s) and compiles it;
later runs load in about 0.3 s from `~/Library/Caches/kohagi/coreml`
(`$KOHAGI_COREML_CACHE` to relocate), which is safe to delete. A checkpoint the
converter cannot honour is refused before anything is written, naming every
reason at once.

`--coreml-buckets` sets the sequence lengths (default `64,128,256,512`; the
largest caps `--max-seq-length`, and 4096 is the longest the converter will
produce).
`--coreml-quantize embeddings` roughly halves a large-vocabulary bundle at no
measured retrieval cost, but a quantized bundle's vectors are not interchangeable
with an fp16 one's, so the two must not share an index — which is why it is not
the default.

To convert ahead of time instead, into a directory to publish or share:

```bash
cargo run --release --bin coreml-convert --features coreml-export -- \
    --model-id cl-nagoya/ruri-v3-130m --out-dir models/ruri-v3-130m-coreml \
    --sequence-lengths 64,128,256,512

kohagi --device coreml --coreml-dir models/ruri-v3-130m-coreml < texts.jsonl
kohagi --device coreml --coreml-model-id takahashim/ruri-v3-130m-coreml < texts.jsonl
```

The lengths share one copy of the weights, so the set costs no disk — four
buckets are the same 260 MB as three. What it costs is one model to open per
length: going from `128,256,512` to the default's `64,128,256,512` took load
from 0.48 s to 0.56 s, and paid for itself after about a hundred short texts by
cutting the per-text cost from 4.3 ms to 3.5 ms. **Match the set to the lengths
your texts actually are.** A bucket nothing lands in is pure overhead — adding
192 to the default, on a corpus where every text is under 32 tokens, cost 0.25 s
of load and bought nothing. [`scripts/convert_coreml.py`](scripts/convert_coreml.py)
does the same conversion through PyTorch and `coremltools`, bit-identical for
`cl-nagoya/ruri-v3-130m`; every model published for Kohagi so far was made with it.

Measured against PyTorch on the same machine — M2, `ruri-v3-130m`, the default
buckets, median of three runs, from `examples/benchmark.py`:

| Input                   |    kohagi (CPU) | kohagi (`--device coreml`) |    torch (MPS) |
| ----------------------- | --------------: | -------------------------: | -------------: |
| 1200 short (~30 tokens) |  7.1 s / 7.4 s  |       **4.0 s / 4.7 s**    | 4.3 s / 13.7 s |
| 240 long (512 tokens)   | 30.8 s / 31.5 s |       **5.9 s / 6.6 s**    | 15.2 s / 24.9 s|

Encode / total, where total adds startup and model load. At 512 tokens the ANE
encodes 2.6× faster than torch/MPS and 5.2× faster than Kohagi's own CPU path.
On short inputs the two are within noise of each other on encode, because what
the ANE gains per token it gives back padding each row to its bucket. The totals
go the other way throughout: torch spends 9–10 s importing and loading per
process against Kohagi's under a second, so a rake task or a per-batch
subprocess sees 2.9× (short) and 3.8× (long).

These moved by a factor of two between runs on the same machine while other work
was going on, so treat them as an order of magnitude rather than a ranking, and
compare against a run of your own.

### `--precision bf16` on AVX512-BF16 CPUs

On Zen 4 (Sapphire Rapids) and newer CPUs, `--precision bf16` uses `bf16` for projection layers while keeping normalization, softmax, and attention scores in `f32`.

Measured on a Ryzen 7 8745H (Zen 4, 8 cores) running Linux, `ruri-v3-130m`,
median of five runs alternating between the two precisions
(`examples/benchmark.py --precision bf16 --skip-torch` produces the times;
peak RSS is from `/usr/bin/time -v`). Times are totals, including startup and
model load. The f32 column drifts a few percent between sessions, so read the
ratios rather than the seconds.

| Input                    |    f32 |              bf16 |          Peak RSS |
| ------------------------ | -----: | ----------------: | ----------------: |
| 1200 short (~30 tokens)  | 11.2 s |  **4.9 s** (2.3×) | 1.19 GB → 0.87 GB |
| 240 long (512 tokens)    | 44.2 s | **22.2 s** (2.0×) | 1.31 GB → 1.06 GB |

Less than half of that is the bf16 arithmetic. The rest comes from two things
a bf16 build also gets, both still f32: AVX-512 kernels for the softmax and
the GELU, which candle evaluates one element at a time
([`src/bf16/softmax.rs`](src/bf16/softmax.rs),
[`src/bf16/geglu.rs`](src/bf16/geglu.rs)), and skipping the three quarters of
the score matrix that ruri-v3's sliding-window layers mask off anyway. What
stays f32 and unfused is the `q·kᵀ` and `att·v` matmuls, which is why the
long row gains less than the short one.

bf16 also pays about a second more at load, converting the weights, which
matters if you spawn a process per small batch.

The resulting embeddings remain very close to f32 output, with cosine similarity around 0.99999, but they are not bit-identical.

bf16 therefore remains opt-in. The default f32 mode produces consistent vectors across machines, which is useful when embeddings generated on different hosts share the same index.

Unsupported CPUs, including Apple Silicon, reject `--precision bf16` at startup rather than silently falling back to f32.

## Accuracy and reproducibility

Kohagi's f32 output matches the Sentence Transformers and PyTorch reference implementation to within f32 rounding error.
On 512-token inputs, `1 - cosine ≈ 3e-12`.

You can verify this on your own texts using [`examples/parity_check.py`](examples/parity_check.py).
See [`examples/README.md`](examples/README.md) for the measured results and the three settings that must match for the comparison to be meaningful.

## The name

In Saeko Himuro’s Heian-era novel series *Nante Suteki ni Japonésque* (『なんて素敵にジャパネスク』), the heroine, Ruri-hime (瑠璃姫), has a lady-in-waiting named Kohagi (小萩).

---

(in Japanese)

## Kohagi (小萩)

Kohagiは[Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-130m) などのModernBERT系文埋め込みモデルをローカル環境で動かすためのCLI / Rustライブラリです。
使い方はシンプルで、標準入力に`{"id","text"}`のJSONLを流すと、標準出力に`{"id","embedding"}`を返します。
外部サービス等を使用せず、バイナリ単体で動作します。GPUも不要です。

```bash
# インストール(リリースのバイナリ、または cargo install kohagi)
kohagi --text "瑠璃も玻璃も照らせば光る"          # 動作確認
kohagi --prefix "検索文書: " < in.jsonl > out.jsonl  # 本番はこちら
```

- モデルは初回のみ Hugging Face Hub から自動ダウンロードします (`--model-path`/`--tokenizer-path` でオフライン運用も可)
- x86_64 (AVX512-BF16 搭載の Zen 4 / Sapphire Rapids 以降)では `--precision bf16` で約 2 倍高速化します(短文 2.3 倍、512 トークン 2.0 倍、cosine ≈ 0.99999、既定は f32。精度は若干落ちます)
- Apple Silicon では `--features metal` でビルドすると `--device metal` が使え、512トークンで CPU の約1.8倍で動きます(出力は f32 のまま変わりません)
- 同様に `--features coreml` でビルドすると `--device coreml` が使え、事前変換したモデルを Apple Neural Engine (ANE) 上で動かせます。512トークンで Metal の約4倍、PyTorch (MPS) 比でも埋め込み処理が3.0倍で、CPU 出力に対し cosine ≈ 0.99999 です(短い入力では ANE が固定長にパディングする分、PyTorch (MPS) やマルチコア CPU の方が速いこともあります)。起動時間を含めた合計では短文2.8倍・512トークン4.4倍になります。ローカルの変換済みモデルは `--coreml-dir`、Hugging Face Hub 上の同じ構成は `--coreml-model-id` で指定します
- 出力は f32 で PyTorch / sentence-transformers と一致するのを確認しています (cosine ≈ 1.0)
- 入出力の契約・exit code(0/2/1)は [PROTOCOL.md](PROTOCOL.md) を参照してください。
  Rails からの呼び出し例は [`examples/rails_open3.rb`](examples/rails_open3.rb) にあります。

また、Ruby では Kohagi 専用の gem である [kohagi-ruby](https://github.com/takahashim/kohagi-ruby)が使えます。

なおmacOSで隔離属性のせいで起動がブロックされた場合は以下を実行して解除してください。

```bash
xattr -dr com.apple.quarantine ~/.local/bin/kohagi
```


Kohagiの名前は氷室冴子『なんて素敵にジャパネスク』に登場する、瑠璃姫の女房である小萩に由来します。

## License

MIT
