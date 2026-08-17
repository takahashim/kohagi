# CoreML bundles

- The Apple Neural Engine runs fixed input shapes, so `--device coreml` runs a
  *converted bundle* rather than the safetensors every other device reads.
- A bundle is a directory holding:
  - one model per sequence length ("bucket"),
  - the `config.json` and `tokenizer.json` it was converted with,
  - and, for a reranker, `head.safetensors`.
- Build with `--features coreml`.

## Converting on first use

- A release build converts a checkpoint for itself.

```bash
kohagi --device coreml < texts.jsonl
kohagi --device coreml --model-id answerdotai/ModernBERT-large < texts.jsonl
kohagi-rerank --device coreml < pairs.jsonl
```

- The first run downloads the checkpoint, converts it and compiles it;
  later runs load from the cache.
- A checkpoint the converter cannot honour is refused before anything is
  written, naming every reason at once.
- The cache lives in `~/Library/Caches/kohagi/coreml`, relocatable with
  `$KOHAGI_COREML_CACHE`, and is safe to delete.
- Its key includes the checkpoint's revision and the emitted graph's version, so
  a stale bundle is never mistaken for a fresh one.

## Choosing bucket lengths

- `--coreml-buckets` sets the sequence lengths.
- Defaults: `64,128,256,512` for `kohagi`, `128,256,512` for `kohagi-rerank`,
  whose pairs fill more of a bucket than a single text does.
- The largest bucket caps `--max-seq-length`, and 4096 is the longest the
  converter will produce.
- Each text is routed to the smallest bucket it fits and padded to exactly that
  length. **Match the set to the lengths your texts actually are.**
- The lengths share one copy of the weights, so the set costs no disk: four
  buckets are the same 260 MB as three.
- What it costs is one model to open per length, and padding for every text that
  lands in a bucket much larger than itself:
  - Going from `128,256,512` to the default `64,128,256,512` took load from
    0.48 s to 0.56 s, and paid for itself after about a hundred short texts by
    cutting the per-text cost from 4.3 ms to 3.5 ms.
  - Adding `192` to the default, on a corpus where every text was under 32
    tokens, cost 0.25 s of load and bought nothing. A bucket nothing lands in is
    pure overhead.

## Quantization

- `--coreml-quantize embeddings` stores the embedding table as int8.
- `all` quantizes every projection too.
- It is not the default, and should not be turned on casually: **a quantized
  bundle's vectors are not interchangeable with an fp16 one's**, so the two must
  not share an index.
- Loading a quantized bundle says so, and `--print-model-info` reports
  `quantization` so a results file can record which one produced its numbers.

## Converting ahead of time

- To produce a directory you can inspect, share or publish:

```bash
cargo run --release --bin coreml-convert --features coreml-export -- \
    --model-id cl-nagoya/ruri-v3-130m --out-dir models/ruri-v3-130m-coreml \
    --sequence-lengths 64,128,256,512
```

- Then load it from a directory, or from a Hub repo holding the same layout:

```bash
kohagi --device coreml --coreml-dir models/ruri-v3-130m-coreml < texts.jsonl
kohagi --device coreml --coreml-model-id takahashim/ruri-v3-130m-coreml < texts.jsonl
```

- `--coreml-dir` wins if both are given.
- When a Hub repo ships both a compiled `.mlmodelc` and a portable `.mlpackage`
  for a bucket, `--coreml-prefer` chooses; only the chosen form is downloaded.

## What a bundle records about itself

- A converted bundle carries the checkpoint it came from, as a Hub id or path
  and as that checkpoint's sha256.
- A bundle holds fp16 copies of the weights rather than the weights themselves,
  so the source digest is the only fingerprint it can offer:

```console
$ kohagi --device coreml --coreml-dir models/ruri-v3-130m-coreml --print-model-info
{"model":"ruri-v3-130m-coreml","backend":"coreml","source":"cl-nagoya/ruri-v3-130m","source_sha256":"1c342581…","buckets":[64,128,256,512],"quantization":"none",…}
```

## When CoreML cannot serve a request

- `--device coreml` exits `3` rather than `1` when the backend cannot serve what
  was asked:
  - the binary was built without the feature,
  - `--max-seq-length` runs past the largest converted bucket,
  - the bundle has no head to score a pair with,
  - or the model is missing.
- It is detected before any output is produced, so a caller can retry on
  `--device cpu`.

