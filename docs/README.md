# Kohagi documentation

- **[Choosing a model](models.md)**: which checkpoints Kohagi runs, how pooling
  is decided, offline files, and how much of a long text the model sees.
- **[Choosing a device](devices.md)**: what `cpu`, `bf16`, `metal`, `cuda` and
  `coreml` each cost, and what each guarantees about the numbers.
- **[CoreML bundles](coreml.md)**: converting for the Apple Neural Engine,
  caching, bucket lengths, quantization, and publishing a bundle.
- **[Retrieval and reranking](reranking.md)**: using `kohagi` and
  `kohagi-rerank` together.

## Elsewhere

- The input and output contract (record shapes, batching, exit codes) is in
  [PROTOCOL.md](../PROTOCOL.md) and [PROTOCOL-rerank.md](../PROTOCOL-rerank.md).
- Every flag is in `kohagi --help` and `kohagi-rerank --help`.
