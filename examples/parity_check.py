"""Check kohagi against the sentence-transformers / PyTorch reference.

kohagi's claim is that its f32 output *is* the reference output, to f32
rounding. This script verifies that on your own machine and texts.

    pip install sentence-transformers
    python examples/parity_check.py --kohagi ./target/release/kohagi

Both sides must be configured identically or the comparison is meaningless.
The three settings that actually bite:

- **prefix** — must match exactly, trailing space included. A missing space
  moves `1 - cosine` to ~3e-3, ten orders of magnitude above the real
  difference. This is the single most common cause of a "mismatch".
- **max_seq_length** — kohagi defaults to 512; sentence-transformers uses
  whatever `sentence_bert_config.json` says (8192 for ruri-v3). Texts longer
  than the shorter limit get truncated on one side only.
- **pooling** — must match the model's `1_Pooling/config.json` (mean for
  ruri-v3 and modernbert-embed).

Cosine is accumulated in float64 on purpose: two f32 vectors this close
saturate f32 arithmetic at `1 - cosine ≈ 1.2e-7`, which measures the
accumulator rather than the vectors.

See examples/README.md for the measured numbers and the reasoning.
"""

import argparse
import json
import subprocess
import sys

import numpy as np
from sentence_transformers import SentenceTransformer, models

SAMPLE_TEXTS = [
    "瑠璃も玻璃も照らせば光る",
    "犬も歩けば棒に当たる",
    "この製品は組み立てが簡単で、説明書も分かりやすかったです。",
    "駅前の駐輪場が不足しているため、増設を要望します。" * 8,
    "長い文章のテスト。" * 200,
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kohagi", default="kohagi", help="path to the kohagi binary")
    p.add_argument("--model-id", default="cl-nagoya/ruri-v3-130m")
    p.add_argument("--prefix", default="検索文書: ")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--precision", default="f32", choices=["f32", "bf16"])
    p.add_argument(
        "--pooling",
        default="model",
        choices=["model", "mean", "cls"],
        help="'model' uses the checkpoint's own 1_Pooling config on both sides; "
        "'mean'/'cls' force that mode on both sides instead",
    )
    p.add_argument("--texts", help="file with one text per line (default: built-in samples)")
    args = p.parse_args()

    texts = SAMPLE_TEXTS
    if args.texts:
        with open(args.texts, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]

    # --- kohagi, over its stdio protocol -------------------------------------
    stdin = "".join(
        json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n" for i, t in enumerate(texts)
    )
    cmd = [
        args.kohagi,
        "--model-id", args.model_id,
        "--prefix", args.prefix,
        "--max-seq-length", str(args.max_seq_length),
        "--precision", args.precision,
    ]
    if args.pooling != "model":
        cmd += ["--pooling", args.pooling]
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if proc.returncode == 1:
        sys.stderr.write(proc.stderr)
        return 1
    got = {json.loads(l)["id"]: json.loads(l)["embedding"] for l in proc.stdout.splitlines()}
    mine = np.array([got[i] for i in range(len(texts))], dtype=np.float64)

    # --- the reference -------------------------------------------------------
    if args.pooling == "model":
        model = SentenceTransformer(args.model_id)
    else:
        # Rebuild the stack so the pooling mode is ours rather than the
        # checkpoint's, which is the only way to exercise a mode the model was
        # not published with.
        body = models.Transformer(args.model_id, max_seq_length=args.max_seq_length)
        head = models.Pooling(body.get_word_embedding_dimension(), pooling_mode=args.pooling)
        model = SentenceTransformer(modules=[body, head])
    model.max_seq_length = args.max_seq_length  # match kohagi's truncation
    ref = model.encode(
        [args.prefix + t for t in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float64)

    # --- compare -------------------------------------------------------------
    cos = (ref * mine).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(mine, axis=1))
    worst = 1 - cos.min()
    print(f"model      : {args.model_id} ({ref.shape[1]} dims, {args.precision})")
    print(f"pooling    : {args.pooling}")
    print(f"texts      : {len(texts)}")
    print(f"mean 1-cos : {1 - cos.mean():.3e}")
    print(f"worst 1-cos: {worst:.3e}")
    print(f"max |diff| : {np.abs(ref - mine).max():.3e}")

    # f32 rounding through a 22-layer encoder lands around 1e-11; bf16 is a
    # different tradeoff and is checked far more loosely.
    limit = 1e-9 if args.precision == "f32" else 1e-3
    ok = worst < limit
    print(f"\n{'OK' if ok else 'FAIL'}: worst 1-cos {worst:.3e} {'<' if ok else '>='} {limit:.0e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
