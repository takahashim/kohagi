#!/usr/bin/env python3
"""Check kohagi-rerank against the sentence-transformers CrossEncoder.

Same claim as tools/parity_check.py makes for embeddings: kohagi-rerank's f32
score *is* the reference score, to f32 rounding. This verifies it on your own
machine and pairs.

    pip install sentence-transformers
    python tools/rerank_parity.py --kohagi ./target/release/kohagi-rerank

Three settings have to match on both sides or the comparison means nothing:

- **max_length** — kohagi-rerank truncates at 512 by default; CrossEncoder uses
  the tokenizer's `model_max_length` (8192 for these rerankers) unless told
  otherwise. A pair longer than the shorter limit is then scored from
  different text on each side. Both are set explicitly here.
- **the sigmoid** — CrossEncoder applies one when the model has a single
  label, which every reranker here does, and so does kohagi-rerank by default.
  `--raw-logits` turns both off together.
- **the pair order** — (query, text), not (text, query). A cross-encoder is
  not symmetric, and swapping them changes the score without failing.

The sample pairs deliberately include a long passage and a code-shaped one:
truncation and code are where a reranker's behaviour is least like its
demonstration examples.
"""

import argparse
import json
import subprocess
import sys

QUERY = "Rubyで配列を並べ替えるには"

SAMPLE_PAIRS = [
    # A passage that answers it.
    (QUERY, "配列の並べ替えには sort と sort_by がある。sort はブロックで比較を指定でき、"
            "sort_by は各要素から取り出したキーで並べ替える。破壊的に並べ替えるなら sort! を使う。"),
    # Related but not an answer.
    (QUERY, "ハッシュは키と値の組を保持するデータ構造で、each で走査すると挿入順に取り出せる。"),
    # Unrelated.
    (QUERY, "駅前の駐輪場が不足しているため、増設を要望します。"),
    # Code, which rerankers read badly and which the mining pipeline hits often.
    (QUERY, "```ruby\n[3, 1, 2].sort  # => [1, 2, 3]\n[3, 1, 2].sort_by { |n| -n }\n```"),
    # Long enough to be truncated at 512 tokens, so both sides must trim the
    # same half of the pair in the same way.
    (QUERY, "配列の操作について説明する。" * 200),
    # A different query, so the batch holds more than one.
    ("感動的な映画について",
     "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kohagi", default="kohagi-rerank", help="path to the kohagi-rerank binary")
    p.add_argument("--model-id", default="hotchpotch/japanese-reranker-xsmall-v2")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--precision", default="f32", choices=["f32", "bf16"])
    p.add_argument("--device", default="cpu", help="kohagi-rerank --device")
    p.add_argument(
        "--raw-logits",
        action="store_true",
        help="compare logits instead of sigmoid scores (turned off on both sides)",
    )
    p.add_argument(
        "--pairs",
        help="JSONL file of {\"query\", \"text\"} pairs (default: built-in samples)",
    )
    args = p.parse_args()

    pairs = SAMPLE_PAIRS
    if args.pairs:
        with open(args.pairs, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        pairs = [(r["query"], r["text"]) for r in rows]

    # --- kohagi-rerank, over its stdio protocol ------------------------------
    stdin = "".join(
        json.dumps({"id": i, "query": q, "text": t}, ensure_ascii=False) + "\n"
        for i, (q, t) in enumerate(pairs)
    )
    cmd = [
        args.kohagi,
        "--model-id", args.model_id,
        "--max-seq-length", str(args.max_seq_length),
        "--precision", args.precision,
        "--device", args.device,
    ]
    if args.device == "coreml":
        # One bucket, matching --max-seq-length: a pair fills a long bucket, and
        # a comparison should not depend on which one it landed in.
        cmd += ["--coreml-buckets", str(args.max_seq_length)]
    if args.raw_logits:
        cmd.append("--raw-logits")
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if proc.returncode == 1:
        sys.stderr.write(proc.stderr)
        return 1
    got = {json.loads(l)["id"]: json.loads(l)["score"] for l in proc.stdout.splitlines()}
    mine = [got[i] for i in range(len(pairs))]

    # --- the reference -------------------------------------------------------
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        args.model_id,
        device="cpu",
        max_length=args.max_seq_length,
        # None keeps the library's own default, which is a sigmoid for a
        # one-label model. Identity is what --raw-logits compares against.
        activation_fn=torch.nn.Identity() if args.raw_logits else None,
    )
    ref = [float(s) for s in model.predict([(q, t) for q, t in pairs])]

    # --- compare -------------------------------------------------------------
    #
    # Against the scale of the scores rather than in absolute terms. A sigmoid
    # score lives in 0..1 and a logit runs to ±8, so the same rounding shows up
    # ~8x larger in the second — measured: worst |diff| 5.1e-07 on sigmoid
    # scores and 4.1e-06 on the logits behind them, which is one phenomenon
    # seen at two magnitudes, not two accuracies. Dividing by the range makes
    # them comparable, and both land at ~5e-07.
    worst = max(abs(a - b) for a, b in zip(ref, mine))
    scale = max(1.0, max(abs(r) for r in ref))
    relative = worst / scale

    print(f"model      : {args.model_id} ({args.precision}, {args.device})")
    print(f"score      : {'logit' if args.raw_logits else 'sigmoid'}")
    print(f"pairs      : {len(pairs)}")
    print()
    print(f"{'reference':>14}  {'kohagi-rerank':>14}  {'|diff|':>9}")
    for a, b in zip(ref, mine):
        print(f"{a:>14.7f}  {b:>14.7f}  {abs(a - b):>9.2e}")
    print(f"\nworst |diff| {worst:.3e} over a score range of {scale:.3g}")

    # ~5e-07 of the range is what an f32 forward through 10 to 25 layers plus a
    # linear head gives; the embedding path's `1 - cos ~ 3e-12` is the same
    # per-element agreement measured on unit vectors.
    #
    # The other two paths are different tradeoffs, not different accuracies of
    # the same one, and are checked as such. `--device coreml` runs the encoder
    # in fp16 on the Neural Engine (the head stays f32 on the CPU): measured
    # over 120 pairs spanning 0.0001 to 0.9995, scores moved by 1.1e-04 on
    # average and 4.9e-03 at worst, with no pair crossing 0.02, 0.1, 0.5 or 0.6
    # and 4 of 840 within-query orderings changing. Judge a converted bundle by
    # that behaviour rather than by this number alone.
    if args.device == "coreml":
        limit = 1e-2
    elif args.precision == "f32":
        limit = 2e-6
    else:
        limit = 1e-2
    ok = relative < limit
    print(f"{'OK' if ok else 'FAIL'}: {relative:.3e} of range {'<' if ok else '>='} {limit:.0e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
