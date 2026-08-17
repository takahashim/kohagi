#!/usr/bin/env python3
"""How often a lower-precision backend moves a pair across a threshold.

`tools/rerank_parity.py --against` derives *where* a flip is possible: for a
threshold t and a logit error d, only pairs whose score is within d*t*(1-t) of
t can cross it. This measures *how often* it actually happens, by sampling from
those bands rather than from the corpus at large.

Sampling uniformly would be the wrong experiment. A band holds a few percent of
a corpus, so a few hundred uniform pairs put a handful in the only region where
the answer can be anything but zero, and observing zero flips there says
nothing — it is what a harmless backend and a catastrophic one both produce.

    python tools/rerank_fp16_bands.py \\
        --scores  /Volumes/Shared/tb-corpus/pair-scores.npy \\
        --pairs   /Volumes/Shared/tb-corpus/triplets-all.jsonl \\
        --model-id hotchpotch/japanese-reranker-xsmall-v2 \\
        --device coreml --against cpu

`--scores` is one score per line of `--pairs`, in the same order, produced by
the same model. The pair text is read back from `--pairs` by index, so the two
files must correspond — check the counts before believing anything here.

Reports, per threshold: the band population, the sample, how many pairs the two
backends put on opposite sides, and the flip count that implies for the whole
corpus. Set `--budget` to the number of flips that would matter to you and the
verdict is printed against it, so the criterion is written down before the
number is known.
"""

import argparse
import json
import random
import subprocess
import sys


def load_pairs(path, wanted):
    """The `query` and `positive` of the wanted line numbers, by one pass."""
    wanted = set(wanted)
    out = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted:
                row = json.loads(line)
                out[i] = (row["query"], row.get("positive") or row.get("text"))
                if len(out) == len(wanted):
                    break
    missing = wanted - set(out)
    if missing:
        sys.exit(
            f"{path} has no line {min(missing)} (asked for {len(wanted)} of them); "
            "does it correspond to --scores?"
        )
    return out


def score(binary, model_id, device, max_seq, pairs):
    """Scores for `pairs` (a list of (query, text)) on one device, in order."""
    stdin = "".join(
        json.dumps({"id": i, "query": q, "text": t}, ensure_ascii=False) + "\n"
        for i, (q, t) in enumerate(pairs)
    )
    cmd = [
        binary,
        "--model-id", model_id,
        "--device", device,
        "--max-seq-length", str(max_seq),
    ]
    if device == "coreml":
        cmd += ["--coreml-buckets", str(max_seq)]
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    # Anything but 0 stops the run. A flip rate counted over the pairs that
    # happened to survive a partial failure is a rate for a different sample
    # than the one the band was drawn from, and it would not look wrong.
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"kohagi-rerank exited {proc.returncode}")
    got = {json.loads(l)["id"]: json.loads(l)["score"] for l in proc.stdout.splitlines()}
    missing = [i for i in range(len(pairs)) if i not in got]
    if missing:
        raise SystemExit(f"kohagi-rerank returned no score for pair {missing[0]}")
    return [got[i] for i in range(len(pairs))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kohagi", default="kohagi-rerank")
    p.add_argument("--model-id", default="hotchpotch/japanese-reranker-xsmall-v2")
    p.add_argument("--scores", required=True, help=".npy of one score per pair")
    p.add_argument("--pairs", required=True, help="JSONL of the pairs those scores are of")
    p.add_argument("--device", default="coreml", help="the backend under test")
    p.add_argument("--against", default="cpu", help="the backend it is judged against")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--thresholds", default="0.02,0.1,0.5,0.6")
    p.add_argument("--width", type=float, default=0.01, help="half-width of the sampled band")
    p.add_argument("--sample", type=int, default=200, help="pairs sampled per band")
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument(
        "--budget",
        type=float,
        help="flips per threshold that would matter, as a count. Printed as a "
        "verdict; decide it before running.",
    )
    args = p.parse_args()

    import numpy as np

    scores = np.load(args.scores)
    rng = random.Random(args.seed)
    print(f"{args.scores}: {len(scores):,} scores")
    print(f"backend under test: --device {args.device} against --device {args.against}")
    print(f"model: {args.model_id}, {args.max_seq_length} tokens\n")

    # Every band's sample is chosen first, so the pair file is read once. It is
    # a gigabyte on a network share; four passes over it cost more than the
    # scoring does.
    chosen = {}
    for t in [float(x) for x in args.thresholds.split(",")]:
        in_band = np.flatnonzero((scores > t - args.width) & (scores < t + args.width))
        if len(in_band) == 0:
            print(f"threshold {t}: nothing within +/-{args.width}")
            continue
        take = min(args.sample, len(in_band))
        chosen[t] = (len(in_band), sorted(rng.sample(list(map(int, in_band)), take)))
    everything = sorted({i for _, picked in chosen.values() for i in picked})
    print(f"reading {len(everything):,} pairs from {args.pairs} ...", flush=True)
    texts = load_pairs(args.pairs, everything)

    rows = []
    for t, (population, picked) in chosen.items():
        take = len(picked)
        pairs = [texts[i] for i in picked]
        a = score(args.kohagi, args.model_id, args.against, args.max_seq_length, pairs)
        b = score(args.kohagi, args.model_id, args.device, args.max_seq_length, pairs)
        flips = sum(1 for x, y in zip(a, b) if (x >= t) != (y >= t))
        rate = flips / take
        expected = rate * population
        rows.append((t, population, take, flips, rate, expected))
        print(
            f"threshold {t}: band +/-{args.width} holds {population:,} pairs; "
            f"sampled {take}, {flips} flipped ({rate:.1%}) -> {expected:,.0f} expected over the band"
        )

    print("\n threshold  band population  sampled  flipped     rate   expected flips" +
          ("   verdict" if args.budget else ""))
    for t, population, take, flips, rate, expected in rows:
        line = (f"  {t:<9} {population:>15,} {take:>8} {flips:>8} {rate:>8.1%} {expected:>16,.0f}")
        if args.budget:
            line += "   " + ("OK" if expected < args.budget else "OVER")
        print(line)
    if args.budget:
        print(f"\nverdict against a budget of {args.budget:,.0f} flips per threshold, set before the run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
