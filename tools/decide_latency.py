#!/usr/bin/env python3
"""Latency of the Jev candidate-scoring path (plan-decide.md §6, latency row).

Two costs matter and this separates them: the model load a process pays once,
and the per-pair forward it pays per candidate. It also reports wall and child
CPU time side by side, because on a loaded machine wall is inflated by
contention while CPU time is not — and the ratio of the two is the machine's
noise made visible rather than a caveat in prose.

    python3 tools/decide_latency.py --device cpu \\
        --kohagi-rerank ./target/release/kohagi-rerank

    # Neural Engine: needs a CoreML build and a converted bundle
    cargo build --release --features coreml,coreml-export --bin kohagi-rerank
    python3 tools/decide_latency.py --device coreml \\
        --kohagi-rerank ./target/release/kohagi-rerank \\
        --coreml-dir ~/Library/Caches/kohagi/coreml/converted/.../<bundle>

Runs are interleaved across K and repeated, so a transient load spike lands on
every K alike and shows up as spread rather than as a wrong median. Per-question
is per-pair x K: the CoreML path runs batch=1 per pair, so its cost is linear in
the candidate count.

Recorded on an M2 (16 GB) on 2026-09-21, `argos1111/modernbert-ja-310m-jev`:

    CPU f32, 1 worker serial : short ~135 ms/pair, 512 tok ~2220 ms/pair
    CoreML (ANE, batch=1)    : short  ~15 ms/pair, 512 tok  ~75 ms/pair
    model load               : CPU ~2.5 s CPU / ~3.5 s wall; CoreML ~0.84 s wall

The CPU wall on that machine was ~3x the CPU time (load average 110-164, rspec
running); the CoreML path is mostly independent of CPU contention.
"""

import argparse
import json
import math
import resource
import statistics
import subprocess
import sys
import time

SHORT = ("質問: 担当部署は？\n状況: 二重請求です。返金してください。"
         "解決しないなら解約します。")
LONG = "質問: 担当部署は？\n状況: " + ("これは非常に長い状況説明です。" * 120)
CANDS = ["billing — 請求・返金", "technical — 技術的な障害", "sales — 新規契約・見積",
         "苦情対応", "解約手続き", "返金処理", "請求書の再発行", "その他"]


def child(cmd, stdin):
    """Run one kohagi-rerank process; return (wall_s, child_cpu_s)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    wall = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"kohagi-rerank exited {proc.returncode}")
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return wall, cpu


def stdin_of(query, k, questions):
    rows = [(query, CANDS[i % len(CANDS)]) for i in range(questions * k)]
    return "".join(json.dumps({"id": i, "query": q, "text": t}, ensure_ascii=False) + "\n"
                   for i, (q, t) in enumerate(rows)), len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kohagi-rerank", default="kohagi-rerank")
    p.add_argument("--model-id", default="argos1111/modernbert-ja-310m-jev")
    p.add_argument("--device", default="cpu", choices=["cpu", "coreml"])
    p.add_argument("--coreml-dir", help="converted bundle for --device coreml")
    p.add_argument("--length", default="short", choices=["short", "long"])
    p.add_argument("--ks", default="1,3,5,9,18", help="candidate counts to sweep")
    p.add_argument("--pairs-per-k", type=int, default=36)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--loads", type=int, default=5)
    args = p.parse_args()

    base = [args.kohagi_rerank, "--model-id", args.model_id]
    if args.device == "coreml":
        if not args.coreml_dir:
            p.error("--coreml-dir is required with --device coreml")
        base += ["--device", "coreml", "--coreml-dir", args.coreml_dir]
    query = SHORT if args.length == "short" else LONG
    ks = [int(x) for x in args.ks.split(",")]

    # Warm the OS file cache and the CoreML bundle, then measure load alone.
    stdin_of(query, 1, 1)
    child(base + ["--raw-logits"], stdin_of(query, 1, 1)[0])
    loads = [child(base + ["--print-model-info"], None) for _ in range(args.loads)]
    t_load = statistics.median(w for w, _ in loads)
    load_cpu = statistics.median(c for _, c in loads)
    print(f"model load : wall {t_load:.2f}s  cpu {load_cpu:.2f}s  "
          f"(n={args.loads}, wall {min(w for w, _ in loads):.2f}-{max(w for w, _ in loads):.2f})")

    samples = {k: [] for k in ks}
    for _ in range(args.reps):
        for k in ks:
            q = max(1, math.ceil(args.pairs_per_k / k))
            stdin, npairs = stdin_of(query, k, q)
            wall, cpu = child(base + ["--raw-logits"], stdin)
            samples[k].append(((wall - t_load) / npairs * 1000,
                               (cpu - load_cpu) / npairs * 1000))

    print(f"\nquery {args.length}, {args.device}, reps={args.reps} interleaved, load subtracted")
    print(f"{'K':>3} {'wall/pair(ms)':>16} {'cpu/pair(ms)':>15} {'per-question wall(ms)':>24}")
    for k in ks:
        walls = [w for w, _ in samples[k]]
        cpus = [c for _, c in samples[k]]
        w = statistics.median(walls)
        c = statistics.median(cpus)
        ratio = c / w if w else 0
        print(f"{k:>3} {w:>8.1f} ({min(walls):.0f}-{max(walls):.0f}) {c:>10.1f} {w * k:>19.1f}"
              f"   cpu/wall {ratio:.2f}")
    if args.device == "cpu":
        print("\ncpu/wall near 1 means little contention; well below 1 means wall is inflated.")
    else:
        print("\nCoreML runs the encoder off-CPU, so cpu/pair is only the head and orchestration;")
        print("wall is the number that matters there, and it is barely affected by CPU contention.")
    print("A subprocess caller also pays the model load above, once per process.")


if __name__ == "__main__":
    main()
