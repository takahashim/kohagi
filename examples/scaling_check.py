"""Measure how Kohagi's CPU throughput scales with worker threads.

    cargo build --release
    python examples/scaling_check.py --kohagi ./target/release/kohagi

Kohagi fans length-bucketed batches across a rayon pool sized to physical
cores. On an 8-core M2 that measured only ~2x over serial, which is either a
real limit worth fixing or an artifact of a busy machine — telling those apart
needs a quiet one, so this script checks before it measures and says so when
the answer is "your machine cannot resolve this".

Two settings are swept independently because they are different mechanisms:

- `RAYON_NUM_THREADS` — how many forwards Kohagi runs at once.
- `VECLIB_MAXIMUM_THREADS` — how many threads Accelerate uses *inside* one
  matmul. Pinned to 1 for the sweep so the outer scaling is measured on its
  own; measured separately at the end.

Apple Silicon is heterogeneous (4 performance + 4 efficiency cores on the base
M1/M2, 8+2 on Pro/Max), so the physical-core default may be handing a quarter
of the work to cores that run it at a fraction of the speed. The sweep covers
thread counts on both sides of the performance-core count for that reason.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

BASE = [
    "この製品は組み立てが簡単で、説明書も分かりやすかったです。",
    "駅前の駐輪場が不足しているため、増設を要望します。",
    "先週の会議で決まった方針について、あらためて共有いたします。",
    "気温の変化が大きいので、体調の管理には十分ご注意ください。",
    "申し込みの締め切りは今月末までとなっておりますのでご了承ください。",
    "検索結果の表示速度が改善され、体感でもかなり速くなりました。",
]


def sysctl(name: str) -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return ""


def preflight() -> tuple[dict, list[str]]:
    """Machine facts, plus the reasons this machine may not be measurable."""
    info = {
        "chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "physical": sysctl("hw.physicalcpu"),
        "p_cores": sysctl("hw.perflevel0.physicalcpu"),
        "e_cores": sysctl("hw.perflevel1.physicalcpu"),
        "load1": os.getloadavg()[0],
        "uptime_days": None,
    }
    try:
        boot = int(sysctl("kern.boottime").split("sec = ")[1].split(",")[0])
        info["uptime_days"] = round((time.time() - boot) / 86400, 1)
    except Exception:
        pass

    free_mb = None
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        page = int(vm.split("page size of ")[1].split(" ")[0])
        free_pages = sum(
            int(line.split(":")[1].strip().rstrip("."))
            for line in vm.splitlines()
            if line.startswith(("Pages free", "Pages inactive", "Pages speculative"))
        )
        free_mb = free_pages * page // (1024 * 1024)
    except Exception:
        pass
    info["free_mb"] = free_mb

    warnings = []
    if info["load1"] > 1.0:
        warnings.append(f"load average is {info['load1']:.2f} — other work is competing for cores")
    if free_mb is not None and free_mb < 3000:
        warnings.append(f"only {free_mb} MB free memory — paging will distort timings")
    if info["uptime_days"] and info["uptime_days"] > 7:
        warnings.append(f"up {info['uptime_days']} days — a reboot clears accumulated pressure")
    return info, warnings


def corpus(n: int) -> str:
    lines = []
    for i in range(n):
        t = BASE[i % len(BASE)] + BASE[(i + 3) % len(BASE)] + f"（整理番号{i}）"
        lines.append(json.dumps({"id": i, "text": t}, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def timed(binary: str, stdin: str, env: dict) -> float:
    e = {**os.environ, **env}
    t0 = time.perf_counter()
    r = subprocess.run([binary, "--device", "cpu"], input=stdin, capture_output=True,
                       text=True, env=e)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.exit(f"Kohagi failed: {r.stderr.strip()[-400:]}")
    return dt


def summarize(runs: list[float]) -> tuple[float, float]:
    return statistics.median(runs), (max(runs) - min(runs)) / statistics.median(runs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kohagi", default="./target/release/kohagi")
    p.add_argument("--texts", type=int, default=1200)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--force", action="store_true", help="measure even on a busy machine")
    args = p.parse_args()

    info, warnings = preflight()
    print(f"chip       : {info['chip']}")
    print(f"cores      : {info['physical']} physical", end="")
    if info["p_cores"] and info["e_cores"]:
        print(f" ({info['p_cores']} performance + {info['e_cores']} efficiency)")
    else:
        print()
    print(f"load / free: {info['load1']:.2f} / {info['free_mb']} MB")

    if warnings:
        print("\nthis machine may not give a usable answer:")
        for w in warnings:
            print(f"  - {w}")
        if not args.force:
            print("\nQuit other applications and retry, or pass --force to measure anyway.")
            return 2
        print("\n--force given; treat everything below as indicative only.\n")

    stdin = corpus(args.texts)
    threads = [1, 2, 3, 4, 6, 8]

    # Interleave: one run of every thread count per round, so thermal drift and
    # background load spread across all of them instead of penalizing whichever
    # was measured last.
    results: dict[int, list[float]] = {t: [] for t in threads}
    print(f"measuring {args.texts} texts, {args.rounds} rounds, interleaved...")
    # One discarded run so the model file is in the page cache; otherwise the
    # first timing carries a disk read that nothing else pays.
    timed(args.kohagi, stdin, {"RAYON_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "1"})
    for r in range(args.rounds):
        for t in threads:
            results[t].append(
                timed(args.kohagi, stdin,
                      {"RAYON_NUM_THREADS": str(t), "VECLIB_MAXIMUM_THREADS": "1"})
            )
        print(f"  round {r + 1}/{args.rounds} done", flush=True)

    serial = statistics.median(results[1])
    print(f"\n{'threads':>8}{'median':>10}{'spread':>9}{'speedup':>9}")
    best_t, best_med = 1, serial
    for t in threads:
        med, spread = summarize(results[t])
        if med < best_med:
            best_t, best_med = t, med
        print(f"{t:>8}{med:>9.2f}s{spread:>8.0%}{serial / med:>8.2f}x")

    # Accelerate's own threading, measured separately at the best outer count.
    blas = statistics.median(
        [timed(args.kohagi, stdin,
               {"RAYON_NUM_THREADS": str(best_t), "VECLIB_MAXIMUM_THREADS": "0"})
         for _ in range(3)]
    )
    print(f"\nwith Accelerate threading unpinned at {best_t} workers: {blas:.2f}s "
          f"({best_med / blas:.2f}x vs pinned)")

    # Judge only the parallel counts against each other. Serial is always far
    # slower, so including it would inflate the "effect" until any amount of
    # noise looked acceptable — which is exactly how a busy machine gets
    # mistaken for a conclusive result.
    ranked = [t for t in threads if t > 1]
    worst_spread = max(summarize(results[t])[1] for t in ranked)
    meds = [statistics.median(results[t]) for t in ranked]
    gap_frac = (max(meds) - min(meds)) / min(meds)

    print("\nverdict")
    if worst_spread > gap_frac:
        print(f"  INCONCLUSIVE — run-to-run spread reaches {worst_spread:.0%} while the")
        print(f"  spread between thread counts is only {gap_frac:.0%}. The noise is larger")
        print("  than what is being measured; these numbers cannot rank thread counts.")
        print("  The serial-vs-parallel speedup above may still be indicative.")
    else:
        print(f"  usable — spread tops out at {worst_spread:.0%}, below the {gap_frac:.0%} effect.")
        print(f"  Best is {best_t} threads at {serial / best_med:.2f}x over serial.")
        if info["p_cores"] and best_t <= int(info["p_cores"]) < int(info["physical"]):
            print(f"  Note that {best_t} <= {info['p_cores']} performance cores: the physical-core")
            print("  default may be losing time to efficiency cores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
