"""Time Kohagi against the sentence-transformers / PyTorch reference.

    python tools/benchmark.py --kohagi ./target/release/kohagi

Both sides are pinned to Kohagi's defaults — mean pooling, L2 normalize,
max_seq_length 512, batch size 64 — because a throughput comparison between
differently-configured runs measures the configuration, not the code.

Two numbers are reported per implementation and they answer different
questions:

- **encode** is compute only. It tells you which runtime multiplies matrices
  faster on this machine.
- **total** includes interpreter startup and model load. It is what a batch
  job or a per-invocation subprocess actually pays, and for short runs it is
  dominated by startup rather than compute.

Startup is measured, not assumed: PyTorch pays it once per process, so a
long-lived worker amortizes it away while a rake task never does. Decide
which row matters by how you plan to deploy, not by which is larger.

Pass --coreml-dir (or --coreml-model-id) to add a `kohagi/coreml` row for the
Apple Neural Engine, alongside the CPU one. It must be the *same* checkpoint
as --model-id, or the two Kohagi rows are not comparable — nothing checks this
for you. Two things about that row:

- **Convert with --compiled.** A directory whose buckets are only
  `.mlpackage` bundles is compiled to `.mlmodelc` on *every* load, which lands
  in the load column: ~20s per invocation on an M2, against ~0.6s for a
  directory that ships `compiled/`. That is a property of the directory, not
  of the ANE.
- **The very first load of a given model pays one more one-time cost** while
  the OS caches its ANE program (~20s). The warm-up run absorbs it.

By default this generates a synthetic Japanese corpus so the numbers are
reproducible without shipping a dataset. Pass --texts to use your own.
"""

import argparse
import json
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


def synth(kind: str, n: int) -> list[str]:
    """Short is ~30 tokens; long overflows 512 so every record hits the cap.

    These are ruri-v3 tokenizer counts. Japanese packs several characters per
    token, so reaching the 512 cap takes more repetitions than an English
    reader might expect — long throughput is dominated by the quadratic
    attention cost only when the records genuinely fill the window.
    """
    reps = 2 if kind == "short" else 60
    return [
        "".join(BASE[(i + k) % len(BASE)] for k in range(reps)) + f"（整理番号{i}）"
        for i in range(n)
    ]


def time_kohagi(
    binary: str, texts: list[str], args, *, coreml: bool = False, device: str | None = None
) -> dict:
    stdin = "".join(
        json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n"
        for i, t in enumerate(texts)
    )
    cmd = [binary, "--model-id", args.model_id, "--max-seq-length", str(args.max_seq_length)]
    if device:
        cmd += ["--device", device]
    if coreml:
        cmd += ["--device", "coreml"]
        if args.coreml_dir:
            cmd += ["--coreml-dir", args.coreml_dir]
        else:
            cmd += ["--coreml-model-id", args.coreml_model_id]
    elif args.precision != "f32":
        cmd += ["--precision", args.precision]

    # A 2-record run is almost entirely startup plus model load, so it
    # isolates the fixed cost that the full run also pays.
    #
    # Take the fastest of two probes. The probe runs in its own process before
    # the full run, so a one-time cost that lands on the *first* invocation of
    # a model — a cold page cache, or the OS caching the CoreML compile of an
    # uncompiled .mlpackage — would be counted as load and then not paid by the
    # full run, pushing `encode` (total - load) below zero. Timing two probes
    # and keeping the smaller one measures the cost the full run actually
    # shares.
    head = "".join(stdin.splitlines(keepends=True)[:2])
    load = min(run_kohagi(cmd, head) for _ in range(2))
    total = run_kohagi(cmd, stdin)

    return {"load": load, "encode": total - load, "total": total}


def run_kohagi(cmd: list[str], stdin: str) -> float:
    """One Kohagi invocation, timed end to end."""
    t0 = time.perf_counter()
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        sys.exit(f"Kohagi failed ({r.returncode}): {r.stderr.strip()}")
    return elapsed


# Run torch in a fresh process, exactly as Kohagi is run. Timing it in-process
# would let Python's module cache serve the second `import sentence_transformers`
# for free, hiding ~3s of the startup cost that a real batch job pays every time.
_ST_SCRIPT = """
import json, sys, time
t0 = time.perf_counter()
from sentence_transformers import SentenceTransformer
texts, model_id, device, msl = json.load(sys.stdin)
m = SentenceTransformer(model_id, device=device)
m.max_seq_length = msl
t_load = time.perf_counter()
m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
t_end = time.perf_counter()
print(json.dumps({"load": t_load - t0, "encode": t_end - t_load, "total": t_end - t0}),
      file=sys.stderr)
"""


def time_st(texts: list[str], device: str, args) -> dict:
    r = subprocess.run(
        [sys.executable, "-c", _ST_SCRIPT],
        input=json.dumps([texts, args.model_id, device, args.max_seq_length]),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        if "ModuleNotFoundError" in r.stderr:
            raise ImportError(r.stderr)
        sys.exit(f"sentence-transformers failed: {r.stderr.strip()[-500:]}")
    return json.loads(r.stderr.strip().splitlines()[-1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kohagi", default="./target/release/kohagi")
    p.add_argument("--model-id", default="cl-nagoya/ruri-v3-130m")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--kind", choices=["short", "long"], default="short")
    p.add_argument("--count", type=int, default=0, help="default: 1200 short / 240 long")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--texts", help="file with one text per line")
    p.add_argument("--device", default="cpu", help="PyTorch device: cpu, mps, cuda")
    p.add_argument(
        "--kohagi-device",
        action="append",
        default=[],
        help="extra Kohagi device to time, e.g. metal. Repeatable; adds one row each.",
    )
    coreml = p.add_mutually_exclusive_group()
    coreml.add_argument(
        "--coreml-dir",
        help="local converted CoreML model directory; adds a kohagi/coreml row. "
        "Must be the same checkpoint as --model-id, and should ship compiled/ "
        "(see the description above).",
    )
    coreml.add_argument(
        "--coreml-model-id",
        help="Hub repo with converted CoreML models; adds a kohagi/coreml row. "
        "Same-checkpoint caveat as --coreml-dir.",
    )
    p.add_argument(
        "--precision",
        choices=["f32", "bf16"],
        default="f32",
        help="Kohagi's --precision; bf16 needs an AVX512-BF16 CPU. torch stays f32, "
        "so a bf16 run compares Kohagi against itself, not against the reference.",
    )
    p.add_argument("--skip-torch", action="store_true")
    args = p.parse_args()

    if args.texts:
        with open(args.texts, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        texts = synth(args.kind, args.count or (1200 if args.kind == "short" else 240))

    print(f"model      : {args.model_id} ({args.precision})")
    if args.coreml_dir or args.coreml_model_id:
        print(f"coreml     : {args.coreml_dir or args.coreml_model_id}")
    print(f"texts      : {len(texts)} ({args.texts or args.kind})")
    print(f"runs       : {args.runs} (median reported)\n")

    # The first run of either implementation pays cold caches — page cache for
    # Kohagi, Metal shader compilation for MPS. Discard it.
    print("warming up...", flush=True)
    time_kohagi(args.kohagi, texts[:8], args)
    for d in args.kohagi_device:
        time_kohagi(args.kohagi, texts[:8], args, device=d)
    if args.coreml_dir or args.coreml_model_id:
        time_kohagi(args.kohagi, texts[:8], args, coreml=True)

    rows = []
    med = lambda rs, k: statistics.median(r[k] for r in rs)  # noqa: E731

    runs = [time_kohagi(args.kohagi, texts, args) for _ in range(args.runs)]
    rows.append(("kohagi", med(runs, "load"), med(runs, "encode"), med(runs, "total")))

    for d in args.kohagi_device:
        runs = [time_kohagi(args.kohagi, texts, args, device=d) for _ in range(args.runs)]
        rows.append(
            (f"kohagi/{d}", med(runs, "load"), med(runs, "encode"), med(runs, "total"))
        )
    if args.coreml_dir or args.coreml_model_id:
        runs = [time_kohagi(args.kohagi, texts, args, coreml=True) for _ in range(args.runs)]
        rows.append(
            ("kohagi/coreml", med(runs, "load"), med(runs, "encode"), med(runs, "total"))
        )

    if not args.skip_torch:
        try:
            time_st(texts[:8], args.device, args)
            runs = [time_st(texts, args.device, args) for _ in range(args.runs)]
            rows.append(
                (f"torch/{args.device}", med(runs, "load"), med(runs, "encode"), med(runs, "total"))
            )
        except ImportError:
            print("sentence-transformers not installed; skipping the comparison\n")

    print(f"\n{'':<14}{'load':>9}{'encode':>9}{'total':>9}")
    for name, load, encode, total in rows:
        print(f"{name:<14}{load:>8.2f}s{encode:>8.2f}s{total:>8.2f}s")

    # `encode` is a subtraction, so it can still come out non-positive when the
    # load estimate is unstable across processes — say a run so short that
    # startup swamps it, or a machine busy enough that the probes and the full
    # run see different conditions. Say so rather than let a negative number
    # pass for a measurement.
    for name, load, encode, _ in rows:
        if encode <= 0:
            print(
                f"\nWARNING: {name} encode is {encode:.2f}s — the {load:.2f}s load "
                "estimate reached the full run, so this row measures startup, not "
                "compute. Raise --count, or quiet the machine, and re-run."
            )

    by_name = {row[0]: row for row in rows}
    if "torch/" + args.device in by_name:
        torch = by_name["torch/" + args.device]
        kohagi = by_name["kohagi"]
        print(f"\nencode : kohagi is {torch[2] / kohagi[2]:.2f}x torch/{args.device}")
        print(f"total  : kohagi is {torch[3] / kohagi[3]:.2f}x torch/{args.device}")
        if "kohagi/coreml" in by_name:
            coreml_row = by_name["kohagi/coreml"]
            print(f"encode : coreml is {torch[2] / coreml_row[2]:.2f}x torch/{args.device}")
            print(f"total  : coreml is {torch[3] / coreml_row[3]:.2f}x torch/{args.device}")
        print(
            "\nNote: torch pays load once per process. If you can amortize it,\n"
            "compare the encode row; if you spawn per batch, compare total."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
