#!/usr/bin/env python3
"""Evaluate a typed-decision task the way notes/plan-decide.md section 6 designs it.

Reads a JSONL evaluation set, scores it with the systems that decision compares,
and reports ranking and probability metrics per split. The plan fixes the
decision's *shape* here so that the numbers, once a real task exists, are not
read subjectively; this script is that shape made runnable.

Input record (one JSON object per line):

    {"id": 1, "split": "main",
     "question": "担当部署は？",
     "state": "二重請求です。返金してください。解決しないなら解約します。",
     "candidates": ["billing — 請求・返金", "technical — 技術的な障害", "sales"],
     "gold": "billing — 請求・返金"}

- `query` may be given instead of `question`/`state`; it is then used verbatim.
  Otherwise the Jev rendering is applied: `質問: {question}\\n状況: {state}`.
- `split` groups the report, e.g. `main`, `long` (state past the window) and
  `unknown` (a label system absent from training). Default `main`.
- `candidates` are strings, already rendered the way the model should read them
  (`billing — 請求・返金`); the caller owns that rendering.
- `gold` is a candidate string or its index. Without it, Top-1 / MRR / NLL / ECE
  report as `-` and only the score distribution is available.

Systems compared:

- `majority`   — always the split's most frequent gold label. The floor. Top-1
  only; a single-label predictor has no per-candidate ranking to score MRR on.
- `embed`      — zero-shot by cosine between the query embedding and each
  candidate embedding (`kohagi`). No NLL/ECE: a cosine is not a probability.
- `jev`        — `argos1111/modernbert-ja-310m-jev` scored per candidate with
  `kohagi-rerank --raw-logits`, softmaxed across one query's candidates. That
  softmax is what turns a score into a probability; it lives here, outside
  kohagi, where plan-decide leaves it.

Controls, on by default (`--no-controls` to skip). Both come from the
open-implementation survey in plan-decide §6:

- `shuffled` — every case scored against the *next* case's context. A model
  that reads the menu rather than the state barely moves; OpenJev reports 95%
  to 22.4% on this control.
- `reversed` / `permuted` — the menu reordered, compared by label rather than
  index (a reversal swaps indices by construction). Reports the flip rate, the
  mean total-variation distance, and whether flips concentrate at low
  confidence. OpenJev reports 3% and 4.5%.

The `hand-tuned LLM zero-shot` row plan-decide lists as an upper bound is not
here: it belongs to whatever LLM you own, not to kohagi.

Usage:

    python3 tools/decide_eval.py --data eval.jsonl \\
        --kohagi ./target/release/kohagi \\
        --kohagi-rerank ./target/release/kohagi-rerank \\
        --embed-model cl-nagoya/ruri-v3-310m \\
        --embed-prefix-query "検索クエリ: " --embed-prefix-doc "検索文書: "

    # plumbing check on the model card's own example, no data file needed
    python3 tools/decide_eval.py --self-test

`--json` prints the same numbers as a JSON object, for a later run to diff
against. `--dump` writes every case's per-candidate scores and the gold rank,
which is what tells a "learned the wrong label system" failure from a "read the
state wrongly" one.
"""

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict

JEV_RENDER = "質問: {question}\n状況: {state}"

JEV = "jev"
EMBED = "embed"
MAJORITY = "majority"
SHUFFLED = "shuffled"


def query_of(record):
    if "query" in record:
        return record["query"]
    return JEV_RENDER.format(question=record["question"], state=record["state"])


def gold_index(record):
    """The candidate index the record points at, or None.

    A `gold` of JSON null is treated as absent, so a set can carry the field in
    every record and only fill it where an answer exists.
    """
    gold = record.get("gold")
    if gold is None:
        return None
    candidates = record["candidates"]
    if isinstance(gold, int):
        if not 0 <= gold < len(candidates):
            raise ValueError(f"record {record.get('id')}: gold index {gold} is out of range")
        return gold
    if gold not in candidates:
        raise ValueError(f"record {record.get('id')}: gold {gold!r} is not among the candidates")
    return candidates.index(gold)


def run_jsonl(binary, args, records):
    """Feed `records` as JSONL, collect the replies keyed by id.

    Exit 2 means some lines were skipped; that is reported and the rest kept,
    because a partial set is still a measurement of that partial set and the
    skip line names why.
    """
    stdin = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    proc = subprocess.run([binary, *args], input=stdin, capture_output=True, text=True)
    if proc.returncode not in (0, 2):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{binary} exited {proc.returncode}")
    if proc.returncode == 2:
        sys.stderr.write(proc.stderr)
    out = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def embed_texts(kohagi, model, prefix, items, what, extra=()):
    """Embed `(id, text)` items under one prefix, return {id: vector}."""
    args = ["--model-id", model]
    if prefix:
        args += ["--prefix", prefix]
    args += list(extra)
    replies = run_jsonl(kohagi, args, [{"id": i, "text": t} for i, t in items])
    missing = [i for i, _ in items if i not in replies]
    if missing:
        raise SystemExit(f"{binary_name(kohagi)} returned no embedding for {what} id {missing[0]}")
    return {i: replies[i]["embedding"] for i, _ in items}


def binary_name(path):
    return path.rsplit("/", 1)[-1]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def jev_logits(rerank, model, records, max_seq, extra=()):
    """Score every (query, candidate) pair; return {(case_key, i): logit}."""
    reqs = []
    for r in records:
        q = query_of(r)
        for i, cand in enumerate(r["candidates"]):
            reqs.append({"id": f"{r['_key']}:{i}", "query": q, "text": cand})
    args = ["--model-id", model, "--raw-logits", "--max-seq-length", str(max_seq)]
    args += list(extra)
    replies = run_jsonl(rerank, args, reqs)
    out = {}
    for key, row in replies.items():
        case, i = key.rsplit(":", 1)
        out[(case, int(i))] = row["score"]
    return out


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    total = sum(es)
    return [e / total for e in es]


def rank_of(scores, gold):
    """1-based rank of `gold` when candidates are ordered by descending score."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return order.index(gold) + 1


def top1_and_mrr(rows, scores_of, gold_of):
    n = len(rows)
    if n == 0:
        return None, None
    hits = 0
    rr = 0.0
    for r in rows:
        scores = scores_of(r)
        gold = gold_of(r)
        if max(range(len(scores)), key=lambda i: scores[i]) == gold:
            hits += 1
        rr += 1.0 / rank_of(scores, gold)
    return hits / n, rr / n


def nll_and_ece(rows, probs_of, gold_of, bins=15):
    """NLL and expected calibration error over the predicted class.

    15 bins because plan-decide fixes that number: fewer hides a miscalibrated
    middle, more is noise at the set sizes a routing task has.
    """
    n = len(rows)
    if n == 0:
        return None, None
    nll = 0.0
    bucket = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        probs = probs_of(r)
        gold = gold_of(r)
        nll -= math.log(max(probs[gold], 1e-12))
        pred = max(range(len(probs)), key=lambda i: probs[i])
        b = min(bins - 1, int(probs[pred] * bins))
        bucket[b][0] += 1
        bucket[b][1] += probs[pred]
        bucket[b][2] += 1.0 if pred == gold else 0.0
    ece = sum(
        count / n * abs(conf_sum / count - correct_sum / count)
        for count, conf_sum, correct_sum in bucket.values()
    )
    return nll / n, ece


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


def context_of(record):
    """The part of a record that is the state, not the question or the menu."""
    return record["query"] if "query" in record else record["state"]


def with_context(record, context):
    r = dict(record)
    if "query" in r:
        r["query"] = context
    else:
        r["state"] = context
    return r


def shuffled_records(records):
    """Rotate every case's context onto the next, so none keeps its own.

    The control a useful decision model must pass: if Top-1 barely moves when
    the state is someone else's, the model is reading the menu and not the
    context. Needs at least two cases; one case has no wrong context to borrow.
    """
    if len(records) < 2:
        return []
    contexts = [context_of(r) for r in records]
    return [with_context(r, contexts[(i + 1) % len(contexts)]) for i, r in enumerate(records)]


def reversed_order(k):
    return list(range(k - 1, -1, -1))


def seeded_order(seed):
    """A fixed permutation per menu size, so a rerun is comparable to the last."""

    def order(k):
        perm = list(range(k))
        random.Random(seed + k).shuffle(perm)
        return perm

    return order


def permuted_records(records, order_of):
    out = []
    for r in records:
        order = order_of(len(r["candidates"]))
        r2 = dict(r)
        r2["candidates"] = [r["candidates"][j] for j in order]
        r2["_order"] = order
        out.append(r2)
    return out


def order_effect(rows, base_of, perm_of):
    """How much one permutation moves the answer, in labels rather than indices.

    Compares the argmax label and the whole distribution, not the index: a
    reversal swaps indices by construction, so an index comparison would report
    a flip for a model that is perfectly order-invariant.
    """
    flips = 0
    tvs = []
    conf_flipped = []
    conf_kept = []
    for r in rows:
        p = base_of(r)
        aligned = [0.0] * len(p)
        for j, orig in enumerate(r["_order"]):
            aligned[orig] = perm_of(r)[j]
        pred_p = max(range(len(p)), key=lambda i: p[i])
        pred_a = max(range(len(aligned)), key=lambda i: aligned[i])
        tvs.append(0.5 * sum(abs(a - b) for a, b in zip(p, aligned)))
        if pred_p != pred_a:
            flips += 1
            conf_flipped.append(p[pred_p])
        else:
            conf_kept.append(p[pred_p])
    n = len(rows)
    return {
        "n": n,
        "flip_rate": flips / n if n else None,
        "mean_tv": statistics.fmean(tvs) if tvs else None,
        "mean_conf_flipped": statistics.fmean(conf_flipped) if conf_flipped else None,
        "mean_conf_kept": statistics.fmean(conf_kept) if conf_kept else None,
    }


def build_systems(records, args):
    """Every system's per-case scores, computed once for all splits.

    Returns `(systems, elapsed)` where `systems[name](record, index)` gives one
    candidate's score or probability.
    """
    qvecs, cvecs = {}, {}
    if not args.no_embed:
        q_items = [(f"q:{r['id']}", query_of(r)) for r in records]
        c_items = [
            (f"c:{r['id']}:{i}", cand)
            for r in records for i, cand in enumerate(r["candidates"])
        ]
        qvecs = embed_texts(args.kohagi, args.embed_model, args.embed_prefix_query, q_items,
                            "query", args.embed_arg)
        cvecs = embed_texts(args.kohagi, args.embed_model, args.embed_prefix_doc, c_items,
                            "candidate", args.embed_arg)

    t0 = time.monotonic()
    logits = jev_logits(args.kohagi_rerank, args.jev_model, records, args.max_seq, args.rerank_arg)
    elapsed = time.monotonic() - t0

    def embed_row(r):
        q = qvecs[f"q:{r['id']}"]
        return [cosine(q, cvecs[f"c:{r['id']}:{i}"]) for i in range(len(r["candidates"]))]

    def jev_row(r):
        return softmax([logits[(r["_key"], i)] for i in range(len(r["candidates"]))])

    systems = {JEV: jev_row}
    if not args.no_embed:
        systems[EMBED] = embed_row
    return systems, elapsed


def evaluate(records, args):
    for r in records:
        r["_key"] = str(r["id"])
    golds = {r["_key"]: gold_index(r) for r in records}
    candidates = {r["_key"]: r["candidates"] for r in records}
    systems, elapsed = build_systems(records, args)

    # Controls (plan-decide §6): the wrong-context control and the two option
    # permutations, all scored with the Jev model. They cost one extra scoring
    # pass each, which is why --no-controls exists for a quick run.
    controls = {}
    if not args.no_controls:
        shuffled = shuffled_records(records)
        if shuffled:
            sh = jev_logits(args.kohagi_rerank, args.jev_model, shuffled, args.max_seq, args.rerank_arg)
            systems[SHUFFLED] = lambda r: softmax(
                [sh[(r["_key"], i)] for i in range(len(r["candidates"]))]
            )
        for name, order_of in (("reversed", reversed_order),
                               ("permuted", seeded_order(args.order_seed))):
            perm = permuted_records(records, order_of)
            controls[name] = {
                "logits": jev_logits(args.kohagi_rerank, args.jev_model, perm, args.max_seq, args.rerank_arg),
                "by_key": {r["_key"]: r for r in perm},
            }

    splits = defaultdict(list)
    for r in records:
        splits[r.get("split", "main")].append(r)

    table = []
    control_table = []
    dump = []
    for split, group in sorted(splits.items()):
        have_gold = [r for r in group if golds[r["_key"]] is not None]
        gold_of = lambda r: golds[r["_key"]]  # noqa: E731

        rows = []
        if have_gold:
            counts = Counter(candidates[r["_key"]][golds[r["_key"]]] for r in have_gold)
            label, hits = counts.most_common(1)[0]
            rows.append((MAJORITY, len(group), hits / len(have_gold), None, None, None, label))
        else:
            rows.append((MAJORITY, len(group), None, None, None, None, None))
        if not args.no_embed:
            t, m = top1_and_mrr(have_gold, systems[EMBED], gold_of)
            rows.append((EMBED, len(group), t, m, None, None, None))
        t, m = top1_and_mrr(have_gold, systems[JEV], gold_of)
        n, e = nll_and_ece(have_gold, systems[JEV], gold_of)
        rows.append((JEV, len(group), t, m, n, e, None))
        if SHUFFLED in systems:
            t, m = top1_and_mrr(have_gold, systems[SHUFFLED], gold_of)
            n, e = nll_and_ece(have_gold, systems[SHUFFLED], gold_of)
            rows.append((SHUFFLED, len(group), t, m, n, e, None))
        table.append((split, rows))

        # The order effect needs each case's permuted candidate list and its
        # original-order probabilities; the permuted record carries `_order`.
        split_controls = {}
        for name, data in controls.items():
            perm_rows = [data["by_key"][r["_key"]] for r in group]

            def perm_of(r, logits=data["logits"]):
                return softmax([logits[(r["_key"], i)] for i in range(len(r["candidates"]))])

            split_controls[name] = order_effect(perm_rows, systems[JEV], perm_of)
        if split_controls:
            control_table.append((split, split_controls))

        for r in group:
            gold = golds[r["_key"]]
            probs = systems[JEV](r)
            cand = r["candidates"]
            dump.append({
                "id": r["id"], "split": r.get("split", "main"),
                "gold": None if gold is None else cand[gold],
                "jev": {cand[i]: round(probs[i], 6) for i in range(len(cand))},
                "jev_rank": None if gold is None else rank_of(probs, gold),
                "embed": None if args.no_embed else
                {cand[i]: round(systems[EMBED](r)[i], 6) for i in range(len(cand))},
            })

    n_pairs = sum(len(c["candidates"]) for c in records)
    if args.json:
        print(json.dumps({
            "splits": [
                {"split": s, "systems": [
                    {"system": name, "n": rn, "top1": t, "mrr": m, "nll": nl, "ece": e,
                     **({"majority_label": extra} if extra else {})}
                    for name, rn, t, m, nl, e, extra in rows]}
                for s, rows in table],
            "controls": [
                {"split": s, "order": {
                    name: {"n": eff["n"], "flip_rate": eff["flip_rate"], "mean_tv": eff["mean_tv"],
                           "mean_conf_flipped": eff["mean_conf_flipped"],
                           "mean_conf_kept": eff["mean_conf_kept"]}
                    for name, eff in sc.items()}}
                for s, sc in control_table],
            "rerank_wall_s": round(elapsed, 3),
            "pairs_per_s": round(n_pairs / elapsed, 1) if elapsed else None,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"jev model  : {args.jev_model}")
        if not args.no_embed:
            print(f"embed model: {args.embed_model}")
        print(f"cases      : {len(records)}  pairs: {n_pairs}")
        rate = f" ({n_pairs / elapsed:.0f} pairs/s)" if elapsed else ""
        print(f"rerank wall: {elapsed:.2f}s{rate}")
        for split, rows in table:
            n = next(rn for _, rn, *_ in rows)
            print(f"\n[{split}]  n={n}")
            print(f"  {'system':10s} {'top1':>8s} {'mrr':>8s} {'nll':>8s} {'ece':>8s}")
            for name, _rn, t, m, nl, e, extra in rows:
                note = f"  ({extra})" if extra else ""
                print(f"  {name:10s} {fmt(t):>8s} {fmt(m):>8s} {fmt(nl):>8s} {fmt(e):>8s}{note}")
        for split, sc in control_table:
            print(f"\n[{split}] controls (jev)")
            for name, eff in sc.items():
                flips = f"{eff['flip_rate'] * eff['n']:.0f}/{eff['n']}"
                print(f"  {name:9s} flips {flips:>7s} ({fmt(eff['flip_rate'])})"
                      f"  mean TV {fmt(eff['mean_tv'])}"
                      f"  conf flipped {fmt(eff['mean_conf_flipped'])}"
                      f" vs kept {fmt(eff['mean_conf_kept'])}")
        if SHUFFLED in systems:
            print("\nshuffled is the wrong-context control: its Top-1 should fall.")
        if control_table:
            print("order flips should be rare and concentrate at low confidence.")
        print("embed nll/ece are '-' by design: a cosine is not a probability.")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for row in dump:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nper-case scores written to {args.dump}", file=sys.stderr)


SELF_TEST = [{
    "id": 0, "split": "main", "question": "担当部署は？",
    "state": "二重請求です。返金してください。解決しないなら解約します。",
    "candidates": ["billing — 請求・返金", "technical — 技術的な障害", "sales"],
    "gold": "billing — 請求・返金",
}]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="JSONL evaluation set (see the module docstring)")
    p.add_argument("--kohagi", default="kohagi", help="path to the kohagi binary")
    p.add_argument("--kohagi-rerank", default="kohagi-rerank", help="path to the kohagi-rerank binary")
    p.add_argument("--jev-model", default="argos1111/modernbert-ja-310m-jev")
    p.add_argument("--embed-model", default="cl-nagoya/ruri-v3-310m",
                   help="the embedding zero-shot baseline (plan-decide names ruri-v3-310m)")
    p.add_argument("--embed-prefix-query", default="", help="prefix for the query side of the embed baseline")
    p.add_argument("--embed-prefix-doc", default="", help="prefix for the label-description side")
    p.add_argument("--no-embed", action="store_true", help="skip the embedding baseline")
    p.add_argument("--no-controls", action="store_true",
                   help="skip the wrong-context control and the option-order permutations")
    p.add_argument("--order-seed", type=int, default=0,
                   help="seed for the permuted-order control (one fixed permutation per menu size)")
    p.add_argument("--rerank-arg", action="append", default=[], metavar="ARG",
                   help="extra argument passed to kohagi-rerank, repeatable "
                        "(e.g. --rerank-arg=--device --rerank-arg=coreml --rerank-arg=--coreml-dir --rerank-arg=DIR)")
    p.add_argument("--embed-arg", action="append", default=[], metavar="ARG",
                   help="extra argument passed to kohagi, repeatable")
    p.add_argument("--max-seq", type=int, default=512)
    p.add_argument("--json", action="store_true", help="print metrics as JSON")
    p.add_argument("--dump", help="write per-case scores to this JSONL, for error analysis")
    p.add_argument("--self-test", action="store_true",
                   help="run the model card's example through the real binaries")
    args = p.parse_args()

    if args.self_test:
        records = SELF_TEST
    elif args.data:
        with open(args.data, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        p.error("--data or --self-test is required")
    if not records:
        p.error("the evaluation set is empty")
    evaluate(records, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
