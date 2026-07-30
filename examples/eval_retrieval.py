#!/usr/bin/env python3
"""Retrieval quality for a Kohagi configuration, on JaCWIR and JQaRA.

This is what says an int8 embedding table costs nothing on retrieval quality and
that the Neural Engine matches the CPU. Keeping it in the repository means a later
change to the encoder, the converter or the quantization can be checked against
the same metric rather than against a remembered figure.

    # the float32 CPU reference, which reproduces the recorded baselines
    python3 examples/eval_retrieval.py --benchmark jacwir -- --device cpu
    python3 examples/eval_retrieval.py --benchmark jqara  -- --device cpu

    # anything else, compared against it
    python3 examples/eval_retrieval.py --benchmark jacwir -- --device coreml
    python3 examples/eval_retrieval.py --benchmark jacwir -- \\
        --device coreml --coreml-quantize embeddings

Everything after `--` is passed to Kohagi verbatim, so any device, model or
quantization can be measured without changing this file.

Recorded for `cl-nagoya/ruri-v3-130m` at `--max-seq-length 512`, float32 on CPU:

    JaCWIR  MAP@10 0.8584   HIT@10 0.9696
    JQaRA   nDCG@10 0.7106

Those are the full sets. `--limit` takes a prefix of the queries, which is enough
to tell that this script's method is the one the recorded numbers came from
without embedding half a million documents: `--limit 50` on JaCWIR measured
MAP@10 0.8303, and `--limit 100` on JQaRA measured nDCG@10 0.7046. A full run is
long — JaCWIR's 5000 queries name about 497k distinct documents — so budget hours
rather than minutes for the numbers that go in a table.

Needs `pip install pyarrow` and network access on the first run (the datasets are
cached under `--cache`, and are not committed).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import urllib.request

# Ruri v3 asks for a task prefix, and the metric moves without it. A model that
# wants none takes --doc-prefix '' --query-prefix ''.
DEFAULTS = {
    "jacwir": dict(doc="検索文書: ", query="検索クエリ: "),
    "jqara": dict(doc="検索文書: ", query="検索クエリ: "),
}

# (repo, config, split) for the Hub's parquet endpoint. JaCWIR keeps its queries
# and its 513k-document collection in two configs, so it needs both.
SOURCES = {
    "jacwir": ("hotchpotch/JaCWIR", "eval", "eval"),
    "jacwir-collection": ("hotchpotch/JaCWIR", "collection", "collection"),
    "jqara": ("hotchpotch/JQaRA", "default", "test"),
}


def parquet_urls(repo, config, split):
    """The parquet shards the Hub converted a dataset into."""
    url = f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def load_table(repo, config, split, cache):
    """Every row of a dataset split, as a list of dicts."""
    import pyarrow.parquet as pq

    os.makedirs(cache, exist_ok=True)
    rows = []
    for i, url in enumerate(parquet_urls(repo, config, split)):
        path = os.path.join(cache, f"{repo.replace('/', '__')}-{split}-{i}.parquet")
        if not os.path.exists(path):
            print(f"  downloading shard {i} ...", file=sys.stderr, flush=True)
            urllib.request.urlretrieve(url, path)
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def embed(kohagi_args, prefix, texts, tag, max_seq):
    """One Kohagi run over `texts`, returning vectors in input order."""
    cmd = [
        *kohagi_args,
        "--max-seq-length",
        str(max_seq),
        "--prefix",
        prefix,
    ]
    payload = "\n".join(
        json.dumps({"id": i, "text": t}, ensure_ascii=False) for i, t in enumerate(texts)
    )
    print(f"  [{tag}] {len(texts)} texts ...", file=sys.stderr, flush=True)
    p = subprocess.run(cmd, input=payload.encode(), capture_output=True)
    if p.returncode != 0:
        sys.exit(f"kohagi failed ({p.returncode}):\n{p.stderr.decode()[-2000:]}")
    out = [None] * len(texts)
    for line in p.stdout.splitlines():
        r = json.loads(line)
        out[r["id"]] = r["embedding"]
    if any(v is None for v in out):
        sys.exit("kohagi returned fewer records than it was given")
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def jacwir(rows, run, cache):
    """MAP@10 and HIT@10: one positive among ~100 candidates per query.

    With a single relevant document per query, average precision is 1/rank, so
    MAP@10 is the mean reciprocal rank truncated at 10.

    Only the documents the retained queries actually name are embedded. The full
    collection is 513k documents; a 5000-query run touches about a hundredth of it,
    and embedding the rest would dominate the runtime without changing a number.
    """
    wanted = {r["positive"] for r in rows}
    for r in rows:
        wanted.update(r["negatives"])

    collection = load_table(*SOURCES["jacwir-collection"], cache)
    docs = {
        d["doc_id"]: f"{d['title']} {d['description']}".strip()
        for d in collection
        if d["doc_id"] in wanted
    }
    missing = wanted - docs.keys()
    if missing:
        sys.exit(f"{len(missing)} candidate documents are not in the collection")

    ids = sorted(docs)
    index = {d: i for i, d in enumerate(ids)}
    doc_vecs = run("doc", [docs[d] for d in ids])
    query_vecs = run("query", [r["query"] for r in rows])

    ap, hit = [], []
    for r, qv in zip(rows, query_vecs):
        candidates = [r["positive"], *r["negatives"]]
        ranked = sorted(
            ((dot(qv, doc_vecs[index[c]]), c) for c in candidates), key=lambda x: -x[0]
        )
        rank = next(i + 1 for i, (_, c) in enumerate(ranked) if c == r["positive"])
        ap.append(1 / rank if rank <= 10 else 0.0)
        hit.append(1.0 if rank <= 10 else 0.0)
    return {"MAP@10": sum(ap) / len(ap), "HIT@10": sum(hit) / len(hit)}


def jqara(rows, run):
    """nDCG@10 over graded relevance, one query at a time."""
    by_query = {}
    for r in rows:
        by_query.setdefault(r["q_id"], {"question": r["question"], "cands": []})
        by_query[r["q_id"]]["cands"].append(
            (f"{r.get('title', '')} {r.get('text', '')}".strip(), int(r["label"]))
        )

    order = sorted(by_query)
    flat, spans = [], []
    for q in order:
        start = len(flat)
        flat.extend(t for t, _ in by_query[q]["cands"])
        spans.append((start, len(flat)))

    doc_vecs = run("doc", flat)
    query_vecs = run("query", [by_query[q]["question"] for q in order])

    def dcg(gains):
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    scores = []
    for (start, end), qv, q in zip(spans, query_vecs, order):
        labels = [lab for _, lab in by_query[q]["cands"]]
        ranked = sorted(
            range(end - start), key=lambda i: -dot(qv, doc_vecs[start + i])
        )
        got = dcg([labels[i] for i in ranked[:10]])
        best = dcg(sorted(labels, reverse=True)[:10])
        scores.append(got / best if best else 0.0)
    return {"nDCG@10": sum(scores) / len(scores)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--benchmark", choices=["jacwir", "jqara"], required=True)
    ap.add_argument(
        "--kohagi",
        default=os.path.join(os.path.dirname(__file__), "..", "target", "release", "kohagi"),
    )
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--doc-prefix", default=None, help="overrides the Ruri v3 default")
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument(
        "--limit", type=int, default=0, help="use only the first N queries (a smoke test)"
    )
    ap.add_argument(
        "--cache",
        default=os.path.join(os.path.expanduser("~"), ".cache", "kohagi-eval"),
        help="where the parquet shards are kept (not committed)",
    )
    ap.add_argument("kohagi_args", nargs="*", help="passed to kohagi after --")
    args = ap.parse_args()

    if not os.path.exists(args.kohagi):
        sys.exit(f"no kohagi at {args.kohagi}; build it first (cargo build --release ...)")

    prefixes = DEFAULTS[args.benchmark]
    doc_prefix = args.doc_prefix if args.doc_prefix is not None else prefixes["doc"]
    query_prefix = args.query_prefix if args.query_prefix is not None else prefixes["query"]

    rows = load_table(*SOURCES[args.benchmark], args.cache)
    if args.limit:
        if args.benchmark == "jqara":
            keep = sorted({r["q_id"] for r in rows})[: args.limit]
            rows = [r for r in rows if r["q_id"] in set(keep)]
        else:
            rows = rows[: args.limit]

    base = [args.kohagi, *args.kohagi_args]

    def run(kind, texts):
        prefix = doc_prefix if kind == "doc" else query_prefix
        return embed(base, prefix, texts, kind, args.max_seq_length)

    metrics = (
        jacwir(rows, run, args.cache)
        if args.benchmark == "jacwir"
        else jqara(rows, run)
    )

    print()
    print(f"benchmark : {args.benchmark}")
    print(f"kohagi    : {' '.join(args.kohagi_args) or '(defaults)'}")
    for name, value in metrics.items():
        print(f"{name:<10}: {value:.4f}")


if __name__ == "__main__":
    main()
