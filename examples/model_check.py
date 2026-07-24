"""Smoke-test Kohagi against any ModernBERT sentence encoder on the Hub.

Kohagi runs more than the default ruri-v3: any ModernBERT encoder that ships a
fast `tokenizer.json` and a sentence-transformers `1_Pooling/config.json`. This
script points Kohagi at one and checks that the embeddings are actually usable,
not merely that the process exited 0 — a retrieval model always returns
plausible floats, so "it ran" proves nothing.

    python examples/model_check.py --kohagi ./target/release/kohagi \
        Alibaba-NLP/gte-modernbert-base

What it checks:

- **pooling is taken from the checkpoint.** Kohagi reads the model's own
  `1_Pooling/config.json` and needs no `--pooling` flag; a CLS model such as
  gte-modernbert-base just works. This script reads the same file only to print
  what Kohagi will pick, and to flag a model that ships none (a reranker or a
  base LM, not a sentence encoder — Kohagi falls back to mean and warns).
- **retrieval structure** — each query's correct document must rank first.
- **paraphrase clustering** — paraphrase pairs must sit closer than unrelated
  sentences.
- **bf16 vs f32**, when the CPU supports it, as a `1 - cosine` figure.

The built-in corpus is English; for a non-English model, pass `--prefix-doc` /
`--prefix-query` if it expects task prefixes, and read the retrieval line as a
sanity check rather than a benchmark. Requires no Python packages — it shells
out to the Kohagi binary and does the arithmetic in the standard library.
"""

import argparse
import json
import math
import subprocess
import sys
import urllib.request

DOCS = [
    "The domestic cat is a small carnivorous mammal kept as a household pet.",
    "Photosynthesis converts light energy into chemical energy in plant cells.",
    "The Rust compiler enforces memory safety without a garbage collector.",
    "Mount Everest is the highest mountain above sea level, in the Himalayas.",
]
QUERIES = [
    "what animal do people keep at home",
    "how do plants turn sunlight into energy",
    "which language has borrow checking instead of a GC",
    "tallest peak on earth",
]
PAIRS = [
    ("A dog is running through the park.", "A canine sprints across the parkland."),
    ("She published a paper on protein folding.", "Her article about how proteins fold appeared."),
]


def declared_pooling(model):
    """What the checkpoint's 1_Pooling config declares, or None if it ships none.

    `config.json` also has a `classifier_pooling` field, which looks
    authoritative and is not: it configures a classification head. Only
    1_Pooling/config.json describes how the sentence embedding is formed, which
    is why Kohagi (and this script) read that file specifically.
    """
    url = f"https://huggingface.co/{model}/raw/main/1_Pooling/config.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            cfg = json.load(r)
    except Exception:
        return None
    for key, name in (("pooling_mode_cls_token", "cls"), ("pooling_mode_mean_tokens", "mean")):
        if cfg.get(key):
            return name
    return None


def embed(kohagi, model, texts, prefix, precision="f32"):
    """Run Kohagi over `texts`, returning (vectors, error). Pooling is left to
    Kohagi's own autodetection — the point of not passing --pooling here."""
    cmd = [kohagi, "--model-id", model, "--prefix", prefix]
    if precision != "f32":
        cmd += ["--precision", precision]
    stdin = "".join(
        json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n"
        for i, t in enumerate(texts)
    )
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()[:300]
    rows = {}
    for line in r.stdout.splitlines():
        d = json.loads(line)
        rows[d["id"]] = d["embedding"]
    return [rows[i] for i in range(len(texts))], None


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def cos_dist(u, v):
    return 1 - dot(u, v) / (math.sqrt(dot(u, u)) * math.sqrt(dot(v, v)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="Hugging Face model id, e.g. nomic-ai/modernbert-embed-base")
    p.add_argument("--kohagi", default="kohagi", help="path to the kohagi binary")
    p.add_argument("--prefix-doc", default="", help="prefix for documents (model-specific)")
    p.add_argument("--prefix-query", default="", help="prefix for queries (model-specific)")
    args = p.parse_args()

    print(f"model    : {args.model}")
    pooling = declared_pooling(args.model)
    if pooling is None:
        print("pooling  : no 1_Pooling/config.json — Kohagi warns and falls back "
              "to mean; this may not be a sentence encoder")
    else:
        note = "" if pooling == "mean" else "   (Kohagi autodetects; no flag needed)"
        print(f"pooling  : {pooling}{note}")

    docs, err = embed(args.kohagi, args.model, DOCS, args.prefix_doc)
    if docs is None:
        print(f"FAILED   : {err}")
        return 1
    queries, _ = embed(args.kohagi, args.model, QUERIES, args.prefix_query)

    print(f"dims     : {len(docs[0])}")
    norms = [math.sqrt(dot(d, d)) for d in docs]
    print(f"L2 norm  : {min(norms):.6f}..{max(norms):.6f}")

    hits, margins = 0, []
    for i, q in enumerate(queries):
        ranked = sorted(((dot(q, d), j) for j, d in enumerate(docs)), reverse=True)
        hits += ranked[0][1] == i
        margins.append(ranked[0][0] - ranked[1][0])
    print(f"retrieval: {hits}/{len(queries)} correct, "
          f"smallest margin over runner-up {min(margins):+.3f}")

    flat = [t for pair in PAIRS for t in pair]
    v, _ = embed(args.kohagi, args.model, flat, args.prefix_doc)
    within = [dot(v[0], v[1]), dot(v[2], v[3])]
    across = [dot(v[0], v[2]), dot(v[0], v[3]), dot(v[1], v[2]), dot(v[1], v[3])]
    verdict = "OK" if min(within) > max(across) else "FAIL"
    print(f"paraphr. : within {min(within):.3f}-{max(within):.3f}  "
          f"across {min(across):.3f}-{max(across):.3f}  [{verdict}]")

    bf16, err = embed(args.kohagi, args.model, DOCS + QUERIES, args.prefix_doc, "bf16")
    if bf16 is None:
        first = err.splitlines()[0][:80] if err else "?"
        print(f"bf16     : unavailable ({first})")
    else:
        f32, _ = embed(args.kohagi, args.model, DOCS + QUERIES, args.prefix_doc)
        worst = max(cos_dist(f32[i], bf16[i]) for i in range(len(f32)))
        print(f"bf16     : vs f32  worst 1-cos {worst:.2e}")

    ok = hits == len(QUERIES) and verdict == "OK"
    print(f"\n{'OK' if ok else 'WEAK'}: "
          f"{'retrieval and paraphrase structure both hold' if ok else 'structure is weaker than expected — check prefixes and that this is a sentence encoder'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
