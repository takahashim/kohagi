#!/usr/bin/env python3
"""Survey every ModernBERT config on the Hugging Face Hub against Kohagi's
CoreML converter.

Answers the question "is the conversion sufficient?" with data rather than a
feeling: fetches config.json (a few KB each) for the most-downloaded ModernBERT
models, runs each through the real `coreml-convert` binary, and reports

  - accepted configs, split into the tested envelope and outside it,
  - rejected configs with the converter's own reasons,
  - models Kohagi cannot use at all (no safetensors),

so that "accepted but never verified" combinations surface with download counts
attached. The first run over 690 configs found 661 accepted and 12 rejected, and
turned up two things worth fixing: `hidden_activation: \"silu\"` was refused
(9 configs, one of them with 325k downloads) and configs written by transformers
5.x could not be loaded at all (94 configs). Both are fixed; a rerun accepts 670
and leaves nothing outside the tested envelope.

Usage (needs the converter built):

    cargo build --release --features coreml,coreml-export --bin coreml-convert
    python3 scripts/survey_modernbert.py [--limit 700] [--min-downloads 50]

Network access to huggingface.co is required; nothing large is downloaded.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter

API = "https://huggingface.co/api/models"
UA = {"User-Agent": "kohagi-survey"}

# What has actually been verified numerically against Kohagi's CPU path
# A config outside this envelope may well work —
# four of these bounds were widened by exactly that check, and silu support was
# added because of it — but it has not been *shown* to work, which is the
# distinction this script exists to make.
TESTED = dict(
    head_dim={26, 32, 64, 80},  # even is required; these are the measured ones
    local_attention={128, 256},
    global_every={1, 3},
    activation={"gelu", "silu"},
)


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r), r.headers.get("Link", "")


def list_models(limit):
    models, url = [], f"{API}?filter=modernbert&sort=downloads&direction=-1&limit=500&full=true"
    while url and len(models) < limit:
        batch, link = get_json(url)
        models.extend(batch)
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1 : part.find(">")]
    return models[:limit]


def classify(binary, config_path):
    """Run the converter's own validation; /dev/null for weights means an
    accepted config fails *after* validation, which is the signal."""
    r = subprocess.run(
        [binary, "--model-path", "/dev/null", "--config-path", config_path,
         "--out-dir", tempfile.gettempdir() + "/kohagi-survey-out",
         "--sequence-lengths", "128"],
        capture_output=True, text=True,
    )
    err = r.stderr
    if "unsupported ModernBERT configuration" in err:
        reasons = [l.strip("- ").split(":")[0] for l in err.splitlines() if l.startswith("- ")]
        return "rejected", reasons
    if "config  :" in err:
        return "accepted", []
    last = err.strip().splitlines()[-1] if err.strip() else "?"
    return "error", [last[:80]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=700)
    ap.add_argument("--min-downloads", type=int, default=50)
    ap.add_argument(
        "--binary",
        default=os.path.join(os.path.dirname(__file__), "..", "target", "release", "coreml-convert"),
    )
    args = ap.parse_args()
    if not os.path.exists(args.binary):
        sys.exit(f"no converter at {args.binary}; build it first (see --help)")

    print("listing models ...", flush=True)
    models = list_models(args.limit * 2)
    targets = [m for m in models if m.get("downloads", 0) >= args.min_downloads
               and not m.get("gated", False)][: args.limit]
    print(f"fetching {len(targets)} configs ...", flush=True)

    tmp = tempfile.mkdtemp(prefix="kohagi-survey-")

    def fetch(m):
        mid = m["id"]
        path = os.path.join(tmp, mid.replace("/", "__") + ".json")
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"https://huggingface.co/{mid}/resolve/main/config.json",
                                       headers=UA), timeout=30) as r:
                open(path, "wb").write(r.read())
            return mid, path
        except Exception:
            return mid, None

    with cf.ThreadPoolExecutor(16) as ex:
        fetched = dict(ex.map(fetch, targets))

    rows = []
    for m in targets:
        path = fetched.get(m["id"])
        if not path:
            continue
        try:
            c = json.load(open(path))
        except Exception:
            continue
        if c.get("model_type") != "modernbert":
            continue
        cls, reasons = classify(args.binary, path)
        heads, hidden = c.get("num_attention_heads"), c.get("hidden_size")
        sib = [s["rfilename"] for s in m.get("siblings", [])]
        rows.append(dict(
            id=m["id"], dl=m.get("downloads", 0), cls=cls, reasons=reasons,
            head_dim=(hidden // heads if heads else None),
            local=c.get("local_attention"), every=c.get("global_attn_every_n_layers"),
            act=c.get("hidden_activation"),
            new_rope_only=("rope_parameters" in c and "global_rope_theta" not in c),
            has_safetensors=any(s.endswith(".safetensors") for s in sib),
        ))

    acc = [r for r in rows if r["cls"] == "accepted"]
    outside = [r for r in acc if r["head_dim"] not in TESTED["head_dim"]
               or r["local"] not in TESTED["local_attention"]
               or r["every"] not in TESTED["global_every"]]
    print(f"\n{len(rows)} ModernBERT configs: "
          f"{len(acc)} accepted, "
          f"{sum(r['cls'] == 'rejected' for r in rows)} rejected, "
          f"{sum(r['cls'] == 'error' for r in rows)} errors")
    print("rejection reasons:", Counter(tuple(r["reasons"]) for r in rows if r["cls"] == "rejected"))
    print("no safetensors   :", sum(not r["has_safetensors"] for r in acc),
          "accepted models Kohagi cannot load at all")
    print("rope_parameters-only:", sum(r["new_rope_only"] for r in rows), "configs")
    print(f"\naccepted but OUTSIDE the tested envelope: {len(outside)}")
    for r in sorted(outside, key=lambda r: -r["dl"])[:15]:
        print(f"  DL{r['dl']:>9}  {r['id']:<55} head_dim={r['head_dim']} "
              f"local={r['local']} every={r['every']}")
    if outside:
        print("\nthese convert without complaint and have never been checked against the")
        print("CPU path; verify the most-downloaded ones and widen TESTED, or tighten the")
        print("converter's checks.")

    out = os.path.join(tmp, "survey.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nfull rows -> {out}")


if __name__ == "__main__":
    main()
