#!/usr/bin/env python3
"""Build the reproducible, local-only data set for ``decide_eval.py``.

The output deliberately is not committed.  It contains redistributed MASSIVE
examples (CC BY 4.0), while the source archive and the exact selection are
recorded in the adjacent JSON manifest.  The three splits exercise distinct
claims from notes/plan-decide.md section 6:

* ``main``: MASSIVE's 18 scenarios, an intent-routing shape seen in training;
* ``unknown``: MASSIVE's finer intents (59 present in ja-JP test), a label
  system the Jev model was not trained to choose (it saw scenarios only);
* ``long``: scenario routing with a neutral audit tail that exceeds the model's
  512-token state window.  The utterance is deliberately first, so this is a
  truncation/cost stress slice, not a claim about naturally long conversations.

MASSIVE test is never used to tune a threshold.  ``--seed`` fixes all sampling
and candidate ordering, making a later run comparable rather than convenient
to re-sample until its result looks good.
"""

import argparse
import hashlib
import json
import random
import tarfile
import urllib.request
from pathlib import Path


MASSIVE_VERSION = "1.1"
MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.1.tar.gz"
)
SOURCE_PATH = f"{MASSIVE_VERSION}/data/ja-JP.jsonl"
QUESTION_SCENARIO = "この発話はどの分野に関するものですか？"
QUESTION_INTENT = "この発話でユーザーがしたいことは何ですか？"


def rendered(label):
    """Keep stable machine labels while giving Japanese text a readable menu."""
    return label.replace("_", " — ")


def read_rows(path):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def source_jsonl(args):
    if args.massive_jsonl:
        return Path(args.massive_jsonl)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "massive-1.1-ja-JP.jsonl"
    if destination.exists():
        return destination

    archive = cache / "amazon-massive-dataset-1.1.tar.gz"
    # A command runner may interrupt a large download.  Do not let its partial
    # destination become a permanent, apparently valid cache hit.
    archive_ok = False
    if archive.exists():
        try:
            with tarfile.open(archive, "r:gz") as tar:
                archive_ok = tar.getmember(SOURCE_PATH).isfile()
        except (tarfile.ReadError, EOFError):
            archive.unlink()
    if not archive_ok:
        partial = archive.with_suffix(archive.suffix + ".part")
        partial.unlink(missing_ok=True)
        print(f"downloading {MASSIVE_URL}")
        with urllib.request.urlopen(MASSIVE_URL, timeout=120) as response, open(partial, "wb") as out:
            expected = response.headers.get("Content-Length")
            while block := response.read(1024 * 1024):
                out.write(block)
        if expected is not None and partial.stat().st_size != int(expected):
            partial.unlink(missing_ok=True)
            raise RuntimeError("MASSIVE download ended before Content-Length bytes arrived")
        try:
            with tarfile.open(partial, "r:gz") as tar:
                if not tar.getmember(SOURCE_PATH).isfile():
                    raise RuntimeError(f"archive has no {SOURCE_PATH}")
        except (tarfile.ReadError, EOFError) as error:
            partial.unlink(missing_ok=True)
            raise RuntimeError("MASSIVE archive is incomplete or corrupt") from error
        partial.replace(archive)
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(SOURCE_PATH)
        with tar.extractfile(member) as source, open(destination, "wb") as out:
            assert source is not None
            while block := source.read(1024 * 1024):
                out.write(block)
    return destination


def long_state(utterance, tail_repetitions):
    # 400 repetitions are intentionally much longer than 512 Japanese tokens.
    # The task signal remains before the tail, so this detects a harmful
    # truncation policy without changing the answer.
    return utterance + "\n\n監査ログ（意図判定には無関係）:\n" + ("確認済み。" * tail_repetitions)


def record(row, split, question, candidates, gold, state=None):
    return {
        "id": f"massive-1.1-ja-JP-{split}-{row['id']}",
        "split": split,
        "question": question,
        "state": row["utt"] if state is None else state,
        "candidates": candidates,
        "gold": rendered(gold),
        "source": {
            "dataset": "MASSIVE 1.1",
            "locale": "ja-JP",
            "partition": "test",
            "source_id": row["id"],
        },
    }


def build(rows, args):
    test = [row for row in rows if row["partition"] == "test"]
    if not test:
        raise ValueError("MASSIVE source contains no test partition")
    rng = random.Random(args.seed)
    rng.shuffle(test)

    # Allocate disjoint source records.  It prevents the same utterance from
    # contributing both a favorable in-distribution and unfavorable unknown
    # score, while keeping every source example in the held-out test partition.
    main_rows = test[:args.main_count]
    unknown_rows = test[args.main_count:args.main_count + args.unknown_count]
    long_rows = test[args.main_count + args.unknown_count:
                     args.main_count + args.unknown_count + args.long_count]
    if len(long_rows) != args.long_count:
        raise ValueError("not enough MASSIVE test rows for requested split sizes")

    scenarios = [rendered(value) for value in sorted({row["scenario"] for row in test})]
    intents = [rendered(value) for value in sorted({row["intent"] for row in test})]
    rows_out = []
    rows_out.extend(record(row, "main", QUESTION_SCENARIO, scenarios, row["scenario"])
                    for row in main_rows)
    rows_out.extend(record(row, "unknown", QUESTION_INTENT, intents, row["intent"])
                    for row in unknown_rows)
    rows_out.extend(record(row, "long", QUESTION_SCENARIO, scenarios, row["scenario"],
                           long_state(row["utt"], args.long_tail_repetitions))
                    for row in long_rows)
    return rows_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output JSONL path (not for version control)")
    parser.add_argument("--cache", default=".cache/decide-eval", help="download/extraction cache")
    parser.add_argument("--massive-jsonl", help="use an already extracted ja-JP.jsonl instead")
    parser.add_argument("--main-count", type=int, default=300)
    parser.add_argument("--unknown-count", type=int, default=300)
    parser.add_argument("--long-count", type=int, default=100)
    parser.add_argument("--long-tail-repetitions", type=int, default=400)
    parser.add_argument("--seed", type=int, default=122)
    args = parser.parse_args()
    if min(args.main_count, args.unknown_count, args.long_count) < 1:
        parser.error("each split count must be positive")
    if args.long_tail_repetitions < 1:
        parser.error("--long-tail-repetitions must be positive")

    source = source_jsonl(args)
    raw = source.read_bytes()
    records = build(read_rows(source), args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as destination:
        for row in records:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "MASSIVE",
        "version": MASSIVE_VERSION,
        "license": "CC BY 4.0",
        "locale": "ja-JP",
        "partition": "test",
        "source_url": MASSIVE_URL,
        "source_path": SOURCE_PATH,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "seed": args.seed,
        "splits": {"main": args.main_count, "unknown": args.unknown_count, "long": args.long_count},
        "long_tail_repetitions": args.long_tail_repetitions,
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} records to {out}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
