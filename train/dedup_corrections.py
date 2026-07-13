#!/usr/bin/env python3
"""Dedupe the correction-UI flywheel log before it feeds training.

`--serve` logs a sample on EVERY tap that changes a symbol's choice, and again for
every symbol on Accept. A user exploring the candidate buttons therefore leaves
wrong labels in the log (the first live session tried `S` and `\\varsigma` before
settling on `5`). Identical strokes serialize identically, so: group by the stroke
JSON, keep the LAST line — the final choice supersedes every exploratory tap.

    python3 train/dedup_corrections.py train/collected/corrections.ndjson
"""
import json
import sys


def main(path: str) -> None:
    with open(path) as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    last: dict[str, str] = {}
    for l in lines:
        d = json.loads(l)
        last[json.dumps(d["strokes"])] = l
    with open(path, "w") as f:
        f.write("\n".join(last.values()) + "\n")
    labels = [json.loads(l)["key"] for l in last.values()]
    print(f"deduped {len(lines)} -> {len(last)} samples: {labels}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train/collected/corrections.ndjson")
