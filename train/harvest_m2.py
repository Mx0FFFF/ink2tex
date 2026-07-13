#!/usr/bin/env python3
"""Turn a guided collection session into per-symbol training samples.

The M2 collector (`collect_expressions.py`) prompts an expression BEFORE the pen
touches glass, so every capture arrives with expression-level ground truth. This
script pushes that truth down to the symbol level by alignment:

    tokens  = tokenize(ground truth)          # '2a+7=3' -> ['2','a','+','7','=','3']
    groups  = ink2tex-desktop --dump-groups   # the REAL pipeline's segmentation
    if len(groups) == len(tokens): zip them   # linear targets read left-to-right

Expressions where the segment count disagrees with the token count are skipped —
that mismatch is exactly the under/over-segmentation (or a truncated capture) that
would poison labels. On honest sessions the yield is high and every sample it emits
carries a label the writer was *asked* to write, which beats any human labelling
pass done after the fact.

    python3 train/harvest_m2.py [--dir train/collected/m2] [--only 000,007,...]
                                [--out train/collected/m2_harvest.ndjson]

`--only` (or `--exclude`) selects expressions by stem — used to keep a held-out
fold's ink out of training entirely.
"""
import argparse
import glob
import json
import os
import re
import subprocess

TOKEN_RE = re.compile(r"\\[a-zA-Z]+|.")


def tokenize(gt: str) -> list[str]:
    return TOKEN_RE.findall(gt.replace(" ", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="train/collected/m2")
    ap.add_argument("--out", default="train/collected/m2_harvest.ndjson")
    ap.add_argument("--bin", default="target/release/ink2tex-desktop")
    ap.add_argument("--model", default="train/expr.iwt")
    ap.add_argument("--labels", default="train/expr.labels.txt")
    ap.add_argument("--only", help="comma-separated stems to include")
    ap.add_argument("--exclude", help="comma-separated stems to skip")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    excl = set(args.exclude.split(",")) if args.exclude else set()

    kept, skipped, samples = 0, [], []
    for ink_path in sorted(glob.glob(os.path.join(args.dir, "*.ink"))):
        stem = os.path.basename(ink_path)[:3]
        if (only is not None and stem not in only) or stem in excl:
            continue
        gt = open(ink_path.replace(".ink", ".gt.txt")).read().strip()
        tokens = tokenize(gt)
        r = subprocess.run(
            [args.bin, "--dump-groups", ink_path, "--model", args.model, "--labels", args.labels],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            skipped.append(f"{stem}(pipeline error)")
            continue
        groups = json.loads(r.stdout)["groups"]
        if len(groups) != len(tokens):
            skipped.append(f"{stem}({len(groups)}seg/{len(tokens)}tok)")
            continue
        for tok, strokes in zip(tokens, groups):
            samples.append({"key": tok, "strokes": strokes})
        kept += 1

    with open(args.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"aligned {kept} expressions -> {len(samples)} labelled symbols -> {args.out}")
    if skipped:
        print(f"skipped {len(skipped)}: {' '.join(skipped)}")


if __name__ == "__main__":
    main()
