#!/usr/bin/env python3
"""Measure the M2 gate: exact-match over the self-handwritten expression corpus.

    python3 train/eval_m2.py [--dir train/collected/m2]

Same binary the device ships, same normalizer as the CROHME harness — one measurement
convention everywhere. Gate: >85% exact match.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_crohme import normalize  # one normalizer for every benchmark

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="train/collected/m2")
    ap.add_argument("--bin", default="target/release/ink2tex-desktop")
    args = ap.parse_args()

    pairs = sorted(f[:-4] for f in os.listdir(args.dir) if f.endswith(".ink"))
    if not pairs:
        sys.exit(f"no corpus in {args.dir} — run collect_expressions.py first")
    exact, results = 0, []
    for stem in pairs:
        gt = open(os.path.join(args.dir, f"{stem}.gt.txt")).read().strip()
        r = subprocess.run(
            [args.bin, "--recognize-expr", os.path.join(args.dir, f"{stem}.ink"),
             "--model", "train/expr.iwt", "--labels", "train/expr.labels.txt"],
            capture_output=True, text=True, timeout=120)
        pred = next((l[7:] for l in r.stdout.splitlines() if l.startswith("LaTeX: ")), "")
        ok = normalize(pred) == normalize(gt)
        exact += ok
        results.append((stem, ok, gt, pred))
    n = len(pairs)
    print(f"M2 corpus: {exact}/{n} exact = {100*exact/n:.1f}%   (gate: >85%)")
    for stem, ok, gt, pred in results:
        if not ok:
            print(f"  ✗ {stem}: wrote {gt!r} → read {pred!r}")

if __name__ == "__main__":
    main()
