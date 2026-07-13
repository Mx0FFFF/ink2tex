#!/usr/bin/env python3
"""Measure the M2 gate AND M4's correction metric over the self-handwritten corpus.

    python3 train/eval_m2.py [--dir train/collected/m2]

One session, two gates, one measurement convention (the CROHME normalizer, the shipped
binary):

  M2: >85% of expressions exact-match with zero corrections.
  M4: the MEDIAN expression needs ≤2 corrections — and corrections are computable, not
      counted by hand: a symbol whose ground-truth token is not the top pick but IS in
      its top-5 candidates costs exactly one tap in the UI. A symbol whose truth is not
      in the candidates at all is uncorrectable-by-tap and scores as infinity (it would
      need retyping) — hiding those would flatter the median dishonestly.
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
    ap.add_argument("--model", default="train/expr.iwt")
    ap.add_argument("--labels", default="train/expr.labels.txt")
    ap.add_argument("--only", help="comma-separated stems: evaluate this subset only "
                                   "(a held-out fold during retrain previews)")
    args = ap.parse_args()

    import json as _json
    pairs = sorted(f[:-4] for f in os.listdir(args.dir) if f.endswith(".ink"))
    if args.only:
        keep = set(args.only.split(","))
        pairs = [p for p in pairs if p in keep]
    if not pairs:
        sys.exit(f"no corpus in {args.dir} — run collect_expressions.py first")
    exact, results, corrections = 0, [], []
    for stem in pairs:
        gt = open(os.path.join(args.dir, f"{stem}.gt.txt")).read().strip()
        r = subprocess.run(
            [args.bin, "--recognize-expr", os.path.join(args.dir, f"{stem}.ink"),
             "--model", args.model, "--labels", args.labels],
            capture_output=True, text=True, timeout=120)
        pred = next((l[7:] for l in r.stdout.splitlines() if l.startswith("LaTeX: ")), "")
        ok = normalize(pred) == normalize(gt)
        exact += ok

        # M4's metric: taps to truth. Parse the per-symbol candidate lists off stdout.
        # GT tokenization: single chars (the M2 targets are built from single-char tokens).
        gt_tokens = list(normalize(gt))
        sym_cands, cur = [], None
        for line in r.stdout.splitlines():
            if line.strip().startswith("symbol "):
                cur = []
                sym_cands.append(cur)
            elif cur is not None and "%" in line and len(line.split()) >= 2:
                cur.append(line.split()[-1])
        if ok:
            taps = 0
        elif len(sym_cands) != len(gt_tokens):
            taps = float("inf")  # segmentation miscount: not fixable by tapping
        else:
            taps = 0
            from eval_crohme import normalize as _n
            for tok, cands in zip(gt_tokens, sym_cands):
                shown = [_n(__import__("re").sub(r"^latex:.*:", "", c)) for c in cands]
                # candidates print as raw labels; compare via the display command
                cmds = [_n(c) for c in cands]
                if cands and (tok == cmds[0] or tok == shown[0]):
                    continue
                if tok in cmds or tok in shown:
                    taps += 1
                else:
                    taps = float("inf")
                    break
        corrections.append(taps)
        results.append((stem, ok, gt, pred, taps))

    n = len(pairs)
    finite = sorted(c for c in corrections)
    median = finite[n // 2] if n else float("inf")
    print(f"M2 gate: {exact}/{n} exact = {100*exact/n:.1f}%   (gate: >85%)")
    print(f"M4 gate: median corrections/expression = {median}   (gate: ≤2; ∞ = not tap-fixable)")
    dist = {}
    for c in corrections:
        k = "∞" if c == float("inf") else str(int(c))
        dist[k] = dist.get(k, 0) + 1
    print(f"  distribution: {dict(sorted(dist.items()))}")
    for stem, ok, gt, pred, taps in results:
        if not ok:
            t = "∞" if taps == float("inf") else int(taps)
            print(f"  ✗ {stem} [{t} taps]: wrote {gt!r} → read {pred!r}")

if __name__ == "__main__":
    main()
