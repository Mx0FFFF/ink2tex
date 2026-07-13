#!/usr/bin/env python3
"""Guided collection of the M2 gate corpus: 100 handwritten expressions with ground truth.

    python3 train/collect_expressions.py [--host root@10.11.99.1] [--start N]

For each target in train/m2_targets.txt: the terminal shows the expression to write, the
tablet records until the pen goes idle (~2 s), the ink is pulled back, and the pair lands
in train/collected/m2/NNN.ink + NNN.gt.txt. Interrupt any time — progress is per-file, and
--start N resumes. HOLD THE TABLET UPRIGHT (portrait); write ONE line per prompt.

Why guided: M2's done-criterion is ">85% exact-match on a 100-expression corpus you
handwrote yourself" — which needs ground truth, and ground truth you don't have to label
after the fact is ground truth you prompted before the pen touched glass.
"""
import argparse
import os
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="root@10.11.99.1")
    ap.add_argument("--targets", default="train/m2_targets.txt")
    ap.add_argument("--out-dir", default="train/collected/m2")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--idle-ms", type=int, default=2000)
    args = ap.parse_args()

    targets = [t.strip() for t in open(args.targets) if t.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    for i, gt in enumerate(targets):
        if i < args.start:
            continue
        ink_path = os.path.join(args.out_dir, f"{i:03d}.ink")
        if os.path.exists(ink_path):
            continue  # resume-friendly
        print(f"\n[{i+1:>3}/{len(targets)}]  write this on the tablet, then lift the pen:\n")
        print(f"        {gt}\n", flush=True)
        r = subprocess.run(
            ["ssh", args.host,
             f"/home/root/ink2tex-rm --record-one --idle-ms {args.idle_ms} --out /home/root/m2.ink"],
        )
        if r.returncode != 0:
            sys.exit("device capture failed — is the tablet awake and plugged in?")
        subprocess.run(["scp", "-q", f"{args.host}:/home/root/m2.ink", ink_path], check=True)
        with open(os.path.join(args.out_dir, f"{i:03d}.gt.txt"), "w") as f:
            f.write(gt + "\n")
        print("  saved ✓")
    print("\ncorpus complete — measure with: python3 train/eval_m2.py")

if __name__ == "__main__":
    main()
