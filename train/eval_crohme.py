#!/usr/bin/env python3
r"""Evaluate the expression pipeline on CROHME — the M3 gate's honest external number.

    python3 train/eval_crohme.py <crohme-inkml-dir> [--limit N] [--bin PATH]

## The licence rules, which are not optional

CROHME is **CC BY-NC-SA**. Per this project's NON-NEGOTIABLE #3 and DESIGN §5:

  - it is used for **evaluation only** — nothing here may feed training;
  - the dataset lives **outside the repository** (point this script at it) and no
    CROHME-derived ink, image or fixture may ever be committed;
  - the only artifact that leaves this script is the *number*, which goes in ROADMAP.

## What it does

Parses each `.inkml` (online strokes + LaTeX ground truth), writes a temporary `.ink`,
runs `ink2tex-desktop --recognize-expr` — the SAME binary and pipeline the device uses —
and scores:

  - **exact match** after LaTeX normalization (the headline; expect it to be low — the
    roadmap says so out loud, and grammar coverage alone caps it: no matrices, no
    multi-line, no \lim/\text);
  - **symbol-bag F1** (multiset of emitted tokens vs ground truth) — movement here is
    meaningful even when exact-match barely twitches, and it separates "recognizes the
    symbols, fumbles the layout" from "recognizes nothing".

Ground-truth LaTeX in CROHME carries markup variance (`$…$`, `\mbox`, spacing braces),
so both sides pass through the same normalizer. Normalization choices are printed with
the result — a number nobody can reproduce is not a number.
"""
import argparse
import collections
import os
import re
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

NS = {"ink": "http://www.w3.org/2003/InkML"}


def parse_inkml(path):
    """-> (strokes [[(x,y),…],…], ground-truth LaTeX or None)"""
    root = ET.parse(path).getroot()
    gt = None
    for ann in root.findall("ink:annotation", NS):
        if ann.get("type") in ("truth", "latex") and ann.text:
            gt = ann.text.strip()
            break
    strokes = []
    for tr in root.findall("ink:trace", NS):
        pts = []
        for tok in (tr.text or "").strip().split(","):
            nums = tok.split()
            if len(nums) >= 2:
                try:
                    pts.append((float(nums[0]), float(nums[1])))
                except ValueError:
                    pass
        if pts:
            strokes.append(pts)
    return strokes, gt


def write_ink(strokes, path):
    """Normalize into [0,1] (aspect preserved) and emit .ink v1."""
    xs = [x for s in strokes for (x, _) in s]
    ys = [y for s in strokes for (_, y) in s]
    x0, y0 = min(xs), min(ys)
    span = max(max(xs) - x0, max(ys) - y0, 1e-6)
    out = bytearray()
    out += b"INK1" + struct.pack("<HHffI", 1, 0, 1.0, 1.0, len(strokes))
    for s in strokes:
        out += struct.pack("<I", len(s))
        for (x, y) in s:
            out += struct.pack("<5fQ", (x - x0) / span, (y - y0) / span, 1.0, 0.0, 0.0, 0)
    with open(path, "wb") as f:
        f.write(bytes(out))


def normalize(tex):
    """Both sides go through this. Every choice listed here is part of the result."""
    t = tex.strip()
    t = t.strip("$")
    t = re.sub(r"\\mbox\s*{([^}]*)}", r"\1", t)
    t = re.sub(r"\\left|\\right", "", t)          # \left( == (
    t = re.sub(r"\\[,;!:]|~", "", t)              # spacing commands
    t = re.sub(r"\s+", "", t)
    t = t.replace("\\lt", "<").replace("\\gt", ">")
    t = re.sub(r"\\cdot(?![a-zA-Z])", "*", t)      # · vs × vs *: normalize multiplication
    t = re.sub(r"\\times(?![a-zA-Z])", "*", t)
    # single-char groups: x^{2} == x^2
    t = re.sub(r"{(\w)}", r"\1", t)
    return t


def tokens(tex):
    return re.findall(r"\\[a-zA-Z]+|.", tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inkml_dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bin", default="target/release/ink2tex-desktop")
    ap.add_argument("--model", default="train/expr.iwt")
    ap.add_argument("--labels", default="train/expr.labels.txt")
    args = ap.parse_args()

    files = sorted(
        os.path.join(dp, f)
        for dp, _, fs in os.walk(args.inkml_dir)
        for f in fs
        if f.endswith(".inkml")
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"no .inkml under {args.inkml_dir}")

    exact = 0
    scored = 0
    f1_sum = 0.0
    fail_parse = 0
    with tempfile.TemporaryDirectory() as td:
        for i, path in enumerate(files):
            try:
                strokes, gt = parse_inkml(path)
            except ET.ParseError:
                fail_parse += 1
                continue
            if not strokes or not gt:
                fail_parse += 1
                continue
            ink_path = os.path.join(td, "e.ink")
            write_ink(strokes, ink_path)
            r = subprocess.run(
                [args.bin, "--recognize-expr", ink_path, "--model", args.model,
                 "--labels", args.labels],
                capture_output=True, text=True, timeout=120,
            )
            pred = ""
            for line in r.stdout.splitlines():
                if line.startswith("LaTeX: "):
                    pred = line[len("LaTeX: "):]
                    break
            p, g = normalize(pred), normalize(gt)
            scored += 1
            if p == g:
                exact += 1
            pt, gt_t = collections.Counter(tokens(p)), collections.Counter(tokens(g))
            inter = sum((pt & gt_t).values())
            prec = inter / max(sum(pt.values()), 1)
            rec = inter / max(sum(gt_t.values()), 1)
            f1_sum += 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
            if (i + 1) % 100 == 0:
                print(f"  … {i+1}/{len(files)}  exact {exact}/{scored}", flush=True)

    print(f"\nCROHME evaluation ({args.inkml_dir}, n={scored}, {fail_parse} unparseable):")
    print(f"  exact match (normalized): {exact}/{scored} = {100*exact/max(scored,1):.1f}%")
    print(f"  symbol-bag F1 (mean):     {100*f1_sum/max(scored,1):.1f}%")
    print("\nnormalization: strip $ \\mbox \\left/\\right spacing-cmds; ×·→*; {c}→c; no-space")


if __name__ == "__main__":
    main()
