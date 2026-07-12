#!/usr/bin/env python3
"""HWRT (write-math) stroke recordings → the same NDJSON the Detexify loader eats.

    python3 train/hwrt_to_ndjson.py hwrt/train-data.csv -o train/detexify_raw/hwrt.ndjson

## Why this dataset exists in the pipeline at all

Detexify has **no `+`, no `-`, no `=`, no digits and no letters** — it is a *symbol-lookup*
corpus, and nobody ever looks up how to type `2` or `x`. That makes `2x + 3 = 7`
unrecognizable no matter how good the segmentation and structure stages are: the classifier
has no such classes to emit. HWRT is the same kind of data (91.9% of it *is* Detexify) but
Martin Thoma's write-math.com also collected the alphabet and the arithmetic, so it carries
the ~26 tokens Detexify is missing.

**It is ODbL** — the same licence as Detexify, so it lands inside the attribution we already
have to give. That is the whole reason it is here and CROHME/MathWriting are not: both of
those have the tokens too, and both are **CC BY-NC-SA**, which must never end up inside a
binary strangers install (DESIGN §5, NON-NEGOTIABLE #3).

    HWRT database, Martin Thoma. https://doi.org/10.5281/zenodo.50022  (ODbL v1.0)

## Still missing after this, and unavailable anywhere permissive: `=`, `(`, `)`

They have to be collected. See ROADMAP.

## Format

`;`-separated CSV: `symbol_id;user_id;data;user_agent`, where `data` is JSON
`[[{"x":…,"y":…,"time":…}, …], …]` — the same shape as Detexify's export except that the
timestamp key is `time` rather than `t` (the Rust loader accepts both).
"""
import argparse
import csv
import json
import sys

csv.field_size_limit(10**9)  # the `data` column is a whole drawing

# The tokens Detexify cannot express. Everything else in HWRT is Detexify data we already
# have, so pulling it in would only duplicate what we trained on.
WANTED = set("0123456789+-<>abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def canonical(latex: str) -> str:
    r"""Key these by their literal LaTeX — `0`, `+`, `x`, `L` — NOT as `latex:latex2e:L`.

    That is not laziness, it avoids a silent data-poisoning collision. Detexify's label
    space already contains `latex:latex2e:L`, `:O`, `:P`, `:S`, `:l`, `:o` — and those are
    the *commands* `\L \O \P \S \l \o` (Ł Ø ¶ § ł ø), **not** the letters. Mapping
    HWRT's plain `L` onto that key would merge two different symbols into one class and
    quietly wreck six of them.

    Literals also fall out correctly at the other end: `symbol_command` passes anything
    without a `latex:` prefix straight through, so `+` renders as `+`. And `structure`'s
    minus-vs-fraction-bar rule keys off the label `"-"`, which now actually exists.
    """
    return latex


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_csv", help="HWRT train-data.csv (or test-data.csv)")
    ap.add_argument("--symbols", help="symbols.csv (default: alongside data_csv)")
    ap.add_argument("-o", "--out", help="output NDJSON (default: stdout)")
    args = ap.parse_args()

    sym_path = args.symbols or args.data_csv.rsplit("/", 1)[0] + "/symbols.csv"
    with open(sym_path) as f:
        symbols = {r["symbol_id"]: r["latex"] for r in csv.DictReader(f, delimiter=";")}

    out = open(args.out, "w") if args.out else sys.stdout
    kept, skipped = 0, 0
    per_class = {}
    with open(args.data_csv) as f:
        for row in csv.DictReader(f, delimiter=";"):
            latex = symbols.get(row["symbol_id"])
            if latex not in WANTED:
                skipped += 1
                continue
            try:
                strokes = json.loads(row["data"])
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not strokes or not any(strokes):
                skipped += 1
                continue
            out.write(json.dumps({"key": canonical(latex), "strokes": strokes}) + "\n")
            per_class[latex] = per_class.get(latex, 0) + 1
            kept += 1
    if args.out:
        out.close()

    print(
        f"{kept} samples over {len(per_class)} classes "
        f"({skipped} rows skipped — Detexify data we already have, or unparseable)",
        file=sys.stderr,
    )
    print("  " + "  ".join(f"{k}:{v}" for k, v in sorted(per_class.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
