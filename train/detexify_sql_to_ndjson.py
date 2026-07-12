#!/usr/bin/env python3
r"""Stream the classic Detexify PostgreSQL dump → newline-delimited JSON.

    python3 train/detexify_sql_to_ndjson.py ~/Downloads/detexify.sql.gz \
        > train/detexify_raw/detexify.ndjson

    # or pipe it straight in, no 1.2 GB temp file:
    python3 train/detexify_sql_to_ndjson.py detexify.sql.gz \
        | ink2tex-desktop --prepare-detexify - --out-dir train/dataset_full \
              --classes train/dataset/classes.txt

This is the ODbL bulk export (`pg_dump` of kirel/detexify-data). Its payload is one
`COPY` block — Postgres' text format, which is *not* CSV:

    COPY samples (id, key, strokes) FROM stdin;
    1<TAB>latex2e-OT1-_textless<TAB>[[[250,103,1362942716695],...]]
    ...
    \.

Fields are tab-separated, one record per physical line, terminated by a lone `\.`.
Literal tabs/newlines/backslashes inside a value are backslash-escaped, and `\N` means
SQL NULL — so the fields must be *unescaped*, not just split.

This script is deliberately dumb transport: it does not interpret the class key
(`latex2e-OT1-_xi`). That normalization is Detexify-format semantics and lives in
Rust (`crates/desktop/src/detexify.rs::normalize_class`), where it is unit-tested and
shared by every export format — not smeared across a one-off Python script.
"""
import argparse
import gzip
import io
import json
import sys

COPY_HEADER = "COPY samples (id, key, strokes) FROM stdin;"

# Postgres COPY text-format escapes (see: COPY, "Text Format").
UNESCAPE = {
    "\\": "\\", "t": "\t", "n": "\n", "r": "\r",
    "b": "\b", "f": "\f", "v": "\v",
}


def unescape(field: str) -> str:
    r"""Undo COPY text-format escaping. `\N` (SQL NULL) is handled by the caller."""
    if "\\" not in field:  # the overwhelmingly common case — don't pay for it
        return field
    out, i, n = [], 0, len(field)
    while i < n:
        c = field[i]
        if c == "\\" and i + 1 < n:
            nxt = field[i + 1]
            if nxt in UNESCAPE:
                out.append(UNESCAPE[nxt])
                i += 2
                continue
            # \xNN hex and \NNN octal also exist; pass anything else through raw
            # rather than guessing and corrupting a class key.
        out.append(c)
        i += 1
    return "".join(out)


def convert(fh, out) -> dict:
    stats = dict(rows=0, emitted=0, null_strokes=0, empty_key=0, malformed=0)

    for line in fh:  # seek to the data block
        if line.startswith(COPY_HEADER):
            break
    else:
        sys.exit(f"error: no `{COPY_HEADER}` in the dump — is this a Detexify pg_dump?")

    for line in fh:
        line = line.rstrip("\n")
        if line == "\\.":  # end-of-COPY marker
            break
        stats["rows"] += 1

        parts = line.split("\t")
        if len(parts) != 3:
            stats["malformed"] += 1
            continue
        _id, key, strokes = parts

        if strokes == "\\N":
            stats["null_strokes"] += 1
            continue
        key = unescape(key)
        if not key.strip():
            stats["empty_key"] += 1
            continue

        # `strokes` is already valid JSON; splice it in verbatim rather than
        # parse→re-serialize 1.2 GB of number arrays for no reason. Only the key
        # needs JSON quoting.
        out.write(f'{{"key":{json.dumps(key)},"strokes":{unescape(strokes)}}}\n')
        stats["emitted"] += 1

    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump", help="detexify.sql.gz (or an uncompressed .sql)")
    ap.add_argument("-o", "--out", help="output NDJSON (default: stdout)")
    args = ap.parse_args()

    opener = gzip.open if args.dump.endswith(".gz") else open
    # errors="replace": a handful of the 2013-era keys carry stray bytes. Replacing
    # them keeps the stream alive; those keys fail normalization downstream and get
    # dropped there, which is where that decision belongs.
    with opener(args.dump, "rt", encoding="utf-8", errors="replace") as fh:
        out = open(args.out, "w") if args.out else io.TextIOWrapper(sys.stdout.buffer)
        try:
            stats = convert(fh, out)
        finally:
            out.flush()
            if args.out:
                out.close()

    print(
        f"{stats['emitted']} samples emitted from {stats['rows']} COPY rows "
        f"(dropped: {stats['null_strokes']} null-strokes, {stats['empty_key']} empty-key, "
        f"{stats['malformed']} malformed)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
