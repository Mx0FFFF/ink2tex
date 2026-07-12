#!/usr/bin/env bash
# Build an installable Toltec/opkg package (.ipk) from the current tree.
#
# The Toltec repo builds recipes itself, in a clean Docker image, from a *public* source
# URL (see packaging/toltec/ink2tex/package). That is the path for an upstream release.
# This script is the local equivalent: it packages the tree you have, so the package can be
# built, inspected and installed on a device before anything is published.
#
# An .ipk is just an `ar` archive of three members, in this order:
#     debian-binary     the literal string "2.0"
#     control.tar.gz    the metadata (+ maintainer scripts)
#     data.tar.gz       the file tree, rooted at /
#
# ⚠️ The payload layout below MUST match package() in the recipe. If you change one,
#    change the other — they are the same package, built two ways.
set -euo pipefail

cd "$(dirname "$0")/.."

PKG=ink2tex
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)
REL=1
ARCH=rm2
TARGET=armv7-unknown-linux-gnueabihf
BIN=target/$TARGET/release/ink2tex-rm
OUT=target/ipk
IPK="$OUT/${PKG}_${VER}-${REL}_${ARCH}.ipk"

if [ ! -f "$BIN" ]; then
    echo "error: $BIN missing — run 'make build-rm' first" >&2
    exit 1
fi
for asset in train/model.iwt train/model.labels.txt ATTRIBUTION.md; do
    [ -f "$asset" ] || { echo "error: $asset missing" >&2; exit 1; }
done

rm -rf "$OUT"
mkdir -p "$OUT"/pkg "$OUT"/ctl
PKGDIR="$OUT/pkg"

# --- payload (mirrors package() in the recipe) --------------------------------
install -D -m 755 "$BIN"                   "$PKGDIR/opt/bin/$PKG"
install -D -m 644 train/model.iwt          "$PKGDIR/opt/usr/share/$PKG/model.iwt"
install -D -m 644 train/model.labels.txt   "$PKGDIR/opt/usr/share/$PKG/model.labels.txt"
install -D -m 644 ATTRIBUTION.md           "$PKGDIR/opt/usr/share/$PKG/ATTRIBUTION.md"

SIZE=$(du -sb "$PKGDIR" | cut -f1)

# --- control ------------------------------------------------------------------
cat > "$OUT/ctl/control" <<EOF
Package: $PKG
Version: $VER-$REL
Description: Recognize handwritten math symbols and get the LaTeX command — on the tablet, offline.
 Draw one symbol with the pen; get the five most likely LaTeX commands, ranked, in about
 16 ms. Everything runs on the device: no cloud, no network, no account. The classifier is
 a hand-written int8 CNN (no ML runtime) trained on the Detexify dataset (ODbL — see
 /opt/usr/share/ink2tex/ATTRIBUTION.md).
Section: math
Priority: optional
Maintainer: Mx0FFFF <Mx0FFFF@users.noreply.github.com>
License: MIT
Architecture: $ARCH
Homepage: https://github.com/Mx0FFFF/ink2tex
Installed-Size: $SIZE
EOF

cat > "$OUT/ctl/postinst" <<'EOF'
#!/bin/sh
echo ""
echo "ink2tex installed. It is a command-line tool — run it over SSH:"
echo "    ink2tex --recognize      # then draw ONE symbol on the tablet"
echo ""
EOF
chmod 755 "$OUT/ctl/postinst"

# --- assemble -----------------------------------------------------------------
# Deterministic: fixed owner, no gzip timestamp — so the same tree always produces the
# same bytes, which is what makes a checksum in a recipe meaningful.
TAR_OPTS="--owner=0 --group=0 --numeric-owner --sort=name --mtime=@0"
# shellcheck disable=SC2086
tar $TAR_OPTS -C "$PKGDIR" -cf - . | gzip -n9 > "$OUT/data.tar.gz"
# shellcheck disable=SC2086
tar $TAR_OPTS -C "$OUT/ctl" -cf - . | gzip -n9 > "$OUT/control.tar.gz"
echo -n "2.0" > "$OUT/debian-binary"

( cd "$OUT" && ar -rc "$(basename "$IPK")" debian-binary control.tar.gz data.tar.gz )

echo "built $IPK ($(du -h "$IPK" | cut -f1))"
echo
echo "  install:  opkg install $(basename "$IPK")      # needs Toltec on the device"
echo "  inspect:  ar -t $IPK"
