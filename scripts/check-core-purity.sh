#!/usr/bin/env bash
# Core-purity guardrail (NON-NEGOTIABLE #2 / docs/core-invariants.md).
#
# crates/core is the entire value of the project and must stay device-free: no
# libremarkable, no /dev/input, no framebuffer, no GUI toolkit, no syscalls. It
# must build and `cargo test` on a laptop. We enforce that structurally by
# inspecting core's *normal* dependency tree and failing if a device/IO crate
# leaked in. If you want one of these, the abstraction is wrong — define a plain
# data type in core and convert at the crates/rm boundary.
set -uo pipefail
cd "$(dirname "$0")/.."

# Crate names that must never appear in ink2tex-core's runtime dep tree.
BANNED=(
  libremarkable evdev evdev-rs framebuffer input-linux
  sdl2 minifb winit softbuffer      # GUI / windowing — desktop-only
  libc nix rustix                    # raw syscalls — core does no I/O
  memmap memmap2 mmap                # mmap happens at the frontend, not in core
  rusb libusb1-sys                   # USB — that's the rm/HID frontend
)

# `--edges normal` = what actually links at runtime (excludes dev/build deps).
# `--prefix none`  = one "name vX.Y.Z" per line; take the name.
names="$(cargo tree -p ink2tex-core --edges normal --prefix none 2>/dev/null \
         | awk 'NF{print $1}' | sort -u)"

if [ -z "$names" ]; then
  echo "core-purity: could not resolve ink2tex-core dependency tree" >&2
  exit 1
fi

violation=0
for b in "${BANNED[@]}"; do
  if printf '%s\n' "$names" | grep -qx "$b"; then
    echo "CORE PURITY VIOLATION: ink2tex-core depends on '$b'" >&2
    violation=1
  fi
done

if [ "$violation" -ne 0 ]; then
  echo "-> core must be device-free (see docs/core-invariants.md)." >&2
  exit 1
fi
echo "core-purity OK — ink2tex-core has no device/IO dependencies."
