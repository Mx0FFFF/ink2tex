#!/usr/bin/env bash
# M5's "the tablet types LaTeX directly into your laptop" — the sanctioned HTTP fallback
# (docs/device.md: usb_f_hid is not in the stock kernel, and DESIGN §7 says ship the
# fallback first). Run `ink2tex --serve` on the tablet, correct the expression in the
# browser, then run this with your cursor in any editor: the corrected LaTeX is fetched
# over usb0 and typed into the focused window.
#
#     scripts/ink2type.sh [host]        # default 10.11.99.1
set -euo pipefail
HOST="${1:-10.11.99.1}"
TEX="$(curl -fsS "http://$HOST:8222/tex")"
[ -n "$TEX" ] || { echo "nothing recognized yet — press Capture in the UI first" >&2; exit 1; }
if command -v xdotool >/dev/null; then
    sleep 0.4  # give the user a beat to focus the target window
    xdotool type --delay 12 -- "$TEX"
elif command -v wtype >/dev/null; then
    sleep 0.4
    wtype -- "$TEX"
else
    # no typing tool: clipboard, then stdout as the last resort
    if command -v xclip >/dev/null; then printf '%s' "$TEX" | xclip -selection clipboard
        echo "(no xdotool/wtype — copied to clipboard instead)" >&2
    else echo "$TEX"; fi
fi
