#!/usr/bin/env bash
# One-shot DEVICE FACTS probe for the reMarkable (see .claude/rules/device.md).
# Gathers everything the device.md table asks for in a single SSH session, then
# you transcribe the answers into that table (it's committed + auto-loaded, so a
# fact recorded once is free for every future session).
#
# Usage:
#   bash scripts/device-facts.sh [user@host]
#   SSH_PASS=... bash scripts/device-facts.sh      # non-interactive (no sshpass needed)
#
# The SSH root password is on the tablet: Settings -> Help -> Copyrights and licenses.
# (That is NOT the screen-unlock PIN.)
set -uo pipefail
HOST="${1:-root@10.11.99.1}"
OUT="${2:-device-facts.out}"

ssh_run() {
  if [ -n "${SSH_PASS:-}" ]; then
    local ap; ap="$(mktemp)"
    printf '#!/bin/sh\necho "%s"\n' "$SSH_PASS" >"$ap"; chmod +x "$ap"
    SSH_ASKPASS_REQUIRE=force SSH_ASKPASS="$ap" setsid -w \
      ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$HOST" "$1"
    local rc=$?; rm -f "$ap"; return $rc
  else
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$HOST" "$1"
  fi
}

# Probes run remotely. Read-only except `modprobe usb_f_hid` (loads a module to
# test availability; harmless and reversible — DESIGN.md §7 asks for exactly this).
read -r -d '' PROBE <<'REMOTE'
echo "==== uname ===="; uname -a
echo "==== SoC / cores / NEON  (device.md row 1) ===="; grep -Ei 'model name|Features|Hardware|processor' /proc/cpuinfo
echo "==== input devices  (device.md row 2: which eventN is the digitizer) ===="; cat /proc/bus/input/devices
echo "==== /dev/input nodes ===="; ls -l /dev/input/
DIG=$(grep -iE -A6 'Name=.*(wacom|digitizer|pen|stylus)' /proc/bus/input/devices | grep -o 'event[0-9]*' | head -1)
echo "digitizer node guess: ${DIG:-UNKNOWN}"
echo "==== digitizer axis ranges  (device.md row 3) ===="
if command -v evtest >/dev/null 2>&1 && [ -n "$DIG" ]; then
  # evtest prints ABS_X/ABS_Y/ABS_PRESSURE/ABS_TILT ranges in its header, then streams.
  timeout 2 evtest "/dev/input/$DIG" </dev/null 2>&1 | sed -n '1,50p'
  echo "(NOTE: orientation-at-corners still needs a human — touch each corner and watch ABS_X/ABS_Y.)"
else
  echo "evtest not installed or node unknown; ranges must be read via ioctl (see crates/rm)."
fi
echo "==== xochitl watchdog / restart policy  (device.md row 5) ===="; systemctl cat xochitl 2>/dev/null | grep -iE 'Watchdog|Restart' || echo "(no xochitl unit or no watchdog/restart directives)"
echo "==== rm2fb / display service present?  (device.md row 4) ===="; systemctl list-units --all 2>/dev/null | grep -iE 'rm2fb|display' || echo "(no rm2fb/display unit visible)"; ls -l /opt/bin 2>/dev/null | grep -iE 'rm2fb|display' || true
echo "==== framebuffer ===="; ls -l /dev/fb* 2>/dev/null; cat /sys/class/graphics/fb0/virtual_size 2>/dev/null
echo "==== usb_f_hid available?  (device.md row 6, needed for M5) ===="; (modprobe usb_f_hid 2>&1 && echo 'modprobe usb_f_hid: OK') || echo 'modprobe usb_f_hid: FAILED'; find /lib/modules -name 'usb_f_hid*' 2>/dev/null; ls /sys/kernel/config/usb_gadget/ 2>/dev/null || echo "(configfs usb_gadget not mounted)"
echo "==== END ===="
REMOTE

echo "Probing $HOST ..." >&2
if ssh_run "$PROBE" | tee "$OUT"; then
  echo >&2
  echo "Raw output saved to $OUT. Transcribe answers into .claude/rules/device.md." >&2
else
  echo "SSH failed. Check the password (Settings -> Help -> Copyrights and licenses) and USB link." >&2
  exit 1
fi
