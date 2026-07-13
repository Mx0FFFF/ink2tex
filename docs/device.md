# crates/rm — the device layer

The **only** device-coupled code in the repo. Keep it under ~500 lines. If it passes ~800, something belongs in `crates/core`.

Its job: read the pen, draw pixels, convert to/from core's normalized types. Nothing else.

## Rules

- **No panics.** No `unwrap()`, no `expect()` in runtime paths. This is a foreground app on an appliance the human is physically holding. A panic means a dead screen and a confused user.
- **Never hardcode `/dev/input/eventN`.** Event numbering is not stable across devices or firmware. Enumerate `/dev/input/event*` and probe with `EVIOCGBIT` / `EVIOCGNAME` for the device advertising `ABS_PRESSURE` + `BTN_TOOL_PEN`.
- **Don't touch xochitl internals.** We depend on the pen and the framebuffer. That's the whole contract, and it's why we survive OS updates that break everyone else.
- **Don't reimplement rm2fb.** Use libremarkable — it bundles the client.
- The digitizer coordinate space is **rotated relative to the display and roughly 10× its resolution** (~20k×15k vs 1872×1404). Get the transform right **once**, in one function, and unit-test it against known corners.

## ⚠️ DEVICE FACTS — verify on first contact, then fill this in

Confirmed on hardware (device: reMarkable 2.0, root SSH over USB at `10.11.99.1`).
Keep this honest — re-verify after a firmware update. This file is committed +
auto-loaded, so every fact here is free for future sessions.

> **Firmware moves under you.** The kernel read `5.4.70-v1.6.2-rm11x` on 2026-07-11 and
> `5.4.70-v1.6.3-rm11x` on 2026-07-12 — the tablet updated itself between sessions. Nothing
> broke (evdev, the digitizer ranges and the transform all still hold), but *check it*, don't
> assume: `ssh root@10.11.99.1 uname -srm`.

**SSH is key-based now** — `~/.ssh/id_ed25519_rm`, with a `~/.ssh/config` entry for
`10.11.99.1`, so `ssh root@10.11.99.1` and every `make` device target run unattended. No
password needed (it is still in Settings → Help → Copyrights and licenses if the key is
ever lost). This is what lets an agent do device work without a human at the keyboard.

| Question | How to check | Answer |
|---|---|---|
| SoC / core / NEON present? | `cat /proc/cpuinfo` | ✅ **Freescale i.MX7 Dual**, 2× Cortex-A7 (ARMv7l rev 5). **NEON present** (`neon vfpv3 vfpv4 idiva idivt`) → the M1 int8 kernel gets a NEON path. `armv7l`. |
| Which event device is the digitizer? | `cat /proc/bus/input/devices` | ✅ **`/dev/input/event1`** — `Wacom I2C Digitizer` (Vendor `0x2d1f`, Product `0x0095`), `ABS=0xf000003`; also symlinked `…/input/touchscreen0`. (event0 = power key; event2 = `pt_mt` capacitive finger touch.) **Still enumerate + ioctl-probe — do not hardcode `event1`**; numbering isn't guaranteed across firmware. |
| Digitizer axis ranges + orientation | `ink2tex-rm --probe` (EVIOCGABS; no `evtest`/python on device — BusyBox) | ✅ **Confirmed 2026-07-11:** `ABS_X 0..20966` (res 100), `ABS_Y 0..15725` (res 100), `ABS_PRESSURE 0..4095`, `ABS_TILT_X/Y −9000..9000`, `ABS_DISTANCE 0..255`. Digitizer aspect `20966/15725 = 1.333` = display 4:3. rm reads these **live at startup** (no hardcoding); `crates/rm/src/transform.rs` maps digitizer→normalized. ✅ **Orientation confirmed 2026-07-11** — a captured 'R' renders upright and un-mirrored, so no flip is needed. |
| Must the **rm2fb server** be running? | try running without it | ⚠️ **`/dev/fb0` is NOT the logical display** — `virtual_size` reads `260,23936` (≠ 1404×1872), so direct fb0 writes won't render. Must go through **libremarkable / rm2fb**. ✅ **Confirmed 2026-07-11: rm2fb is NOT installed** (`systemctl is-active rm2fb` → `inactive`; nothing in `/opt/bin`, `/opt/lib`, `/dev/shm`). So on-screen `--ink` needs rm2fb installed first (Toltec `display` pkg). Capture (`--record`) needs no framebuffer and runs alongside xochitl — **verified: 12 strokes / 2745 points captured**. |
| Does the digitizer report the **eraser end**, and does it look like drawing? | `cat /proc/bus/input/devices` → the `KEY=` bitmask | ✅ **Yes — and it looks *identical* to drawing.** `KEY=1c03` (word 10) sets bits `0x140` `BTN_TOOL_PEN`, **`0x141` `BTN_TOOL_RUBBER`**, `0x14a` `BTN_TOUCH`, `0x14b`/`0x14c` `BTN_STYLUS`/`2`. While the **eraser** is in range the kernel still emits `BTN_TOUCH` and a full `ABS_X/Y/PRESSURE` stream — so a capture that watches only `BTN_TOUCH` records *erasing* as ink. `capture.rs` gates on which tool is in range. **Anything you add that reads the pen must ask the same question.** |
| Can we tell what xochitl thinks the pen is doing (lasso, highlighter, eraser tool)? | — | ❌ **No, and there is no bit for it.** We read raw evdev *below* xochitl: a selection lasso, a toolbar tap and a drawn symbol are all just the tip on glass. This is not a bug we can fix in `crates/rm` — it is the cost of the "we depend only on the pen and the framebuffer" contract. **The fix is to own the screen while capturing** (stop xochitl, or go through rm2fb). Until then, `--recognize` assumes xochitl is on a blank page with the pen tool selected. |
| Is `systemctl stop xochitl` sufficient, or does the watchdog (`WatchdogSec`) restart it? | `systemctl cat xochitl` | ✅ **`Restart=on-failure`, `WatchdogSec=60`.** A crash/kill is restarted within ~60 s; use a clean `systemctl stop xochitl` (what `make run` does) and expect the watchdog to bring xochitl back afterward. |
| Is `usb_f_hid` in the stock kernel? (M5 "tablet types LaTeX") | `modprobe usb_f_hid` | ❌ **NOT available** — "Module usb_f_hid not found" in `/lib/modules/5.4.70-v1.6.2-rm11x`; the loaded gadget is `g_ether`. ⇒ M5 needs a kernel module **or** the HTTP-over-`usb0` fallback (DESIGN.md §7 — ship the fallback first). |
| Actual pen→screen latency | measure with the live inker | ⏳ **PENDING** — needs rm2fb + the on-screen inker. (This is the *inking* loop, not inference — see the row below.) |
| Inference latency + does armv7 agree with x86? | `ink2tex-rm --recognize --from <ink>` | ✅ **Re-confirmed 2026-07-12** on the full-corpus model: **mean 15.6 ms/symbol** (max 17.8, n=9) — well inside M1's `<50 ms`. And the Cortex-A7 top-5 is **bit-identical to x86**, probabilities to the last decimal, over 3 symbols. The hand-rolled int8 kernel is arch-consistent: no float drift, no NEON surprises. Re-run this after any change to `core::classify` — it is the cheapest possible check that the device still agrees with the trainer. |

## ⛔ Toltec bootstrap will SOFT-BRICK this device — do not run it

Toltec supports OS builds **2.6.1.71 – 3.3.2.1666** and its own site warns installing on
anything newer soft-bricks. This tablet runs a 2026-06 build (`/etc/version` 20260629…),
far past the ceiling. Consequences, recorded 2026-07-13:
- **No rm2fb** on this device until the display-shim ecosystem catches up with new OS.
  The `--ink` DU-waveform renderer stays dormant (code kept; compiles; unverifiable here).
- **On-screen inking is delivered by cohabitation instead**: capture reads evdev in
  parallel with xochitl, which renders the pen at its native ~21 ms. Every collection and
  live test in this project ran that way.
- Packages we publish to Toltec serve *its* users (on supported firmware); this device
  sideloads via `make ipk` / `scp`.

**Back up the device before the first deploy.** Soft-bricking is real; the community wiki is full of people who skipped this.

## Waveforms

E-Ink refresh mode is a real tradeoff, not a detail. Use a fast, low-quality waveform (`DU`-class) for live inking — it's the difference between the pen feeling attached to the ink and feeling like a laggy mess. Use a high-quality full refresh only for UI transitions and to clear ghosting. If inking feels wrong, this is almost always why.

## Self-screenshot

Add a debug flag that dumps the framebuffer to `/tmp/screen.png` (libremarkable's framebuffer + the `image` crate, ~10 lines). `make screenshot` pulls it back. Without this, nobody can verify device rendering without physically looking at the tablet — which means the agent can't verify it at all.
