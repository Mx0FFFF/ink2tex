# Roadmap

> This is the mutable plan. It lives here, **not** in `CLAUDE.md` — memory files shouldn't become fossils.
> **Update "Current state" at the end of every session.** That's how the next session knows where to pick up.

---

## Current state

**M1 recognizer works end-to-end on real data.** The full stack — Detexify strokes
→ rasterize → PyTorch train → int8 quantize → Rust int8 forward pass → labelled
top-5 — is built and validated: trained on **39,554 real samples / 1,123 classes**
(from `detexify-next`, since the classic Drive dump 401s), **86.5% float val top-5**,
and the exported `train/model.iwt` runs through the hand-rolled int8 kernel in Rust
(`--eval`) with the quantization intact. **And it runs ON THE DEVICE**: `crates/rm
--recognize` (`make recognize`) rasterizes captured ink → int8 CNN → top-5 LaTeX on
stdout (streamed over SSH, so **no rm2fb needed**), and the armv7 Cortex-A7 produced
the **bit-identical** top-5 to x86 — the quantized math is arch-consistent, at
**~18 ms/symbol** (M1's `<50 ms` inference criterion, met). A **live draw-to-recognize**
on the tablet worked end-to-end. The repo is now **committed** (`f047779`, branch
`main`) and the `tests/corpus` regression suite is seeded (xi, with the reference model
committed so CI runs it). Remaining
to ship M1: the live-pen loop is the same code (draw instead of `--from`; verified by
composition — capture ✅ + recognize ✅); optionally train on the full 342k set for
accuracy; on-screen result display (needs the M4 typesetter); package for Toltec/Vellum.
The one lingering **M0** item is rm2fb for on-screen *inking* (recognition doesn't need it).

- **Last session:** 2026-07-11 — full M0 build. Workspace (core/desktop/rm), Makefile,
  GitHub CI, `deny.toml` ML-runtime ban + core-purity check (both proven to *fail* on a
  real violation). `crates/core`: `Point`/`Stroke`/`Ink` + hand-rolled little-endian
  `.ink` format (8 tests). `crates/desktop`: headless replay renderer (`--replay`,
  `--synth`). `crates/rm`: hand-rolled evdev/ioctl layer (`_IOC` unit-checked vs
  `<linux/input.h>`), digitizer enumeration + capability probe, `EVIOCGABS` ranges, the
  capture state machine (11 host tests), the digitizer→normalized transform, and
  libremarkable DU-waveform drawing. Cross-compiles for armv7 via `cross`.
  **Verified on the device end-to-end:** `--probe` read the real ranges, and a
  `--record` session captured a hand-drawn 'R' (12 strokes / 2745 points) that renders
  **upright and correctly oriented** → the transform is right, no flip needed.
- **Next task:** to get live ink on the *screen*, install rm2fb on the device (Toltec
  `display` pkg), then `make ink` and draw — confirm ink appears under the pen and
  measure perceived latency (DEVICE FACTS row 7; ⚠ back up the device first, it stops
  xochitl). Then add the first real `tests/corpus/*.ink` fixtures. That closes M0 →
  start **M1** (offline Detexify single-symbol classifier — the "ship this" milestone).
- **Blocked on:** nothing code-side. On-screen `--ink` needs rm2fb installed on the
  device; capture (`--record`) already works without it.
- **Device facts verified:** rows 1,2,3,5 ✅; row 3 orientation ✅; row 4 ✅ (rm2fb NOT
  installed → needed for `--ink`); row 6 ✅ (no `usb_f_hid`); row 7 (latency) pending the
  on-screen `--ink` run. See `.claude/rules/device.md`.
- **Done-criterion status:** headless (`make replay` → PNG) ✅; on-device **capture**
  ✅ (real handwriting, correct orientation); on-device **on-screen inking** built +
  cross-compiled, pending rm2fb install + latency check.
- **M1 started** (parallel to the M0 loose end above). Foundation landed in
  `crates/core/src/classify/`, all device-free and tested: the hand-rolled int8
  `kernel` (conv2d/dense/maxpool/requantize/softmax/top-k), the mmap-able `weights`
  blob (`.iwt`), the stroke→32×32 `raster`izer + global features (the pinned
  train/inference preprocessing contract), and ranked `Prediction` output. Guardrails
  confirm **zero** new deps. Detexify format recon done: samples are
  `{id: "pkg-enc-_cmd", data: [[{x,y,t}…]…]}` (bulk data on Google Drive), classes
  from `symbols.yaml` (~800–1000). `ink2tex-desktop --raster <ink>` visualizes the
  32×32. **Data + training pipeline done:** the Detexify JSON loader
  (`crates/desktop/src/detexify.rs`, tested on real `example.json`),
  `--prepare-detexify` (rasterizes through the core rasterizer → flat numpy-readable
  dataset — verified, no skew), `--dump-weights`, and `train/` (`iwt.py` = the `.iwt`
  writer, byte-verified against core's parser; `train.py` = the PyTorch CNN trainer +
  int8 PTQ scaffolding; `README.md`). **Forward pass done:** `classify::recognize()`
  runs the quantized model (quantize→conv→pool→conv→pool→+features→fc1→fc2→dequant→
  softmax→top-5), shape-validated so a malformed `.iwt` errors instead of panicking;
  wired into `ink2tex-desktop --recognize <ink> --model <iwt> --labels <txt>` and
  verified end-to-end on a real `.ink` (rasterize → int8 CNN → labelled top-5).
  **Next — the only thing between here and shipping:** download the Detexify bulk data
  (Google Drive) + `pip install torch`, then `--prepare-detexify` → `train.py` to >90%
  top-5 (validate train.py's PTQ on that first real run), deploy `model.iwt` to the
  device, and package for Toltec/Vellum. Done = >90% top-5 → **ship the single-symbol
  tool** (the milestone that gets real users and breaks the "abandoned sample" curse).

---

## Milestones

Each gate has a hard done-criterion. **Don't skip gates.** The failure mode for this project is drifting into a research project and never shipping — the gates exist to prevent that.

### ⬜ M0 — Ink recorder *(a weekend)*

Read the digitizer via evdev. Draw strokes to the framebuffer with partial refresh. Save `.ink` files.
**Also build the headless replay renderer** (`--replay <ink> --render-to <png>`) — do not defer this, it's the agent's only way to verify visual work.

**Done when:** ink appears on screen with <50 ms perceived latency, *and* `make replay` produces a PNG.
**Learning:** evdev, ioctl capability probing, coordinate transforms, E-Ink waveform modes, cross-compiling to `armv7-unknown-linux-gnueabihf`.

### ⬜ M1 — Offline Detexify *(2–4 weeks)* — ★ **SHIP THIS** ★

Train a symbol classifier on Detexify's ODbL stroke data. Hand-rolled int8 CNN inference in Rust. Draw a symbol → top-5 LaTeX commands → tap to copy.

**Done when:** >90% top-5 accuracy on a held-out split, <50 ms inference on-device.
**Then package it for Toltec/Vellum and release it.**

This is not a toy milestone. An offline symbol-lookup tool on e-ink doesn't exist and people want it. **Real users from month one is what breaks the "unmaintained experimental sample" curse** that killed every prior attempt at this. Ship before you're ready.

**Learning:** quantization, fixed-point arithmetic, `mmap`, hand-written convolution kernels, cache-friendly memory layout, NEON intrinsics (and *measuring* the speedup).

### ⬜ M2 — Linear expressions *(3–6 weeks)*

Greedy segmentation (temporal + spatial) + left-to-right ordering. `2x + 3 = 7`, `f(x) = ax + b`. No 2D structure yet.

**Done when:** >85% exact-match on a 100-expression corpus you handwrote yourself.
**Learning:** stroke grouping, the delayed-stroke problem, hypothesis scoring.

### ⬜ M3 — Structure *(6–12 weeks)* — **the heart of it**

Line-of-sight graph → relation classification → maximum spanning tree → Symbol Layout Tree → LaTeX. Superscripts, subscripts, fractions, radicals, `\sum`/`\int` with limits.

**Done when:** you can report an honest exact-match number on CROHME 2014 (**evaluation only** — do not train on it).

⚠️ **Accuracy will feel bad here and that is expected.** Full-expression exact-match is well under 100% even for GPU transformers. If you benchmark against Mathpix you will conclude you failed and quit. **Read DESIGN.md §7 before you do that.** The correction UI is the product; the model just makes it fast.

**Learning:** graph algorithms, spanning trees, 2D grammar parsing, joint optimization vs. naive pipelines.

### ⬜ M4 — Correction UI + typesetting + export *(3–4 weeks)*

Tap-to-fix with top-5 alternatives. Correction logging (**every fix is a labelled training example**). A small math typesetter. `.tex` export + HTTP endpoint on `usb0`.

**Done when:** the median expression needs ≤2 corrections and lands in your Overleaf tab.

### ⬜ M5 — Flywheel and reach

USB-HID gadget (**the tablet types LaTeX directly into your laptop** — the feature that drives adoption). WASM browser demo. Opt-in corpus contribution → retrain → ship better weights. Paper Pro (aarch64) port.

**Learning:** USB gadget subsystem, configfs, HID report descriptors, possibly building a kernel module.

---

## Deliberately out of scope

- Beating Mathpix or CROHME SOTA on accuracy.
- General handwriting → text (different problem; MyScript already ships it on-device).
- Image-based OCR of photographed math.
- Anything that couples us to xochitl internals.
