# Roadmap

> This is the mutable plan. It lives here, and it is kept honest — plans shouldn't become fossils.
> **Update "Current state" at the end of every session.** That's how the next session knows where to pick up.

---

## Current state

**M1's accuracy gate is MET on the full corpus — and a bug that would have broken the
model on the tablet is fixed.** The classic Detexify bulk dump turned out to be sitting
in `~/Downloads/detexify.sql.gz` all along (this file used to say it was inaccessible):
**210,454 samples, 5.3× the 39,554** we had been training on. It is now ingested
end-to-end, **zero samples dropped**, and the shipped model is trained on it.

**Held-out top-5, through the int8 kernel the device actually runs: 96.8% (micro),
85.9% (macro).** The M1 done-criterion (>90% top-5) is met with room. Quantization is
essentially free — PyTorch float scores 96.86 / 86.31 on the same rows.

The gain, as a *controlled* A/B (identical held-out split, identical hyperparameters,
only the number of training rows differs — `train.py --train-subsample`):

| train rows | top-5 micro | **top-5 macro** | train top-5 |
|---|---|---|---|
| 39,554 (the old corpus's size) | 93.2% | 70.9% | 96.6% |
| **189,554 (full)** | **96.9%** | **86.3%** | 97.6% |

⚠️ **Macro is the number to watch, not micro.** The real corpus is heavily imbalanced
(`\int`: 3,937 samples; median class: 53; 159 classes under ten) — that is genuine usage
frequency, not a defect, but it means micro accuracy is dominated by the head and can
read as excellent while the tail is unusable. The extra data bought **+15.4 points of
macro** and only +3.6 micro: it went almost entirely into the tail, which is exactly
where it was needed. Both `--eval` and `train.py` now report macro.

⚠️ **The old "90.8%" is not comparable to these** and has been retired: it was measured
on a different (class-balanced) val set, and with the broken features below.

**Now capacity-bound, not data-bound.** The train/val gap collapsed from 3.4 points to
**0.7** (97.6 vs 96.9). More data will not move this much further; **model capacity is
the next accuracy lever** — `fc1` is 64 units feeding a 1,123-way softmax, which is a
tight bottleneck. There is inference budget for it (~18 ms/symbol against a 50 ms
criterion).

### 🐛 The bug: `global_features` was not scale-invariant (would have broken on-device)

Five of the seven global features were emitted in **raw coordinate units** — arc length,
and the start/end points straight off the stroke. The bitmap channel is invariant for
free (it aspect-fits) and the online channel is invariant *and tested*; this vector was
neither, and nobody had tested it.

It only ever worked because detexify-next's coordinates *happened* to be normalized like
the device's. The bulk dump ships **raw pixels**, so: scoring the old model on the same
symbols in pixel units gave **7.9% top-5** instead of ~90% — and a model trained on the
dump would have failed exactly as hard **on the tablet, on real ink, in the user's
hands**. (Corroborating detail: the unbounded `w/h` aspect term hit ~8e5 on a minus sign,
and all seven features share one int8 scale with the 1,384 CNN/online activations they
are concatenated to — epoch-0 loss was 125 where `ln(1123) ≈ 7`.)

Features are now dimensionless (relative to the ink's own bbox) and bounded, with the
invariance test that was missing. The invariant is written down in
`docs/core-invariants.md` — **if you add a feature, add its invariance test in the
same commit.** This also silently fixes M2/M3: `line.rs` computes features per *segmented
symbol*, so `sx/sy` used to encode a symbol's absolute position in the expression.

### The two corpora are the same drawings — do not merge them naively

detexify-next is a class-**balanced subsample** of the classic dump: **97.4% of its rows
have a shape-twin in the dump, yet zero match byte-for-byte** (normalized floats vs raw
pixels, so de-duplication does not catch them). Merging and splitting at random puts the
same drawing in train *and* val and silently inflates the held-out number — the very
metric this milestone gates on. `train.py` now splits by **shape group**, which makes
that impossible and costs no data. Details in `train/README.md`.

### Also this session

- **Ingest:** `train/detexify_sql_to_ndjson.py` (streams the `pg_dump` COPY block) →
  `--prepare-detexify`, which now **streams** (`-` = stdin, so a 1 GB dump never lands on
  disk) and takes `--classes` to **pin the label space**. Pinning makes datasets
  concatenable and keeps `model.labels.txt` stable — the new model's labels are
  byte-identical to the old one's, so it is a drop-in swap for anything deployed.
- Classic keys (`latex2e-OT1-_xi`) are normalized to canonical symbolIds
  (`latex:latex2e:xi`) in `detexify::normalize_class`, plus an 18-entry alias table for
  the punctuation the old key format can't express (`_&` → `ampersand`, `[` →
  `lbracket`). Result: **0 of 210,454 samples dropped.**
- **`\sqrt{}` emitted the literal `\sqrt-lbrace-rbrace`** — an empty brace argument
  collapses its dash, which the symbolId→LaTeX mapper didn't handle. A class with >1,000
  samples. Fixed + tested.
- **`--recognize` printed internal symbolIds, not LaTeX** — on both desktop and the
  device, whose docstring already claimed otherwise. It now prints `\sum  (latex:latex2e:sum)`.
- **Corpus regression suite: 1 → 13 cases.** One test case was guarding the entire
  classifier. `--export-corpus` mints fixtures from real Detexify handwriting; and
  `.expected.tex` now holds actual LaTeX (it held a raw symbolId), so the suite covers
  the symbolId→command mapper too — where `\sqrt{}` was broken.
- `make dataset` / `make train` / `make eval` now exist; the pipeline was only in a shell
  history before.

---

**M3 (2-D structure) built — the recognizer now emits real math, not just a list of
symbols.** `core::structure` turns positioned symbols into a Symbol Layout Tree via
geometry + class rules — baseline + super/subscripts, fractions (nested,
minus-disambiguated per §4.3), radicals (including over a fraction), and `\sum`/`\int`
limits — and `core::latex` emits the string (with a Detexify `symbolId`→command
mapper). `core::segment` now clusters strokes into symbols by **2-D proximity** (so a
fraction's stacked bar/numerator/denominator stay separate for structure to
reassemble; this replaced the M2 left-to-right split and serves both). The whole
pipeline `core::recognize_expression` (ink → segment → classify → structure → LaTeX),
exposed as `ink2tex-desktop --recognize-expr`, was verified end-to-end: a hand-drawn
fraction layout parses to `\frac{}{}`. **17 structure tests.** Deferred: the learned
relation MLP, robust segmentation of stacked-bar symbols (`=`/`≡`/`Ξ`), and an honest
CROHME exact-match number (needs the full-data model + more segmentation work).

**M1 recognizer works end-to-end on real data.** The full stack — Detexify strokes
→ rasterize → PyTorch train → int8 quantize → Rust int8 forward pass → labelled
top-5 — is built and validated (see "Current state" above for the numbers, now on the
full 210k corpus). Training is seeded + best-on-validation checkpointed, so a given
dataset reproduces its model. The architecture's accuracy levers, in the order they
landed: affine augmentation + dropout, then the **online channel** (DESIGN §3b) — the
free temporal signal a small 1-D conv reads off the resampled
`[dx, dy, pen_up, curvature]` pen trajectory (`core::classify::online`), fused with the
bitmap CNN at the fc1 layer — then the full corpus. *Next lever is capacity, not data.*
The exported `train/model.iwt` runs through the hand-rolled int8 kernel in Rust
(`--eval`) with the quantization intact. **And it runs ON THE DEVICE**: `crates/rm
--recognize` (`make recognize`) rasterizes captured ink → int8 CNN → top-5 LaTeX on
stdout (streamed over SSH, so **no rm2fb needed**), and the armv7 Cortex-A7 produced
the **bit-identical** top-5 to x86 — the quantized math is arch-consistent, at
**15.6 ms/symbol** (M1's `<50 ms` inference criterion, met) — **re-verified 2026-07-12 with
the full-corpus model and the fixed features**, still bit-identical to x86. A **live
draw-to-recognize** on the tablet worked end-to-end.
Remaining to ship M1: **package for Toltec/Vellum**. (An on-screen *result* display needs
the M4 typesetter; the stdout-over-SSH tool works today.)
The one lingering **M0** item is rm2fb for on-screen *inking* (recognition doesn't need it).

- **Last session:** 2026-07-12 — the full 210k Detexify corpus (see "Current state"), the
  `global_features` scale-invariance bug, the shape-group split, and the corpus suite
  1 → 13. The shipped model is retrained; `train/model.iwt` is the full-corpus one and its
  labels are byte-identical to the previous file.
- **✅ Deployed and verified on hardware (2026-07-12).** The full-corpus model + the fixed
  features run on the tablet: **mean 15.6 ms/symbol** (max 17.8, n=9 — M1's `<50 ms` met with
  3× headroom), and the armv7 top-5 is **bit-identical to x86** across 3 symbols,
  probabilities to the last decimal. So the int8 kernel is arch-consistent, and the
  preprocessing contract now holds on *both* sides of the wire. **Both M1 done-criteria are
  met.** SSH is key-based now (`~/.ssh/id_ed25519_rm`), so device targets run unattended.
- **✅ Packaged for Toltec (2026-07-12).** `make ipk` → `ink2tex_0.1.0-1_rm2.ipk` (332 KB),
  a well-formed opkg package; `packaging/toltec/ink2tex/package` is the recipe to submit.
  **Verified by installing it on the tablet** (payload unpacked at `/opt` exactly as opkg
  would): `ink2tex --recognize` with *no flags* found its own weights and answered in 17.7 ms;
  then removed, leaving the device as it was. `installdepends` is empty — it needs no rm2fb
  and no launcher, so it runs on a stock device with nothing but Toltec. The weights now
  ship with their ODbL attribution, which is an obligation, not a courtesy.
- **⛔ Next task — and it is YOUR decision, not a coding task: publish the source.** Toltec
  builds recipes in a clean Docker image from a **public** source URL checked against a
  sha256. This repo has no remote, so `source=` is a placeholder and `sha256sums=(SKIP)`.
  Push it, tag `v0.1.0`, fill in those two lines, open a PR against `toltec-dev/toltec`
  (`testing` branch). **That is the only thing left between here and M1 shipped.**
  See `packaging/README.md`.
- **After that**, the next accuracy lever is **model capacity, not data**: `fc1` is 64 units
  into a 1,123-way softmax, the train/val gap is down to 0.7 points, and ~34 ms of the 50 ms
  inference budget is unspent.
- **Blocked on:** publishing the repo (above). Nothing code-side.

<details><summary>Earlier sessions</summary>

- **2026-07-11** — full M0 build. Workspace (core/desktop/rm), Makefile,
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
- **M0 loose end:** to get live ink on the *screen*, install rm2fb on the device (Toltec
  `display` pkg), then `make ink` and draw — confirm ink appears under the pen and
  measure perceived latency (DEVICE FACTS row 7; ⚠ back up the device first, it stops
  xochitl).
- **Device facts verified:** rows 1,2,3,5 ✅; row 3 orientation ✅; row 4 ✅ (rm2fb NOT
  installed → needed for `--ink`); row 6 ✅ (no `usb_f_hid`); row 7 (latency) pending the
  on-screen `--ink` run. See `docs/device.md`.
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

</details>

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

> ✅ **Accuracy: met** — 96.8% top-5 (85.9% macro) through the int8 kernel, on a
> shape-group-held-out split of the full 210k corpus.
> ✅ **Latency: met** — 15.6 ms/symbol mean on the armv7 Cortex-A7 (2026-07-12), with the
> top-5 bit-identical to x86.
> ✅ **Packaged** — `make ipk`; installed and run on the tablet from `/opt`.
> **Every gate is green. `git push` is the last step, and it is a decision, not a task.**

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
