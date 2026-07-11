# ink2tex — On-device handwritten math → LaTeX for e-ink tablets

> Draw an equation on your reMarkable. Get LaTeX. No cloud, no subscription, no network.

**Status:** design doc / project spec
**Target:** reMarkable 2 first, reMarkable Paper Pro second, other e-ink devices free
**Languages:** Rust core + hand-rolled inference kernels (drop to C/unsafe/NEON in hot loops)

---

## 1. The thesis

Every serious handwritten-math-expression-recognition (HMER) system available today falls into one of three buckets:

| System | Approach | Why it doesn't work here |
|---|---|---|
| **MyScript** (what reMarkable licenses) | Online, stroke-based, excellent | Closed, proprietary, behind a paid subscription |
| **Mathpix** | Cloud image OCR | Closed, paid, requires network |
| **pix2tex / LaTeX-OCR, BTTR, ICAL, TAMER, NAMER** | Image encoder + transformer decoder | GPU-scale. Hundreds of MB. Will not run on a Cortex-A7 |
| **Detexify** | Online, stroke-based, open, ODbL data | **Single symbols only.** No expressions, no structure |

Now here is the thing worth staring at.

CROHME — the standard benchmark — distributes **InkML files containing actual stroke trajectories**, with ground truth at both the symbol level (segmentation + labels) and the expression level (MathML/LaTeX structure).

And every modern research system opens those files and **immediately rasterizes the strokes into bitmap images**, discarding the temporal signal. BTTR: *"We transform the handwritten stroke trajectory information in the InkML files to offline images in bitmap format for training and testing."* ICAL, TAMER, and NAMER all say the same thing.

They do this because on a GPU, an image encoder + transformer decoder wins the benchmark.

**We don't have a GPU. We have a 1.2 GHz dual-core Cortex-A7 and 1 GB of RAM — and we have the strokes.**

Online recognition (using the pen trajectory) is *orders of magnitude* cheaper than offline image recognition, because the strokes are already segmented, already ordered, already noise-free. Detexify achieved useful symbol accuracy with **k-nearest-neighbours over histogram features** — no neural network at all.

So the inversion is:

> **The signal the research community throws away is precisely the signal that makes on-device recognition possible. And the reMarkable hands it to you for free, at 200+ Hz, with pressure and tilt.**

Nobody has built an **open, offline, on-device, stroke-based recognizer that handles structure** (fractions, exponents, radicals, limits). That's the gap. That's the project.

---

## 2. Non-goals

- **Not** trying to beat Mathpix or CROHME SOTA on exact-match accuracy. See §7.
- **Not** doing general handwriting → text (that's a different, easier problem and MyScript already ships on-device).
- **Not** doing image-based OCR of photographed math.
- **Not** binding to xochitl internals. We depend on the pen and the framebuffer, nothing else.

---

## 3. Architecture

The single most important decision in this document:

**The recognizer is a device-agnostic library. The reMarkable is a thin frontend.**

```
ink2tex/
├── core/          Rust lib. ZERO device dependencies. Runs & tests on your laptop.
│   ├── ink/        Stroke types, resampling, normalization, .ink file format
│   ├── segment/    Stroke → symbol-candidate grouping (the hard part)
│   ├── classify/   Hand-rolled quantized CNN inference (the low-level part)
│   ├── structure/  LOS graph → relation edges → spanning tree → layout tree
│   ├── latex/      Tree → LaTeX string
│   └── corpus/     Load/save labelled ink; the regression harness
│
├── train/         Python + PyTorch. Offline. Exports a flat weights blob.
│
├── desktop/       SDL/minifb dev harness. Draw with a mouse, SEE every pipeline
│                  stage rendered (groups, graph, tree). Debugger + contributor
│                  on-ramp + data collection tool. No tablet required.
│
├── rm/            libremarkable frontend. ~500 lines. The ONLY device-coupled code.
│
├── wasm/          core → wasm32. Free browser demo. Contributor magnet + data funnel.
│
└── tests/corpus/  Every bug report = an .ink file + expected LaTeX. CI runs them.
```

**Why this survives when the rmkit sample didn't:**

1. **`core` has no device deps.** `cargo test` runs the entire recognizer on any machine. Contributors don't need to own a reMarkable.
2. **When an OS update breaks rm2fb, exactly one file breaks** — `rm/`. The recognizer is untouched.
3. **Accuracy is a tracked number, not a vibe.** The corpus is checked in. Regressions are caught by CI.
4. **The core ports for free.** Paper Pro (aarch64), Supernote, Boox, and the browser all just need a new frontend. That multiplies the contributor pool by 10x.
5. **The WASM demo is the funnel.** People try it in a browser, contribute ink, file bugs with reproducible `.ink` attachments.

---

## 4. The pipeline

```
[Wacom digitizer: /dev/input/eventN]
  │  evdev: EV_ABS (ABS_X, ABS_Y, ABS_PRESSURE, ABS_TILT_X/Y)
  │         EV_KEY (BTN_TOOL_PEN, BTN_TOUCH), EV_SYN
  ▼
[1] INK CAPTURE ─────────► Stroke { points: [{x, y, pressure, tilt, t_us}] }
  │  resample to constant arc-length · smooth · normalize scale/position
  ▼
[2] SEGMENTATION ────────► candidate stroke groups (symbol hypotheses)
  │  temporal adjacency + spatial coherence + delayed-stroke reattachment
  ▼
[3] CLASSIFICATION ──────► each group → ranked top-k symbol classes + confidences
  │  tiny int8 CNN over a 32×32 render + online stroke features
  ▼
[4] STRUCTURE ───────────► Symbol Layout Tree (SLT)
  │  line-of-sight graph → relation classification → max spanning tree
  ▼
[5] LATEX EMIT ──────────► "\frac{x^2 + 1}{\sqrt{y}}"
  ▼
[6] CORRECTION UI ───────► tap-to-fix. Every fix is logged as training data.
  ▼
[7] EXPORT ──────────────► file · HTTP over usb0 · USB-HID (tablet types into your laptop)
```

### Stage 1 — Ink capture

Read the digitizer directly from `/dev/input/eventN`. **Do not hardcode the event number** — enumerate `/dev/input/event*` and probe with `EVIOCGBIT`/`EVIOCGNAME` ioctls to find the device advertising `ABS_PRESSURE` and `BTN_TOOL_PEN`. Device ordering is not stable across models.

Digitizer coordinate space is much larger than the screen (~20k × 15k vs 1872 × 1404 on rM2) and is **rotated relative to the display**. Get this transform right once, in one place.

> **Teaches:** Linux input subsystem, evdev protocol, `EV_SYN` event batching, ioctl-based capability probing, coordinate transforms, monotonic timestamps.

### Stage 2 — Segmentation (the genuinely hard part)

Group strokes into symbols. Most symbols are 1–3 strokes (`x` is 2, `=` is 2, `≡` is 3), so cap group size at ~4.

Baseline heuristic: strokes that are **temporally adjacent** AND **spatially coherent** (bbox overlap, or minimum inter-stroke distance below a threshold scaled by stroke size) belong together.

**The killer problem is delayed strokes.** People dot the `i` later. Cross the `t` later. Draw the fraction bar *after* the numerator and denominator. Add the bar over `\bar{x}` at the end. Naive temporal grouping falls apart.

The fix is to stop treating segmentation as a separate stage:

> **Segmentation, classification, and parsing must be jointly optimized.** Generate a *lattice* of plausible groupings, score each hypothesis by (classifier confidence + structural plausibility), and beam-search for the best consistent partition. The classifier is the oracle that scores segmentation hypotheses.

Concretely: enumerate contiguous-in-time stroke subsets of size ≤ 4 that are spatially coherent, classify each, then beam-search over consistent partitions using the structure parser as a plausibility check. Add a reattachment pass that tries gluing small isolated strokes to nearby groups and keeps the change if classifier confidence improves.

For M2 (linear expressions), greedy temporal+spatial grouping gets you 80% of the way. Don't build the lattice until M3.

> **Teaches:** search algorithms, beam search, dynamic programming, hypothesis scoring, the difference between a pipeline and a jointly-optimized system.

### Stage 3 — Classification

**Input representation — two channels, and the second one is the whole point:**

- **(a) Offline channel:** render the stroke group to a 32×32 anti-aliased bitmap, aspect-preserving, centred.
- **(b) Online channel:** resample the stroke to a fixed 64 points, encode as `[dx, dy, pen_up, curvature]`. **This is the free information the research systems discard.**

**Model:** small CNN on (a) + small 1D conv on (b) → concat → dense → softmax over ~120 classes (CROHME's 111 + extras).

For M1, ship just the CNN over (a) plus a handful of global stroke features (stroke count, aspect ratio, arc length, start/end position). Add (b) in M2.

**Inference: hand-roll it. Do not link TFLite or ONNX Runtime.**

Write `conv2d`, `relu`, `maxpool`, `dense`, `softmax` yourself. For a model this small that's <400 lines of Rust. Quantize to int8. Export weights as a flat little-endian blob with a small header, `mmap` it at startup.

This is not masochism — it's the right engineering call *and* the best learning in the project:
- Zero heavy dependencies → a self-contained ~2 MB static binary
- No cross-compilation nightmare for armv7/aarch64
- You control the memory layout, so you can make it cache-friendly
- You can write a NEON path for the convolution inner loop and *measure the speedup*

Check `/proc/cpuinfo` on your device to confirm NEON is present. If it isn't, scalar int8 is still fast enough — this model is tiny.

> **Teaches:** quantization and fixed-point arithmetic, memory layout and cache behaviour, `mmap`, SIMD/NEON intrinsics, im2col vs. direct convolution, why a 500 KB model beats a 100 MB one when you have 1 GB of RAM.

**The size trap you will hit:** in isolation, `.` vs `\cdot` vs `\bullet` are indistinguishable. So are `-` vs `\_` vs a fraction bar, `,` vs `'`, and `x` vs `\times`. **Do not try to solve this in the classifier.** Pass relative-size and baseline-position features forward and let the *structure* stage disambiguate: a horizontal bar with content above **and** below is a fraction bar; the same bar with symbols left and right on the same baseline is a minus sign. This is exactly why the pipeline has to be jointly optimized.

### Stage 4 — Structure

Turn a bag of positioned symbols into a tree. This is a **2D grammar parsing problem** and it's the most intellectually satisfying part of the project.

1. **Nodes** = classified symbols with bounding boxes.
2. **Line-of-sight graph:** connect two symbols if a ray between them isn't fully occluded by a third.
3. **Relation classification** on each edge → `{Right, Superscript, Subscript, Above, Below, Inside, None}`.
   Features: relative centroid offset, size ratio, bbox overlap, estimated baseline, **and the classes of both endpoints**.
   - **v1: pure geometry + class-aware rules.** `Inside` is only legal if the parent is `\sqrt`. `Above`/`Below` only for fraction bars, `\sum`, `\int`, `\lim`, `\bar`. This gets you surprisingly far with zero ML.
   - **v2: a tiny learned MLP** over the feature vector. Train it on CROHME's **SymLG (symbol-level label graph)** ground truth from the ICDAR 2023 release — which is *literally the label-graph format this algorithm consumes*.
4. **Extract a maximum spanning tree** over the weighted relation graph (Edmonds' algorithm for the directed case) → the **Symbol Layout Tree**.

### Stage 5 — LaTeX emission

Recursive tree walk. Handle `\frac{}{}`, `^{}`, `_{}`, `\sqrt[]{}`, `\sum_{}^{}`, `\int_{}^{}`, `\begin{matrix}`. Trivial once the tree is right. Emit MathML too — it's nearly free and doubles the utility.

### Stage 6 — Correction UI (**this is the actual product**)

Read §7 before you argue with this.

- Recognized expression renders as **typeset math** next to your ink.
- **Tap a symbol** → popup with the classifier's top-5 alternatives → tap to fix.
- **Long-press between two symbols** → force a relation ("make this a superscript").
- **Lasso strokes** → force them into one symbol (fixes segmentation).
- **Every correction appends `(ink, correct_label)` to a local corpus file.**
- Opt-in: "contribute my corrections" → grows an open ODbL ink corpus → retrain → ship better weights.

**The correction log is the dataset.** That's the flywheel, and it's how the project gets better after v1 instead of rotting.

**Sub-project: the math typesetter.** To render typeset math on-device you need a small math layout engine. Implementing the subset of TeX's box-and-glue / mlist-to-hlist rules you actually need, with bundled Computer Modern glyphs, is a self-contained and genuinely fun module — and *a small, dependency-free, embeddable LaTeX-math-to-bitmap renderer in Rust is a valuable open-source library in its own right.* Nobody has one for e-ink. (v1: just show the LaTeX source string. That's honest and useful. Build the renderer at M4.)

### Stage 7 — Export (the feature that drives adoption)

- **v1:** write a `.tex` file to the filesystem. Pull it over SSH.
- **v2:** a tiny HTTP server bound to `usb0` (the reMarkable already exposes `10.11.99.1`). `GET /latest.tex` from your laptop. Raw TCP sockets — good practice, ~100 lines.
- **v3 — the killer feature: USB HID gadget.** The reMarkable **already runs a USB gadget** (that's how it presents as a network device). Add an HID keyboard function via configfs, write HID reports to `/dev/hidg0`, and **the tablet types the LaTeX directly into whatever has focus on your laptop.**

  > *Write math on your reMarkable → it appears in your Overleaf tab.*

  **Verify `usb_f_hid` is available in the stock kernel before committing to this.** If it isn't, you're building a kernel module — which is a great advanced exercise, and the community already does this (ghostwriter had to deal with `uinput` missing from the Paper Pro kernel). **Fallback:** the HTTP endpoint plus a 30-line host-side clipboard daemon. Ship the fallback first.

> **Teaches:** USB gadget subsystem, configfs, HID report descriptors, character devices, BSD sockets, kernel module building.

---

## 5. Training data

| Source | What it gives you | Format | License |
|---|---|---|---|
| **[Detexify data](https://github.com/kirel/detexify-data)** | ~1000 LaTeX symbol classes, crowd-sourced, **stored as full strokes** | stroke JSON | **ODbL** (attribution + share-alike on the database) |
| **[CROHME](https://www.iapr-tc11.org/mediawiki/index.php/CROHME:_Competition_on_Recognition_of_Online_Handwritten_Mathematical_Expressions)** / [ICDAR 2023 on Zenodo](https://zenodo.org/records/8428035) | 8,836 expressions, 111 symbol classes, **symbol-level AND expression-level ground truth**, plus SymLG label graphs | InkML | ⚠️ **check carefully** — see below |
| **Your own corrections** | Real ink from real reMarkable users, on the actual digitizer | `.ink` | yours to license |

### ⚠️ The license trap — read this before you train anything you ship

A widely-mirrored derived CROHME dataset is distributed under **CC BY-NC-SA** (non-commercial, share-alike). Terms vary across CROHME editions and mirrors.

**Recommended strategy to keep the license story clean:**

- **Train the shipped weights on Detexify (ODbL) + your own collected corpus.** ODbL is workable: attribute, and share-alike if you redistribute the *database*. A trained model is arguably a "Produced Work," which just needs attribution — but read the license yourself, don't take my word for it.
- **Use CROHME for evaluation and research only.** Report your CROHME 2014/2016/2019 numbers for credibility, but don't bake NC-licensed data into the binary you ship to users.
- This also means: **your own collected corpus is strategically valuable.** An open, permissively-licensed, modern corpus of online handwritten math ink — captured on a real high-res digitizer — doesn't currently exist. Building one is a genuine contribution to the field, independent of the app.

---

## 6. Milestones

Each one has a hard done-criterion. Don't move on until you hit it.

### M0 — Ink recorder *(a weekend)*
Rust + libremarkable. Read the digitizer, render strokes to the framebuffer with partial refresh (use the fast `DU` waveform for low-latency inking), save `.ink` files.

**Done when:** you can write on the screen with sub-50 ms perceived latency and dump strokes to a file.
**Teaches:** evdev, ioctl capability probing, rm2fb, E-Ink waveform modes, cross-compiling to `armv7-unknown-linux-gnueabihf`.

### M1 — Offline Detexify *(2–4 weeks)* — **★ SHIP THIS ★**
Train a symbol classifier on Detexify's ODbL stroke data. Hand-roll int8 inference in Rust. Draw a symbol → top-5 LaTeX commands → tap to copy.

**Done when:** >90% top-5 accuracy on a held-out split, <50 ms inference on-device.

**This is already a real, useful, non-duplicate application.** Package it for Toltec/Vellum and release it. Users from month one is exactly what breaks the "unmaintained experimental sample" curse — you'll have people filing bugs and sending you ink before you've written a single line of the structure parser.

### M2 — Linear expressions *(3–6 weeks)*
Greedy segmentation + left-to-right ordering. `2x + 3 = 7`, `f(x) = ax + b`. No 2D structure yet.

**Done when:** >85% exact-match on a 100-expression handwritten linear corpus you collected yourself.

### M3 — Structure *(6–12 weeks)* — **the heart of it**
LOS graph → relation classification → maximum spanning tree → SLT → LaTeX. Superscripts, subscripts, fractions, radicals, `\sum`/`\int` with limits.

**Done when:** you can report an honest exact-match number on CROHME 2014. Even 40–50% from a hand-rolled on-device system is a respectable result — and §7 explains why that's a *shipping product*, not a failure.

### M4 — Correction UI + typesetting + export *(3–4 weeks)*
Tap-to-fix, correction logging, math renderer, `.tex` out, HTTP on `usb0`.

**Done when:** the median expression needs ≤2 corrections and lands in your Overleaf tab.

### M5 — Flywheel and reach
USB-HID typing. WASM browser demo. Opt-in corpus contribution → retrain → ship better weights. Paper Pro (aarch64) port — which, given the architecture in §3, should be mostly a frontend swap.

---

## 7. The accuracy reframe — read this or you will quit at M3

**HMER is a research-grade problem.** Full-expression exact-match accuracy is well below 100% *even for state-of-the-art image transformers running on GPUs*. If you benchmark yourself against Mathpix and expect to win, you will conclude the project failed and abandon it.

**That framing is wrong, and the reason is the hardware you're on.**

You are not building a batch OCR system that has to be right the first time on a scanned page. You are building an **interactive tool on a device with a pen in the user's hand.** Correction is nearly free: tap a symbol, pick from the top-5 alternatives, done.

> **A 70%-accurate model with a 2-tap correction UI is a 99%-accurate *workflow*.** The correction UI isn't a fallback for when the model fails — **the correction UI is the product**, and the model is the thing that makes it fast.

And it compounds: every correction is a labelled training example on the real digitizer. The model gets better precisely where real users actually struggle.

Design for correction from day one. Get top-k out of the classifier from M1. Never build a pipeline stage that can't expose its alternatives to the UI.

---

## 8. Known risks and open questions

| Risk | Mitigation |
|---|---|
| **CROHME licensing may be NC** | Train shipped weights on Detexify (ODbL) + own corpus. Use CROHME for eval only. |
| **`usb_f_hid` may not be in the stock kernel** | Verify early. Ship the HTTP-endpoint fallback first; treat HID as a stretch. |
| **NEON availability on i.MX7** | Check `/proc/cpuinfo`. Scalar int8 fallback is fast enough regardless. |
| **rm2fb breaks on OS updates** | Don't reimplement it — use libremarkable, which bundles the client. Keep device code in one file. |
| **Delayed strokes wreck naive segmentation** | Don't build the naive version and hope. Plan for the hypothesis lattice from the start (§4.2). |
| **Paper Pro is aarch64 + secure boot + qtfb** | Don't start there. Port at M5, once the core is proven and device-agnostic. |
| **Scope creep into a research project** | The milestone gates exist for this reason. M1 ships a real app in a month. Guard that. |

---

## 9. Prior art to read (and credit)

- **Detexify** — Daniel Kirsch. [Detexify explained](https://gist.github.com/kirel/149896) is a genuinely great short read on the kNN + histogram-feature approach, written by someone who describes himself as having stumbled into pattern recognition. Start here.
- **Zanibbi & Blostein**, *Recognition and retrieval of mathematical expressions* — the survey. The line-of-sight graph / label-graph / spanning-tree family of methods comes from this group.
- **CROHME competition papers** (Mouchère et al.) — for the evaluation methodology and the SymLG format.
- **BTTR / CoMER / ICAL / TAMER / NAMER** — read these to understand what the image-based SOTA does, and *why you are deliberately not doing that.*
- **libremarkable** — your device layer. Bundles the rm2fb client and handles the Wacom digitizer.
- **rmkit** — the existing experimental math sample lives here. Read it, credit it, supersede it.

---

## 10. Naming

`ink2tex` is a placeholder; it's descriptive and greppable. Alternatives if you want more character: **Sigil**, **Quill**, **Marginalia**, **Chalk**.

---

## 11. First commit

```bash
# 1. Confirm what you're actually running on
ssh root@10.11.99.1
cat /proc/cpuinfo          # NEON? cores? confirm the SoC
ls /dev/input/             # find the digitizer
cat /proc/bus/input/devices

# 2. Watch raw pen events. Feel the data before you write any code.
evtest /dev/input/eventN

# 3. Set up the toolchain
rustup target add armv7-unknown-linux-gnueabihf

# 4. M0: read the digitizer, draw the stroke, save the file.
```

Start with M0. It's a weekend. And the moment you see your own ink appear on that screen from code you wrote — reading raw evdev events, transforming coordinates, driving an E-Ink partial refresh — the rest of this document stops being a plan and starts being a project.
