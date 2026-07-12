# ink2tex

**Draw a math symbol on your reMarkable. Get the LaTeX command. Offline.**

```
$ ink2tex --recognize          # draw ∑ with the pen
  1.  64.9%  \sum             (latex:latex2e:sum)
  2.  33.9%  \Sigma           (latex:latex2e:Sigma)
  3.   1.2%  \Upsigma         (latex:upgreek:Upsigma)
inference: 15.6 ms
```

No cloud. No network. No account. The recognizer runs on the tablet's own 2011-era
Cortex-A7, in ~16 milliseconds, from a 164 KB model.

| | |
|---|---|
| **Accuracy** | **96.8%** top-5 (85.9% macro) on a held-out split of 210k real drawings |
| **Speed** | **15.6 ms**/symbol on-device (mean, n=9) |
| **Size** | 450 KB binary + 164 KB weights |
| **Dependencies** | **no ML runtime.** No ONNX, no TFLite, no torch. |

## Install

Via [Toltec](https://toltec-dev.org):

```bash
opkg install ink2tex
ink2tex --recognize
```

It needs no launcher and no rm2fb — it reads the pen through evdev and prints to stdout,
so it runs on a stock device with nothing but Toltec.

## Why it's built this way

The int8 CNN is **hand-written** — `conv2d`, `maxpool`, `dense`, `softmax`, quantization,
all of it, in ~400 lines of dependency-free Rust ([`crates/core/src/classify/`](crates/core/src/classify/)).
That is not stubbornness. A 1 GB armv7 appliance is precisely where a general ML runtime is
the wrong tool: it costs tens of megabytes, drags in a cross-compilation nightmare, and buys
nothing a fixed 5-layer network needs. The whole model is a flat little-endian blob that is
`mmap`'d and multiplied.

The classifier reads two channels: the **bitmap** (a 32×32 rasterization) and the **online
signal** — the pen's actual trajectory, resampled to `[dx, dy, pen_up, curvature]`. That
second one is free information a photo of your handwriting simply doesn't have, and it is
worth about a point and a half of top-5.

**Top-5, never top-1.** Every stage exposes ranked alternatives, because for handwritten
math the correction UI *is* the product — a model that is right 70% of the time and wrong
usefully beats one that is right 75% of the time and wrong opaquely.

## Status

Single symbols work, on real hardware, today. Expressions (`\frac`, super/subscripts,
radicals, `\sum` limits) parse in `crates/core/src/structure.rs` but do not have a UI yet.
See [`ROADMAP.md`](ROADMAP.md) for what is real and what is not — it is kept honest.

## Build from source

```bash
cargo test --workspace        # 87 tests, no device needed
make build-rm                 # cross-compile → armv7
make ipk                      # → target/ipk/ink2tex_0.1.0-1_rm2.ipk
make deploy-model && make recognize
```

Retraining, including the Detexify ingest, is documented in [`train/README.md`](train/README.md).
`crates/core` has zero device dependencies and is fully testable on a laptop — that split is
enforced in CI, and it is why an OS update that breaks the framebuffer breaks one file rather
than the project.

## Licence and attribution

Code: **MIT OR Apache-2.0**.

The shipped weights are trained on the [Detexify](https://github.com/kirel/detexify-data)
dataset — 210,454 drawings contributed by its users — which is **ODbL**. See
[`ATTRIBUTION.md`](ATTRIBUTION.md); it ships inside the package too, because it has to.
CROHME is used for evaluation only and is never trained on.
