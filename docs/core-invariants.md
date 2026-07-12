# crates/core — purity rules

This crate is the entire value of the project. Everything else is a frontend.

## It must stay device-free

No `libremarkable`. No `/dev/input`. No framebuffer. No `evdev`. No device paths.
It must build and test on a laptop with `cargo test -p ink2tex-core`.

**CI enforces this** (`cargo tree -p ink2tex-core` is checked against a banned list). If you find yourself wanting a device type in here, the abstraction is wrong: define a plain data type in core and convert at the `crates/rm` boundary.

`Stroke` is `{ points: Vec<Point { x, y, pressure, tilt_x, tilt_y, t_us }> }` in **normalized** coordinates. The digitizer→screen transform lives in `crates/rm`, not here.

## No ML runtime dependencies. Ever.

Banned: `tract*`, `ort`, `candle-*`, `tflite`, `burn`, `torch-sys`, and anything similar.

`classify/` implements `conv2d`, `relu`, `maxpool`, `dense`, `softmax` **by hand**, int8-quantized. This is deliberate. It is the richest learning surface in the project and it is also the correct engineering choice for a 1 GB armv7 device. CI will fail the build if you add one.

## Stage contract

Every stage is `&Input -> Result<Output>`, pure, and independently testable.
Every stage that ranks anything **must expose top-k**, not just the argmax. The correction UI depends on it.

## The preprocessing contract: every classifier input is dimensionless

`rasterize`, `online_features` and `global_features` run in two places — offline to build training tensors, and on the device to featurize live ink. Ink reaches them in whatever units its source uses: Detexify's bulk dump is in **screen pixels**, detexify-next in **0–1 floats**, `crates/rm` in **normalized device coords**. So every value handed to the model must depend on the *shape alone*: relative to the ink's own bounding box, and bounded.

This is not theoretical. `global_features` emitted a raw arc length and raw start/end points, which went unnoticed only because the training corpus happened to be normalized like the device already. Score that model on the same symbols in pixel units and it collapses to **7.9% top-5** where it otherwise gets 90%+ — and a model trained the other way round would have failed exactly as hard *on the tablet, in the user's hands*. The bitmap and online channels were fine throughout, because their invariance is unit-tested. That vector's never was.

Keep each feature **O(1)** too: all seven share a single int8 scale with the 1,384 CNN/online activations they're concatenated to, so one wild value (the old unbounded `w/h` aspect reached ~8e5 on a minus sign) quantizes everything else to zero.

**Add a feature → add its invariance test in the same commit.**

## The two algorithmic traps

**Delayed strokes.** People dot the `i` later, cross the `t` later, draw the fraction bar *after* the numerator and denominator. Naive temporal grouping will break on real ink. Don't build the naive version and hope — the design calls for a hypothesis lattice scored by classifier confidence + structural plausibility (DESIGN.md §4.2).

**Size-ambiguous symbols.** `-` vs fraction bar vs `\_`; `.` vs `\cdot` vs `\bullet`; `x` vs `\times`; `,` vs `'`. These are *unresolvable* by the classifier in isolation. Do not attempt to fix this in `classify/`. Pass size and baseline-position features forward and let `structure/` disambiguate: a horizontal bar with content above **and** below is a fraction bar; the same bar with symbols left and right on one baseline is a minus sign.

If you catch yourself adding classes to the classifier to fix an ambiguity, stop — it belongs in structure.

## Testing

`tests/corpus/*.ink` + `*.expected.tex` is the regression suite and the project's immune system. Every bug fix adds a case. Accuracy is a tracked number, never a vibe.
