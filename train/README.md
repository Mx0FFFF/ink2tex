# train/ — offline training for the M1 symbol classifier

Python + PyTorch, **offline**. Produces a flat int8 `model.iwt` blob that the
device-free Rust core `mmap`s and runs (see `crates/core/src/classify`). Nothing
here ships to the device.

## The one rule: don't rasterize in Python

Training bitmaps are produced by `ink2tex-desktop --prepare-detexify`, which
rasterizes strokes through the **same** `crates/core` rasterizer the device uses at
inference time. If you rasterize in Python instead, the model trains on a slightly
different pixel distribution than it sees on-device and accuracy silently rots. So
the pipeline is: Rust rasterizes → Python trains on the pre-rasterized tensors.

## Data

Detexify's crowd-sourced stroke data (~342k samples, ~1000 classes) is **ODbL**.
Get the export JSON via the [detexify-data](https://github.com/kirel/detexify-data)
repo (a Google-Drive link + `example.json` showing the format). Attribute it, and —
per the project's non-negotiables — **do not train shipped weights on CROHME**
(license risk); use CROHME for evaluation only.

## Workflow

```bash
# 0. one-time
pip install torch numpy

# 1. Rust rasterizes the Detexify JSON into a training dataset (no skew).
cargo run -p ink2tex-desktop -- --prepare-detexify detexify.json --out-dir train/dataset
#    → train/dataset/{images.u8, features.f32, labels.u32, classes.txt, meta.json}

# 2. train the int8 CNN and export the weights blob (+ labels).
python train/train.py --data train/dataset --out train/model.iwt

# 3. sanity-check the blob with core's own parser.
cargo run -p ink2tex-desktop -- --dump-weights train/model.iwt

# 4. deploy model.iwt + model.labels.txt to the device (the classifier mmaps it).
```

## Files

- `iwt.py` — byte-exact mirror of `crates/core/src/classify/weights.rs`. The
  producer side of the `.iwt` contract; verified against core via `--dump-weights`.
- `train.py` — loads the dataset, trains the small CNN, does symmetric int8
  post-training quantization, and exports via `iwt.py`. The architecture and the
  quantization scheme documented at the top of `train.py` are exactly what the Rust
  forward pass implements.

## Model (M1 baseline)

`1×32×32 → conv(1→8,3×3)·relu·pool → conv(8→16,3×3)·relu·pool → flatten + 7 global
features → dense(→64)·relu → dense(→classes)`. Symmetric int8 (zero-point 0), i32
accumulation. Done-criterion: >90% top-5 on a held-out split, <50 ms on-device —
then package for Toltec/Vellum and release.
