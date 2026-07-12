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

Detexify's crowd-sourced stroke data is **ODbL**. Attribute it, and — per the project's
non-negotiables — **do not train shipped weights on CROHME** (license risk); CROHME is
for evaluation only.

It ships in two exports, and they are the *same drawings* in different clothes:

| export | shape | samples | coords | class key |
|---|---|---|---|---|
| **classic bulk dump** (`detexify.sql.gz`, a `pg_dump`) | `COPY` block, `[x,y,t]` points | **210,454** | raw pixels | `latex2e-OT1-_xi` |
| detexify-next | JSON, `{x,y}` points | 39,554 | normalized 0–1 | `latex:latex2e:xi` |

**Use the bulk dump.** detexify-next is a class-*balanced subsample* of it — 97.4% of its
rows have a shape-twin in the dump — so it is 5.3× smaller and no more diverse. The two
vocabularies are reconciled by `detexify::normalize_class` (plus an 18-entry alias table
for the punctuation the old key format couldn't express), which lands **every** dump
sample in the same 1,123-class space.

⚠️ Because the corpora overlap, do **not** merge them and split at random: the same
drawing lands in both train and val and the held-out score inflates. `train.py` splits by
*shape group*, which makes that impossible — see `shape_groups()`.

The corpus is also heavily **imbalanced** (`\int`: 3,937 samples; median class: 53; 159
classes under ten). That is real usage frequency, not a defect — but it means micro
accuracy alone flatters the model, so training reports macro (per-class) accuracy too.

## Workflow

```bash
# 0. one-time
pip install torch numpy

# 1. the pg_dump → NDJSON (streams; `-` pipes straight into Rust, no 1 GB temp file).
python3 train/detexify_sql_to_ndjson.py ~/Downloads/detexify.sql.gz \
    -o train/detexify_raw/detexify.ndjson

# 2. Rust rasterizes it into a training dataset — the same rasterizer inference uses,
#    so there is no train/infer skew. --classes pins the label space.
cargo run --release -p ink2tex-desktop -- \
    --prepare-detexify train/detexify_raw/detexify.ndjson \
    --out-dir train/dataset_full --classes train/model.labels.txt
#    → {images.u8, features.f32, online.f32, labels.u32, classes.txt, meta.json}

# 3. train the int8 CNN, export the weights blob (+ labels), and keep the held-out split.
python3 train/train.py --data train/dataset_full --out train/model.iwt \
    --dump-val train/dataset_val

# 4. check the blob with core's own parser, and score it through the *int8* kernel —
#    the number that matters is the one the device will actually produce.
cargo run --release -p ink2tex-desktop -- --dump-weights train/model.iwt
cargo run --release -p ink2tex-desktop -- --eval train/dataset_val --model train/model.iwt

# 5. deploy model.iwt + model.labels.txt to the device (the classifier mmaps it).
```

Because `--classes` pins the label space, datasets are **concatenable**: `--data dirA dirB`
just works, and `model.labels.txt` stays stable across retrains.

## Files

- `detexify_sql_to_ndjson.py` — the bulk dump's `COPY` block → NDJSON. Dumb transport;
  it does not interpret class keys (that lives in Rust, unit-tested).
- `iwt.py` — byte-exact mirror of `crates/core/src/classify/weights.rs`. The
  producer side of the `.iwt` contract; verified against core via `--dump-weights`.
- `train.py` — loads the dataset(s), splits by shape group, trains the small CNN, does
  symmetric int8 post-training quantization, and exports via `iwt.py`. The architecture
  and the quantization scheme documented at the top of `train.py` are exactly what the
  Rust forward pass implements.

## Model (M1 baseline)

`1×32×32 → conv(1→8,3×3)·relu·pool → conv(8→16,3×3)·relu·pool → flatten + 7 global
features → dense(→64)·relu → dense(→classes)`. Symmetric int8 (zero-point 0), i32
accumulation. Done-criterion: >90% top-5 on a held-out split, <50 ms on-device —
then package for Toltec/Vellum and release.
