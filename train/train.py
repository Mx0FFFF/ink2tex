#!/usr/bin/env python3
"""Train the M1 symbol classifier and export an int8 `.iwt` blob for on-device
inference.

    pip install torch numpy
    python3 train/train.py --data train/dataset --out train/model.iwt

The dataset comes from `ink2tex-desktop --prepare-detexify` — which rasterizes
through the SAME Rust rasterizer the device uses. Do NOT rasterize in Python: that
reintroduces the train/inference skew this whole split exists to prevent.

Pipeline: load dataset → train a small float CNN (mini-batched) → symmetric int8
post-training quantization (PTQ) → write `model.iwt` (via iwt.py) + `.labels.txt`.

The export tensor names and quantization scheme are the exact contract the Rust
forward pass (`crates/core/src/classify/model.rs::recognize`) implements. For each
layer L we export `L.w` (int8, with its weight scale), `L.b` (int32 bias at the
accumulator scale `in_scale·sw`), and `L.in_scale` (the activation scale feeding L,
from calibration). Rust derives the requantize multipliers `in_scale[L]·sw[L] /
in_scale[L+1]` itself, and dequantizes the final layer with `in_scale·sw`.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iwt  # noqa: E402  (local module, byte-exact mirror of weights.rs)

# --- architecture (must match crates/core/src/classify/model.rs) -------------
# 1×32×32 → conv1(1→8,3×3,pad1)+relu+pool2 → 8×16×16
#         → conv2(8→16,3×3,pad1)+relu+pool2 → 16×8×8 → flatten(1024)
#   online 4×64 → conv1d(4→12,k5,stride2)+relu → flatten(360)   (DESIGN §3b)
#         → concat [cnn 1024, online 360, 7 global feats] → fc1(→64)+relu → fc2
CONV1, CONV2, FC1, KSIZE = 8, 16, 64, 3
# Online branch — MUST match core::classify::online (ONLINE_CHANNELS / ONLINE_POINTS
# / ONLINE_STRIDE) and model.rs (which reads OC/KW from the tensor, stride fixed).
ONLINE_CH, ONLINE_LEN = 4, 64
ONLINE_OC, ONLINE_KW, ONLINE_STRIDE = 12, 5, 2


def load_one(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    n, size, nf = meta["n"], meta["size"], meta["num_features"]
    X = np.fromfile(os.path.join(d, "images.u8"), np.uint8).reshape(n, 1, size, size)
    F = np.fromfile(os.path.join(d, "features.f32"), "<f4").reshape(n, nf)
    y = np.fromfile(os.path.join(d, "labels.u32"), "<u4").astype(np.int64)
    classes = [c for c in open(os.path.join(d, "classes.txt")).read().split("\n") if c]
    # Online trajectory channel (DESIGN §3b) — present iff prepare emitted online.f32.
    O = None
    if meta.get("online_len"):
        O = np.fromfile(os.path.join(d, "online.f32"), "<f4").reshape(n, ONLINE_CH, ONLINE_LEN)
    return X, O, F, y, classes, size, nf


def load_dataset(dirs, max_samples=0):
    """Load one or more dataset dirs and concatenate them.

    Concatenating is only meaningful because `--prepare-detexify --classes` pins the
    label space: every dir indexes the *same* class list, so label ids line up. Loading
    dirs prepared against different vocabularies would silently mislabel everything, so
    that is checked, not assumed.
    """
    parts = [load_one(d) for d in dirs]
    classes, size, nf = parts[0][4], parts[0][5], parts[0][6]
    for d, p in zip(dirs[1:], parts[1:]):
        if p[4] != classes or p[5] != size or p[6] != nf:
            sys.exit(f"error: {d} was prepared against a different label space/geometry — "
                     f"re-run --prepare-detexify --classes with the same class list")
    X = np.concatenate([p[0] for p in parts])
    F = np.concatenate([p[2] for p in parts])
    y = np.concatenate([p[3] for p in parts])
    O = None if any(p[1] is None for p in parts) else np.concatenate([p[1] for p in parts])
    if len(dirs) > 1:
        print("  " + " + ".join(f"{len(p[3])} from {d}" for d, p in zip(dirs, parts)), flush=True)
    if max_samples and len(y) > max_samples:
        sel = np.random.default_rng(0).permutation(len(y))[:max_samples]
        X, F, y = X[sel], F[sel], y[sel]
        O = O[sel] if O is not None else None
    return X, O, F, y, classes, size, nf


def shape_groups(X, y):
    """Group id per row: (binarized bitmap, label). Rows that look identical to the
    model — the same drawing — share a group.

    The bulk corpus contains the same drawing more than once: people resubmit, and the
    detexify-next corpus turned out to be a re-encoding of the very same drawings (97.4%
    of it has a shape-twin in the Postgres dump, yet *zero* rows match byte-for-byte,
    because one ships normalized floats and the other raw pixels). A plain random row
    split scatters those twins across train and val, and the held-out score — the M1
    gate — inflates for free. Splitting whole groups makes that impossible, and unlike
    de-duplication it throws no data away.
    """
    packed = np.packbits(X.reshape(len(y), -1) > 0, axis=1)          # 1 bit/pixel
    key = np.concatenate([packed, y.astype("<u4").view(np.uint8).reshape(-1, 4)], axis=1)
    _, gid = np.unique(key, axis=0, return_inverse=True)
    return gid.ravel()


def build_model(nf, n_classes, size):
    import torch
    import torch.nn as nn

    flat = CONV2 * (size // 4) * (size // 4)
    online_w = (ONLINE_LEN - ONLINE_KW) // ONLINE_STRIDE + 1
    online_flat = ONLINE_OC * online_w

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, CONV1, KSIZE, padding=1)
            self.c2 = nn.Conv2d(CONV1, CONV2, KSIZE, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.o1 = nn.Conv1d(ONLINE_CH, ONLINE_OC, ONLINE_KW, stride=ONLINE_STRIDE)
            self.f1 = nn.Linear(flat + online_flat + nf, FC1)
            self.f2 = nn.Linear(FC1, n_classes)
            self.drop = nn.Dropout(0.3) # the long tail is still thin: 159 classes have <10 samples

        def forward(self, x, online, feats):
            x = self.pool(torch.relu(self.c1(x)))
            x = self.pool(torch.relu(self.c2(x)))
            o = torch.relu(self.o1(online)).flatten(1) # online 1-D conv branch
            # Concat order MUST match model.rs: [cnn flatten, online flatten, globals].
            x = torch.cat([x.flatten(1), o, feats], dim=1)
            x = self.drop(torch.relu(self.f1(x)))
            return self.f2(x)

    return Net()


def augment(x):
    """Random affine (rotation, scale, translation) on a batch of 1×H×W bitmaps —
    cheap synthetic variety that makes the classifier robust to how a symbol is drawn —
    the same glyph rotated, bigger, off-centre. Train-time ONLY (inference sees the clean
    rasterization). It was worth +3 points top-5 back when the corpus was a 39.5k
    subsample; it still earns its keep on the tail classes, which have a handful of
    samples each no matter how big the corpus gets."""
    import torch

    b = x.shape[0]
    ang = (torch.rand(b) - 0.5) * 0.4 # ±0.2 rad
    scl = 1.0 + (torch.rand(b) - 0.5) * 0.3 # ±15%
    tx = (torch.rand(b) - 0.5) * 0.2
    ty = (torch.rand(b) - 0.5) * 0.2
    cos, sin = torch.cos(ang) * scl, torch.sin(ang) * scl
    theta = torch.zeros(b, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, align_corners=False)


def _batch(X, O, F, y, idx):
    """Materialize one mini-batch as tensors (uint8 → float32/255 on the fly, so the
    full 210k×32×32 corpus never lives in memory as float)."""
    import torch

    xb = torch.from_numpy(X[idx].astype(np.float32) / 255.0)
    ob = (
        torch.from_numpy(O[idx])
        if O is not None
        else torch.zeros(len(idx), ONLINE_CH, ONLINE_LEN)
    )
    fb = torch.from_numpy(F[idx])
    yb = torch.from_numpy(y[idx])
    return xb, ob, fb, yb


def evaluate(model, X, O, F, y, idx, bs, n_classes=0):
    """Top-5 (micro), and — if asked — the *macro* average over classes.

    The raw corpus is wildly imbalanced: `\\int` has 3,937 samples, the median class 53,
    and 159 classes fewer than ten. Micro accuracy is dominated by the head, so it can
    look excellent while the tail is unusable. Macro (mean per-class recall) is the
    number that notices. Report both; trust neither alone.
    """
    import torch

    model.eval()
    hit = np.zeros(len(idx), bool)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb, ob, fb, yb = _batch(X, O, F, y, idx[i : i + bs])
            logits = model(xb, ob, fb)
            k = min(5, logits.shape[1])
            top5 = logits.topk(k, dim=1).indices
            hit[i : i + len(yb)] = (top5 == yb[:, None]).any(1).numpy()
    micro = hit.mean() if len(idx) else float("nan")
    if not n_classes:
        return micro
    yv = y[idx]
    per = [hit[yv == c].mean() for c in range(n_classes) if (yv == c).any()]
    return micro, float(np.mean(per)), len(per)


def split_by_group(gid, val_frac, seed=0):
    """Hold out whole shape-groups (see `shape_groups`), never individual rows."""
    groups = np.unique(gid)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_frac))
    is_val = np.isin(gid, groups[:n_val])
    return np.where(~is_val)[0], np.where(is_val)[0]


def train(model, X, O, F, y, gid, epochs, lr, val_frac, bs, n_classes, subsample=0):
    import torch

    tr, va = split_by_group(gid, val_frac)
    dupes = len(y) - len(np.unique(gid))
    print(
        f"split by shape-group: {len(tr)} train / {len(va)} val "
        f"({len(np.unique(gid))} groups, {dupes} rows share a group with another)",
        flush=True,
    )
    if subsample and subsample < len(tr):
        # Shrink the TRAIN side only, and only *after* the split, so a small-data run and
        # a full-data run are scored on the identical held-out rows. Subsampling before
        # the split would move the val set too, and "more data helped" would be
        # indistinguishable from "the val set got easier".
        tr = np.random.default_rng(1).permutation(tr)[:subsample]
        print(f"train subsampled to {len(tr)} rows (val untouched — controlled A/B)", flush=True)

    import copy

    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        rng.shuffle(tr)
        total = 0.0
        for i in range(0, len(tr), bs):
            xb, ob, fb, yb = _batch(X, O, F, y, tr[i : i + bs])
            xb = augment(xb) # train-time affine augmentation (bitmap only; online left clean)
            opt.zero_grad()
            loss = loss_fn(model(xb, ob, fb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        acc = evaluate(model, X, O, F, y, va, bs) if va.size else float("nan")
        # Keep the best-on-validation weights, not the noisy final epoch.
        if acc > best_val:
            best_val, best_state = acc, copy.deepcopy(model.state_dict())
        print(f"epoch {ep:3d}  loss {total / max(1, len(tr)):.4f}  val top-5 {acc:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    micro, macro, seen = evaluate(model, X, O, F, y, va, bs, n_classes)
    tr_micro = evaluate(model, X, O, F, y, tr[: len(va)], bs)  # same size, for the gap
    print(
        f"\nbest epoch — val top-5 {micro:.4f} (micro) | {macro:.4f} (macro, over the "
        f"{seen} classes present in val) | train top-5 {tr_micro:.4f}",
        flush=True,
    )
    return model, va


# --- symmetric int8 post-training quantization -------------------------------
def qscale(w):
    """Per-tensor symmetric scale: real ≈ scale * q, q in [-127, 127]."""
    m = float(np.max(np.abs(w))) if w.size else 0.0
    return (m / 127.0) if m > 0 else 1.0


def quant_i8(w, scale):
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8)


def calibrate_input_scales(model, X, O, F, n_calib=2048):
    """Capture the max-abs activation feeding each layer over a calibration batch,
    so activations quantize symmetrically. Returns s_in per layer. Note `f1.in_scale`
    covers the whole concatenated input (cnn + online + globals), which is exactly
    what the Rust forward pass quantizes all three parts at."""
    import torch

    acts = {}
    hooks = []
    layers = [("c1", model.c1), ("c2", model.c2), ("o1", model.o1), ("f1", model.f1), ("f2", model.f2)]
    for name, mod in layers:
        hooks.append(
            mod.register_forward_pre_hook(
                lambda _m, inp, nm=name: acts.__setitem__(
                    nm, max(acts.get(nm, 0.0), float(inp[0].abs().max()))
                )
            )
        )
    model.eval()
    m = min(n_calib, len(X))
    with torch.no_grad():
        xb, ob, fb, _ = _batch(X, O, F, np.zeros(m, np.int64), np.arange(m))
        model(xb, ob, fb)
    for h in hooks:
        h.remove()
    return {k: (v / 127.0 if v > 0 else 1.0) for k, v in acts.items()}


def export(model, classes, out_path, s_in):
    """Write the int8 model as `.iwt` (contract in the module docstring) + labels."""
    w = iwt.WeightsWriter()
    layers = [("c1", model.c1), ("c2", model.c2), ("o1", model.o1), ("f1", model.f1), ("f2", model.f2)]
    for name, mod in layers:
        W = mod.weight.detach().cpu().numpy()
        if name == "o1":
            # Conv1d weight [OC,IC,KW] → the 4-D [OC,IC,1,KW] the Rust reader expects
            # (row-major layout is identical; the size-1 axis just makes it rank-4).
            W = W.reshape(W.shape[0], W.shape[1], 1, W.shape[2])
        b = mod.bias.detach().cpu().numpy()
        sw = qscale(W)
        acc_scale = s_in[name] * sw  # scale of the int32 accumulator
        w.i8(f"{name}.w", list(W.shape), sw, quant_i8(W, sw))
        w.i32(f"{name}.b", list(b.shape), np.round(b / acc_scale).astype(np.int32))
        w.f32(f"{name}.in_scale", [1], [s_in[name]])
    w.write(out_path)
    with open(os.path.splitext(out_path)[0] + ".labels.txt", "w") as f:
        f.write("\n".join(classes) + "\n")
    print(f"wrote {out_path} ({len(w.to_bytes())} bytes) + labels for {len(classes)} classes")


def dump_val(d, X, O, F, y, classes, va, size, nf):
    """Write the held-out split as a dataset dir, so the Rust int8 evaluator
    (`ink2tex-desktop --eval`) can score *any* model on exactly these rows. That is what
    makes an old-vs-new comparison a controlled experiment rather than two numbers
    measured on two different val sets."""
    os.makedirs(d, exist_ok=True)
    X[va].astype(np.uint8).tofile(os.path.join(d, "images.u8"))
    F[va].astype("<f4").tofile(os.path.join(d, "features.f32"))
    y[va].astype("<u4").tofile(os.path.join(d, "labels.u32"))
    if O is not None:
        O[va].astype("<f4").tofile(os.path.join(d, "online.f32"))
    with open(os.path.join(d, "classes.txt"), "w") as f:
        f.write("\n".join(classes) + "\n")
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"n": len(va), "size": size, "num_features": nf,
                   "num_classes": len(classes),
                   "online_len": ONLINE_CH * ONLINE_LEN if O is not None else 0}, f)
    print(f"wrote the {len(va)}-row held-out split to {d}/ (for --eval)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+",
                    help="dataset dir(s) from --prepare-detexify; several are concatenated "
                         "(they must share a pinned --classes label space)")
    ap.add_argument("--out", default="train/model.iwt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = use all")
    ap.add_argument("--train-subsample", type=int, default=0, metavar="N",
                    help="train on only N rows of the train split (val untouched). Use this "
                         "to measure what more DATA buys, holding everything else fixed.")
    ap.add_argument("--dump-val", metavar="DIR",
                    help="write the held-out split here as a dataset dir, for --eval")
    args = ap.parse_args()

    import torch

    # Deterministic weight init + augmentation, so a given dataset reproduces the same
    # model — the same reproducibility ethos as the rasterizer and the int8 kernel.
    torch.manual_seed(0)

    X, O, F, y, classes, size, nf = load_dataset(args.data, args.max_samples)
    chan = "+online" if O is not None else "no-online"
    print(
        f"{len(y)} samples, {len(classes)} classes, {size}×{size} bitmaps, {nf} features, {chan}",
        flush=True,
    )
    gid = shape_groups(X, y)
    model = build_model(nf, len(classes), size)
    model, va = train(model, X, O, F, y, gid, args.epochs, args.lr, args.val_frac,
                      args.batch_size, len(classes), args.train_subsample)
    if args.dump_val:
        dump_val(args.dump_val, X, O, F, y, classes, va, size, nf)
    s_in = calibrate_input_scales(model, X, O, F)
    export(model, classes, args.out, s_in)


if __name__ == "__main__":
    main()
