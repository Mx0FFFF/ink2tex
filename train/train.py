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
#         → concat 7 global features → 1031 → fc1(→64)+relu → fc2(→classes)
CONV1, CONV2, FC1, KSIZE = 8, 16, 64, 3


def load_dataset(d, max_samples=0):
    meta = json.load(open(os.path.join(d, "meta.json")))
    n, size, nf = meta["n"], meta["size"], meta["num_features"]
    X = np.fromfile(os.path.join(d, "images.u8"), np.uint8).reshape(n, 1, size, size)
    F = np.fromfile(os.path.join(d, "features.f32"), "<f4").reshape(n, nf)
    y = np.fromfile(os.path.join(d, "labels.u32"), "<u4").astype(np.int64)
    classes = [c for c in open(os.path.join(d, "classes.txt")).read().split("\n") if c]
    if max_samples and n > max_samples:
        sel = np.random.default_rng(0).permutation(n)[:max_samples]
        X, F, y = X[sel], F[sel], y[sel]
    return X, F, y, classes, size, nf


def build_model(nf, n_classes, size):
    import torch
    import torch.nn as nn

    flat = CONV2 * (size // 4) * (size // 4)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, CONV1, KSIZE, padding=1)
            self.c2 = nn.Conv2d(CONV1, CONV2, KSIZE, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.f1 = nn.Linear(flat + nf, FC1)
            self.f2 = nn.Linear(FC1, n_classes)

        def forward(self, x, feats):
            x = self.pool(torch.relu(self.c1(x)))
            x = self.pool(torch.relu(self.c2(x)))
            x = torch.cat([x.flatten(1), feats], dim=1)
            x = torch.relu(self.f1(x))
            return self.f2(x)

    return Net()


def _batch(X, F, y, idx):
    """Materialize one mini-batch as tensors (uint8 → float32/255 on the fly, so the
    full 342k×32×32 set never lives in memory as float)."""
    import torch

    xb = torch.from_numpy(X[idx].astype(np.float32) / 255.0)
    fb = torch.from_numpy(F[idx])
    yb = torch.from_numpy(y[idx])
    return xb, fb, yb


def evaluate(model, X, F, y, idx, bs):
    import torch

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb, fb, yb = _batch(X, F, y, idx[i : i + bs])
            logits = model(xb, fb)
            k = min(5, logits.shape[1])
            top5 = logits.topk(k, dim=1).indices
            correct += (top5 == yb[:, None]).any(1).sum().item()
    return correct / max(1, len(idx))


def train(model, X, F, y, epochs, lr, val_frac, bs):
    import torch

    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    cut = max(1, int(n * (1 - val_frac)))
    tr, va = idx[:cut].copy(), idx[cut:].copy()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        rng.shuffle(tr)
        total = 0.0
        for i in range(0, len(tr), bs):
            xb, fb, yb = _batch(X, F, y, tr[i : i + bs])
            opt.zero_grad()
            loss = loss_fn(model(xb, fb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        acc = evaluate(model, X, F, y, va, bs) if va.size else float("nan")
        print(f"epoch {ep:3d}  loss {total / max(1, len(tr)):.4f}  val top-5 {acc:.3f}", flush=True)
    return model


# --- symmetric int8 post-training quantization -------------------------------
def qscale(w):
    """Per-tensor symmetric scale: real ≈ scale * q, q in [-127, 127]."""
    m = float(np.max(np.abs(w))) if w.size else 0.0
    return (m / 127.0) if m > 0 else 1.0


def quant_i8(w, scale):
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8)


def calibrate_input_scales(model, X, F, n_calib=2048):
    """Capture the max-abs activation feeding each layer over a calibration batch,
    so activations quantize symmetrically. Returns s_in per layer."""
    import torch

    acts = {}
    hooks = []
    for name, mod in [("c1", model.c1), ("c2", model.c2), ("f1", model.f1), ("f2", model.f2)]:
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
        model(*_batch(X, F, np.zeros(m, np.int64), np.arange(m))[:2])
    for h in hooks:
        h.remove()
    return {k: (v / 127.0 if v > 0 else 1.0) for k, v in acts.items()}


def export(model, classes, out_path, s_in):
    """Write the int8 model as `.iwt` (contract in the module docstring) + labels."""
    w = iwt.WeightsWriter()
    for name, mod in [("c1", model.c1), ("c2", model.c2), ("f1", model.f1), ("f2", model.f2)]:
        W = mod.weight.detach().cpu().numpy()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir from --prepare-detexify")
    ap.add_argument("--out", default="train/model.iwt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = use all")
    args = ap.parse_args()

    X, F, y, classes, size, nf = load_dataset(args.data, args.max_samples)
    print(f"{len(y)} samples, {len(classes)} classes, {size}×{size} bitmaps, {nf} features", flush=True)
    model = build_model(nf, len(classes), size)
    train(model, X, F, y, args.epochs, args.lr, args.val_frac, args.batch_size)
    s_in = calibrate_input_scales(model, X, F)
    export(model, classes, args.out, s_in)


if __name__ == "__main__":
    main()
