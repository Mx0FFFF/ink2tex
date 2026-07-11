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


def load_dataset(d, max_samples=0):
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
    if max_samples and n > max_samples:
        sel = np.random.default_rng(0).permutation(n)[:max_samples]
        X, F, y = X[sel], F[sel], y[sel]
        O = O[sel] if O is not None else None
    return X, O, F, y, classes, size, nf


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
            self.drop = nn.Dropout(0.3) # regularize: 35 samples/class is little data

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
    cheap synthetic variety that effectively multiplies our limited data and makes the
    classifier robust to how a symbol is drawn. Train-time ONLY (inference sees the
    clean rasterization). This is the main accuracy lever when more data isn't
    available (the full 342k Detexify Drive dump is inaccessible)."""
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
    full 342k×32×32 set never lives in memory as float)."""
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


def evaluate(model, X, O, F, y, idx, bs):
    import torch

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb, ob, fb, yb = _batch(X, O, F, y, idx[i : i + bs])
            logits = model(xb, ob, fb)
            k = min(5, logits.shape[1])
            top5 = logits.topk(k, dim=1).indices
            correct += (top5 == yb[:, None]).any(1).sum().item()
    return correct / max(1, len(idx))


def train(model, X, O, F, y, epochs, lr, val_frac, bs):
    import torch

    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    cut = max(1, int(n * (1 - val_frac)))
    tr, va = idx[:cut].copy(), idx[cut:].copy()

    import copy

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
        print(f"restored best epoch (val top-5 {best_val:.3f})", flush=True)
    return model


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir from --prepare-detexify")
    ap.add_argument("--out", default="train/model.iwt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = use all")
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
    model = build_model(nf, len(classes), size)
    train(model, X, O, F, y, args.epochs, args.lr, args.val_frac, args.batch_size)
    s_in = calibrate_input_scales(model, X, O, F)
    export(model, classes, args.out, s_in)


if __name__ == "__main__":
    main()
