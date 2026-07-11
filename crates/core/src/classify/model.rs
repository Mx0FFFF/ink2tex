//! The M1 forward pass: a quantized `.iwt` model + a 32×32 bitmap + global features
//! → ranked top-k symbol predictions. It composes the `kernel` primitives in the
//! exact order and with the exact scales the trainer (`train/train.py`) quantized
//! against — that shared contract is the whole reason the int8 math lines up.
//!
//! Scale chaining (symmetric int8, zero-point 0). Each layer L exports its weight
//! scale `sw` (on the i8 tensor) plus `L.in_scale` (the activation scale feeding it,
//! from calibration). The requantize multiplier from an accumulator into the next
//! layer's int8 is `in_scale[L] * sw[L] / in_scale[L+1]`; the final layer dequantizes
//! with `in_scale * sw` straight to f32 logits.
//!
//! A weights blob is *external input*, so every shape is validated up front:
//! `recognize` returns `Error::BadWeights` rather than indexing out of bounds.

use super::kernel::{conv2d_i8, dense_i8, dequantize_i32, maxpool2d_i8, quantize_i8, requantize};
use super::{rank_logits, Prediction, Weights};
use crate::error::{Error, Result};

fn scalar(w: &Weights, name: &str) -> Result<f32> {
    w.get(name)
        .and_then(|t| t.as_f32().first().copied())
        .ok_or(Error::BadWeights("missing scale tensor"))
}

/// Run the M1 CNN. `bitmap` is `size*size` grayscale in `[0,1]` (from `rasterize`);
/// `features` is the `global_features` vector. Returns the ranked top-`k`.
pub fn recognize(
    w: &Weights,
    bitmap: &[f32],
    features: &[f32],
    size: usize,
    k: usize,
) -> Result<Vec<Prediction>> {
    let miss = || Error::BadWeights("model tensor missing");
    let (c1, c2, f1, f2) = (
        w.get("c1.w").ok_or_else(miss)?,
        w.get("c2.w").ok_or_else(miss)?,
        w.get("f1.w").ok_or_else(miss)?,
        w.get("f2.w").ok_or_else(miss)?,
    );
    let (c1b, c2b, f1b, f2b) = (
        w.get("c1.b").ok_or_else(miss)?.as_i32(),
        w.get("c2.b").ok_or_else(miss)?.as_i32(),
        w.get("f1.b").ok_or_else(miss)?.as_i32(),
        w.get("f2.b").ok_or_else(miss)?.as_i32(),
    );
    let (s_c1, s_c2, s_f1, s_f2) = (
        scalar(w, "c1.in_scale")?,
        scalar(w, "c2.in_scale")?,
        scalar(w, "f1.in_scale")?,
        scalar(w, "f2.in_scale")?,
    );

    // Shape validation — reject a malformed model instead of panicking.
    if c1.dims.len() != 4 || c2.dims.len() != 4 || f1.dims.len() != 2 || f2.dims.len() != 2 {
        return Err(Error::BadWeights("unexpected model tensor rank"));
    }
    if size < 4 || bitmap.len() != size * size {
        return Err(Error::BadWeights("bitmap size mismatch"));
    }
    let (oc1, ic1, kh1, kw1) = dims4(c1);
    let (oc2, ic2, kh2, kw2) = dims4(c2);
    let (f1_out, f1_in) = (f1.dims[0] as usize, f1.dims[1] as usize);
    let (n_classes, f2_in) = (f2.dims[0] as usize, f2.dims[1] as usize);
    if ic1 != 1 || ic2 != oc1 {
        return Err(Error::BadWeights("conv channel mismatch"));
    }

    // input → conv1 → relu → pool
    let x = quantize_i8(bitmap, s_c1);
    let (acc, h, wd) = conv2d_i8(
        &x,
        1,
        size,
        size,
        c1.as_i8(),
        oc1,
        kh1,
        kw1,
        &c1b,
        1,
        (kh1 - 1) / 2,
    );
    let a = requantize(&acc, s_c1 * c1.scale / s_c2, true);
    let (a, h, wd) = maxpool2d_i8(&a, oc1, h, wd, 2, 2);

    // conv2 → relu → pool
    let (acc, h, wd) = conv2d_i8(
        &a,
        oc1,
        h,
        wd,
        c2.as_i8(),
        oc2,
        kh2,
        kw2,
        &c2b,
        1,
        (kh2 - 1) / 2,
    );
    let a = requantize(&acc, s_c2 * c2.scale / s_f1, true);
    let (a, _h, _w) = maxpool2d_i8(&a, oc2, h, wd, 2, 2);

    // flatten (already CHW-flat) + global features, quantized at the fc1 input scale
    let mut fc1_in = a;
    fc1_in.extend(quantize_i8(features, s_f1));
    if fc1_in.len() != f1_in {
        return Err(Error::BadWeights("fc1 input size mismatch"));
    }
    let acc = dense_i8(&fc1_in, f1.as_i8(), f1_out, &f1b);
    let hdn = requantize(&acc, s_f1 * f1.scale / s_f2, true);
    if hdn.len() != f2_in {
        return Err(Error::BadWeights("fc2 input size mismatch"));
    }

    // fc2 → dequantize → softmax → top-k
    let acc = dense_i8(&hdn, f2.as_i8(), n_classes, &f2b);
    let logits = dequantize_i32(&acc, s_f2 * f2.scale);
    Ok(rank_logits(&logits, k))
}

fn dims4(t: &super::weights::Tensor) -> (usize, usize, usize, usize) {
    (
        t.dims[0] as usize,
        t.dims[1] as usize,
        t.dims[2] as usize,
        t.dims[3] as usize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::WeightsWriter;

    // A tiny but structurally-real model over a 4×4 input (4→pool→2→pool→1, so the
    // conv stack flattens to `oc2` features + 7 globals). `f1w`/`f1b`/`f2w`/`f2b` let
    // a test wire a deterministic decision; conv weights are arbitrary (all ones).
    fn tiny(f1w: &[i8], f1b: &[i32], f2w: &[i8], f2b: &[i32], n_classes: u32) -> Vec<u8> {
        let mut w = WeightsWriter::new();
        w.i8("c1.w", &[1, 1, 3, 3], 0.01, &[1i8; 9]);
        w.i32("c1.b", &[1], &[0]);
        w.f32("c1.in_scale", &[1], &[0.1]);
        w.i8("c2.w", &[1, 1, 3, 3], 0.01, &[1i8; 9]);
        w.i32("c2.b", &[1], &[0]);
        w.f32("c2.in_scale", &[1], &[0.1]);
        w.i8("f1.w", &[f1b.len() as u32, 8], 0.01, f1w); // fc1 input = oc2(1) + 7 = 8
        w.i32("f1.b", &[f1b.len() as u32], f1b);
        w.f32("f1.in_scale", &[1], &[0.1]);
        w.i8("f2.w", &[n_classes, f1b.len() as u32], 0.01, f2w);
        w.i32("f2.b", &[n_classes], f2b);
        w.f32("f2.in_scale", &[1], &[0.1]);
        w.finish()
    }

    #[test]
    fn runs_and_returns_a_valid_distribution() {
        let blob = tiny(&[1i8; 16], &[0, 0], &[1i8; 4], &[0, 0], 2);
        let w = Weights::parse(&blob).unwrap();
        let bitmap = vec![0.5f32; 16];
        let feats = [0.2f32; 7];
        let preds = recognize(&w, &bitmap, &feats, 4, 2).unwrap();
        assert_eq!(preds.len(), 2);
        let sum: f32 = preds.iter().map(|p| p.prob).sum();
        assert!((sum - 1.0).abs() < 1e-4 || sum < 1.0); // top-2 is a prefix of the full softmax
                                                        // deterministic
        let again = recognize(&w, &bitmap, &feats, 4, 2).unwrap();
        assert_eq!(preds, again);
    }

    #[test]
    fn wired_weights_pick_the_expected_class() {
        // fc1 ignores its input (weights 0) but bias forces a positive hidden vector;
        // fc2 routes all of it to class 0 → class 0 must win, through the full stack.
        let blob = tiny(&[0i8; 16], &[1000, 1000], &[100, 100, 0, 0], &[0, 0], 2);
        let w = Weights::parse(&blob).unwrap();
        let preds = recognize(&w, &[0.5f32; 16], &[0.0f32; 7], 4, 2).unwrap();
        assert_eq!(preds[0].class, 0);
        assert!(preds[0].prob > preds[1].prob);
    }

    #[test]
    fn malformed_model_errors_without_panic() {
        // Missing tensors.
        let empty = WeightsWriter::new().finish();
        let w = Weights::parse(&empty).unwrap();
        assert!(recognize(&w, &[0.0; 16], &[0.0; 7], 4, 2).is_err());
        // Wrong bitmap size.
        let blob = tiny(&[1i8; 16], &[0, 0], &[1i8; 4], &[0, 0], 2);
        let w = Weights::parse(&blob).unwrap();
        assert!(recognize(&w, &[0.0; 9], &[0.0; 7], 4, 2).is_err());
    }
}
