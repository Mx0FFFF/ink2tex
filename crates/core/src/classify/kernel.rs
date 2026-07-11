//! The hand-rolled int8 inference kernel — `conv2d` / `maxpool` / `dense` /
//! `softmax` written by hand, no TFLite / ONNX / candle (CI enforces this). This is
//! deliberate: it's the single richest thing on the learning surface *and* the
//! right engineering call for a 1 GB armv7 device (a ~2 MB static binary, no
//! cross-compile pain, and a memory layout we control).
//!
//! ## Systems concept: quantization + integer accumulation
//! A float CNN stores weights as f32 and multiplies in f32. We instead store them
//! as **int8** with a single f32 `scale` per tensor (symmetric, zero-point 0):
//! the real value ≈ `scale * q`. The heavy inner loop — the multiply-accumulate —
//! then runs entirely in integers: `i8 × i8 → i16`, summed into an `i32`
//! accumulator. That's 4× less memory traffic than f32 (cache-friendly on a
//! Cortex-A7) and maps directly onto NEON's `vmlal`-class instructions later. We
//! keep the accumulator in i32 so it can't overflow: a 3×3×64 conv is ≤ 576 terms
//! of ≤ 127×127, ~9.3 M, far under i32's ~2.1 B range.
//!
//! The design splits cleanly: the MAC primitives are **pure integer** (exactly
//! testable, no float rounding), and a separate `requantize` step folds the scales
//! back in. Direct convolution here is the readable baseline; im2col + a NEON inner
//! loop (and *measuring* the speedup) is a later optimization, not a rewrite.

/// Integer 2D convolution: i8 activations ⊛ i8 weights → **i32 accumulators**.
///
/// Layout (row-major): input CHW `[in_c][in_h][in_w]`; weights OIHW
/// `[out_c][in_c][kh][kw]`; `bias[out_c]` is i32, pre-quantized to the accumulator
/// scale `sx*sw`. Returns `(acc, out_h, out_w)` with `acc` laid out CHW.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_i8(
    input: &[i8],
    in_c: usize,
    in_h: usize,
    in_w: usize,
    weights: &[i8],
    out_c: usize,
    kh: usize,
    kw: usize,
    bias: &[i32],
    stride: usize,
    pad: usize,
) -> (Vec<i32>, usize, usize) {
    let out_h = (in_h + 2 * pad - kh) / stride + 1;
    let out_w = (in_w + 2 * pad - kw) / stride + 1;
    let mut out = vec![0i32; out_c * out_h * out_w];

    for oc in 0..out_c {
        for oy in 0..out_h {
            for ox in 0..out_w {
                // Start from the (pre-quantized) bias so it lands in the accumulator.
                let mut acc = bias.get(oc).copied().unwrap_or(0);
                for ic in 0..in_c {
                    for ky in 0..kh {
                        // Map output+kernel position back to input, undoing padding.
                        let iy = (oy * stride + ky) as isize - pad as isize;
                        if iy < 0 || iy >= in_h as isize {
                            continue; // padded region contributes 0
                        }
                        for kx in 0..kw {
                            let ix = (ox * stride + kx) as isize - pad as isize;
                            if ix < 0 || ix >= in_w as isize {
                                continue;
                            }
                            let w = weights[((oc * in_c + ic) * kh + ky) * kw + kx] as i32;
                            let x = input[(ic * in_h + iy as usize) * in_w + ix as usize] as i32;
                            acc += w * x;
                        }
                    }
                }
                out[(oc * out_h + oy) * out_w + ox] = acc;
            }
        }
    }
    (out, out_h, out_w)
}

/// Fully-connected layer: `out[o] = Σ_i weights[o*in + i] * input[i] + bias[o]`,
/// in i32. `weights` is row-major `[out_features][in_features]`.
pub fn dense_i8(input: &[i8], weights: &[i8], out_features: usize, bias: &[i32]) -> Vec<i32> {
    let in_f = input.len();
    let mut out = vec![0i32; out_features];
    for (o, slot) in out.iter_mut().enumerate() {
        let mut acc = bias.get(o).copied().unwrap_or(0);
        let row = &weights[o * in_f..(o + 1) * in_f];
        for (w, x) in row.iter().zip(input) {
            acc += *w as i32 * *x as i32;
        }
        *slot = acc;
    }
    out
}

/// Fold the scales back in and drop to int8 for the next layer: `q = round(acc *
/// multiplier)`, where `multiplier = sx*sw/s_out`. `relu` clamps negatives to 0.
/// We clamp to `[-127, 127]` (symmetric; -128 is left unused).
pub fn requantize(acc: &[i32], multiplier: f32, relu: bool) -> Vec<i8> {
    acc.iter()
        .map(|&a| {
            let mut v = (a as f32 * multiplier).round() as i32;
            if relu && v < 0 {
                v = 0;
            }
            v.clamp(-127, 127) as i8
        })
        .collect()
}

/// Quantize real values (a `[0,1]` bitmap or a feature vector) to int8 for the
/// first layer: `q = round(v / scale)`, clamped to `[-127, 127]`.
pub fn quantize_i8(vals: &[f32], scale: f32) -> Vec<i8> {
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    vals.iter()
        .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Int8 max-pooling over `k×k` windows, stride `stride`, on a CHW tensor. Pooling
/// commutes with the affine dequant, so the scale is unchanged.
pub fn maxpool2d_i8(
    input: &[i8],
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
) -> (Vec<i8>, usize, usize) {
    let oh = (h - k) / stride + 1;
    let ow = (w - k) / stride + 1;
    let mut out = vec![i8::MIN; c * oh * ow];
    for ch in 0..c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut m = i8::MIN;
                for ky in 0..k {
                    for kx in 0..k {
                        let v = input[(ch * h + oy * stride + ky) * w + ox * stride + kx];
                        if v > m {
                            m = v;
                        }
                    }
                }
                out[(ch * oh + oy) * ow + ox] = m;
            }
        }
    }
    (out, oh, ow)
}

/// Turn integer accumulators back into real values: `r = scale * acc`. Used on the
/// final logits before softmax (`scale = sx*sw` of the last layer).
pub fn dequantize_i32(acc: &[i32], scale: f32) -> Vec<f32> {
    acc.iter().map(|&a| a as f32 * scale).collect()
}

/// Numerically-stable softmax (subtract the max before exp).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Ranked `(index, value)` pairs, highest first — the top-k the correction UI needs.
/// Stable order on ties (lower index first).
pub fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.into_iter().take(k).map(|i| (i, values[i])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_matches_hand_computation() {
        // input·W^T + bias, all integer.
        let input = [1i8, 2, 3];
        let weights = [1i8, 0, -1, /* row0 */ 2, 2, 2 /* row1 */];
        let bias = [10i32, 0];
        let out = dense_i8(&input, &weights, 2, &bias);
        assert_eq!(out, vec![1 - 3 + 10, 2 + 4 + 6]); // [8, 12]
    }

    #[test]
    fn conv2d_matches_hand_computation() {
        // 1×3×3 input, one 2×2 diagonal kernel, stride 1, no pad → 2×2 output.
        let input = [1i8, 2, 3, 4, 5, 6, 7, 8, 9];
        let weights = [1i8, 0, 0, 1]; // [[1,0],[0,1]]
        let (out, oh, ow) = conv2d_i8(&input, 1, 3, 3, &weights, 1, 2, 2, &[0], 1, 0);
        assert_eq!((oh, ow), (2, 2));
        // e.g. top-left = 1*1 + 5*1 = 6; bottom-right = 5*1 + 9*1 = 14.
        assert_eq!(out, vec![6, 8, 12, 14]);
    }

    #[test]
    fn conv2d_padding_keeps_size_with_identity_center() {
        let input = [1i8, 2, 3, 4]; // 1×2×2
        let center = [0i8, 0, 0, 0, 1, 0, 0, 0, 0]; // 3×3, only the center tap set
        let (out, oh, ow) = conv2d_i8(&input, 1, 2, 2, &center, 1, 3, 3, &[0], 1, 1);
        assert_eq!((oh, ow), (2, 2)); // pad 1 with a 3×3 kernel preserves H,W
        assert_eq!(out, vec![1, 2, 3, 4]); // a center-only kernel is the identity
    }

    #[test]
    fn maxpool_takes_window_maxima() {
        let input: Vec<i8> = (0..16).collect(); // 1×4×4, values 0..15
        let (out, oh, ow) = maxpool2d_i8(&input, 1, 4, 4, 2, 2);
        assert_eq!((oh, ow), (2, 2));
        assert_eq!(out, vec![5, 7, 13, 15]);
    }

    #[test]
    fn requantize_clamps_and_relus() {
        let acc = [300i32, -300, 50];
        assert_eq!(requantize(&acc, 0.5, false), vec![127, -127, 25]);
        assert_eq!(requantize(&acc, 0.5, true), vec![127, 0, 25]);
    }

    #[test]
    fn quantize_rounds_and_clamps() {
        assert_eq!(
            quantize_i8(&[0.0, 0.5, 1.0, 2.0], 0.01),
            vec![0, 50, 100, 127]
        );
        assert_eq!(quantize_i8(&[-0.005, 0.005], 0.01), vec![-1, 1]);
    }

    #[test]
    fn dequantize_scales_back() {
        assert_eq!(dequantize_i32(&[2, -3], 0.5), vec![1.0, -1.5]);
    }

    #[test]
    fn softmax_is_a_distribution_and_peaks() {
        let p = softmax(&[0.0, 0.0]);
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6);
        let peak = softmax(&[10.0, 0.0, 0.0]);
        assert!(peak[0] > 0.99);
        assert!((peak.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn top_k_ranks_highest_first() {
        let probs = [0.1, 0.5, 0.3, 0.05, 0.05];
        let top = top_k(&probs, 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 2);
        assert_eq!(top.len(), 2);
    }
}
