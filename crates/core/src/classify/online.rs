//! The **online input channel** (DESIGN §3(b)) — the free temporal signal the
//! research systems throw away when they rasterize strokes to bitmaps.
//!
//! A symbol's pen trajectory (its strokes, in order) is resampled to a fixed `N`
//! points by arc length, then encoded as four channels: `dx`, `dy` (the velocity
//! direction between consecutive resampled points), `pen_up` (1 at a stroke
//! boundary), and `curvature` (the turning angle). Positions are normalized to the
//! symbol's bounding box, so it is scale- and translation-invariant, and the whole
//! thing is deterministic — the exact same featurization runs at training and at
//! inference (like the rasterizer), so there is no skew.
//!
//! Output layout is **channel-major**, matching a 1-D conv's `[C, L]`:
//! `[dx_0..dx_N, dy_0..dy_N, pen_up_0..pen_up_N, curv_0..curv_N]`.

use crate::stroke::Stroke;

/// Resampled sequence length.
pub const ONLINE_POINTS: usize = 64;
/// Channels: dx, dy, pen_up, curvature.
pub const ONLINE_CHANNELS: usize = 4;
/// Stride of the online 1-D conv (downsamples in place of a pool, since the int8
/// `maxpool2d_i8` is square and can't pool a height-1 tensor). **Must match the
/// trainer's `Conv1d(stride=…)`** — the model forward pass (`model.rs`) reads the
/// conv's channel/kernel dims from the weight tensor but takes this stride as fixed.
pub const ONLINE_STRIDE: usize = 2;

/// One resampled point: normalized position + which stroke it belongs to.
struct Rp {
    x: f32,
    y: f32,
    stroke: usize,
}

/// Compute the online-channel feature `[ONLINE_CHANNELS × n]`, channel-major. Returns
/// all-zeros for degenerate input (no points / zero arc length).
pub fn online_features(strokes: &[Stroke], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ONLINE_CHANNELS * n];
    if n == 0 {
        return out;
    }

    // Gather all points with their stroke index.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ss = Vec::new();
    for (si, stroke) in strokes.iter().enumerate() {
        for p in &stroke.points {
            xs.push(p.x);
            ys.push(p.y);
            ss.push(si);
        }
    }
    if xs.len() < 2 {
        return out;
    }

    // Normalize positions to the bbox (centered, larger side = 1).
    let (mut nx, mut ny, mut mx, mut my) = (xs[0], ys[0], xs[0], ys[0]);
    for i in 0..xs.len() {
        nx = nx.min(xs[i]);
        ny = ny.min(ys[i]);
        mx = mx.max(xs[i]);
        my = my.max(ys[i]);
    }
    let scale = (mx - nx).max(my - ny).max(1e-6);
    let (cx, cy) = ((nx + mx) * 0.5, (ny + my) * 0.5);
    for i in 0..xs.len() {
        xs[i] = (xs[i] - cx) / scale;
        ys[i] = (ys[i] - cy) / scale;
    }

    // Cumulative arc length — only *within* a stroke (a pen-up jump adds nothing).
    let mut cum = vec![0.0f32; xs.len()];
    for k in 1..xs.len() {
        let d = if ss[k] == ss[k - 1] {
            ((xs[k] - xs[k - 1]).powi(2) + (ys[k] - ys[k - 1]).powi(2)).sqrt()
        } else {
            0.0
        };
        cum[k] = cum[k - 1] + d;
    }
    let total = cum[xs.len() - 1];
    if total <= 1e-6 {
        return out;
    }

    // Resample n points equally spaced by arc length.
    let mut rp: Vec<Rp> = Vec::with_capacity(n);
    let mut k = 0usize;
    for i in 0..n {
        let t = if n == 1 {
            0.0
        } else {
            i as f32 * total / (n as f32 - 1.0)
        };
        while k + 1 < xs.len() && cum[k + 1] < t {
            k += 1;
        }
        if k + 1 >= xs.len() {
            rp.push(Rp {
                x: xs[k],
                y: ys[k],
                stroke: ss[k],
            });
        } else {
            let seg = cum[k + 1] - cum[k];
            if seg > 1e-9 {
                let f = ((t - cum[k]) / seg).clamp(0.0, 1.0);
                rp.push(Rp {
                    x: xs[k] + f * (xs[k + 1] - xs[k]),
                    y: ys[k] + f * (ys[k + 1] - ys[k]),
                    stroke: ss[k],
                });
            } else {
                rp.push(Rp {
                    x: xs[k],
                    y: ys[k],
                    stroke: ss[k],
                });
            }
        }
    }

    // Encode dx, dy, pen_up.
    let mut dx = vec![0.0f32; n];
    let mut dy = vec![0.0f32; n];
    for i in 0..n {
        if i + 1 < n {
            dx[i] = rp[i + 1].x - rp[i].x;
            dy[i] = rp[i + 1].y - rp[i].y;
            out[2 * n + i] = if rp[i + 1].stroke != rp[i].stroke {
                1.0
            } else {
                0.0
            };
        } else {
            out[2 * n + i] = 1.0; // end of trajectory
        }
        out[i] = dx[i];
        out[n + i] = dy[i];
    }

    // Curvature = signed turning angle between successive velocity vectors, in [-1,1].
    for i in 1..n.saturating_sub(1) {
        let (ax, ay, bx, by) = (dx[i - 1], dy[i - 1], dx[i], dy[i]);
        let cross = ax * by - ay * bx;
        let dot = ax * bx + ay * by;
        out[3 * n + i] = cross.atan2(dot) / core::f32::consts::PI;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stroke::Point;

    fn stroke(pts: &[(f32, f32)]) -> Stroke {
        Stroke {
            points: pts
                .iter()
                .map(|&(x, y)| Point::new(x, y, 1.0, 0.0, 0.0, 0))
                .collect(),
        }
    }

    #[test]
    fn empty_is_zeros() {
        assert!(online_features(&[], 64).iter().all(|&v| v == 0.0));
        assert!(
            online_features(&[stroke(&[(0.5, 0.5)])], 64) // single point
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn output_shape() {
        let f = online_features(&[stroke(&[(0.1, 0.5), (0.9, 0.5)])], 64);
        assert_eq!(f.len(), ONLINE_CHANNELS * 64);
    }

    #[test]
    fn horizontal_stroke_moves_in_x() {
        // A left-to-right line: dx > 0, dy ≈ 0, low curvature, pen_up only at the end.
        let n = 16;
        let f = online_features(&[stroke(&[(0.1, 0.5), (0.5, 0.5), (0.9, 0.5)])], n);
        let (dx, dy, penup) = (&f[0..n], &f[n..2 * n], &f[2 * n..3 * n]);
        assert!(dx[0] > 0.0, "dx should be positive");
        assert!(
            dy.iter().all(|&v| v.abs() < 1e-3),
            "dy ~ 0 for a horizontal line"
        );
        assert_eq!(penup[n - 1], 1.0); // end
        assert!(penup[..n - 1].iter().all(|&v| v == 0.0)); // single stroke, no interior lifts
    }

    #[test]
    fn two_strokes_mark_a_pen_up() {
        let n = 32;
        let f = online_features(
            &[
                stroke(&[(0.1, 0.2), (0.4, 0.2)]),
                stroke(&[(0.6, 0.8), (0.9, 0.8)]),
            ],
            n,
        );
        let penup = &f[2 * n..3 * n];
        // Exactly one interior pen-up (the boundary), plus the terminal one.
        let interior_lifts = penup[..n - 1].iter().filter(|&&v| v > 0.5).count();
        assert_eq!(interior_lifts, 1);
    }

    #[test]
    fn scale_and_translation_invariant() {
        let a = online_features(&[stroke(&[(0.3, 0.4), (0.5, 0.6), (0.4, 0.3)])], 32);
        let big = stroke(&[(0.6, 0.8), (1.0, 1.2), (0.8, 0.6)]); // 2× + shifted-ish
        let b = online_features(&[big], 32);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4, "features should be scale-invariant");
        }
    }

    #[test]
    fn deterministic() {
        let s = [stroke(&[(0.2, 0.2), (0.8, 0.3), (0.5, 0.9)])];
        assert_eq!(online_features(&s, 64), online_features(&s, 64));
    }
}
