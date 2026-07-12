//! Stroke-group → `size×size` bitmap — the classifier's **offline input channel**
//! (DESIGN.md §4.3 (a): "render the stroke group to a 32×32 anti-aliased bitmap,
//! aspect-preserving, centred").
//!
//! ## The train/inference-skew footgun
//! This exact preprocessing runs in *two* places: Python, to build training
//! bitmaps, and Rust, to featurize live ink on the device. If they differ by even
//! a pixel, the model sees a different distribution at inference than it trained on
//! and accuracy silently rots. So the algorithm here is **deterministic** and
//! pinned by these constants — the training pipeline preprocesses through the same
//! definition (ideally by calling this code, not re-deriving it). Anti-aliasing is
//! plain supersample-then-box-downsample: draw into an `SS×` canvas as 1-bit
//! coverage, then average each `SS×SS` block into a `[0,1]` grey value.

use crate::stroke::Stroke;

/// Supersampling factor (anti-aliasing quality). Part of the pinned contract.
const SS: usize = 4;
/// Border kept clear around the ink, as a fraction of the canvas.
const MARGIN: f32 = 0.10;
/// Ink half-width in output pixels (stroke thickness).
const INK_RADIUS_PX: f32 = 0.5;

/// Number of scalar global stroke features emitted alongside the bitmap.
pub const NUM_FEATURES: usize = 7;

/// Render `strokes` (in normalized coords) to a `size×size` grayscale bitmap,
/// row-major, values in `[0, 1]` where 1.0 is full ink coverage. Aspect-preserving
/// and centered: the ink's bounding-box center maps to the canvas center and its
/// larger dimension fills the content area. Scale- and translation-invariant, so a
/// symbol drawn big in a corner and small in the middle rasterize the same.
pub fn rasterize(strokes: &[Stroke], size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; size * size];
    if size == 0 {
        return out;
    }
    let Some((min_x, min_y, max_x, max_y)) = bounds(strokes) else {
        return out; // no points → blank
    };

    let hi = size * SS;
    let (cx, cy) = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
    let ext = (max_x - min_x).max(max_y - min_y);
    let content = hi as f32 * (1.0 - 2.0 * MARGIN);
    // A degenerate extent (single point / axis-aligned line) → scale 0, which keeps
    // that axis pinned to the center rather than exploding.
    let scale = if ext > f32::EPSILON {
        content / ext
    } else {
        0.0
    };
    let half = hi as f32 * 0.5;
    let map = |x: f32, y: f32| ((x - cx) * scale + half, (y - cy) * scale + half);
    let radius = INK_RADIUS_PX * SS as f32;

    let mut hi_buf = vec![0u8; hi * hi];
    for s in strokes {
        match s.points.as_slice() {
            [] => {}
            [p] => {
                let (px, py) = map(p.x, p.y);
                stamp_disk(&mut hi_buf, hi, px, py, radius);
            }
            pts => {
                for w in pts.windows(2) {
                    let (x0, y0) = map(w[0].x, w[0].y);
                    let (x1, y1) = map(w[1].x, w[1].y);
                    stamp_segment(&mut hi_buf, hi, x0, y0, x1, y1, radius);
                }
            }
        }
    }

    // Box-downsample SS×SS → coverage fraction in [0,1] (this is the anti-aliasing).
    let cells = (SS * SS) as f32;
    for oy in 0..size {
        for ox in 0..size {
            let mut covered = 0u32;
            for sy in 0..SS {
                for sx in 0..SS {
                    if hi_buf[(oy * SS + sy) * hi + (ox * SS + sx)] != 0 {
                        covered += 1;
                    }
                }
            }
            out[oy * size + ox] = covered as f32 / cells;
        }
    }
    out
}

/// A small vector of shape cues the classifier uses alongside the bitmap (DESIGN
/// §4.3: "a handful of global stroke features"). Fields:
/// `[stroke_count, elongation, arc_length, start_x, start_y, end_x, end_y]`.
/// Elongation and arc length disambiguate size-lost symbols (e.g. `.` vs `-`); the
/// structure stage later refines size-ambiguous cases (DESIGN §4.3).
///
/// # Every feature here is **dimensionless**. It must stay that way.
///
/// This is the same train/inference-skew footgun the module header warns about, and it
/// is *easier* to trip here than in the bitmap. `rasterize` aspect-fits into its own
/// canvas, so it cannot help but be scale-invariant. This function looks at raw
/// coordinates, and the ink reaching it arrives in whatever units its source happens to
/// use: Detexify's bulk dump is in **screen pixels** (a stroke spans ~300 units),
/// detexify-next ships **0–1 normalized floats**, and `crates/rm` hands us **normalized
/// device ink**. Emit an arc length or a start point in raw units and the model bakes
/// the corpus's coordinate system into its weights.
///
/// That is not hypothetical — it is the bug this shape was written to fix. The features
/// were previously raw (`arc`, and `sx/sy/ex/ey` straight off the point), which nobody
/// noticed only because detexify-next's coordinates *happened* to already be normalized
/// like the device's. Scoring that model against the same symbols in pixel coordinates
/// gave **7.9% top-5** where it otherwise gets 90%+ — and a model trained the other way
/// round would have failed exactly as hard on the tablet, on real ink, in the user's
/// hands. The bitmap and online channels were fine throughout; the giveaway is that
/// their invariance is *tested* and this vector's never was. It is now
/// (`features_are_scale_and_translation_invariant`).
pub fn global_features(strokes: &[Stroke]) -> [f32; NUM_FEATURES] {
    let count = strokes.iter().filter(|s| !s.points.is_empty()).count();
    let (nx, ny, mx, my) = bounds(strokes).unwrap_or((0.0, 0.0, 0.0, 0.0));
    let (w, h) = (mx - nx, my - ny);

    // The ink's own frame: translate by the bounding box origin, scale by its longest
    // side. Uniform (not per-axis) so the aspect ratio isn't squashed out — `elong`
    // below reports that separately.
    let span = w.max(h);
    let span = if span > 1e-6 { span } else { 1.0 }; // a single dot has no extent
    let rel = |p: &crate::stroke::Point| ((p.x - nx) / span, (p.y - ny) / span);

    let mut arc = 0.0f32;
    for s in strokes {
        for p in s.points.windows(2) {
            let (dx, dy) = (p[1].x - p[0].x, p[1].y - p[0].y);
            arc += (dx * dx + dy * dy).sqrt();
        }
    }

    // Elongation in [0, 1]: 0 = a vertical sliver, 0.5 = square, 1 = a horizontal bar.
    // Deliberately NOT the raw w/h ratio, which is unbounded — a minus sign is a few
    // thousandths tall and would emit an aspect in the hundreds. All seven features
    // share ONE int8 scale with the 1,384 CNN/online activations they are concatenated
    // to (see model.rs), so a single wild feature drags that scale up and quantizes
    // everything else to near-zero. Keep every feature O(1).
    let elong = if w + h > 1e-6 { w / (w + h) } else { 0.5 };

    let first = strokes.iter().flat_map(|s| s.points.first()).next();
    let last = strokes.iter().rev().flat_map(|s| s.points.last()).next();
    let (sx, sy) = first.map_or((0.0, 0.0), rel);
    let (ex, ey) = last.map_or((0.0, 0.0), rel);
    [count as f32, elong, arc / span, sx, sy, ex, ey]
}

fn bounds(strokes: &[Stroke]) -> Option<(f32, f32, f32, f32)> {
    let mut it = strokes.iter().flat_map(|s| s.points.iter());
    let p0 = it.next()?;
    let (mut nx, mut ny, mut mx, mut my) = (p0.x, p0.y, p0.x, p0.y);
    for p in it {
        nx = nx.min(p.x);
        ny = ny.min(p.y);
        mx = mx.max(p.x);
        my = my.max(p.y);
    }
    Some((nx, ny, mx, my))
}

/// Set every hi-res pixel whose center lies within `r` of `(cx, cy)`.
fn stamp_disk(buf: &mut [u8], hi: usize, cx: f32, cy: f32, r: f32) {
    let r2 = r * r;
    let hif = hi as f32;
    let x0 = (cx - r).floor().clamp(0.0, hif) as usize;
    let x1 = (cx + r).ceil().clamp(0.0, hif) as usize;
    let y0 = (cy - r).floor().clamp(0.0, hif) as usize;
    let y1 = (cy + r).ceil().clamp(0.0, hif) as usize;
    for iy in y0..y1 {
        for ix in x0..x1 {
            let dx = ix as f32 + 0.5 - cx;
            let dy = iy as f32 + 0.5 - cy;
            if dx * dx + dy * dy <= r2 {
                buf[iy * hi + ix] = 1;
            }
        }
    }
}

/// Stamp disks along a segment (a thick, round-capped line).
fn stamp_segment(buf: &mut [u8], hi: usize, x0: f32, y0: f32, x1: f32, y1: f32, r: f32) {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let steps = (dx * dx + dy * dy).sqrt().ceil().max(1.0) as usize;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        stamp_disk(buf, hi, x0 + dx * t, y0 + dy * t, r);
    }
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

    // Center of mass of the ink, in pixel coords, plus total coverage.
    fn centroid(img: &[f32], size: usize) -> (f32, f32, f32) {
        let (mut sx, mut sy, mut s) = (0.0, 0.0, 0.0);
        for y in 0..size {
            for x in 0..size {
                let v = img[y * size + x];
                sx += v * x as f32;
                sy += v * y as f32;
                s += v;
            }
        }
        (sx / s.max(1e-6), sy / s.max(1e-6), s)
    }

    #[test]
    fn empty_is_blank() {
        assert!(rasterize(&[], 32).iter().all(|&v| v == 0.0));
        assert!(rasterize(&[stroke(&[])], 32).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn values_are_in_unit_range() {
        let img = rasterize(&[stroke(&[(0.2, 0.3), (0.8, 0.7)])], 32);
        assert!(img.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert!(img.iter().any(|&v| v > 0.0)); // something got drawn
    }

    #[test]
    fn ink_is_centered() {
        // A symmetric drawing off in a corner must still land centered. Closed
        // square (5th point returns to start) so its ink centroid == bbox center.
        let img = rasterize(
            &[stroke(&[
                (0.05, 0.05),
                (0.25, 0.05),
                (0.25, 0.25),
                (0.05, 0.25),
                (0.05, 0.05),
            ])],
            32,
        );
        let (cx, cy, total) = centroid(&img, 32);
        assert!(total > 0.0);
        assert!((cx - 16.0).abs() < 2.0, "cx={cx}");
        assert!((cy - 16.0).abs() < 2.0, "cy={cy}");
    }

    #[test]
    fn scale_and_translation_invariant() {
        // Same shape, drawn tiny-and-centered vs big-and-shifted → same framing.
        let shape = [(0.40, 0.45), (0.60, 0.45), (0.50, 0.60)];
        let small = rasterize(&[stroke(&shape)], 32);
        let big: Vec<_> = shape
            .iter()
            .map(|&(x, y)| (x * 3.0 + 0.1, y * 3.0 - 0.2))
            .collect();
        let large = rasterize(&[stroke(&big)], 32);
        let (ax, ay, atot) = centroid(&small, 32);
        let (bx, by, btot) = centroid(&large, 32);
        assert!(
            (ax - bx).abs() < 1.5 && (ay - by).abs() < 1.5,
            "centroids differ"
        );
        assert!((atot - btot).abs() / atot < 0.15, "ink area differs a lot");
    }

    #[test]
    fn horizontal_line_hugs_the_middle_rows() {
        let img = rasterize(&[stroke(&[(0.1, 0.5), (0.9, 0.5)])], 32);
        let (_, cy, _) = centroid(&img, 32);
        assert!((cy - 16.0).abs() < 2.0); // vertically centered
                                          // Top and bottom rows are clear.
        assert!(img[0..32].iter().all(|&v| v == 0.0));
        assert!(img[31 * 32..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn is_deterministic() {
        let s = [stroke(&[(0.2, 0.2), (0.8, 0.3), (0.5, 0.9)])];
        assert_eq!(rasterize(&s, 32), rasterize(&s, 32));
    }

    #[test]
    fn features_capture_shape() {
        let s = [
            stroke(&[(0.0, 0.0), (1.0, 0.0)]),
            stroke(&[(0.5, 0.0), (0.5, 0.5)]),
        ];
        let f = global_features(&s);
        assert_eq!(f.len(), NUM_FEATURES);
        assert_eq!(f[0], 2.0); // two strokes
        assert!(f[2] >= 1.4); // arc length ≥ 1.0 (horizontal) + 0.5 (vertical)
        assert_eq!((f[3], f[4]), (0.0, 0.0)); // start point
    }

    /// The test that wasn't here — and the bug it would have caught.
    ///
    /// `rasterize`'s invariance is asserted above; this vector's never was. So it drifted
    /// into emitting raw coordinates, which only worked because the training corpus
    /// happened to be normalized like the device. Feed the identical glyph in the bulk
    /// dump's pixel units and the model that scores 90%+ collapses to 7.9% top-5.
    #[test]
    fn features_are_scale_and_translation_invariant() {
        let shape = [(0.10, 0.20), (0.60, 0.25), (0.35, 0.70)];
        let unit = [stroke(&shape)];
        // The same glyph as the Detexify bulk dump ships it: hundreds of pixels, offset.
        let as_pixels: Vec<_> = shape
            .iter()
            .map(|&(x, y)| (x * 480.0 + 120.0, y * 480.0 + 37.0))
            .collect();
        let pixels = [stroke(&as_pixels)];

        let (a, b) = (global_features(&unit), global_features(&pixels));
        for i in 0..NUM_FEATURES {
            assert!(
                (a[i] - b[i]).abs() < 1e-3,
                "feature {i} carries the coordinate system ({} vs {}): a model trained on \
                 one corpus cannot survive another — nor the device",
                a[i],
                b[i],
            );
        }
    }

    /// Every feature shares ONE int8 scale with the 1,384 CNN/online activations it is
    /// concatenated to, so one wild value quantizes all the others to zero.
    #[test]
    fn features_stay_bounded_for_a_flat_stroke() {
        // A minus sign: wide, and — bar the pen's jitter — of no height at all. The old
        // raw `w/h` aspect reported ~800,000 here.
        let f = global_features(&[stroke(&[(0.1, 0.5), (0.9, 0.5000001)])]);
        assert!(
            f.iter().all(|v| v.is_finite() && v.abs() <= 20.0),
            "unbounded feature would wreck int8 calibration: {f:?}",
        );
    }
}
