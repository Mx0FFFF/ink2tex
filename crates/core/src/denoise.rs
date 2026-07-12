//! Drop stray taps before they become symbols.
//!
//! We read the pen through raw evdev, *below* xochitl, so we capture everything the pen
//! physically does — including taps on xochitl's toolbar. Those arrive as tiny strokes,
//! `segment` dutifully calls each one a symbol, and `structure`, seeing a small mark off
//! the baseline, dutifully makes it a superscript. A row of `α Σ Π √ ∞` came back as
//! `…\infty^{\slash}`. The pipeline was not wrong; it was fed rubbish.
//!
//! # The obvious filter is wrong and would delete real mathematics
//!
//! "Drop small strokes" deletes `\cdot`, the decimal point in `3.14`, and the dot on an
//! `i` — all of which are exactly as small as a stray tap. `docs/core-invariants.md` says
//! it outright: `.` vs `\cdot` vs `\bullet` are *unresolvable* by size. And the asymmetry
//! is brutal: the correction UI can fix a wrong symbol, but it cannot resurrect one that
//! was silently deleted before the user ever saw it. **When in doubt, keep the stroke.**
//!
//! # What actually separates a tap from a dot is its neighbours, not its size
//!
//! A `\cdot` sits *between* its operands. The dot of an `i` sits *over* its stem. They
//! belong to the writing. A toolbar tap sits alone in the margin. So a stroke is only
//! discarded when it is **both** far smaller than the typical stroke **and** further from
//! every other stroke than most of a symbol's width.
//!
//! Both thresholds are measured, not guessed — from real device captures (`--strokes`):
//!
//! | | size (× median) | nearest neighbour (× median) |
//! |---|---|---|
//! | stray taps | 0.02 – 0.08 | 1.12 – 3.43 |
//! | real symbols | 0.85 – 1.96 | 0.38 – 1.15 |
//!
//! Note the overlap in the *second* column: a hand-drawn `∞` sat 1.15 median-widths from
//! anything else — **more isolated than two of the taps**. Isolation alone would have
//! deleted it. Only the conjunction is safe.
//!
//! Distance is measured **point-to-point, not bounding-box to bounding-box**. A selection
//! lasso has a bbox that encloses the entire page, which makes every stray tap look
//! "adjacent" to it. Scale is the **median** stroke, not the largest, for the same reason:
//! one giant lasso stroke (9× the median, in a real capture) destroys a max-based scale,
//! and the median shrugs it off. The median does assume noise is the *minority* — if more
//! than half your strokes are taps, this gives up rather than guess, which is the right
//! failure.

use crate::stroke::Stroke;

/// A stroke this much smaller than the median is a candidate for noise — but only a
/// candidate. `\cdot` lives here too.
const NOISE_MAX_SIZE: f32 = 0.25;
/// …and it is only *discarded* if nothing else comes this close to it.
const NOISE_MIN_GAP: f32 = 0.8;

/// Remove stray taps. Everything else — including every deliberate dot — is kept.
pub fn denoise(strokes: &[Stroke]) -> Vec<Stroke> {
    keep_indices(strokes)
        .into_iter()
        .map(|i| strokes[i].clone())
        .collect()
}

/// Which strokes survive, **as indices into the input**.
///
/// This is the primitive, and the copying version is built on it, because the indices are
/// what callers actually need: `LineSymbol::strokes` refers back to the user's original
/// ink, and the correction UI has to highlight the strokes *they* drew — not offsets into
/// some filtered copy they never saw.
pub fn keep_indices(strokes: &[Stroke]) -> Vec<usize> {
    let live: Vec<usize> = (0..strokes.len())
        .filter(|&i| !strokes[i].points.is_empty())
        .collect();
    if live.len() < 2 {
        // Nothing to be isolated *from*. A lone dot is a `\cdot`, not noise.
        return live;
    }

    let diags: Vec<f32> = live.iter().map(|&i| diagonal(&strokes[i])).collect();
    let scale = median(&mut diags.clone());
    if scale <= f32::EPSILON {
        return live; // degenerate: every stroke is a single point
    }

    live.iter()
        .enumerate()
        .filter(|&(n, &i)| {
            if diags[n] >= NOISE_MAX_SIZE * scale {
                return true; // big enough to be a symbol, whatever it is
            }
            // Small — so: is anything near it? Only the handful of strokes that get this
            // far pay for the point-walk, and a tap has a couple of dozen points, not
            // hundreds.
            let gap = live
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| nearest_point_distance(&strokes[i], &strokes[j]))
                .fold(f32::MAX, f32::min);
            gap < NOISE_MIN_GAP * scale // close to something → it belongs to the writing
        })
        .map(|(_, &i)| i)
        .collect()
}

fn diagonal(s: &Stroke) -> f32 {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in &s.points {
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);
    }
    (x1 - x0).hypot(y1 - y0)
}

fn nearest_point_distance(a: &Stroke, b: &Stroke) -> f32 {
    let mut best = f32::MAX;
    for p in &a.points {
        for q in &b.points {
            best = best.min((p.x - q.x).hypot(p.y - q.y));
        }
    }
    best
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.get(v.len() / 2).copied().unwrap_or(0.0)
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

    /// A symbol-sized blob: a little box `size` across at `(x, y)`.
    fn symbol(x: f32, y: f32, size: f32) -> Stroke {
        stroke(&[
            (x, y),
            (x + size, y),
            (x + size, y + size),
            (x, y + size),
            (x, y),
        ])
    }

    /// A dot: a few samples in one spot, as the digitizer actually delivers one.
    fn dot(x: f32, y: f32) -> Stroke {
        stroke(&[(x, y), (x + 0.003, y + 0.001), (x + 0.001, y + 0.003)])
    }

    #[test]
    fn drops_an_isolated_tap() {
        let ink = [
            symbol(0.30, 0.50, 0.10),
            symbol(0.45, 0.50, 0.10),
            dot(0.05, 0.05), // a tap up in the corner, far from the writing
        ];
        let kept = denoise(&ink);
        assert_eq!(kept.len(), 2, "the isolated tap should be gone");
    }

    /// The whole point. `a · b` — the dot is exactly as small as a tap, and must survive.
    #[test]
    fn keeps_a_cdot_between_its_operands() {
        let ink = [
            symbol(0.30, 0.50, 0.10),
            dot(0.43, 0.55), // the `\cdot`, sitting between them
            symbol(0.48, 0.50, 0.10),
        ];
        let kept = denoise(&ink);
        assert_eq!(
            kept.len(),
            3,
            "a \\cdot is not noise — it belongs to the expression"
        );
    }

    /// …and the dot of an `i`, which hovers *above* its stem rather than beside it.
    #[test]
    fn keeps_the_dot_of_an_i() {
        let ink = [
            symbol(0.30, 0.50, 0.10), // the stem
            dot(0.34, 0.46),          // its dot, just above
            symbol(0.45, 0.50, 0.10),
        ];
        assert_eq!(denoise(&ink).len(), 3);
    }

    /// If *everything* is a dot (`\ldots`), nothing is unusually small — so nothing goes.
    /// This falls out of measuring size against the median rather than an absolute.
    #[test]
    fn keeps_everything_when_the_ink_is_all_dots() {
        let ink = [dot(0.30, 0.50), dot(0.40, 0.50), dot(0.50, 0.50)];
        assert_eq!(
            denoise(&ink).len(),
            3,
            "\\ldots is three dots, not three taps"
        );
    }

    #[test]
    fn a_lone_dot_is_a_symbol_not_noise() {
        assert_eq!(denoise(&[dot(0.5, 0.5)]).len(), 1);
    }

    #[test]
    fn empty_strokes_are_dropped_and_nothing_panics() {
        let ink = [symbol(0.3, 0.5, 0.1), Stroke::new(), symbol(0.5, 0.5, 0.1)];
        assert_eq!(denoise(&ink).len(), 2);
        assert!(denoise(&[]).is_empty());
    }
}
