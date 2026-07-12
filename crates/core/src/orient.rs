//! Landscape detection — because people hold the tablet sideways to write equations.
//!
//! The digitizer's frame does not rotate with the user's grip: the very first live
//! equation test (`2x + 3 = 7`, 2026-07-13) arrived rotated 90°, every glyph classified
//! as garbage, and the ink had to be counter-rotated by hand. That grip wasn't a fluke —
//! landscape is the *natural* way to write a long expression on a portrait tablet.
//!
//! ## How detection works, and why the classifier gets the last word
//!
//! Writing runs along a line. If the segmented symbols spread mostly **vertically**, the
//! ink was written landscape — but that alone cannot say *which* landscape: rotated
//! clockwise or counter-clockwise, the line becomes horizontal either way. Geometry has
//! no answer; **glyphs do**. Both candidate rotations are built, a few symbols from each
//! are classified, and whichever orientation the model recognizes with more confidence
//! wins. Upside-down text loses that vote exactly like sideways text does — a `3` rotated
//! 180° is not a `3` to the model.
//!
//! Portrait ink never pays for any of this: a horizontal line short-circuits to a no-op.
//! The cost when rotation *is* needed is 2 orientations × ≤4 symbols ≈ a dozen extra
//! forward passes, ~200 ms on the device — once per expression, not per stroke.

use crate::classify::{global_features, online_features, rasterize, recognize, Weights};
use crate::denoise::keep_indices;
use crate::error::Result;
use crate::segment::segment;
use crate::stroke::{Ink, Point, Stroke};

/// How many symbols vote in the orientation ballot. More adds latency, not accuracy:
/// the two candidate rotations differ by 180°, and even one clean glyph separates them.
const BALLOT: usize = 4;

/// Rotate 90° clockwise in normalized coords (y down): `(x, y) → (1−y, x)`.
pub fn rotate_cw(ink: &Ink) -> Ink {
    transform(ink, |x, y| (1.0 - y, x))
}

/// Rotate 90° counter-clockwise: `(x, y) → (y, 1−x)`.
pub fn rotate_ccw(ink: &Ink) -> Ink {
    transform(ink, |x, y| (y, 1.0 - x))
}

fn transform(ink: &Ink, f: impl Fn(f32, f32) -> (f32, f32)) -> Ink {
    Ink {
        source_width: ink.source_height,
        source_height: ink.source_width,
        strokes: ink
            .strokes
            .iter()
            .map(|s| Stroke {
                points: s
                    .points
                    .iter()
                    .map(|p| {
                        let (x, y) = f(p.x, p.y);
                        Point::new(x, y, p.pressure, p.tilt_x, p.tilt_y, p.t_us)
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Does the symbol line run vertically? (Pure geometry — the cheap first gate.)
///
/// Measured over segmented symbol centers, not strokes: a single tall glyph like `∫` has
/// vertical *stroke* spread but is one symbol; three symbols stacked is a line. Requires
/// at least 3 symbols — below that "line direction" is not meaningful, and single-symbol
/// lookup (M1) must never be second-guessed.
pub fn line_is_vertical(ink: &Ink) -> bool {
    let keep = keep_indices(&ink.strokes);
    let kept: Vec<Stroke> = keep.iter().map(|&i| ink.strokes[i].clone()).collect();
    let groups = segment(&kept);
    if groups.len() < 3 {
        return false;
    }
    let centers: Vec<(f32, f32)> = groups
        .iter()
        .map(|g| {
            let pts = g.iter().flat_map(|&i| kept[i].points.iter());
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for p in pts {
                x0 = x0.min(p.x);
                y0 = y0.min(p.y);
                x1 = x1.max(p.x);
                y1 = y1.max(p.y);
            }
            ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        })
        .collect();
    let spread = |get: fn(&(f32, f32)) -> f32| {
        let lo = centers.iter().map(get).fold(f32::MAX, f32::min);
        let hi = centers.iter().map(get).fold(f32::MIN, f32::max);
        hi - lo
    };
    // Comfortably vertical, not merely square: a 2-D construct (fraction, matrix-ish
    // blob) must not trigger a rotation.
    spread(|c| c.1) > 1.6 * spread(|c| c.0)
}

/// Mean top-1 confidence over the first `BALLOT` symbols — the orientation ballot.
fn confidence(ink: &Ink, weights: &Weights) -> Result<f32> {
    let keep = keep_indices(&ink.strokes);
    let kept: Vec<Stroke> = keep.iter().map(|&i| ink.strokes[i].clone()).collect();
    let groups = segment(&kept);
    let (mut total, mut n) = (0.0f32, 0usize);
    for g in groups.iter().take(BALLOT) {
        let strokes: Vec<Stroke> = g.iter().map(|&i| kept[i].clone()).collect();
        let preds = recognize(
            weights,
            &rasterize(&strokes, 32),
            &global_features(&strokes),
            &online_features(&strokes, crate::classify::ONLINE_POINTS),
            32,
            1,
        )?;
        total += preds.first().map_or(0.0, |p| p.prob);
        n += 1;
    }
    Ok(if n == 0 { 0.0 } else { total / n as f32 })
}

/// Give back ink the classifier can read: portrait ink unchanged, landscape ink rotated
/// to whichever upright orientation the model recognizes with more confidence.
///
/// The **original orientation competes in the ballot**, and wins ties. That is not a
/// nicety: an isolated fraction is *geometrically indistinguishable* from a vertical
/// three-symbol line — centers stacked, no horizontal spread — so any purely geometric
/// rule would flip every lone fraction on the page. Geometry here only decides whether a
/// ballot is held at all; upright ink then defends itself by classifying well (`a`, a
/// bar, `b` — high confidence), while genuinely sideways ink classifies as garbage and
/// loses to its own rotation.
pub fn auto_orient(ink: &Ink, weights: &Weights) -> Result<Ink> {
    if !line_is_vertical(ink) {
        return Ok(ink.clone());
    }
    let as_is = confidence(ink, weights)?;
    let cw = rotate_cw(ink);
    let ccw = rotate_ccw(ink);
    let (score_cw, score_ccw) = (confidence(&cw, weights)?, confidence(&ccw, weights)?);
    Ok(if as_is >= score_cw && as_is >= score_ccw {
        ink.clone()
    } else if score_cw >= score_ccw {
        cw
    } else {
        ccw
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stroke(pts: &[(f32, f32)]) -> Stroke {
        Stroke {
            points: pts
                .iter()
                .map(|&(x, y)| Point::new(x, y, 1.0, 0.0, 0.0, 0))
                .collect(),
        }
    }
    fn glyph(x: f32, y: f32) -> Stroke {
        stroke(&[(x, y), (x + 0.05, y), (x + 0.05, y + 0.07), (x, y + 0.07)])
    }

    #[test]
    fn rotations_are_inverse_of_each_other() {
        let ink = Ink {
            source_width: 1.0,
            source_height: 1.0,
            strokes: vec![stroke(&[(0.2, 0.7), (0.4, 0.1)])],
        };
        let back = rotate_ccw(&rotate_cw(&ink));
        for (a, b) in ink.strokes[0].points.iter().zip(&back.strokes[0].points) {
            assert!((a.x - b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6);
        }
    }

    #[test]
    fn a_horizontal_line_is_not_vertical() {
        let ink = Ink {
            source_width: 1.0,
            source_height: 1.0,
            strokes: vec![glyph(0.2, 0.5), glyph(0.4, 0.51), glyph(0.6, 0.5)],
        };
        assert!(!line_is_vertical(&ink));
    }

    #[test]
    fn a_vertical_line_of_symbols_is_vertical() {
        let ink = Ink {
            source_width: 1.0,
            source_height: 1.0,
            strokes: vec![glyph(0.5, 0.2), glyph(0.51, 0.4), glyph(0.5, 0.6)],
        };
        assert!(line_is_vertical(&ink));
    }

    /// One or two symbols carry no line direction — M1 single-symbol lookup must never
    /// be second-guessed into a rotation.
    #[test]
    fn too_few_symbols_never_trigger_rotation() {
        let ink = Ink {
            source_width: 1.0,
            source_height: 1.0,
            strokes: vec![glyph(0.5, 0.2), glyph(0.5, 0.6)],
        };
        assert!(!line_is_vertical(&ink));
    }

    /// An isolated fraction IS a vertical line, geometrically — stacked centers, no
    /// horizontal spread. No geometry can tell them apart, which is exactly why
    /// `auto_orient` holds a ballot instead of rotating on geometry alone: the upright
    /// fraction competes as-is, classifies confidently, and wins. This test pins the
    /// trigger; the ballot's verdict is pinned end-to-end on real ink in the desktop
    /// corpus tests (a rotated equation rotates, horizontal captures do not move).
    #[test]
    fn a_fraction_shape_triggers_the_ballot_not_a_blind_rotation() {
        let ink = Ink {
            source_width: 1.0,
            source_height: 1.0,
            strokes: vec![
                glyph(0.45, 0.38),                     // numerator
                stroke(&[(0.40, 0.50), (0.56, 0.50)]), // bar
                glyph(0.45, 0.56),                     // denominator
            ],
        };
        assert!(
            line_is_vertical(&ink),
            "the trigger fires; the ballot decides"
        );
    }
}
