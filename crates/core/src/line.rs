//! Linear-expression recognizer (M2): segment strokes into symbols left-to-right,
//! classify each with the int8 CNN, and return per-symbol ranked predictions in
//! reading order.
//!
//! No 2-D structure yet — fractions, exponents, and radicals are M3. This handles a
//! single left-to-right line like `2x + 3 = 7`: segmentation (`crate::segment`)
//! gives the symbols in order, and each is fed through the same classifier M1 uses.

use crate::classify::{global_features, rasterize, recognize, Prediction, Weights};
use crate::error::Result;
use crate::segment::segment;
use crate::stroke::{Ink, Stroke};

/// One recognized symbol on the line: the stroke indices it was segmented from, and
/// its ranked top-k candidates (kept ranked — the correction UI needs alternatives).
pub struct LineSymbol {
    pub strokes: Vec<usize>,
    pub predictions: Vec<Prediction>,
}

/// Segment `ink` into symbols and classify each, returning them left-to-right.
pub fn recognize_line(ink: &Ink, weights: &Weights, k: usize) -> Result<Vec<LineSymbol>> {
    let mut out = Vec::new();
    for group in segment(&ink.strokes) {
        let strokes: Vec<Stroke> = group.iter().map(|&i| ink.strokes[i].clone()).collect();
        let bitmap = rasterize(&strokes, 32);
        let feats = global_features(&strokes);
        let predictions = recognize(weights, &bitmap, &feats, 32, k)?;
        out.push(LineSymbol {
            strokes: group,
            predictions,
        });
    }
    Ok(out)
}
