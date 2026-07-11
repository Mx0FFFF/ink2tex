//! Linear-expression recognizer (M2): segment strokes into symbols left-to-right,
//! classify each with the int8 CNN, and return per-symbol ranked predictions in
//! reading order.
//!
//! No 2-D structure yet — fractions, exponents, and radicals are M3. This handles a
//! single left-to-right line like `2x + 3 = 7`: segmentation (`crate::segment`)
//! gives the symbols in order, and each is fed through the same classifier M1 uses.

use crate::classify::{
    global_features, online_features, rasterize, recognize, Labels, Prediction, Weights,
    ONLINE_POINTS,
};
use crate::error::Result;
use crate::latex::{symbol_command, to_latex};
use crate::segment::segment;
use crate::stroke::{Ink, Stroke};
use crate::structure::{parse as parse_structure, BBox, Symbol as PosSymbol};

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
        let online = online_features(&strokes, ONLINE_POINTS);
        let predictions = recognize(weights, &bitmap, &feats, &online, 32, k)?;
        out.push(LineSymbol {
            strokes: group,
            predictions,
        });
    }
    Ok(out)
}

/// The whole pipeline: **ink → segment → classify → 2-D structure → LaTeX** (M1 +
/// M2 + M3). Each segmented symbol contributes its top-1 label (mapped to a LaTeX
/// token so `structure` can spot `√`, fraction bars, big operators) and its stroke
/// bounding box; `structure::parse` then builds the layout tree and `to_latex`
/// renders it. Falls back to an empty string on empty input.
pub fn recognize_expression(
    ink: &Ink,
    weights: &Weights,
    labels: &Labels,
    k: usize,
) -> Result<String> {
    let mut symbols = Vec::new();
    for ls in recognize_line(ink, weights, k)? {
        let Some(top) = ls.predictions.first() else {
            continue;
        };
        let token = symbol_command(labels.get(top.class).unwrap_or(""));
        if let Some(bbox) = strokes_bbox(ink, &ls.strokes) {
            symbols.push(PosSymbol::new(token, bbox));
        }
    }
    Ok(to_latex(&parse_structure(&symbols)))
}

/// Bounding box (normalized coords) of a group of strokes.
fn strokes_bbox(ink: &Ink, idx: &[usize]) -> Option<BBox> {
    let mut pts = idx
        .iter()
        .filter_map(|&i| ink.strokes.get(i))
        .flat_map(|s| s.points.iter());
    let p0 = pts.next()?;
    let (mut nx, mut ny, mut mx, mut my) = (p0.x, p0.y, p0.x, p0.y);
    for p in pts {
        nx = nx.min(p.x);
        ny = ny.min(p.y);
        mx = mx.max(p.x);
        my = my.max(p.y);
    }
    Some(BBox::new(nx, ny, mx, my))
}
