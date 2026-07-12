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
use crate::denoise::keep_indices;
use crate::error::Result;
use crate::latex::{symbol_command, to_latex};
use crate::segment::segment;
use crate::stroke::{Ink, Stroke};
use crate::structure::{parse as parse_structure, BBox, Symbol as PosSymbol};
use crate::vocab::in_expression_vocab;

/// One recognized symbol on the line: the stroke indices it was segmented from, and
/// its ranked top-k candidates (kept ranked — the correction UI needs alternatives).
pub struct LineSymbol {
    pub strokes: Vec<usize>,
    pub predictions: Vec<Prediction>,
}

/// How hard the training-frequency prior is divided out in expression ranking. 1.0 is the
/// textbook value (Menon et al., logit adjustment): score ∝ p(c|ink) / count(c)^TAU, i.e.
/// re-target the classifier from the *training* prior to a uniform one. It is a constant,
/// not a knob — the moment it becomes tunable, someone tunes it to their three favourite
/// test drawings (ask me how I know).
const PRIOR_TAU: f32 = 1.0;

/// If the expression vocabulary captures less than this much raw probability, the model is
/// confidently saying "this is something exotic" — masking would replace a right answer
/// with a hallucinated in-vocab one. Fall back to the unmasked ranking instead.
const VOCAB_MASS_FLOOR: f32 = 0.02;

/// Re-rank a classifier distribution for **expression** context: keep the expression
/// vocabulary, divide out the training prior, renormalize.
///
/// Two corrections in one, and the pairing is load-bearing:
///
/// 1. **The mask** implements DESIGN §4.3, which specifies the expression recognizer over
///    "~120 classes", not the 1,188-class *lookup* space M2 accidentally inherited. A
///    hand-drawn `x` was losing to `\upchi` and `\mathcal{{X}}` — labels that are simply
///    not plausible readings of anything in `2x + 3 = 7`.
/// 2. **The prior division** fixes what the mask alone cannot: Detexify's frequencies are
///    those of a *lookup* service (`\chi`: 958 samples; `x`: 59 — nobody looks up how to
///    type x), which is close to the inverse of what a pen writes. Dividing by
///    `count^TAU` re-targets to a uniform prior. Without the mask this correction is
///    dangerous — it catapults every 50-sample exotic class; the mask removes them first.
///
/// Falls back to the unmasked ranking when the vocabulary captures almost no probability
/// mass (see `VOCAB_MASS_FLOOR`). Returned probabilities are renormalized posteriors under
/// the uniform prior — ranked honestly, but not calibrated confidences.
pub fn expression_rank(
    preds: &[Prediction],
    labels: &Labels,
    counts: Option<&[u32]>,
    k: usize,
) -> Vec<Prediction> {
    let kept: Vec<&Prediction> = preds
        .iter()
        .filter(|p| labels.get(p.class).is_some_and(in_expression_vocab))
        .collect();
    let mass: f32 = kept.iter().map(|p| p.prob).sum();
    if kept.is_empty() || mass < VOCAB_MASS_FLOOR {
        return preds.iter().take(k).cloned().collect(); // confidently exotic: don't mask
    }

    let mut adjusted: Vec<Prediction> = kept
        .into_iter()
        .map(|p| {
            let prior = counts
                .and_then(|c| c.get(p.class))
                .map_or(1.0, |&n| (n.max(1) as f32).powf(PRIOR_TAU));
            Prediction {
                class: p.class,
                prob: p.prob / prior,
            }
        })
        .collect();
    let total: f32 = adjusted.iter().map(|p| p.prob).sum();
    if total > 0.0 {
        for p in &mut adjusted {
            p.prob /= total;
        }
    }
    adjusted.sort_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    adjusted.truncate(k);
    adjusted
}

/// Segment `ink` into symbols and classify each, returning them left-to-right.
///
/// This is the **expression** path, and it ranks over the expression vocabulary (see
/// `expression_rank`) — the M1 lookup path (`--recognize`) deliberately does not.
/// `counts` is the per-class training-sample count aligned with `labels` (the
/// `.counts.txt` the trainer writes beside the labels file); `None` skips the prior
/// correction but keeps the mask.
///
/// Returns the **oriented ink** alongside the symbols, and every downstream consumer
/// must use it. The first landscape capture taught why the hard way: orientation was
/// once internal to this function, classification saw rotated glyphs (right labels!),
/// but the structure parse computed its bboxes from the caller's original vertical ink —
/// and laid perfectly-recognized symbols out as `2\frac{{>_{{=}}}}{{x^{{+}}}}`. Stroke
/// *indices* survive rotation; *coordinates* do not.
pub fn recognize_line(
    ink: &Ink,
    weights: &Weights,
    labels: &Labels,
    counts: Option<&[u32]>,
    k: usize,
) -> Result<(Ink, Vec<LineSymbol>)> {
    // Landscape grip first: if the symbol line runs vertically, the tablet was held
    // sideways and every glyph is rotated 90° — rotate upright before anything else
    // looks at the ink (see `orient`). Portrait ink short-circuits to a clone.
    let ink = &crate::orient::auto_orient(ink, weights)?;

    // Stray taps first: we read the pen below xochitl, so a tap on its toolbar arrives as
    // a tiny stroke, and without this `segment` calls it a symbol and `structure` makes it
    // a superscript. `keep_indices` (not `denoise`) because `LineSymbol::strokes` must
    // keep pointing at the *user's* strokes, not at a filtered copy they never saw.
    let keep = keep_indices(&ink.strokes);
    let kept: Vec<Stroke> = keep.iter().map(|&i| ink.strokes[i].clone()).collect();

    let mut out = Vec::new();
    for group in segment(&kept) {
        let strokes: Vec<Stroke> = group.iter().map(|&i| kept[i].clone()).collect();
        let bitmap = rasterize(&strokes, 32);
        let feats = global_features(&strokes);
        let online = online_features(&strokes, ONLINE_POINTS);
        // Ask the classifier for a DEEP list before masking: on real ink the right
        // everyday token has been observed as far down as rank 15 of the unmasked
        // ranking. Mask a top-5 and it is already gone.
        let deep = recognize(weights, &bitmap, &feats, &online, 32, 40)?;
        let predictions = expression_rank(&deep, labels, counts, k);
        out.push(LineSymbol {
            strokes: group.iter().map(|&i| keep[i]).collect(), // back to original indices
            predictions,
        });
    }
    Ok((ink.clone(), out))
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
    counts: Option<&[u32]>,
    k: usize,
) -> Result<String> {
    let (ink, line) = recognize_line(ink, weights, labels, counts, k)?;
    let ink = &ink; // the ORIENTED ink — bboxes must come from the same frame as the labels
    let mut symbols = Vec::new();
    for ls in line {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Labels {
        Labels::from_lines("x\nlatex:latex2e:chi\nlatex:stmaryrd:Lbag\nlatex:latex2e:rightarrow\n+")
    }
    fn p(class: usize, prob: f32) -> Prediction {
        Prediction { class, prob }
    }

    #[test]
    fn masks_tokens_that_are_not_expression_vocabulary() {
        // \Lbag out-scores everything — and is not a plausible reading of anything in
        // `2x + 3 = 7`, so it must not survive into the expression ranking.
        let preds = [p(2, 0.60), p(1, 0.25), p(0, 0.10)];
        let out = expression_rank(&preds, &labels(), None, 5);
        assert!(out.iter().all(|q| q.class != 2), "\\Lbag survived the mask");
        assert_eq!(out[0].class, 1, "ranking among survivors must be preserved");
    }

    #[test]
    fn prior_division_lets_the_everyday_token_win() {
        // The measured case: chi 958 training samples, x 59. Raw 18% vs 3% — chi wins.
        // Divided by the training prior, x wins: nobody looks up how to type x, so its
        // lookup-corpus count says nothing about how often a pen writes it.
        let preds = [p(1, 0.18), p(0, 0.03)];
        let mut counts = vec![0u32; 5];
        counts[0] = 59;
        counts[1] = 958;
        let out = expression_rank(&preds, &labels(), Some(&counts), 5);
        assert_eq!(
            out[0].class, 0,
            "x must out-rank chi once the prior is divided out"
        );
        let total: f32 = out.iter().map(|q| q.prob).sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "renormalized probs must sum to 1"
        );
    }

    #[test]
    fn confidently_exotic_ink_is_not_masked_into_a_hallucination() {
        // 97% \Lbag, a whisper of everything else: the model is sure this is something
        // exotic. Masking would crown a 1% token as "the answer". Fall back instead.
        let preds = [p(2, 0.985), p(0, 0.008), p(1, 0.007)];
        let out = expression_rank(&preds, &labels(), None, 2);
        assert_eq!(out[0].class, 2, "the confident exotic answer must survive");
        assert_eq!(out.len(), 2, "fallback still honours k");
    }

    #[test]
    fn missing_counts_still_masks_but_skips_the_prior() {
        let preds = [p(1, 0.18), p(0, 0.03), p(2, 0.5)];
        let out = expression_rank(&preds, &labels(), None, 5);
        assert_eq!(
            out[0].class, 1,
            "without counts, raw ranking holds among vocab"
        );
        assert!(out.iter().all(|q| q.class != 2));
    }
}
