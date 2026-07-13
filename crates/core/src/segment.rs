//! Stroke → symbol clustering. Groups strokes into individual symbols by **2-D
//! spatial proximity** (a symbol is a compact cluster of nearby strokes), then
//! orders the symbols left-to-right.
//!
//! Unlike a pure left-to-right split, proximity clustering keeps a fraction's
//! numerator, bar, and denominator as *separate* symbols (they are vertically
//! stacked at the same x), so the 2-D structure stage (`crate::structure`) can
//! re-assemble them as `\frac`. It also keeps the crossing strokes of an `x` or the
//! dot-and-stem of an `i` together.
//!
//! v1 uses a single proximity threshold scaled by the median stroke size. A tight
//! stack of horizontal bars (`=`, `≡`, `Ξ`) may over-split — distinguishing that
//! from a fraction needs the joint segmentation/parse lattice of DESIGN §4.2, which
//! is later work. The delayed-stroke problem is likewise deferred.

use crate::stroke::Stroke;

/// Strokes whose x-projections are disjoint get only this fraction of the merge
/// threshold — side-by-side strokes must nearly touch to be one symbol (see the
/// union loop in `segment` for the measured rationale).
const SIDE_BY_SIDE_TOUCH: f32 = 0.35;

/// `[min_x, min_y, max_x, max_y]`.
type Bbox = [f32; 4];

fn bbox(s: &Stroke) -> Option<Bbox> {
    let mut it = s.points.iter();
    let p0 = it.next()?;
    let mut b = [p0.x, p0.y, p0.x, p0.y];
    for p in it {
        b[0] = b[0].min(p.x);
        b[1] = b[1].min(p.y);
        b[2] = b[2].max(p.x);
        b[3] = b[3].max(p.y);
    }
    Some(b)
}

/// Chebyshev gap between two bboxes: 0 when they overlap along a dimension.
///
/// This is a **lower bound** on the distance between the strokes' actual ink, never more —
/// which is what makes it sound as a cheap pre-filter for `ink_gap` below, and useless as a
/// clustering rule on its own.
fn gap(a: &Bbox, b: &Bbox) -> f32 {
    let dx = (a[0] - b[2]).max(b[0] - a[2]).max(0.0);
    let dy = (a[1] - b[3]).max(b[1] - a[3]).max(0.0);
    dx.max(dy)
}

/// The real thing: the closest approach between two strokes' **ink**.
///
/// ## Why bounding boxes are not enough — the `√` bug
/// Clustering on bbox gap alone merges any stroke whose box *contains* another's, because
/// their boxes intersect and the gap is 0 — at **any** threshold. That is not a corner case:
/// it is how a square root is written. A hand-drawn `√x+1` puts the tick to the left of the
/// contents and the overbar above them, so the radical's box encloses `x`, `+` and `1`, and
/// all six strokes collapsed into one "symbol" (the classifier, shown a whole expression as
/// one glyph, answered `\mathscr{F}` at 13.9%). Tall parentheses do the same thing.
///
/// The radical's *ink*, though, is nowhere near the contents — measured on that very
/// capture, its nearest point was 0.0298 away against a 0.0143 merge threshold. So compare
/// ink to ink, and the radical separates while the two crossing strokes of an `x` (0.0007
/// apart) still merge, which is exactly what we want.
///
/// It compares **polyline to polyline**, not sample point to sample point. Points would be a
/// *nearly* correct shortcut — real ink is dense — but "nearly" is doing a lot of work
/// there: two strokes that genuinely cross can have every sample far from every other, and
/// an `x` dashed off in a few samples would shatter into two symbols.
///
/// It is a **predicate, not a distance**, because the caller only ever asks "close enough to
/// merge?" — so it can stop at the first hit instead of finding the true minimum.
///
/// **The per-segment pruning is not optional.** The per-*stroke* bbox test in `segment` is
/// worthless in exactly the case this function exists for: when one box contains another it
/// reports 0, and all O(segments²) pairs get walked. Measured on real captures, the
/// unpruned walk cost 18 ms for a `√x+1` and **107 ms** for a page with a lasso on it — on
/// x86, against a 50 ms budget on a CPU several times slower. So each *segment* pair is
/// rejected by its own bounding box first: a few flops instead of ~40.
fn ink_within(a: &Stroke, b: &Stroke, thresh: f32) -> bool {
    // Degenerate: a stroke of one sample has no segment to walk.
    if a.points.len() < 2 || b.points.len() < 2 {
        return a.points.iter().any(|p| {
            b.points
                .iter()
                .any(|q| (p.x - q.x).hypot(p.y - q.y) < thresh)
        });
    }

    for pq in a.points.windows(2) {
        let (ax0, ax1) = (pq[0].x.min(pq[1].x) - thresh, pq[0].x.max(pq[1].x) + thresh);
        let (ay0, ay1) = (pq[0].y.min(pq[1].y) - thresh, pq[0].y.max(pq[1].y) + thresh);
        for rs in b.points.windows(2) {
            // Cheap reject: if b's segment box misses a's segment box grown by `thresh`,
            // nothing on it can be within `thresh`. This is what makes the walk affordable.
            if rs[0].x.min(rs[1].x) > ax1
                || rs[0].x.max(rs[1].x) < ax0
                || rs[0].y.min(rs[1].y) > ay1
                || rs[0].y.max(rs[1].y) < ay0
            {
                continue;
            }
            if seg_seg_dist(
                (pq[0].x, pq[0].y),
                (pq[1].x, pq[1].y),
                (rs[0].x, rs[0].y),
                (rs[1].x, rs[1].y),
            ) < thresh
            {
                return true;
            }
        }
    }
    false
}

type Pt = (f32, f32);

/// Distance between two line segments — **0 if they intersect**, which is the whole point:
/// that is what makes the crossing strokes of an `x` one symbol.
fn seg_seg_dist(p: Pt, p2: Pt, q: Pt, q2: Pt) -> f32 {
    let (r, s) = ((p2.0 - p.0, p2.1 - p.1), (q2.0 - q.0, q2.1 - q.1));
    let denom = r.0 * s.1 - r.1 * s.0;
    let qp = (q.0 - p.0, q.1 - p.1);
    if denom.abs() > 1e-12 {
        let t = (qp.0 * s.1 - qp.1 * s.0) / denom;
        let u = (qp.0 * r.1 - qp.1 * r.0) / denom;
        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
            return 0.0; // proper intersection
        }
    }
    // Parallel, or they miss: the closest approach is then from some endpoint.
    point_seg_dist(p, q, q2)
        .min(point_seg_dist(p2, q, q2))
        .min(point_seg_dist(q, p, p2))
        .min(point_seg_dist(q2, p, p2))
}

fn point_seg_dist(p: Pt, a: Pt, b: Pt) -> f32 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let len2 = ab.0 * ab.0 + ab.1 * ab.1;
    if len2 <= f32::EPSILON {
        return (p.0 - a.0).hypot(p.1 - a.1);
    }
    let t = (((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / len2).clamp(0.0, 1.0);
    (p.0 - (a.0 + t * ab.0)).hypot(p.1 - (a.1 + t * ab.1))
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
    }
    x
}

/// Cluster strokes into symbols by proximity, ordered left-to-right. Returns groups
/// of indices into `strokes`; empty strokes are dropped.
pub fn segment(strokes: &[Stroke]) -> Vec<Vec<usize>> {
    let items: Vec<(usize, Bbox)> = strokes
        .iter()
        .enumerate()
        .filter_map(|(i, s)| bbox(s).map(|b| (i, b)))
        .collect();
    if items.is_empty() {
        return Vec::new();
    }

    // Merge threshold = a fraction of the median stroke size (its larger side) — taken
    // over COMPACT strokes only. Line-like strokes (tall parentheses, bars, slashes) have
    // a diagonal that measures their *length*, not the writing's symbol scale, and they
    // can be a large share of the population: in a real-glyph `(x+1)`, the parens and the
    // flagged `1` pushed the median from ~0.08 to 0.126 and the threshold to within 12%
    // of a normal inter-symbol gap — one slightly-tight writer away from `x+` fusing into
    // a single blob (which then classifies as `\aleph`, of all things). Compact strokes
    // measure the x-height; the exotic shapes get to *use* the threshold, not set it.
    let mut sizes: Vec<f32> = items
        .iter()
        .filter(|(_, b)| {
            let (w, h) = (b[2] - b[0], b[3] - b[1]);
            w.max(h) / w.min(h).max(1e-6) <= 2.5
        })
        .map(|(_, b)| (b[2] - b[0]).max(b[3] - b[1]))
        .collect();
    if sizes.is_empty() {
        // Nothing compact (a lone `=`, a page of slashes): fall back to every stroke.
        sizes = items
            .iter()
            .map(|(_, b)| (b[2] - b[0]).max(b[3] - b[1]))
            .collect();
    }
    sizes.sort_by(|a, c| a.partial_cmp(c).unwrap_or(core::cmp::Ordering::Equal));
    let thresh = 0.25 * sizes[sizes.len() / 2].max(1e-6);

    // Union strokes whose *ink* comes within `thresh`. The bbox test is only a cheap
    // rejection: it is a lower bound, so "boxes far apart" already means "ink far apart",
    // and we skip the point walk. It must never be the thing that decides a merge — see
    // `ink_gap` for the `√` that six strokes of it collapsed into one symbol.
    let n = items.len();
    let mut parent: Vec<usize> = (0..n).collect();
    for a in 0..n {
        for b in (a + 1)..n {
            // Strokes that are DISJOINT in x must nearly touch to merge; only
            // x-overlapping strokes get the full threshold. This is why handwriting
            // is segmentable at all: symbols advance horizontally, so a genuine
            // multi-stroke symbol overlaps itself in x (x's crossing, ='s stacked
            // bars, t's crossbar, i's dot), while two neighbouring symbols do not.
            // Measured on the first live subtraction: the `5x` product was written
            // 0.0057 apart — inside the 0.0073 threshold, so it fused and read as
            // `\ast` — with x-projection gaps of +0.002/+0.003; every real
            // multi-stroke symbol on the same page overlapped by −0.015 or more.
            // K-style arms that touch their stem (gap ≈ 0.0005) stay well inside
            // the reduced allowance.
            let x_gap = items[a].1[0].max(items[b].1[0]) - items[a].1[2].min(items[b].1[2]);
            let allow = if x_gap > 0.0 {
                SIDE_BY_SIDE_TOUCH * thresh
            } else {
                thresh
            };
            if gap(&items[a].1, &items[b].1) >= allow {
                continue; // boxes too far ⇒ ink too far
            }
            if ink_within(&strokes[items[a].0], &strokes[items[b].0], allow) {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent[ra] = rb;
            }
        }
    }

    // Collect components, tracking each group's leftmost x for ordering.
    let mut roots: Vec<usize> = Vec::new();
    let mut groups: Vec<(f32, Vec<usize>)> = Vec::new();
    for (a, (orig, bbox)) in items.iter().enumerate() {
        let r = find(&mut parent, a);
        match roots.iter().position(|&x| x == r) {
            Some(g) => {
                groups[g].0 = groups[g].0.min(bbox[0]);
                groups[g].1.push(*orig);
            }
            None => {
                roots.push(r);
                groups.push((bbox[0], vec![*orig]));
            }
        }
    }
    groups.sort_by(|g, h| g.0.partial_cmp(&h.0).unwrap_or(core::cmp::Ordering::Equal));
    merge_stacked_bars(groups.into_iter().map(|(_, idx)| idx).collect(), strokes)
}

/// Merge the two bars of an `=` into one symbol group.
///
/// Proximity clustering can never do this on its own: the bars of a handwritten `=` sit
/// 0.7–0.9 bar-widths apart (measured on device ink), while the merge threshold is a
/// quarter of the median stroke size — so `=` always splits into two groups, each
/// classifies as `-`, and `2x + 3 = 7` reads `2x + 3 - - 7`. The classifier has real
/// device-collected `=` samples now; it just never gets shown the whole glyph.
///
/// Two groups merge only when they are BOTH single wide bars, of similar width, strongly
/// x-overlapping, and close vertically. Each guard kills a real false positive:
/// - *similar width* keeps a fraction bar from pairing with a `-` inside its numerator
///   (`\frac{{a-b}}{{c}}` — the fraction bar is 2-3× wider);
/// - *strong x-overlap* keeps neighbouring minus signs on one baseline apart
///   (`a - b - c` — their bars never overlap in x);
/// - *bar shape* keeps letters and digits out entirely.
///
/// `≡` (three bars) merges its closest two and leaves one `-` — the honest v1 limit,
/// same family as DESIGN §4.2's stacked-bar caveat, and `≡` is a single trained glyph
/// when drawn compactly anyway.
fn merge_stacked_bars(groups: Vec<Vec<usize>>, strokes: &[Stroke]) -> Vec<Vec<usize>> {
    let group_bbox = |g: &[usize]| -> Option<Bbox> {
        let mut it = g.iter().filter_map(|&i| bbox(&strokes[i]));
        let first = it.next()?;
        Some(it.fold(first, |a, b| {
            [
                a[0].min(b[0]),
                a[1].min(b[1]),
                a[2].max(b[2]),
                a[3].max(b[3]),
            ]
        }))
    };
    // A bar, allowing for slant. The first live equation test failed here: the user's
    // `=` was written ~30° downhill, its bars' axis-aligned aspect fell below any sane
    // threshold, and the merge never fired — one bar then classified as `\setminus`
    // (which is exactly what a slanted bar is, to an upright-trained model). So measure
    // elongation in the stroke's OWN frame: take the endpoint direction, require it
    // within ±40° of horizontal, require the path to be straight (no `(`-curves), and
    // require the rotated-frame aspect to clear the bar threshold.
    let is_bar = |g: &[usize]| -> Option<(Bbox, f32)> {
        if g.len() != 1 {
            return None;
        }
        let pts = &strokes[g[0]].points;
        let (p0, p1) = (pts.first()?, pts.last()?);
        let (dx, dy) = (p1.x - p0.x, p1.y - p0.y);
        let len = dx.hypot(dy);
        let theta = dy.atan2(dx.abs()); // sign-folded: direction within ±90°
        if theta.abs() > 40f32.to_radians() {
            return None; // too steep to be an = bar (that is a `/`, `|` or `1`)
        }
        // Straightness: a curve's path is much longer than its endpoint span.
        let path: f32 = pts
            .windows(2)
            .map(|w| (w[1].x - w[0].x).hypot(w[1].y - w[0].y))
            .sum();
        if len < 1e-6 || path / len > 1.35 {
            return None;
        }
        // Aspect in the bar's own frame: thickness = max perpendicular deviation.
        let (ux, uy) = (dx / len, dy / len);
        let thick = pts
            .iter()
            .map(|p| ((p.x - p0.x) * uy - (p.y - p0.y) * ux).abs())
            .fold(0.0f32, f32::max);
        (len / thick.max(1e-6) > 2.5).then(|| (group_bbox(g).unwrap(), theta))
    };

    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut i = 0;
    while i < groups.len() {
        let merged = (|| {
            let (a, ta) = is_bar(&groups[i])?;
            // Left-to-right ordering puts the partner bar adjacent (same leftmost x).
            let (b, tb) = is_bar(groups.get(i + 1)?)?;
            if (ta - tb).abs() > 25f32.to_radians() {
                return None; // the two bars of one `=` slant TOGETHER
            }
            let (wa, wb) = (a[2] - a[0], b[2] - b[0]);
            let overlap = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
            let vgap = ((a[1] + a[3]) / 2.0 - (b[1] + b[3]) / 2.0).abs();
            if wa.max(wb) / wa.min(wb).max(1e-6) >= 2.4
                || overlap <= 0.6 * wa.min(wb)
                || vgap >= 1.2 * wa.max(wb)
            {
                return None;
            }
            // The corridor test — what actually separates `=` from a fraction bar
            // paired with a `-` in its numerator. A real `=` has EMPTY SPACE beyond
            // both bars in their shared x-corridor; a fraction always has content
            // there (numerator above, denominator below). This is also what lets the
            // width guard relax from 1.8 to 2.4: a live `=` was drawn with its top
            // bar at 1.9× ratio, split into two minus signs, and — both being
            // hairline "bases" — parsed as `-^{-}`.
            let (upper, lower) = if a[1] + a[3] <= b[1] + b[3] { (a, b) } else { (b, a) };
            let (cx0, cx1) = (a[0].max(b[0]), a[2].min(b[2]));
            let reach = 2.0 * vgap;
            let crowded = groups.iter().enumerate().any(|(k, g)| {
                k != i && k != i + 1
                    && group_bbox(g).is_some_and(|c| {
                        let ccx = (c[0] + c[2]) / 2.0;
                        let ccy = (c[1] + c[3]) / 2.0;
                        ccx > cx0
                            && ccx < cx1
                            && ((ccy > upper[1] - reach && ccy < upper[1])
                                || (ccy > lower[3] && ccy < lower[3] + reach))
                    })
            });
            (!crowded).then(|| {
                let mut g = groups[i].clone();
                g.extend(&groups[i + 1]);
                g
            })
        })();
        match merged {
            Some(g) => {
                out.push(g);
                i += 2;
            }
            None => {
                out.push(groups[i].clone());
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    // --- what an *enveloping* stroke does to proximity clustering ------------------
    //
    // These are the cases that motivated `ink_gap`. Clustering on bounding boxes merges
    // any stroke whose box contains another's — at any threshold — and that is not a
    // corner case, it is how `√` is written.

    /// The bug, in the notation that actually triggers it. A `√` drawn over its contents
    /// encloses them; on a real capture this collapsed all six strokes into one "symbol"
    /// and the classifier answered `\mathscr{F}`. It must stay four.
    #[test]
    fn a_radical_does_not_swallow_the_expression_underneath_it() {
        let radical = stroke(&[(0.30, 0.50), (0.33, 0.58), (0.36, 0.38), (0.62, 0.38)]);
        let g = segment(&[
            radical,
            glyph(0.39, 0.42, 0.05, 0.07), // x
            glyph(0.46, 0.42, 0.05, 0.07), // +
            glyph(0.53, 0.42, 0.04, 0.07), // 1
        ]);
        assert_eq!(g.len(), 4, "√ merged with its contents: {g:?}");
    }

    /// …but strokes that genuinely *touch* must still merge, or every `x` and `+` shatters.
    #[test]
    fn crossing_strokes_of_one_symbol_still_merge() {
        let g = segment(&[
            stroke(&[(0.40, 0.40), (0.46, 0.48)]), // the two strokes of an `x`
            stroke(&[(0.46, 0.40), (0.40, 0.48)]),
        ]);
        assert_eq!(g.len(), 1, "an `x` is one symbol, not two: {g:?}");
    }

    /// A fraction must stay in three pieces so `structure` can rebuild it as \frac.
    #[test]
    fn a_fraction_stays_numerator_bar_denominator() {
        let g = segment(&[
            glyph(0.40, 0.30, 0.06, 0.08),
            stroke(&[(0.36, 0.42), (0.52, 0.42)]),
            glyph(0.40, 0.47, 0.06, 0.08),
        ]);
        assert_eq!(g.len(), 3, "fraction collapsed: {g:?}");
    }

    /// The two bars of a handwritten `=` sit ~0.8 bar-widths apart — far beyond the
    /// proximity threshold — so without the stacked-bar merge every `=` splits into two
    /// `-` symbols and `2x+3=7` reads `2x+3--7`. Geometry from real device ink.
    #[test]
    fn an_equals_sign_is_one_symbol_not_two_minuses() {
        let g = segment(&[
            stroke(&[(0.40, 0.470), (0.46, 0.472)]), // top bar
            stroke(&[(0.40, 0.516), (0.46, 0.514)]), // bottom bar, ~0.75 widths below
        ]);
        assert_eq!(g.len(), 1, "= split into {g:?}");
    }

    /// …but three minus signs on one baseline must NOT chain-merge: their bars never
    /// overlap in x, which is the guard that separates `a - b - c` from `=`.
    #[test]
    fn minus_signs_on_a_baseline_stay_separate() {
        let g = segment(&[
            stroke(&[(0.20, 0.50), (0.26, 0.50)]),
            stroke(&[(0.40, 0.50), (0.46, 0.50)]),
            stroke(&[(0.60, 0.50), (0.66, 0.50)]),
        ]);
        assert_eq!(g.len(), 3, "minuses merged: {g:?}");
    }

    /// …and a fraction bar must not swallow a minus inside the numerator: the widths
    /// differ by >1.8x, which is what that guard is for. (`\frac{a-b}{c}` territory.)
    #[test]
    fn a_fraction_bar_does_not_pair_with_a_numerator_minus() {
        let g = segment(&[
            stroke(&[(0.42, 0.42), (0.48, 0.42)]), // '-' in the numerator, narrow
            stroke(&[(0.30, 0.50), (0.62, 0.50)]), // fraction bar, ~5x wider
        ]);
        assert_eq!(g.len(), 2, "fraction bar swallowed the minus: {g:?}");
    }

    /// The first live equation failed here: an `=` written ~17° downhill has bars whose
    /// axis-aligned aspect is mediocre, and the old horizontal-only bar test let them
    /// through as two separate `\setminus`-ish strokes. The bar test measures elongation
    /// in the stroke's own frame now.
    #[test]
    fn a_slanted_equals_still_merges() {
        let g = segment(&[
            stroke(&[(0.40, 0.470), (0.46, 0.488)]), // top bar, sloping down-right
            stroke(&[(0.40, 0.516), (0.46, 0.534)]), // bottom bar, same slant
        ]);
        assert_eq!(g.len(), 1, "slanted = split: {g:?}");
    }

    /// …but curves must never pass the bar test: a curve's path is far longer than its
    /// endpoint span, which is what the straightness guard measures. Two stacked arcs
    /// (think `⌒` over `⌣`) satisfy every OTHER `=` condition — similar width, strong
    /// x-overlap, small vertical gap — so straightness is the only thing keeping them
    /// from reading as an equals sign.
    #[test]
    fn stacked_curves_are_not_an_equals_sign() {
        let g = segment(&[
            stroke(&[(0.40, 0.46), (0.43, 0.41), (0.46, 0.46)]), // top arc ⌒
            stroke(&[(0.40, 0.52), (0.43, 0.57), (0.46, 0.52)]), // bottom arc ⌣
        ]);
        assert_eq!(g.len(), 2, "arcs merged as an =: {g:?}");
    }

    /// A selection lasso is not notation, but it is what exposed this, and it must not
    /// absorb the page either.
    #[test]
    fn an_enclosing_loop_does_not_absorb_what_it_encloses() {
        let loop_ = stroke(&[
            (0.20, 0.30),
            (0.70, 0.30),
            (0.70, 0.70),
            (0.20, 0.70),
            (0.20, 0.30),
        ]);
        let g = segment(&[
            loop_,
            glyph(0.35, 0.45, 0.06, 0.08),
            glyph(0.45, 0.45, 0.06, 0.08),
        ]);
        assert_eq!(g.len(), 3, "the loop swallowed its contents: {g:?}");
    }

    use super::*;
    use crate::stroke::Point;

    /// A stroke whose bbox is `[min_x,max_x] × [min_y,max_y]` (two opposite corners).
    fn stroke(pts: &[(f32, f32)]) -> Stroke {
        Stroke {
            points: pts
                .iter()
                .map(|&(x, y)| Point::new(x, y, 1.0, 0.0, 0.0, 0))
                .collect(),
        }
    }

    /// A closed glyph-sized outline — a real symbol's ink, not just its corners.
    fn glyph(x: f32, y: f32, w: f32, h: f32) -> Stroke {
        stroke(&[(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
    }

    fn boxed(min_x: f32, max_x: f32, min_y: f32, max_y: f32) -> Stroke {
        Stroke {
            points: vec![
                Point::new(min_x, min_y, 1.0, 0.0, 0.0, 0),
                Point::new(max_x, max_y, 1.0, 0.0, 0.0, 1),
            ],
        }
    }

    #[test]
    fn two_separated_symbols_split() {
        let s = vec![boxed(0.00, 0.15, 0.40, 0.60), boxed(0.40, 0.55, 0.40, 0.60)];
        assert_eq!(segment(&s), vec![vec![0], vec![1]]);
    }

    /// The first live subtraction, verbatim: `2x - 5x = 80` written on device,
    /// subsampled but with every stroke's endpoints, bbox extremes and closest-approach
    /// points kept EXACT. The `5` and the second `x` sit 0.0057 apart — inside the
    /// 0.0073 distance threshold — and fused into one blob that classified as `\ast`.
    /// Their x-projections are disjoint (+0.002), which is what the side-by-side rule
    /// keys on; the x's own crossing strokes overlap in x by −0.015 and must stay merged.
    #[test]
    #[allow(clippy::approx_constant)] // measured pen coordinates; two happen to look like 1/π and log₁₀e
    fn live_tight_product_5x_splits_but_x_stays_whole() {
        let s = vec![
            stroke(&[(0.1667, 0.3257), (0.1749, 0.3221), (0.1758, 0.3220), (0.1843, 0.3245), (0.1902, 0.3321), (0.1860, 0.3430), (0.1779, 0.3496), (0.1758, 0.3501), (0.1730, 0.3479), (0.1753, 0.3414), (0.1884, 0.3408), (0.1973, 0.3446), (0.2001, 0.3466), (0.2010, 0.3461)]), // 2
            stroke(&[(0.2397, 0.3335), (0.2397, 0.3338), (0.2397, 0.3338), (0.2395, 0.3341), (0.2389, 0.3353), (0.2373, 0.3372), (0.2349, 0.3401), (0.2323, 0.3430), (0.2317, 0.3436), (0.2290, 0.3469), (0.2271, 0.3492), (0.2256, 0.3507), (0.2247, 0.3518), (0.2247, 0.3518)]), // x, stroke 1
            stroke(&[(0.2191, 0.3395), (0.2191, 0.3397), (0.2197, 0.3397), (0.2205, 0.3399), (0.2218, 0.3401), (0.2235, 0.3407), (0.2257, 0.3412), (0.2282, 0.3418), (0.2306, 0.3425), (0.2323, 0.3430), (0.2327, 0.3428), (0.2350, 0.3439), (0.2374, 0.3446), (0.2396, 0.3449), (0.2406, 0.3448)]), // x, stroke 2
            stroke(&[(0.2743, 0.3388), (0.2738, 0.3389), (0.2736, 0.3390), (0.2736, 0.3388), (0.2736, 0.3390), (0.2737, 0.3390), (0.2743, 0.3392), (0.2762, 0.3397), (0.2793, 0.3399), (0.2808, 0.3402), (0.2830, 0.3399), (0.2866, 0.3399), (0.2908, 0.3397), (0.2948, 0.3395), (0.2949, 0.3394)]), // -
            stroke(&[(0.3536, 0.3221), (0.3435, 0.3230), (0.3347, 0.3232), (0.3336, 0.3258), (0.3305, 0.3345), (0.3301, 0.3363), (0.3301, 0.3366), (0.3307, 0.3370), (0.3414, 0.3362), (0.3497, 0.3397), (0.3498, 0.3401), (0.3495, 0.3421), (0.3440, 0.3473), (0.3315, 0.3512), (0.3302, 0.3510), (0.3290, 0.3503), (0.3321, 0.3468), (0.3330, 0.3462)]), // 5
            stroke(&[(0.3731, 0.3363), (0.3737, 0.3351), (0.3739, 0.3352), (0.3739, 0.3353), (0.3739, 0.3354), (0.3737, 0.3353), (0.3720, 0.3372), (0.3667, 0.3425), (0.3654, 0.3438), (0.3600, 0.3492), (0.3570, 0.3522), (0.3571, 0.3524), (0.3574, 0.3523), (0.3588, 0.3496), (0.3587, 0.3466)]), // x, stroke 1
            stroke(&[(0.3559, 0.3401), (0.3556, 0.3395), (0.3555, 0.3398), (0.3555, 0.3398), (0.3558, 0.3398), (0.3571, 0.3404), (0.3595, 0.3415), (0.3626, 0.3429), (0.3656, 0.3440), (0.3667, 0.3443), (0.3706, 0.3455), (0.3747, 0.3462), (0.3765, 0.3462), (0.3782, 0.3461)]), // x, stroke 2
            stroke(&[(0.4052, 0.3301), (0.4051, 0.3298), (0.4052, 0.3297), (0.4055, 0.3296), (0.4074, 0.3297), (0.4109, 0.3296), (0.4154, 0.3295), (0.4210, 0.3291), (0.4261, 0.3286), (0.4304, 0.3283), (0.4337, 0.3282), (0.4343, 0.3282), (0.4372, 0.3282), (0.4384, 0.3286)]), // = top bar
            stroke(&[(0.4164, 0.3423), (0.4175, 0.3427), (0.4186, 0.3427), (0.4201, 0.3430), (0.4219, 0.3429), (0.4240, 0.3429), (0.4247, 0.3432), (0.4264, 0.3427), (0.4290, 0.3426), (0.4317, 0.3424), (0.4345, 0.3423), (0.4372, 0.3418), (0.4396, 0.3419), (0.4418, 0.3418)]), // = bottom bar
            stroke(&[(0.5055, 0.3187), (0.5058, 0.3161), (0.5026, 0.3153), (0.4933, 0.3177), (0.4799, 0.3266), (0.4798, 0.3268), (0.4798, 0.3271), (0.4799, 0.3281), (0.4879, 0.3339), (0.4970, 0.3392), (0.4906, 0.3447), (0.4884, 0.3457), (0.4815, 0.3431), (0.4813, 0.3423), (0.4813, 0.3421), (0.4918, 0.3322), (0.5081, 0.3205), (0.5091, 0.3196), (0.5092, 0.3193), (0.5060, 0.3198)]), // 8
            stroke(&[(0.5369, 0.3191), (0.5353, 0.3180), (0.5296, 0.3203), (0.5212, 0.3280), (0.5209, 0.3286), (0.5194, 0.3334), (0.5194, 0.3340), (0.5202, 0.3369), (0.5274, 0.3415), (0.5347, 0.3427), (0.5362, 0.3426), (0.5431, 0.3393), (0.5463, 0.3324), (0.5463, 0.3318), (0.5441, 0.3215), (0.5400, 0.3174), (0.5397, 0.3174), (0.5384, 0.3179)]), // 0
        ];
        let g = segment(&s);
        assert_eq!(
            g,
            vec![
                vec![0],       // 2
                vec![1, 2],    // x
                vec![3],       // -
                vec![4],       // 5   — was fused with the x below
                vec![5, 6],    // x
                vec![7, 8],    // =   (stacked-bar merge)
                vec![9],       // 8
                vec![10],      // 0
            ],
            "the live `2x-5x=80` must segment into exactly 8 symbols"
        );
    }

    /// A live `=` whose top bar came out at 1.9× the bottom bar's width (people are
    /// not rulers) — the old 1.8 similar-width guard split it into two minus signs,
    /// which then parsed `-^{-}`. Real geometry from the guided session (083). The
    /// corridor is empty beyond both bars, which is what licenses the merge.
    #[test]
    fn a_sloppy_width_equals_still_merges() {
        let s = vec![
            stroke(&[(0.413, 0.650), (0.433, 0.648), (0.453, 0.651)]), // lower bar, w=0.040
            stroke(&[(0.416, 0.636), (0.427, 0.635), (0.437, 0.638)]), // upper bar, w=0.021
            glyph(0.468, 0.633, 0.025, 0.027),                         // the 2 after it
        ];
        let g = segment(&s);
        assert_eq!(g.len(), 2, "= must be one symbol, then the 2: {g:?}");
    }

    /// …but the relaxed width guard must NOT glue a numerator's `-` to the fraction
    /// bar below it: the corridor beyond that pair contains the denominator, and
    /// content-in-corridor is what separates a fraction from an `=`.
    #[test]
    fn a_fraction_bar_does_not_pair_with_the_numerator_minus() {
        let s = vec![
            stroke(&[(0.42, 0.40), (0.47, 0.40), (0.52, 0.40)]),       // numerator minus
            stroke(&[(0.38, 0.50), (0.47, 0.50), (0.56, 0.50)]),       // fraction bar (~1.9×)
            glyph(0.44, 0.56, 0.06, 0.08),                             // denominator c, in-corridor
        ];
        let g = segment(&s);
        assert_eq!(g.len(), 3, "fraction must stay bar/num/den: {g:?}");
    }

    /// The side-by-side rule must not split symbols whose strokes touch without
    /// crossing: a K-style arm meeting its stem has a *positive* x-projection gap
    /// but near-zero ink distance.
    #[test]
    fn touching_side_by_side_strokes_still_merge() {
        let s = vec![
            stroke(&[(0.30, 0.30), (0.30, 0.40), (0.30, 0.50)]), // stem
            stroke(&[(0.301, 0.40), (0.34, 0.32)]),              // upper arm, starts ON the stem
            stroke(&[(0.301, 0.40), (0.34, 0.50)]),              // lower arm
            glyph(0.40, 0.35, 0.08, 0.10),                       // a neighbour that must stay separate
        ];
        assert_eq!(segment(&s), vec![vec![0, 1, 2], vec![3]]);
    }

    #[test]
    fn overlapping_strokes_are_one_symbol() {
        // Crossing strokes of an 'x' — overlapping bboxes → one symbol.
        let s = vec![boxed(0.10, 0.30, 0.40, 0.60), boxed(0.12, 0.28, 0.42, 0.58)];
        assert_eq!(segment(&s), vec![vec![0, 1]]);
    }

    #[test]
    fn vertical_stack_splits() {
        // A fraction's bar / numerator / denominator (stacked at one x) → 3 symbols,
        // which `structure` then re-assembles into \frac.
        let s = vec![
            boxed(0.20, 0.60, 0.49, 0.51), // bar
            boxed(0.38, 0.42, 0.28, 0.40), // numerator (above)
            boxed(0.38, 0.42, 0.60, 0.72), // denominator (below)
        ];
        assert_eq!(segment(&s).len(), 3);
    }

    #[test]
    fn orders_left_to_right() {
        let s = vec![
            boxed(0.40, 0.55, 0.40, 0.60), // middle (idx 0)
            boxed(0.80, 0.95, 0.40, 0.60), // right  (idx 1)
            boxed(0.00, 0.15, 0.40, 0.60), // left   (idx 2)
        ];
        assert_eq!(segment(&s), vec![vec![2], vec![0], vec![1]]);
    }

    #[test]
    fn empty_and_blank_strokes() {
        assert!(segment(&[]).is_empty());
        assert!(segment(&[Stroke::new()]).is_empty());
    }
}
