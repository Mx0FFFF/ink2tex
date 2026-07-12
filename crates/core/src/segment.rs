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

    // Merge threshold = a fraction of the median stroke size (its larger side).
    let mut sizes: Vec<f32> = items
        .iter()
        .map(|(_, b)| (b[2] - b[0]).max(b[3] - b[1]))
        .collect();
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
            if gap(&items[a].1, &items[b].1) >= thresh {
                continue; // boxes too far ⇒ ink too far
            }
            if ink_within(&strokes[items[a].0], &strokes[items[b].0], thresh) {
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
    groups.into_iter().map(|(_, idx)| idx).collect()
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
