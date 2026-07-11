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
fn gap(a: &Bbox, b: &Bbox) -> f32 {
    let dx = (a[0] - b[2]).max(b[0] - a[2]).max(0.0);
    let dy = (a[1] - b[3]).max(b[1] - a[3]).max(0.0);
    dx.max(dy)
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

    // Union strokes whose bboxes are within `thresh` (in both x and y).
    let n = items.len();
    let mut parent: Vec<usize> = (0..n).collect();
    for a in 0..n {
        for b in (a + 1)..n {
            if gap(&items[a].1, &items[b].1) < thresh {
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
    use super::*;
    use crate::stroke::Point;

    /// A stroke whose bbox is `[min_x,max_x] × [min_y,max_y]` (two opposite corners).
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
