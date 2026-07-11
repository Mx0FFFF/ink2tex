//! Stroke → symbol-group segmentation for **linear** expressions (M2).
//!
//! Greedy and spatial: sort strokes left-to-right by bounding-box center, then walk
//! them merging horizontally-overlapping or near-adjacent strokes into one symbol
//! and splitting on a visible gap (scaled by the typical symbol height). Most
//! symbols are 1–3 strokes, so groups are capped at 4.
//!
//! This is deliberately the "80%" heuristic DESIGN §4.2 prescribes for M2 — good for
//! a single left-to-right line like `2x + 3 = 7`. It does **not** solve delayed
//! strokes (dotting an `i` after moving on, a fraction bar drawn last): that needs
//! the jointly-optimized hypothesis lattice, which is M3. Don't build it here.

use crate::stroke::Stroke;

/// Most symbols are 1–3 strokes (`=` is 2, `≡` is 3); cap a group at 4.
const MAX_GROUP: usize = 4;

#[derive(Clone, Copy)]
struct Bbox {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

impl Bbox {
    fn center_x(&self) -> f32 {
        (self.min_x + self.max_x) * 0.5
    }
    fn height(&self) -> f32 {
        self.max_y - self.min_y
    }
}

fn bbox(s: &Stroke) -> Option<Bbox> {
    let mut it = s.points.iter();
    let p0 = it.next()?;
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (p0.x, p0.x, p0.y, p0.y);
    for p in it {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    Some(Bbox {
        min_x,
        max_x,
        min_y,
        max_y,
    })
}

fn cmp(a: f32, b: f32) -> core::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(core::cmp::Ordering::Equal)
}

/// Group strokes into symbols, ordered left-to-right. Returns groups of indices
/// into `strokes` (empty strokes are dropped).
pub fn segment(strokes: &[Stroke]) -> Vec<Vec<usize>> {
    // Index every non-empty stroke with its bbox, ordered left-to-right.
    let mut items: Vec<(usize, Bbox)> = strokes
        .iter()
        .enumerate()
        .filter_map(|(i, s)| bbox(s).map(|b| (i, b)))
        .collect();
    if items.is_empty() {
        return Vec::new();
    }
    items.sort_by(|(_, a), (_, b)| cmp(a.center_x(), b.center_x()));

    // The gap that separates two symbols, scaled by the typical symbol height (so it
    // is invariant to how big the drawing is).
    let mut heights: Vec<f32> = items.iter().map(|(_, b)| b.height()).collect();
    heights.sort_by(|&a, &b| cmp(a, b));
    let gap = 0.25 * heights[heights.len() / 2].max(1e-6);

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_max_x = f32::NEG_INFINITY;
    for (i, b) in &items {
        // Join the current symbol if this stroke overlaps it (or sits within `gap`
        // of its right edge) and the group isn't already full.
        let joins = !cur.is_empty() && cur.len() < MAX_GROUP && b.min_x <= cur_max_x + gap;
        if !joins && !cur.is_empty() {
            groups.push(std::mem::take(&mut cur));
            cur_max_x = f32::NEG_INFINITY;
        }
        cur.push(*i);
        cur_max_x = cur_max_x.max(b.max_x);
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stroke::Point;

    /// A stroke whose bbox is exactly `[min_x,max_x] × [min_y,max_y]` (two corners).
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
        // Height 0.6 → gap threshold 0.15; the 0.20 gap between them is visible.
        let s = vec![boxed(0.05, 0.40, 0.2, 0.8), boxed(0.60, 0.95, 0.2, 0.8)];
        let g = segment(&s);
        assert_eq!(g, vec![vec![0], vec![1]]);
    }

    #[test]
    fn overlapping_strokes_are_one_symbol() {
        // Two crossing strokes of an 'x': same x-range → one symbol.
        let s = vec![boxed(0.1, 0.5, 0.1, 0.9), boxed(0.1, 0.5, 0.9, 0.1)];
        assert_eq!(segment(&s), vec![vec![0, 1]]);
    }

    #[test]
    fn orders_left_to_right_regardless_of_draw_order() {
        // Drawn middle, right, left → segmented left, middle, right.
        let s = vec![
            boxed(0.40, 0.60, 0.2, 0.8), // middle (idx 0)
            boxed(0.80, 1.00, 0.2, 0.8), // right  (idx 1)
            boxed(0.00, 0.20, 0.2, 0.8), // left   (idx 2)
        ];
        assert_eq!(segment(&s), vec![vec![2], vec![0], vec![1]]);
    }

    #[test]
    fn caps_group_size() {
        // Five overlapping strokes → 4 + 1.
        let s: Vec<Stroke> = (0..5).map(|_| boxed(0.1, 0.5, 0.1, 0.9)).collect();
        let g = segment(&s);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].len(), 4);
        assert_eq!(g[1].len(), 1);
    }

    #[test]
    fn empty_and_blank_strokes() {
        assert!(segment(&[]).is_empty());
        assert!(segment(&[Stroke::new()]).is_empty()); // blank stroke dropped
    }
}
