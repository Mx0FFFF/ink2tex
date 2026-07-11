//! The normalized ink data model. Plain data types, no I/O, no device coupling:
//! the raw-digitizer -> normalized-canvas transform happens in `crates/rm`, and
//! everything here is already in normalized coordinates.

/// One digitizer sample, in **normalized** (canvas) coordinates — not raw device
/// units. `x`/`y` are typically in `[0, 1]` with `y` pointing down (screen
/// convention); `pressure` in `[0, 1]`; `tilt_*` in radians as the pen reports.
/// `t_us` is microseconds since the start of capture (monotonic) — the online
/// stroke features in `classify/` (dx, dy, curvature over time) depend on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub pressure: f32,
    pub tilt_x: f32,
    pub tilt_y: f32,
    pub t_us: u64,
}

impl Point {
    pub fn new(x: f32, y: f32, pressure: f32, tilt_x: f32, tilt_y: f32, t_us: u64) -> Self {
        Self {
            x,
            y,
            pressure,
            tilt_x,
            tilt_y,
            t_us,
        }
    }
}

/// A single pen-down..pen-up trajectory.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stroke {
    pub points: Vec<Point>,
}

impl Stroke {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, p: Point) {
        self.points.push(p);
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl FromIterator<Point> for Stroke {
    fn from_iter<I: IntoIterator<Item = Point>>(it: I) -> Self {
        Stroke {
            points: it.into_iter().collect(),
        }
    }
}

/// A captured drawing: a bag of strokes plus the native source dimensions the
/// normalization was performed against. `source_width` / `source_height` exist
/// purely so a renderer can reproduce the correct **aspect ratio** (normalized
/// coordinates alone lose it); `0.0` means "unknown, assume square pixels".
#[derive(Debug, Clone, PartialEq)]
pub struct Ink {
    pub source_width: f32,
    pub source_height: f32,
    pub strokes: Vec<Stroke>,
}

impl Default for Ink {
    fn default() -> Self {
        Ink {
            source_width: 0.0,
            source_height: 0.0,
            strokes: Vec::new(),
        }
    }
}

impl Ink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: record the native source dimensions (for aspect-correct rendering).
    pub fn with_source(mut self, w: f32, h: f32) -> Self {
        self.source_width = w;
        self.source_height = h;
        self
    }

    pub fn push(&mut self, s: Stroke) {
        self.strokes.push(s);
    }

    /// Total sample count across all strokes.
    pub fn point_count(&self) -> usize {
        self.strokes.iter().map(|s| s.points.len()).sum()
    }

    /// Axis-aligned bounding box over every point as `(min_x, min_y, max_x, max_y)`,
    /// or `None` if there are no points. Used by the renderer to frame the drawing.
    pub fn bounds(&self) -> Option<(f32, f32, f32, f32)> {
        let mut it = self.strokes.iter().flat_map(|s| s.points.iter());
        let first = it.next()?;
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (first.x, first.y, first.x, first.y);
        for p in it {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        Some((min_x, min_y, max_x, max_y))
    }

    /// Source aspect ratio (w / h) if known and valid, else `None`.
    pub fn aspect_ratio(&self) -> Option<f32> {
        if self.source_width > 0.0 && self.source_height > 0.0 {
            Some(self.source_width / self.source_height)
        } else {
            None
        }
    }
}
