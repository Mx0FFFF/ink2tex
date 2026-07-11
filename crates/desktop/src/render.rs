//! Headless stroke rasterizer (tiny-skia — pure Rust, no system libs). Maps
//! normalized ink coordinates into a framed PNG, preserving the source aspect
//! ratio so the output is shaped like the tablet would show it.

use std::path::Path;

use anyhow::{anyhow, Result};
use ink2tex_core::Ink;
use tiny_skia::{
    Color, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Rect, Stroke as SkStroke, Transform,
};

/// Long edge of the rendered PNG, in pixels.
const LONG_EDGE: f32 = 1000.0;
/// Fraction of the canvas kept as a margin around the ink.
const MARGIN: f32 = 0.06;
/// Fallback aspect ratio when the ink didn't record source dims (reMarkable 2 shape).
const RM2_ASPECT: f32 = 1872.0 / 1404.0;

pub fn render_to_png(ink: &Ink, out: &Path) -> Result<()> {
    let aspect = ink.aspect_ratio().unwrap_or(RM2_ASPECT);
    let (w, h) = if aspect >= 1.0 {
        (LONG_EDGE, LONG_EDGE / aspect)
    } else {
        (LONG_EDGE * aspect, LONG_EDGE)
    };
    let (wp, hp) = (w.round().max(1.0) as u32, h.round().max(1.0) as u32);

    let mut pm = Pixmap::new(wp, hp).ok_or_else(|| anyhow!("invalid pixmap size {wp}x{hp}"))?;
    pm.fill(Color::from_rgba8(0xFA, 0xFA, 0xFA, 0xFF));

    // Map normalized [0,1]^2 into the canvas, inset by MARGIN.
    let m = MARGIN;
    let (sx, sy) = (w * (1.0 - 2.0 * m), h * (1.0 - 2.0 * m));
    let (ox, oy) = (w * m, h * m);
    let map = |x: f32, y: f32| (ox + x * sx, oy + y * sy);

    let mut paint = Paint::default();
    paint.set_color(Color::from_rgba8(0x1A, 0x1A, 0x1A, 0xFF));
    paint.anti_alias = true;

    for stroke in &ink.strokes {
        // A lone sample can't form a polyline — draw a small dot so it's visible.
        if stroke.points.len() < 2 {
            if let Some(p) = stroke.points.first() {
                let (px, py) = map(p.x, p.y);
                if let Some(rect) = Rect::from_xywh(px - 1.5, py - 1.5, 3.0, 3.0) {
                    pm.fill_rect(rect, &paint, Transform::identity(), None);
                }
            }
            continue;
        }

        let mut pb = PathBuilder::new();
        let (x0, y0) = map(stroke.points[0].x, stroke.points[0].y);
        pb.move_to(x0, y0);
        for p in &stroke.points[1..] {
            let (x, y) = map(p.x, p.y);
            pb.line_to(x, y);
        }

        if let Some(path) = pb.finish() {
            // Width tracks mean pressure so pressure capture is visible in the PNG.
            let mean_p =
                stroke.points.iter().map(|p| p.pressure).sum::<f32>() / stroke.points.len() as f32;
            let sk = SkStroke {
                width: 1.5 + 3.0 * mean_p.clamp(0.0, 1.0),
                line_cap: LineCap::Round,
                line_join: LineJoin::Round,
                ..Default::default()
            };
            pm.stroke_path(&path, &paint, &sk, Transform::identity(), None);
        }
    }

    pm.save_png(out).map_err(|e| anyhow!("save_png: {e}"))?;
    Ok(())
}
