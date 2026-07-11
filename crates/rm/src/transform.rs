//! Digitizer → normalized-screen coordinate transform. This is the *one* place
//! that knows the pen's raw axis geometry; everything downstream lives in
//! normalized [0,1] screen space (x→right, y→down), which is what
//! `ink2tex_core::Point` stores. (`.claude/rules/device.md`: "Get the transform
//! right once, in one function, and unit-test it against known corners.")
//!
//! ## Systems concept: coordinate spaces
//! The Wacom digitizer under the rM2 glass is not aligned with the display. It
//! reports a much larger grid (~20k × ~15k vs 1404 × 1872) and is rotated 90°: the
//! digitizer's X runs along the display's *long* edge, its Y along the *short*
//! edge. So screen_x comes from digitizer-Y, and screen_y from digitizer-X (with a
//! flip so "up the page" is up on screen).
//!
//! The exact axis *ranges* are read from the device at startup (`EVIOCGABS`), so
//! the only thing baked in here is the rotation/flip — a physical property of how
//! the panel is mounted. ✅ Confirmed on hardware 2026-07-11: a captured 'R'
//! renders upright and un-mirrored, so this mapping is correct for the rM2 mount
//! (no flip needed). The tests below lock the math to a coherent bijection.

use crate::evdev::AbsInfo;

pub struct Transform {
    x: AbsInfo,
    y: AbsInfo,
}

impl Transform {
    pub fn new(x: AbsInfo, y: AbsInfo) -> Self {
        Self { x, y }
    }

    /// Map a raw digitizer sample to normalized screen coords in [0,1], y-down.
    pub fn to_norm(&self, raw_x: i32, raw_y: i32) -> (f32, f32) {
        let fx = frac(raw_x, &self.x); // along digitizer X (display long edge)
        let fy = frac(raw_y, &self.y); // along digitizer Y (display short edge)
        let screen_x = fy; // short edge → screen width
        let screen_y = 1.0 - fx; // long edge → screen height, flipped
        (screen_x, screen_y)
    }

    /// Normalize pressure to [0,1] against the reported ABS_PRESSURE range.
    pub fn norm_pressure(&self, raw: i32, p: &AbsInfo) -> f32 {
        frac(raw, p)
    }
}

fn frac(v: i32, a: &AbsInfo) -> f32 {
    let span = (a.maximum - a.minimum) as f32;
    if span <= 0.0 {
        return 0.0;
    }
    (((v - a.minimum) as f32) / span).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ax(min: i32, max: i32) -> AbsInfo {
        AbsInfo {
            minimum: min,
            maximum: max,
            ..AbsInfo::default()
        }
    }

    // Representative rM2 ranges; the real ones come from `ink2tex-rm --probe` and
    // are recorded in .claude/rules/device.md. The mapping math is range-agnostic.
    fn rm2() -> Transform {
        Transform::new(ax(0, 20966), ax(0, 15725))
    }

    #[test]
    fn center_maps_to_center() {
        let (sx, sy) = rm2().to_norm(20966 / 2, 15725 / 2);
        assert!((sx - 0.5).abs() < 1e-3, "sx={sx}");
        assert!((sy - 0.5).abs() < 1e-3, "sy={sy}");
    }

    #[test]
    fn corners_are_a_coherent_bijection() {
        let t = rm2();
        let (sx, sy) = t.to_norm(0, 0);
        assert!(
            (sx - 0.0).abs() < 1e-3 && (sy - 1.0).abs() < 1e-3,
            "({sx},{sy})"
        );
        let (sx, sy) = t.to_norm(20966, 15725);
        assert!(
            (sx - 1.0).abs() < 1e-3 && (sy - 0.0).abs() < 1e-3,
            "({sx},{sy})"
        );
    }

    #[test]
    fn out_of_range_is_clamped() {
        let (sx, sy) = rm2().to_norm(-5000, 999_999);
        assert!((0.0..=1.0).contains(&sx) && (0.0..=1.0).contains(&sy));
    }
}
