//! The ink-capture state machine: raw evdev events → `ink2tex_core::Ink`.
//!
//! ## Systems concept: EV_SYN batching
//! evdev doesn't deliver "a point". It delivers a *stream* of independent axis
//! updates — `EV_ABS ABS_X`, `EV_ABS ABS_Y`, `EV_ABS ABS_PRESSURE`, key presses
//! (`BTN_TOOL_PEN` when the pen enters range, `BTN_TOUCH` when the tip presses the
//! glass) — and then an `EV_SYN`/`SYN_REPORT` that means "everything since the last
//! SYN is one coherent sample; latch it now." So we accumulate the latest raw axis
//! values as they arrive and only *emit a point* on `SYN_REPORT`, while the tip is
//! down. This is pure logic over an event stream — no device needed — so it is
//! unit-tested below with synthetic events.

use crate::evdev::{
    AbsInfo, InputEvent, ABS_PRESSURE, ABS_TILT_X, ABS_TILT_Y, ABS_X, ABS_Y, BTN_TOOL_PEN,
    BTN_TOOL_RUBBER, BTN_TOUCH, EV_ABS, EV_KEY, EV_SYN, SYN_REPORT,
};
use crate::transform::Transform;
use ink2tex_core::{Ink, Point, Stroke};

/// A live-draw hint, emitted when a sample is latched under the tip. `from` is the
/// previous normalized point in the current stroke (`None` = first point of a new
/// stroke). The drawing frontend turns this into one E-Ink line segment.
// `to`/`pressure` are consumed by the arm drawing frontend (`--ink`); the host
// `--record` build ignores the segment, so let them look unused there.
#[allow(dead_code)]
pub struct Segment {
    pub from: Option<(f32, f32)>,
    pub to: (f32, f32),
    pub pressure: f32,
}

pub struct Capture {
    tf: Transform,
    pressure: AbsInfo,
    tilt_x: AbsInfo,
    tilt_y: AbsInfo,
    // latest raw axis values seen since the last SYN_REPORT
    raw_x: i32,
    raw_y: i32,
    raw_p: i32,
    raw_tx: i32,
    raw_ty: i32,
    tip_down: bool,
    /// The eraser end of the Marker is what's in range (`BTN_TOOL_RUBBER`). Erasing looks
    /// identical to drawing on this bus — same `BTN_TOUCH`, same coordinate stream — so
    /// this flag is the only thing standing between "the user rubbed something out" and
    /// "the user drew a symbol".
    eraser: bool,
    t0_us: Option<u64>,
    last_norm: Option<(f32, f32)>,
    stroke: Stroke,
    ink: Ink,
}

impl Capture {
    /// Build from the digitizer's axis metadata (read via `EVIOCGABS`). Takes the
    /// axes directly rather than `&Digitizer` so tests don't fabricate a file fd.
    pub fn from_axes(
        x: AbsInfo,
        y: AbsInfo,
        pressure: AbsInfo,
        tilt_x: AbsInfo,
        tilt_y: AbsInfo,
    ) -> Self {
        Capture {
            tf: Transform::new(x, y),
            pressure,
            tilt_x,
            tilt_y,
            raw_x: 0,
            raw_y: 0,
            raw_p: 0,
            raw_tx: 0,
            raw_ty: 0,
            tip_down: false,
            eraser: false,
            t0_us: None,
            last_norm: None,
            stroke: Stroke::new(),
            ink: Ink::new().with_source(1404.0, 1872.0), // rM2 display, portrait
        }
    }

    /// Feed one event. Returns a draw hint when a point is latched under the tip.
    pub fn process(&mut self, ev: &InputEvent) -> Option<Segment> {
        match ev.kind {
            EV_ABS => {
                match ev.code {
                    ABS_X => self.raw_x = ev.value,
                    ABS_Y => self.raw_y = ev.value,
                    ABS_PRESSURE => self.raw_p = ev.value,
                    ABS_TILT_X => self.raw_tx = ev.value,
                    ABS_TILT_Y => self.raw_ty = ev.value,
                    _ => {}
                }
                None
            }
            // Which END of the Marker is in range? The eraser emits `BTN_TOUCH` and a full
            // coordinate stream exactly like the tip does, so without this the user's
            // *erasing* is recorded as ink — and then handed to the classifier as if they
            // had drawn it.
            EV_KEY if ev.code == BTN_TOOL_RUBBER => {
                self.eraser = ev.value != 0;
                if self.eraser {
                    self.tip_down = false;
                    self.end_stroke(); // flipped mid-stroke: keep what was drawn, stop here
                }
                None
            }
            EV_KEY if ev.code == BTN_TOUCH => {
                let down = ev.value != 0 && !self.eraser; // erasing is not drawing
                if !down {
                    self.end_stroke(); // tip lifted → finish the stroke
                }
                self.tip_down = down;
                None
            }
            EV_KEY if ev.code == BTN_TOOL_PEN && ev.value == 0 => {
                // Pen left detection range entirely — end any stroke in progress.
                self.tip_down = false;
                self.end_stroke();
                None
            }
            EV_SYN if ev.code == SYN_REPORT && self.tip_down => Some(self.commit_point(ev.t_us())),
            _ => None,
        }
    }

    fn commit_point(&mut self, t_us: u64) -> Segment {
        let t0 = *self.t0_us.get_or_insert(t_us);
        let (nx, ny) = self.tf.to_norm(self.raw_x, self.raw_y);
        let pressure = self.tf.norm_pressure(self.raw_p, &self.pressure);
        let tilt_x = signed_frac(self.raw_tx, &self.tilt_x);
        let tilt_y = signed_frac(self.raw_ty, &self.tilt_y);
        self.stroke.push(Point::new(
            nx,
            ny,
            pressure,
            tilt_x,
            tilt_y,
            t_us.saturating_sub(t0),
        ));
        let from = self.last_norm.replace((nx, ny));
        Segment {
            from,
            to: (nx, ny),
            pressure,
        }
    }

    fn end_stroke(&mut self) {
        if !self.stroke.is_empty() {
            self.ink.push(std::mem::take(&mut self.stroke));
        }
        self.last_norm = None;
    }

    /// Flush the in-progress stroke and return the captured drawing.
    pub fn finish(mut self) -> Ink {
        self.end_stroke();
        self.ink
    }
}

/// Map a symmetric axis (±max, e.g. tilt) into [-1, 1] against its reported range.
fn signed_frac(v: i32, a: &AbsInfo) -> f32 {
    let span = (a.maximum - a.minimum) as f32;
    if span <= 0.0 {
        return 0.0;
    }
    (2.0 * ((v - a.minimum) as f32 / span) - 1.0).clamp(-1.0, 1.0)
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

    fn cap() -> Capture {
        Capture::from_axes(
            ax(0, 20966),
            ax(0, 15725),
            ax(0, 4095),
            ax(-9000, 9000), // confirmed ABS_TILT_X range on hardware
            ax(-9000, 9000),
        )
    }

    fn ev(kind: u16, code: u16, value: i32, us: i64) -> InputEvent {
        InputEvent {
            tv_sec: 0,
            tv_usec: us as libc::suseconds_t,
            kind,
            code,
            value,
        }
    }

    // Latch one sample: set X, Y, pressure, then SYN_REPORT.
    fn sample(c: &mut Capture, x: i32, y: i32, p: i32, us: i64) -> Option<Segment> {
        c.process(&ev(EV_ABS, ABS_X, x, us));
        c.process(&ev(EV_ABS, ABS_Y, y, us));
        c.process(&ev(EV_ABS, ABS_PRESSURE, p, us));
        c.process(&ev(EV_SYN, SYN_REPORT, 0, us))
    }

    /// Erasing is not drawing. The rM2 digitizer advertises `BTN_TOOL_RUBBER` (its KEY
    /// bitmask has bit 0x141), and while the eraser is in range it still emits `BTN_TOUCH`
    /// and a full coordinate stream — so a capture that only watches `BTN_TOUCH` silently
    /// records the user rubbing something out, and then classifies it as a symbol.
    #[test]
    fn eraser_end_is_not_captured_as_ink() {
        let mut c = cap();
        c.process(&ev(EV_KEY, BTN_TOOL_RUBBER, 1, 0)); // pen flipped over
        c.process(&ev(EV_KEY, BTN_TOUCH, 1, 0)); // eraser pressed to the glass
        for i in 0..5 {
            sample(
                &mut c,
                5000 + i * 100,
                6000 + i * 100,
                2000,
                10_000 * i as i64,
            );
        }
        c.process(&ev(EV_KEY, BTN_TOUCH, 0, 60_000));
        assert!(c.finish().strokes.is_empty(), "erasing was recorded as ink");
    }

    /// …and the tip still works once the pen is flipped back.
    #[test]
    fn tip_still_draws_after_the_eraser_leaves_range() {
        let mut c = cap();
        c.process(&ev(EV_KEY, BTN_TOOL_RUBBER, 1, 0));
        c.process(&ev(EV_KEY, BTN_TOUCH, 1, 0));
        sample(&mut c, 5000, 6000, 2000, 0);
        c.process(&ev(EV_KEY, BTN_TOUCH, 0, 10_000));
        c.process(&ev(EV_KEY, BTN_TOOL_RUBBER, 0, 20_000)); // flipped back to the tip
        c.process(&ev(EV_KEY, BTN_TOOL_PEN, 1, 20_000));
        c.process(&ev(EV_KEY, BTN_TOUCH, 1, 20_000));
        for i in 0..3 {
            sample(
                &mut c,
                7000 + i * 100,
                8000,
                2000,
                30_000 + 10_000 * i as i64,
            );
        }
        c.process(&ev(EV_KEY, BTN_TOUCH, 0, 70_000));
        let ink = c.finish();
        assert_eq!(ink.strokes.len(), 1, "only the tip's stroke should survive");
        assert_eq!(ink.strokes[0].points.len(), 3);
    }

    #[test]
    fn one_stroke_from_pen_down_to_up() {
        let mut c = cap();
        c.process(&ev(EV_KEY, BTN_TOOL_PEN, 1, 0));
        c.process(&ev(EV_KEY, BTN_TOUCH, 1, 0));
        for i in 0..5 {
            let seg = sample(
                &mut c,
                5000 + i * 100,
                6000 + i * 100,
                2000,
                10_000 * i as i64,
            );
            assert!(seg.is_some());
        }
        c.process(&ev(EV_KEY, BTN_TOUCH, 0, 60_000));
        let ink = c.finish();
        assert_eq!(ink.strokes.len(), 1);
        assert_eq!(ink.strokes[0].points.len(), 5);
    }

    #[test]
    fn no_points_emitted_before_tip_touches() {
        let mut c = cap();
        c.process(&ev(EV_KEY, BTN_TOOL_PEN, 1, 0)); // hovering, tip not down
        assert!(sample(&mut c, 5000, 6000, 0, 0).is_none());
        assert_eq!(c.finish().strokes.len(), 0);
    }

    #[test]
    fn two_strokes_separated_by_pen_lift() {
        let mut c = cap();
        c.process(&ev(EV_KEY, BTN_TOUCH, 1, 0));
        sample(&mut c, 100, 100, 1000, 0);
        sample(&mut c, 200, 200, 1000, 5_000);
        c.process(&ev(EV_KEY, BTN_TOUCH, 0, 10_000));
        c.process(&ev(EV_KEY, BTN_TOUCH, 1, 20_000));
        sample(&mut c, 300, 300, 1000, 25_000);
        let ink = c.finish();
        assert_eq!(ink.strokes.len(), 2);
        assert_eq!(ink.strokes[0].points.len(), 2);
        assert_eq!(ink.strokes[1].points.len(), 1);
    }

    #[test]
    fn first_point_of_stroke_has_no_from() {
        let mut c = cap();
        c.process(&ev(EV_KEY, BTN_TOUCH, 1, 0));
        let s0 = sample(&mut c, 100, 100, 1000, 0).unwrap();
        assert!(s0.from.is_none());
        let s1 = sample(&mut c, 200, 200, 1000, 5_000).unwrap();
        assert!(s1.from.is_some());
    }

    #[test]
    fn normalized_coords_stay_in_unit_square_and_time_is_relative() {
        let mut c = cap();
        c.process(&ev(EV_KEY, BTN_TOUCH, 1, 0));
        sample(&mut c, 0, 0, 0, 1_000_000); // first sample at t=1s
        sample(&mut c, 20966, 15725, 4095, 1_050_000);
        let ink = c.finish();
        for p in &ink.strokes[0].points {
            assert!((0.0..=1.0).contains(&p.x) && (0.0..=1.0).contains(&p.y));
        }
        // t_us is measured from the first latched sample.
        assert_eq!(ink.strokes[0].points[0].t_us, 0);
        assert_eq!(ink.strokes[0].points[1].t_us, 50_000);
    }
}
