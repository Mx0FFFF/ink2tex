//! Synthetic pen strokes, injected into the **real digitizer's** event node — how
//! ink2tex draws on the panel without ever touching a framebuffer.
//!
//! ## Systems concept: `write(2)` on an evdev node is input in reverse
//! Everyone knows `/dev/input/eventN` can be `read()`. Less known: it can be
//! **written**. The evdev write handler (`evdev_write` → `input_inject_event`)
//! feeds your `struct input_event` into the input core as if the hardware had
//! produced it, and the core fans it out to *every* reader of that device —
//! xochitl included. Two consequences worth internalizing:
//!
//!   1. **No virtual device needed.** uinput (see `fake-pen`) creates a *new*
//!      device node, which only helps processes that enumerate devices after it
//!      appears. Writing to the existing node reaches the process that opened the
//!      real Wacom at boot — the one that owns the screen.
//!   2. **The capability bitmap is the contract.** `input_inject_event` drops
//!      events the device never advertised (`is_event_supported` checks
//!      `dev->evbit/keybit/absbit`). We only send what the Wacom already claims:
//!      `ABS_X/Y/PRESSURE`, `BTN_TOOL_PEN`, `BTN_TOOL_RUBBER`, `BTN_TOUCH`.
//!
//! This is the engine of the beautifier (`--beautify`): recognize the sloppy
//! ink, retrace it with the RUBBER tool (xochitl erases it), then "handwrite"
//! the typeset layout with the PEN tool. xochitl renders at its native latency
//! and — the part no framebuffer hack gets for free — **persists the result into
//! the notebook**, so the pretty version survives sync and export.
//!
//! Cohabitation honesty: whatever brush/eraser the human last picked in the
//! toolbar is what our strokes render with. We write with their pen. That is the
//! contract's cost and its charm.

use std::io::{self, Write};
use std::thread::sleep;
use std::time::Duration;

use std::os::unix::io::AsRawFd;

use crate::evdev::{AbsInfo, Digitizer, InputEvent};

const EV_SYN: u16 = 0x00;
const EV_KEY: u16 = 0x01;
const EV_ABS: u16 = 0x03;
const SYN_REPORT: u16 = 0;

const BTN_TOOL_PEN: u16 = 0x140;
const BTN_TOOL_RUBBER: u16 = 0x141;
const BTN_TOUCH: u16 = 0x14a;

const ABS_X: u16 = 0x00;
const ABS_Y: u16 = 0x01;
const ABS_PRESSURE: u16 = 0x18;
const ABS_DISTANCE: u16 = 0x19;

#[derive(Clone, Copy, PartialEq)]
pub enum Tool {
    Pen,
    Rubber,
}

pub struct Injector {
    node: std::fs::File,
    x: AbsInfo,
    y: AbsInfo,
    pressure: AbsInfo,
    /// Advertised by the device? (The rM2 Wacom has the eraser bit; a device
    /// without it would render our "erase" pass as INK.)
    pub has_rubber: bool,
    /// Microseconds of dwell per emitted sample — the real digitizer's ~200 Hz
    /// cadence. Going faster corrupts xochitl's rendering: its stroke pipeline
    /// expects hardware timing, and a teleporting pen produces joined glyphs and
    /// dropped strokes (measured on the first live beautify).
    pub dwell_us: u64,
}

impl Injector {
    /// Open the digitizer's node for writing. Separate fd from the read side:
    /// the capture path keeps its `O_RDONLY|O_NONBLOCK` fd untouched.
    pub fn open(dig: &Digitizer) -> io::Result<Self> {
        let node = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&dig.path)?;
        let has_rubber = dig.has_rubber;
        Ok(Self {
            node,
            x: dig.x,
            y: dig.y,
            pressure: dig.pressure,
            has_rubber,
            dwell_us: 5000,
        })
    }

    /// Is the HARDWARE pen in range or touching right now? Between our own
    /// strokes every bit we set has been released, so any set bit here is the
    /// human's — the interlock that keeps injection from interleaving with a
    /// live stroke in the kernel's shared input state (both-tools-in-range is a
    /// state a real Wacom cannot produce, and xochitl's reaction is undefined).
    pub fn hardware_busy(&self) -> io::Result<bool> {
        let keys = crate::evdev::current_keys(self.node.as_raw_fd())?;
        Ok(crate::evdev::key_is_down(&keys, BTN_TOOL_PEN)
            || crate::evdev::key_is_down(&keys, BTN_TOOL_RUBBER)
            || crate::evdev::key_is_down(&keys, BTN_TOUCH))
    }

    /// Block until the hardware pen leaves the glass and its hover range, or
    /// give up after `max_wait`.
    pub fn wait_hardware_clear(&self, max_wait: Duration) -> io::Result<()> {
        let t0 = std::time::Instant::now();
        while self.hardware_busy()? {
            if t0.elapsed() > max_wait {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "pen stayed in range — refusing to inject through a live pen",
                ));
            }
            sleep(Duration::from_millis(50));
        }
        Ok(())
    }

    /// Normalized screen coords ([0,1], x→right, y→down) → raw digitizer counts.
    /// Exact inverse of `Transform::to_norm`: screen x comes from digitizer Y,
    /// screen y from flipped digitizer X.
    fn to_raw(&self, sx: f32, sy: f32) -> (i32, i32) {
        let fx = (1.0 - sy).clamp(0.0, 1.0); // fraction along digitizer X
        let fy = sx.clamp(0.0, 1.0); //          fraction along digitizer Y
        let raw = |f: f32, a: &AbsInfo| a.minimum + (f * (a.maximum - a.minimum) as f32) as i32;
        (raw(fx, &self.x), raw(fy, &self.y))
    }

    fn send(&mut self, events: &[InputEvent]) -> io::Result<()> {
        // SAFETY of the transmute-free path: InputEvent is #[repr(C)] and Copy;
        // we hand the kernel exactly the bytes it defines. Timestamps are zero —
        // `input_inject_event` stamps events at delivery, same as hardware.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                events.as_ptr() as *const u8,
                std::mem::size_of_val(events),
            )
        };
        self.node.write_all(bytes)
    }

    fn ev(kind: u16, code: u16, value: i32) -> InputEvent {
        let mut e = InputEvent::zeroed();
        e.kind = kind;
        e.code = code;
        e.value = value;
        e
    }

    /// Trace one polyline with the given tool: approach → touch → glide → lift.
    /// Points are normalized screen coords. Consecutive points are interpolated
    /// so no hop exceeds ~1/300 of the screen — xochitl draws segments between
    /// samples, and a sparse polyline would render as visible chords.
    pub fn stroke(&mut self, tool: Tool, pts: &[(f32, f32)]) -> io::Result<()> {
        let Some(&(x0, y0)) = pts.first() else {
            return Ok(());
        };
        // Interlock: never begin a stroke while the human's pen is in range.
        self.wait_hardware_clear(Duration::from_secs(20))?;
        let btn_tool = match tool {
            Tool::Pen => BTN_TOOL_PEN,
            Tool::Rubber => BTN_TOOL_RUBBER,
        };
        let press = (self.pressure.maximum as f32 * 0.6) as i32;
        match self.stroke_inner(btn_tool, press, x0, y0, pts) {
            Ok(()) => Ok(()),
            Err(e) => {
                // Best-effort release: never leave touch/tool latched in the
                // kernel — the duplicate-state filter would silently swallow
                // the NEXT stroke's touch-down.
                let _ = self.send(&[
                    Self::ev(EV_ABS, ABS_PRESSURE, 0),
                    Self::ev(EV_KEY, BTN_TOUCH, 0),
                    Self::ev(EV_KEY, btn_tool, 0),
                    Self::ev(EV_SYN, SYN_REPORT, 0),
                ]);
                Err(e)
            }
        }
    }

    fn stroke_inner(
        &mut self,
        btn_tool: u16,
        press: i32,
        x0: f32,
        y0: f32,
        pts: &[(f32, f32)],
    ) -> io::Result<()> {

        // The pen appears in range above the first point and APPROACHES like a
        // hand: several hover samples with falling distance. A real pen never
        // teleports from out-of-range to touching in one frame, and xochitl's
        // pipeline visibly mistrusts one that does.
        let (rx, ry) = self.to_raw(x0, y0);
        self.send(&[
            Self::ev(EV_KEY, btn_tool, 1),
            Self::ev(EV_ABS, ABS_X, rx),
            Self::ev(EV_ABS, ABS_Y, ry),
            Self::ev(EV_ABS, ABS_DISTANCE, 70),
            Self::ev(EV_SYN, SYN_REPORT, 0),
        ])?;
        sleep(Duration::from_micros(self.dwell_us));
        for d in [50, 30, 15, 5] {
            self.send(&[
                Self::ev(EV_ABS, ABS_DISTANCE, d),
                Self::ev(EV_SYN, SYN_REPORT, 0),
            ])?;
            sleep(Duration::from_micros(self.dwell_us));
        }

        // Touch down, pressure ramping over two frames like a landing pen tip.
        self.send(&[
            Self::ev(EV_ABS, ABS_DISTANCE, 0),
            Self::ev(EV_KEY, BTN_TOUCH, 1),
            Self::ev(EV_ABS, ABS_PRESSURE, press / 3),
            Self::ev(EV_SYN, SYN_REPORT, 0),
        ])?;
        sleep(Duration::from_micros(self.dwell_us));
        self.send(&[
            Self::ev(EV_ABS, ABS_PRESSURE, press),
            Self::ev(EV_SYN, SYN_REPORT, 0),
        ])?;

        // Glide through densified points.
        let mut last = (x0, y0);
        for &(px, py) in pts {
            for (ix, iy) in densify(last, (px, py), 1.0 / 300.0) {
                let (rx, ry) = self.to_raw(ix, iy);
                self.send(&[
                    Self::ev(EV_ABS, ABS_X, rx),
                    Self::ev(EV_ABS, ABS_Y, ry),
                    Self::ev(EV_ABS, ABS_PRESSURE, press),
                    Self::ev(EV_SYN, SYN_REPORT, 0),
                ])?;
                sleep(Duration::from_micros(self.dwell_us));
            }
            last = (px, py);
        }

        // Lift with a pressure ramp, hover a beat, then leave range — and give
        // xochitl a real inter-stroke pause. A human's fastest stroke-to-stroke
        // gap is ~80 ms; ours must not be the place xochitl first meets 6 ms.
        self.send(&[
            Self::ev(EV_ABS, ABS_PRESSURE, press / 3),
            Self::ev(EV_SYN, SYN_REPORT, 0),
        ])?;
        sleep(Duration::from_micros(self.dwell_us));
        self.send(&[
            Self::ev(EV_ABS, ABS_PRESSURE, 0),
            Self::ev(EV_KEY, BTN_TOUCH, 0),
            Self::ev(EV_ABS, ABS_DISTANCE, 30),
            Self::ev(EV_SYN, SYN_REPORT, 0),
        ])?;
        sleep(Duration::from_micros(self.dwell_us * 2));
        self.send(&[
            Self::ev(EV_KEY, btn_tool, 0),
            Self::ev(EV_SYN, SYN_REPORT, 0),
        ])?;
        sleep(Duration::from_millis(80));
        Ok(())
    }
}

/// Intermediate points from `a` (exclusive) to `b` (inclusive), stepping ≤ `max_hop`.
fn densify(a: (f32, f32), b: (f32, f32), max_hop: f32) -> Vec<(f32, f32)> {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let dist = dx.hypot(dy);
    let n = (dist / max_hop).ceil().max(1.0) as usize;
    (1..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            (a.0 + dx * t, a.1 + dy * t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn injector(x: AbsInfo, y: AbsInfo) -> Injector {
        Injector {
            node: std::fs::OpenOptions::new()
                .write(true)
                .open("/dev/null")
                .expect("/dev/null"),
            x,
            y,
            pressure: AbsInfo {
                maximum: 4095,
                ..Default::default()
            },
            has_rubber: true,
            dwell_us: 0,
        }
    }

    fn ax(max: i32) -> AbsInfo {
        AbsInfo {
            minimum: 0,
            maximum: max,
            ..Default::default()
        }
    }

    /// to_raw must be the exact inverse of Transform::to_norm — inject a point,
    /// read it back through the forward transform, land where you started.
    #[test]
    fn inverse_roundtrips_through_the_forward_transform() {
        let inj = injector(ax(20966), ax(15725));
        let fwd = crate::transform::Transform::new(ax(20966), ax(15725));
        for &(sx, sy) in &[(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.25, 0.8), (0.9, 0.1)] {
            let (rx, ry) = inj.to_raw(sx, sy);
            let (bx, by) = fwd.to_norm(rx, ry);
            assert!(
                (bx - sx).abs() < 2e-4 && (by - sy).abs() < 2e-4,
                "({sx},{sy}) -> raw ({rx},{ry}) -> ({bx},{by})"
            );
        }
    }

    #[test]
    fn densify_never_hops_farther_than_the_cap() {
        let pts = densify((0.0, 0.0), (0.1, 0.0), 1.0 / 300.0);
        assert!(pts.len() >= 30);
        let mut last = (0.0, 0.0);
        for p in pts {
            assert!((p.0 - last.0).hypot(p.1 - last.1) <= 1.0 / 300.0 + 1e-6);
            last = p;
        }
        assert!((last.0 - 0.1).abs() < 1e-6);
    }
}
