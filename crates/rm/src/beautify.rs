//! The beautifier daemon (`--beautify`): write sloppy math, **hold the pen still**,
//! watch it become typeset — reMarkable's snap-to-shape gesture, for formulas.
//!
//! One cycle:
//!   1. capture strokes as the human writes (same evdev tap as everything else);
//!   2. the TRIGGER is a hold: pen in contact, not moving, for ~1.2 s — the exact
//!      gesture xochitl uses for shape-snapping, so the hand already knows it;
//!   3. recognize the expression (minus the hold-dot itself), build the SLT;
//!   4. ERASE the sloppy ink: retrace every captured stroke with the RUBBER tool —
//!      xochitl removes it like a hand-held eraser would;
//!   5. REWRITE: scale the typeset stroke plan (Hershey glyphs + rules) into the
//!      region the ink occupied and trace it with the PEN tool.
//!
//! xochitl renders and persists both steps — the beautified formula is real
//! notebook content. While injecting we are also *reading* the node (fan-out
//! delivers our own strokes back), so injection happens with capture paused and
//! the queue drained afterwards; otherwise the daemon would try to beautify its
//! own handwriting, forever.
//!
//! Cohabitation caveats, honestly: strokes render with whatever pen/eraser the
//! human last selected (fineliner recommended; the default eraser works). And a
//! hold at the END of a glyph stroke triggers too — finish the formula, lift,
//! then press-and-hold on free space.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ink2tex_core::classify::{Labels, Weights};
use ink2tex_core::{analyze, compose_slt, typeset, Ink};

use crate::evdev::{self, Digitizer, InputEvent};
use crate::inject::{Injector, Tool};
use crate::STOP;

/// Contact counts as "held still" when it stays within this box (normalized
/// screen units; ~0.8% ≈ 11 px) …
const HOLD_EPS: f32 = 0.008;
/// … for at least this long.
const HOLD_MS: u64 = 1200;

pub struct Beautifier {
    weights: Weights<'static>,
    labels: Labels,
    counts: Option<Vec<u32>>,
}

impl Beautifier {
    pub fn new(model: &str, labels_path: &str, counts: Option<Vec<u32>>) -> Result<Self> {
        let blob: &'static [u8] = Box::leak(
            std::fs::read(model)
                .with_context(|| format!("reading {model}"))?
                .into_boxed_slice(),
        );
        Ok(Self {
            weights: Weights::parse(blob).context("parsing expr model")?,
            labels: Labels::from_lines(
                &std::fs::read_to_string(labels_path)
                    .with_context(|| format!("reading {labels_path}"))?,
            ),
            counts,
        })
    }

    pub fn run(&self) -> Result<()> {
        let dig = evdev::find_digitizer().context("locating the pen digitizer")?;
        let mut inj = Injector::open(&dig).context("opening digitizer for write")?;
        eprintln!("beautifier ready: write a formula, then HOLD the pen still to snap it");
        while !STOP.load(Ordering::SeqCst) {
            let Some(ink) = capture_until_hold(&dig)? else {
                continue; // spurious hold with no writing before it
            };
            eprintln!(
                "hold detected — {} strokes captured (+ the hold itself)",
                ink.strokes.len() - 1
            );
            match self.beautify(&mut inj, &ink) {
                Ok(latex) => eprintln!("beautified: {latex}"),
                Err(e) => eprintln!("beautify failed (ink left untouched): {e:#}"),
            }
            drain(&dig)?; // our own injected strokes echo back; discard them
        }
        Ok(())
    }

    /// The transaction: recognize FIRST (so a failure leaves the page untouched),
    /// only then erase and rewrite. `ink`'s LAST stroke is the hold gesture: it
    /// is excluded from recognition (it is a trigger, not content) but included
    /// in the erase — the first live run left a dark dot at every hold because
    /// the erase pass never saw the popped stroke.
    fn beautify(&self, inj: &mut Injector, ink: &Ink) -> Result<String> {
        let mut expr = ink.clone();
        // The last stroke carries the hold gesture. If it is a pure dot it is
        // trigger-only; if the human held at the END of a writing stroke, that
        // stroke is real content and must still be recognized (either way the
        // erase pass covers it — it is part of `ink`).
        if let Some(last) = expr.strokes.last() {
            let path_len: f32 = last
                .points
                .windows(2)
                .map(|w| (w[1].x - w[0].x).hypot(w[1].y - w[0].y))
                .sum();
            if path_len < 3.0 * HOLD_EPS {
                expr.strokes.pop();
            }
        }
        if expr.strokes.is_empty() {
            anyhow::bail!("nothing but the hold itself was written");
        }
        let (_oriented, symbols) = analyze(
            &expr,
            &self.weights,
            &self.labels,
            self.counts.as_deref(),
            5,
        )?;
        let choices = vec![0usize; symbols.len()];
        let slt = compose_slt(&symbols, &choices);
        let plan = typeset::to_strokes(&slt);
        if plan.polylines.is_empty() {
            anyhow::bail!("stroke plan came back empty");
        }
        if !plan.missing.is_empty() {
            eprintln!("  (no glyph yet for {:?} — leaving gaps)", plan.missing);
        }
        let latex = ink2tex_core::compose(&symbols, &choices).0;

        if !inj.has_rubber {
            anyhow::bail!("digitizer has no eraser tool — beautify would double-draw");
        }
        // Let xochitl finish rendering the human's last stroke (and the hold)
        // before the eraser arrives — erasing mid-render loses the race.
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Erase: retrace every original stroke — including the hold-dot — with
        // the rubber, three passes with slight vertical offsets so the eraser's
        // disk covers the full drawn line width, not just its spine. Original
        // screen coordinates, not the oriented ones.
        for dy in [0.0f32, 0.004, -0.004] {
            for stroke in &ink.strokes {
                let pts: Vec<(f32, f32)> =
                    stroke.points.iter().map(|p| (p.x, p.y + dy)).collect();
                inj.stroke(Tool::Rubber, &pts)?;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(300));

        // Rewrite: fit the plan into the box the sloppy ink occupied — same
        // height, left-aligned, centred vertically. Height is the anchor because
        // a typeset line is usually WIDER than cramped handwriting, and running
        // off the right edge is worse than being slightly small.
        let (x0, y0, _x1, y1) = ink_bbox(&expr);
        // Match the handwriting's height, but never run off the right edge and
        // never go degenerate: a formula written at the far right margin must
        // come back small, not mirrored (a negative min() here once flipped the
        // whole rewrite upside-down).
        let room_right = (0.96 - x0).max(0.05);
        let scale = ((y1 - y0) / plan.h.max(1.0))
            .min(room_right / plan.w.max(1.0))
            .max(1e-5);
        let oy = y0 + ((y1 - y0) - plan.h * scale) / 2.0;
        for pl in &plan.polylines {
            let pts: Vec<(f32, f32)> = pl
                .iter()
                .map(|&(px, py)| (x0 + px * scale, oy + py * scale))
                .collect();
            inj.stroke(Tool::Pen, &pts)?;
        }
        Ok(latex)
    }
}

fn ink_bbox(ink: &Ink) -> (f32, f32, f32, f32) {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for s in &ink.strokes {
        for p in &s.points {
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
    }
    (x0, y0, x1, y1)
}

/// Capture strokes until the pen is HELD still in contact. Returns the ink with
/// the hold stroke removed — or None if the hold arrived before any writing.
fn capture_until_hold(dig: &Digitizer) -> Result<Option<Ink>> {
    let mut cap =
        crate::capture::Capture::from_axes(dig.x, dig.y, dig.pressure, dig.tilt_x, dig.tilt_y);
    let mut buf = [InputEvent::zeroed(); 64];
    let mut anchor: Option<((f32, f32), Instant)> = None;

    loop {
        if STOP.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let mut pfd = libc::pollfd {
            fd: dig.fd.raw(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid pollfd, millisecond timeout.
        let ready = unsafe { libc::poll(&mut pfd, 1, 50) };
        if ready < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                return Ok(None);
            }
            return Err(e).context("poll on the digitizer");
        }
        if ready > 0 {
            let n = match evdev::read_events(dig.fd.raw(), &mut buf) {
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => return Ok(None),
                Err(e) => return Err(e).context("reading pen events"),
            };
            for ev in &buf[..n] {
                if let Some(seg) = cap.process(ev) {
                    // A latched ink point. Movement resets the hold anchor;
                    // stillness lets it age toward the trigger.
                    let p = seg.to;
                    match &anchor {
                        Some((a, _)) if (p.0 - a.0).hypot(p.1 - a.1) > HOLD_EPS => {
                            anchor = Some((p, Instant::now()));
                        }
                        None => anchor = Some((p, Instant::now())),
                        _ => {}
                    }
                }
            }
        }
        // The pen leaving contact resets the hold — a hold spans one press.
        if !cap.touching() {
            anchor = None;
        }
        if let Some((_, since)) = anchor {
            if cap.touching() && since.elapsed() >= Duration::from_millis(HOLD_MS) {
                // Wait (bounded) for the lift so the hold stroke closes.
                let lift_t0 = Instant::now();
                while cap.touching()
                    && !STOP.load(Ordering::SeqCst)
                    && lift_t0.elapsed() < Duration::from_secs(10)
                {
                    let ready = unsafe { libc::poll(&mut pfd, 1, 50) };
                    if ready > 0 {
                        match evdev::read_events(dig.fd.raw(), &mut buf) {
                            Ok(n) => {
                                for ev in &buf[..n] {
                                    cap.process(ev);
                                }
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => break,
                            Err(e) => return Err(e).context("reading pen events"),
                        }
                    }
                }
                let ink = cap.finish();
                if ink.strokes.len() < 2 {
                    return Ok(None); // just the hold — nothing written before it
                }
                return Ok(Some(ink)); // last stroke = the hold; beautify() knows
            }
        }
    }
}

/// Discard whatever is queued on the node (our own injected echo).
fn drain(dig: &Digitizer) -> Result<()> {
    let mut buf = [InputEvent::zeroed(); 64];
    loop {
        let mut pfd = libc::pollfd {
            fd: dig.fd.raw(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid pollfd, zero timeout — a pure "anything left?" probe.
        let ready = unsafe { libc::poll(&mut pfd, 1, 0) };
        if ready < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue; // EINTR is not "queue empty" — ask again
            }
            return Err(e).context("polling during drain");
        }
        if ready == 0 {
            return Ok(());
        }
        if let Err(e) = evdev::read_events(dig.fd.raw(), &mut buf) {
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e).context("draining injected echo");
        }
    }
}
