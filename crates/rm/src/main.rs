//! ink2tex-rm — the reMarkable 2 frontend. THE ONLY device-coupled code in the
//! repo (~500 LOC ceiling; see `.claude/rules/device.md`).
//!
//! Modes:
//!   --probe                        enumerate + ioctl-probe the digitizer, print ranges
//!   --record [--out P] [--dur S]   read the pen, save strokes as .ink (no drawing)
//!   --ink    [--out P] [--dur S]   same, but also draw live with a DU waveform (arm)
//!   --recognize --model M [--labels L] [--from INK | --dur S]
//!                                  draw (or load --from) a symbol → int8 CNN → top-5
//!                                  LaTeX on stdout (streamed back over SSH; no rm2fb)
//!
//! No `unwrap()`/`expect()` in runtime paths — a panic here is a dead screen.

mod capture;
#[cfg(target_arch = "arm")]
mod draw;
mod evdev;
mod transform;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use ink2tex_core::Ink;

/// Set from a signal handler (SIGINT/SIGTERM) or the `--dur` alarm (SIGALRM) to
/// break the blocking read loop and finish cleanly.
static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_stop(_sig: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--probe") | None => probe(),
        Some("--record") => record(&args),
        Some("--ink") => ink(&args),
        Some("--recognize") => recognize(&args),
        Some("-h") | Some("--help") => {
            print_usage();
            Ok(())
        }
        Some(other) => {
            eprintln!("ink2tex-rm: unknown argument '{other}'");
            print_usage();
            std::process::exit(2);
        }
    }
}

fn print_usage() {
    eprintln!("usage: ink2tex-rm [--probe | --record | --ink | --recognize] [options]");
    eprintln!("  --probe                      find the digitizer, print its axis ranges");
    eprintln!("  --record  [--out P][--dur S]  capture pen strokes to an .ink file");
    eprintln!("  --ink     [--out P][--dur S]  capture AND draw live (device; needs rm2fb)");
    eprintln!("  --recognize --model M [--labels L] [--from INK | --dur S]");
    eprintln!("                               draw (or --from a file) a symbol -> top-5 LaTeX");
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn capture_args(args: &[String]) -> (String, Duration) {
    let out = flag(args, "--out").unwrap_or_else(|| "/home/root/out.ink".to_string());
    let secs = flag(args, "--dur")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(60);
    (out, Duration::from_secs(secs))
}

/// Enumerate `/dev/input/event*`, find the pen digitizer by capability, and dump
/// the axis ranges the kernel reports (fills DEVICE FACTS row 3, grounds the transform).
fn probe() -> Result<()> {
    let d = evdev::find_digitizer().context("locating the pen digitizer")?;
    println!("digitizer node : {}", d.path);
    println!("device name    : {}", d.name);
    let show = |label: &str, a: &evdev::AbsInfo| {
        println!(
            "  {label:<13} min={:<6} max={:<6} fuzz={} flat={} res={}",
            a.minimum, a.maximum, a.fuzz, a.flat, a.resolution
        );
    };
    show("ABS_X", &d.x);
    show("ABS_Y", &d.y);
    show("ABS_PRESSURE", &d.pressure);
    show("ABS_TILT_X", &d.tilt_x);
    show("ABS_TILT_Y", &d.tilt_y);
    show("ABS_DISTANCE", &d.distance);
    let (dw, dh) = (
        (d.x.maximum - d.x.minimum) as f64,
        (d.y.maximum - d.y.minimum) as f64,
    );
    if dw > 0.0 && dh > 0.0 {
        println!(
            "  digitizer aspect = {:.3}  (rM2 display 1404x1872 = {:.3})",
            dw / dh,
            1872.0 / 1404.0
        );
    }
    Ok(())
}

/// Shared read loop: block on pen events, feed the capture state machine, invoke
/// `on_segment` for each latched point (drawing or not), and return the captured ink.
///
/// ## Systems concept: interrupting a blocking read
/// `read()` on the evdev fd blocks until the pen moves. We install SIGINT/SIGTERM
/// handlers and an `alarm(dur)` (→ SIGALRM); when any fires, the in-flight `read`
/// returns `EINTR`, which we treat as "stop" — so the app exits promptly whether the
/// human hits Ctrl-C or the duration elapses, even with the pen idle.
fn run_capture(
    duration: Duration,
    mode_label: &str,
    mut on_segment: impl FnMut(&capture::Segment),
) -> Result<Ink> {
    let dig = evdev::find_digitizer().context("locating the pen digitizer")?;
    eprintln!(
        "digitizer: {} ({}) — mode: {}",
        dig.path, dig.name, mode_label
    );

    // SAFETY: handlers only do an atomic store (async-signal-safe); alarm is a plain
    // timer. All process-global, but this is a single-threaded binary.
    unsafe {
        libc::signal(libc::SIGINT, on_stop as extern "C" fn(libc::c_int) as usize);
        libc::signal(
            libc::SIGTERM,
            on_stop as extern "C" fn(libc::c_int) as usize,
        );
        libc::signal(
            libc::SIGALRM,
            on_stop as extern "C" fn(libc::c_int) as usize,
        );
        libc::alarm(duration.as_secs().min(u32::MAX as u64) as libc::c_uint);
    }

    let mut cap = capture::Capture::from_axes(dig.x, dig.y, dig.pressure, dig.tilt_x, dig.tilt_y);
    eprintln!(
        "capturing up to {}s — draw on the tablet (Ctrl-C to stop early)...",
        duration.as_secs()
    );

    let mut buf = [evdev::InputEvent::zeroed(); 64];
    while !STOP.load(Ordering::SeqCst) {
        let n = match evdev::read_events(dig.fd.raw(), &mut buf) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => break, // signal/alarm
            Err(e) => return Err(e).context("reading pen events"),
        };
        for ev in &buf[..n] {
            if let Some(seg) = cap.process(ev) {
                on_segment(&seg);
            }
        }
    }
    Ok(cap.finish())
}

fn save(ink: &Ink, out: &str) -> Result<()> {
    std::fs::write(out, ink.encode()).with_context(|| format!("saving {out}"))?;
    eprintln!(
        "saved {} strokes / {} points to {out}",
        ink.strokes.len(),
        ink.point_count()
    );
    Ok(())
}

fn record(args: &[String]) -> Result<()> {
    let (out, dur) = capture_args(args);
    let ink = run_capture(dur, "record (no drawing)", |_seg| {})?;
    save(&ink, &out)
}

/// Recognize a symbol: draw one (or load `--from` a captured `.ink`), rasterize it,
/// run the int8 CNN from the deployed model, and print the top-5 LaTeX candidates.
/// This is a thin wrapper over the device-free `core::classify` — all the real work
/// (rasterize, quantize, conv/dense, softmax) is in core.
fn recognize(args: &[String]) -> Result<()> {
    use ink2tex_core::classify::{
        global_features, online_features, rasterize, Labels, Weights, ONLINE_POINTS,
    };
    use ink2tex_core::latex::symbol_command;

    let model = flag(args, "--model").context("--recognize needs --model <iwt>")?;
    let ink = match flag(args, "--from") {
        Some(path) => {
            let bytes = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
            Ink::decode(&bytes).with_context(|| format!("parsing {path} as .ink"))?
        }
        None => {
            let dur = flag(args, "--dur")
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(Duration::from_secs(20));
            run_capture(dur, "recognize (draw one symbol)", |_seg| {})?
        }
    };

    let blob = std::fs::read(&model).with_context(|| format!("reading model {model}"))?;
    let weights = Weights::parse(&blob).context("parsing model .iwt")?;
    let t0 = std::time::Instant::now();
    let bitmap = rasterize(&ink.strokes, 32);
    let feats = global_features(&ink.strokes);
    let online = online_features(&ink.strokes, ONLINE_POINTS);
    let preds = ink2tex_core::classify::recognize(&weights, &bitmap, &feats, &online, 32, 5)
        .context("classifier forward pass")?;
    eprintln!(
        "inference: {:.2} ms (rasterize + int8 CNN)",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let labels = match flag(args, "--labels") {
        Some(p) => Some(Labels::from_lines(
            &std::fs::read_to_string(&p).with_context(|| format!("reading labels {p}"))?,
        )),
        None => None,
    };
    eprintln!("recognized {} strokes:", ink.strokes.len());
    for (i, p) in preds.iter().enumerate() {
        // What the user wants is the LaTeX, not Detexify's internal symbolId. Keep the id
        // alongside it — it is what tells look-alike classes apart when a result surprises
        // you (there are, for instance, two indistinguishable vertical-bar classes).
        match labels.as_ref().and_then(|l| l.get(p.class)) {
            Some(id) => println!(
                "  {}. {:>5.1}%  {:<16} ({id})",
                i + 1,
                p.prob * 100.0,
                symbol_command(id)
            ),
            None => println!("  {}. {:>5.1}%  class {}", i + 1, p.prob * 100.0, p.class),
        }
    }
    Ok(())
}

#[cfg(target_arch = "arm")]
fn ink(args: &[String]) -> Result<()> {
    let (out, dur) = capture_args(args);
    let mut screen = draw::Screen::open();
    screen.clear();
    let ink = run_capture(dur, "ink (live DU draw)", |seg| {
        // First point of a stroke has no predecessor — draw a dot (from == to).
        let from = seg.from.unwrap_or(seg.to);
        screen.ink_segment(from, seg.to, seg.pressure);
    })?;
    save(&ink, &out)
}

#[cfg(not(target_arch = "arm"))]
fn ink(_args: &[String]) -> Result<()> {
    anyhow::bail!("--ink is device-only (needs the framebuffer); use --record or --recognize")
}
