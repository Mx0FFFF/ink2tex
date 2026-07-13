//! ink2tex-rm — the reMarkable 2 frontend. THE ONLY device-coupled code in the
//! repo (~500 LOC ceiling; see `docs/device.md`).
//!
//! Modes:
//!   --probe                        enumerate + ioctl-probe the digitizer, print ranges
//!   --record [--out P] [--dur S]   read the pen, save strokes as .ink (no drawing)
//!   --ink    [--out P] [--dur S]   same, but also draw live with a DU waveform (arm)
//!   --collect --symbol S [--count N]
//!                                  draw S repeatedly -> NDJSON training samples (the tokens
//!                                  no permissively-licensed dataset has: `=`, `(`, `)`)
//!   --expr [--from INK | --dur S]     write a whole EXPRESSION → LaTeX on stdout.
//!                                  Auto-orients landscape ink; uses the expression
//!                                  model (expr.iwt — `make deploy-expr`).
//!   --recognize [--from INK | --dur S] [--model M] [--labels L]
//!                                  draw (or load --from) a symbol → int8 CNN → top-5
//!                                  LaTeX on stdout (streamed back over SSH; no rm2fb).
//!                                  Weights resolve themselves — see `default_asset`.
//!
//! No `unwrap()`/`expect()` in runtime paths — a panic here is a dead screen.

mod capture;
#[cfg(target_arch = "arm")]
mod draw;
mod evdev;
mod serve;
mod transform;

use std::path::Path;
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
        Some("--collect") => collect(&args),
        Some("--expr") => expr(&args),
        Some("--serve") => serve_cmd(&args),
        Some("--record-one") => record_one_cmd(&args),
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
    eprintln!(
        "ink2tex {} — handwritten math → LaTeX, on the tablet, offline.",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!();
    eprintln!("usage: ink2tex [--probe | --record | --ink | --recognize] [options]");
    eprintln!("  --recognize [--from INK] [--dur S]");
    eprintln!("                               draw a symbol (or --from a file) -> top-5 LaTeX");
    eprintln!("  --probe                      find the digitizer, print its axis ranges");
    eprintln!("  --record  [--out P][--dur S]  capture pen strokes to an .ink file");
    eprintln!("  --ink     [--out P][--dur S]  capture AND draw live (device; needs rm2fb)");
    eprintln!("  --collect --symbol S [--count N] [--out P] [--idle-ms MS]");
    eprintln!("                               draw S over and over -> NDJSON training samples");
    eprintln!("  --expr    [--from INK][--dur S]  write an EXPRESSION -> LaTeX (uses expr.iwt)");
    eprintln!("  --serve   [--port P][--idle-ms MS]  correction UI at http://10.11.99.1:8222");
    eprintln!();
    eprintln!(
        "  --model M / --labels L       override the weights (default: /opt/usr/share/ink2tex,"
    );
    eprintln!("                               else alongside the binary)");
    eprintln!();
    eprintln!("Draw one symbol, get the five most likely LaTeX commands. No cloud, no network.");
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Where the model lives when nobody says. A packaged install must Just Work — a user
/// who typed `opkg install ink2tex` should be able to type `ink2tex --recognize`, not
/// have to know where opkg put the weights.
///
/// Searched in order: the Toltec install prefix, then **next to the binary**, which is
/// what makes a bare `scp`-to-`/home/root` deployment work too (`make deploy-model`).
/// An explicit `--model` always wins.
fn default_asset(args: &[String], flag_name: &str, file: &str) -> Result<String> {
    if let Some(p) = flag(args, flag_name) {
        return Ok(p);
    }
    let mut tried = vec![format!("/opt/usr/share/ink2tex/{file}")];
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        tried.push(dir.join(file).to_string_lossy().into_owned());
    }
    tried
        .iter()
        .find(|p| Path::new(p).is_file())
        .cloned()
        .with_context(|| {
            format!(
                "no {file} found (looked in {}); pass {flag_name} <path>",
                tried.join(", ")
            )
        })
}

/// `--device <path>` bypasses enumeration (used to point at a synthetic uinput pen).
fn device_arg(args: &[String]) -> Option<String> {
    flag(args, "--device")
}

fn capture_args(args: &[String]) -> (String, Duration) {
    let out = flag(args, "--out").unwrap_or_else(|| "/home/root/out.ink".to_string());
    let secs = flag(args, "--dur")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(60);
    (out, Duration::from_secs(secs))
}

/// Collect training samples for one symbol: draw it `--count` times, get NDJSON out.
///
/// ## Why this exists
/// Detexify has no `+`, `-`, `=`, digits or letters — it is a symbol-*lookup* corpus, and
/// nobody looks up how to type `2`. HWRT (ODbL) supplies 65 of the missing tokens, but
/// **`=`, `(` and `)` are in no permissively-licensed dataset that exists**: they live only in
/// CROHME and MathWriting, both CC BY-NC-SA, which must never enter a shipped binary. So the
/// equals sign has to be drawn. This is the tool that draws it — and the seed of the open,
/// permissively-licensed corpus DESIGN §5 argues does not yet exist and ought to.
///
/// Output is the same NDJSON the Detexify loader already eats, so collected ink feeds
/// straight into `--prepare-detexify --classes` with no new format and no second parser to
/// drift out of sync.
///
/// ## Systems concept: putting a *deadline* on a blocking read — and why `poll`, not `alarm`
/// A sample ends when the pen goes **idle**, not when a stroke ends: `=` is two strokes with
/// a short gap between them, `(` is one. So the loop has to ask "has anything happened in the
/// last N ms?" — a question a blocking `read()` cannot answer, because it only wakes when the
/// pen moves. `poll()` waits on the fd *with a timeout* and returns 0 when nothing arrived,
/// which is exactly that question. (An `alarm`/`SIGALRM` would also interrupt the read — and
/// see `run_capture` for how that goes wrong: `signal()` sets SA_RESTART and the read
/// silently resumes. `poll` has no such trap.)
fn collect(args: &[String]) -> Result<()> {
    use std::io::Write as _;

    let symbol = flag(args, "--symbol").context("--collect needs --symbol <latex>, e.g. '='")?;
    let count: usize = flag(args, "--count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let out_path = flag(args, "--out").unwrap_or_else(|| "/home/root/collected.ndjson".to_string());
    let idle_ms: u64 = flag(args, "--idle-ms")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500);
    let dev = device_arg(args);

    let dig = match dev.as_deref() {
        Some(p) => evdev::open_digitizer(p).with_context(|| format!("opening {p}"))?,
        None => evdev::find_digitizer().context("locating the pen digitizer")?,
    };

    // SAFETY: the handler only does an atomic store. `sa_flags = 0` for the same reason as in
    // `run_capture` — we want a syscall to fail rather than silently restart.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_stop as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        for sig in [libc::SIGINT, libc::SIGTERM] {
            libc::sigaction(sig, &sa, std::ptr::null_mut());
        }
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .with_context(|| format!("opening {out_path}"))?;

    eprintln!("collecting {count} samples of {symbol:?} → {out_path}");
    eprintln!("draw it, lift the pen, wait {idle_ms} ms. Ctrl-C stops and keeps what you have.");
    // Samples are stored as-drawn, and a lone glyph cannot orient itself (the expression
    // path's ballot needs a line of them). Both real collection sessions so far were
    // drawn landscape and needed rescue-rotation at ingest — say it up front instead.
    eprintln!("⚠ hold the tablet UPRIGHT (portrait). Landscape samples train a sideways model.\n");

    let mut saved = 0usize;
    while saved < count && !STOP.load(Ordering::SeqCst) {
        eprint!("  [{:>3}/{count}] draw  {symbol}  … ", saved + 1);
        let _ = std::io::stderr().flush();

        let ink = record_one(&dig, idle_ms)?;
        if STOP.load(Ordering::SeqCst) {
            eprintln!("(stopped)");
            break;
        }
        if ink.strokes.is_empty() {
            eprintln!("nothing captured — again");
            continue;
        }
        writeln!(file, "{}", ndjson_sample(&symbol, &ink))?;
        file.flush()?; // one at a time: a crash costs the last drawing, not the whole set
        saved += 1;
        eprintln!(
            "✓ {} stroke(s) / {} points",
            ink.strokes.len(),
            ink.point_count()
        );
    }

    eprintln!("\nwrote {saved} sample(s) of {symbol:?} to {out_path}");
    Ok(())
}

/// Record until the pen has been idle for `idle_ms` — *after* something was actually drawn.
fn record_one(dig: &evdev::Digitizer, idle_ms: u64) -> Result<Ink> {
    let mut cap = capture::Capture::from_axes(dig.x, dig.y, dig.pressure, dig.tilt_x, dig.tilt_y);
    let mut buf = [evdev::InputEvent::zeroed(); 64];
    let mut last_ink: Option<std::time::Instant> = None;

    loop {
        if STOP.load(Ordering::SeqCst) {
            break;
        }
        // Wake when the pen does something — or every 100 ms regardless, so the idle deadline
        // still fires while the pen sits perfectly still and sends nothing at all.
        let mut pfd = libc::pollfd {
            fd: dig.fd.raw(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid pollfd and a plain millisecond timeout.
        let ready = unsafe { libc::poll(&mut pfd, 1, 100) };
        if ready < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                break; // Ctrl-C
            }
            return Err(e).context("poll on the digitizer");
        }
        if ready > 0 {
            let n = match evdev::read_events(dig.fd.raw(), &mut buf) {
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => break,
                Err(e) => return Err(e).context("reading pen events"),
            };
            for ev in &buf[..n] {
                // A *latched point* — ink, not a hover. Hovering must not keep the sample
                // alive, or the deadline never fires while a hand rests near the screen.
                if cap.process(ev).is_some() {
                    last_ink = Some(std::time::Instant::now());
                }
            }
        }
        if let Some(t) = last_ink {
            if t.elapsed() >= std::time::Duration::from_millis(idle_ms) {
                break; // drawn, then quiet — that is one sample
            }
        }
    }
    Ok(cap.finish())
}

/// One NDJSON record in the exact shape `detexify.rs` already parses, so collected ink needs
/// no new format and no second loader to drift out of sync with the first.
///
/// Coordinates are the normalized ones `Capture` emits; nothing downstream cares about scale
/// any more (that is what made `global_features` dimensionless). `t` is milliseconds, which
/// is the unit the loader expects.
fn ndjson_sample(symbol: &str, ink: &Ink) -> String {
    let strokes: Vec<String> = ink
        .strokes
        .iter()
        .map(|s| {
            let pts: Vec<String> = s
                .points
                .iter()
                .map(|p| format!(r#"{{"x":{:.5},"y":{:.5},"t":{}}}"#, p.x, p.y, p.t_us / 1000))
                .collect();
            format!("[{}]", pts.join(","))
        })
        .collect();
    let key = symbol.replace('\\', "\\\\").replace('"', "\\\"");
    format!(r#"{{"key":"{key}","strokes":[{}]}}"#, strokes.join(","))
}

/// Capture one expression's ink for the server: record until the pen goes idle.
pub(crate) fn capture_expression(idle_ms: u64) -> Result<Ink> {
    let dig = evdev::find_digitizer().context("locating the pen digitizer")?;
    record_one(&dig, idle_ms)
}

/// Capture exactly one idle-terminated drawing and save it — the primitive the guided
/// expression-corpus collector (M2) drives over SSH, once per target expression.
fn record_one_cmd(args: &[String]) -> Result<()> {
    let out = flag(args, "--out").unwrap_or_else(|| "/home/root/one.ink".to_string());
    let idle_ms: u64 = flag(args, "--idle-ms").and_then(|s| s.parse().ok()).unwrap_or(2000);
    let dig = match device_arg(args).as_deref() {
        Some(p) => evdev::open_digitizer(p).with_context(|| format!("opening {p}"))?,
        None => evdev::find_digitizer().context("locating the pen digitizer")?,
    };
    let ink = record_one(&dig, idle_ms)?;
    std::fs::write(&out, ink.encode()).with_context(|| format!("writing {out}"))?;
    eprintln!("saved {} strokes / {} points to {out}", ink.strokes.len(), ink.point_count());
    Ok(())
}

/// M4: serve the correction UI over usb0. See `serve.rs` — this is just argument plumbing.
fn serve_cmd(args: &[String]) -> Result<()> {
    let model = default_asset(args, "--model", "expr.iwt")?;
    let labels = default_asset(args, "--labels", "expr.labels.txt")?;
    let counts = default_asset(args, "--counts", "expr.counts.txt")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.lines().filter_map(|l| l.trim().parse().ok()).collect());
    let port: u16 = flag(args, "--port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8222);
    let idle_ms: u64 = flag(args, "--idle-ms")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500);
    let log = flag(args, "--log").unwrap_or_else(|| "/home/root/corrections.ndjson".to_string());
    serve::Server::new(&model, &labels, counts, &log)?.run(port, idle_ms)
}

/// Recognize a whole EXPRESSION: capture (or `--from`) a line of math, run the full
/// core pipeline — auto-orientation, denoising, segmentation (stacked-bar `=` merge),
/// per-symbol int8 classification over the expression vocabulary with the training-prior
/// correction, 2-D structure — and print LaTeX. Everything happens on this CPU.
///
/// Uses the *expression* model (`expr.iwt` + labels + counts, deployed by
/// `make deploy-expr`), not the M1 lookup model: the lookup model has no digits, letters
/// or operators, and expression ranking is masked to ~190 expression-plausible tokens —
/// the two modes answer different questions (DESIGN §4.3).
fn expr(args: &[String]) -> Result<()> {
    use ink2tex_core::classify::{Labels, Weights};
    use ink2tex_core::latex::symbol_command;

    let model = default_asset(args, "--model", "expr.iwt")?;
    let labels_path = default_asset(args, "--labels", "expr.labels.txt")?;
    let ink = match flag(args, "--from") {
        Some(path) => {
            let bytes = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
            Ink::decode(&bytes).with_context(|| format!("parsing {path} as .ink"))?
        }
        None => {
            let dur = flag(args, "--dur")
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(Duration::from_secs(30));
            let dev = device_arg(args);
            run_capture(
                dur,
                "expr (write one line of math)",
                dev.as_deref(),
                |_seg| {},
            )?
        }
    };

    let blob = std::fs::read(&model).with_context(|| format!("reading model {model}"))?;
    let weights = Weights::parse(&blob).context("parsing expr model .iwt")?;
    let labels = Labels::from_lines(
        &std::fs::read_to_string(&labels_path)
            .with_context(|| format!("reading labels {labels_path}"))?,
    );
    // Counts unlock the training-prior correction; without them `x` loses to `\chi`
    // (958 lookup-corpus samples vs 59). Degrade politely rather than refuse.
    let counts: Option<Vec<u32>> = default_asset(args, "--counts", "expr.counts.txt")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.lines().filter_map(|l| l.trim().parse().ok()).collect());
    if counts.is_none() {
        eprintln!("(no expr.counts.txt — prior correction off, expect worse letters/digits)");
    }

    let t0 = std::time::Instant::now();
    let latex = ink2tex_core::recognize_expression(&ink, &weights, &labels, counts.as_deref(), 3)
        .context("expression recognizer")?;
    let (_oriented, line) =
        ink2tex_core::recognize_line(&ink, &weights, &labels, counts.as_deref(), 3)
            .context("per-symbol ranking")?;
    eprintln!(
        "recognized {} symbol(s) in {:.0} ms (incl. orientation ballot if held)",
        line.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );

    println!("LaTeX: {latex}");
    for (i, s) in line.iter().enumerate() {
        let alts: Vec<String> = s
            .predictions
            .iter()
            .map(|p| {
                labels
                    .get(p.class)
                    .map(|l| format!("{} {:.0}%", symbol_command(l), p.prob * 100.0))
                    .unwrap_or_else(|| format!("class{} ", p.class))
            })
            .collect();
        println!("  {}. {}", i + 1, alts.join("  |  "));
    }
    Ok(())
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
    device: Option<&str>,
    mut on_segment: impl FnMut(&capture::Segment),
) -> Result<Ink> {
    let dig = match device {
        Some(p) => evdev::open_digitizer(p).with_context(|| format!("opening {p}"))?,
        None => evdev::find_digitizer().context("locating the pen digitizer")?,
    };
    eprintln!(
        "digitizer: {} ({}) — mode: {}",
        dig.path, dig.name, mode_label
    );

    // SAFETY: handlers only do an atomic store (async-signal-safe); alarm is a plain
    // timer. All process-global, but this is a single-threaded binary.
    //
    // ## Systems concept: SA_RESTART, and why `signal()` is a trap here
    // This *must* be `sigaction` with `sa_flags = 0`, not `signal()`. glibc's `signal()`
    // gives you BSD semantics — it sets **SA_RESTART** — so when the alarm fires the
    // kernel silently *restarts* the blocked `read()` instead of failing it with `EINTR`.
    // The handler still sets STOP, but the loop below is parked inside `read()` and never
    // gets to look at it. The app then hangs forever with the pen idle, which is exactly
    // the case `--dur` exists for. (This was not theoretical: it hung on the tablet, and
    // only ever looked fine because a still-moving pen kept waking the read.)
    // With `sa_flags = 0` the `read()` returns `EINTR`, we break, and the ink gets saved.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_stop as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0; // NOT SA_RESTART — we *want* read() to fail with EINTR
        for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGALRM] {
            libc::sigaction(sig, &sa, std::ptr::null_mut());
        }
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
    let dev = device_arg(args);
    let (out, dur) = capture_args(args);
    let ink = run_capture(dur, "record (no drawing)", dev.as_deref(), |_seg| {})?;
    save(&ink, &out)
}

/// Recognize a symbol: draw one (or load `--from` a captured `.ink`), rasterize it,
/// run the int8 CNN from the deployed model, and print the top-5 LaTeX candidates.
/// This is a thin wrapper over the device-free `core::classify` — all the real work
/// (rasterize, quantize, conv/dense, softmax) is in core.
fn recognize(args: &[String]) -> Result<()> {
    let dev = device_arg(args);
    use ink2tex_core::classify::{
        global_features, online_features, rasterize, Labels, Weights, ONLINE_POINTS,
    };
    use ink2tex_core::latex::symbol_command;

    let model = default_asset(args, "--model", "model.iwt")?;
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
            run_capture(
                dur,
                "recognize (draw one symbol)",
                dev.as_deref(),
                |_seg| {},
            )?
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

    // Labels sit beside the model, so they resolve the same way. Missing labels degrade
    // to bare class indices rather than failing — a top-5 of numbers still beats an error.
    let labels = match default_asset(args, "--labels", "model.labels.txt") {
        Ok(p) => Some(Labels::from_lines(
            &std::fs::read_to_string(&p).with_context(|| format!("reading labels {p}"))?,
        )),
        Err(_) => None,
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
    let dev = device_arg(args);
    let (out, dur) = capture_args(args);
    let mut screen = draw::Screen::open();
    screen.clear();
    let ink = run_capture(dur, "ink (live DU draw)", dev.as_deref(), |seg| {
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
