//! The correction UI, served by the tablet (M4).
//!
//! DESIGN §7: "the correction UI *is* the product; the model just makes it fast." The
//! roadmap wanted it on the E-Ink panel — that path is blocked on this firmware (rm2fb
//! soft-bricks > 3.3.2, see docs/device.md), and M4's own spec names the alternative:
//! an **HTTP endpoint on `usb0`**. So the tablet serves a self-contained page at
//! `http://10.11.99.1:8222`: press *Capture*, write a line of math on the tablet, and the
//! browser shows the typeset result with every symbol's top-k as one-tap corrections.
//!
//! Everything meaningful happens in `core` (`analyze`, `compose`, the typesetter); this
//! module is plumbing: a `std`-only HTTP loop — no framework, no async, one connection at
//! a time, which is exactly right for an appliance with one user. Query-string parameters
//! only, so there is no request-body parsing to get wrong.
//!
//! **Every fix is a labelled training example** (the M4 flywheel): a correction appends
//! the symbol's strokes + its human-chosen label to `corrections.ndjson` — the same
//! NDJSON the training pipeline already ingests — and *Accept* logs the whole expression
//! as confirmed ground truth.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use anyhow::{Context, Result};
use ink2tex_core::classify::{Labels, Weights};
use ink2tex_core::latex::symbol_command;
use ink2tex_core::{analyze, compose, AnalyzedSymbol, Ink};

const UI: &str = include_str!("ui.html");

struct Session {
    oriented: Ink,
    symbols: Vec<AnalyzedSymbol>,
    choices: Vec<usize>,
}

pub struct Server {
    weights: Weights<'static>,
    labels: Labels,
    counts: Option<Vec<u32>>,
    log_path: String,
    session: Option<Session>,
}

impl Server {
    pub fn new(
        model: &str,
        labels_path: &str,
        counts: Option<Vec<u32>>,
        log_path: &str,
    ) -> Result<Self> {
        // The server lives for the process: leaking the weight blob buys 'static borrows
        // without an ownership dance. 172 KB, once.
        let blob: &'static [u8] = Box::leak(
            std::fs::read(model)
                .with_context(|| format!("reading {model}"))?
                .into_boxed_slice(),
        );
        Ok(Server {
            weights: Weights::parse(blob).context("parsing expr model")?,
            labels: Labels::from_lines(
                &std::fs::read_to_string(labels_path)
                    .with_context(|| format!("reading {labels_path}"))?,
            ),
            counts,
            log_path: log_path.to_string(),
            session: None,
        })
    }

    pub fn run(&mut self, port: u16, idle_ms: u64) -> Result<()> {
        let listener = TcpListener::bind(("0.0.0.0", port))
            .with_context(|| format!("binding 0.0.0.0:{port}"))?;
        eprintln!("correction UI: http://10.11.99.1:{port}  (Ctrl-C to stop)");
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            if let Err(e) = self.handle(&mut stream, idle_ms) {
                eprintln!("request error: {e:#}");
                let _ = respond(&mut stream, 500, "text/plain", e.to_string().as_bytes());
            }
        }
        Ok(())
    }

    fn handle(&mut self, stream: &mut TcpStream, idle_ms: u64) -> Result<()> {
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf)?;
        let req = String::from_utf8_lossy(&buf[..n]);
        let line = req.lines().next().unwrap_or_default();
        let mut parts = line.split_whitespace();
        let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        let q = |k: &str| -> Option<String> {
            query.split('&').find_map(|kv| {
                kv.split_once('=')
                    .filter(|(key, _)| *key == k)
                    .map(|(_, v)| v.to_string())
            })
        };

        match (method, path) {
            ("GET", "/") => respond(stream, 200, "text/html; charset=utf-8", UI.as_bytes()),
            ("POST", "/capture") => {
                // Blocks until the pen goes idle — the browser's fetch waits alongside.
                let ink = crate::capture_expression(idle_ms)?;
                let (oriented, symbols) =
                    analyze(&ink, &self.weights, &self.labels, self.counts.as_deref(), 5)?;
                let choices = vec![0; symbols.len()];
                let json = self.state_json(&symbols, &choices);
                self.session = Some(Session {
                    oriented,
                    symbols,
                    choices,
                });
                respond(stream, 200, "application/json", json.as_bytes())
            }
            ("POST", "/correct") => {
                let i: usize = q("i").and_then(|v| v.parse().ok()).context("i")?;
                let c: usize = q("c").and_then(|v| v.parse().ok()).context("c")?;
                let Some(s) = self.session.as_mut() else {
                    return respond(stream, 409, "text/plain", b"no capture yet");
                };
                if i >= s.symbols.len() || c >= s.symbols[i].candidates.len() {
                    return respond(stream, 400, "text/plain", b"index out of range");
                }
                let was = s.choices[i];
                s.choices[i] = c;
                let log_path = self.log_path.clone();
                if c != was {
                    // The flywheel: this exact ink now has a human-confirmed label.
                    let (label, _) = &s.symbols[i].candidates[c];
                    log_sample(&log_path, &s.oriented, &s.symbols[i], label)?;
                }
                let symbols = std::mem::take(&mut s.symbols);
                let choices = s.choices.clone();
                let json = self.state_json(&symbols, &choices);
                if let Some(s) = self.session.as_mut() {
                    s.symbols = symbols; // put them back — take() avoided a self-borrow clash
                }
                respond(stream, 200, "application/json", json.as_bytes())
            }
            ("POST", "/accept") => {
                let Some(s) = self.session.as_ref() else {
                    return respond(stream, 409, "text/plain", b"no capture yet");
                };
                // Whole expression confirmed: every symbol's final label is ground truth.
                for (sym, &choice) in s.symbols.iter().zip(&s.choices) {
                    if let Some((label, _)) = sym.candidates.get(choice) {
                        log_sample(&self.log_path, &s.oriented, sym, label)?;
                    }
                }
                respond(stream, 200, "text/plain", b"logged")
            }
            ("GET", "/tex") => {
                let body = self
                    .session
                    .as_ref()
                    .map(|s| compose(&s.symbols, &s.choices).0)
                    .unwrap_or_default();
                respond(stream, 200, "text/plain; charset=utf-8", body.as_bytes())
            }
            _ => respond(stream, 404, "text/plain", b"not found"),
        }
    }

    fn state_json(&self, symbols: &[AnalyzedSymbol], choices: &[usize]) -> String {
        let (latex, svg) = compose(symbols, choices);
        let mut out = String::from("{");
        out.push_str(&format!(
            "\"latex\":{},\"svg\":{},\"symbols\":[",
            js_str(&latex),
            js_str(&svg)
        ));
        for (i, s) in symbols.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("{{\"choice\":{},\"candidates\":[", choices[i]));
            for (j, (label, prob)) in s.candidates.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "{{\"cmd\":{},\"p\":{:.3}}}",
                    js_str(&symbol_command(label)),
                    prob
                ));
            }
            out.push_str("]}");
        }
        out.push_str("]}");
        out
    }
}

/// One corrected/confirmed symbol → one training sample, in the exact NDJSON shape the
/// pipeline already ingests (`train/collected/`). Coordinates are the oriented ink's.
fn log_sample(path: &str, oriented: &Ink, sym: &AnalyzedSymbol, label: &str) -> Result<()> {
    use std::fmt::Write as _;
    let mut strokes = String::from("[");
    for (si, &idx) in sym.stroke_indices.iter().enumerate() {
        if si > 0 {
            strokes.push(',');
        }
        strokes.push('[');
        for (pi, p) in oriented.strokes[idx].points.iter().enumerate() {
            if pi > 0 {
                strokes.push(',');
            }
            let _ = write!(
                strokes,
                r#"{{"x":{:.5},"y":{:.5},"t":{}}}"#,
                p.x,
                p.y,
                p.t_us / 1000
            );
        }
        strokes.push(']');
    }
    strokes.push(']');
    let line = format!("{{\"key\":{},\"strokes\":{strokes}}}\n", js_str(label));
    use std::io::Write as _;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(line.as_bytes())?;
    Ok(())
}

fn js_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn respond(stream: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> Result<()> {
    let status = match code {
        200 => "200 OK",
        400 => "400 Bad Request",
        404 => "404 Not Found",
        409 => "409 Conflict",
        _ => "500 Internal Server Error",
    };
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    Ok(())
}
