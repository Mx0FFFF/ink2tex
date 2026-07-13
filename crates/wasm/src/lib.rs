//! The browser demo (M5), with no `wasm-bindgen` — deliberately.
//!
//! The whole project's ethos is "hand-roll the layer you're supposed to understand": the
//! int8 kernel instead of an ML runtime, `sigaction` instead of a signal crate, uinput by
//! `ioctl`. WASM's FFI boundary is the same kind of layer. These exports speak raw linear
//! memory — the JS side allocates through `alloc`, writes plain little-endian floats, and
//! reads UTF-8 back out of the module's memory. ~40 lines of glue on each side, zero
//! dependencies, and you can see every byte cross the boundary.
//!
//! ## Wire format (JS → wasm), all little-endian f32:
//!   [n_strokes, len_0, x,y,x,y,…, len_1, x,y,…]      (normalized 0–1 coords, y down)
//! Weights/labels/counts arrive as raw bytes of the exact files the device uses.
//! Result (wasm → JS): UTF-8 JSON `{"latex":…, "svg":…}`, returned as (ptr, len).

use ink2tex_core::classify::{Labels, Weights};
use ink2tex_core::{Ink, Point, Stroke};

/// Bump allocator handle: JS asks for a buffer, writes into it, passes the pointer back.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// The one entry point. Returns a pointer to `[len: u32][json bytes…]`, or null on error.
///
/// # Safety
/// All pointers must come from `alloc` with the stated lengths — this is the raw FFI
/// boundary, and the JS glue is the only intended caller.
#[no_mangle]
pub unsafe extern "C" fn recognize_expr(
    strokes_ptr: *const f32,
    strokes_len: usize,
    weights_ptr: *const u8,
    weights_len: usize,
    labels_ptr: *const u8,
    labels_len: usize,
    counts_ptr: *const u8,
    counts_len: usize,
) -> *const u8 {
    let floats = std::slice::from_raw_parts(strokes_ptr, strokes_len);
    let weights_bytes = std::slice::from_raw_parts(weights_ptr, weights_len);
    let labels_bytes = std::slice::from_raw_parts(labels_ptr, labels_len);
    let counts_bytes = std::slice::from_raw_parts(counts_ptr, counts_len);

    let json = run(floats, weights_bytes, labels_bytes, counts_bytes)
        .unwrap_or_else(|e| format!(r#"{{"error":"{e}"}}"#));
    let bytes = json.into_bytes();
    let mut out = Vec::with_capacity(4 + bytes.len());
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&bytes);
    let p = out.as_ptr();
    std::mem::forget(out);
    p
}

fn run(floats: &[f32], weights: &[u8], labels: &[u8], counts: &[u8]) -> Result<String, String> {
    // decode the stroke wire format
    let mut i = 0usize;
    let next = |i: &mut usize| -> Result<f32, String> {
        let v = *floats.get(*i).ok_or("truncated stroke data")?;
        *i += 1;
        Ok(v)
    };
    let n_strokes = next(&mut i)? as usize;
    let mut strokes = Vec::with_capacity(n_strokes);
    for _ in 0..n_strokes {
        let len = next(&mut i)? as usize;
        let mut points = Vec::with_capacity(len);
        for j in 0..len {
            let x = next(&mut i)?;
            let y = next(&mut i)?;
            points.push(Point::new(x, y, 1.0, 0.0, 0.0, (j as u64) * 8_000));
        }
        strokes.push(Stroke { points });
    }
    let ink = Ink {
        source_width: 1.0,
        source_height: 1.0,
        strokes,
    };

    let weights = Weights::parse(weights).map_err(|e| e.to_string())?;
    let labels = Labels::from_lines(std::str::from_utf8(labels).map_err(|e| e.to_string())?);
    let counts: Vec<u32> = std::str::from_utf8(counts)
        .map_err(|e| e.to_string())?
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();

    let (_oriented, symbols) = ink2tex_core::analyze(&ink, &weights, &labels, Some(&counts), 5)
        .map_err(|e| e.to_string())?;
    let choices = vec![0usize; symbols.len()];
    let (latex, svg) = ink2tex_core::compose(&symbols, &choices);
    // Top-k or it didn't happen: the demo page needs every symbol's ranked
    // alternatives to offer corrections, same as the on-device UI.
    let mut syms = String::from("[");
    for (i, s) in symbols.iter().enumerate() {
        if i > 0 {
            syms.push(',');
        }
        syms.push_str("{\"candidates\":[");
        for (j, (label, p)) in s.candidates.iter().enumerate() {
            if j > 0 {
                syms.push(',');
            }
            syms.push_str(&format!(
                r#"{{"cmd":{},"p":{:.3}}}"#,
                json_str(&ink2tex_core::latex::symbol_command(label)),
                p
            ));
        }
        syms.push_str("]}");
    }
    syms.push(']');
    Ok(format!(
        r#"{{"latex":{},"svg":{},"symbols":{}}}"#,
        json_str(&latex),
        json_str(&svg),
        syms
    ))
}

fn json_str(s: &str) -> String {
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
