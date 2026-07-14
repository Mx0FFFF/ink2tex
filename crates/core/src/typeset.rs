//! The small math typesetter (M4): Symbol Layout Tree → SVG.
//!
//! This is what lets a person *see* what the recognizer thinks before they correct it —
//! DESIGN §7's whole argument. It is deliberately a typesetter for OUR SLT, not a LaTeX
//! engine: it lays out exactly the constructs `structure` can produce (baseline runs,
//! super/subscripts, fractions, radicals), which means it can never disagree with the
//! parse it renders. Everything is computed in a private em-based coordinate system and
//! emitted as a single self-contained `<svg>` — no fonts fetched, no CSS, no scripts, so
//! it can be inlined into the usb0 web UI (which must work with zero network) or, later,
//! rasterized onto the E-Ink panel itself.
//!
//! Metrics are honest approximations (a glyph is ~0.6 em wide, scripts shrink to 0.7×,
//! a fraction bar adds 0.06 em of rule): good enough to *read*, nowhere near TeX. The
//! milestone asks for "a small math typesetter", and small is a feature — ~200 lines,
//! testable by asserting geometry, replaceable wholesale if the project ever wants Knuth.

use crate::latex::symbol_command;
use crate::structure::{Base, Slt, Term};

/// A laid-out box: everything is measured in em × 1000 (integer math, no float drift in
/// tests). `baseline` is the distance from the box top to the baseline.
#[derive(Clone, Debug)]
struct LBox {
    w: i32,
    h: i32,
    baseline: i32,
    items: Vec<Item>,
}

/// One drawable, positioned relative to its box's top-left.
#[derive(Clone, Debug)]
enum Item {
    Text {
        x: i32,
        y: i32,
        size: i32,
        s: String,
    },
    Line {
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        stroke: i32,
    },
}

const EM: i32 = 1000;
const CHAR_W: i32 = 600; // average glyph advance, in em/1000
const SCRIPT_SCALE: i32 = 700; // scripts render at 0.7×
const RULE: i32 = 60; // fraction bar / radical overbar thickness

fn scale(v: i32, factor: i32) -> i32 {
    (v as i64 * factor as i64 / 1000) as i32
}

/// Display form of a symbol label: the LaTeX command with the backslash-noise reduced to
/// something a human can read in a box (`\alpha` → `α` is out of scope — we render the
/// command name itself, which is unambiguous and font-independent).
fn display(label: &str) -> String {
    symbol_command(label)
}

/// The advance the tracer will actually use for this token: one composed glyph
/// for a `\command`, per-char Hershey advances for a literal run, CHAR_W for
/// anything we cannot draw. Layout and tracing MUST agree here — the first
/// live beautify boxed `\pm` at 600 em per character of its command NAME while
/// tracing one 630-em glyph, leaving two character-cells of dead page after it.
fn token_advance(s: &str) -> i32 {
    if s.starts_with('\\') {
        return crate::glyphs::strokes(s)
            .map(|g| g.advance as i32)
            .unwrap_or(CHAR_W);
    }
    s.chars()
        .map(|c| {
            crate::glyphs::strokes(&c.to_string())
                .map(|g| g.advance as i32)
                .unwrap_or(CHAR_W)
        })
        .sum::<i32>()
        .max(CHAR_W / 2)
}

fn text_box(s: &str, size: i32) -> LBox {
    let w = scale(token_advance(s), size);
    let h = scale(EM, size);
    LBox {
        w,
        h,
        baseline: scale(800, size), // glyph baseline sits at 0.8 em from the top
        items: vec![Item::Text {
            x: 0,
            y: scale(800, size),
            size,
            s: s.to_string(),
        }],
    }
}

fn shift(items: &[Item], dx: i32, dy: i32) -> Vec<Item> {
    items
        .iter()
        .map(|it| match it {
            Item::Text { x, y, size, s } => Item::Text {
                x: x + dx,
                y: y + dy,
                size: *size,
                s: s.clone(),
            },
            Item::Line {
                x1,
                y1,
                x2,
                y2,
                stroke,
            } => Item::Line {
                x1: x1 + dx,
                y1: y1 + dy,
                x2: x2 + dx,
                y2: y2 + dy,
                stroke: *stroke,
            },
        })
        .collect()
}

/// Lay out a horizontal run: boxes sit on a shared baseline, with a possibly
/// different gap at every boundary (`gaps.len() == boxes.len() - 1`; missing
/// entries fall back to the last, or 0).
fn hbox_gaps(boxes: Vec<LBox>, gaps: &[i32]) -> LBox {
    let baseline = boxes.iter().map(|b| b.baseline).max().unwrap_or(0);
    let depth = boxes.iter().map(|b| b.h - b.baseline).max().unwrap_or(0);
    let mut items = Vec::new();
    let mut x = 0;
    for (i, b) in boxes.iter().enumerate() {
        items.extend(shift(&b.items, x, baseline - b.baseline));
        x += b.w
            + if i + 1 < boxes.len() {
                gaps.get(i).or(gaps.last()).copied().unwrap_or(0)
            } else {
                0
            };
    }
    LBox {
        w: x.max(0),
        h: baseline + depth,
        baseline,
        items,
    }
}

/// TeX's insight, reduced to one function: how much air a token gets depends on
/// its grammatical class. Relations (=, <) breathe widest, binary operators
/// (+, −, ±) get a medium cushion, everything else sits close. Uniform spacing
/// is exactly what makes machine output look non-typeset.
fn spacing_class(t: &Term) -> i32 {
    let Base::Symbol(l) = &t.base else { return 0 };
    match display(l).as_str() {
        "=" | "<" | ">" | "\\leq" | "\\geq" | "\\neq" => 150,
        "+" | "-" | "\\pm" | "\\times" | "\\cdot" | "\\div" => 70,
        _ => 0,
    }
}

fn layout_slt(slt: &Slt, size: i32) -> LBox {
    let boxes: Vec<LBox> = slt.terms.iter().map(|t| layout_term(t, size)).collect();
    if boxes.is_empty() {
        return text_box(" ", size);
    }
    // Per-boundary gaps: base air plus each neighbour's class cushion.
    let gaps: Vec<i32> = slt
        .terms
        .windows(2)
        .map(|w| scale(90 + spacing_class(&w[0]) + spacing_class(&w[1]), size))
        .collect();
    hbox_gaps(boxes, &gaps)
}

fn layout_term(t: &Term, size: i32) -> LBox {
    let base = match &t.base {
        Base::Symbol(l) => text_box(&display(l), size),
        Base::Frac { num, den } => {
            let n = layout_slt(num, scale(size, 850));
            let d = layout_slt(den, scale(size, 850));
            let w = n.w.max(d.w) + scale(200, size);
            let bar = scale(RULE, size);
            // Clearance above and below the bar: parens and descenders in a
            // numerator otherwise bottom out at EXACTLY the bar's top edge and
            // merge with it at pen width.
            let gap = scale(120, size);
            let mut items = Vec::new();
            items.extend(shift(&n.items, (w - n.w) / 2, 0));
            items.push(Item::Line {
                x1: 0,
                y1: n.h + gap + bar / 2,
                x2: w,
                y2: n.h + gap + bar / 2,
                stroke: bar,
            });
            items.extend(shift(&d.items, (w - d.w) / 2, n.h + gap + bar + gap));
            LBox {
                w,
                h: n.h + gap + bar + gap + d.h,
                // the fraction bar sits on the maths axis, ~0.55 em above baseline
                baseline: n.h + gap + bar / 2 + scale(300, size),
                items,
            }
        }
        Base::Sqrt(inner) => {
            let c = layout_slt(inner, size);
            let tick = scale(450, size); // the √ hook's width
            let bar = scale(RULE, size);
            let mut items = vec![
                // the radical: down-stroke, up-stroke, then the overbar
                Item::Line {
                    x1: 0,
                    y1: c.h * 6 / 10,
                    x2: tick / 2,
                    y2: c.h,
                    stroke: bar,
                },
                Item::Line {
                    x1: tick / 2,
                    y1: c.h,
                    x2: tick,
                    y2: bar / 2,
                    stroke: bar,
                },
                Item::Line {
                    x1: tick,
                    y1: bar / 2,
                    x2: tick + c.w + scale(100, size),
                    y2: bar / 2,
                    stroke: bar,
                },
            ];
            items.extend(shift(
                &c.items,
                tick + scale(50, size),
                bar + scale(60, size),
            ));
            LBox {
                w: tick + c.w + scale(150, size),
                h: c.h + bar + scale(60, size),
                baseline: c.baseline + bar + scale(60, size),
                items,
            }
        }
    };

    let ssize = scale(size, SCRIPT_SCALE);
    let (mut w, mut items) = (base.w, base.items.clone());
    let mut top_overshoot = 0;
    let mut bottom_overshoot = 0;
    if let Some(sup) = &t.sup {
        let s = layout_slt(sup, ssize);
        // superscript: its baseline sits ~0.45 em above the base's baseline
        let dy = base.baseline - scale(450, size) - s.baseline;
        items.extend(shift(&s.items, base.w + scale(40, size), dy));
        w = w.max(base.w + scale(40, size) + s.w);
        top_overshoot = top_overshoot.max(-dy);
    }
    if let Some(sub) = &t.sub {
        let s = layout_slt(sub, ssize);
        let dy = base.baseline + scale(250, size) - s.baseline;
        items.extend(shift(&s.items, base.w + scale(40, size), dy));
        w = w.max(base.w + scale(40, size) + s.w);
        bottom_overshoot = bottom_overshoot.max(dy + s.h - base.h);
    }
    if top_overshoot > 0 {
        items = shift(&items, 0, top_overshoot);
    }
    LBox {
        w,
        h: base.h + top_overshoot + bottom_overshoot.max(0),
        baseline: base.baseline + top_overshoot,
        items,
    }
}

/// A typeset expression as pen polylines — what the beautifier hands to the
/// injector to "handwrite" back onto the page.
pub struct StrokePlan {
    /// Polylines in this plan's own coordinate space (em×1000 units, y down).
    pub polylines: Vec<Vec<(f32, f32)>>,
    pub w: f32,
    pub h: f32,
    /// Tokens we had no glyph for (skipped, leaving their advance as a gap).
    pub missing: Vec<String>,
}

/// Lay out the SLT and trace every item as polylines: text through the Hershey
/// single-stroke glyphs, rules (fraction bars, radicals) as line segments. The
/// caller scales/translates the plan into page coordinates.
pub fn to_strokes(slt: &Slt) -> StrokePlan {
    let b = layout_slt(slt, EM);
    let mut plan = StrokePlan {
        polylines: Vec::new(),
        w: b.w as f32,
        h: b.h as f32,
        missing: Vec::new(),
    };
    // (extents corrected against the traced ink below — a wide trailing glyph
    // can overhang the layout box, and the beautifier scales by these numbers)
    for it in &b.items {
        match it {
            Item::Text { x, y, size, s } => {
                // (x, y) is the run's left edge on its baseline. Multi-char runs
                // (function names, spelled-out commands) advance glyph by glyph.
                let k = *size as f32 / 1000.0;
                let mut pen_x = *x as f32;
                let token_is_command = s.starts_with('\\');
                let units: Vec<String> = if token_is_command {
                    vec![s.clone()]
                } else {
                    s.chars().map(|c| c.to_string()).collect()
                };
                for u in units {
                    match crate::glyphs::strokes(&u) {
                        Some(g) => {
                            for pl in &g.polylines {
                                plan.polylines.push(
                                    pl.iter()
                                        .map(|&(gx, gy)| {
                                            (pen_x + gx * k, *y as f32 + (gy - 800.0) * k)
                                        })
                                        .collect(),
                                );
                            }
                            pen_x += g.advance * k;
                        }
                        None => {
                            plan.missing.push(u);
                            pen_x += 600.0 * k; // leave the advance as a gap
                        }
                    }
                }
            }
            Item::Line { x1, y1, x2, y2, .. } => {
                plan.polylines
                    .push(vec![(*x1 as f32, *y1 as f32), (*x2 as f32, *y2 as f32)]);
            }
        }
    }
    // Normalize to the TRACED INK's bounding box. The layout box is a full em
    // tall (ascender + descender headroom) but a typical formula's ink spans
    // ~600/1000 of it — scaling by the box height rendered every rewrite at
    // 40-60% of the handwriting's size. The beautifier matches ink to ink.
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for pl in &plan.polylines {
        for &(x, y) in pl {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x1 > x0 && y1 > y0 {
        for pl in &mut plan.polylines {
            for p in pl {
                p.0 -= x0;
                p.1 -= y0;
            }
        }
        plan.w = x1 - x0;
        plan.h = y1 - y0;
    }
    plan
}

/// Render an SLT as a self-contained SVG document.
pub fn to_svg(slt: &Slt) -> String {
    let b = layout_slt(slt, EM);
    let pad = 150;
    let (w, h) = (b.w + 2 * pad, b.h + 2 * pad);
    let mut out = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="serif">"#
    );
    for it in shift(&b.items, pad, pad) {
        match it {
            Item::Text { x, y, size, s } => {
                let esc = s
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;");
                out.push_str(&format!(
                    r#"<text x="{x}" y="{y}" font-size="{size}">{esc}</text>"#
                ));
            }
            Item::Line {
                x1,
                y1,
                x2,
                y2,
                stroke,
            } => {
                out.push_str(&format!(
                    r#"<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="currentColor" stroke-width="{stroke}"/>"#
                ));
            }
        }
    }
    out.push_str("</svg>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{parse, BBox, Symbol};

    fn sym(label: &str, x0: f32, y0: f32, x1: f32, y1: f32) -> Symbol {
        Symbol::new(label, BBox::new(x0, y0, x1, y1))
    }

    #[test]
    fn a_baseline_run_renders_left_to_right_text() {
        let slt = parse(&[
            sym("2", 0.20, 0.40, 0.26, 0.48),
            sym("x", 0.28, 0.40, 0.34, 0.48),
        ]);
        let svg = to_svg(&slt);
        assert!(svg.starts_with("<svg"));
        let x2 = svg.find(">2<").expect("2 rendered");
        let xx = svg.find(">x<").expect("x rendered");
        assert!(x2 < xx, "2 must be emitted before x");
    }

    #[test]
    fn a_fraction_renders_a_bar_between_num_and_den() {
        let slt = parse(&[
            sym("a", 0.40, 0.30, 0.46, 0.38),
            sym("-", 0.36, 0.42, 0.52, 0.44),
            sym("b", 0.40, 0.47, 0.46, 0.55),
        ]);
        let svg = to_svg(&slt);
        assert!(svg.contains("<line"), "the fraction bar must be drawn");
        assert!(svg.contains(">a<") && svg.contains(">b<"));
    }

    #[test]
    fn a_superscript_is_smaller_than_its_base() {
        let slt = parse(&[
            sym("x", 0.28, 0.42, 0.34, 0.50),
            sym("2", 0.35, 0.34, 0.39, 0.39),
        ]);
        let svg = to_svg(&slt);
        // base at font-size 1000, script at 700
        assert!(svg.contains(r#"font-size="1000">x"#));
        assert!(svg.contains(r#"font-size="700">2"#));
    }

    #[test]
    fn a_radical_draws_its_overbar_and_contents() {
        let slt = parse(&[
            sym("\\sqrt{}", 0.00, 0.20, 0.75, 0.80),
            sym("x", 0.25, 0.40, 0.45, 0.60),
        ]);
        let svg = to_svg(&slt);
        assert!(svg.matches("<line").count() >= 3, "tick + hook + overbar");
        assert!(svg.contains(">x<"));
    }

    #[test]
    fn svg_escapes_angle_brackets_in_labels() {
        let slt = parse(&[sym("<", 0.20, 0.40, 0.26, 0.48)]);
        let svg = to_svg(&slt);
        assert!(svg.contains("&lt;"), "raw < must not leak into markup");
    }

    /// The stroke plan for the quadratic formula's right-hand side — a fraction
    /// whose numerator holds a ± and a radical — must trace every token with a
    /// glyph (nothing missing), keep every polyline inside the plan's own bounds,
    /// and include the fraction bar and radical rules as line segments.
    #[test]
    fn quadratic_formula_traces_completely() {
        let slt = parse(&[
            sym("x", 0.10, 0.44, 0.14, 0.50),
            sym("=", 0.16, 0.45, 0.20, 0.48),
            // numerator: -b ± √(b²-4ac) → simplified layout: -b\pm\sqrt{b}
            sym("-", 0.24, 0.36, 0.27, 0.365),
            sym("b", 0.28, 0.32, 0.31, 0.38),
            sym("\\pm", 0.32, 0.33, 0.35, 0.38),
            sym("\\sqrt{}", 0.36, 0.30, 0.46, 0.40),
            sym("b", 0.39, 0.33, 0.42, 0.385),
            // the fraction bar
            sym("-", 0.22, 0.44, 0.48, 0.45),
            // denominator: 2a
            sym("2", 0.30, 0.50, 0.33, 0.56),
            sym("a", 0.34, 0.51, 0.37, 0.56),
        ]);
        let plan = to_strokes(&slt);
        assert!(plan.missing.is_empty(), "missing glyphs: {:?}", plan.missing);
        assert!(plan.polylines.len() > 10, "too few strokes: {}", plan.polylines.len());
        for pl in &plan.polylines {
            for &(x, y) in pl {
                assert!(
                    x >= -1.0 && x <= plan.w + 1.0 && y >= -1.0 && y <= plan.h + 1.0,
                    "({x},{y}) escapes the plan bounds {}x{}",
                    plan.w,
                    plan.h
                );
            }
        }
        // Rules present: the fraction bar + at least the radical's three lines.
        let two_point: usize = plan.polylines.iter().filter(|p| p.len() == 2).count();
        assert!(two_point >= 4, "expected bar+radical rules, got {two_point}");
    }
}
