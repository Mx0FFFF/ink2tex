//! Single-stroke vector glyphs — the "pretty handwriting" the beautifier writes
//! back onto the page through xochitl.
//!
//! The data is Dr. A. V. Hershey's 1967 plotter font set (the `futural` /
//! Simplex face), which is close to perfect for this job for a reason that is
//! not a coincidence: Hershey designed for pen plotters, and injecting strokes
//! into a tablet's digitizer IS a pen plotter — a polyline is drawn by a moving
//! pen tip, there is no fill, no hinting, no rasterizer. The glyph data is
//! permissively redistributable with attribution (see ATTRIBUTION.md); the JHF
//! parser below is ours.
//!
//! ## The JHF format, briefly
//! One glyph per record: 5 bytes of id (ignored), 3 bytes of vertex count, then
//! vertex pairs as ASCII bytes offset from `'R'` (`ord(c) - ord('R')`, so `R` is
//! zero). The FIRST pair is the left/right advance bounds, `" R"` is pen-up, and
//! records longer than a line simply wrap — the vertex count is the only truth.
//! Y grows downward; the baseline sits at +9, capital tops at −12 (21 units of
//! cap height), which we map into `typeset`'s em space (baseline 800/1000).

/// A glyph ready to trace: polylines in em×1000 space (y down, baseline at 800,
/// left edge at x=0) plus the horizontal advance.
#[derive(Clone, Debug)]
pub struct Glyph {
    pub polylines: Vec<Vec<(f32, f32)>>,
    pub advance: f32,
}

const FUTURAL: &str = include_str!("glyphs/futural.jhf");

/// (left bound, right bound, polylines) in raw Hershey units.
type RawGlyph = (f32, f32, Vec<Vec<(f32, f32)>>);
/// Hershey units → em/1000: 21 units of cap height become 600 (typeset's glyphs
/// are ~0.6 em tall over a 0.8-em baseline, same proportions as its text boxes).
const UNIT: f32 = 600.0 / 21.0;
const BASELINE_EM: f32 = 800.0;
const BASELINE_HERSHEY: f32 = 9.0;

/// Parse every glyph record in a JHF blob. Wrapped lines are joined by trusting
/// the vertex count, not the line structure.
fn parse_jhf(data: &str) -> Vec<RawGlyph> {
    let mut glyphs = Vec::new();
    let mut buf = String::new();
    for line in data.lines() {
        if line.is_empty() {
            continue;
        }
        buf.push_str(line);
        let Some(n) = buf.get(5..8).and_then(|s| s.trim().parse::<usize>().ok()) else {
            buf.clear();
            continue;
        };
        if buf.len() < 8 + 2 * n {
            continue; // record wraps onto the next line
        }
        let body = &buf.as_bytes()[8..8 + 2 * n];
        let coord = |b: u8| (b as i32 - 'R' as i32) as f32;
        let (l, r) = (coord(body[0]), coord(body[1]));
        let mut polylines = Vec::new();
        let mut cur: Vec<(f32, f32)> = Vec::new();
        for pair in body[2..].chunks_exact(2) {
            if pair == b" R" {
                if cur.len() > 1 {
                    polylines.push(std::mem::take(&mut cur));
                } else {
                    cur.clear();
                }
            } else {
                cur.push((coord(pair[0]), coord(pair[1])));
            }
        }
        if cur.len() > 1 {
            polylines.push(cur);
        }
        glyphs.push((l, r, polylines));
        buf.clear();
    }
    glyphs
}

/// Hershey-space glyph → em-space glyph (baseline 800, left edge 0).
fn to_em(l: f32, r: f32, polylines: &[Vec<(f32, f32)>]) -> Glyph {
    Glyph {
        polylines: polylines
            .iter()
            .map(|pl| {
                pl.iter()
                    .map(|&(x, y)| {
                        (
                            (x - l) * UNIT,
                            BASELINE_EM + (y - BASELINE_HERSHEY) * UNIT,
                        )
                    })
                    .collect()
            })
            .collect(),
        advance: (r - l) * UNIT,
    }
}

/// Hand-composed glyphs for tokens Hershey's ASCII face lacks. Same Hershey
/// coordinate conventions (y down, baseline +9), so they inherit the em mapping.
fn composed(cmd: &str) -> Option<RawGlyph> {
    let g = |l: f32, r: f32, pls: &[&[(f32, f32)]]| {
        Some((l, r, pls.iter().map(|p| p.to_vec()).collect()))
    };
    match cmd {
        "\\pm" => g(
            -11.0,
            11.0,
            &[
                &[(0.0, -10.0), (0.0, 4.0)],  // vertical of the +
                &[(-8.0, -3.0), (8.0, -3.0)], // horizontal of the +
                &[(-8.0, 9.0), (8.0, 9.0)],   // the bar underneath
            ],
        ),
        "\\times" => g(
            -9.0,
            9.0,
            &[&[(-6.0, -7.0), (6.0, 5.0)], &[(6.0, -7.0), (-6.0, 5.0)]],
        ),
        "\\cdot" => g(
            -4.0,
            4.0,
            &[&[(-1.0, 0.0), (1.0, 0.0), (1.0, 2.0), (-1.0, 2.0), (-1.0, 0.0)]],
        ),
        "\\leq" => g(
            -10.0,
            10.0,
            &[&[(7.0, -9.0), (-7.0, -1.0), (7.0, 7.0)], &[(-7.0, 11.0), (7.0, 11.0)]],
        ),
        "\\geq" => g(
            -10.0,
            10.0,
            &[&[(-7.0, -9.0), (7.0, -1.0), (-7.0, 7.0)], &[(-7.0, 11.0), (7.0, 11.0)]],
        ),
        "\\neq" => g(
            -10.0,
            10.0,
            &[
                &[(-9.0, -3.0), (9.0, -3.0)],
                &[(-9.0, 3.0), (9.0, 3.0)],
                &[(4.0, -10.0), (-4.0, 10.0)],
            ],
        ),
        _ => None,
    }
}

/// Strokes for one display token, or None if we cannot draw it yet. `cmd` is
/// what `latex::symbol_command` produces: a literal ASCII char ("x", "2", "+"),
/// or a LaTeX command ("\\pm"). Multi-char ASCII runs (function names like
/// "sin") are the caller's job to split.
///
/// Two finishing passes make the output read as TYPESET math instead of plotter
/// output:
///  - **Chaikin smoothing** (two rounds of corner cutting): Hershey glyphs are
///    sparse polylines, and at pen scale their curves render as visible facets —
///    the `2`'s bowl was an obvious hexagon on the panel;
///  - **italic slant for letters**: mathematics sets variables in italic; a
///    12° shear about the baseline is most of that look for a stroke font.
pub fn strokes(cmd: &str) -> Option<Glyph> {
    let mut g = raw_strokes(cmd)?;
    for pl in &mut g.polylines {
        *pl = chaikin(pl, 2);
    }
    if cmd.len() == 1 && cmd.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
        const SLANT: f32 = 0.21; // tan(12°)
        for pl in &mut g.polylines {
            for p in pl.iter_mut() {
                p.0 += (800.0 - p.1) * SLANT;
            }
        }
        // The shear widens the occupied box a little; advance follows suit so
        // a slanted letter does not lean into its neighbour.
        g.advance += 800.0 * SLANT * 0.35;
    }
    Some(g)
}

fn raw_strokes(cmd: &str) -> Option<Glyph> {
    if let Some((l, r, pls)) = composed(cmd) {
        return Some(to_em(l, r, &pls));
    }
    let mut chars = cmd.chars();
    let (c, rest) = (chars.next()?, chars.next());
    if rest.is_some() || !(' '..='~').contains(&c) {
        return None; // multi-char or non-ASCII: not this table's problem
    }
    let table = parsed_futural();
    let (l, r, pls) = table.get(c as usize - ' ' as usize)?;
    Some(to_em(*l, *r, pls))
}

/// Chaikin's corner-cutting: each round replaces every interior corner with two
/// points at 1/4 and 3/4 of its adjoining segments. Endpoints are preserved, so
/// glyphs keep their exact extents; straight two-point lines pass through
/// untouched (nothing to cut).
fn chaikin(pl: &[(f32, f32)], rounds: usize) -> Vec<(f32, f32)> {
    let mut cur: Vec<(f32, f32)> = pl.to_vec();
    for _ in 0..rounds {
        if cur.len() < 3 {
            return cur;
        }
        let mut next = Vec::with_capacity(cur.len() * 2);
        next.push(cur[0]);
        for w in cur.windows(2) {
            let (a, b) = (w[0], w[1]);
            next.push((a.0 * 0.75 + b.0 * 0.25, a.1 * 0.75 + b.1 * 0.25));
            next.push((a.0 * 0.25 + b.0 * 0.75, a.1 * 0.25 + b.1 * 0.75));
        }
        next.push(*cur.last().expect("nonempty"));
        cur = next;
    }
    cur
}

/// The parse is cheap (~3.5 KB of text) but not free; do it once.
fn parsed_futural() -> &'static [RawGlyph] {
    use std::sync::OnceLock;
    static PARSED: OnceLock<Vec<RawGlyph>> = OnceLock::new();
    PARSED.get_or_init(|| parse_jhf(FUTURAL))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn futural_parses_all_96_ascii_glyphs() {
        assert_eq!(parsed_futural().len(), 96);
    }

    #[test]
    fn an_a_is_three_polylines_on_the_baseline() {
        let g = strokes("A").expect("A");
        assert_eq!(g.polylines.len(), 3, "two legs + crossbar");
        // Legs end ON the baseline (em y=800), cap top at 200.
        let ys: Vec<f32> = g.polylines.iter().flatten().map(|p| p.1).collect();
        assert!((ys.iter().cloned().fold(f32::MIN, f32::max) - 800.0).abs() < 1.0);
        assert!((ys.iter().cloned().fold(f32::MAX, f32::min) - 200.0).abs() < 1.0);
    }

    #[test]
    fn wrapped_records_parse_by_vertex_count() {
        // '@' (index 32) is the most complex futural glyph and wraps in the file;
        // if the line-joining logic were wrong it would vanish or shatter.
        let g = strokes("@").expect("@ must parse despite wrapping");
        assert!(!g.polylines.is_empty());
    }

    #[test]
    fn the_m2_target_alphabet_is_fully_drawable() {
        // Every character the guided-session targets can produce must have a glyph:
        // this is the beautifier's write-back vocabulary floor.
        for c in "0123456789abcdefghijklmnopqrstuvwxyz+-=<>()".chars() {
            assert!(strokes(&c.to_string()).is_some(), "no glyph for {c:?}");
        }
        assert!(strokes("\\pm").is_some(), "quadratic formula needs ±");
    }

    #[test]
    fn advances_are_positive_and_sane() {
        for c in ["i", "m", "1", "(", "x"] {
            let g = strokes(c).expect(c);
            assert!(
                g.advance > 100.0 && g.advance < 1400.0,
                "{c}: advance {}",
                g.advance
            );
        }
    }
}
