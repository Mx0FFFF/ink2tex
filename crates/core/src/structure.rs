//! Symbol Layout Tree (SLT) construction — turning a bag of positioned, classified
//! symbols into a tree (DESIGN §4.4, "the heart of it"). v1 is pure geometry +
//! class-aware rules; the learned relation MLP is later work.
//!
//! Coordinates are normalized with **y increasing downward** (screen convention):
//! "higher on the page" is a *smaller* y. The parse has two phases:
//!   1. `form_units` — collapse composite constructs (fractions, radicals) into
//!      single units, recursively parsing their sub-regions.
//!   2. `baseline` — walk the units left-to-right, attaching super/subscripts (and
//!      big-operator limits) to each base, recursing into every region.
//!
//! Size-ambiguous symbols are resolved here, not in the classifier (DESIGN §4.3): a
//! horizontal bar with content *above and below* is a fraction; the same bar with
//! symbols left/right on one baseline is a minus sign.

/// Axis-aligned bounding box in normalized coords (y down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl BBox {
    pub fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }
    pub fn cx(&self) -> f32 {
        (self.min_x + self.max_x) * 0.5
    }
    pub fn cy(&self) -> f32 {
        (self.min_y + self.max_y) * 0.5
    }
    pub fn w(&self) -> f32 {
        self.max_x - self.min_x
    }
    pub fn h(&self) -> f32 {
        self.max_y - self.min_y
    }
    fn area(&self) -> f32 {
        self.w() * self.h()
    }
    fn union(&self, o: &BBox) -> BBox {
        BBox::new(
            self.min_x.min(o.min_x),
            self.min_y.min(o.min_y),
            self.max_x.max(o.max_x),
            self.max_y.max(o.max_y),
        )
    }
    /// Is `o`'s center inside this box (both axes)? Used to find a radical's content.
    fn contains_center(&self, o: &BBox) -> bool {
        o.cx() >= self.min_x && o.cx() <= self.max_x && o.cy() >= self.min_y && o.cy() <= self.max_y
    }
    /// Does this box fully contain `o`? (An enclosing structure, e.g. a `√`.)
    fn contains(&self, o: &BBox) -> bool {
        self.min_x <= o.min_x
            && self.max_x >= o.max_x
            && self.min_y <= o.min_y
            && self.max_y >= o.max_y
    }
    /// Is this box's x-center within `o`'s x-range? Used to find a fraction's num/den.
    fn center_in_x(&self, o: &BBox) -> bool {
        self.cx() >= o.min_x && self.cx() <= o.max_x
    }
}

/// A classified symbol with its position.
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    pub label: String,
    pub bbox: BBox,
}

impl Symbol {
    pub fn new(label: impl Into<String>, bbox: BBox) -> Self {
        Self {
            label: label.into(),
            bbox,
        }
    }
}

/// A Symbol Layout Tree: a horizontal run of terms, left-to-right.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Slt {
    pub terms: Vec<Term>,
}

/// One term on a baseline: a base with optional super/subscript sub-trees.
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    pub base: Base,
    pub sup: Option<Slt>,
    pub sub: Option<Slt>,
}

/// What sits on the baseline: a plain symbol, a fraction, or a radical.
#[derive(Clone, Debug, PartialEq)]
pub enum Base {
    Symbol(String),
    Frac { num: Slt, den: Slt },
    Sqrt(Slt),
}

/// A resolved layout unit: a bbox plus the base it represents (a plain symbol or an
/// already-built composite). Baseline parsing works over these.
#[derive(Clone, Debug)]
struct Unit {
    bbox: BBox,
    base: Base,
}

/// Parse positioned symbols into an SLT.
pub fn parse(symbols: &[Symbol]) -> Slt {
    baseline(form_units(symbols))
}

/// Threshold factor (fraction of a symbol's height) for region classification.
const SCRIPT_DY: f32 = 0.25; // vertical offset that makes a neighbour a super/subscript

/// Is this symbol a plausible fraction *bar*? A wide, short stroke, or a labelled
/// rule/minus. (Whether it's actually a fraction is decided by content above/below.)
fn is_bar_candidate(s: &Symbol) -> bool {
    matches!(s.label.as_str(), "-" | "\\frac" | "\\hline" | "_")
        || (s.bbox.h() > 1e-6 && s.bbox.w() / s.bbox.h() > 2.5)
}

/// Is this symbol the radical *glyph*?
///
/// It must accept **every label the classifier can emit for a `√`**, and there are three of
/// them, because Detexify keeps the same glyph under three classes:
///
/// | class | `symbol_command` |
/// |---|---|
/// | `latex:latex2e:sqrt-lbrace-rbrace` | `\sqrt{}` |
/// | `latex:textcomp:textsurd` | `\textsurd` |
/// | `latex:latex2e:surd` | `\surd` |
///
/// They are indistinguishable ink, so the model splits its probability across them — on the
/// corpus fixture it says `\sqrt{}` 65.8%, `\textsurd` 28.0%, `\surd` 3.7%. Matching only
/// one would leave the radical unrecognized roughly a third of the time, *silently*: it
/// would come out as a plain symbol with its contents dangling beside it rather than inside.
///
/// This is exactly how `\sqrt` came to be broken end-to-end while 17 structure tests passed
/// — the tests said `"\\sqrt"`, and the classifier never says that. **A label gate here has
/// to speak the classifier's vocabulary, not LaTeX's.**
fn is_sqrt(label: &str) -> bool {
    matches!(
        label,
        "\\sqrt" | "\\sqrt{}" | "\\surd" | "\\textsurd" | "√" | "sqrt"
    )
}

/// A composite construct found in the pool, with the indices it consumes.
enum Composite {
    Frac {
        above: Vec<usize>,
        below: Vec<usize>,
        members: Vec<usize>,
        region: BBox,
    },
    Sqrt {
        inside: Vec<usize>,
        members: Vec<usize>,
        region: BBox,
    },
}

impl Composite {
    fn region(&self) -> BBox {
        match self {
            Composite::Frac { region, .. } | Composite::Sqrt { region, .. } => *region,
        }
    }
}

/// The outermost composite in the pool, if any — the one whose enclosing region is
/// largest. Picking the biggest region first makes nesting resolve correctly whether
/// a `√` wraps a fraction or a fraction's numerator holds a `√`.
fn best_composite(pool: &[Symbol]) -> Option<Composite> {
    let mut best: Option<Composite> = None;
    let mut consider = |c: Composite| {
        // (MSRV 1.80: `map_or`, not the 1.82 `is_none_or`.)
        if best
            .as_ref()
            .map_or(true, |b| c.region().area() > b.region().area())
        {
            best = Some(c);
        }
    };

    // Fractions: a bar candidate with content both above and below (in its x-span).
    for (i, s) in pool.iter().enumerate() {
        if !is_bar_candidate(s) {
            continue;
        }
        let bar = s.bbox;
        let (mut above, mut below) = (Vec::new(), Vec::new());
        let mut region = bar;
        for (k, t) in pool.iter().enumerate() {
            // Skip a structure that *encloses* the bar (e.g. a √ around the whole
            // fraction) — it's an outer composite, not this fraction's num/den.
            if k != i && !t.bbox.contains(&bar) && t.bbox.center_in_x(&bar) {
                region = region.union(&t.bbox);
                if t.bbox.cy() < bar.cy() {
                    above.push(k);
                } else {
                    below.push(k);
                }
            }
        }
        if !above.is_empty() && !below.is_empty() {
            let mut members = vec![i];
            members.extend(above.iter().chain(below.iter()));
            consider(Composite::Frac {
                above,
                below,
                members,
                region,
            });
        }
    }

    // Radicals: a `√` whose bbox encloses the centers of some following symbols.
    for (i, s) in pool.iter().enumerate() {
        if !is_sqrt(&s.label) {
            continue;
        }
        let inside: Vec<usize> = pool
            .iter()
            .enumerate()
            .filter(|&(k, t)| k != i && s.bbox.contains_center(&t.bbox))
            .map(|(k, _)| k)
            .collect();
        if !inside.is_empty() {
            let mut members = vec![i];
            members.extend(&inside);
            consider(Composite::Sqrt {
                inside,
                members,
                region: s.bbox,
            });
        }
    }
    best
}

/// Collapse composites (fractions, radicals) into single units, outermost first;
/// everything left becomes a plain symbol unit.
fn form_units(symbols: &[Symbol]) -> Vec<Unit> {
    let mut pool: Vec<Symbol> = symbols.to_vec();
    let mut units: Vec<Unit> = Vec::new();

    while let Some(comp) = best_composite(&pool) {
        let pick =
            |idx: &[usize]| -> Vec<Symbol> { idx.iter().map(|&k| pool[k].clone()).collect() };
        let (unit, members) = match comp {
            Composite::Frac {
                above,
                below,
                members,
                region,
            } => (
                Unit {
                    bbox: region,
                    base: Base::Frac {
                        num: parse(&pick(&above)),
                        den: parse(&pick(&below)),
                    },
                },
                members,
            ),
            Composite::Sqrt {
                inside,
                members,
                region,
            } => (
                Unit {
                    bbox: region,
                    base: Base::Sqrt(parse(&pick(&inside))),
                },
                members,
            ),
        };
        units.push(unit);
        pool = pool
            .iter()
            .enumerate()
            .filter(|(k, _)| !members.contains(k))
            .map(|(_, s)| s.clone())
            .collect();
    }

    for s in &pool {
        units.push(Unit {
            bbox: s.bbox,
            base: Base::Symbol(s.label.clone()),
        });
    }
    units
}

fn opt(slt: Slt) -> Option<Slt> {
    if slt.terms.is_empty() {
        None
    } else {
        Some(slt)
    }
}

/// Walk units left-to-right, attaching super/subscripts (and big-op limits) to each
/// base. Recurses into every region.
fn baseline(mut units: Vec<Unit>) -> Slt {
    units.sort_by(|a, b| {
        a.bbox
            .min_x
            .partial_cmp(&b.bbox.min_x)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut terms = Vec::new();
    let mut i = 0;
    while i < units.len() {
        let base_box = units[i].bbox;

        // Collect the run of following units that are super/subscripts of this base
        // (for a big operator like `\sum` these are its over/under limits, which sit
        // in the same upper/lower regions), until one returns to the baseline.
        let mut sup = Vec::new();
        let mut sub = Vec::new();
        let mut j = i + 1;
        while j < units.len() {
            let u = &units[j];
            let dy = u.bbox.cy() - base_box.cy();
            let thresh = SCRIPT_DY * base_box.h().max(1e-6);
            let above = dy < -thresh;
            let below = dy > thresh;
            if above {
                sup.push(u.clone());
                j += 1;
            } else if below {
                sub.push(u.clone());
                j += 1;
            } else {
                break;
            }
        }

        terms.push(Term {
            base: units[i].base.clone(),
            sup: opt(baseline(sup)),
            sub: opt(baseline(sub)),
        });
        i = j;
    }
    Slt { terms }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(label: &str, min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Symbol {
        Symbol::new(label, BBox::new(min_x, min_y, max_x, max_y))
    }

    // Build the SLT and render it, so tests read as expected LaTeX.
    fn latex(symbols: &[Symbol]) -> String {
        crate::latex::to_latex(&parse(symbols))
    }

    #[test]
    fn horizontal_sequence() {
        // a + b  (all on one baseline, y in [0.4, 0.6])
        let s = [
            sym("a", 0.0, 0.4, 0.1, 0.6),
            sym("+", 0.15, 0.4, 0.25, 0.6),
            sym("b", 0.3, 0.4, 0.4, 0.6),
        ];
        assert_eq!(latex(&s), "a+b");
    }

    #[test]
    fn superscript() {
        // x^2 : x on baseline, 2 up-and-right (smaller y)
        let s = [
            sym("x", 0.0, 0.4, 0.15, 0.7),
            sym("2", 0.16, 0.20, 0.26, 0.40),
        ];
        assert_eq!(latex(&s), "x^{2}");
    }

    #[test]
    fn subscript() {
        // x_i : i down-and-right (larger y)
        let s = [
            sym("x", 0.0, 0.3, 0.15, 0.6),
            sym("i", 0.16, 0.55, 0.24, 0.80),
        ];
        assert_eq!(latex(&s), "x_{i}");
    }

    #[test]
    fn superscript_then_baseline() {
        // x^2 + 1
        let s = [
            sym("x", 0.0, 0.4, 0.12, 0.7),
            sym("2", 0.13, 0.20, 0.22, 0.40),
            sym("+", 0.30, 0.45, 0.40, 0.60),
            sym("1", 0.45, 0.42, 0.52, 0.62),
        ];
        assert_eq!(latex(&s), "x^{2}+1");
    }

    #[test]
    fn both_scripts() {
        // x_i^2  (i lower-right, 2 upper-right)
        let s = [
            sym("x", 0.0, 0.35, 0.15, 0.65),
            sym("2", 0.16, 0.12, 0.25, 0.32),
            sym("i", 0.16, 0.62, 0.24, 0.85),
        ];
        assert_eq!(latex(&s), "x_{i}^{2}");
    }

    #[test]
    fn nested_superscript() {
        // x^{2y} : 2 and y both in the superscript region
        let s = [
            sym("x", 0.0, 0.4, 0.12, 0.7),
            sym("2", 0.13, 0.15, 0.20, 0.35),
            sym("y", 0.21, 0.15, 0.28, 0.38),
        ];
        assert_eq!(latex(&s), "x^{2y}");
    }

    #[test]
    fn simple_fraction() {
        // \frac{a}{b}: wide bar, a above, b below (x-centers within the bar).
        let s = [
            sym("-", 0.10, 0.49, 0.50, 0.51), // bar (wide, short)
            sym("a", 0.25, 0.20, 0.35, 0.40),
            sym("b", 0.25, 0.60, 0.35, 0.80),
        ];
        assert_eq!(latex(&s), "\\frac{a}{b}");
    }

    #[test]
    fn minus_is_not_a_fraction() {
        // a - b: the bar has content to the sides, not above/below → minus.
        let s = [
            sym("a", 0.00, 0.45, 0.10, 0.55),
            sym("-", 0.15, 0.49, 0.25, 0.51),
            sym("b", 0.30, 0.45, 0.40, 0.55),
        ];
        assert_eq!(latex(&s), "a-b");
    }

    #[test]
    fn fraction_in_a_sequence() {
        // 1 + \frac{a}{b}
        let s = [
            sym("1", 0.00, 0.45, 0.08, 0.60),
            sym("+", 0.12, 0.47, 0.22, 0.57),
            sym("-", 0.30, 0.50, 0.70, 0.52), // bar
            sym("a", 0.45, 0.30, 0.55, 0.45),
            sym("b", 0.45, 0.57, 0.55, 0.72),
        ];
        assert_eq!(latex(&s), "1+\\frac{a}{b}");
    }

    #[test]
    fn nested_fraction() {
        // \frac{\frac{a}{b}}{c}: outer bar widest; its numerator is another fraction.
        let s = [
            sym("=", 0.10, 0.50, 0.60, 0.52), // outer bar (widest)
            sym("-", 0.20, 0.30, 0.50, 0.31), // inner bar
            sym("a", 0.30, 0.18, 0.40, 0.28),
            sym("b", 0.30, 0.33, 0.40, 0.43),
            sym("c", 0.30, 0.60, 0.40, 0.75),
        ];
        assert_eq!(latex(&s), "\\frac{\\frac{a}{b}}{c}");
    }

    /// The other radical tests here all say `"\\sqrt"` — a label **the classifier never
    /// emits**. That is how `\sqrt` stayed broken end-to-end while every one of them passed.
    /// This pins the vocabulary the classifier actually speaks.
    #[test]
    fn the_radical_labels_the_classifier_actually_emits_are_recognized() {
        // On a real capture the model said: \sqrt{} 67.2%, \textsurd 30.3%, \surd 1.7%.
        // All three are the same ink. Miss any one and the radical silently degrades into a
        // plain symbol with its contents dangling beside it instead of inside it.
        for label in ["\\sqrt{}", "\\textsurd", "\\surd", "\\sqrt"] {
            let out = latex(&[
                sym(label, 0.00, 0.20, 0.75, 0.80),
                sym("a", 0.25, 0.40, 0.45, 0.60),
                sym("b", 0.45, 0.40, 0.65, 0.60),
            ]);
            assert_eq!(
                out, "\\sqrt{ab}",
                "{label} was not treated as a radical — its contents fell outside it"
            );
        }
    }

    #[test]
    fn simple_radical() {
        // \sqrt{x}: a big √ enclosing x.
        let s = [
            sym("\\sqrt", 0.00, 0.20, 0.50, 0.80), // radical encloses the radicand
            sym("x", 0.20, 0.40, 0.35, 0.60),
        ];
        assert_eq!(latex(&s), "\\sqrt{x}");
    }

    #[test]
    fn radical_over_expression() {
        // \sqrt{a+b}
        let s = [
            sym("\\sqrt", 0.00, 0.20, 0.75, 0.80),
            sym("a", 0.15, 0.40, 0.25, 0.60),
            sym("+", 0.30, 0.42, 0.40, 0.58),
            sym("b", 0.45, 0.40, 0.55, 0.60),
        ];
        assert_eq!(latex(&s), "\\sqrt{a+b}");
    }

    #[test]
    fn radical_in_sequence() {
        // 1 + \sqrt{x}
        let s = [
            sym("1", 0.00, 0.40, 0.08, 0.60),
            sym("+", 0.12, 0.42, 0.22, 0.58),
            sym("\\sqrt", 0.30, 0.25, 0.70, 0.75),
            sym("x", 0.45, 0.40, 0.58, 0.60),
        ];
        assert_eq!(latex(&s), "1+\\sqrt{x}");
    }

    #[test]
    fn radical_over_fraction() {
        // \sqrt{\frac{a}{b}} — the √ region is larger than the fraction, so it is the
        // outer composite; its radicand re-parses to the fraction.
        let s = [
            sym("\\sqrt", 0.05, 0.15, 0.65, 0.85), // biggest region → outermost
            sym("-", 0.20, 0.48, 0.50, 0.52),      // fraction bar, inside the √
            sym("a", 0.30, 0.25, 0.40, 0.40),
            sym("b", 0.30, 0.60, 0.40, 0.75),
        ];
        assert_eq!(latex(&s), "\\sqrt{\\frac{a}{b}}");
    }

    #[test]
    fn sum_with_limits() {
        // \sum_{i}^{n} : n over the operator, i under it.
        let s = [
            sym("\\sum", 0.10, 0.35, 0.40, 0.65),
            sym("n", 0.18, 0.12, 0.32, 0.28), // above
            sym("i", 0.18, 0.72, 0.32, 0.88), // below
        ];
        assert_eq!(latex(&s), "\\sum_{i}^{n}");
    }

    #[test]
    fn integral_with_limits() {
        // \int_{a}^{b} : b upper-right, a lower-right (script-style limits).
        let s = [
            sym("\\int", 0.10, 0.20, 0.22, 0.80),
            sym("b", 0.24, 0.18, 0.34, 0.36), // upper
            sym("a", 0.24, 0.66, 0.34, 0.84), // lower
        ];
        assert_eq!(latex(&s), "\\int_{a}^{b}");
    }

    #[test]
    fn sum_over_fraction_expression() {
        // \sum_{i}^{n}\frac{1}{i} — operator with limits, then a fraction on the line.
        let s = [
            sym("\\sum", 0.05, 0.35, 0.30, 0.65),
            sym("n", 0.12, 0.12, 0.24, 0.28),
            sym("i", 0.12, 0.72, 0.24, 0.88),
            sym("-", 0.40, 0.49, 0.70, 0.51), // fraction bar
            sym("1", 0.52, 0.28, 0.60, 0.45),
            sym("i", 0.52, 0.56, 0.60, 0.72),
        ];
        assert_eq!(latex(&s), "\\sum_{i}^{n}\\frac{1}{i}");
    }
}
