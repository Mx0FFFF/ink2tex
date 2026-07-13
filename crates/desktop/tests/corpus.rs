//! The corpus regression suite — the project's immune system (docs/core-invariants.md,
//! DESIGN.md §3). Every `tests/corpus/<name>.ink` must classify with its true label
//! (`tests/corpus/<name>.expected.tex`) somewhere in the top-5. Drop a captured
//! symbol + its expected LaTeX and it's covered forever; a code change that breaks
//! recognition of a known symbol fails CI.
//!
//! Runs against the committed reference model (`train/model.iwt`). If that's absent
//! (e.g. a fresh checkout before training), it skips rather than failing.

use std::path::{Path, PathBuf};

use ink2tex_core::classify::{
    global_features, online_features, rasterize, recognize, Labels, Weights, ONLINE_POINTS,
};
use ink2tex_core::latex::symbol_command;
use ink2tex_core::Ink;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn corpus_symbols_classify_in_top5() {
    let root = workspace_root();
    let model_path = root.join("train/model.iwt");
    if !model_path.exists() {
        eprintln!(
            "skipping corpus test: {} missing (run train/train.py)",
            model_path.display()
        );
        return;
    }

    let blob = std::fs::read(&model_path).expect("read model");
    let weights = Weights::parse(&blob).expect("parse model");
    let labels = Labels::from_lines(
        &std::fs::read_to_string(root.join("train/model.labels.txt")).expect("read labels"),
    );

    let corpus = root.join("tests/corpus");
    let mut checked = 0;
    for entry in std::fs::read_dir(&corpus).expect("read tests/corpus") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("ink") {
            continue;
        }
        let expected = std::fs::read_to_string(path.with_extension("expected.tex"))
            .unwrap_or_else(|_| panic!("missing .expected.tex for {}", path.display()));
        let expected = expected.trim();

        let ink = Ink::decode(&std::fs::read(&path).expect("read .ink")).expect("decode .ink");
        let preds = recognize(
            &weights,
            &rasterize(&ink.strokes, 32),
            &global_features(&ink.strokes),
            &online_features(&ink.strokes, ONLINE_POINTS),
            32,
            5,
        )
        .expect("classify");

        // Compare LaTeX, not raw symbolIds — so this suite also covers the
        // symbolId→command mapper, where `\sqrt{}` once emitted `\sqrt-lbrace-rbrace`.
        let top5: Vec<String> = preds
            .iter()
            .map(|p| symbol_command(labels.get(p.class).unwrap_or("?")))
            .collect();
        assert!(
            top5.iter().any(|t| t == expected),
            "{}: expected {expected:?} not in top-5 {top5:?}",
            file_name(&path),
        );
        checked += 1;
    }

    assert!(checked > 0, "no corpus cases in {}", corpus.display());
    eprintln!("corpus: {checked} case(s) classified correctly (top-5)");
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_string()
}

/// `\sqrt` end-to-end, on real ink — the thing 17 passing structure tests did not check.
///
/// The structure tests hand `structure::parse` pre-positioned symbols labelled `"\sqrt"`,
/// a string **the classifier never emits** (it says `\sqrt{}`, `\textsurd` or `\surd`,
/// splitting its probability across all three). And they never touch segmentation, which
/// used to merge the radical with its own contents. Both bugs were invisible to them.
///
/// This drives the whole pipeline — ink → denoise → segment → classify → structure → LaTeX —
/// over a real capture of a hand-drawn `√x+1`, and only asks the one thing that matters:
/// the contents ended up *inside* the radical.
#[test]
fn a_real_hand_drawn_radical_nests_its_contents() {
    let root = workspace_root();
    let model_path = root.join("train/model.iwt");
    if !model_path.exists() {
        eprintln!("skipping: {} missing", model_path.display());
        return;
    }
    let blob = std::fs::read(&model_path).expect("read model");
    let weights = Weights::parse(&blob).expect("parse");
    let labels = Labels::from_lines(
        &std::fs::read_to_string(root.join("train/model.labels.txt")).expect("labels"),
    );
    let ink = Ink::decode(
        &std::fs::read(root.join("crates/core/tests/data/radical_over_expression.ink"))
            .expect("read fixture"),
    )
    .expect("decode");

    let out =
        ink2tex_core::recognize_expression(&ink, &weights, &labels, None, 3).expect("recognize");
    assert!(
        out.starts_with("\\sqrt{") && out.len() > "\\sqrt{}".len(),
        "the radical did not take its contents as an argument: {out:?}"
    );
}

/// Every expression-vocabulary entry must exist in the shipped label space. A dead entry
/// is not an error anywhere else — it just silently narrows the mask, and the first
/// symptom is some everyday token quietly losing to an exotic one again.
#[test]
fn vocabulary_entries_exist_in_the_label_space() {
    let root = workspace_root();
    let path = root.join("train/expr.labels.txt");
    if !path.exists() {
        eprintln!("skipping: {} missing", path.display());
        return;
    }
    let text = std::fs::read_to_string(&path).expect("labels");
    let have: std::collections::HashSet<&str> = text.lines().map(str::trim).collect();
    let dead: Vec<&&str> = ink2tex_core::vocab::EXPRESSION_TOKENS
        .iter()
        .filter(|t| !have.contains(**t))
        .collect();
    assert!(
        dead.is_empty(),
        "vocab entries the classifier can never emit: {dead:?}"
    );
}

/// The everyday-token guarantee, pinned against real ink and the CURRENT expression
/// model. Asserts what is actually promised: `x` wins outright; `+` and `1` are within
/// correction reach (top-5 of their symbol). Asserting an exact LaTeX string here would
/// pin one model's lucky top-1s and break on every retrain — the correction UI is the
/// product, so correction reach is the contract.
#[test]
fn everyday_tokens_win_on_real_ink_in_expression_mode() {
    let root = workspace_root();
    let model_path = root.join("train/expr.iwt");
    if !model_path.exists() {
        eprintln!("skipping: {} missing", model_path.display());
        return;
    }
    let blob = std::fs::read(&model_path).expect("read model");
    let weights = Weights::parse(&blob).expect("parse");
    let labels = Labels::from_lines(
        &std::fs::read_to_string(root.join("train/expr.labels.txt")).expect("labels"),
    );
    let counts: Vec<u32> = std::fs::read_to_string(root.join("train/expr.counts.txt"))
        .expect("counts")
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let ink = Ink::decode(
        &std::fs::read(root.join("crates/core/tests/data/radical_over_expression.ink"))
            .expect("fixture"),
    )
    .expect("decode");

    let (_ink, line) =
        ink2tex_core::recognize_line(&ink, &weights, &labels, Some(&counts), 5).expect("recognize");
    assert_eq!(line.len(), 4, "√ x + 1 should segment to 4 symbols");
    let names: Vec<Vec<&str>> = line
        .iter()
        .map(|s| {
            s.predictions
                .iter()
                .filter_map(|p| labels.get(p.class))
                .collect()
        })
        .collect();
    assert_eq!(names[1][0], "x", "x must be top-1, got {:?}", names[1]);
    assert!(
        names[2].contains(&"+"),
        "+ must be in correction reach: {:?}",
        names[2]
    );
    assert!(
        names[3].contains(&"1"),
        "1 must be in correction reach: {:?}",
        names[3]
    );
}

/// The first full equation ever recognized end-to-end: a real `2x + 3 = 7`, drawn on the
/// tablet (2026-07-13; rotated upright — it was written in landscape, and the capture
/// frame is portrait). Pins the whole chain at once: proximity segmentation, the
/// slant-aware stacked-bar merge (the `=` was written ~17° downhill and used to read as
/// `\setminus` + `-`), baseline detrending (the line drifts downhill and used to parse as
/// a tower of subscripts), mixed-height script regions (the tall `2` used to take `x` as
/// a subscript), the expression vocabulary, and the training-prior correction.
///
/// The `7` reads as `>` at top-1 — honestly: drawn without a top-left hook it IS a wide
/// open angle — so the assertion demands its truth within correction reach, which is the
/// product's actual contract.
#[test]
fn the_first_full_equation_2x_plus_3_equals_7() {
    let root = workspace_root();
    let model_path = root.join("train/expr.iwt");
    if !model_path.exists() {
        eprintln!("skipping: {} missing", model_path.display());
        return;
    }
    let blob = std::fs::read(&model_path).expect("read model");
    let weights = Weights::parse(&blob).expect("parse");
    let labels = Labels::from_lines(
        &std::fs::read_to_string(root.join("train/expr.labels.txt")).expect("labels"),
    );
    let counts: Vec<u32> = std::fs::read_to_string(root.join("train/expr.counts.txt"))
        .expect("counts")
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let ink = Ink::decode(
        &std::fs::read(root.join("crates/core/tests/data/equation_2x_plus_3_eq_7.ink"))
            .expect("fixture"),
    )
    .expect("decode");

    let (_ink, line) =
        ink2tex_core::recognize_line(&ink, &weights, &labels, Some(&counts), 5).expect("recognize");
    let names: Vec<Vec<&str>> = line
        .iter()
        .map(|s| {
            s.predictions
                .iter()
                .filter_map(|p| labels.get(p.class))
                .collect()
        })
        .collect();
    assert_eq!(line.len(), 6, "2, x, +, 3, =, 7 — six symbols, = merged");
    for (i, want) in ["2", "x", "+", "3", "="].iter().enumerate() {
        assert_eq!(&names[i][0], want, "symbol {i}: {:?}", names[i]);
    }
    assert!(
        names[5].contains(&"7"),
        "7 must be in correction reach: {:?}",
        names[5]
    );

    let latex = ink2tex_core::recognize_expression(&ink, &weights, &labels, Some(&counts), 3)
        .expect("latex");
    assert!(
        latex.starts_with("2x+3="),
        "flat baseline parse, got {latex:?}"
    );
}

/// The SAME equation, exactly as the digitizer delivered it: rotated 90°, because the
/// tablet was held in landscape — the natural grip for a long expression. This is the
/// capture that used to require hand-rotation. `orient::auto_orient` must detect the
/// vertical symbol line, hold its three-way ballot (as-is / CW / CCW), and land on the
/// same reading as the manually-rotated fixture above.
#[test]
fn a_landscape_equation_orients_itself() {
    let root = workspace_root();
    let model_path = root.join("train/expr.iwt");
    if !model_path.exists() {
        eprintln!("skipping: {} missing", model_path.display());
        return;
    }
    let blob = std::fs::read(&model_path).expect("read model");
    let weights = Weights::parse(&blob).expect("parse");
    let labels = Labels::from_lines(
        &std::fs::read_to_string(root.join("train/expr.labels.txt")).expect("labels"),
    );
    let counts: Vec<u32> = std::fs::read_to_string(root.join("train/expr.counts.txt"))
        .expect("counts")
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let ink = Ink::decode(
        &std::fs::read(root.join("crates/core/tests/data/equation_landscape_raw.ink"))
            .expect("fixture"),
    )
    .expect("decode");

    let latex = ink2tex_core::recognize_expression(&ink, &weights, &labels, Some(&counts), 3)
        .expect("latex");
    assert!(
        latex.starts_with("2x+3="),
        "landscape ink did not orient itself: {latex:?}"
    );
}

/// `(x+1)` composited from REAL collected glyphs (device ink; only the layout is
/// synthetic), spaced like a tight writer — gaps ≈ 0.3 x-heights. This is the tall-paren
/// case: line-like strokes (parens, the flagged 1) used to inflate the median stroke
/// size that sets the merge threshold, and `x+` fused into one blob that classified as
/// `\aleph`. The threshold's median is now taken over compact strokes only, and this
/// fixture holds the line: five symbols, parens included, every glyph correct.
#[test]
fn tight_tall_parens_do_not_swallow_their_contents() {
    let root = workspace_root();
    let model_path = root.join("train/expr.iwt");
    if !model_path.exists() {
        eprintln!("skipping: {} missing", model_path.display());
        return;
    }
    let blob = std::fs::read(&model_path).expect("read model");
    let weights = Weights::parse(&blob).expect("parse");
    let labels = Labels::from_lines(
        &std::fs::read_to_string(root.join("train/expr.labels.txt")).expect("labels"),
    );
    let counts: Vec<u32> = std::fs::read_to_string(root.join("train/expr.counts.txt"))
        .expect("counts")
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let ink = Ink::decode(
        &std::fs::read(root.join("crates/core/tests/data/parens_tight_composite.ink"))
            .expect("fixture"),
    )
    .expect("decode");

    let latex = ink2tex_core::recognize_expression(&ink, &weights, &labels, Some(&counts), 3)
        .expect("latex");
    assert_eq!(
        latex, "(x+1)",
        "the tight-paren composite must parse exactly"
    );
}

/// The correction round-trip (M4's core loop) on real ink: analyze the equation, correct
/// the one wrong symbol (`7` read as `>`) by choosing from ITS OWN candidate list, and
/// the recomposed expression must come out exactly right. This is "every fix is one tap"
/// as a testable property — and `compose` must re-parse structure from scratch, because a
/// corrected label can legitimately change the layout.
#[test]
fn correcting_the_seven_by_one_tap_yields_the_exact_equation() {
    let root = workspace_root();
    let model_path = root.join("train/expr.iwt");
    if !model_path.exists() {
        eprintln!("skipping: {} missing", model_path.display());
        return;
    }
    let blob = std::fs::read(&model_path).expect("read model");
    let weights = Weights::parse(&blob).expect("parse");
    let labels = Labels::from_lines(
        &std::fs::read_to_string(root.join("train/expr.labels.txt")).expect("labels"),
    );
    let counts: Vec<u32> = std::fs::read_to_string(root.join("train/expr.counts.txt"))
        .expect("counts")
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let ink = Ink::decode(
        &std::fs::read(root.join("crates/core/tests/data/equation_2x_plus_3_eq_7.ink"))
            .expect("fixture"),
    )
    .expect("decode");

    let (_oriented, symbols) =
        ink2tex_core::analyze(&ink, &weights, &labels, Some(&counts), 5).expect("analyze");
    assert_eq!(symbols.len(), 6);

    // Uncorrected: the known state of the world.
    let (latex, svg) = ink2tex_core::compose(&symbols, &[0, 0, 0, 0, 0, 0]);
    assert!(latex.starts_with("2x+3="), "uncorrected: {latex}");
    assert!(svg.starts_with("<svg"), "the typesetter must render it");

    // One tap: pick `7` from the last symbol's own candidates.
    let seven = symbols[5]
        .candidates
        .iter()
        .position(|(l, _)| l == "7")
        .expect("7 must be in correction reach");
    let (latex, _svg) = ink2tex_core::compose(&symbols, &[0, 0, 0, 0, 0, seven]);
    assert_eq!(latex, "2x+3=7", "one correction must finish the equation");
}
