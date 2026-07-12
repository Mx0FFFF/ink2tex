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
    let path = root.join("train/model_v2.labels.txt");
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

/// The fix for "the model cannot say x": on a real capture of `√x+1`, the expression
/// path must now put the literal `x` and `+` at top-1 of their symbols. This pins the
/// whole chain — deep-k, vocabulary mask, training-prior division — against real ink;
/// the unit tests only ever see synthetic distributions.
#[test]
fn everyday_tokens_win_on_real_ink_in_expression_mode() {
    let root = workspace_root();
    let model_path = root.join("train/model_v2.iwt");
    if !model_path.exists() {
        eprintln!("skipping: {} missing", model_path.display());
        return;
    }
    let blob = std::fs::read(&model_path).expect("read model");
    let weights = Weights::parse(&blob).expect("parse");
    let labels = Labels::from_lines(
        &std::fs::read_to_string(root.join("train/model_v2.labels.txt")).expect("labels"),
    );
    let counts: Vec<u32> = std::fs::read_to_string(root.join("train/model_v2.counts.txt"))
        .expect("counts")
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let ink = Ink::decode(
        &std::fs::read(root.join("crates/core/tests/data/radical_over_expression.ink"))
            .expect("fixture"),
    )
    .expect("decode");

    let out = ink2tex_core::recognize_expression(&ink, &weights, &labels, Some(&counts), 3)
        .expect("recognize");
    assert!(
        out.contains('x') && out.contains('+'),
        "expected the literals x and + inside the expression, got {out:?}"
    );
}
