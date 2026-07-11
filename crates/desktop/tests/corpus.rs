//! The corpus regression suite — the project's immune system (CLAUDE.md §4,
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

        let top5: Vec<&str> = preds
            .iter()
            .map(|p| labels.get(p.class).unwrap_or("?"))
            .collect();
        assert!(
            top5.contains(&expected),
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
