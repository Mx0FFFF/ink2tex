//! `denoise` against ink that actually came off a tablet.
//!
//! `noisy_row.ink` is a real capture: a hand-drawn `α Σ Π √ ∞`, plus three stray taps on
//! xochitl's toolbar that the digitizer handed us as strokes because we read the pen below
//! xochitl and cannot see what tool it thinks is selected. Before this filter, the taps
//! became symbols and `structure` turned them into superscripts: `…\infty^{\slash}`.
//!
//! Synthetic unit tests can be tuned until they pass. This one cannot.

use ink2tex_core::denoise::denoise;
use ink2tex_core::Ink;

#[test]
fn strips_the_toolbar_taps_from_a_real_capture_and_keeps_every_symbol() {
    let bytes = include_bytes!("data/noisy_row.ink");
    let ink = Ink::decode(bytes).expect("decode noisy_row.ink");
    assert_eq!(ink.strokes.len(), 8, "fixture: 5 symbols + 3 taps");

    let kept = denoise(&ink.strokes);
    assert_eq!(
        kept.len(),
        5,
        "expected the 5 symbols and none of the 3 taps"
    );

    // And the *right* five: every survivor must be one of the big strokes. The taps are
    // 21-28 points; the symbols are 424-653. Checking the count alone would pass even if
    // the filter kept three taps and dropped three symbols.
    for s in &kept {
        assert!(
            s.points.len() > 100,
            "kept a {}-point stroke — that is a tap, not a symbol",
            s.points.len()
        );
    }
}
