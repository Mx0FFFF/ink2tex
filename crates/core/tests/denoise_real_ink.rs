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

/// `radical_over_expression.ink` is a real capture of a hand-drawn `√x+1`, written the way
/// it is printed: the tick on the left, the overbar spanning right, and `x + 1` tucked
/// underneath. That makes the radical's bounding box *enclose* its contents — and
/// bbox-based clustering merged all six strokes into one "symbol", which the classifier
/// then read as `\mathscr{F}` at 13.9%. `\sqrt` was broken end-to-end, while all 17
/// structure tests passed, because those hand-feed positioned symbols and never touch
/// segmentation.
#[test]
fn a_real_hand_drawn_radical_does_not_swallow_its_contents() {
    use ink2tex_core::segment::segment;

    let bytes = include_bytes!("data/radical_over_expression.ink");
    let ink = Ink::decode(bytes).expect("decode radical_over_expression.ink");
    assert_eq!(ink.strokes.len(), 6, "fixture: √ + x(2 strokes) + +(2) + 1");

    let groups = segment(&ink.strokes);
    assert_eq!(
        groups.len(),
        4,
        "√, x, +, 1 — got {} group(s): {groups:?}",
        groups.len()
    );
    // The radical must be alone: it is stroke 0 and it must not have absorbed anything.
    let radical = groups
        .iter()
        .find(|g| g.contains(&0))
        .expect("stroke 0 (the radical) is in some group");
    assert_eq!(
        radical.len(),
        1,
        "the radical dragged strokes in with it: {radical:?}"
    );
}
