//! Integration-level check that the public `.ink` API round-trips the way the
//! frontends use it (encode -> bytes -> decode). Error-path coverage lives in the
//! `format.rs` unit tests. This is the first entry in the project's regression
//! suite; every bug fix from here on adds a `tests/corpus/*.ink` case.

use ink2tex_core::{Ink, Point, Stroke};

#[test]
fn public_api_roundtrip() {
    let mut ink = Ink::new().with_source(1872.0, 1404.0);
    let stroke: Stroke = (0..32)
        .map(|i| {
            let t = i as f32 / 31.0;
            Point::new(
                t,
                0.5 + 0.25 * (t * std::f32::consts::TAU).sin(),
                0.7,
                0.0,
                0.0,
                i as u64 * 5_000,
            )
        })
        .collect();
    ink.push(stroke);

    let decoded = Ink::decode(&ink.encode()).expect("decode");
    assert_eq!(ink, decoded);
    assert_eq!(decoded.strokes.len(), 1);
    assert_eq!(decoded.point_count(), 32);
    assert_eq!(decoded.aspect_ratio(), Some(1872.0 / 1404.0));
}
