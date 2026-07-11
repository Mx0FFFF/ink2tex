//! A deterministic synthetic `.ink` so the renderer — and M0's `make replay`
//! done-criterion — can be exercised without a tablet. No randomness: the bytes
//! are identical every run, which keeps it usable as a CI fixture.

use std::f32::consts::TAU;

use ink2tex_core::{Ink, Point, Stroke};

pub fn sample_ink() -> Ink {
    // reMarkable 2 screen aspect, so the PNG is framed like the device.
    let mut ink = Ink::new().with_source(1872.0, 1404.0);

    // Stroke 1: a sine wave across the middle, pressure varying along it.
    let sine: Stroke = (0..=120)
        .map(|i| {
            let u = i as f32 / 120.0;
            let x = 0.10 + 0.80 * u;
            let y = 0.45 - 0.18 * (u * 2.0 * TAU).sin();
            let pressure = 0.35 + 0.50 * (u * TAU).sin().abs();
            Point::new(x, y, pressure, 0.0, 0.0, (i as u64) * 6_000)
        })
        .collect();
    ink.push(sine);

    // Stroke 2: a diagonal.
    let diag: Stroke = (0..=40)
        .map(|i| {
            let u = i as f32 / 40.0;
            Point::new(
                0.15 + 0.30 * u,
                0.75 - 0.30 * u,
                0.8,
                0.0,
                0.0,
                800_000 + (i as u64) * 4_000,
            )
        })
        .collect();
    ink.push(diag);

    // Stroke 3: a small closed loop (a crude 'o').
    let ring: Stroke = (0..=60)
        .map(|i| {
            let a = (i as f32 / 60.0) * TAU;
            Point::new(
                0.72 + 0.06 * a.cos(),
                0.70 + 0.08 * a.sin(),
                0.6,
                0.0,
                0.0,
                1_200_000 + (i as u64) * 5_000,
            )
        })
        .collect();
    ink.push(ring);

    ink
}
