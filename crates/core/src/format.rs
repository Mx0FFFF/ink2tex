//! The `.ink` binary container (little-endian, hand-rolled — no serde, so core
//! stays dependency-free and cross-compiles clean). Layout, all little-endian:
//!
//! ```text
//! magic:         [u8;4] = b"INK1"
//! version:       u16    = 1
//! flags:         u16    = 0  (reserved)
//! source_width:  f32
//! source_height: f32
//! stroke_count:  u32
//! repeat stroke_count times:
//!     point_count: u32
//!     repeat point_count times (28 bytes each):
//!         x, y, pressure, tilt_x, tilt_y : 5 x f32
//!         t_us                           : u64
//! ```
//!
//! There is no file I/O here: `encode` returns a `Vec<u8>`, `decode` consumes a
//! `&[u8]`. The frontends (`crates/desktop`, `crates/rm`) own `std::fs`.

use crate::error::{Error, Result};
use crate::stroke::{Ink, Point, Stroke};

const MAGIC: [u8; 4] = *b"INK1";
const VERSION: u16 = 1;
const POINT_BYTES: usize = 5 * 4 + 8; // five f32 + one u64 = 28

impl Ink {
    /// Serialize to the `.ink` byte layout documented above.
    pub fn encode(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(20 + self.strokes.len() * 4 + self.point_count() * POINT_BYTES);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&self.source_width.to_le_bytes());
        out.extend_from_slice(&self.source_height.to_le_bytes());
        out.extend_from_slice(&(self.strokes.len() as u32).to_le_bytes());
        for s in &self.strokes {
            out.extend_from_slice(&(s.points.len() as u32).to_le_bytes());
            for p in &s.points {
                out.extend_from_slice(&p.x.to_le_bytes());
                out.extend_from_slice(&p.y.to_le_bytes());
                out.extend_from_slice(&p.pressure.to_le_bytes());
                out.extend_from_slice(&p.tilt_x.to_le_bytes());
                out.extend_from_slice(&p.tilt_y.to_le_bytes());
                out.extend_from_slice(&p.t_us.to_le_bytes());
            }
        }
        out
    }

    /// Parse the `.ink` byte layout. Rejects bad magic, unknown versions, and
    /// truncation — never panics on malformed or hostile input.
    pub fn decode(bytes: &[u8]) -> Result<Ink> {
        let mut c = Cursor::new(bytes);
        let magic = c.take_array::<4>()?;
        if magic != MAGIC {
            return Err(Error::BadMagic {
                found: magic,
                expected: MAGIC,
            });
        }
        let version = c.take_u16()?;
        if version != VERSION {
            return Err(Error::UnsupportedVersion(version, VERSION));
        }
        let _flags = c.take_u16()?;
        let source_width = c.take_f32()?;
        let source_height = c.take_f32()?;
        let stroke_count = c.take_u32()? as usize;

        // Cap the pre-allocation so a corrupt count can't request a huge Vec; the
        // loop still errors cleanly via `take_*` if the data is actually short.
        let mut strokes = Vec::with_capacity(stroke_count.min(4096));
        for _ in 0..stroke_count {
            let point_count = c.take_u32()? as usize;
            let mut points = Vec::with_capacity(point_count.min(4096));
            for _ in 0..point_count {
                points.push(Point {
                    x: c.take_f32()?,
                    y: c.take_f32()?,
                    pressure: c.take_f32()?,
                    tilt_x: c.take_f32()?,
                    tilt_y: c.take_f32()?,
                    t_us: c.take_u64()?,
                });
            }
            strokes.push(Stroke { points });
        }
        Ok(Ink {
            source_width,
            source_height,
            strokes,
        })
    }
}

/// Minimal bounds-checked little-endian reader. Returns `Truncated` instead of
/// panicking when the buffer is too short.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated {
            offset: self.pos,
            needed: n,
            available: self.buf.len().saturating_sub(self.pos),
        })?;
        if end > self.buf.len() {
            return Err(Error::Truncated {
                offset: self.pos,
                needed: n,
                available: self.buf.len().saturating_sub(self.pos),
            });
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn take_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take_array::<2>()?))
    }
    fn take_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take_array::<4>()?))
    }
    fn take_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take_array::<8>()?))
    }
    fn take_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take_array::<4>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Ink {
        let mut ink = Ink::new().with_source(20967.0, 15725.0);
        ink.push(Stroke::from_iter([
            Point::new(0.10, 0.20, 0.5, 0.0, 0.0, 0),
            Point::new(0.11, 0.22, 0.6, 0.01, -0.01, 8_000),
        ]));
        ink.push(Stroke::from_iter([Point::new(
            0.90, 0.80, 0.9, 0.2, 0.1, 16_000,
        )]));
        ink
    }

    #[test]
    fn roundtrip_preserves_everything() {
        let ink = sample();
        let decoded = Ink::decode(&ink.encode()).unwrap();
        assert_eq!(ink, decoded);
    }

    #[test]
    fn empty_ink_roundtrips() {
        let ink = Ink::new();
        assert_eq!(ink, Ink::decode(&ink.encode()).unwrap());
    }

    #[test]
    fn header_layout_is_stable() {
        let bytes = sample().encode();
        assert_eq!(&bytes[0..4], b"INK1");
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), VERSION);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample().encode();
        bytes[0] = b'X';
        assert!(matches!(Ink::decode(&bytes), Err(Error::BadMagic { .. })));
    }

    #[test]
    fn rejects_bad_version() {
        let mut bytes = sample().encode();
        bytes[4] = 0xFF; // corrupt the version's low byte
        assert!(matches!(
            Ink::decode(&bytes),
            Err(Error::UnsupportedVersion(_, _))
        ));
    }

    #[test]
    fn rejects_truncation() {
        let bytes = sample().encode();
        // Chop the last point's t_us in half.
        assert!(matches!(
            Ink::decode(&bytes[..bytes.len() - 4]),
            Err(Error::Truncated { .. })
        ));
    }

    #[test]
    fn corrupt_count_errors_without_panicking() {
        // A stroke_count of 4 billion with no following data must error cleanly.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"INK1");
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0f32.to_le_bytes());
        bytes.extend_from_slice(&0f32.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(Ink::decode(&bytes), Err(Error::Truncated { .. })));
    }
}
