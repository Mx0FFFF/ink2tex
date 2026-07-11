//! The `.iwt` weights blob — a flat little-endian container of named quantized
//! tensors that the Python trainer emits and core consumes. Layout, all LE:
//!
//! ```text
//! magic:        [u8;4] = b"IW01"
//! version:      u16 = 1
//! flags:        u16 = 0
//! tensor_count: u32
//! repeat tensor_count times:
//!     name_len: u16,  name: [name_len] utf-8
//!     dtype:    u8    (0 = i8, 1 = i32, 2 = f32)
//!     scale:    f32   (dequant scale for i8 tensors; 0.0 otherwise)
//!     ndim:     u8,   dims: [ndim] u32
//!     data_len: u32,  data: [data_len] bytes  (i32/f32 stored little-endian)
//! ```
//!
//! ## Systems concept: mmap + zero-copy
//! The frontend `mmap`s this file so the OS pages it in on demand and shares it
//! across processes — the weights never hit the heap as a `Vec`. Core is handed the
//! resulting `&[u8]` and parses it **borrowing** in place: `as_i8()` reinterprets
//! the mapped bytes directly (i8 and u8 share size/alignment, and every bit pattern
//! is a valid i8), so the bulk weight arrays are truly zero-copy. Only the small
//! i32 bias / f32 vectors are decoded into `Vec`s.

use crate::error::{Error, Result};

const MAGIC: [u8; 4] = *b"IW01";
const VERSION: u16 = 1;

const DTYPE_I8: u8 = 0;
const DTYPE_I32: u8 = 1;
const DTYPE_F32: u8 = 2;

/// One named tensor, borrowing its bytes from the mapped blob.
pub struct Tensor<'a> {
    pub name: &'a str,
    pub dtype: u8,
    pub scale: f32,
    pub dims: Vec<u32>,
    data: &'a [u8],
}

impl<'a> Tensor<'a> {
    /// Element count = product of dims.
    pub fn len(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Zero-copy view of an i8 tensor's data. Returns `&[]` if the dtype is wrong.
    pub fn as_i8(&self) -> &'a [i8] {
        if self.dtype != DTYPE_I8 {
            return &[];
        }
        // SAFETY: i8 and u8 have identical size/alignment and every byte is a valid
        // i8; the slice length is unchanged.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr() as *const i8, self.data.len()) }
    }

    /// Decode an i32 tensor (small bias vectors) into a `Vec`.
    pub fn as_i32(&self) -> Vec<i32> {
        if self.dtype != DTYPE_I32 {
            return Vec::new();
        }
        self.data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Decode an f32 tensor into a `Vec`.
    pub fn as_f32(&self) -> Vec<f32> {
        if self.dtype != DTYPE_F32 {
            return Vec::new();
        }
        self.data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// A parsed weights blob: the tensor directory, each borrowing the mapped bytes.
pub struct Weights<'a> {
    tensors: Vec<Tensor<'a>>,
}

impl<'a> Weights<'a> {
    /// Parse a blob. Validates magic/version and that each tensor's `data_len`
    /// matches its dtype × element count — so downstream `as_*` calls are safe.
    pub fn parse(blob: &'a [u8]) -> Result<Weights<'a>> {
        let mut c = Reader::new(blob);
        if c.take(4)? != MAGIC {
            return Err(Error::BadWeights("bad magic (expected IW01)"));
        }
        if c.u16()? != VERSION {
            return Err(Error::BadWeights("unsupported version"));
        }
        let _flags = c.u16()?;
        let count = c.u32()? as usize;

        let mut tensors = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let name_len = c.u16()? as usize;
            let name = core::str::from_utf8(c.take(name_len)?)
                .map_err(|_| Error::BadWeights("tensor name not utf-8"))?;
            let dtype = c.u8()?;
            let scale = c.f32()?;
            let ndim = c.u8()? as usize;
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                dims.push(c.u32()?);
            }
            let data_len = c.u32()? as usize;
            let data = c.take(data_len)?;

            let elem = dims.iter().map(|&d| d as usize).product::<usize>();
            let unit = match dtype {
                DTYPE_I8 => 1,
                DTYPE_I32 | DTYPE_F32 => 4,
                _ => return Err(Error::BadWeights("unknown dtype")),
            };
            if data_len != elem * unit {
                return Err(Error::BadWeights("tensor data length != dims × dtype size"));
            }
            tensors.push(Tensor {
                name,
                dtype,
                scale,
                dims,
                data,
            });
        }
        Ok(Weights { tensors })
    }

    /// Look up a tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor<'a>> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Iterate the tensors in file order.
    pub fn tensors(&self) -> impl Iterator<Item = &Tensor<'a>> {
        self.tensors.iter()
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Serializes the blob. The Python trainer mirrors this exact byte layout; keeping
/// a Rust writer here gives us a reference encoder and lets tests round-trip.
#[derive(Default)]
pub struct WeightsWriter {
    tensors: Vec<u8>,
    count: u32,
}

impl WeightsWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn i8(&mut self, name: &str, dims: &[u32], scale: f32, data: &[i8]) {
        // SAFETY: i8→u8 reinterpret, identical layout.
        let bytes = unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
        self.push(name, DTYPE_I8, scale, dims, bytes);
    }

    pub fn i32(&mut self, name: &str, dims: &[u32], data: &[i32]) {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.push(name, DTYPE_I32, 0.0, dims, &bytes);
    }

    pub fn f32(&mut self, name: &str, dims: &[u32], data: &[f32]) {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.push(name, DTYPE_F32, 0.0, dims, &bytes);
    }

    fn push(&mut self, name: &str, dtype: u8, scale: f32, dims: &[u32], data: &[u8]) {
        self.tensors
            .extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.tensors.extend_from_slice(name.as_bytes());
        self.tensors.push(dtype);
        self.tensors.extend_from_slice(&scale.to_le_bytes());
        self.tensors.push(dims.len() as u8);
        for d in dims {
            self.tensors.extend_from_slice(&d.to_le_bytes());
        }
        self.tensors
            .extend_from_slice(&(data.len() as u32).to_le_bytes());
        self.tensors.extend_from_slice(data);
        self.count += 1;
    }

    pub fn finish(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.tensors.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&self.count.to_le_bytes());
        out.extend_from_slice(&self.tensors);
        out
    }
}

/// Minimal bounds-checked LE reader (mirrors the `.ink` cursor; returns
/// `BadWeights` rather than panicking on truncation).
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or(Error::BadWeights("truncated blob"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_named_tensors() {
        let mut w = WeightsWriter::new();
        w.i8("conv1.w", &[2, 1, 2, 2], 0.005, &[1, 2, 3, 4, 5, 6, 7, 8]);
        w.i32("conv1.b", &[2], &[-10, 20]);
        w.f32("meta.scale", &[1], &[0.0125]);
        let blob = w.finish();

        let parsed = Weights::parse(&blob).unwrap();
        assert_eq!(parsed.len(), 3);

        let c = parsed.get("conv1.w").unwrap();
        assert_eq!(c.dims, vec![2, 1, 2, 2]);
        assert!((c.scale - 0.005).abs() < 1e-9);
        assert_eq!(c.as_i8(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(c.len(), 8);

        assert_eq!(parsed.get("conv1.b").unwrap().as_i32(), vec![-10, 20]);
        assert_eq!(parsed.get("meta.scale").unwrap().as_f32(), vec![0.0125]);
        assert!(parsed.get("missing").is_none());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut blob = WeightsWriter::new().finish();
        blob[0] = b'X';
        assert!(matches!(Weights::parse(&blob), Err(Error::BadWeights(_))));
    }

    #[test]
    fn rejects_truncation() {
        let mut w = WeightsWriter::new();
        w.i8("t", &[4], 1.0, &[1, 2, 3, 4]);
        let blob = w.finish();
        assert!(matches!(
            Weights::parse(&blob[..blob.len() - 2]),
            Err(Error::BadWeights(_))
        ));
    }
}
