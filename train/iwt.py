"""Writer for the `.iwt` weights blob — the byte-exact mirror of
`crates/core/src/classify/weights.rs`. This is the contract between the PyTorch
trainer (producer) and the Rust inference kernel (consumer); the two MUST agree
byte-for-byte, so this file is deliberately tiny and dependency-free (numpy
optional). Verified against core's parser via `ink2tex-desktop --dump-weights`.

Layout (all little-endian):
    magic "IW01" | version u16=1 | flags u16=0 | tensor_count u32
    per tensor: name_len u16, name utf8, dtype u8 (0=i8,1=i32,2=f32),
                scale f32, ndim u8, dims[ndim] u32, data_len u32, data
"""

import struct

DTYPE_I8, DTYPE_I32, DTYPE_F32 = 0, 1, 2


def _flatten(x):
    for v in x:
        if isinstance(v, (list, tuple)):
            yield from _flatten(v)
        else:
            yield v


def _to_bytes(data, dtype):
    """A numpy array or a (possibly nested) iterable of numbers → LE bytes."""
    if hasattr(data, "astype"):  # numpy array
        np_dtype = {DTYPE_I8: "<i1", DTYPE_I32: "<i4", DTYPE_F32: "<f4"}[dtype]
        return data.astype(np_dtype).tobytes()
    fmt = {DTYPE_I8: "<b", DTYPE_I32: "<i", DTYPE_F32: "<f"}[dtype]
    cast = float if dtype == DTYPE_F32 else int
    return b"".join(struct.pack(fmt, cast(v)) for v in _flatten(data))


class WeightsWriter:
    def __init__(self):
        self._tensors = []  # (name, dtype, scale, dims, data_bytes)

    def i8(self, name, dims, scale, data):
        self._tensors.append((name, DTYPE_I8, float(scale), list(dims), _to_bytes(data, DTYPE_I8)))

    def i32(self, name, dims, data):
        self._tensors.append((name, DTYPE_I32, 0.0, list(dims), _to_bytes(data, DTYPE_I32)))

    def f32(self, name, dims, data):
        self._tensors.append((name, DTYPE_F32, 0.0, list(dims), _to_bytes(data, DTYPE_F32)))

    def to_bytes(self):
        out = bytearray(b"IW01")
        out += struct.pack("<HHI", 1, 0, len(self._tensors))
        for name, dtype, scale, dims, data in self._tensors:
            nb = name.encode("utf-8")
            out += struct.pack("<H", len(nb)) + nb
            out += struct.pack("<B", dtype)
            out += struct.pack("<f", scale)
            out += struct.pack("<B", len(dims))
            for d in dims:
                out += struct.pack("<I", int(d))
            out += struct.pack("<I", len(data)) + data
        return bytes(out)

    def write(self, path):
        with open(path, "wb") as f:
            f.write(self.to_bytes())
