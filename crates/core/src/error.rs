//! Core's error type. Core uses `thiserror` and returns `Result` everywhere; it
//! must never panic on malformed input (that would be a frontend crash on an
//! appliance the human is holding). Binaries convert these into `anyhow`.

use thiserror::Error;

/// Everything that can go wrong inside the device-free core.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    /// An `.ink` buffer didn't start with the `INK1` magic.
    #[error("not an .ink file: bad magic {found:x?} (expected {expected:x?})")]
    BadMagic { found: [u8; 4], expected: [u8; 4] },

    /// The `.ink` format version isn't one this build understands.
    #[error("unsupported .ink version {0} (this build supports {1})")]
    UnsupportedVersion(u16, u16),

    /// The buffer ended mid-record — truncated or corrupt.
    #[error("truncated .ink: needed {needed} more bytes at offset {offset}, had {available}")]
    Truncated {
        offset: usize,
        needed: usize,
        available: usize,
    },

    /// A weights blob was malformed (bad magic, unsupported version, or a tensor
    /// whose declared shape doesn't match its byte length).
    #[error("malformed weights blob: {0}")]
    BadWeights(&'static str),
}

/// Core's ubiquitous result alias.
pub type Result<T> = core::result::Result<T, Error>;
