//! ink2tex-core — the device-free recognizer. Everything here builds and tests
//! on a laptop (`cargo test -p ink2tex-core`): no libremarkable, no `/dev/input`,
//! no framebuffer. The digitizer->screen transform lives in `crates/rm`, not here
//! (see `docs/core-invariants.md`).
//!
//! M0 ships the ink data model (`Point` / `Stroke` / `Ink`) and the `.ink`
//! container format. The segmentation / classification / structure / latex stages
//! (DESIGN.md §4) land in later milestones.

pub mod classify;
pub mod error;
pub mod format;
pub mod latex;
pub mod line;
pub mod segment;
pub mod stroke;
pub mod structure;

pub use classify::Prediction;
pub use error::{Error, Result};
pub use latex::to_latex;
pub use line::{recognize_expression, recognize_line, LineSymbol};
pub use segment::segment;
pub use stroke::{Ink, Point, Stroke};
pub use structure::{parse as parse_structure, BBox, Slt, Symbol as PositionedSymbol};
