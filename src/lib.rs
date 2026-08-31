//! serra — analytical multi-material meshes from voxelized segmentations.
//!
//! The crate is organised as a pipeline, one module per stage. See
//! `docs/developer.md` for how they fit together.

pub mod grid;
pub mod orient;
pub mod tables;

use pyo3::prelude::*;

#[pymodule]
fn _serra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
