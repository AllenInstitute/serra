//! serra — analytical multi-material meshes from voxelized segmentations.
//!
//! The crate is organised as a pipeline, one module per stage. See
//! `docs/developer.md` for how they fit together.

pub mod extract;
pub mod grid;
pub mod mesh;
pub mod orient;
pub mod place;
pub mod python;
pub mod simplify;
pub mod tables;

use pyo3::prelude::*;

#[pymodule]
fn _serra_mesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    python::register(m)?;
    Ok(())
}
