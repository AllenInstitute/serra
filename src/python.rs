//! PyO3 bindings — the only module that knows about Python.
//!
//! The Rust side returns plain numpy arrays; the friendly `Mesh` type and all
//! file IO live in the Python package, where they are far easier to maintain.
//!
//! Label dtypes are handled by a single generic function that the compiler
//! monomorphises, rather than by generating one wrapper class per dtype. (zmesh
//! renders eight such classes from a Jinja template and pastes them into its
//! .pyx by hand.)

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::extract::{extract, Extraction};
use crate::grid::{Label, VolumeView};
use crate::mesh::{build, MeshOptions, TriangleMesh};
use crate::orient::Layout;

/// Arrays handed back for one object: vertices, faces, and optionally normals.
type MeshArrays<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<u32>>,
    Option<Bound<'py, PyArray2<f32>>>,
);

/// Extracts every object's surface from a labelled volume in a single pass.
#[pyclass(module = "serra._serra")]
pub struct Mesher {
    resolution: [f64; 3],
    layout: Layout,
    shape: [usize; 3],
    extraction: Option<Extraction>,
}

fn run<'py, T: Label + numpy::Element>(
    py: Python<'py>,
    array: PyReadonlyArray3<'py, T>,
    close: bool,
) -> ([usize; 3], Extraction) {
    let view = array.as_array();
    let s = view.shape();
    let shape = [s[0], s[1], s[2]];
    // The extraction touches no Python objects, so the GIL is released
    // (pyo3 0.29 spells this `detach`) and other threads can run.
    let extraction = py.detach(move || extract(&VolumeView::new(view, close)));
    (shape, extraction)
}

#[pymethods]
impl Mesher {
    #[new]
    #[pyo3(signature = (voxel_resolution=vec![1.0, 1.0, 1.0], axis_order="XYZ", y_down=false))]
    fn new(voxel_resolution: Vec<f64>, axis_order: &str, y_down: bool) -> PyResult<Self> {
        if voxel_resolution.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "voxel_resolution must have 3 entries, got {}",
                voxel_resolution.len()
            )));
        }
        // `is_finite` also rejects NaN, so the comparison never sees one.
        if voxel_resolution.iter().any(|&r| !r.is_finite() || r <= 0.0) {
            return Err(PyValueError::new_err(
                "voxel_resolution entries must be finite and positive",
            ));
        }
        let mut layout = Layout::from_axis_order(axis_order)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if y_down {
            layout = layout.with_flipped_physical_axis(1);
        }
        Ok(Mesher {
            resolution: [
                voxel_resolution[0],
                voxel_resolution[1],
                voxel_resolution[2],
            ],
            layout,
            shape: [0; 3],
            extraction: None,
        })
    }

    /// Mesh a 3-D array of unsigned integer labels.
    ///
    /// With `close`, the volume is treated as if surrounded by background, so
    /// objects reaching the array edge come back sealed. This costs nothing:
    /// the border is virtual, not a padded copy of the input.
    #[pyo3(signature = (labels, close=false))]
    fn mesh(&mut self, py: Python<'_>, labels: &Bound<'_, PyAny>, close: bool) -> PyResult<()> {
        // Free the previous result before building the next one, so peak memory
        // is one extraction rather than two.
        self.extraction = None;

        macro_rules! attempt {
            ($t:ty) => {
                if let Ok(array) = labels.extract::<PyReadonlyArray3<$t>>() {
                    let (shape, extraction) = run(py, array, close);
                    self.shape = shape;
                    self.extraction = Some(extraction);
                    return Ok(());
                }
            };
        }
        attempt!(u8);
        attempt!(u16);
        attempt!(u32);
        attempt!(u64);

        Err(PyTypeError::new_err(
            "labels must be a 3-D numpy array of uint8, uint16, uint32 or uint64. \
             Signed arrays can be reinterpreted without copying, e.g. \
             labels.view(np.uint32) for an int32 array.",
        ))
    }

    /// Label ids present in the volume, ascending. Background (0) is excluded.
    fn ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        match &self.extraction {
            Some(e) => e.labels.clone().into_pyarray(py),
            None => Vec::<u64>::new().into_pyarray(py),
        }
    }

    /// Whether a label is present.
    fn contains(&self, label: u64) -> bool {
        self.extraction
            .as_ref()
            .is_some_and(|e| e.position_of(label).is_some())
    }

    /// Arrays for one object, or `None` if the label is absent.
    #[pyo3(signature = (label, normals=false))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        label: u64,
        normals: bool,
    ) -> PyResult<Option<MeshArrays<'py>>> {
        let Some(extraction) = self.extraction.as_ref() else {
            return Err(PyValueError::new_err(
                "no volume has been meshed yet; call mesh() first",
            ));
        };
        let Some(raw) = extraction.mesh(label) else {
            return Ok(None);
        };

        let options = MeshOptions {
            resolution: self.resolution,
            layout: self.layout,
            shape: self.shape,
            normals,
        };
        let built = py.detach(move || build(raw, &options));
        Ok(Some(to_arrays(py, built)))
    }

    /// Drop one object's surface, freeing its memory.
    fn erase(&mut self, label: u64) -> bool {
        match self.extraction.as_mut() {
            Some(e) => e.remove(label),
            None => false,
        }
    }

    /// Drop every stored surface.
    fn clear(&mut self) {
        self.extraction = None;
        self.shape = [0; 3];
    }

    fn __len__(&self) -> usize {
        self.extraction.as_ref().map_or(0, |e| e.labels.len())
    }

    fn __repr__(&self) -> String {
        format!(
            "Mesher(voxel_resolution={:?}, objects={})",
            self.resolution,
            self.__len__()
        )
    }
}

fn to_arrays(py: Python<'_>, m: TriangleMesh) -> MeshArrays<'_> {
    let nv = m.vertices.len();
    let nf = m.faces.len();

    let vertices = Array2::from_shape_vec((nv, 3), m.vertices.into_iter().flatten().collect())
        .expect("vertex buffer is a multiple of 3");
    let faces = Array2::from_shape_vec((nf, 3), m.faces.into_iter().flatten().collect())
        .expect("face buffer is a multiple of 3");
    let normals = m.normals.map(|n| {
        Array2::from_shape_vec((nv, 3), n.into_iter().flatten().collect())
            .expect("normal buffer is a multiple of 3")
            .into_pyarray(py)
    });

    (vertices.into_pyarray(py), faces.into_pyarray(py), normals)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mesher>()?;
    Ok(())
}
