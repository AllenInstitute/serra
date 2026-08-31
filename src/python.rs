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
use rayon::prelude::*;

use crate::extract::{extract_parallel, ExtractOptions, Extraction};
use crate::grid::{Label, VolumeView};
use crate::mesh::{build, MeshOptions, TriangleMesh};
use crate::orient::Layout;
use crate::place::{relax, Relaxation};

/// Arrays handed back for one object: vertices, faces, and optionally normals.
type MeshArrays<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<u32>>,
    Option<Bound<'py, PyArray2<f32>>>,
);

/// Extracts every object's surface from a labelled volume in a single pass.
#[pyclass(module = "serra_mesh._serra_mesh")]
pub struct Mesher {
    resolution: [f64; 3],
    layout: Layout,
    relaxation: Relaxation,
    threads: usize,
    /// A private pool, so a thread count set here cannot be overridden by
    /// RAYON_NUM_THREADS and cannot disturb other rayon users in the process.
    /// `None` means either fully sequential or rayon's global pool.
    pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    shape: [usize; 3],
    extraction: Option<Extraction>,
}

fn run<'py, T: Label + numpy::Element>(
    py: Python<'py>,
    array: PyReadonlyArray3<'py, T>,
    close: bool,
    relaxation: Relaxation,
    threads: usize,
    pool: Option<&rayon::ThreadPool>,
    owned_cells: Option<[usize; 3]>,
) -> ([usize; 3], Extraction) {
    let view = array.as_array();
    let s = view.shape();
    let shape = [s[0], s[1], s[2]];

    // Relaxation must not move vertices whose one-ring the chunk does not
    // fully contain, so the outermost layer of cells is pinned. With `close`
    // there is nothing beyond that layer to be missing, so nothing is pinned
    // and the whole surface is free to smooth.
    let options = ExtractOptions {
        mark_boundary: relaxation.iterations > 0 && !close,
        owned_cells,
    };

    // Neither step touches a Python object, so the GIL is released (pyo3 0.29
    // spells this `detach`) and other threads can run.
    let work = move || {
        let mut extraction = extract_parallel(&VolumeView::new(view, close), &options, threads);
        if relaxation.iterations > 0 {
            if threads == 1 {
                for mesh in extraction.meshes.iter_mut() {
                    relax(mesh, &relaxation);
                }
            } else {
                // Each object's surface is a separate graph, so relaxation is
                // embarrassingly parallel and cannot depend on scheduling: a
                // mesh is only ever touched by the worker that owns it.
                extraction
                    .meshes
                    .par_iter_mut()
                    .for_each(|mesh| relax(mesh, &relaxation));
            }
        }
        extraction
    };

    let extraction = py.detach(move || match pool {
        Some(p) => p.install(work),
        None => work(),
    });
    (shape, extraction)
}

#[pymethods]
impl Mesher {
    #[new]
    #[pyo3(signature = (
        voxel_resolution=vec![1.0, 1.0, 1.0],
        axis_order="XYZ",
        y_down=false,
        relaxation=0,
        max_deviation=0.5,
        relaxation_step=0.5,
        threads=0,
    ))]
    fn new(
        voxel_resolution: Vec<f64>,
        axis_order: &str,
        y_down: bool,
        relaxation: u32,
        max_deviation: f64,
        relaxation_step: f64,
        threads: usize,
    ) -> PyResult<Self> {
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
        if !max_deviation.is_finite() || max_deviation < 0.0 {
            return Err(PyValueError::new_err(
                "max_deviation must be finite and non-negative",
            ));
        }
        if !relaxation_step.is_finite() || relaxation_step <= 0.0 || relaxation_step > 1.0 {
            return Err(PyValueError::new_err("relaxation_step must lie in (0, 1]"));
        }
        // 0 means "use every core", which is rayon's global pool. Any other
        // value gets a private pool so the choice is honoured exactly.
        let pool = if threads > 1 {
            Some(std::sync::Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| PyValueError::new_err(format!("thread pool: {e}")))?,
            ))
        } else {
            None
        };
        Ok(Mesher {
            resolution: [
                voxel_resolution[0],
                voxel_resolution[1],
                voxel_resolution[2],
            ],
            layout,
            threads,
            pool,
            relaxation: Relaxation {
                iterations: relaxation,
                max_deviation,
                step: relaxation_step,
            },
            shape: [0; 3],
            extraction: None,
        })
    }

    /// Mesh a 3-D array of unsigned integer labels.
    ///
    /// With `close`, the volume is treated as if surrounded by background, so
    /// objects reaching the array edge come back sealed. This costs nothing:
    /// the border is virtual, not a padded copy of the input.
    #[pyo3(signature = (labels, close=false, owned_shape=None))]
    fn mesh(
        &mut self,
        py: Python<'_>,
        labels: &Bound<'_, PyAny>,
        close: bool,
        owned_shape: Option<Vec<usize>>,
    ) -> PyResult<()> {
        // Free the previous result before building the next one, so peak memory
        // is one extraction rather than two.
        self.extraction = None;

        let owned_cells = match owned_shape {
            None => None,
            Some(v) if v.len() == 3 => Some([v[0], v[1], v[2]]),
            Some(v) => {
                return Err(PyValueError::new_err(format!(
                    "owned_shape must have 3 entries, got {}",
                    v.len()
                )))
            }
        };

        macro_rules! attempt {
            ($t:ty) => {
                if let Ok(array) = labels.extract::<PyReadonlyArray3<$t>>() {
                    let (shape, extraction) = run(
                        py,
                        array,
                        close,
                        self.relaxation,
                        self.threads,
                        self.pool.as_deref(),
                        owned_cells,
                    );
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

    /// Threads actually used: the configured count, or every core when 0.
    #[getter]
    fn effective_threads(&self) -> usize {
        if self.threads == 0 {
            rayon::current_num_threads()
        } else {
            self.threads
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Mesher(voxel_resolution={:?}, threads={}, objects={})",
            self.resolution,
            self.effective_threads(),
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
