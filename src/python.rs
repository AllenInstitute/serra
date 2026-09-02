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
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::dice::{dice as dice_mesh, DiceOptions};
use crate::extract::{extract_parallel, ExtractOptions, Extraction};
use crate::grid::{Label, VolumeView};
use crate::mesh::{build, MeshOptions, TriangleMesh};
use crate::orient::Layout;
use crate::simplify::{simplify, SimplifyOptions};
use crate::smooth::{fair, scatter, smooth_with, Fairing, Relaxation, Scratch, Smoothing, Taubin};

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
    smoothing: Smoothing,
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
    smoothing: Smoothing,
    threads: usize,
    pool: Option<&rayon::ThreadPool>,
    owned_cells: Option<[usize; 3]>,
) -> ([usize; 3], Extraction) {
    let view = array.as_array();
    let s = view.shape();
    let shape = [s[0], s[1], s[2]];

    // Smoothing must not move vertices whose one-ring the chunk does not
    // fully contain, so the outermost layer of cells is pinned. With `close`
    // there is nothing beyond that layer to be missing, so nothing is pinned
    // and the whole surface is free to smooth.
    let options = ExtractOptions {
        mark_boundary: smoothing.is_active() && (!close || owned_cells.is_some()),
        owned_cells,
    };

    // Neither step touches a Python object, so the GIL is released (pyo3 0.29
    // spells this `detach`) and other threads can run.
    let work = move || {
        let mut extraction = extract_parallel(&VolumeView::new(view, close), &options, threads);
        if smoothing.is_cell_domain() {
            // One pass over the shared cell field, then every vertex reads its
            // cell. Both copies of a wall become the same number, so they cannot
            // disagree however many iterations run.
            if let Smoothing::Fairing(params) = smoothing {
                fair(&mut extraction.cells, &params, threads != 1);
            }
            let cells = &extraction.cells;
            if threads == 1 {
                for mesh in extraction.meshes.iter_mut() {
                    scatter(cells, mesh);
                }
            } else {
                extraction
                    .meshes
                    .par_iter_mut()
                    .for_each(|mesh| scatter(cells, mesh));
            }
        } else if smoothing.is_active() {
            // One scratch buffer per worker, carried across every mesh that
            // worker handles. Smoothing a chunk means smoothing thousands of
            // surfaces, and allocating the adjacency afresh for each one costs
            // more than the arithmetic does.
            if threads == 1 {
                let mut scratch = Scratch::default();
                for mesh in extraction.meshes.iter_mut() {
                    smooth_with(mesh, &smoothing, &mut scratch);
                }
            } else {
                // Each object's surface is a separate graph, so smoothing is
                // embarrassingly parallel and cannot depend on scheduling: a
                // mesh is only ever touched by the worker that owns it.
                extraction
                    .meshes
                    .par_iter_mut()
                    .for_each_init(Scratch::default, |scratch, mesh| {
                        smooth_with(mesh, &smoothing, scratch)
                    });
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
        taubin=0,
        taubin_pass_band=0.1,
        taubin_lambda=0.63,
        fairing=0,
        fairing_step=0.5,
        fairing_junction_rule=true,
        fairing_taubin=false,
        fairing_pass_band=0.1,
        fairing_lambda=0.63,
        threads=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        voxel_resolution: Vec<f64>,
        axis_order: &str,
        y_down: bool,
        relaxation: u32,
        max_deviation: f64,
        relaxation_step: f64,
        taubin: u32,
        taubin_pass_band: f64,
        taubin_lambda: f64,
        fairing: u32,
        fairing_step: f64,
        fairing_junction_rule: bool,
        fairing_taubin: bool,
        fairing_pass_band: f64,
        fairing_lambda: f64,
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
        // Two filters over the same vertices would be a compounding of bounds
        // nobody could reason about, and the answer to "which one" is a
        // decision, not a blend.
        if [relaxation > 0, taubin > 0, fairing > 0]
            .iter()
            .filter(|&&on| on)
            .count()
            > 1
        {
            return Err(PyValueError::new_err(
                "set only one of relaxation, taubin or fairing",
            ));
        }
        if fairing > 0 && (!fairing_step.is_finite() || fairing_step <= 0.0 || fairing_step > 1.0) {
            return Err(PyValueError::new_err("fairing_step must lie in (0, 1]"));
        }
        let taubin_params = Taubin {
            iterations: taubin,
            pass_band: taubin_pass_band,
            lambda: taubin_lambda,
            max_deviation,
        };
        if taubin > 0 && !taubin_params.is_valid() {
            return Err(PyValueError::new_err(format!(
                "taubin_pass_band and taubin_lambda must lie in (0, 1) and give a \
                 negative mu larger in magnitude than lambda; got pass_band={taubin_pass_band}, \
                 lambda={taubin_lambda}, mu={}",
                taubin_params.mu()
            )));
        }
        let smoothing = if fairing > 0 {
            Smoothing::Fairing(Fairing {
                iterations: fairing,
                step: fairing_step,
                max_deviation,
                junction_rule: fairing_junction_rule,
                pass_band: if fairing_taubin {
                    Some(fairing_pass_band)
                } else {
                    None
                },
                lambda: fairing_lambda,
            })
        } else if taubin > 0 {
            Smoothing::Taubin(taubin_params)
        } else {
            Smoothing::Laplacian(Relaxation {
                iterations: relaxation,
                max_deviation,
                step: relaxation_step,
            })
        };
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
            smoothing,
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
                        self.smoothing,
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
    ///
    /// `reduction_factor` asks for that many times fewer faces;
    /// `max_error` caps how far simplification may move any vertex.
    #[pyo3(signature = (label, normals=false, reduction_factor=0, max_error=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        label: u64,
        normals: bool,
        reduction_factor: u32,
        max_error: Option<f64>,
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
        // Following zmesh: max_error defaults to one voxel of the coarsest axis.
        let max_error =
            max_error.unwrap_or_else(|| self.resolution.iter().cloned().fold(f64::MIN, f64::max));
        if max_error < 0.0 {
            return Err(PyValueError::new_err("max_error must be non-negative"));
        }

        let built = py.detach(move || {
            let mut mesh = build(raw, &options);
            if reduction_factor > 1 {
                let target = mesh.faces.len() / reduction_factor as usize;
                simplify(
                    &mut mesh,
                    &SimplifyOptions {
                        target_faces: target,
                        max_error,
                    },
                );
            }
            mesh
        });
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

/// Cut a mesh into the cells of a regular grid.
///
/// Returns `{(i, j, k): (vertices, faces)}` for the cells that hold anything.
/// Every triangle lands in exactly one cell, wholly inside it, and two cells
/// that meet agree on the geometry between them bit for bit — see
/// [`crate::dice`] for why that needs no tolerance.
#[pyfunction]
#[pyo3(signature = (vertices, faces, chunk_shape, grid_origin, grid_size))]
fn dice<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f32>,
    faces: PyReadonlyArray2<'py, u32>,
    chunk_shape: [f64; 3],
    grid_origin: [f64; 3],
    grid_size: [i64; 3],
) -> PyResult<Bound<'py, PyDict>> {
    let v = vertices.as_array();
    let f = faces.as_array();
    if v.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vertices must be (N, 3), got {:?}",
            v.shape()
        )));
    }
    if f.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "faces must be (M, 3), got {:?}",
            f.shape()
        )));
    }
    for axis in 0..3 {
        if chunk_shape[axis].is_nan() || chunk_shape[axis] <= 0.0 {
            return Err(PyValueError::new_err(
                "chunk_shape must be positive on every axis",
            ));
        }
        if grid_size[axis] < 1 {
            return Err(PyValueError::new_err(
                "grid_size must be at least 1 on every axis",
            ));
        }
    }

    let vertex_list: Vec<[f32; 3]> = (0..v.shape()[0])
        .map(|i| [v[[i, 0]], v[[i, 1]], v[[i, 2]]])
        .collect();
    let nv = vertex_list.len() as u32;
    let mut face_list = Vec::with_capacity(f.shape()[0]);
    for i in 0..f.shape()[0] {
        let face = [f[[i, 0]], f[[i, 1]], f[[i, 2]]];
        if face.iter().any(|&idx| idx >= nv) {
            return Err(PyValueError::new_err(format!(
                "face {i} references a vertex past the end of the array"
            )));
        }
        face_list.push(face);
    }
    let mesh = TriangleMesh {
        vertices: vertex_list,
        faces: face_list,
        normals: None,
        pinned: Vec::new(),
    };

    let options = DiceOptions {
        chunk_shape,
        grid_origin,
        grid_size,
    };
    let cells = py.detach(move || dice_mesh(&mesh, &options));

    let out = PyDict::new(py);
    for cell in cells {
        let arrays = to_arrays(
            py,
            TriangleMesh {
                vertices: cell.vertices,
                faces: cell.faces,
                normals: None,
                pinned: Vec::new(),
            },
        );
        out.set_item(
            (cell.index[0], cell.index[1], cell.index[2]),
            (arrays.0, arrays.1),
        )?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mesher>()?;
    m.add_function(wrap_pyfunction!(dice, m)?)?;
    Ok(())
}
