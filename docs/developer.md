# Developer guide

## Building from source

Needs a Rust toolchain, 1.74 or newer ([rustup](https://rustup.rs)). Then:

```bash
pip install .                   # builds release and installs
```

To work on the code, an editable build that you re-run after any Rust change:

```bash
uv sync
uv run maturin develop --release
```

**Always pass `--release`.** `maturin develop` defaults to debug, which is ~24×
slower: 0.147 s to mesh a 128³ volume against 0.006 s.

To produce a wheel:

```bash
maturin build --release         # target/wheels/*.whl
```

`abi3-py39` means one wheel per platform covers CPython 3.9 and up.

## Layout

The crate is a pipeline, one module per stage. Data flows top to bottom.

| module | responsibility |
| --- | --- |
| `src/orient.rs` | Axis order and handedness, as index arithmetic |
| `src/grid.rs` | Zero-copy view over the caller's array |
| `src/tables.rs` | Compile-time cube tables |
| `src/place.rs` | Where a cell's vertex goes, plus relaxation |
| `src/extract.rs` | The single pass, and its parallel banding |
| `src/mesh.rs` | Physical coordinates, triangulation, normals |
| `src/python.rs` | PyO3 bindings — the only module that knows about Python |
| `python/serra_mesh/` | `Mesher` front end, `Mesh` container, file formats |

Two rules keep this tidy. Orientation is applied **once, on output**, so the hot
loop never thinks about axis order. And Python appears in exactly one module, so
the core is testable as plain Rust (`cargo test --no-default-features`).

## The algorithm

A **cell** is a 2×2×2 block of voxels. A cell whose eight corners are not all the
same label contributes a vertex. A **quad** is dual to a voxel edge whose two
voxels differ, and is built from the four cells surrounding that edge.

### One traversal is enough

The four cells around the edge leaving voxel `p` along axis `a` have minimum
corners at `p − du·e_u − dv·e_v` for `du, dv ∈ {0,1}`. None exceeds `p`, so by
the time the loop reaches `p` every cell it needs already exists. Only two cell
layers stay live.

On a real 512³ volume with 2524 labels, 16.7% of cells are on a boundary and
84.8% of those carry exactly two labels, so the early-out on uniform cells does
most of the work. Cost tracks boundary area, not object count.

### Vertex placement

`tables::CENTROID` maps the 12-bit pattern of which cube edges cross a label
boundary to a position, in **1/256-voxel fixed point**. Integer positions are
what make two chunks agree bit-for-bit on a shared seam: there is no
floating-point association order to get wrong.

### Manifoldness

An object's dual surface is the boundary of its voxel set. Non-manifold edges
and vertices appear exactly where that set touches itself only diagonally.
`tables::TABLES.split` partitions a cell's corners into 6-connected components,
and the cell emits **one vertex per component**, which separates those sheets.
At most four components, hit by the two checkerboard masks.

Recording per cell *which vertex each corner belongs to* makes the quad step a
direct lookup — the corner index of `p` inside each neighbour is known from
`(du, dv, a)` — instead of re-deriving component ids for four cells per quad.

### Triangulating a quad

The shorter diagonal wins. The subtlety: a wall between two labels is emitted
twice, once per label with the ring reversed, and reversing swaps which index
pair *names* each diagonal. The rule is therefore stated over unordered point
pairs, so both copies pick the same diagonal and stay exactly coincident.

## Parallelism

The volume is split into bands of cell layers along axis 2, one per worker, each
with a private label table.

A band cannot emit its own first layer's quads, since those read the layer below
which belongs to the previous band. Rather than duplicating that layer, each
band skips them and a serial pass produces them afterwards from the two
neighbouring bands' exported layers.

!!! danger "Ordering, not just geometry"
    That seam pass must splice its quads **between** the bands they sit between,
    not append them at the end. Appending produces geometrically correct meshes
    whose face order depends on the band count — so output would differ between
    machines with different core counts. This was a real bug during development;
    `tests/test_determinism.py` compares fingerprints across eight thread counts
    to catch a regression.

## Testing

```bash
cargo test --no-default-features        # tables, grid, orientation, relaxation
uv run maturin develop --release
uv run pytest tests -q                  # everything else
```

| file | covers |
| --- | --- |
| `tests/test_api.py` | Binding surface, dtypes, serialization |
| `tests/test_analytic_geometry.py` | Volume, area, normals against exact shapes |
| `tests/test_manifold.py` | Self-contact, multi-label junctions, banding |
| `tests/test_seams.py` | The chunked-meshing contract |
| `tests/test_determinism.py` | Order, dtype, threads, strides |
| `tests/test_relaxation.py` | Smoothing quality and its deviation bound |

Helpers live in `tests/conftest.py`, including a proper non-manifold **vertex**
check. An edge-only check is not enough: two blocks meeting at a single corner
use every edge exactly twice, yet the shared vertex is a pinch point. The link
of each vertex must be a single cycle.

!!! note "Axis conventions in tests"
    Masks are built with `z, y, x = np.ogrid[...]`, so `x` varies along **array
    axis 2**. Anything not axis-symmetric — an ellipsoid's semi-axes, an analytic
    normal — must respect that. Getting it wrong once made a correct mesh look
    like it had a 52° normal error.

## Benchmarks

```bash
uv sync --group bench                        # pyvista/VTK are not installed by default
python bench/compare_zmesh.py serra          # one implementation per process
python bench/compare_zmesh.py zmesh
python bench/render_comparison.py --zmesh ../zmesh --out docs/images
```

Run each backend in its own process so peak RSS reflects only that
implementation.
