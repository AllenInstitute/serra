# GAMer, and why serra's smoothing is in the wrong domain

Two papers were put to serra as a possible source of technique:

> Yu, Z., Holst, M. J., Cheng, Y., & McCammon, J. A. (2008). *Feature-preserving
> adaptive mesh generation for molecular shape modeling and simulation*. Journal
> of Molecular Graphics and Modelling, 26(8), 1370–1380.

> Lee, C. T., Laughlin, J. G., Angliviel de La Beaumelle, N., Amaro, R. E.,
> McCammon, J. A., Ramamoorthi, R., Holst, M., & Rangamani, P. (2020). *3D mesh
> processing using GAMer 2 to enable reaction-diffusion simulations in realistic
> cellular geometries*. PLoS Computational Biology, 16(4), e1007756.

They were not considered when serra was designed; it follows Frisken (2022)
multi-label SurfaceNets. GAMer is a different family — a mesh *post-processing*
toolchain whose contribution begins after iso-contouring, not an extractor.

!!! note "Two projects share the name"

    `github.com/gamer-project/gamer` is a GPU-accelerated adaptive mesh
    refinement code for **astrophysics**. The mesh-conditioning GAMer described
    above is `github.com/ctlee/gamer` (PyPI `pygamer`).

Everything below is measured with `bench/cell_smoothing.py` on the committed
MICrONS neuropil fixture, unless a source is given.

## Why a naive multi-material port would be slow — and worse than slow

**GAMer's front end is single-material by construction.** GAMer 1 blurs atoms
into a scalar field and extracts one isosurface at one isovalue; GAMer 2 takes
surface meshes from contour tiling. Neither ingests a label volume. For *N*
labels you run it *N* times.

The obvious cost is the traversal. Even with the standard optimisation of
cropping each object to its bounding box, on the neuropil fixture — 20,840
objects, median 56 voxels — the bounding boxes sum to 33.5 G voxels against
134 M in the array. **A per-label pipeline touches 250× the volume**, because
neuropil objects are long thin processes whose boxes overlap enormously. serra's
single pass is the whole reason it is viable at 10⁵–10⁶ objects per chunk.

The deeper objection is not speed. In dense neuropil **88.6% of distinct vertex
positions belong to more than one label**: most surface is *interface*, not
exterior boundary. 76.3% of boundary cells hold exactly two labels, 13.7% hold
three or more. A mesh-domain algorithm whose neighbourhood is one label's own
surface therefore processes the same wall twice, from two different
neighbourhoods, and the two copies drift apart.

## serra already has this bug

serra's own relaxation runs per label. Measured on a 128³ neuropil subvolume
with one shared Jacobi operator, six iterations, `max_deviation = 0.5`:

| smoothing | area / raw | mean dihedral | angles in [40°,80°] | volume / voxel truth | wall drift, median | max |
| --- | --- | --- | --- | --- | --- | --- |
| none | 100.0% | 25.1° | 67.8% | 97.99% | 0 | 0 |
| **per label** (serra today) | 84.3% | 16.9° | 86.5% | 84.00% | 0.103 vx | **2.224 vx** |
| cell domain | 93.4% | 21.3° | 87.5% | 93.92% | **0** | **0** |
| cell domain + LST | 97.3% | 23.4° | 87.7% | 97.12% | **0** | **0** |

Two labels' copies of the same wall end up as much as **2.2 voxels — 71 nm —
apart**. After smoothing, the segmentation is no longer a partition of space:
adjacent objects gap or interpenetrate.

The cell-domain rows are also, at identical operator and iteration count,
markedly *less destructive* — 93.9% of the true volume against 84.0%. A shared
node is pulled by neighbours drawn from every label meeting there, and those
neighbours agree with each other; a per-label copy is pulled only by its own
label's surface, which curves away from the wall.

## The fix: smooth in the cell domain

serra places one vertex per cell as a pure function of that cell's eight corner
labels (`cell_vertex`, `src/place.rs`), so **every label present in a cell
already receives the identical position** — and then serra copies it into each
label's mesh and smooths the copies independently. Smoothing one shared position
per cell instead:

- makes drift **exactly zero** by construction, at any iteration count;
- costs **2.04×** less, there being 2.06 label-copies per boundary cell;
- makes a per-cell structure tensor affordable, one tensor serving every label.

This is Frisken's own formulation, which serra departed from. It is not a new
idea; it is a gap being closed.

**Wrinkle, measured:** the extractor splits a cell's vertex per surface *sheet*
where a label touches itself only diagonally, and all sheets share the cell
position, so welding on position alone would fuse sheets that are deliberately
separate. That is 0.237% of (cell, label) pairs, affecting 0.483% of cells. The
prototype gives them private nodes and reports the count.

## What ports from GAMer, and what does not

**The quality-improvement stage does not port, because serra does not have the
problem it solves.** GAMer's angle-based smoothing and vertex-removal decimation
exist to repair marching cubes run on a *smooth scalar field*, where the
isosurface crosses a cube edge at an arbitrary interpolated position and two
vertices can land arbitrarily close together. The 2008 paper's input meshes have
a minimum angle of **0.02°** and a maximum of 179.1°. On label data there is no
interpolation, and neither mesher produces slivers:

| | min angle | max angle | in [40°,80°] | below 10° |
| --- | --- | --- | --- | --- |
| zmesh (marching cubes) | 26.2° | 127.6° | 43.1% | 0.00% |
| serra | 13.7° | 131.1° | 67.2% | 0.00% |
| serra, `relaxation=3` | 14.2° | 140.3° | 73.0% | 0.00% |
| GAMer's MC input (published) | **0.02°** | 179.1° | ~50% | — |
| after GAMer's pipeline (published) | 20.2° | 130.7° | 85–90% | — |

serra starts where GAMer finishes and has no degenerate triangles to repair.
There is headroom to 85–90%, but it is polish, not repair — and GAMer's figure
comes from three mechanisms together (an angle-based target, the structure
tensor, and edge flips), only one of which is discussed below.

**The local structure tensor does port**, with its purpose corrected.

## What the local structure tensor actually does

T = Σ n nᵀ over the incident wall normals; damp the smoothing step by
1/(1+γλₖ) along each eigenvector. It is tempting to read this as an anisotropic
replacement for serra's isotropic `max_deviation` box. That reading is wrong.

On a locally flat patch every incident normal is the same **n**, so T is rank one
with its large eigenvalue along the *surface normal*. Damping therefore
suppresses motion **off** the surface and leaves motion **along** it free. On a
crease the normals span two directions and only motion along the crease line
survives; at a corner nothing moves. It is a *slide along the surface, do not
move off it* constraint. It bounds normal displacement and bounds tangential
displacement not at all — so it does not replace `max_deviation`, which bounds
total displacement from the data. The two are orthogonal.

Two properties make it fit serra specifically:

- **n nᵀ is sign-invariant**, so the winding flip between the two sides of a
  shared wall is irrelevant and one tensor serves every label in the cell.
- **No eigendecomposition is needed.** Because the eigenvectors are orthonormal
  and T = Σ λₖ eₖ eₖᵀ, the damped update is identically
  **x̂ = x + (I + γT)⁻¹ (x̄ − x)**. T is a sum of outer products, so I + γT is
  symmetric positive definite with eigenvalues ≥ 1: always invertible, well
  conditioned, and solvable in closed form. `np.linalg.eigh` costs 784 ns per
  3×3; the closed-form solve is roughly 30× cheaper.

One correction worth recording, because it is easy to get wrong: a wall between
A and B is emitted into *both* meshes. Since n nᵀ is sign-invariant, gathering
faces per label does not cancel that duplication — it **doubles** it, weighting
interior walls twice against walls facing background. The tensor must be
assembled over each wall once, in the cell domain.

### Measured, at matched displacement

Comparing two smoothers at equal parameters is meaningless when they move
vertices by different amounts: the one that moves less always looks like it
preserves more. Both arms below are the same Jacobi Laplacian, swept over
iteration count and read off at a matched mean displacement.

| fixture | mean move | constraint | volume / true | area / raw | dihedral |
| --- | --- | --- | --- | --- | --- |
| **sphere r=20** (control) | 0.186 vx | isotropic | 98.76% | 96.7% | **1.8°** |
| | | LST | 99.48% | 97.8% | 4.5° |
| **cylinder r=12** | 0.219 vx | isotropic | 96.86% | 93.2% | 3.1° |
| | | LST | **98.56%** | 96.2% | 4.1° |
| **box 24³** | 0.195 vx | isotropic | 96.45% | 93.4% | 2.6° |
| | | LST | **98.49%** | 97.0% | 2.5° |

Read the dihedral column alongside the volume. On the **sphere** the LST loses
less volume but also smooths far less — 4.5° against 1.8° — and at *matched
smoothing* the isotropic bound is marginally better. That is the correct control
result: a sphere's structure tensor is near-isotropic, so there is no anisotropy
to exploit and the LST must not win there.

On the **box** both reach the same dihedral, 2.5°, and the LST loses 1.0% of
volume where the isotropic bound loses 3.0%. The box's exact bevel loss is
3n − 2 = 70 voxels, 0.51%, so isotropic smoothing is destroying the edges and
corners at six times that rate while the LST largely holds them. That is the
mechanism working as advertised, on the fixture where it should.

## The finding that limits all of this

Per-label QEM simplification **already destroys wall coincidence**, before any
of the above is considered. Same subvolume, `relaxation=0`, measuring the share
of distinct vertex positions belonging to more than one label:

| | shared positions |
| --- | --- |
| no simplification | **88.5%** |
| `reduction_factor=2` | 43.9% |
| `reduction_factor=10` | 16.8% |

PyChunkedGraph's served path calls `get(id, reduction_factor=100, max_error=40)`.
So for the meshes the pipeline actually stores, the coincident-wall guarantee is
gone at simplification regardless of what smoothing does. Cell-domain smoothing
is worth having for the LOD-0 and analysis paths — anywhere volumes or contact
areas are measured off the meshes — but it does not by itself make the *served*
meshes a partition. Doing that would need the simplifier made cell-aware too,
which is a substantially larger change.

## Where that leaves it

- The hypothesis is right: a naive multi-material port is slow (250×) and, more
  importantly, structurally wrong for data that is 88.6% shared interface.
- The clever way is to move smoothing from the label domain to the cell domain.
  It is cheaper, it is less destructive, and it makes drift exactly zero.
- The structure tensor is a genuine improvement on sharp features, is far
  cheaper than it first appears, and is a *tangential* constraint that
  complements `max_deviation` rather than replacing it.
- None of it fixes the served path on its own, because simplification breaks
  coincidence first.

Reproduce with:

```bash
uv sync --group bench
python bench/cell_smoothing.py
```

The reference implementation could not be used for a head-to-head: `pygamer`
2.0.7 (Aug 2021) ships source-only, and its vendored pybind11 predates CPython
3.11's opaque `PyFrameObject`, so it does not build against a current
interpreter. The published figures above are quoted from the papers.
