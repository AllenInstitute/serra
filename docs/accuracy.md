# Accuracy and smoothing

Every figure here is checked by the test suite, and the tolerances there are set
from these measurements rather than from round numbers.

## Analytic shapes

Measured against exact volume and area, isotropic voxels, no relaxation:

| shape | volume | area | mean normal error |
| --- | --- | --- | --- |
| sphere, r = 8…32 | < 0.11% | +2.8…3.0% | 8.1° |
| ellipsoid 30/20/12 | +0.17% | +3.3% | 8.6° |
| cylinder, any axis | −1.48% | −0.47% | |
| torus R25 r8 | +0.24% | +2.9% | |
| ellipsoid on 4×4×40 nm voxels | −0.21% | | |

Every one comes out closed, 2-manifold and correctly oriented, with the right
Euler characteristic — including χ = 0 for the torus, so genus survives meshing.

## Two systematic biases

**Area is overstated by about 3%** on smooth surfaces and does not shrink with
resolution. Placing a cell's vertex at the centroid of its edge crossings leaves
the surface slightly faceted. Marching cubes has the same non-convergence at
roughly +9%.

**Sharp convex edges are bevelled.** An n³ box loses exactly `3n − 2` of volume:
an edge effect, so the relative error falls as 1/n². Marching cubes has the
opposite bias — exact on axis-aligned boxes, 9% over on spheres. For the rounded
shapes serra targets this does not bite; a sphere lands within 0.06%.

## Relaxation

`relaxation=k` runs `k` iterations of constrained smoothing. On a sphere of
radius 20:

| k | volume | area | mean normal | 95th pct | triangle-area CV |
| --- | --- | --- | --- | --- | --- |
| 0 | −0.11% | +2.95% | 8.1° | 17.5° | 0.247 |
| 3 | −0.52% | **+0.24%** | 4.0° | 9.3° | 0.224 |
| 5 | −0.78% | −0.25% | 2.9° | 6.8° | 0.220 |
| 10 | −1.43% | −0.87% | 1.8° | 4.2° | 0.216 |
| 25 | −3.39% | −2.24% | **1.0°** | 2.0° | 0.212 |

Normal accuracy converges with iterations, which it does not under local
placement alone. `k = 3` is a good default when you want it: area error drops
by an order of magnitude for a modest cost.

### Bounding the deviation

Laplacian smoothing shrinks volume. `max_deviation` (in voxels, default 0.5)
caps how far any vertex may move from where the data put it, enforced per axis
against the *original* position rather than per step. Tightening it monotonically
reduces shrinkage:

| max_deviation | volume error at k = 25 |
| --- | --- |
| 0.1 | −1.09% |
| 0.25 | −2.99% |
| 0.5 | −6.42% |

Relaxation preserves topology: genus survives, and one-voxel-thick sheets do not
collapse.

!!! warning "Objects that touch each other"
    Relaxation currently runs per object. Two objects sharing a wall hold
    separate copies of it and smooth against different neighbourhoods near the
    wall's rim, so the wall may no longer be exactly coincident from both sides.
    Each object stays individually watertight and manifold. At `relaxation=0`
    shared walls are exactly coincident, which the test suite checks.


## Validation against zmesh on real data

`bench/validate_winding.py` performs the check described in `CLAUDE.md`: classify
sample points as inside or outside each mesh with a robust generalized winding
number, confirm the volumes agree, and confirm that where the two disagree the
points lie *close to the surface*. That last part is what distinguishes a
genuine difference in where the surface was placed from an actual defect — a
hole or an inverted patch produces disagreements scattered through the volume.

The fixture is `data/microns_neuropil.npy.gz`: a 512³ cutout at 32×32×40 nm from
`gs://iarpa_microns/minnie/minnie65/seg_m1300`, taken from dense neuropil near
the centre of the imaged column. It holds **20,840 objects with a median size of
56 voxels**, and no single object exceeds 4.1% of the volume — deliberately not a
cell body, since serra is built for volumes with very many small objects.

Over 24 objects spanning the size range, 20,000 sample points each:

| | value |
| --- | --- |
| serra volume / true voxel volume | **0.982** (0.918–1.005) |
| zmesh volume / true voxel volume | 0.968 (0.918–0.991) |
| serra / zmesh volume | **1.015** (1.000–1.031) |
| winding-number agreement | 94.7% of sampled points |
| disagreeing points, distance to surface | median **0.47** voxels, worst **2.11** |
| all sampled points, distance to surface | median 4.98 voxels |

Two things to read from this.

**serra is closer to the voxel truth than zmesh** — 0.982 against 0.968 — and
both undershoot, which is expected: a surface drawn through the voxel boundary
cuts the corners off a blocky object.

**Every disagreement is a boundary effect.** Disagreeing points sit 11× closer to
the surface than a typical sampled point, and none is further than 2.11 voxels
from it. The two meshers place the surface slightly differently within about a
voxel, and agree everywhere else. Nothing is scattered through the interior,
which is what a hole or an inverted normal would produce.

The raw agreement rate of 94.7% is not a defect either: most of these objects are
tiny, so uniform sampling in their bounding box puts a large share of points
within a voxel of the surface. Agreement is 99%+ on the larger objects and falls
on the smallest purely through the surface-to-volume ratio.

### Reproducing

```bash
uv sync --group bench
python bench/download_microns.py            # re-fetch the cutout
python bench/download_microns.py --survey   # re-run the region search
python bench/validate_winding.py
```

## Two smoothing filters

`relaxation=k` is Frisken's constrained Laplacian fairing: each pass moves a
vertex a fraction of the way to the average of its neighbours, bounded by
`max_deviation`, with seam vertices pinned. It is diffusion, and diffusion
shrinks.

`taubin=k` alternates a positive step `lambda` with a larger negative one, `mu`,
derived from `taubin_pass_band`. The pair is a low-pass filter rather than a
diffusion: the low graph frequencies that carry an object's bulk come through at
close to unit gain, so the surface can be smoothed hard without the volume
draining away. Each iteration costs two passes instead of one. Everything else
is shared — the same adjacency, the same pinning, the same `max_deviation`
bound.

Both run inside `mesh()`, so a mesh is always smoothed **before** `get()`
simplifies it, which is the order that measures better (see below).

### On an analytic sphere

Radius 32 voxels, isotropic. Normal error is the mean angle between each face
normal and the true surface normal.

| | area error | volume error | normal error |
| --- | --- | --- | --- |
| no smoothing | +2.80% | −0.23% | 12.52° |
| `relaxation=3` | +0.28% | −0.39% | 5.16° |
| `relaxation=10` | −0.42% | −0.75% | 1.76° |
| `relaxation=20` | −0.82% | −1.27% | 0.97° |
| `taubin=3` | +1.29% | −0.22% | 8.57° |
| `taubin=10` | +0.47% | −0.19% | 5.19° |
| `taubin=20` | +0.25% | −0.14% | 3.80° |
| `taubin=40` | +0.19% | −0.05% | 2.94° |

Read the columns together. Relaxation is far more efficient at buying normals —
3 passes reach what Taubin needs 20 for — but it pays in volume, and past
`relaxation=10` the area error has gone *negative*: the sphere is now smaller
than the data says. Taubin's volume error moves the other way, towards zero.

### On real neuropil, which is what settles it

A sphere has almost no surface to lose. A 200 nm spine neck does. 24 objects
between the 70th and 99th size percentile in the MICrONS cutout, closed ones
only, measured against the voxel count:

| | volume / true | area / unsmoothed |
| --- | --- | --- |
| no smoothing | 99.98% | 100.0% |
| `relaxation=3` | 97.51% | 89.4% |
| `relaxation=10` | 93.06% | 83.2% |
| `taubin=3` | **100.12%** | 95.5% |
| `taubin=10` | **100.44%** | 93.0% |
| `taubin=20` | **100.93%** | 92.2% |

Laplacian relaxation eats 2.5% of the volume at `k=3` and 7% at `k=10`. Taubin
holds it. **Prefer `taubin` when the meshes will be measured**, and `relaxation`
when only appearance matters and the budget is tight.

### What it costs

512³ MICrONS volume, 2523 objects, Apple M4 Pro (14 cores), median of three
runs. `get()` covers extracting every object.

| | `mesh()` | `get()` | total | 1 thread |
| --- | --- | --- | --- | --- |
| no smoothing | 0.28 s | 0.98 s | **1.26 s** | 1.62 s |
| `relaxation=3` | 0.48 s | 0.92 s | 1.40 s | 3.21 s |
| `taubin=2` | 0.51 s | 0.94 s | 1.45 s | 3.49 s |
| `taubin=3` | 0.57 s | 0.94 s | 1.51 s | 4.03 s |
| `taubin=5` | 0.68 s | 0.96 s | 1.64 s | 4.88 s |

Smoothing is added work and cannot be free. What it is, is **no more expensive
than the relaxation pass that was already there**: `taubin=2` costs about what
`relaxation=3` does, and the whole smoothing stage is roughly a third cheaper
than it was before this work, because the adjacency structure is now built
without a global compaction pass and the position buffers are `f32`.

Cost splits about evenly between building the vertex adjacency (once per object,
whatever the iteration count) and the passes themselves, so the *first*
iteration is much more expensive than the tenth.

### Order against simplification

Smooth first. At a 10× reduction, with VTK quadric decimation on both paths so
the order is the only variable:

| order | mean distance to full-res | worst | volume kept | roughness |
| --- | --- | --- | --- | --- |
| simplify only | 0.100 vx | 0.70 | 97.8% | 33.6° |
| Taubin then simplify | 0.103 vx | 0.48 | 98.2% | 26.3° |
| simplify then Taubin | 0.325 vx | 2.71 | 88.6% | 22.8° |

Smoothing first costs essentially nothing in fidelity and improves the worst
case; smoothing afterwards deviates 3× further and loses 11% of the volume,
because a decimated mesh has no high-frequency detail left to remove and the
filter eats structure instead. serra gets this order for free: smoothing happens
in `mesh()`, simplification in `get()`.

### Smoothing a chunked volume

Both filters pin the outermost layer of cells, so seam vertices stay
bit-identical between neighbouring chunks and stitching by exact vertex equality
still works. This is *not* true of a post-hoc smoother applied to the output.
Measured with VTK's windowed-sinc filter on a sphere across 3×3×3 chunks, all
2,628 seam vertices move whatever the window:

| window | boundary smoothing | max seam displacement | stitches? |
| --- | --- | --- | --- |
| Blackman | on | 1.7 × 10⁻¹ voxel | no |
| Blackman | off | 1.7 × 10⁻³ voxel | no |
| Hamming | off | 3.9 × 10⁻² voxel | no |
| Nuttall | off | 4.9 × 10⁻⁵ voxel | no |

Blackman preserves borders far better than the default, but "far better" is not
"exactly", and exact is what welding needs. If you must smooth after the fact,
snap every seam vertex back to the extractor's position afterwards: stitching
then succeeds, and the assembled surface lands a median 0.003 voxels from where
smoothing the whole volume in one piece would have put it.

The trade-off `serra` accepts by pinning instead: a chunk's interior smooths
slightly more than the band around its seams, so a stitched surface is
self-consistent and watertight but not identical to the same volume smoothed in
one piece.

### Reproducing

```bash
uv sync --group bench
python bench/taubin.py
```
