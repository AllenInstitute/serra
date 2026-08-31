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
