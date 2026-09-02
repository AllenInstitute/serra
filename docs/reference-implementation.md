# Against other SurfaceNets implementations

serra cites Frisken (2022) as the method it implements, and the paper's author
publishes a C++ implementation alongside it:
[surfaceNets-multi-material](https://github.com/sarahfrisken/surfaceNets-multi-material).
This page is what happened when we built it and ran it on the same 512³
connectomics volume the [performance table](../#performance) uses.

The short version: **serra is about ten times faster than the reference as
published, but almost all of that gap is a memory-layout accident rather than
anything algorithmic.** Fix the layout in three lines and the reference's
smoothing loop is marginally *faster* than serra's on one thread. Where serra
holds a durable advantage is elsewhere — threading, extraction, initial vertex
placement, and shrink compensation.

## Building it on macOS

The repo ships `SurfaceNets.sln`, which is a red herring. Only
`Source/Application` needs Qt and OpenGL. `Source/SNLib` — the whole library,
1,613 lines across five `.cpp` files — includes nothing but `<vector> <map>
<set> <array> <string> <algorithm> <cstdlib> <exception> <type_traits>
<time.h>`. It is portable ISO C++ and `clang++ -std=c++17` compiles all five
files with zero errors and two cosmetic brace warnings.

So there is no port. There is a compile command:

```bash
make -C bench/frisken stock       # or: make FRISKEN=/path/to/clone stock
```

`bench/frisken/sn_bench.cpp` is the missing `main()`: it reads a raw `uint16`
volume, times construction, relaxation and extraction, and prints one JSON line
with the same keys as the other backends, so it drops into the existing harness:

```bash
python bench/compare_zmesh.py frisken --relax 20
```

The clone is never modified. `make patched` copies the sources into
`build/patched/` and applies `bench/frisken/patches/*.patch` to the copy.

!!! warning "Two coordinate conventions to convert between"
    `MMCellMap` walks `i` fastest, so its `arraySize[0]` is numpy's **axis 2** —
    the mirror of serra's convention, where `voxel_resolution[i]` scales array
    axis `i`. And it pads the volume by one voxel on every face and reports
    positions in the *padded* index space, so raw output sits exactly **+1 voxel**
    from the segmentation it came from on every axis. `sn_bench` corrects both,
    and reverses triangle winding to undo the reflection. On a cubic volume the
    axis swap is invisible, which is what makes it dangerous.

    With both corrections the unsmoothed mesh of a test sphere encloses
    **8255.0 units of volume for exactly 8255 voxels**, and is watertight. That
    equality is the check that the driver is driving the library correctly.

## Performance

512³ connectomics volume, 2524 objects, Apple M4 Pro (14 cores), everything
built `-O3 -DNDEBUG`, one process per backend so peak RSS is never contaminated.

| | serra (1 thread) | serra (14 threads) | zmesh | reference (stock) | reference (patched) |
| --- | --- | --- | --- | --- | --- |
| traverse the volume | 1.86 s | **0.47 s** | 0.89 s | 16.04 s | 1.94 s |
| smooth, 20 sweeps | 10.49 s | — | n/a | 112.89 s | **8.61 s** |
| smooth, 20 sweeps + Taubin | 20.68 s | **2.24 s** | n/a | not implemented | not implemented |
| extract every object | **0.99 s** | 0.99 s | 5.45 s | 92.71 s | 18.66 s |
| peak RSS | **2.96 GB** | 4.00 GB | 3.53 GB | 13.99 GB | 5.51 GB |

The reference's extraction figure is the sum of three phases it keeps separate:
`labels()` (5.45 s stock, 1.26 s patched), building the quad list (5.26 s /
1.46 s), and the `objData()` loop (82.0 s / 15.94 s).

Vertex counts are not directly comparable — the reference emits 50.5M vertices
to serra's 44.8M — because it treats label 0 as an ordinary material and meshes
the background along with everything else.

### The 88-byte cell flag is the whole story

`MMCellFlag` stores 19 bit masks as **non-static** `const unsigned int` data
members, and declares a `virtual` destructor. Both are invisible at the call
site and neither carries information:

```cpp
const unsigned int m_leftFaceCrossingBits = (1 << LeftFaceShift) | ...;
// ... 18 more, replicated in every instance, plus a vptr
```

`sizeof(MMCellFlag)` is **88 bytes**, of which 84 are the same constants in
every cell. That pushes `sizeof(Cell)` to 112 B, and `MMCellMap` allocates
`new Cell[(nx+2)(ny+2)(nz+2)]` — dense over the whole padded volume, boundary
or not. At 512³ that is 135.8M cells and **15.2 GB**.

Making the masks `static constexpr` and dropping `virtual` — patch 01, a
three-line change — takes `Cell` from **112 bytes to 24**, and the OBJ output
stays byte-identical. Patch 02 indexes quads by label so `objData()` stops
rescanning the global quad list once per object. Together:

| | stock | patched | speedup |
| --- | --- | --- | --- |
| construct | 16.04 s | 1.94 s | 8.3× |
| relax ×20 | 112.89 s | 8.61 s | **13.1×** |
| extract all | 92.71 s | 18.66 s | 5.0× |
| peak RSS | 13.99 GB | 5.51 GB | 2.5× |

Nothing about the algorithm changed. A 4.7× smaller working set is worth 13× on
the relaxation loop because relaxation is pure pointer-chasing over the cell
array, and at 112 B/cell almost every neighbour lookup is a cache miss.

### So how fast is the algorithm, really?

Per smoothing sweep, single-threaded, same 6-face-neighbour stencil:

| | per sweep |
| --- | --- |
| reference, stock | 5.64 s |
| serra, `fairing` | 0.525 s |
| reference, patched | **0.431 s** |

The patched reference is **1.2× faster per sweep than serra**. That is the
honest result, and the likely reason is a deliberate serra tradeoff: the
reference relaxes **Gauss-Seidel**, updating `vertexOffset` in place and reading
neighbours' already-updated values, which needs no second buffer. serra runs
**Jacobi** so that output is byte-identical regardless of how the volume is
split across threads. Determinism costs a buffer's worth of memory traffic.

serra buys that back and more by threading — the reference's in-place update is
order-dependent and cannot be parallelised as written, while serra does 20
Taubin passes (40 sweeps) in 2.24 s on 14 cores.

!!! note "Sweeps are not iterations"
    Gauss-Seidel propagates information across the whole array within a single
    sweep, so it converges in roughly half as many sweeps as Jacobi. Equal sweep
    counts are equal *work*, not equal *smoothing*. Any accuracy comparison has
    to match convergence, not the iteration counter.

## Two behavioural differences that matter

**The reference places every vertex at the cell centre.** `initCell` sets
`vertexOffset` to `(0.5, 0.5, 0.5)` and nothing else ever computes an
edge-crossing centroid; all surface shape comes from `relax()`. serra places at
the centroid of the crossings up front, which is why serra's *unsmoothed* mesh
already scores 0.109 voxels of mean error on the
[analytic tube](accuracy.md#judged-against-an-analytic-solid) against zmesh's
0.141. Comparing the two "with smoothing off" compares different algorithms —
the reference's unrelaxed output is exactly the blocky voxel surface, which is
what the 8255.0-for-8255-voxels check above demonstrates.

It also means the construct-phase timings above are not measuring equal work:
serra computes crossing centroids during traversal and the reference does not.

**The reference has no shrink compensation.** `relax()` is a pure Laplacian, so
it loses volume monotonically with sweep count — the effect
[measured here](accuracy.md#two-smoothing-filters) at 12–15% of a radius-4 tube
and 41% of a radius-2 one at 20 sweeps. serra's `fairing_taubin=True` alternates
λ and μ steps to cancel it. On thin structures, which is the connectomics
operating regime, this matters more than any timing on this page.

## What the reference got right, and serra now does too

`relax()` skips faces whose `faceCrossingType` is `NoFaceCrossing`. serra's
cell-domain fairing excludes uniform faces for the same reason, derived
independently from the observation that the literal 6-face stencil collapses a
one-voxel sheet to zero thickness. The reference agreeing is a useful
confirmation that the exclusion is part of the method and not a workaround.

The `EdgeVertex` junction rule — averaging only over `JunctionFace` neighbours
so junction vertices slide along the junction curve — is likewise present in
both, and serra's `fairing_junction_rule` is the same rule.

## Against VTK's vtkSurfaceNets3D

VTK ships its own SurfaceNets, `vtkSurfaceNets3D`, from the same Frisken
lineage, reachable from Python as
[`ImageData.contour_labels`](https://docs.pyvista.org/api/core/_autosummary/pyvista.imagedatafilters.contour_labels).
It is a far closer comparison than zmesh: one pass over the volume, every
material at once, dual vertices inside cells.

Two settings decide whether the comparison means anything. pyvista's
`boundary_style` defaults to `"external"`, which returns only the outside of
the foreground and skips every wall between adjacent objects — on this volume
that is a third of the faces. And `smoothing` defaults **on** at 16 iterations.
Both are set explicitly below.

### The two agree exactly

With serra's `close=True` matching VTK's `pad_background=True`, the two produce
the same mesh on the 512³ neuropil volume:

| | objects | vertices | faces |
| --- | --- | --- | --- |
| serra, `close=True` | 2523 | 46.5M | 93.1M |
| VTK, `boundary_style="all"` | 2523 | 46.5M | 93.1M |

Not close — equal, and equal again at 128³ (156 objects, 1,831,136 faces each).
On a single sphere both give 3038 vertices and 6072 triangles enclosing exactly
8255.0 units of volume for 8255 voxels. Frisken's own C++ agrees on all three
figures too, which is the strongest evidence available that the method is being
implemented correctly in all three places.

### Performance

512³, 2524 labels, Apple M4 Pro, one process each. **VTK's PyPI wheel ships the
`Sequential` SMP backend**, so `vtkSurfaceNets3D` runs single-threaded whatever
it is asked for — the middle column is the like-for-like one.

| | serra (1 thread) | VTK (1 thread) | serra (14 threads) |
| --- | --- | --- | --- |
| traverse the volume | **1.96 s** | 3.5 s | **0.45 s** |
| smoothing, 20 sweeps | 10.5 s | **4.2 s** | 2.2 s |
| one indexed mesh per object | **1.03 s** | 16.9 s | 1.01 s |
| peak RSS, traversal only | **2.7 GB** | 7–10 GB | 4.1 GB |

Three things worth reading off that table honestly.

**VTK's traversal is 1.8× slower than serra's on one thread**, and serra's
threading widens it to 7.8×. But VTK's is not a Python-overhead artifact:
converting the numpy array into `vtkImageData` is free on an F-contiguous
volume, and the 3.5 s is time inside the C++ filter. Wiring the C++ up directly
would not change it.

**VTK's smoothing is about 2.5× faster than serra's fairing per sweep** — 0.21 s
against 0.52 s over a comparable number of points. serra's Jacobi update needs
a second buffer where VTK's does not, which is the price of output that is
byte-identical however the volume is split across threads.

**Peak memory is the largest gap, 3–4×.** VTK holds the whole volume as one
polydata — 23.8M points and 51.8M faces before the per-object split — and
serra's peak is both lower and far more repeatable, 2.7 GB against a figure
that moved between 6.6 and 9.9 GB across runs.

### VTK does not give you per-object meshes

`vtkSurfaceNets3D` returns one polydata for the whole volume, with a
two-component `boundary_labels` cell array naming the materials either side of
each face. Each shared wall is stored once. A connectomics pipeline needs the
opposite — one indexed mesh per segment, each wall present in both neighbours —
and VTK has no filter for that, so the 16.9 s above is `bench/compare_zmesh.py`
doing it in numpy, bucketed in a single pass rather than filtered once per
label. A C++ implementation would beat it comfortably. The point is not that
16.9 s is VTK's number; it is that the step exists at all, and that serra's
1.03 s already includes it.

## Reproducing

```bash
make -C bench/frisken stock
make -C bench/frisken patched

python bench/compare_zmesh.py frisken --relax 20
python bench/compare_zmesh.py frisken --relax 20 \
    --frisken-bin bench/frisken/build/patched/sn_bench
python bench/compare_zmesh.py serra --threads 1 --fairing 20
python bench/compare_zmesh.py serra --fairing 20 --fairing-taubin
python bench/compare_zmesh.py zmesh

python bench/compare_zmesh.py vtk                     # boundary_style=all
python bench/compare_zmesh.py vtk --smoothing-iterations 20
python bench/compare_zmesh.py serra --threads 1 --close   # matches VTK exactly
```
