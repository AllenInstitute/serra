# Against other SurfaceNets implementations

serra implements the method in [Frisken (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9623606/).
Two other implementations of that paper exist: the author's own
[C++ library](https://github.com/sarahfrisken/surfaceNets-multi-material), and
VTK's `vtkSurfaceNets3D`, reachable from Python as
[`ImageData.contour_labels`](https://docs.pyvista.org/api/core/_autosummary/pyvista.imagedatafilters.contour_labels).
Both were built and run against the same 512³ connectomics volume serra is
benchmarked on. Unlike zmesh — marching cubes, one object at a time — these are
the same algorithm, so the comparison is direct.

## Results

### All three produce the same mesh

512³ neuropil volume, no smoothing:

| | objects | vertices | faces |
| --- | --- | --- | --- |
| serra | 2523 | 46.5M | 93.1M |
| VTK `vtkSurfaceNets3D` | 2523 | 46.5M | 93.1M |
| Frisken C++ | 2524 | 50.5M | 100.4M |

serra and VTK are equal, not close, and equal again at 128³ (156 objects,
1,831,136 faces each). Frisken's counts are higher only because it meshes the
background as an ordinary material; on a single sphere all three give **3038
vertices and 6072 triangles**, watertight, enclosing exactly 8255.0 units of
volume for 8255 voxels.

Three independent implementations agreeing to the face is the strongest
available evidence that the method is right in all three.

### Speed and memory

512³, 2524 labels, Apple M4 Pro, `-O3`, one process each. VTK and Frisken are
single-threaded, so **serra (1 thread) is the like-for-like column**.

| | serra (1 thr) | serra (14 thr) | VTK | Frisken (as published) | Frisken (patched) |
| --- | --- | --- | --- | --- | --- |
| traverse the volume | **1.96 s** | **0.45 s** | 3.5 s | 16.0 s | 1.94 s |
| smoothing, per sweep | 0.52 s | 0.11 s | **0.21 s** | 5.64 s | 0.43 s |
| one mesh per object | **1.03 s** | 1.01 s | 16.9 s | 92.7 s | 18.7 s |
| peak RSS | **2.7 GB** | 4.1 GB | 7–10 GB | 14.0 GB | 5.5 GB |

- **serra is fastest at traversal**, 1.8× over VTK on one thread and 7.8× using
  all 14. VTK cannot be threaded from Python: its PyPI wheel ships the
  `Sequential` SMP backend.
- **VTK smooths 2.5× faster than serra per sweep.** serra runs Jacobi, which
  needs a second buffer, so that output is byte-identical however the volume is
  split across threads. VTK and Frisken update in place and cannot be.
- **serra uses 3–4× less memory than VTK**, and far more predictably — VTK's
  peak moved between 6.6 and 9.9 GB across runs.
- **Frisken's published library is ~10× slower than serra, but its algorithm is
  not.** Three lines of layout fix — see below — bring it level.

### Where serra differs

| | serra | VTK and Frisken |
| --- | --- | --- |
| unsmoothed vertex position | centroid of the edge crossings | centre of the cell |
| shrink compensation | Taubin λ/μ (`fairing_taubin`) | none (VTK: constrained) |
| output | one indexed mesh per object | one polydata for the whole volume |

The placement difference is the one that shows up in accuracy: serra's
unsmoothed mesh scores 0.109 voxels of mean surface error against an analytic
tube where a cell-centre mesh scores 0.240. See
[accuracy](accuracy.md#judged-against-an-analytic-solid).

Frisken's pure Laplacian loses **15.7% of a radius-4 tube's volume at 20
sweeps** against serra's 0.42% with Taubin. serra's own Laplacian, run as a
control, loses 14.97% — the same operator, so the difference is Taubin, not the
domain.

## Methods and details

### Building the reference implementations

The Frisken repo ships a Visual Studio solution, but only `Source/Application`
needs Qt. `Source/SNLib` is 1,613 lines of ISO C++ with only standard-library
includes, and `clang++ -std=c++17` compiles all five files with zero errors:

```bash
make -C bench/frisken stock       # or: make FRISKEN=/path/to/clone stock
make -C bench/frisken patched
```

`bench/frisken/sn_bench.cpp` is the missing `main()` — it prints the same JSON
keys as `bench/compare_zmesh.py` so it drops into that harness. The clone is
never modified; `patched` copies the sources out and applies
`bench/frisken/patches/*.patch` to the copy.

VTK needs no build; it comes from the `pyvista` bench dependency group.

### Why the published Frisken library is slow

`MMCellFlag` stores 19 bit masks as **non-static** `const unsigned int` data
members and declares a `virtual` destructor. Neither carries information, and
together they make `sizeof(MMCellFlag)` **88 bytes**, 84 of them the same
constants in every cell. `sizeof(Cell)` is then 112 B, and `MMCellMap`
allocates `new Cell[(nx+2)(ny+2)(nz+2)]` — dense over the padded volume,
boundary or not. At 512³ that is 135.8M cells and 15.2 GB.

Patch 01 makes the masks `static constexpr` and drops `virtual`, taking `Cell`
to **24 bytes** with byte-identical output. Patch 02 indexes quads by label so
`objData()` stops rescanning the global quad list once per object. Relaxation
gains 13.1× from a 4.7× smaller working set, because it is pure pointer-chasing
over the cell array and at 112 B/cell nearly every neighbour lookup misses cache.

### Settings that make the comparison fair

Three defaults would each have invalidated the result:

- **pyvista's `boundary_style` defaults to `"external"`**, returning only the
  outside of the foreground and skipping every wall between adjacent objects —
  a third of the faces here. Set to `"all"`.
- **pyvista's `smoothing` defaults on** at 16 iterations, which would compare a
  smoothed VTK mesh against an unsmoothed serra one. Off unless asked for.
- **serra's `close` defaults off** while VTK's `pad_background` and Frisken's
  padding are always on. With `close=True` the face counts agree exactly;
  without it serra emits fewer.

The Frisken `.vcxproj` configuration inspected carries `Optimization=Disabled`;
all figures here are `-O3 -DNDEBUG`.

### Two conversions the reference needs

`MMCellMap` walks `i` fastest, so its `arraySize[0]` is numpy's **axis 2** — the
mirror of serra's convention. It also pads by a voxel on every face and reports
positions in the *padded* index space, so raw output sits **+1 voxel** from the
segmentation on every axis. `sn_bench` corrects both and reverses triangle
winding to undo the reflection. On a cubic volume the axis swap is invisible,
which is what makes it dangerous.

### On the per-object numbers

`vtkSurfaceNets3D` returns one polydata with a two-component `boundary_labels`
cell array, each shared wall stored once. A connectomics pipeline needs the
opposite — one indexed mesh per segment, each wall in both neighbours — and VTK
has no filter for it. The 16.9 s above is `bench/compare_zmesh.py` doing it in
numpy, bucketed in one pass rather than filtered per label; a C++ implementation
would beat it comfortably. The point is that the step exists at all, and that
serra's 1.03 s already includes it.

Frisken's 92.7 s is the library's own `objData()`, which makes two full passes
over every quad in the volume per label.

### Sweeps are not iterations

VTK and Frisken update in place, propagating information across the array within
one sweep, so they converge in roughly half as many sweeps as serra's Jacobi.
Equal sweep counts are equal *work*, not equal *smoothing*. On the analytic tube
the two land within 3–6% of each other at matched sweep counts, because the
movement clamp binds before the difference in convergence rate matters.

### Reproducing

```bash
python bench/compare_zmesh.py serra --threads 1 --close   # matches VTK exactly
python bench/compare_zmesh.py vtk
python bench/compare_zmesh.py vtk --smoothing-iterations 20
python bench/compare_zmesh.py frisken --relax 20
python bench/compare_zmesh.py frisken --relax 20 \
    --frisken-bin bench/frisken/build/patched/sn_bench
python bench/analytic_tube.py --radius 4                  # accuracy
```
