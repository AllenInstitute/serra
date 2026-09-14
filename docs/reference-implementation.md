# Against other SurfaceNets implementations

serra implements the method in [Frisken (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9623606/).
Two other implementations of that paper exist: the author's own
[C++ library](https://github.com/sarahfrisken/surfaceNets-multi-material), and
VTK's `vtkSurfaceNets3D`, reachable from Python as
[`ImageData.contour_labels`](https://docs.pyvista.org/api/core/_autosummary/pyvista.imagedatafilters.contour_labels).
Both were built and run against the same 512³ connectomics volume serra is
benchmarked on. Because all three implement the same algorithm, the comparison
is direct — mesh against mesh, face for face.

zmesh is in both tables below as well, but on different terms. It is marching
cubes, so it is a scale reference rather than a check: it
is *expected* to disagree on content, and the interesting question about it is not whether
it matches but what it costs.

## Results

### The three SurfaceNets agree; marching cubes does not

512³ neuropil volume, no smoothing, boundary padding on in every row:

| | objects | vertices | faces |
| --- | --- | --- | --- |
| serra | 2523 | 46,547,344 | **93,061,024** |
| VTK `vtkSurfaceNets3D` | 2523 | 46,486,892 | **93,061,024** |
| Frisken C++ | 2524 | 50.5M | 100.4M |
| zmesh (marching cubes) | 2523 | 46,530,512 | 93,031,372 |

serra and VTK agree on the face count **exactly**, and again at 128³ (156
objects, 1,831,136 faces each). Their vertex counts differ by 0.13%: VTK returns
one polydata with each shared wall stored once, and splitting that into one mesh
per label re-derives each label's vertex set, which counts a junction vertex
slightly differently from how serra emits it. Frisken's counts are higher only
because it meshes the background as an ordinary material; on a single sphere all
three give **3038 vertices and 6072 triangles**, watertight, enclosing exactly
8255.0 units of volume for 8255 voxels.

Three independent implementations agreeing to the face is the strongest
available evidence that the method is right in all three.

### Speed and memory

512³, 2524 labels, Apple M4 Pro, `-O3`, one process each. VTK, Frisken and
zmesh are all single-threaded, so **serra (1 thread) is the like-for-like
column**.

| | serra (1 thr) | serra (14 thr) | zmesh | VTK | Frisken (as published) | Frisken (patched) |
| --- | --- | --- | --- | --- | --- | --- |
| traverse the volume | 2.10 s | **0.63 s** | 2.15 s | 3.65 s | 16.0 s | 1.94 s |
| smoothing, per sweep | 0.55 s | **0.06 s** | — | 0.20 s | 5.64 s | 0.43 s |
| one mesh per object | **1.12 s** | 1.11 s | 23.8 s | 17.3 s | 92.7 s | 18.7 s |
| peak RSS | **2.97 GB** | 4.04 GB | 5.13 GB | 8.0–9.7 GB | 14.0 GB | 5.5 GB |

The two Frisken columns are carried over from the run that
[built the reference](#building-the-reference-implementations); every other
column was measured in one sitting on an idle machine. Rebuilding Frisken's
stock library to re-measure it costs 15 GB of resident memory, and its algorithm
has not changed.

- **serra is fastest at traversal**, 1.7× over VTK on one thread and 5.8× using
  all 14. VTK cannot be threaded from Python: its PyPI wheel ships the
  `Sequential` SMP backend.
- **zmesh traverses at serra's single-threaded speed and is 21× slower at
  delivery.** Marching cubes emits a triangle soup — every triangle carries its
  own three vertices, with no sharing — so turning that into indexed meshes
  means deduplicating 268.6M soup vertices down to 45.0M unique, per object,
  through a hash map: 83 ns each, which is simply what a hash-map insertion
  costs. serra never creates the duplicates. One vertex per cell per connected
  component, with quads referencing those indices directly, so extraction is a
  coordinate conversion and a triangulation at 23 ns per vertex. The third row
  is an architectural difference, not a tuning one.
- **zmesh has no smoothing**, at any setting; its dash means *not applicable*,
  not zero. serra's fairing, VTK's constrained smoothing and Frisken's
  relaxation are all sweeps over the extracted surface.
- **VTK smooths 2.8× faster than serra on one thread**, and serra overtakes it
  by 3× on 14. serra runs Jacobi, which needs a second buffer, so its output is
  byte-identical however the volume is split across threads. VTK and Frisken
  update in place and cannot be.
- **serra uses 2.7–3.3× less memory than VTK**, and far more predictably —
  VTK's peak moved between 8.0 and 9.7 GB across three runs of the same command.
- **Frisken's published library is ~8× slower than serra, but its algorithm is
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

Four defaults would each have invalidated the result:

- **pyvista's `boundary_style` defaults to `"external"`**, returning only the
  outside of the foreground and skipping every wall between adjacent objects —
  a third of the faces here. Set to `"all"`.
- **pyvista's `smoothing` defaults on** at 16 iterations, which would compare a
  smoothed VTK mesh against an unsmoothed serra one. Off unless asked for.
- **serra's `close` defaults off** while VTK's `pad_background` and Frisken's
  padding are always on. With `close=True` the face counts agree exactly;
  without it serra emits fewer (89,182,314 against 93,061,024). Every serra
  figure above is `--close`.
- **zmesh's `close` also defaults off, and costs it far more than it costs
  serra.** serra's is a virtual border and is free-ish: 1.77 s → 2.10 s on one
  thread. zmesh's wrapper allocates a padded C-order copy of the volume — 0.54
  GB at 512³ `uint32`, plus a transposing copy because this volume is
  Fortran-ordered, which also moves the traversal off zmesh's Fortran-order loop
  onto its C-order one. That nearly doubles it: **1.12 s unpadded against 2.15 s
  padded**. The table uses the padded run so that every column is padded, but
  the difference is the wrapper's, not the algorithm's. Peak RSS does *not* show
  the copy — 5.77 GB unpadded against 5.13 GB padded — because zmesh's peak is
  set by the soup dedup in extraction, which swamps a half-gigabyte volume and
  varies by more than that between runs. zmesh also never subtracts the pad
  again, so its vertices sit **+1 voxel** from everyone else's; that moves no
  count in the first table, and is corrected in
  [accuracy](accuracy.md#judged-against-an-analytic-solid), where coordinates
  do matter.

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
has no filter for it. The 17.3 s above is `bench/compare_zmesh.py` doing it in
numpy, bucketed in one pass rather than filtered per label; a C++ implementation
would beat it comfortably. The point is that the step exists at all, and that
serra's 1.12 s already includes it.

Frisken's 92.7 s is the library's own `objData()`, which makes two full passes
over every quad in the volume per label.

zmesh's 23.8 s is the other shape this problem takes. It already returns one
mesh per object, so nothing needs splitting — the cost is deduplicating the
triangle soup that marching cubes produced in the first place, which is a step
neither SurfaceNets implementation has to do at all.

### Sweeps are not iterations

VTK and Frisken update in place, propagating information across the array within
one sweep, so they converge in roughly half as many sweeps as serra's Jacobi.
Equal sweep counts are equal *work*, not equal *smoothing*. On the analytic tube
the two land within 3–6% of each other at matched sweep counts, because the
movement clamp binds before the difference in convergence rate matters.

### Reproducing

```bash
python bench/compare_zmesh.py serra --threads 1  --close   # matches VTK exactly
python bench/compare_zmesh.py serra --threads 14 --close
python bench/compare_zmesh.py serra --threads 1  --close --fairing 20
python bench/compare_zmesh.py zmesh --close --zmesh ../zmesh
python bench/compare_zmesh.py zmesh --zmesh ../zmesh       # unpadded; the pad's cost
python bench/compare_zmesh.py vtk
python bench/compare_zmesh.py vtk --smoothing-iterations 20
python bench/compare_zmesh.py frisken --relax 20
python bench/compare_zmesh.py frisken --relax 20 \
    --frisken-bin bench/frisken/build/patched/sn_bench
python bench/analytic_tube.py --radius 4                   # accuracy
```

`--zmesh` points at a local zmesh checkout, because zmesh is deliberately not a
declared dependency and the comparison should run against the build you have
rather than whatever is on PyPI. A zmesh is usually installed in the environment
too, so a path that does not resolve falls back to it silently; the `zmesh_path`
field in the JSON records which one actually ran. Per-sweep smoothing is
`smooth_s / 20` from the `--fairing` and `--smoothing-iterations` runs, and the
VTK peak is the range across three runs of the same command.
