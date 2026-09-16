# Recommended workflow

```python
import serra_mesh

mesher = serra_mesh.Mesher(
    voxel_resolution=[32, 32, 40],   # your voxel size, in nm
    fairing=20,                      # smoothing: cell-domain Taubin
    fairing_taubin=True,
)
mesher.mesh(volume, close=True)
```

That is the recommended configuration for almost every job. What each part is
doing:

| setting | recommendation | why |
| --- | --- | --- |
| `voxel_resolution` | your voxel size, in physical units | vertices come out in those units, so nothing downstream has to rescale |
| `fairing=20` + `fairing_taubin=True` | use it | cell-domain Taubin: touching objects keep a bit-identical shared wall, and thin processes keep their volume |
| `max_deviation` | leave at `0.5` | caps how far smoothing may move any vertex from the data |
| `close` | `True` for a whole volume | seals objects that touch the array edge; the border is virtual, so it costs no memory |
| `threads` | `0` alone, `1` inside a pool | `0` uses every core; `1` stops parallel workers fighting over them |
| `reduction_factor` (on `get()`) | `0` for measurement, `2`–`10` for display | simplification is topology- and seam-preserving, but it is still a loss |

Smoothing runs inside `mesh()`, so it always happens **before** `get()`
simplifies — the order that measures better. You do not have to sequence it
yourself.

If your objects never touch each other, `taubin=20` is equivalent and slightly
cheaper in bookkeeping. Everything else is a trade-off worth understanding
before you change it: see [Accuracy and smoothing](accuracy.md).

!!! warning "Labels must be unsigned"
    `uint8/16/32/64` only. Signed arrays raise, rather than silently
    reinterpreting — `data.view(np.uint32)` converts without copying. Label 0
    is background and is never meshed.

## Every mesh in a volume that fits in memory

One pass over the voxels builds every object's surface; `get_all()` then yields
them in ascending label order, one at a time, so you never hold more than one.

```python
import serra_mesh

mesher = serra_mesh.Mesher(
    voxel_resolution=[32, 32, 40],
    fairing=20,
    fairing_taubin=True,
)
mesher.mesh(volume, close=True)

print(f"{len(mesher)} objects")

for mesh in mesher.get_all():
    mesh.save(f"meshes/{mesh.id}.ply")
```

For a volume with many objects, extraction — not meshing — is where the memory
goes, because every surface is held until you drop it. Release each one as you
finish with it:

```python
for label in mesher.ids():
    mesh = mesher.get(int(label))
    mesh.save(f"meshes/{label}.ply")
    mesher.erase(label)          # frees that object's surface
```

`get()` also takes `normals=True` for unit vertex normals, and
`reduction_factor` / `max_error` to simplify on the way out. A `Mesh` carries
`vertices`, `faces`, `normals` and `id`, and writes itself out with `to_obj()`,
`to_ply()`, `to_precomputed()` or `save()` (which picks the format from the
extension).

## Levels of detail

Ask the same object for several reductions. The mesh is smoothed once, in
`mesh()`, and each `get()` simplifies that result independently:

```python
for lod, factor in enumerate([1, 4, 16, 64]):
    mesh = mesher.get(label, reduction_factor=factor)
    write_lod(lod, mesh)
```

`reduction_factor` is a request, not a guarantee: `max_error` also bounds how
far a vertex may move, and it defaults to the largest voxel dimension, so past
some factor the face count stops falling. Raise `max_error` if you need a
coarser level than that allows.

Simplification applies the link condition, so a closed 2-manifold stays one,
and it never collapses a pinned seam vertex — chunks still stitch after
simplification, at the cost of the band along each seam staying at full
resolution.

## Cutting a mesh into fragments

Neuroglancer's multi-resolution format wants each level of detail cut into
fragments on a regular grid. `dice()` makes that cut, on a finished mesh —
straight from `get()`, or from `stitch()` after a chunked run:

```python
CHUNK = [1024, 1024, 1280]        # 32 voxels at 32×32×40 nm

pieces = serra_mesh.dice(mesh, chunk_shape=CHUNK)

for (i, j, k), fragment in pieces.items():
    write_fragment(i, j, k, fragment)
```

`chunk_shape` is in the mesh's own units, so it is your fragment size in voxels
times the voxel resolution. Only cells that hold something come back. Every
triangle lands in exactly one cell and lies wholly inside it, so a triangle
straddling a cell boundary is split and the total face count **rises** — 33,708
to 36,552 on a sphere cut into eight cells.

!!! warning "Pass a shared grid when dicing more than one object"
    `grid_origin` defaults to the mesh's own lower corner and `grid_size` to
    whatever covers it, which puts every object on a different grid. For a
    volume of objects, pass the same `grid_origin` and `grid_size` for all of
    them — otherwise fragments that should tile together do not.

For writing the format itself, ask for the integer lattice directly:

```python
pieces = serra_mesh.dice(
    mesh,
    chunk_shape=CHUNK,
    grid_origin=volume_origin,     # shared across every object
    grid_size=grid_size,
    quantization_bits=10,          # neuroglancer allows 10 or 16
)
```

Vertices then come back as integers in `[0, 2**bits - 1]` across the cell that
holds them — where `cloudvolume.to_stored_model_space` would put them, so that
step can be skipped.

It also makes the fragments agree *exactly* rather than very nearly, which is
the reason to prefer it. The format quantizes each fragment over its own cell
and has discarded the original coordinates by the time anything could reconcile
two sides that disagree, and at coordinates of a thousand voxels one float32
step is about a nanometre — enough to read as a crack. Rounding onto one grid
spanning the whole volume before the split collapses that difference: a shared
vertex comes out as `2**bits - 1` in the lower cell and `0` in the upper one.

Classification against a cutting plane is exact, with no epsilon, which is what
makes a T-junction impossible — both triangles holding an edge see the same two
coordinates, so either both split it or neither does. Concatenating the pieces
and welding by exact equality reproduces the input mesh.

## A volume too large for memory

Mesh it in chunks. Each chunk owns a disjoint block of voxels and reads two
voxels past it wherever a neighbour exists; `owned_shape` then emits faces only
for the owned block, so every face belongs to exactly one chunk and the pieces
join with no duplicates.

```python
import numpy as np
import serra_mesh

RESOLUTION = np.array([32, 32, 40])
OWN = 256      # voxels owned along each axis
HALO = 2       # dual contouring reads two cell layers past the seam

pieces = {}    # label -> [(mesh, offset), ...]

for index in np.ndindex(*grid_shape):
    lo = np.array(index) * OWN
    owned = np.minimum(OWN, volume_shape - lo)        # last chunk owns rest
    hi = np.minimum(lo + owned + HALO, volume_shape)  # halo on high side

    chunk = read_subvolume(lo, hi)                     # your reader

    mesher = serra_mesh.Mesher(
        voxel_resolution=RESOLUTION.tolist(),
        fairing=20,
        fairing_taubin=True,
        threads=1,                                     # one chunk per process
    )
    mesher.mesh(chunk, owned_shape=owned)

    offset = lo * RESOLUTION
    for mesh in mesher.get_all():
        pieces.setdefault(mesh.id, []).append((mesh, offset))

for label, parts in pieces.items():
    whole = serra_mesh.stitch(parts, dedup_faces=False, id=label)
    whole.save(f"meshes/{label}.ply")
```

Welding is by exact coordinate equality — no tolerance, no snapping, no repair
pass — because serra derives positions from integers in units of 1/256 of a
voxel and both chunks compute a shared cell identically. `dedup_faces=False` is
right here because `owned_shape` already made each face unique; set it `True`
only if you meshed without it.

!!! note "Objects that touch themselves diagonally"
    Where a surface touches itself only at a diagonal, serra emits two
    coincident-but-distinct vertices to keep the object 2-manifold. `stitch`
    welds by position, so it merges that pair and pinches the surface there,
    leaving a few boundary edges on an otherwise closed object. This is a
    property of `stitch` and not of chunking — passing a
    whole-volume mesh through it as a single piece does the same thing. Mesh
    those objects in one piece if you need them exactly closed.

Two things to get right:

- **The halo is two voxels, not one.** One voxel is enough for marching cubes,
  but serra places a vertex inside each cell, so a one-voxel overlap leaves the
  seam cell layer belonging to neither chunk and the joined surface has a
  ring-shaped hole. See [Chunked meshing](chunked.md).
- **Leave `close=False` on chunks.** It seals the whole array, which is what
  you want for a complete volume but would cap an interior chunk off at its
  seams. If you need the finished surface sealed at the volume's outer
  boundary, pad the volume with one voxel of background label before
  chunking — real background does the same job as the virtual border.

Collecting every piece of every object in one dict, as above, assumes the
finished meshes fit in memory even though the voxels did not. For a genuinely
large run, write each chunk's pieces out keyed by label and stitch one label at
a time.

The pieces are also independent, so the chunk loop parallelises across
processes with `threads=1`. A chunk's mesh is reproducible from that chunk's
own array alone, at any smoothing setting, so nothing outside the halo can
perturb it.

## Checking the result

```python
mesh = mesher.get(label)

mesh.is_closed()               # no boundary edges
mesh.count_boundary_edges()    # how many, if not
mesh.volume(), mesh.area()     # enclosed volume and surface area
```

With `close=True` on a whole volume every object comes back closed, 2-manifold
and correctly oriented. Without it, objects running off the array edge are
legitimately open there — that is the surface genuinely continuing outside the
cutout, not a defect.

## Mistakes worth knowing about

| symptom | cause |
| --- | --- |
| ring-shaped holes along chunk seams | one voxel of halo instead of two |
| duplicate faces after joining | meshed without `owned_shape`, stitched with `dedup_faces=False` |
| chunks capped off at their seams | `close=True` on an interior chunk |
| a parallel run slower than one process | `threads=0` in every worker, all claiming every core |
| cracks between neighbouring fragments | diced per object without a shared `grid_origin`, or without `quantization_bits` |
| thin processes lost volume | plain Laplacian (`relaxation`, or `fairing` without `fairing_taubin`) |
| smoothing made a decimated mesh worse | smoothed after simplifying; serra's order already avoids this |
| `TypeError` on `mesh()` | signed labels — `.view(np.uint32)` first |

## Next

- [Chunked meshing](chunked.md) — the seam contract in detail.
- [Accuracy and smoothing](accuracy.md) — what those settings buy, measured.
- [API reference](reference.md) — every parameter.
