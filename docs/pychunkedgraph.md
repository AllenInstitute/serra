# Using serra in PyChunkedGraph

serra can replace zmesh in the PyChunkedGraph meshing pipeline, but **two
changes to PyChunkedGraph are required first**. Without them the pipeline fails
silently: meshes render correctly while being full of holes and topologically
disconnected across chunks, which is precisely what the cross-chunk graph
depends on.

Every claim on this page is checked by a script that needs no PyChunkedGraph
install:

```bash
uv sync --group bench
python bench/pcg_compat.py
```

Line numbers refer to `pychunkedgraph/meshing/` as of `zmesh==1.7.0`,
`dracopy==1.3.0`.

## Why anything has to change

Marching cubes reads **one cell per face**: a triangle depends only on the eight
voxels of the cube it sits in. Dual contouring reads **two cell layers per
face**: a quad is dual to a voxel edge and is built from the four cells
surrounding it. Everything below follows from that one difference.

## Change 1 — the halo must be two voxels

**`pychunkedgraph/meshing/meshgen.py:930`**

```diff
-    high_padding = 1
+    high_padding = 2
```

That constant is the whole change. It already flows to both places that need
it: the segmentation fetch (`meshgen.py:959` and `:967`, reaching
`meshgen_utils.py:166`) and the Draco quantization range (`meshgen.py:955` →
`:594`, where `min_quantization_range` is computed from
`mesh_block_shape + high_padding`). No separate range bump is needed.

For consistency the same default should move in the three other places it is
declared, so higher-layer stitching agrees with what layer 2 produced:

| file | line | current |
| --- | --- | --- |
| `meshgen_utils.py` | 153 | `def get_ws_seg_for_chunk(cg, chunk_id, mip, overlap_vx=1)` |
| `meshgen.py` | 579 | `get_draco_encoding_settings_for_chunk(..., high_padding: int = 1)` |
| `meshgen.py` | 1096 | `chunk_stitch_remeshing_task(..., high_padding=1, ...)` |
| `meshgen.py` | 1241 | `chunk_initial_sharded_stitching_task(..., high_padding=1, ...)` |

**This costs no extra downloads.** `chunk_end = chunk_start + mip_chunk_size +
overlap_vx` extends only in +x/+y/+z, and 1 and 2 both land inside the *next*
chunk. The factor-of-two saving the positive-only halo was designed for is
preserved exactly; only the slice end moves by one voxel. Extending backwards
is what would pull in the previous chunk and triple the fetches.

**What happens without it.** Measured on a 72³ volume cut into 24³ chunks:

| | faces produced | missing | broken edges |
| --- | --- | --- | --- |
| sphere, `high_padding=1` | 31,344 | **2,592** | 2,496 |
| sphere, `high_padding=2` | 33,936 | 0 | 0 |
| slab on a seam, `high_padding=1` | 25,296 | **1,840** | 1,808 |
| slab on a seam, `high_padding=2` | 27,136 | 0 | 0 |

A face that was never emitted cannot be recovered by welding, so this is not
something the stitcher can paper over.

## Change 2 — the weld predicate must cover the seam cell

**`pychunkedgraph/meshing/meshgen.py:743`**

```diff
-        are_chunk_aligned = (vertices == quantized_chunk_boundary).any(axis=1)
+        # Dual-contouring vertices sit inside the cell straddling the boundary
+        # rather than on the boundary plane, so match the cell, not the plane.
+        voxel = np.array(cg.meta.cv.mip_resolution(mip), dtype=vertices.dtype)
+        are_chunk_aligned = (
+            np.abs(vertices - quantized_chunk_boundary) <= voxel
+        ).any(axis=1)
```

**Why.** Marching cubes places vertices on cube edges, so a whole sheet of them
lands exactly on the shared plane and exact float equality finds them. serra
places one vertex inside each cell, so none lie on the plane.

Measured on a sphere spanning a chunk boundary at x = 24:

```
vertices shared across the seam                : 288
matched by  vertices == boundary               : 0
matched by  |vertices - boundary| <= 1 voxel   : 288
they lie at x in [24.168, 24.832] — inside the seam cell
```

**Zero of 288.** `merge_draco_meshes_across_boundaries` would weld nothing and
degrade to plain concatenation. The result renders correctly and is completely
disconnected at every seam.

!!! warning "This failure is silent"
    Nothing in the pipeline validates watertightness, and the only error
    handling around meshing is a bare `except:` on the Draco encode. An
    unwelded seam produces no warning at any stage.

### A softer, related issue

**`pychunkedgraph/meshing/mesh_analysis.py:40-44`** selects boundary vertices
with `vertices[:, axis] == np.amax(...)` — relative to the mesh's own extent
rather than an absolute plane. It therefore still returns a non-empty set with
serra, just one offset by up to half a voxel into the seam cell. The find-path
"precision mode" centroid it feeds will be correspondingly offset. Worth
knowing; not a blocker.

## What already matches, and needs no change

Confirmed by `bench/pcg_compat.py`, check 4:

| assumption | status |
| --- | --- |
| `Mesher(resolution)` with positional nm resolution | matches |
| `mesh(seg)` leaves block faces open (`close=False` default) | matches |
| `ids()`, `erase(id)` | matches |
| `get(id, reduction_factor=100, max_error=…)` | matches |
| `mesh.vertices[:] += chunk_offset` — mutable in place | matches |
| `vertices` float32 `(N,3)`, `faces` uint32 `(M,3)` | matches |
| `.flatten("C")` for `DracoPy.encode_mesh_to_buffer` | matches |
| `to_precomputed()` for the non-Draco debug path | matches |
| origin at the *centre* of voxel `[0,0,0]` | matches |
| `[x, y, z]` axis order, no transpose | matches |

`serra_mesh.Mesher` is therefore a drop-in for `zmesh.Mesher` at every call site
in `meshgen.py` (lines 953, 974, 979-983, 998).

### Draco quantization is not a problem

The lattice is `floor(min(resolution)/sqrt(2))` = 22 nm ≈ 0.69 voxel at
32×32×40 nm, and the `sqrt(2)` is calibrated to marching cubes' *minimum vertex
separation*. Dual-contouring vertices have no such minimum, so the concern is
reasonable — but measured on real data the damage is near-identical:

| | vertices collapsed | zero-area triangles |
| --- | --- | --- |
| zmesh | 3.6% | 3.9% |
| serra | 3.9% | 4.1% |

A pre-existing property of the pipeline, not something serra introduces. No
change needed.

## What serra must do, and does

The pipeline welds by exact coordinate equality, so serra has to make seam
vertices bit-identical between chunks that never see each other. It does:
positions are derived from integers in units of 1/256 of a voxel, and a cell
straddling a seam is present in both chunks and computed from the same eight
voxels by both.

Call `mesh()` with `owned_shape` so each face belongs to exactly one chunk:

```python
mesher = serra_mesh.Mesher(voxel_resolution=cv.mip_resolution(mip), threads=1)
mesher.mesh(seg, close=False, owned_shape=list(mip_chunk_size))
```

Without `owned_shape` both sides of a seam emit the wall between them, and
because `merge_draco_meshes_across_boundaries` deduplicates vertices but not
faces, those duplicates survive as non-manifold edges.

`threads=1` matters in this pipeline: workers already run one chunk per
process, and the default would have every one of them claim every core.

Measured end to end, 26 chunks of a sphere and 18 of a slab, vertex
deduplication only: face counts equal to the whole-volume mesh, zero broken
edges, zero duplicate faces.

## Sequencing

1. Land change 1. Harmless on its own — zmesh tolerates a wider halo, it just
   fetches one more voxel and quantizes over a slightly larger cube.
2. Land change 2. Mostly harmless on its own, but not entirely: widening the
   predicate puts more vertices through the `np.unique` weld. Marching-cubes
   vertices on the plane are still selected and still weld the same way, and
   non-coincident vertices simply pass through — but any *coincident* vertices
   within a voxel of a boundary that previously stayed separate will now be
   merged. The code's own comment at `meshgen.py:50-52` calls interior duplicate
   vertices an unresolved problem, so merging them is arguably an improvement,
   but it is a behaviour change and should be confirmed on a test dataset rather
   than assumed.

   A tighter predicate is possible — serra's shared vertices lie specifically in
   `[boundary, boundary + 1 voxel]`, not symmetrically around it — but the sign
   depends on which side of each axis the chunk sits, so the symmetric form is
   simpler and safe.
3. Swap the mesher.

Because both changes are no-ops for zmesh, they can be deployed and verified
before serra is introduced at all.
