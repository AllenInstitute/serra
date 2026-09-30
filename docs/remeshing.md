# Multi-material remeshing and levels of detail (in progress)

This page records the design for serra's third remeshing phase: remeshing and
simplification done together, on every label's surface at once, so that
touching objects still fit exactly and triangles stay well shaped at every
level of detail. It is a plan and a progress log; the parts marked done are in
the code.

## Why the simplifier has to change

`get(label, reduction_factor=...)` simplifies each label's mesh on its own, with
quadric error collapse. A wall between two objects exists once in each of their
meshes, and the two copies are simplified independently, so they stop
coinciding: on dense neuropil the share of vertex positions belonging to more
than one label falls from 88.5% unsimplified to 16.8% at a reduction factor of
10 (see [Related work](related-work.md#the-finding-that-limits-all-of-this)).
Quadric simplification also trades triangle shape for fidelity, which is where
slivers come from.

## Decisions

| question | decision |
| --- | --- |
| what each level optimises | curvature-adaptive edge length, every collapse bounded by `max_error` |
| chunk seams at coarse levels | seam curves simplified in one dimension, identically in both chunks |
| API | a per-chunk `mesher.lods(levels=[...])` building every level for all labels at once; the default is one full-resolution level |
| maps between levels | fine→coarse and coarse→fine, each optional, to save memory |

## Design

**One copy of each wall.** A `WallMesh` stores every wall once,
tagged with the label on each side (background counts as one). Vertices are
shared between labels. Where three or more labels meet, an edge simply has
three or more triangles. Every operation is decided once per wall, so shared
walls fit exactly by construction.

**What each vertex may do** follows Faraj et al.'s multi-material remesher. It
is read from the label pairs around the vertex and kept up to date as the mesh
changes:

- **wall vertex**: all its triangles on one wall. May move in that wall and
  collapse into a wall or curve vertex.
- **curve vertex**: on a curve where three or more labels meet. Moves and
  collapses only along its curve.
- **corner**, **pinch** and **seam** vertices: never move. Seam curves are the
  exception at coarse levels, below.

**The loop** is Botsch–Kobbelt's isotropic remeshing, with a target edge length
that varies with curvature (Dunyach et al. 2013), in physical units. Each pass:

1. **split** edges longer than 4/3 of the target;
2. **collapse** edges shorter than 4/5 of it, unless the collapse would move the
   surface further than `max_error` from the fine one, or break any label's
   manifoldness (the link condition, checked for every label at the edge), or
   merge two different curves;
3. **flip** wall edges to even out vertex degree;
4. **relax** wall vertices within the surface and curve vertices along their
   curve, and project them back onto the fine surface.

Each level of detail starts from the one before, so all levels together cost
about as much as the first. Every pass is scheduled like the edge flips
already are: operations that share no triangle are chosen by a fixed priority
and applied together, so the result does not depend on the thread count.

**Seams.** A chunk's boundary is a curve of vertices both neighbouring chunks
see. At coarse levels it is simplified first, in one dimension, along itself
and within the seam plane, by a rule that reads only those vertices. Both chunks
reach the same coarse seam, and the interior is then remeshed with it held
fixed. Chunks keep stitching exactly, with no strip of fine triangles along
every seam.

**Maps between levels.** As collapses happen, each fine vertex is kept located
as a triangle of the coarse mesh and barycentric coordinates in it, and each
coarse vertex as the same on the fine mesh (in the manner of MAPS, Lee et al.
1998). Either direction can be turned off.

## Progress

| step | what | status |
| --- | --- | --- |
| 3a | `WallMesh` (`src/walls.rs`): every wall once, and each label's surface recovered exactly | **done** |
| 3b | vertex classification and the four operations on the shared mesh (`src/remesh.rs`) | **done** |
| 3c | the adaptive remeshing loop, with `max_error` against the fine surface | next |
| 3d | seam curves simplified identically in both chunks | |
| 3e | vertex maps between levels, each direction optional | |
| 3f | `mesher.lods()`, documentation and benchmarks against the current simplifier | |

### 3a: the shared wall mesh

Built from an extraction by pairing each label's quad with the other label's
copy of it. The four cells a quad is built from identify the voxel edge it is
dual to, so sorting on them pairs the copies. The copies' vertices are then
merged cell by cell. Extraction must be unsmoothed or cell-domain (`fairing`),
since per-label smoothing moves the copies apart; building from such an
extraction is refused.

Checked against `build()` on every label of a 192³ MICrONS crop, closed, open
and after `fairing=20` + Taubin: all 1,692 labels come back with identical
triangles, windings and vertex counts. The shared mesh holds 55.6% of the
per-label triangle count. Two things had to be got right on the way:

- **Merging can join two sheets of one label** in a cell with no ambiguous face.
  Take a label at two corners at opposite ends of a body diagonal, and a second
  label along an edge that touches both. About 30,000 vertices of the crop are
  like this. Each of the label's members of such a group keeps its own alias
  vertex, and the faces record which one each corner means, so a label always
  sees exactly its own vertices.
- **The manifold repair depended on vertex numbering.** Splitting one suspect
  vertex changes the edges its neighbours see, so the result depended on which
  was processed first. The repair now goes in order of cell, which is the same
  whichever mesh the surface came from. That changes `get()` only where two
  suspects share an edge, and makes it depend on the geometry alone.

Building the 192³ crop's shared mesh takes about 2.6 s, single-threaded, as a
pass after extraction. Moving it into the extraction pass is the fix once
everything else is in place.

### 3b: classification and the four operations

`Remesh` (`src/remesh.rs`) holds the shared mesh in editable form, with the
faces around each vertex, and offers `split`, `collapse`, `flip` and
`relocate`. Each checks whether it is allowed and leaves the mesh untouched if
not, so a remeshing policy can try operations and keep those that succeed. Vertex
and edge kinds are read from the label pairs of the triangles around them, so
they stay correct as the mesh changes.

A collapse must pass all of these:

- the link condition on the whole mesh, **and on each label's surface alone**. A
  vertex can be joined to both ends through one label's faces while the face
  opposite it on the edge belongs to another label;
- a label with faces at both ends must have a face on the edge itself.
  Otherwise the edge belongs to other labels' walls, and collapsing it glues two
  separate points of this label's surface together, which the link test cannot
  see;
- no two faces may end up on the same three vertices. A tetrahedron otherwise
  collapses into two coincident triangles, a closed "surface" the next collapse
  deletes outright;
- a curve must not close up on itself, and no face may turn over or become
  degenerate.

The last three were each found by the fuzz tests, not anticipated. Those tests
run thousands of random operations and check after every batch that every
label's surface is still a 2-manifold with the same Euler characteristic and
open-edge count. A second variant mostly collapses, which drives small
components down to the fewest faces their topology allows. By default each runs
4 seeds; `SERRA_FUZZ_SEEDS=120` ran both clean, about a million operations.

**Locked vertices** keep every triangle around them exactly as it is: pinned
seam vertices, vertices on any label's open rim, vertices from ambiguous cells,
and pinches. Two points about them:

- **The rim has to be judged per label.** Where one label's surface ends at the
  volume boundary, other labels' walls can still meet along the same edge, so
  the combined mesh shows no rim there. The fuzz tests found this too.
- **It was a lot of the surface.** On the 192³ crop 2.0% of vertices were locked
  and 5.9% of triangles touched one, which at a 10× reduction would have been
  over half the output. Resolved in 3c; see below.

Surfaces cut open by the volume boundary can touch that boundary at a single
vertex. `get()` has always produced this for open volumes, in about 1 label in
12 of the synthetic fixtures, and stitching resolves it. The fuzz checker exempts
rim vertices from the one-fan test for that reason; they are locked, so
remeshing cannot change them.

Setting up `Remesh` takes 4.9 s on the 192³ crop, on top of 2.6 s to build the
shared mesh. Both are single-threaded passes, and both are on the list to move
into the extraction.

### 3c, part 1: no more frozen patches

Vertices from ambiguous cells were locked because extraction's manifold repair
decides at them, per label, and needs to see their triangles unchanged. They are
now resolved once, up front, when `Remesh` is built. In the repair's own order
(by cell), each label's fans at the vertex are found the way its repair would
find them, from that label's own view of its neighbours. Then:

- **where every label agrees**, the vertex is split for real, one vertex per
  group of fans;
- **where they conflict**, and one label needs two vertices where another,
  whose single fan touches both, needs one, the shared vertex stays and the
  label that needs more gets an *alias* per extra fan. This is the same
  mechanism the pinches of 3a use.

Every label's vertex identities are then explicit, so extraction no longer runs
the repair at all. A vertex with aliases is **fixed**: it never moves and
nothing collapses into it, but the triangles around it can be split, flipped and
collapsed, with its aliases carried onto them. Aliases are keyed by face, vertex
and side rather than by corner, so they survive a triangle's corners being
reordered. Splitting an edge that ends at a fixed vertex is refused, because a
label seeing that vertex twice sees two edges there.

Two things this took:

- **An earlier attempt split what it could and locked the rest.** That left
  36-44% of ambiguous vertices locked, and it could not match `get()`. The
  repair splits an unresolvable vertex before reaching its neighbours, and
  resolving those neighbours against the unsplit vertex decided differently.
  Aliases resolve every vertex, so the order matches the repair's again.
- **Suspects are per label.** The repair splits a vertex for one label and not
  another, so the shared mesh now records each label's own suspects rather than
  deriving them from the cell.

On the 192³ crop, every one of the 1,692 labels still matches `get()` exactly,
closed and open, with no repair at extraction. In the closed volume 0.38% of
vertices are fixed and **no** triangle is frozen, against 5.9% before. Open, 2.2%
of triangles stay frozen: the open rim and the chunk's outer cell layer, which
3d addresses. The fuzz tests, now exercising operations around fixed vertices,
pass 120 seeds of each variant.
