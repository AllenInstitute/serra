# Chunked meshing

To mesh a volume larger than memory, cut it into chunks and join the results.
serra is designed so that join needs nothing but vertex deduplication — no
tolerance, no snapping, no repair pass.

## The contract

Each chunk **owns a disjoint range of voxels** and is passed to `mesh()` with a
**one-voxel halo on every side**. Neighbouring input arrays therefore overlap by
two voxels.

```python
OWN = 512
HALO = 1

lo = index * OWN
hi = lo + OWN
chunk = volume[max(lo - HALO, 0) : hi + HALO, ...]

mesher = serra_mesh.Mesher(voxel_resolution=[32, 32, 40], threads=1)
mesher.mesh(chunk, close=False)
```

Use `close=False` for interior chunks. `close=True` seals the array's faces,
which is what you want for a whole volume but would cap a chunk off at its seams.

## Why two voxels and not one

serra places a vertex inside each *cell* — each 2×2×2 block of voxels — rather
than on voxel edges. For two chunks to agree, the layer of cells straddling the
seam has to exist in both.

With a one-voxel overlap it does not. The lower chunk's last cell ends exactly
where the upper chunk's first cell begins, so neither produces the layer in
between and the joined surface has a ring-shaped hole. This is measured in the
test suite: with a one-voxel overlap the lower chunk contributes **zero**
vertices to the seam band; with a one-voxel halo both contribute the same 56,
matching the whole-volume mesh bit for bit.

!!! note "Marching cubes differs here"
    zmesh needs only a one-voxel overlap, because its vertices sit on voxel
    edges rather than inside cells. If you are porting a pipeline, this is the
    one thing you must change.

## Relaxation does not widen the halo

Iterative smoothing normally spreads one cell per iteration, so `k` iterations
would need `k + 1` voxels of halo. serra instead **pins the outermost layer of
cells** — precisely the vertices whose one-ring the chunk does not fully
contain — so relaxation never reads past the halo.

A chunk's mesh is therefore reproducible from that chunk's own array alone, at
any `relaxation` setting. Objects outside the halo cannot perturb it.

The deliberate trade-off: a chunk's interior smooths slightly more than the band
around its seams, so a stitched surface is self-consistent and watertight, but
not identical to the same volume meshed in one piece.

## Threading

Set `threads=1` when running one chunk per process, or every process will try
to claim every core:

```python
serra_mesh.Mesher(threads=1)
```

See [the README](https://github.com/AllenInstitute/serra#controlling-parallelism)
for the scaling numbers.
