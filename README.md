# serra

Analytical multi-material meshes from voxelized segmentations.

`serra` turns a 3-D array of integer labels into one triangle mesh per label. It is
built for connectomics-scale data, where a single chunk may contain hundreds of
thousands of distinct objects, so it makes **one pass over the volume** regardless of
how many labels are present.

```python
import serra

mesher = serra.Mesher(voxel_resolution=[4, 4, 40])
mesher.mesh(cutout)          # a 3-D array of integer labels
mesh = mesher.get(504)       # mesh.vertices, mesh.faces
```

## What makes it different

serra uses **multi-label surface nets** (dual contouring) rather than marching cubes.
Vertices sit inside cells at a position determined by where the label boundary
crosses the cell, instead of being pinned to voxel-edge midpoints. That removes the
staircase artifact, which shows up most clearly in surface area.

Measured on analytically-known spheres (isotropic voxels), against `zmesh`:

| sphere radius | zmesh area error | serra area error |
| ------------- | ---------------- | ---------------- |
| 10 voxels     | +8.52%           | +2.89%           |
| 20 voxels     | +9.31%           | +2.97%           |
| 40 voxels     | +8.77%           | +2.80%           |

Marching-cubes area error does not shrink as resolution rises — it is a systematic
bias, not a sampling error. With the optional relaxation pass enabled
(`relaxation=3`) serra's area error drops to **+0.38%**, while volume stays within
0.2% of analytic in every case.

Other properties:

- **2-manifold per object.** No non-manifold vertices or edges. Cells where a label
  touches itself only diagonally get their vertex split per connected component.
- **Watertight** inside the volume; open only where an object runs off the edge.
- **Deterministic.** Identical output regardless of thread count or whether the
  input array is C- or Fortran-ordered.
- **Chunk-seam exact.** Meshed with a 1-voxel halo, vertices on a shared seam are
  bit-identical between neighbouring chunks, so chunks stitch by vertex dedup alone.

## Chunked meshing

Each chunk owns a disjoint range of voxels and is passed to `mesh()` with a
**1-voxel halo on every side** — so neighbouring input arrays overlap by 2 voxels.
This is what makes the dual cells along a seam shared between both chunks, and
therefore what makes their vertices identical.

Enabling `relaxation=k` widens the required halo to `k + 1`, because relaxation
propagates position information one cell per iteration.

## Status

Under active development. See `docs/` for the user guide and the developer guide to
the module layout.

## License

MIT
