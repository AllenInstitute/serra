# serra

Analytical multi-material meshes from voxelized segmentations.

serra turns a 3-D array of integer labels into one triangle mesh per label. It
is built for connectomics-scale data, where a single chunk may hold hundreds of
thousands of distinct objects, so it makes **one pass over the volume** however
many labels are present.

```python
import serra_mesh

mesher = serra_mesh.Mesher(voxel_resolution=[32, 32, 40])
mesher.mesh(cutout)          # a 3-D array of unsigned integer labels
mesh = mesher.get(504)       # mesh.vertices, mesh.faces
```

## Install

```bash
pip install serra-mesh          # see the repository README for the current name
```

Wheels are built against Python's stable ABI, so one wheel per platform covers
Python 3.9 and up.

## What it guarantees

- **2-manifold per object.** No non-manifold vertices or edges, including where
  an object touches itself only diagonally.
- **Watertight** inside the volume; open only where an object runs off the edge,
  unless `close=True`.
- **Deterministic.** Byte-identical output regardless of thread count, memory
  order, dtype width or platform.
- **Chunk-seam exact.** Meshed with a one-voxel halo, vertices on a shared seam
  are bit-identical between neighbouring chunks.

## Why not marching cubes

Marching cubes pins vertices to voxel-edge midpoints, which leaves axis-aligned
terracing. It shows up most clearly in surface area, where the error does not
shrink as resolution rises:

| sphere radius | zmesh (marching cubes) | serra |
| --- | --- | --- |
| 10 voxels | +8.52% | +2.89% |
| 20 voxels | +9.31% | +2.97% |
| 40 voxels | +8.77% | +2.80% |

See [Accuracy and smoothing](accuracy.md) for the full picture, including what
the optional relaxation pass buys.
