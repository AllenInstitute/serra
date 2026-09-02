"""Cutting a finished mesh into a regular grid of cells."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from serra_mesh import _serra_mesh
from serra_mesh.mesh import Mesh

__all__ = ["dice"]


def dice(
    mesh: Mesh,
    chunk_shape: Sequence[float],
    grid_origin: Optional[Sequence[float]] = None,
    grid_size: Optional[Sequence[int]] = None,
) -> Dict[Tuple[int, int, int], Mesh]:
    """Cut ``mesh`` into the cells of a regular grid.

    Returns ``{(i, j, k): Mesh}`` for the cells that hold anything. Every
    triangle lands in exactly one cell and lies wholly inside it, and two cells
    that meet agree on the geometry along the plane between them **bit for
    bit** -- so concatenating the pieces and deduplicating vertices by exact
    equality reproduces the input mesh, with no tolerance and no repair pass.

    This is what the neuroglancer multi-resolution mesh format needs: it
    quantizes each fragment over the cell it is filed under, so anything
    outside is clamped flat onto the cell face, and it has thrown the original
    coordinates away by the time a weld could reconcile two sides that
    disagree.

    Parameters
    ----------
    chunk_shape:
        Cell size along each axis, in the mesh's own units.
    grid_origin:
        The grid's lower corner. Defaults to the mesh's own minimum, so cell
        ``(0, 0, 0)`` starts where the mesh does.
    grid_size:
        How many cells the grid holds along each axis. Indices are clamped into
        it, so a vertex lying exactly on the far face of the last cell stays in
        that cell rather than starting a new one. Defaults to whatever covers
        the mesh.

    Notes
    -----
    Classification against a cutting plane is **exact** -- a vertex is below,
    above, or on it, with no epsilon. That is what makes a T-junction
    impossible: a plane crosses the interior of a shared edge only when one end
    is strictly below and the other strictly above, and both triangles holding
    that edge see the same two coordinates, so either both split it or neither
    does. Treating a vertex within some tolerance of a plane as lying on it is
    what tears a surface -- the triangle whose corner was snapped stops being
    split while its neighbour across the far edge still is, leaving a vertex in
    the middle of an edge the first triangle spans.

    The cost is a sliver triangle where a vertex sits a hair off a plane. That
    is harmless: quantization collapses it and the decoder drops it, which
    leaves its edges held by exactly the two triangles either side.

    Examples
    --------
    >>> pieces = dice(mesh, chunk_shape=[64, 64, 64])      # doctest: +SKIP
    >>> sorted(pieces)[:2]                                  # doctest: +SKIP
    [(0, 0, 0), (0, 0, 1)]
    """
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (M, 3), got {faces.shape}")

    chunk_shape = np.asarray(chunk_shape, dtype=np.float64)
    if chunk_shape.shape != (3,):
        raise ValueError("chunk_shape must have 3 entries")
    if np.any(chunk_shape <= 0):
        raise ValueError("chunk_shape must be positive on every axis")

    if len(vertices) == 0 or len(faces) == 0:
        return {}

    if grid_origin is None:
        grid_origin = np.floor(vertices.min(axis=0)).astype(np.float64)
    else:
        grid_origin = np.asarray(grid_origin, dtype=np.float64)
        if grid_origin.shape != (3,):
            raise ValueError("grid_origin must have 3 entries")

    if grid_size is None:
        span = vertices.max(axis=0).astype(np.float64) - grid_origin
        grid_size = np.maximum(np.ceil(span / chunk_shape), 1).astype(np.int64)
    else:
        grid_size = np.asarray(grid_size, dtype=np.int64)
        if grid_size.shape != (3,):
            raise ValueError("grid_size must have 3 entries")
        if np.any(grid_size < 1):
            raise ValueError("grid_size must be at least 1 on every axis")

    pieces = _serra_mesh.dice(
        vertices,
        faces,
        [float(v) for v in chunk_shape],
        [float(v) for v in grid_origin],
        [int(v) for v in grid_size],
    )
    label = getattr(mesh, "id", None)
    return {
        index: Mesh(vertices=v, faces=f, id=label)
        for index, (v, f) in pieces.items()
    }
