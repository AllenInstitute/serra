"""Joining meshes from adjacent chunks."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from serra_mesh.mesh import Mesh

__all__ = ["stitch"]


def stitch(
    pieces: Iterable[Tuple[Mesh, Sequence[float]]],
    dedup_faces: bool = True,
    id: Optional[int] = None,
) -> Mesh:
    """Join per-chunk meshes of one object into a single mesh.

    Parameters
    ----------
    pieces:
        ``(mesh, offset)`` pairs, where ``offset`` is the chunk's origin in the
        same units as the mesh's vertices. Meshes are translated by their
        offset before joining.
    dedup_faces:
        Also drop faces that appear more than once. Needed only when the
        chunks were meshed *without* ``owned_shape``, in which case both sides
        of a seam emit the wall between them. Harmless otherwise.
    id:
        Label to record on the result.

    Returns
    -------
    A single mesh whose seam vertices have been welded.

    Notes
    -----
    Welding is by **exact** coordinate equality, with no tolerance. That is
    sound because serra derives vertex positions from integers in units of
    1/256 of a voxel, and a cell straddling a seam is present in both chunks
    and computed identically by both. It is also why the chunks must have been
    meshed with two voxels of halo: with one, the cells along the seam are
    split between the chunks and the faces there are produced by neither, which
    no amount of welding can repair.
    """
    vertex_index: dict[bytes, int] = {}
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []

    for mesh, offset in pieces:
        if len(mesh.faces) == 0:
            continue
        moved = mesh.vertices.astype(np.float64) + np.asarray(offset, dtype=np.float64)
        # float32 keys, so two chunks that computed the same position produce
        # the same bytes.
        keys = np.ascontiguousarray(moved, dtype=np.float32)
        remap = np.empty(len(keys), np.int64)
        for i, key in enumerate(keys):
            k = key.tobytes()
            slot = vertex_index.get(k)
            if slot is None:
                slot = len(vertices)
                vertex_index[k] = slot
                vertices.append(key)
            remap[i] = slot
        faces.append(remap[mesh.faces])

    if not vertices:
        return Mesh(
            vertices=np.zeros((0, 3), np.float32),
            faces=np.zeros((0, 3), np.uint32),
            id=id,
        )

    joined = np.concatenate(faces) if faces else np.zeros((0, 3), np.int64)
    if dedup_faces and len(joined):
        _, keep = np.unique(np.sort(joined, axis=1), axis=0, return_index=True)
        joined = joined[np.sort(keep)]

    return Mesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=joined.astype(np.uint32),
        id=id,
    )
