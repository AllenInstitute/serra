"""The :class:`Mesher` front end."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np

from serra import _serra
from serra.mesh import Mesh

__all__ = ["Mesher"]


class Mesher:
    """Extracts one mesh per label from a 3-D array of integer labels.

    The whole volume is traversed once, however many labels it contains, so
    cost is driven by boundary area rather than by object count.

    Parameters
    ----------
    voxel_resolution:
        Physical size of a voxel along each **array** axis. The units are
        yours; nanometres are the usual choice for EM data.
    axis_order:
        Which physical axis each array axis runs along. ``"XYZ"`` (the default)
        means the array is indexed ``[x, y, z]``; ``"ZYX"`` means ``[z, y, x]``.
        No copy or transpose is made — this only changes how coordinates are
        emitted.
    y_down:
        Set for the image convention where Y increases downward.

    Notes
    -----
    **Chunked meshing.** To mesh a large volume in pieces, give each chunk a
    one-voxel halo on every side, so neighbouring input arrays overlap by two
    voxels. Vertices on a shared seam are then bit-identical between chunks and
    the pieces stitch together by vertex deduplication alone. A smaller overlap
    leaves the dual cells along the seam unshared, and the surfaces will not
    meet.

    Examples
    --------
    >>> mesher = Mesher(voxel_resolution=[4, 4, 40])
    >>> mesher.mesh(cutout)                       # doctest: +SKIP
    >>> mesh = mesher.get(504)                    # doctest: +SKIP
    >>> mesh.vertices.shape, mesh.faces.shape     # doctest: +SKIP
    """

    def __init__(
        self,
        voxel_resolution: Sequence[float] = (1.0, 1.0, 1.0),
        axis_order: str = "XYZ",
        y_down: bool = False,
    ):
        self.voxel_resolution = np.asarray(voxel_resolution, dtype=np.float64)
        self.axis_order = axis_order
        self.y_down = y_down
        self._inner = _serra.Mesher(
            voxel_resolution=[float(v) for v in self.voxel_resolution],
            axis_order=axis_order,
            y_down=y_down,
        )

    def mesh(self, data: np.ndarray, close: bool = False) -> "Mesher":
        """Extract every object's surface from ``data``.

        Parameters
        ----------
        data:
            3-D array of unsigned integer labels (uint8/16/32/64). C- and
            Fortran-ordered arrays give identical results. Label 0 is
            background and is never meshed.
        close:
            Treat the volume as surrounded by background, so objects touching
            the array edge come back sealed. The border is virtual, so this
            costs no extra memory.

        Returns
        -------
        The mesher, so calls can be chained.
        """
        if data.ndim != 3:
            raise ValueError(f"expected a 3-D array, got shape {data.shape}")
        if data.dtype.kind == "i":
            raise TypeError(
                f"signed labels ({data.dtype}) are not supported; reinterpret "
                f"them without copying, e.g. data.view(np.uint{data.dtype.itemsize * 8})"
            )
        if data.dtype.kind != "u":
            raise TypeError(f"expected unsigned integer labels, got {data.dtype}")
        self._inner.mesh(data, close=close)
        return self

    def ids(self) -> np.ndarray:
        """Labels present in the volume, ascending. Excludes background."""
        return self._inner.ids()

    def get(self, label: int, normals: bool = False) -> Mesh:
        """The mesh for one object.

        Raises
        ------
        KeyError
            If the label is not present. Use ``label in mesher`` to check.
        """
        result = self._inner.get(int(label), normals=normals)
        if result is None:
            raise KeyError(f"label {label} is not present in the meshed volume")
        vertices, faces, vertex_normals = result
        return Mesh(
            vertices=vertices, faces=faces, normals=vertex_normals, id=int(label)
        )

    def get_all(self, normals: bool = False) -> Iterator[Mesh]:
        """Every object's mesh, in ascending label order.

        Yields one at a time so the caller can process and discard, rather than
        holding every mesh in memory at once.
        """
        for label in self.ids():
            yield self.get(int(label), normals=normals)

    def erase(self, label: int) -> bool:
        """Drop one object's surface, freeing its memory."""
        return self._inner.erase(int(label))

    def clear(self) -> None:
        """Drop every stored surface."""
        self._inner.clear()

    def __contains__(self, label: int) -> bool:
        return self._inner.contains(int(label))

    def __len__(self) -> int:
        return len(self._inner)

    def __iter__(self) -> Iterable[int]:
        return iter(self.ids())

    def __repr__(self) -> str:
        return (
            f"Mesher(voxel_resolution={self.voxel_resolution.tolist()}, "
            f"axis_order={self.axis_order!r}, objects={len(self)})"
        )
