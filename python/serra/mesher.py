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
    relaxation:
        Iterations of constrained smoothing. Zero (the default) uses purely
        local vertex placement. Raising it gives a smoother surface, more
        accurate area and better normals, at some cost in speed.
    max_deviation:
        How far relaxation may move a vertex from where local placement put it,
        in voxels. This bounds how far the surface can stray from the data and
        stops smoothing from shrinking objects away.
    relaxation_step:
        Fraction of the way to the neighbour average per iteration, in (0, 1].
    threads:
        How many threads to use. ``0`` (the default) uses every core; ``1``
        runs fully sequentially; any other value uses exactly that many.

        Set ``threads=1`` when you are already parallelising at a higher level
        — running one chunk per process in a pipeline, for instance — otherwise
        each process will try to claim every core and they will fight.

        Values above 1 get a private thread pool, so the setting is honoured
        exactly, is not overridden by ``RAYON_NUM_THREADS``, and does not
        disturb other users of rayon in the same process. Only ``threads=0``
        defers to ``RAYON_NUM_THREADS``.

        Output is byte-identical whatever this is set to.

    Notes
    -----
    **Chunked meshing.** To mesh a large volume in pieces, give each chunk a
    one-voxel halo on every side, so neighbouring input arrays overlap by two
    voxels. Vertices on a shared seam are then bit-identical between chunks and
    the pieces stitch together by vertex deduplication alone. A smaller overlap
    leaves the dual cells along the seam unshared, and the surfaces will not
    meet.

    One voxel of halo is enough at *any* ``relaxation`` setting. Relaxation
    holds the outermost layer of cells fixed, so it never reads past the halo
    and a chunk's mesh stays reproducible from that chunk's array alone. The
    trade-off is that a chunk's interior is smoothed slightly more than the band
    around its seams, so a stitched surface is self-consistent and watertight
    but not identical to the same volume meshed in one piece.

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
        relaxation: int = 0,
        max_deviation: float = 0.5,
        relaxation_step: float = 0.5,
        threads: int = 0,
    ):
        self.voxel_resolution = np.asarray(voxel_resolution, dtype=np.float64)
        self.axis_order = axis_order
        self.y_down = y_down
        self.relaxation = int(relaxation)
        self.max_deviation = float(max_deviation)
        self.relaxation_step = float(relaxation_step)
        self.threads = int(threads)
        if self.relaxation < 0:
            raise ValueError("relaxation must be non-negative")
        if self.threads < 0:
            raise ValueError("threads must be non-negative (0 means all cores)")
        self._inner = _serra.Mesher(
            voxel_resolution=[float(v) for v in self.voxel_resolution],
            axis_order=axis_order,
            y_down=y_down,
            relaxation=self.relaxation,
            max_deviation=self.max_deviation,
            relaxation_step=self.relaxation_step,
            threads=self.threads,
        )

    @property
    def effective_threads(self) -> int:
        """Threads actually used: the configured count, or every core when 0."""
        return self._inner.effective_threads

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
            f"axis_order={self.axis_order!r}, relaxation={self.relaxation}, "
            f"threads={self.effective_threads}, objects={len(self)})"
        )
