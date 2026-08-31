"""The :class:`Mesher` front end."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

from serra_mesh import _serra_mesh
from serra_mesh.mesh import Mesh

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
        self._inner = _serra_mesh.Mesher(
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

    def mesh(
        self,
        data: np.ndarray,
        close: bool = False,
        owned_shape: Optional[Sequence[int]] = None,
    ) -> "Mesher":
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
        owned_shape:
            When meshing one chunk of a larger volume, how many voxels along
            each axis this chunk owns, counted from index 0. Faces are emitted
            only for the owned region, so each one belongs to exactly one
            chunk: concatenating the chunks and deduplicating vertices then
            reproduces the whole-volume mesh with no duplicate faces.

            **Wherever a neighbouring chunk exists, the array must extend two
            voxels past the owned region along that axis.** Dual contouring
            reads two cell layers per face, so one voxel of halo — enough for
            marching cubes — leaves a hole along the seam. Only the positive
            side needs it, which keeps the fetch within the next chunk.

            At the far edge of the volume there is no neighbour and no halo is
            needed, so the last chunk simply owns everything it holds.

            Leave it ``None`` when the array is the whole volume.

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
        if owned_shape is not None:
            owned_shape = [int(v) for v in owned_shape]
            if len(owned_shape) != 3:
                raise ValueError(
                    f"owned_shape must have 3 entries, got {len(owned_shape)}"
                )
            for axis, (owned, size) in enumerate(zip(owned_shape, data.shape)):
                if owned < 1:
                    raise ValueError("owned_shape entries must be positive")
                if owned > size:
                    raise ValueError(
                        f"axis {axis}: cannot own {owned} voxels of an array with "
                        f"{size}"
                    )
        self._inner.mesh(data, close=close, owned_shape=owned_shape)
        return self

    def ids(self) -> np.ndarray:
        """Labels present in the volume, ascending. Excludes background."""
        return self._inner.ids()

    def get(
        self,
        label: int,
        normals: bool = False,
        reduction_factor: int = 0,
        max_error: Optional[float] = None,
    ) -> Mesh:
        """The mesh for one object.

        Parameters
        ----------
        normals:
            Also compute unit vertex normals.
        reduction_factor:
            Ask for this many times fewer faces. ``0`` or ``1`` means no
            simplification.
        max_error:
            Cap on how far simplification may move a vertex, in the same units
            as ``voxel_resolution``. Defaults to the largest voxel dimension,
            matching zmesh.

        Notes
        -----
        Simplification preserves topology: it applies the link condition, so a
        closed 2-manifold stays a closed 2-manifold, and rejects collapses that
        would duplicate a face or flip a normal.

        It also leaves the chunk seam alone. Vertices pinned during extraction
        are never collapsed, which keeps not just their positions but the edges
        between them intact — so chunks still stitch after simplification.
        A consequence is that the band along a chunk seam stays at full
        resolution.

        Raises
        ------
        KeyError
            If the label is not present. Use ``label in mesher`` to check.
        """
        if reduction_factor < 0:
            raise ValueError("reduction_factor must be non-negative")
        result = self._inner.get(
            int(label),
            normals=normals,
            reduction_factor=int(reduction_factor),
            max_error=max_error,
        )
        if result is None:
            raise KeyError(f"label {label} is not present in the meshed volume")
        vertices, faces, vertex_normals = result
        return Mesh(
            vertices=vertices, faces=faces, normals=vertex_normals, id=int(label)
        )

    def get_all(self, normals: bool = False, **kwargs) -> Iterator[Mesh]:
        """Every object's mesh, in ascending label order.

        Yields one at a time so the caller can process and discard, rather than
        holding every mesh in memory at once.
        """
        for label in self.ids():
            yield self.get(int(label), normals=normals, **kwargs)

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
