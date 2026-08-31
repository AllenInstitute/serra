"""The :class:`Mesh` container and its file formats."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["Mesh"]


@dataclass
class Mesh:
    """A triangle mesh for a single object.

    Attributes
    ----------
    vertices:
        ``(N, 3)`` float32 array of physical coordinates.
    faces:
        ``(M, 3)`` uint32 array of indices into ``vertices``.
    normals:
        ``(N, 3)`` float32 array of unit vertex normals, or ``None``.
    id:
        The label this mesh was extracted from.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    id: Optional[int] = None

    def __len__(self) -> int:
        return len(self.faces)

    def __repr__(self) -> str:
        return (
            f"Mesh(id={self.id}, vertices={len(self.vertices)}, "
            f"faces={len(self.faces)}, normals={self.normals is not None})"
        )

    @property
    def nbytes(self) -> int:
        """Total size of the underlying arrays."""
        total = self.vertices.nbytes + self.faces.nbytes
        if self.normals is not None:
            total += self.normals.nbytes
        return total

    def is_empty(self) -> bool:
        return len(self.faces) == 0

    def triangles(self) -> np.ndarray:
        """``(M, 3, 3)`` array of triangle corner coordinates."""
        return self.vertices[self.faces]

    # -- geometry -----------------------------------------------------------

    def area(self) -> float:
        """Total surface area."""
        t = self.triangles()
        cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())

    def volume(self) -> float:
        """Signed enclosed volume, via the divergence theorem.

        Only meaningful for a closed mesh. A negative result means the winding
        is inverted.
        """
        a, b, c = (self.vertices[self.faces[:, i]].astype(np.float64) for i in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def is_closed(self) -> bool:
        """Whether every edge is shared by exactly two triangles."""
        return self.count_boundary_edges() == 0

    def count_boundary_edges(self) -> int:
        """Number of undirected edges not shared by exactly two triangles.

        Nonzero means the surface is open — expected where an object runs off
        the edge of the volume and ``close`` was not used.
        """
        if len(self.faces) == 0:
            return 0
        e = np.concatenate(
            [self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]
        )
        e = np.sort(e, axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        return int((counts != 2).sum())

    # -- serialization ------------------------------------------------------

    def to_obj(self) -> bytes:
        """Wavefront OBJ. Indices are 1-based, per the format."""
        lines = [f"v {x} {y} {z}" for x, y, z in self.vertices]
        if self.normals is not None:
            lines += [f"vn {x} {y} {z}" for x, y, z in self.normals]
            lines += [
                f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}"
                for a, b, c in self.faces
            ]
        else:
            lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.faces]
        return ("\n".join(lines) + "\n").encode("utf-8")

    def to_ply(self) -> bytes:
        """Binary little-endian PLY."""
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(self.vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(self.faces)}\n"
            "property list uchar int vertex_index\n"
            "end_header\n"
        ).encode("ascii")

        verts = np.ascontiguousarray(self.vertices, dtype="<f4").tobytes()
        # Each face record is a count byte followed by three int32 indices.
        faces = np.empty(len(self.faces), dtype=[("n", "u1"), ("v", "<i4", 3)])
        faces["n"] = 3
        faces["v"] = self.faces
        return header + verts + faces.tobytes()

    def to_precomputed(self) -> bytes:
        """Neuroglancer precomputed mesh fragment."""
        return (
            struct.pack("<I", len(self.vertices))
            + np.ascontiguousarray(self.vertices, dtype="<f4").tobytes()
            + np.ascontiguousarray(self.faces, dtype="<u4").tobytes()
        )

    @classmethod
    def from_precomputed(cls, data: bytes, id: Optional[int] = None) -> "Mesh":
        """Parse a Neuroglancer precomputed mesh fragment."""
        (n_vertices,) = struct.unpack("<I", data[:4])
        end = 4 + n_vertices * 12
        vertices = np.frombuffer(data[4:end], dtype="<f4").reshape(n_vertices, 3)
        faces = np.frombuffer(data[end:], dtype="<u4").reshape(-1, 3)
        return cls(vertices=vertices.copy(), faces=faces.copy(), id=id)

    def save(self, path: str) -> None:
        """Write to ``path``, choosing the format from its extension."""
        lowered = path.lower()
        if lowered.endswith(".ply"):
            payload = self.to_ply()
        elif lowered.endswith(".obj"):
            payload = self.to_obj()
        else:
            raise ValueError(
                f"unknown mesh extension: {path!r} (expected .ply or .obj)"
            )
        with open(path, "wb") as handle:
            handle.write(payload)
