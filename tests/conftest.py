"""Shape builders and mesh-validity helpers shared by the test suite.

Note on axis conventions: the voxel masks here are built with
``z, y, x = np.ogrid[...]``, so ``x`` varies along **array axis 2** and ``z``
along array axis 0. Analytic quantities that are not axis-symmetric — an
ellipsoid's semi-axes, a surface normal — must respect that, or a correct mesh
will look badly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------------------------------
# shape builders
# --------------------------------------------------------------------------


def sphere_mask(radius: float, pad: int = 4) -> np.ndarray:
    """Solid sphere, centred, with `pad` voxels of background around it."""
    n = int(2 * radius) + 2 * pad
    c = (n - 1) / 2
    z, y, x = np.ogrid[:n, :n, :n]
    mask = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= radius * radius
    return mask.astype(np.uint32)


def ellipsoid_mask(
    semi_axes: tuple[float, float, float], pad: int = 4
) -> tuple[np.ndarray, tuple[float, float, float], float]:
    """Solid ellipsoid.

    `semi_axes` is given in **array-axis order**, so it can be compared
    against vertex coordinates directly.

    Returns the mask, the array-axis-ordered semi-axes, and the centre.
    """
    n = int(2 * max(semi_axes)) + 2 * pad
    c = (n - 1) / 2
    grids = np.ogrid[:n, :n, :n]
    q = sum(((grids[k] - c) / semi_axes[k]) ** 2 for k in range(3))
    return (q <= 1).astype(np.uint32), semi_axes, c


def cylinder_mask(radius: float, height: float, axis: int, pad: int = 4) -> np.ndarray:
    """Solid circular cylinder whose long axis is array axis `axis`."""
    shape = [int(2 * radius) + 2 * pad] * 3
    shape[axis] = int(height) + 2 * pad
    grids = np.ogrid[: shape[0], : shape[1], : shape[2]]
    centres = [(s - 1) / 2 for s in shape]
    perpendicular = [k for k in range(3) if k != axis]
    radial = sum((grids[k] - centres[k]) ** 2 for k in perpendicular)
    along = np.abs(grids[axis] - centres[axis]) <= height / 2
    return ((radial <= radius * radius) & along).astype(np.uint32)


def torus_mask(major: float, minor: float, pad: int = 4) -> np.ndarray:
    """Solid torus in the axis-1/axis-2 plane, revolved about array axis 0."""
    n = int(2 * (major + minor)) + 2 * pad
    c = (n - 1) / 2
    z, y, x = np.ogrid[:n, :n, :n]
    q = np.sqrt((x - c) ** 2 + (y - c) ** 2) - major
    return ((q**2 + (z - c) ** 2) <= minor * minor).astype(np.uint32)


def box_mask(side: int, pad: int = 4) -> np.ndarray:
    a = np.zeros((side + 2 * pad,) * 3, np.uint32)
    a[pad : pad + side, pad : pad + side, pad : pad + side] = 1
    return a


# --------------------------------------------------------------------------
# mesh validity
# --------------------------------------------------------------------------


def undirected_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.sort(e, axis=1)


def euler_characteristic(mesh) -> int:
    """V - E + F. Two for a sphere-like surface, zero for a torus."""
    n_edges = len(np.unique(undirected_edges(mesh.faces), axis=0))
    return len(mesh.vertices) - n_edges + len(mesh.faces)


def non_manifold_edge_count(mesh) -> int:
    """Edges not shared by exactly two triangles."""
    if len(mesh.faces) == 0:
        return 0
    _, counts = np.unique(undirected_edges(mesh.faces), axis=0, return_counts=True)
    return int((counts != 2).sum())


def non_manifold_vertex_count(mesh) -> int:
    """Vertices whose link is not a single cycle.

    A vertex is manifold when the triangles around it form one closed fan. Take
    the neighbours of a vertex as nodes and join the two neighbours contributed
    by each incident triangle: the link is a single cycle exactly when that
    graph is connected and every node has degree two.

    This catches pinch points that an edge-only check misses — two cones
    meeting at a single apex have every edge used twice.
    """
    links: dict[int, dict[int, set[int]]] = {}
    for a, b, c in mesh.faces:
        for v, p, q in ((a, b, c), (b, c, a), (c, a, b)):
            link = links.setdefault(int(v), {})
            link.setdefault(int(p), set()).add(int(q))
            link.setdefault(int(q), set()).add(int(p))

    bad = 0
    for link in links.values():
        if any(len(neighbours) != 2 for neighbours in link.values()):
            bad += 1
            continue
        # Walk one cycle and check it covers every node.
        start = next(iter(link))
        seen = {start}
        previous, current = start, next(iter(link[start]))
        while current != start:
            seen.add(current)
            nxt = [n for n in link[current] if n != previous]
            if not nxt:
                break
            previous, current = current, nxt[0]
        if len(seen) != len(link):
            bad += 1
    return bad


def normal_angle_error(mesh, analytic) -> np.ndarray:
    """Per-vertex angle, in degrees, between mesh and analytic normals.

    `analytic` maps an ``(N, 3)`` array of vertex coordinates to unnormalised
    outward normals.
    """
    reference = analytic(mesh.vertices.astype(np.float64))
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    got = mesh.normals.astype(np.float64)
    cos = np.clip(np.einsum("ij,ij->i", reference, got), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def assert_valid_closed_surface(mesh, expected_euler: int = 2) -> None:
    """Every structural property a closed serra mesh must have."""
    assert len(mesh.faces) > 0, "mesh is empty"
    assert non_manifold_edge_count(mesh) == 0, "non-manifold or boundary edges"
    assert non_manifold_vertex_count(mesh) == 0, "non-manifold vertices"
    assert euler_characteristic(mesh) == expected_euler, (
        f"Euler characteristic {euler_characteristic(mesh)}, expected {expected_euler}"
    )
    assert mesh.volume() > 0, "inverted winding: normals point inward"
    assert mesh.faces.max() < len(mesh.vertices), "face index out of range"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sphere20():
    import serra_mesh

    mask = sphere_mask(20)
    mesher = serra_mesh.Mesher().mesh(mask, close=True)
    return mesher.get(1, normals=True), mask
