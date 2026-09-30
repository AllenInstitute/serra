"""Edge flips: `edge_flips`.

Flipping replaces the shared edge of two triangles with the other diagonal of
their quad when that improves the worse triangle. No vertex moves. What has to
hold:

* **It helps**, most where it can: on anisotropic voxels, where a triangle's
  shape depends on which diagonal it was cut along.
* **It keeps a valid surface**: closed, manifold, same Euler characteristic,
  volume nearly unchanged.
* **Both copies of a shared wall flip identically**, so touching objects still
  meet exactly. Each label's mesh is flipped separately, so this is the
  property most worth testing.
* **Chunks still stitch**, and output does not depend on thread count.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import assert_valid_closed_surface, sphere_mask

import serra_mesh


def meshed(mask, label=1, **kwargs):
    return serra_mesh.Mesher(**kwargs).mesh(mask, close=True).get(label)


def quality(mesh):
    """Mean shape quality and fraction of triangles with an angle under 20 deg."""
    p = mesh.vertices[mesh.faces].astype(np.float64)
    edges = [p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(edges[0], -edges[2]), axis=1)
    l2 = sum((e * e).sum(1) for e in edges)
    lengths = np.sort(np.stack([np.linalg.norm(e, axis=1) for e in edges], 1), 1)
    sin_min = 2 * area / (lengths[:, 1] * lengths[:, 2])
    theta = np.degrees(np.arcsin(np.clip(sin_min, 0, 1)))
    return (4 * np.sqrt(3) * area / l2).mean(), (theta < 20).mean()


def triangle_keys(mesh):
    """Each triangle as the sorted bytes of its three corners."""
    corners = mesh.vertices[mesh.faces]
    return {b"".join(sorted(t.tobytes() for t in tri)) for tri in corners}


ANISO = dict(voxel_resolution=[4, 4, 40], fairing=20, fairing_taubin=True)


# --------------------------------------------------------------------------
# it helps
# --------------------------------------------------------------------------


def test_it_improves_triangles_on_anisotropic_voxels():
    mask = sphere_mask(16.0)
    q0, thin0 = quality(meshed(mask, **ANISO))
    q1, thin1 = quality(meshed(mask, edge_flips=3, **ANISO))
    assert q1 > q0 + 0.05
    assert thin1 < 0.8 * thin0


def test_it_never_makes_the_worst_triangle_worse():
    """Each flip improves the worse of its two triangles, so the minimum over
    the whole mesh cannot drop."""
    mask = sphere_mask(16.0)

    def worst(mesh):
        p = mesh.vertices[mesh.faces].astype(np.float64)
        edges = [p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(edges[0], -edges[2]), axis=1)
        return (4 * np.sqrt(3) * area / sum((e * e).sum(1) for e in edges)).min()

    assert worst(meshed(mask, edge_flips=3, **ANISO)) >= worst(meshed(mask, **ANISO))


# --------------------------------------------------------------------------
# a valid surface, nearly the same shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{"edge_flips": 3}, {"edge_flips": 3, **ANISO}, {"edge_flips": 3, "fairing": 10}],
)
def test_it_is_still_a_valid_closed_surface(kwargs):
    assert_valid_closed_surface(meshed(sphere_mask(12.0), **kwargs))


def test_vertices_and_face_count_are_unchanged():
    mask = sphere_mask(12.0)
    a = meshed(mask, **ANISO)
    b = meshed(mask, edge_flips=3, **ANISO)
    assert len(a.faces) == len(b.faces)
    # get() emits vertices in first-use order, which flips can change, so
    # compare the sets.
    assert {r.tobytes() for r in a.vertices} == {r.tobytes() for r in b.vertices}


def test_the_surface_does_not_get_less_accurate():
    """Flipping a pair that is not quite flat trims or adds the thin tetrahedron
    between its two diagonals, so volume does move a little. It must not move
    away from the truth: judged against the exact sphere, the error in volume
    and the mean distance of the triangles from the surface may not grow.
    """
    radius = 16.0
    mask = sphere_mask(radius)
    res = np.array(ANISO["voxel_resolution"], dtype=float)
    true_volume = 4 / 3 * np.pi * radius**3 * res.prod()
    centre = (mask.shape[0] - 1) / 2 * res

    def radial_error(mesh):
        # Triangle centroids, back in voxels so every axis counts equally.
        c = mesh.vertices[mesh.faces].astype(np.float64).mean(axis=1)
        return np.abs(np.linalg.norm((c - centre) / res, axis=1) - radius).mean()

    a = meshed(mask, **ANISO)
    b = meshed(mask, edge_flips=3, **ANISO)
    assert abs(b.volume() - true_volume) <= abs(a.volume() - true_volume)
    assert radial_error(b) <= radial_error(a) + 1e-3
    # And it is small in absolute terms.
    assert b.volume() == pytest.approx(a.volume(), rel=5e-3)


def test_zero_passes_changes_nothing():
    mask = sphere_mask(10.0)
    a = meshed(mask, **ANISO)
    b = meshed(mask, edge_flips=0, **ANISO)
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.faces, b.faces)


# --------------------------------------------------------------------------
# multi-material: both copies of a wall agree
# --------------------------------------------------------------------------


def three_labels(radius=14.0, pad=4):
    """A sphere cut in half by a slanted plane: labels 1 and 2, inside 3.

    Slanted so the shared wall is not grid-aligned and has something to flip.
    """
    n = int(2 * radius) + 2 * pad
    c = (n - 1) / 2
    z, y, x = np.ogrid[:n, :n, :n]
    inside = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= radius * radius
    side = (np.arange(n)[:, None, None] - c) + 0.6 * (y - c) + 0.3 * (z - c) < 0
    a = np.full((n, n, n), 3, np.uint32)
    a[inside & side] = 1
    a[inside & ~side] = 2
    return a


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voxel_resolution": [4, 4, 40]},
        ANISO,
        {"fairing": 20, "fairing_taubin": True, "fairing_tangential": 5},
    ],
)
def test_shared_walls_stay_coincident(kwargs):
    """Every wall triangle of label 1 facing label 2 is also one of label 2's.

    Checked through positions, since each mesh numbers its vertices itself. Run
    with and without flips, and the wall must be the same size both ways and
    must actually have changed, or the test would pass by flipping nothing.
    """
    a = three_labels()
    walls = {}
    for flips in (0, 3):
        mesher = serra_mesh.Mesher(edge_flips=flips, **kwargs).mesh(a, close=True)
        one, two, three = (mesher.get(label) for label in (1, 2, 3))
        k1, k2, k3 = triangle_keys(one), triangle_keys(two), triangle_keys(three)
        # Label 1's triangles are each on a wall with label 2 or label 3.
        assert k1 <= k2 | k3
        walls[flips] = k1 & k2
        # Every triangle of 1 not shared with 2 must be shared with 3, and
        # vice versa: nothing is left belonging to a single label.
        assert (k1 - k2) <= k3 and (k2 - k1) <= k3
        for label, euler in ((1, 2), (2, 2), (3, 4)):
            assert_valid_closed_surface(mesher.get(label), expected_euler=euler)
    assert len(walls[3]) == len(walls[0]) > 300
    assert walls[3] != walls[0], "no flip happened on the shared wall"


# --------------------------------------------------------------------------
# chunking and determinism
# --------------------------------------------------------------------------


S = 24
N = 3 * S
HALO = 2


def chunked_sphere():
    g = np.indices((N, N, N))
    c = (N - 1) / 2
    return (((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= 30**2).astype(
        np.uint32
    )


def decompose(volume, splits, **mesher_kwargs):
    """Mesh in chunks, PyChunkedGraph style: positive-only halo, owned_shape."""
    ranges = [list(range(0, N, S)) if k in splits else [0] for k in range(3)]
    pieces = []
    for origin in itertools.product(*ranges):
        window, owned = [], []
        for k in range(3):
            extent = S if k in splits else N
            end = min(origin[k] + extent, N)
            owned.append(end - origin[k])
            window.append(slice(origin[k], min(end + HALO, N)))
        mesher = serra_mesh.Mesher(**mesher_kwargs).mesh(
            volume[tuple(window)], close=False, owned_shape=owned
        )
        if 1 in mesher:
            offset = np.array(origin, dtype=float) * mesher.voxel_resolution
            pieces.append((mesher.get(1), offset))
    return pieces


@pytest.mark.parametrize("splits,name", [({0}, "x"), ({0, 1, 2}, "xyz")])
def test_chunks_still_stitch_watertight(splits, name):
    volume = chunked_sphere()
    kwargs = dict(edge_flips=3, **ANISO)
    joined = serra_mesh.stitch(decompose(volume, splits, **kwargs), dedup_faces=False)
    reference = serra_mesh.Mesher(**kwargs).mesh(volume, close=False).get(1)
    assert len(joined.faces) == len(reference.faces)
    assert joined.count_boundary_edges() == 0
    assert joined.is_closed()


def test_output_is_identical_whatever_the_thread_count():
    a = three_labels(radius=18.0)
    one = serra_mesh.Mesher(threads=1, edge_flips=3, **ANISO).mesh(a, close=True)
    many = serra_mesh.Mesher(threads=4, edge_flips=3, **ANISO).mesh(a, close=True)
    for label in (1, 2, 3):
        assert np.array_equal(one.get(label).vertices, many.get(label).vertices)
        assert np.array_equal(one.get(label).faces, many.get(label).faces)


def test_it_composes_with_simplification():
    mesh = serra_mesh.Mesher(edge_flips=3, **ANISO).mesh(sphere_mask(16.0), close=True)
    reduced = mesh.get(1, reduction_factor=4)
    assert_valid_closed_surface(reduced)


def test_negative_passes_are_rejected():
    with pytest.raises(ValueError, match="edge_flips must be non-negative"):
        serra_mesh.Mesher(edge_flips=-1)
