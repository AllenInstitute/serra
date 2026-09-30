"""Tangential fairing: `fairing_tangential`.

These sweeps move each cell toward its neighbour average, but only within the
surface, so triangles even out without the shape changing. Three things are
asserted:

* **It does its job.** Triangle areas get more uniform and slivers go away.
* **It does nothing else.** Volume, area and distance from the true surface are
  unchanged, and no vertex strays past ``max_deviation``.
* **It respects the multi-material structure.** Walls shared by two labels stay
  coincident, vertices on a curve where three labels meet stay on that curve,
  and seams, threads and stitching behave exactly as for the other filters.
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
    """Area coefficient of variation, and mean shape quality.

    Quality is 4*sqrt(3)*area / (sum of squared edge lengths): 1 for an
    equilateral triangle, 0 for a degenerate one.
    """
    p = mesh.vertices[mesh.faces].astype(np.float64)
    edges = [p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(edges[0], -edges[2]), axis=1)
    l2 = sum((e * e).sum(1) for e in edges)
    return area.std() / area.mean(), (4 * np.sqrt(3) * area / l2).mean()


def nearest_distance(points, reference):
    """Distance from each point to the closest of `reference`.

    Vertex order is not comparable between two meshes: a quad is split along its
    shorter diagonal, so moving vertices changes the triangulation and with it
    the emission order. Brute force, because the fixtures are small.
    """
    out = np.empty(len(points))
    for i in range(0, len(points), 512):
        block = points[i : i + 512]
        d = np.linalg.norm(block[:, None, :] - reference[None, :, :], axis=2)
        out[i : i + 512] = d.min(axis=1)
    return out


def radial_error(mesh, centre, radius):
    return np.abs(np.linalg.norm(mesh.vertices - centre, axis=1) - radius)


# --------------------------------------------------------------------------
# it does its job
# --------------------------------------------------------------------------


def test_triangles_become_more_uniform():
    mask = sphere_mask(20.0)
    plain_cv, plain_q = quality(meshed(mask))
    cv, q = quality(meshed(mask, fairing_tangential=5))
    # Measured: 0.260 -> 0.228 and 0.902 -> 0.922. Not more, because one vertex
    # per cell fixes how many triangles cover each patch of surface; see
    # docs/accuracy.md.
    assert cv < 0.92 * plain_cv
    assert q > plain_q + 0.01


def test_it_also_improves_an_already_faired_surface():
    mask = sphere_mask(20.0)
    faired = dict(fairing=20, fairing_taubin=True)
    before, _ = quality(meshed(mask, **faired))
    after, _ = quality(meshed(mask, fairing_tangential=10, **faired))
    assert after < before


# --------------------------------------------------------------------------
# it does nothing else
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "faired", [{}, {"fairing": 20, "fairing_taubin": True}, {"fairing": 10}]
)
def test_the_shape_does_not_change(faired):
    mask = sphere_mask(20.0)
    before = meshed(mask, **faired)
    after = meshed(mask, fairing_tangential=10, **faired)
    assert after.volume() == pytest.approx(before.volume(), rel=1e-3)
    assert after.area() == pytest.approx(before.area(), rel=5e-3)

    n = mask.shape[0]
    centre = np.full(3, (n - 1) / 2)
    # Distance from the true sphere, before and after: moving only within the
    # surface must not move the surface away from the truth.
    assert (
        radial_error(after, centre, 20.0).mean()
        <= radial_error(before, centre, 20.0).mean() + 0.005
    )


def test_it_is_still_a_valid_closed_surface():
    assert_valid_closed_surface(meshed(sphere_mask(12.0), fairing_tangential=10))


@pytest.mark.parametrize("max_deviation", [0.0, 0.125, 0.5])
def test_no_vertex_strays_past_the_deviation_bound(max_deviation):
    mask = sphere_mask(10.0)
    plain = meshed(mask)
    moved = meshed(mask, fairing_tangential=20, max_deviation=max_deviation)
    drift = nearest_distance(moved.vertices, plain.vertices).max()
    # The bound is per axis, so the worst case in Euclidean distance is the
    # diagonal of the cube it defines. One fixed-point unit of slack for the
    # rounding back to 1/256 of a voxel.
    assert drift <= np.sqrt(3) * (max_deviation + 1.0 / 256) + 1e-6


def test_zero_sweeps_changes_nothing():
    mask = sphere_mask(10.0)
    a = meshed(mask, fairing=5)
    b = meshed(mask, fairing=5, fairing_tangential=0)
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.faces, b.faces)


def test_it_runs_on_its_own():
    mask = sphere_mask(10.0)
    assert not np.array_equal(
        meshed(mask).vertices, meshed(mask, fairing_tangential=3).vertices
    )


# --------------------------------------------------------------------------
# multi-material structure
# --------------------------------------------------------------------------


def three_labels(radius=14.0, pad=4):
    """A sphere cut in half by a plane: labels 1 and 2, inside background 3.

    The two halves share a flat wall, and all three labels meet on the circle
    where the plane cuts the sphere.
    """
    n = int(2 * radius) + 2 * pad
    c = (n - 1) / 2
    z, y, x = np.ogrid[:n, :n, :n]
    inside = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= radius * radius
    a = np.full((n, n, n), 3, np.uint32)
    # Array axis 0 is the mesh's x with the default axis order.
    a[inside & (np.arange(n)[:, None, None] < c)] = 1
    a[inside & (np.arange(n)[:, None, None] >= c)] = 2
    return a, c, radius


def triangle_keys(mesh):
    """Each triangle as the sorted bytes of its three corners."""
    corners = mesh.vertices[mesh.faces]
    return {b"".join(sorted(t.tobytes() for t in tri)) for tri in corners}


def test_shared_walls_stay_coincident():
    """Every triangle of the wall between labels 1 and 2 is in both meshes.

    Checked through positions rather than indices: the two meshes number their
    vertices independently. Because tangential sweeps move each cell once, for
    every label there, the two copies of a wall cannot diverge.
    """
    a, _, _ = three_labels()
    for kwargs in [{}, {"fairing_tangential": 10}]:
        mesher = serra_mesh.Mesher(**kwargs).mesh(a, close=True)
        one, two = mesher.get(1), mesher.get(2)
        shared = triangle_keys(one) & triangle_keys(two)
        # The wall is a disc of radius 14, some 600 voxels of area.
        assert len(shared) > 600, kwargs
        # The background is a shell: the volume's outer face and the sphere's
        # cavity, so two components.
        for label, euler in ((1, 2), (2, 2), (3, 4)):
            assert_valid_closed_surface(mesher.get(label), expected_euler=euler)


def junction_vertices(mesher):
    """Vertices present in all three labels' meshes: the junction circle."""
    sets = [
        {row.tobytes() for row in mesher.get(label).vertices} for label in (1, 2, 3)
    ]
    common = sets[0] & sets[1] & sets[2]
    return np.array([np.frombuffer(k, np.float32) for k in common], dtype=np.float64)


def test_junction_vertices_stay_on_the_junction():
    """The three-label curve is a circle in the plane x = c, of radius r.

    A sheet move there would drag those vertices off the curve into one of the
    walls meeting at it. Moving only along the curve keeps them where they
    were, within the smoothing of the curve itself.
    """
    a, c, r = three_labels()
    before = junction_vertices(serra_mesh.Mesher().mesh(a, close=True))
    after = junction_vertices(
        serra_mesh.Mesher(fairing_tangential=10).mesh(a, close=True)
    )
    assert len(before) > 50 and len(after) == len(before)

    def off_curve(p):
        # Distance from the circle x = c, (y - c)^2 + (z - c)^2 = r^2.
        axial = p[:, 0] - c
        radial = np.hypot(p[:, 1] - c, p[:, 2] - c) - r
        return np.hypot(axial, radial)

    assert off_curve(after).mean() <= off_curve(before).mean() + 0.01
    assert off_curve(after).max() <= off_curve(before).max() + 0.05

    # And they do move, along the curve: their spacing around the circle evens
    # out (measured 0.20 -> 0.12). Holding every junction vertex still would
    # pass the checks above, so this is what shows the curve rule is working.
    def spacing_cv(p):
        angle = np.sort(np.arctan2(p[:, 2] - c, p[:, 1] - c))
        gaps = np.diff(np.r_[angle, angle[0] + 2 * np.pi])
        return gaps.std() / gaps.mean()

    assert spacing_cv(after) < 0.8 * spacing_cv(before)


# --------------------------------------------------------------------------
# chunking and determinism, as for every other filter
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
            pieces.append((mesher.get(1), np.array(origin, dtype=float)))
    return pieces


@pytest.mark.parametrize("splits,name", [({0}, "x"), ({0, 1, 2}, "xyz")])
@pytest.mark.parametrize("faired", [{}, {"fairing": 8, "fairing_taubin": True}])
def test_chunks_still_stitch_watertight(splits, name, faired):
    volume = chunked_sphere()
    kwargs = dict(fairing_tangential=5, **faired)
    joined = serra_mesh.stitch(decompose(volume, splits, **kwargs), dedup_faces=False)
    reference = serra_mesh.Mesher(**kwargs).mesh(volume, close=False).get(1)
    assert len(joined.faces) == len(reference.faces)
    assert joined.count_boundary_edges() == 0
    assert joined.is_closed()


def test_output_is_identical_whatever_the_thread_count():
    a, _, _ = three_labels(radius=18.0)
    kwargs = dict(fairing=4, fairing_tangential=6)
    one = serra_mesh.Mesher(threads=1, **kwargs).mesh(a, close=True)
    many = serra_mesh.Mesher(threads=4, **kwargs).mesh(a, close=True)
    for label in (1, 2, 3):
        assert np.array_equal(one.get(label).vertices, many.get(label).vertices)
        assert np.array_equal(one.get(label).faces, many.get(label).faces)


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("other", [{"relaxation": 3}, {"taubin": 3}])
def test_it_counts_as_fairing_for_mutual_exclusion(other):
    with pytest.raises(ValueError, match="only one of relaxation, taubin or fairing"):
        serra_mesh.Mesher(fairing_tangential=3, **other)


def test_negative_sweeps_are_rejected():
    with pytest.raises(ValueError, match="fairing_tangential must be non-negative"):
        serra_mesh.Mesher(fairing_tangential=-1)


def test_the_step_is_checked_when_only_tangential_runs():
    with pytest.raises(ValueError, match="fairing_step"):
        serra_mesh.Mesher(fairing_tangential=3, fairing_step=0.0)
