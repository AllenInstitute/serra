"""Quadric error simplification, and the guarantees it has to keep.

Decimation is where a mesher usually starts producing quietly broken output:
edge collapse can create a pinched vertex or an edge shared by three triangles
without any obvious symptom, and it can erode a chunk's seam so the pieces stop
fitting together. Both are checked here directly rather than assumed.

The three properties that matter:

* **Topology survives.** No non-manifold edges or vertices, and the Euler
  characteristic is unchanged — including for a torus, where a careless
  collapse would close the hole.
* **The seam survives.** Chunks simplified independently still stitch into a
  watertight surface.
* **`max_error` means something.** No vertex ends up further than that from
  where it started.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import (
    euler_characteristic,
    non_manifold_edge_count,
    non_manifold_vertex_count,
    sphere_mask,
    torus_mask,
)

import serra_mesh

HUGE = 1e9  # effectively "no displacement limit"


@pytest.fixture(scope="module")
def sphere():
    return serra_mesh.Mesher().mesh(sphere_mask(16), close=True)


@pytest.fixture(scope="module")
def torus():
    return serra_mesh.Mesher().mesh(torus_mask(20.0, 7.0), close=True)


def nearest_distance(points, reference):
    """Distance from each point to the closest of `reference`, in chunks."""
    worst = 0.0
    for block in np.array_split(np.asarray(points, dtype=np.float64), 16):
        if not len(block):
            continue
        d = np.linalg.norm(block[:, None, :] - reference[None, :, :], axis=-1)
        worst = max(worst, float(d.min(axis=1).max()))
    return worst


class TestTopologyIsPreserved:
    """The core guarantee: simplification cannot break the mesh."""

    @pytest.mark.parametrize("reduction", [2, 5, 10, 50, 100, 1000])
    def test_sphere_stays_a_closed_manifold(self, sphere, reduction):
        mesh = sphere.get(1, reduction_factor=reduction, max_error=HUGE)
        assert non_manifold_edge_count(mesh) == 0
        assert non_manifold_vertex_count(mesh) == 0
        assert euler_characteristic(mesh) == 2
        assert mesh.is_closed()
        assert mesh.volume() > 0, "winding inverted"

    @pytest.mark.parametrize("reduction", [2, 5, 10, 50, 100, 1000])
    def test_genus_survives(self, torus, reduction):
        """A hole is the thing a careless collapse quietly closes."""
        mesh = torus.get(1, reduction_factor=reduction, max_error=HUGE)
        assert non_manifold_edge_count(mesh) == 0
        assert non_manifold_vertex_count(mesh) == 0
        assert euler_characteristic(mesh) == 0, "the torus lost its hole"
        assert mesh.is_closed()

    def test_separate_components_stay_separate(self):
        """Two objects sharing a label must not be welded into one."""
        a = np.zeros((44, 20, 20), np.uint32)
        g = np.indices(a.shape)
        for cx in (11, 33):
            a[((g[0] - cx) ** 2 + (g[1] - 10) ** 2 + (g[2] - 10) ** 2) <= 6**2] = 1
        mesher = serra_mesh.Mesher().mesh(a, close=True)
        mesh = mesher.get(1, reduction_factor=20, max_error=HUGE)
        assert non_manifold_edge_count(mesh) == 0
        assert non_manifold_vertex_count(mesh) == 0
        # Two sphere-like components, so chi = 4 rather than 2.
        assert euler_characteristic(mesh) == 4

    def test_a_thin_sheet_is_not_collapsed_away(self):
        a = np.zeros((20, 20, 20), np.uint32)
        a[10, 4:16, 4:16] = 1
        mesh = (
            serra_mesh.Mesher()
            .mesh(a, close=True)
            .get(1, reduction_factor=100, max_error=HUGE)
        )
        assert mesh.volume() > 0
        assert mesh.is_closed()
        assert non_manifold_vertex_count(mesh) == 0

    def test_faces_index_within_the_vertex_array(self, sphere):
        mesh = sphere.get(1, reduction_factor=50, max_error=HUGE)
        assert mesh.faces.max() < len(mesh.vertices)

    def test_no_degenerate_faces(self, sphere):
        mesh = sphere.get(1, reduction_factor=50, max_error=HUGE)
        f = mesh.faces
        assert not (
            (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
        ).any()
        t = mesh.triangles()
        area = 0.5 * np.linalg.norm(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1
        )
        assert (area > 0).all(), "zero-area triangle produced"


class TestReduction:
    @pytest.mark.parametrize("reduction", [2, 4, 10])
    def test_target_is_approximately_met(self, sphere, reduction):
        full = len(sphere.get(1).faces)
        mesh = sphere.get(1, reduction_factor=reduction, max_error=HUGE)
        assert len(mesh.faces) == pytest.approx(full / reduction, rel=0.05)

    def test_more_reduction_means_fewer_faces(self, sphere):
        counts = [
            len(sphere.get(1, reduction_factor=r, max_error=HUGE).faces)
            for r in (2, 5, 10, 50)
        ]
        assert counts == sorted(counts, reverse=True)

    @pytest.mark.parametrize("reduction", [0, 1])
    def test_no_reduction_requested_is_a_no_op(self, sphere, reduction):
        plain = sphere.get(1)
        mesh = sphere.get(1, reduction_factor=reduction)
        np.testing.assert_array_equal(plain.vertices, mesh.vertices)
        np.testing.assert_array_equal(plain.faces, mesh.faces)

    def test_volume_degrades_gracefully(self, sphere):
        full = sphere.get(1).volume()
        for reduction, tolerance in [(2, 0.01), (10, 0.02), (50, 0.05)]:
            mesh = sphere.get(1, reduction_factor=reduction, max_error=HUGE)
            assert mesh.volume() / full - 1 == pytest.approx(0, abs=tolerance)

    def test_negative_reduction_rejected(self, sphere):
        with pytest.raises(ValueError, match="non-negative"):
            sphere.get(1, reduction_factor=-1)


class TestDisplacementBound:
    """`max_error` is a geometric bound, not just a quadric threshold.

    A quadric error is a sum of squared distances to planes and grows with
    vertex valence, so thresholding it alone would let vertices wander further
    in busy neighbourhoods. The bound is enforced against each vertex's
    original position instead.
    """

    @pytest.mark.parametrize("max_error", [0.5, 1.0, 2.0, 4.0])
    def test_no_vertex_moves_further_than_allowed(self, sphere, max_error):
        base = sphere.get(1).vertices.astype(np.float64)
        mesh = sphere.get(1, reduction_factor=1000, max_error=max_error)
        assert nearest_distance(mesh.vertices, base) <= max_error + 1e-6

    def test_a_tight_bound_prevents_simplification(self, sphere):
        """Adjacent dual vertices are about a voxel apart, so nothing qualifies."""
        full = len(sphere.get(1).faces)
        assert len(sphere.get(1, reduction_factor=1000, max_error=0.1).faces) == full

    def test_a_looser_bound_allows_more(self, sphere):
        counts = [
            len(sphere.get(1, reduction_factor=1000, max_error=e).faces)
            for e in (0.5, 1.0, 2.0, 4.0)
        ]
        assert counts == sorted(counts, reverse=True), counts

    def test_default_bound_is_the_coarsest_voxel(self):
        """Matches zmesh, which defaults to one voxel of the largest axis."""
        mesher = serra_mesh.Mesher(voxel_resolution=[1, 1, 4]).mesh(
            sphere_mask(12), close=True
        )
        default = mesher.get(1, reduction_factor=1000)
        explicit = mesher.get(1, reduction_factor=1000, max_error=4.0)
        assert len(default.faces) == len(explicit.faces)

    def test_negative_bound_rejected(self, sphere):
        with pytest.raises(ValueError, match="non-negative"):
            sphere.get(1, reduction_factor=2, max_error=-1.0)


class TestChunkSeamsSurvive:
    """Chunks simplified independently must still stitch."""

    S, N, HALO = 24, 72, 2

    def volume(self):
        g = np.indices((self.N, self.N, self.N))
        c = (self.N - 1) / 2
        return (((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= 30**2).astype(
            np.uint32
        )

    def decompose(self, volume, reduction):
        pieces = []
        for origin in itertools.product(range(0, self.N, self.S), repeat=3):
            window, owned = [], []
            for k in range(3):
                end = min(origin[k] + self.S, self.N)
                owned.append(end - origin[k])
                window.append(slice(origin[k], min(end + self.HALO, self.N)))
            mesher = serra_mesh.Mesher().mesh(
                volume[tuple(window)], close=False, owned_shape=owned
            )
            if 1 in mesher:
                pieces.append(
                    (
                        mesher.get(1, reduction_factor=reduction, max_error=HUGE),
                        np.array(origin, dtype=float),
                    )
                )
        return pieces

    @pytest.mark.parametrize("reduction", [0, 2, 5, 20])
    def test_stitched_result_stays_watertight(self, reduction):
        joined = serra_mesh.stitch(
            self.decompose(self.volume(), reduction), dedup_faces=False
        )
        assert non_manifold_edge_count(joined) == 0
        assert non_manifold_vertex_count(joined) == 0
        assert joined.is_closed(), f"seam leaked at reduction_factor={reduction}"

    def test_simplification_actually_happened(self):
        """Guards against the seam holding only because nothing was removed."""
        plain = serra_mesh.stitch(self.decompose(self.volume(), 0))
        reduced = serra_mesh.stitch(self.decompose(self.volume(), 5))
        assert len(reduced.faces) < len(plain.faces) / 2

    def test_the_open_rim_of_a_chunk_is_left_alone(self):
        """The rim is what neighbouring chunks weld to, so it must not erode."""
        volume = self.volume()
        window = volume[0 : self.S + self.HALO]
        mesher = serra_mesh.Mesher().mesh(
            window, close=False, owned_shape=[self.S, self.N, self.N]
        )

        def rim(mesh):
            edges = np.sort(
                np.concatenate(
                    [
                        mesh.faces[:, [0, 1]],
                        mesh.faces[:, [1, 2]],
                        mesh.faces[:, [2, 0]],
                    ]
                ),
                axis=1,
            )
            uniq, counts = np.unique(edges, axis=0, return_counts=True)
            border = np.unique(uniq[counts != 2])
            return np.unique(
                np.round(mesh.vertices[border].astype(np.float64) * 256).astype(
                    np.int64
                ),
                axis=0,
            )

        before = rim(mesher.get(1))
        for reduction in (2, 5, 20):
            after = rim(mesher.get(1, reduction_factor=reduction, max_error=HUGE))
            assert before.shape == after.shape and np.array_equal(before, after), (
                f"rim changed at reduction_factor={reduction}"
            )


class TestDeterminism:
    @pytest.mark.parametrize("reduction", [2, 10])
    def test_repeatable(self, sphere, reduction):
        a = sphere.get(1, reduction_factor=reduction, max_error=HUGE)
        b = sphere.get(1, reduction_factor=reduction, max_error=HUGE)
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.faces, b.faces)

    @pytest.mark.parametrize("threads", [1, 4, 0])
    def test_thread_count_does_not_matter(self, threads):
        mesher = serra_mesh.Mesher(threads=threads).mesh(sphere_mask(16), close=True)
        mesh = mesher.get(1, reduction_factor=10, max_error=HUGE)
        reference = (
            serra_mesh.Mesher(threads=1)
            .mesh(sphere_mask(16), close=True)
            .get(1, reduction_factor=10, max_error=HUGE)
        )
        np.testing.assert_array_equal(mesh.vertices, reference.vertices)
        np.testing.assert_array_equal(mesh.faces, reference.faces)


class TestNormals:
    def test_normals_are_recomputed_after_simplification(self, sphere):
        mesh = sphere.get(1, normals=True, reduction_factor=10, max_error=HUGE)
        assert mesh.normals is not None
        assert mesh.normals.shape == mesh.vertices.shape
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)

    def test_normals_still_point_outward(self, sphere):
        mesh = sphere.get(1, normals=True, reduction_factor=20, max_error=HUGE)
        centre = mesh.vertices.astype(np.float64).mean(axis=0)
        radial = mesh.vertices.astype(np.float64) - centre
        dots = np.einsum("ij,ij->i", radial, mesh.normals.astype(np.float64))
        assert (dots > 0).mean() > 0.98
