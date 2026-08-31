"""The optional relaxation pass.

Relaxation smooths the surface while holding the outermost layer of cells fixed.
Pinning that layer is what keeps the halo at one voxel no matter how many
iterations run: a cell in the last layer has neighbours in the layer beyond,
which the chunk does not contain, so those are exactly the vertices whose
one-ring is incomplete. Everything else has a complete neighbourhood, so the
whole pass depends only on data the chunk already holds.

The consequence, deliberately accepted: a chunk's interior smooths slightly more
than the band around its seams, so a stitched surface is self-consistent and
watertight but not identical to the same volume meshed in one piece.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    assert_valid_closed_surface,
    normal_angle_error,
    sphere_mask,
    torus_mask,
)

import serra_mesh

RADIUS = 20.0


@pytest.fixture(scope="module")
def sphere():
    return sphere_mask(RADIUS), (int(2 * RADIUS) + 8 - 1) / 2


def relaxed(mask, k, **kwargs):
    mesher = serra_mesh.Mesher(relaxation=k, **kwargs).mesh(mask, close=True)
    return mesher.get(1, normals=True)


class TestQuality:
    """Relaxation should improve the two metrics local placement cannot."""

    def test_area_error_shrinks_with_iterations(self, sphere):
        mask, _ = sphere
        exact = 4 * np.pi * RADIUS**2
        errors = [abs(relaxed(mask, k).area() / exact - 1) for k in (0, 1, 2, 3)]
        assert errors == sorted(errors, reverse=True), f"not monotone: {errors}"
        # +2.95% unrelaxed down to roughly +0.24% by k=3.
        assert errors[0] > 0.02
        assert errors[-1] < 0.01

    def test_normal_error_converges(self, sphere):
        """Unlike area under local placement, this genuinely goes to zero."""
        mask, centre = sphere
        means = [
            normal_angle_error(relaxed(mask, k), lambda v: v - centre).mean()
            for k in (0, 3, 10, 25)
        ]
        assert means == sorted(means, reverse=True), f"not monotone: {means}"
        assert means[0] > 7.0  # ~8.1 degrees
        assert means[-1] < 2.0  # ~1.0 degrees

    def test_triangle_areas_get_more_uniform(self, sphere):
        mask, _ = sphere

        def spread(mesh):
            t = mesh.triangles()
            a = 0.5 * np.linalg.norm(
                np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1
            )
            return a.std() / a.mean()

        assert spread(relaxed(mask, 10)) < spread(relaxed(mask, 0))

    def test_zero_iterations_changes_nothing(self, sphere):
        mask, _ = sphere
        plain = serra_mesh.Mesher().mesh(mask, close=True).get(1)
        explicit = relaxed(mask, 0)
        np.testing.assert_array_equal(plain.vertices, explicit.vertices)
        np.testing.assert_array_equal(plain.faces, explicit.faces)


class TestStructureIsPreserved:
    @pytest.mark.parametrize("k", [1, 3, 10, 50])
    def test_sphere_stays_a_valid_closed_surface(self, sphere, k):
        mask, _ = sphere
        assert_valid_closed_surface(relaxed(mask, k), expected_euler=2)

    @pytest.mark.parametrize("k", [3, 25])
    def test_genus_is_preserved(self, k):
        """Smoothing must not close the hole in a torus."""
        assert_valid_closed_surface(relaxed(torus_mask(25.0, 8.0), k), expected_euler=0)

    @pytest.mark.parametrize("k", [3, 25])
    def test_thin_features_survive(self, k):
        """A one-voxel-thick sheet is the easiest thing to smooth away."""
        a = np.zeros((20, 20, 20), np.uint32)
        a[10, 4:16, 4:16] = 1
        mesh = serra_mesh.Mesher(relaxation=k).mesh(a, close=True).get(1)
        assert mesh.volume() > 0, "sheet collapsed or inverted"
        assert_valid_closed_surface(mesh, expected_euler=2)


class TestDeviationBound:
    """`max_deviation` is the guarantee that the surface tracks the data."""

    def _max_per_axis_shift(self, mask, k, max_deviation):
        base = (
            serra_mesh.Mesher()
            .mesh(mask, close=True)
            .get(1)
            .vertices.astype(np.float64)
        )
        moved = (
            serra_mesh.Mesher(relaxation=k, max_deviation=max_deviation)
            .mesh(mask, close=True)
            .get(1)
            .vertices.astype(np.float64)
        )
        # Vertex order is not stable across settings (the quad diagonal choice
        # depends on position), so compare each vertex with its nearest original.
        worst = 0.0
        for chunk in np.array_split(moved, 16):
            dist = np.linalg.norm(chunk[:, None, :] - base[None, :, :], axis=-1)
            nearest = dist.argmin(1)
            worst = max(worst, float(np.abs(chunk - base[nearest]).max()))
        return worst

    @pytest.mark.parametrize("max_deviation", [0.1, 0.25, 0.5])
    def test_no_vertex_moves_further_than_allowed(self, sphere, max_deviation):
        mask, _ = sphere
        shift = self._max_per_axis_shift(mask, 50, max_deviation)
        # One 1/256-voxel of slack for the final quantisation back to fixed point.
        assert shift <= max_deviation + 1.0 / 256 + 1e-6, f"moved {shift}"

    def test_tighter_bounds_reduce_shrinkage(self, sphere):
        """Laplacian smoothing shrinks; the bound is what limits it."""
        mask, _ = sphere
        exact = 4 / 3 * np.pi * RADIUS**3
        losses = [
            abs(relaxed(mask, 25, max_deviation=md).volume() / exact - 1)
            for md in (0.1, 0.25, 0.5)
        ]
        assert losses == sorted(losses), f"not monotone: {losses}"
        assert losses[0] < 0.02

    def test_zero_deviation_pins_everything(self, sphere):
        mask, _ = sphere
        plain = serra_mesh.Mesher().mesh(mask, close=True).get(1)
        frozen = relaxed(mask, 25, max_deviation=0.0)
        np.testing.assert_array_equal(plain.vertices, frozen.vertices)


class TestValidation:
    def test_negative_iterations_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            serra_mesh.Mesher(relaxation=-1)

    def test_negative_deviation_rejected(self):
        with pytest.raises(ValueError, match="finite and non-negative"):
            serra_mesh.Mesher(max_deviation=-0.1)

    @pytest.mark.parametrize("step", [0.0, -0.5, 1.5, float("nan")])
    def test_step_outside_the_unit_interval_rejected(self, step):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            serra_mesh.Mesher(relaxation=1, relaxation_step=step)

    def test_repr_mentions_relaxation(self):
        assert "relaxation=3" in repr(serra_mesh.Mesher(relaxation=3))


class TestDeterminism:
    @pytest.mark.parametrize("k", [1, 3, 25])
    def test_repeatable(self, sphere, k):
        mask, _ = sphere
        first = relaxed(mask, k)
        second = relaxed(mask, k)
        np.testing.assert_array_equal(first.vertices, second.vertices)
        np.testing.assert_array_equal(first.faces, second.faces)

    @pytest.mark.parametrize("k", [3, 25])
    def test_memory_order_does_not_matter(self, sphere, k):
        mask, _ = sphere
        c_order = np.ascontiguousarray(mask)
        f_order = np.asfortranarray(mask)
        np.testing.assert_array_equal(
            relaxed(c_order, k).vertices, relaxed(f_order, k).vertices
        )
