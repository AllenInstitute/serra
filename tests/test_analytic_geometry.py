"""Shapes whose volume, area and normals are known analytically.

Tolerances here are not arbitrary — they come from measured behaviour, and each
one is justified where it is asserted. Two systematic biases matter:

* **Area is overstated by about 3%** on smooth surfaces, and does not shrink
  with resolution. Placing a cell's vertex at the centroid of its edge
  crossings leaves the surface slightly faceted. (Marching cubes is far worse
  here: about +9%, also non-converging.)
* **Sharp convex edges are bevelled**, costing volume in proportion to edge
  length. Smooth shapes are unaffected; a cylinder's two flat end rims are not.

Both are properties of purely local vertex placement, which is what buys exact
chunk seams. The optional relaxation pass trades halo width for improvement on
both counts.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    assert_valid_closed_surface,
    cylinder_mask,
    ellipsoid_mask,
    normal_angle_error,
    sphere_mask,
    torus_mask,
)

import serra

# Smooth closed surfaces sit in this band; see the module docstring.
SMOOTH_AREA_BAND = (0.0, 0.04)


def mesh_of(mask, resolution=(1, 1, 1), normals=False):
    mesher = serra.Mesher(voxel_resolution=list(resolution)).mesh(mask, close=True)
    return mesher.get(1, normals=normals)


class TestSphere:
    @pytest.mark.parametrize("radius", [8, 16, 32])
    def test_structure(self, radius):
        assert_valid_closed_surface(mesh_of(sphere_mask(radius)), expected_euler=2)

    @pytest.mark.parametrize("radius", [8, 16, 32])
    def test_volume_is_essentially_exact(self, radius):
        mesh = mesh_of(sphere_mask(radius))
        exact = 4 / 3 * np.pi * radius**3
        assert mesh.volume() / exact - 1 == pytest.approx(0, abs=0.005)

    @pytest.mark.parametrize("radius", [8, 16, 32])
    def test_area_bias_is_small_and_positive(self, radius):
        mesh = mesh_of(sphere_mask(radius))
        exact = 4 * np.pi * radius**2
        error = mesh.area() / exact - 1
        low, high = SMOOTH_AREA_BAND
        assert low < error < high, f"r={radius}: area error {error:+.4f}"

    def test_area_bias_does_not_grow_with_resolution(self):
        """The bias is systematic, not divergent.

        Marching cubes shows the same non-convergence at a much larger
        magnitude; this test pins serra's down so a regression that reintroduces
        staircasing is caught.
        """
        errors = []
        for radius in (8, 16, 32):
            mesh = mesh_of(sphere_mask(radius))
            errors.append(mesh.area() / (4 * np.pi * radius**2) - 1)
        assert max(errors) - min(errors) < 0.01, f"area error drifting: {errors}"

    @pytest.mark.parametrize("radius", [8, 16])
    def test_normals_point_along_the_radius(self, radius):
        mask = sphere_mask(radius)
        centre = (mask.shape[0] - 1) / 2
        mesh = mesh_of(mask, normals=True)
        angles = normal_angle_error(mesh, lambda v: v - centre)
        # Measured mean is ~8 degrees, p95 ~17. Faceting of the dual surface,
        # not a directional error: no vertex is anywhere near backwards.
        assert angles.mean() < 10.0, f"mean normal error {angles.mean():.2f} deg"
        assert np.percentile(angles, 95) < 20.0
        assert angles.max() < 45.0, "a normal is wildly misoriented"


class TestEllipsoid:
    SEMI_AXES = (12.0, 20.0, 30.0)  # array-axis order

    def test_structure(self):
        mask, _, _ = ellipsoid_mask(self.SEMI_AXES)
        assert_valid_closed_surface(mesh_of(mask), expected_euler=2)

    def test_volume(self):
        mask, semi, _ = ellipsoid_mask(self.SEMI_AXES)
        exact = 4 / 3 * np.pi * semi[0] * semi[1] * semi[2]
        assert mesh_of(mask).volume() / exact - 1 == pytest.approx(0, abs=0.005)

    def test_area(self):
        mask, semi, _ = ellipsoid_mask(self.SEMI_AXES)
        a, b, c = semi
        # Thomsen's approximation, itself accurate to ~1%, so the band is
        # widened to accommodate it.
        p = 1.6075
        exact = (
            4 * np.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)
        )
        error = mesh_of(mask).area() / exact - 1
        assert -0.01 < error < 0.05, f"area error {error:+.4f}"

    def test_normals_follow_the_analytic_gradient(self):
        mask, semi, centre = ellipsoid_mask(self.SEMI_AXES)
        mesh = mesh_of(mask, normals=True)

        def gradient(v):
            return np.stack(
                [(v[:, k] - centre) / semi[k] ** 2 for k in range(3)], axis=1
            )

        angles = normal_angle_error(mesh, gradient)
        assert angles.mean() < 10.0, f"mean normal error {angles.mean():.2f} deg"
        assert np.percentile(angles, 95) < 20.0


class TestCylinder:
    RADIUS, HEIGHT = 15.0, 40.0

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_structure_is_the_same_along_every_axis(self, axis):
        mask = cylinder_mask(self.RADIUS, self.HEIGHT, axis)
        assert_valid_closed_surface(mesh_of(mask), expected_euler=2)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_volume(self, axis):
        mask = cylinder_mask(self.RADIUS, self.HEIGHT, axis)
        # The mask keeps voxels within height/2 of centre inclusive, so it spans
        # height + 1 voxels along the axis.
        exact = np.pi * self.RADIUS**2 * (self.HEIGHT + 1)
        error = mesh_of(mask).volume() / exact - 1
        # Looser than the sphere: the two flat end rims are sharp circular
        # edges, and bevelling them costs volume.
        assert abs(error) < 0.025, f"axis={axis}: volume error {error:+.4f}"

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_area(self, axis):
        mask = cylinder_mask(self.RADIUS, self.HEIGHT, axis)
        height = self.HEIGHT + 1
        exact = 2 * np.pi * self.RADIUS * height + 2 * np.pi * self.RADIUS**2
        error = mesh_of(mask).area() / exact - 1
        # Here the +3% smooth-surface bias and the rim bevel work against each
        # other, landing near zero.
        assert abs(error) < 0.03, f"axis={axis}: area error {error:+.4f}"

    def test_all_three_orientations_agree(self):
        """A cylinder is a cylinder however it is laid out in the array."""
        volumes, areas = [], []
        for axis in (0, 1, 2):
            mesh = mesh_of(cylinder_mask(self.RADIUS, self.HEIGHT, axis))
            volumes.append(mesh.volume())
            areas.append(mesh.area())
        assert max(volumes) - min(volumes) < 1e-3 * max(volumes)
        assert max(areas) - min(areas) < 1e-3 * max(areas)


class TestTorus:
    MAJOR, MINOR = 25.0, 8.0

    def test_genus_one_surface_has_euler_characteristic_zero(self):
        """The topology check that matters: a hole must survive meshing."""
        mesh = mesh_of(torus_mask(self.MAJOR, self.MINOR))
        assert_valid_closed_surface(mesh, expected_euler=0)

    def test_volume(self):
        mesh = mesh_of(torus_mask(self.MAJOR, self.MINOR))
        exact = 2 * np.pi**2 * self.MAJOR * self.MINOR**2
        assert mesh.volume() / exact - 1 == pytest.approx(0, abs=0.01)

    def test_area(self):
        mesh = mesh_of(torus_mask(self.MAJOR, self.MINOR))
        exact = 4 * np.pi**2 * self.MAJOR * self.MINOR
        error = mesh.area() / exact - 1
        low, high = SMOOTH_AREA_BAND
        assert low < error < high, f"area error {error:+.4f}"


class TestAnisotropy:
    def test_physical_ellipsoid_on_an_anisotropic_grid(self):
        """A sphere in physical space sampled on 4x4x40 nm voxels.

        This is the real connectomics case: the voxel grid is anisotropic, and
        the mesh must come out with the right physical volume, which only works
        if resolution is applied correctly per axis.
        """
        resolution = (4.0, 4.0, 40.0)
        semi = (200.0, 160.0, 400.0)  # physical nanometres, array-axis order
        shape = [int(2 * semi[k] / resolution[k]) + 8 for k in range(3)]
        grids = np.ogrid[: shape[0], : shape[1], : shape[2]]
        centres = [(s - 1) / 2 for s in shape]
        q = sum(
            ((grids[k] - centres[k]) * resolution[k] / semi[k]) ** 2 for k in range(3)
        )
        mask = (q <= 1).astype(np.uint32)

        mesh = mesh_of(mask, resolution=resolution)
        assert_valid_closed_surface(mesh, expected_euler=2)
        exact = 4 / 3 * np.pi * semi[0] * semi[1] * semi[2]
        assert mesh.volume() / exact - 1 == pytest.approx(0, abs=0.01)

    def test_scaling_a_single_axis_scales_volume_proportionally(self):
        mask = sphere_mask(12)
        base = mesh_of(mask).volume()
        for axis in range(3):
            resolution = [1.0, 1.0, 1.0]
            resolution[axis] = 7.0
            scaled = mesh_of(mask, resolution=resolution).volume()
            assert scaled == pytest.approx(base * 7.0, rel=1e-4)
