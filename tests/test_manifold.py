"""Manifoldness, including the configurations that break naive dual contouring.

Every object serra returns must be a 2-manifold surface: no edge shared by more
than two triangles, and no vertex whose surrounding triangles form more than one
fan. The second condition is the hard one — two blocks meeting at a single
corner have every edge used exactly twice, yet the shared vertex is a pinch
point. serra avoids it by splitting a cell's vertex once per 6-connected
component of the label inside that cell.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    assert_valid_closed_surface,
    non_manifold_edge_count,
    non_manifold_vertex_count,
    sphere_mask,
)

import serra_mesh


def mesh_all(mask, close=True):
    mesher = serra_mesh.Mesher().mesh(mask, close=close)
    return {int(i): mesher.get(int(i)) for i in mesher.ids()}


class TestSelfContact:
    def test_corner_contact_is_split(self):
        """Two voxels sharing only a corner: the classic pinch point."""
        a = np.zeros((7, 7, 7), np.uint32)
        a[2, 2, 2] = 1
        a[3, 3, 3] = 1
        mesh = mesh_all(a)[1]
        assert non_manifold_edge_count(mesh) == 0
        assert non_manifold_vertex_count(mesh) == 0
        # Two separate closed cubes: 8 vertices and 6 quads (12 triangles) each.
        assert len(mesh.vertices) == 16
        assert len(mesh.faces) == 24

    def test_edge_contact_is_split(self):
        """Two voxels sharing only an edge — a non-manifold edge if unsplit."""
        a = np.zeros((7, 7, 7), np.uint32)
        a[2, 2, 2] = 1
        a[3, 3, 2] = 1
        mesh = mesh_all(a)[1]
        assert non_manifold_edge_count(mesh) == 0
        assert non_manifold_vertex_count(mesh) == 0
        assert len(mesh.faces) == 24

    def test_diagonal_chain_stays_manifold(self):
        """A staircase of corner-touching voxels: many pinch points at once."""
        a = np.zeros((16, 16, 16), np.uint32)
        for i in range(1, 12):
            a[i, i, i] = 1
        mesh = mesh_all(a)[1]
        assert non_manifold_edge_count(mesh) == 0
        assert non_manifold_vertex_count(mesh) == 0
        # Eleven independent cubes.
        assert len(mesh.faces) == 11 * 6 * 2

    def test_checkerboard_is_the_worst_case(self):
        """Alternating labels put four components in every single cell."""
        n = 9
        idx = np.indices((n, n, n)).sum(axis=0)
        a = (idx % 2 + 1).astype(np.uint32)
        a[0, :, :] = a[-1, :, :] = 0
        a[:, 0, :] = a[:, -1, :] = 0
        a[:, :, 0] = a[:, :, -1] = 0

        meshes = mesh_all(a)
        assert set(meshes) == {1, 2}
        for label, mesh in meshes.items():
            assert non_manifold_edge_count(mesh) == 0, f"label {label}"
            assert non_manifold_vertex_count(mesh) == 0, f"label {label}"
            assert mesh.volume() > 0, f"label {label} inverted"


class TestMultiLabel:
    def test_two_touching_spheres_are_each_closed(self):
        mask = sphere_mask(10)
        n = mask.shape[0]
        a = np.zeros((2 * n, n, n), np.uint32)
        a[:n] = mask
        a[n:] = mask * 2
        meshes = mesh_all(a)
        assert set(meshes) == {1, 2}
        for label, mesh in meshes.items():
            assert non_manifold_edge_count(mesh) == 0, f"label {label} is open"
            assert non_manifold_vertex_count(mesh) == 0, f"label {label}"
            assert mesh.volume() > 0

    def test_three_labels_meeting_along_a_line(self):
        """A triple junction: each object closed, no shared non-manifold edge."""
        a = np.zeros((14, 14, 8), np.uint32)
        centre = 7
        for i in range(1, 13):
            for j in range(1, 13):
                for k in range(1, 7):
                    di, dj = i - centre, j - centre
                    if di * di + dj * dj > 30:
                        continue
                    if di >= 0 and dj >= 0:
                        a[i, j, k] = 1
                    elif di < 0 and dj >= 0:
                        a[i, j, k] = 2
                    else:
                        a[i, j, k] = 3
        meshes = mesh_all(a)
        assert set(meshes) == {1, 2, 3}
        for label, mesh in meshes.items():
            assert non_manifold_edge_count(mesh) == 0, f"label {label}"
            assert non_manifold_vertex_count(mesh) == 0, f"label {label}"
            assert mesh.volume() > 0, f"label {label}"

    def test_a_label_fully_enclosed_by_another(self):
        """An inclusion. The outer shell has two boundary components."""
        a = np.zeros((15, 15, 15), np.uint32)
        a[2:13, 2:13, 2:13] = 1
        a[6:9, 6:9, 6:9] = 2
        meshes = mesh_all(a)
        assert set(meshes) == {1, 2}
        for label, mesh in meshes.items():
            assert non_manifold_edge_count(mesh) == 0, f"label {label}"
            assert non_manifold_vertex_count(mesh) == 0, f"label {label}"
        # The shell encloses the outer box minus the inner one, and its two
        # sphere-like boundary components give a total Euler characteristic of 4.
        from conftest import euler_characteristic

        assert euler_characteristic(meshes[1]) == 4
        assert euler_characteristic(meshes[2]) == 2

    def test_shared_wall_geometry_is_identical_from_both_sides(self):
        """A wall between two labels must be the same surface seen twice."""
        a = np.zeros((12, 9, 9), np.uint32)
        a[1:5, 1:8, 1:8] = 1
        a[5:9, 1:8, 1:8] = 2
        meshes = mesh_all(a)

        def triangle_keys(mesh):
            quantised = np.round(mesh.vertices.astype(np.float64) * 256).astype(
                np.int64
            )
            tris = quantised[mesh.faces]
            tris = np.sort(tris.reshape(len(tris), 3, 3), axis=1)
            return {t.tobytes() for t in tris}

        shared = triangle_keys(meshes[1]) & triangle_keys(meshes[2])
        # The interface spans 7x7 voxel faces: 49 quads, 98 triangles.
        assert len(shared) == 98


class TestOpenSurfaces:
    def test_object_running_off_the_edge_is_open_without_close(self):
        a = np.zeros((8, 8, 8), np.uint32)
        a[0:4, :, :] = 1
        mesh = mesh_all(a, close=False)[1]
        assert non_manifold_edge_count(mesh) > 0, "should be open at the face"

    def test_close_seals_it_and_restores_manifoldness(self):
        a = np.zeros((8, 8, 8), np.uint32)
        a[0:4, :, :] = 1
        mesh = mesh_all(a, close=True)[1]
        assert_valid_closed_surface(mesh, expected_euler=2)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_close_works_on_every_face(self, axis):
        a = np.zeros((9, 9, 9), np.uint32)
        index = [slice(1, 8)] * 3
        index[axis] = slice(0, 4)
        a[tuple(index)] = 1
        assert_valid_closed_surface(mesh_all(a, close=True)[1], expected_euler=2)

    def test_object_filling_the_whole_volume(self):
        """Degenerate but legal: every voxel is the same label."""
        a = np.ones((6, 6, 6), np.uint32)
        assert_valid_closed_surface(mesh_all(a, close=True)[1], expected_euler=2)


class TestBandedExtraction:
    """The seam pass is separate code from the main traversal.

    Extraction is split into bands along one axis, and the quads on each band's
    first cell layer are produced by a distinct routine afterwards. If it got
    the vertex bookkeeping wrong the surface would tear along regularly spaced
    planes, so these check watertightness specifically under threading.
    """

    @pytest.mark.parametrize("threads", [1, 2, 4, 8, 14])
    def test_sphere_stays_closed_however_it_is_banded(self, threads):
        mask = sphere_mask(24)
        mesher = serra_mesh.Mesher(threads=threads).mesh(mask, close=True)
        assert_valid_closed_surface(mesher.get(1), expected_euler=2)

    @pytest.mark.parametrize("threads", [1, 4, 14])
    def test_many_objects_all_stay_closed(self, threads):
        """A tall stack, so several bands cut through every object."""
        a = np.zeros((30, 30, 120), np.uint32)
        grid = np.indices(a.shape)
        for n in range(6):
            centre = (15, 15, 12 + n * 19)
            d = sum((grid[k] - centre[k]) ** 2 for k in range(3))
            a[d <= 8**2] = n + 1
        mesher = serra_mesh.Mesher(threads=threads).mesh(a, close=True)
        assert len(mesher) == 6
        for label in mesher.ids():
            assert_valid_closed_surface(mesher.get(int(label)), expected_euler=2)

    def test_an_object_spanning_every_band_stays_closed(self):
        """A tube running the full length of the banded axis."""
        a = np.zeros((24, 24, 160), np.uint32)
        y, x = np.ogrid[:24, :24]
        disc = (y - 11.5) ** 2 + (x - 11.5) ** 2 <= 7**2
        a[:, :, 4:156][disc] = 1
        for threads in (1, 14):
            mesh = serra_mesh.Mesher(threads=threads).mesh(a, close=True).get(1)
            assert_valid_closed_surface(mesh, expected_euler=2)


class TestKnownSplitLimitation:
    """A cell can carry more surface sheets than the label has components.

    serra splits a cell's vertex once per 6-connected component of the label's
    corners. That is not always enough: if the label is connected but the
    *background* inside the cell is not, the surface passes through the cell as
    two separate sheets, and they end up sharing the single vertex.

    The smallest case is a cell with six label corners whose two background
    corners are body-diagonal (mask 0b11100111). The label is 6-connected, so
    one vertex is emitted, but there is a sheet around each background corner.

    Found on real connectomics data, where it affects 10 of 133 objects in a
    160^3 cutout. The fix is to split per connected component of the crossing
    edges rather than of the label's corners.
    """

    @staticmethod
    def pinch_volume():
        a = np.zeros((8, 8, 8), np.uint32)
        for x, y, z in [
            (1, 0, 1),
            (1, 1, 1),
            (1, 1, 2),
            (2, 0, 1),
            (2, 0, 2),
            (2, 1, 2),
        ]:
            a[x + 2, y + 2, z + 2] = 1
        return a

    def test_the_configuration_is_what_we_think_it_is(self):
        """Guards the diagnosis, so this stays meaningful if the fix changes."""
        a = self.pinch_volume()
        mask = 0
        for c in range(8):
            dx, dy, dz = c & 1, (c >> 1) & 1, (c >> 2) & 1
            if a[3 + dx, 2 + dy, 3 + dz]:
                mask |= 1 << c
        assert mask == 0b11100111
        background = [c for c in range(8) if not (mask >> c) & 1]
        # Body-diagonal: they differ along all three axes, so they are not
        # 6-connected and form two separate background components.
        assert bin(background[0] ^ background[1]).count("1") == 3

    @pytest.mark.xfail(
        reason="splitting is per label component, not per surface sheet",
        strict=True,
    )
    def test_no_pinch_point(self):
        mesh = serra_mesh.Mesher().mesh(self.pinch_volume(), close=True).get(1)
        assert non_manifold_vertex_count(mesh) == 0

    def test_edges_are_still_fine(self):
        """Only the vertex is pinched; no edge is over-used."""
        mesh = serra_mesh.Mesher().mesh(self.pinch_volume(), close=True).get(1)
        assert non_manifold_edge_count(mesh) == 0
