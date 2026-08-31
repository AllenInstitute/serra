"""The Python API surface: construction, dtypes, retrieval, serialization."""

import numpy as np
import pytest

import serra


@pytest.fixture
def cube():
    """A 3x3x3 solid label sitting well inside a 7^3 volume."""
    a = np.zeros((7, 7, 7), np.uint32)
    a[2:5, 2:5, 2:5] = 1
    return a


@pytest.fixture
def meshed(cube):
    return serra.Mesher(voxel_resolution=[1, 1, 1]).mesh(cube, close=True)


def test_version_is_exposed():
    assert isinstance(serra.__version__, str)
    assert serra.__version__.count(".") == 2


class TestMesherConstruction:
    def test_defaults_to_unit_voxels(self):
        assert serra.Mesher().voxel_resolution.tolist() == [1.0, 1.0, 1.0]

    @pytest.mark.parametrize("resolution", [[1, 2], [1, 2, 3, 4]])
    def test_rejects_wrong_length_resolution(self, resolution):
        with pytest.raises(ValueError, match="3 entries"):
            serra.Mesher(voxel_resolution=resolution)

    @pytest.mark.parametrize("resolution", [[0, 1, 1], [-1, 1, 1], [np.nan, 1, 1]])
    def test_rejects_non_positive_resolution(self, resolution):
        with pytest.raises(ValueError, match="finite and positive"):
            serra.Mesher(voxel_resolution=resolution)

    @pytest.mark.parametrize(
        "order,message",
        [("XY", "exactly 3"), ("XYW", "only contain"), ("XXY", "repeats")],
    )
    def test_rejects_bad_axis_order(self, order, message):
        with pytest.raises(ValueError, match=message):
            serra.Mesher(axis_order=order)

    def test_accepts_every_axis_permutation(self):
        for order in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
            assert serra.Mesher(axis_order=order) is not None

    def test_repr_is_clean(self):
        text = repr(serra.Mesher(voxel_resolution=[4, 4, 40]))
        assert "4.0, 4.0, 40.0" in text
        # numpy scalars must not leak into the representation.
        assert "np.float64" not in text


class TestMeshInput:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_all_unsigned_dtypes_agree(self, cube, dtype):
        reference = serra.Mesher().mesh(cube, close=True).get(1)
        got = serra.Mesher().mesh(cube.astype(dtype), close=True).get(1)
        np.testing.assert_array_equal(reference.vertices, got.vertices)
        np.testing.assert_array_equal(reference.faces, got.faces)

    def test_c_and_fortran_order_are_identical(self, cube):
        c_order = np.ascontiguousarray(cube)
        f_order = np.asfortranarray(cube)
        assert f_order.flags.f_contiguous and not f_order.flags.c_contiguous

        a = serra.Mesher().mesh(c_order, close=True).get(1)
        b = serra.Mesher().mesh(f_order, close=True).get(1)
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.faces, b.faces)

    def test_rejects_signed_labels_with_a_useful_hint(self, cube):
        with pytest.raises(TypeError, match="view"):
            serra.Mesher().mesh(cube.astype(np.int32))

    def test_rejects_float_labels(self, cube):
        with pytest.raises(TypeError, match="unsigned"):
            serra.Mesher().mesh(cube.astype(np.float32))

    @pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5, 5)])
    def test_rejects_wrong_dimensionality(self, shape):
        with pytest.raises(ValueError, match="3-D"):
            serra.Mesher().mesh(np.zeros(shape, np.uint32))

    def test_mesh_returns_self_for_chaining(self, cube):
        mesher = serra.Mesher()
        assert mesher.mesh(cube, close=True) is mesher

    def test_remeshing_replaces_previous_result(self, cube):
        mesher = serra.Mesher().mesh(cube, close=True)
        assert mesher.ids().tolist() == [1]
        other = np.zeros((7, 7, 7), np.uint32)
        other[3, 3, 3] = 42
        mesher.mesh(other, close=True)
        assert mesher.ids().tolist() == [42]


class TestRetrieval:
    def test_ids_are_sorted_and_exclude_background(self):
        a = np.zeros((11, 5, 5), np.uint32)
        for n, label in enumerate([98_340_797, 7, 4_000_000_000, 12]):
            a[2 * n + 1, 2, 2] = label
        ids = serra.Mesher().mesh(a, close=True).ids()
        assert ids.tolist() == [7, 12, 98_340_797, 4_000_000_000]

    def test_ids_is_empty_before_meshing(self):
        assert serra.Mesher().ids().tolist() == []

    def test_get_before_mesh_is_an_error(self):
        with pytest.raises(ValueError, match="mesh\\(\\) first"):
            serra.Mesher().get(1)

    def test_missing_label_raises_key_error(self, meshed):
        with pytest.raises(KeyError, match="not present"):
            meshed.get(999)

    def test_arrays_have_the_documented_dtypes(self, meshed):
        mesh = meshed.get(1, normals=True)
        assert mesh.vertices.dtype == np.float32
        assert mesh.faces.dtype == np.uint32
        assert mesh.normals.dtype == np.float32
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        assert mesh.normals.shape == mesh.vertices.shape
        assert mesh.id == 1

    def test_normals_are_omitted_by_default(self, meshed):
        assert meshed.get(1).normals is None

    def test_normals_are_unit_length(self, meshed):
        normals = meshed.get(1, normals=True).normals
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)

    def test_faces_index_within_the_vertex_array(self, meshed):
        mesh = meshed.get(1)
        assert mesh.faces.max() < len(mesh.vertices)

    def test_get_all_yields_every_object_in_order(self):
        a = np.zeros((11, 5, 5), np.uint32)
        for n, label in enumerate([5, 3, 9]):
            a[2 * n + 1, 2, 2] = label
        mesher = serra.Mesher().mesh(a, close=True)
        assert [m.id for m in mesher.get_all()] == [3, 5, 9]

    def test_container_protocol(self, meshed):
        assert len(meshed) == 1
        assert 1 in meshed
        assert 999 not in meshed
        assert list(meshed) == [1]

    def test_erase_and_clear(self, meshed):
        assert meshed.erase(1) is True
        assert meshed.erase(1) is False
        assert len(meshed) == 0

        meshed.clear()
        assert len(meshed) == 0


class TestGeometry:
    def test_a_solid_cube_is_closed(self, meshed):
        mesh = meshed.get(1)
        assert mesh.is_closed()
        assert mesh.count_boundary_edges() == 0

    @pytest.mark.parametrize("n", [3, 5, 10, 20, 40])
    def test_cube_volume_follows_the_bevel_law(self, n):
        """Dual contouring bevels sharp convex edges, and by a known amount.

        A cube's vertices sit on the voxel-centre boundary, so an ideal
        reconstruction would enclose exactly n^3. Placing each cell's vertex at
        the centroid of its edge crossings instead rounds the twelve sharp
        edges, losing exactly ``3n - 2`` of volume — an edge effect, not a
        surface or volume one, so the *relative* error falls off as 1/n^2.

        This is the price of a smooth surface, and it is why marching cubes
        reproduces an axis-aligned box exactly while overstating a sphere's
        area by 9%. For the rounded biological shapes serra targets it is
        negligible: a sphere comes out within 0.06%.
        """
        a = np.zeros((n + 4,) * 3, np.uint32)
        a[2 : 2 + n, 2 : 2 + n, 2 : 2 + n] = 1
        volume = serra.Mesher().mesh(a, close=True).get(1).volume()
        assert volume == pytest.approx(n**3 - (3 * n - 2), abs=0.05)

    def test_volume_is_positive_meaning_normals_point_outward(self, meshed):
        assert meshed.get(1).volume() > 0

    def test_anisotropic_resolution_scales_volume(self, cube):
        iso = serra.Mesher(voxel_resolution=[1, 1, 1]).mesh(cube, close=True).get(1)
        aniso = serra.Mesher(voxel_resolution=[4, 4, 40]).mesh(cube, close=True).get(1)
        assert aniso.volume() == pytest.approx(iso.volume() * 4 * 4 * 40, rel=1e-4)

    @pytest.mark.parametrize("order", ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
    def test_every_axis_order_keeps_normals_outward(self, cube, order):
        mesh = serra.Mesher(axis_order=order).mesh(cube, close=True).get(1)
        assert mesh.volume() > 0, f"{order} turned the object inside out"

    def test_y_down_keeps_normals_outward(self, cube):
        mesh = serra.Mesher(y_down=True).mesh(cube, close=True).get(1)
        assert mesh.volume() > 0

    def test_without_close_a_touching_object_is_open(self):
        a = np.zeros((5, 5, 5), np.uint32)
        a[0:2, :, :] = 1
        assert not serra.Mesher().mesh(a).get(1).is_closed()
        assert serra.Mesher().mesh(a, close=True).get(1).is_closed()


class TestSerialization:
    def test_precomputed_round_trip(self, meshed):
        mesh = meshed.get(1)
        restored = serra.Mesh.from_precomputed(mesh.to_precomputed(), id=1)
        np.testing.assert_array_equal(mesh.vertices, restored.vertices)
        np.testing.assert_array_equal(mesh.faces, restored.faces)
        assert restored.id == 1

    def test_ply_header_and_size(self, meshed):
        mesh = meshed.get(1)
        blob = mesh.to_ply()
        assert blob.startswith(b"ply\nformat binary_little_endian 1.0\n")
        assert f"element vertex {len(mesh.vertices)}".encode() in blob
        header_end = blob.index(b"end_header\n") + len(b"end_header\n")
        expected = len(mesh.vertices) * 12 + len(mesh.faces) * 13
        assert len(blob) - header_end == expected

    def test_obj_is_one_indexed(self, meshed):
        text = meshed.get(1).to_obj().decode()
        faces = [line for line in text.splitlines() if line.startswith("f ")]
        assert len(faces) == len(meshed.get(1).faces)
        assert min(int(tok) for line in faces for tok in line.split()[1:]) >= 1

    def test_save_rejects_unknown_extension(self, meshed, tmp_path):
        with pytest.raises(ValueError, match="unknown mesh extension"):
            meshed.get(1).save(str(tmp_path / "mesh.xyz"))

    @pytest.mark.parametrize("suffix", [".ply", ".obj"])
    def test_save_writes_a_file(self, meshed, tmp_path, suffix):
        path = tmp_path / f"mesh{suffix}"
        meshed.get(1).save(str(path))
        assert path.stat().st_size > 0

    def test_mesh_repr_and_nbytes(self, meshed):
        mesh = meshed.get(1)
        assert "Mesh(id=1" in repr(mesh)
        assert mesh.nbytes == mesh.vertices.nbytes + mesh.faces.nbytes
        assert not mesh.is_empty()
        assert mesh.triangles().shape == (len(mesh.faces), 3, 3)
