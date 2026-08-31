"""The chunked-meshing contract.

serra is meant to mesh a large volume in pieces. That works only if two
neighbouring chunks compute *bit-identical* vertices along their shared seam, so
the pieces can be joined by deduplicating vertices with no geometric tolerance
and no repair pass.

The contract: each chunk owns a disjoint range of voxels and is passed with a
**one-voxel halo on every side**, so neighbouring input arrays overlap by two
voxels. That is what makes the layer of dual cells straddling the seam present
in both chunks, and therefore computed identically by both.

Exact comparison is legitimate here because vertex coordinates are derived from
integers in units of 1/256 of a voxel — a power of two, so the conversion to
float32 is lossless at these magnitudes. Nothing depends on floating-point
association order.
"""

from __future__ import annotations

import numpy as np
import pytest

import serra_mesh

# Chunk A owns voxels [0, SPLIT); chunk B owns [SPLIT, size).
SPLIT = 32


def tube_volume(size=64, cross=24, radius=7.0):
    """A tube running along axis 0, clear of the outer volume boundary."""
    a = np.zeros((size, cross, cross), np.uint32)
    centre = (cross - 1) / 2
    y, z = np.ogrid[:cross, :cross]
    disc = (y - centre) ** 2 + (z - centre) ** 2 <= radius * radius
    for i in range(2, size - 2):
        a[i][disc] = 1
    return a


def blobs_volume(size=64, cross=28):
    """Several objects, some straddling the seam, some not."""
    a = np.zeros((size, cross, cross), np.uint32)
    grid = np.indices((size, cross, cross))
    for n, (cx, label) in enumerate([(16, 3), (SPLIT, 7), (48, 11)]):
        d = (
            (grid[0] - cx) ** 2
            + (grid[1] - cross // 2) ** 2
            + (grid[2] - cross // 2) ** 2
        )
        a[d <= 9**2] = label
    return a


def meshed_vertices(array, axis, offset, label):
    """Vertices of one object, shifted back into global coordinates."""
    mesher = serra_mesh.Mesher().mesh(array, close=False)
    if label not in mesher:
        return np.zeros((0, 3), np.float64)
    v = mesher.get(label).vertices.astype(np.float64).copy()
    v[:, axis] += offset
    return v


def quantise(vertices):
    """Exact integer keys, sorted, for set comparison."""
    if len(vertices) == 0:
        return np.zeros((0, 3), np.int64)
    keys = np.round(vertices * 256).astype(np.int64)
    return np.unique(keys, axis=0)


def seam_band(vertices, axis, lo, hi):
    """Vertices produced by the dual cell layer straddling the seam.

    That layer has minimum corner at ``SPLIT - 1``, so its vertices lie between
    ``SPLIT - 1`` and ``SPLIT``.
    """
    if len(vertices) == 0:
        return quantise(vertices)
    inside = (vertices[:, axis] >= lo) & (vertices[:, axis] <= hi)
    return quantise(vertices[inside])


def split_with_halo(array, axis, halo):
    """Slice into two chunks, each carrying `halo` voxels past its own range."""
    size = array.shape[axis]
    lower = [slice(None)] * 3
    upper = [slice(None)] * 3
    lower[axis] = slice(0, min(SPLIT + halo, size))
    upper[axis] = slice(max(SPLIT - halo, 0), size)
    return array[tuple(lower)], array[tuple(upper)], max(SPLIT - halo, 0)


class TestSeamAgreement:
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_seam_vertices_are_bit_identical_across_chunks(self, axis):
        """The core guarantee, on every axis."""
        array = np.moveaxis(tube_volume(), 0, axis).copy()
        a_chunk, b_chunk, b_offset = split_with_halo(array, axis, halo=1)

        a_seam = seam_band(meshed_vertices(a_chunk, axis, 0, 1), axis, SPLIT - 1, SPLIT)
        b_seam = seam_band(
            meshed_vertices(b_chunk, axis, b_offset, 1), axis, SPLIT - 1, SPLIT
        )

        assert len(a_seam) > 0, "no vertices found on the seam"
        assert a_seam.shape == b_seam.shape
        assert np.array_equal(a_seam, b_seam), "seam vertices differ between chunks"

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_seam_vertices_match_the_whole_volume(self, axis):
        """Agreement with ground truth, not just with each other."""
        array = np.moveaxis(tube_volume(), 0, axis).copy()
        a_chunk, _, _ = split_with_halo(array, axis, halo=1)

        chunked = seam_band(
            meshed_vertices(a_chunk, axis, 0, 1), axis, SPLIT - 1, SPLIT
        )
        whole = seam_band(meshed_vertices(array, axis, 0, 1), axis, SPLIT - 1, SPLIT)

        assert len(whole) > 0
        assert np.array_equal(chunked, whole)

    def test_agreement_holds_for_several_objects_at_once(self):
        array = blobs_volume()
        a_chunk, b_chunk, b_offset = split_with_halo(array, 0, halo=1)

        # Label 7 is centred on the seam, so it is the one that must match.
        a_seam = seam_band(meshed_vertices(a_chunk, 0, 0, 7), 0, SPLIT - 1, SPLIT)
        b_seam = seam_band(
            meshed_vertices(b_chunk, 0, b_offset, 7), 0, SPLIT - 1, SPLIT
        )
        whole = seam_band(meshed_vertices(array, 0, 0, 7), 0, SPLIT - 1, SPLIT)

        assert len(whole) > 0
        assert np.array_equal(a_seam, b_seam)
        assert np.array_equal(a_seam, whole)

    def test_a_wider_halo_also_works(self):
        """Extra halo is harmless — the owned region is unaffected."""
        array = tube_volume()
        for halo in (1, 2, 4):
            a_chunk, b_chunk, b_offset = split_with_halo(array, 0, halo=halo)
            a_seam = seam_band(meshed_vertices(a_chunk, 0, 0, 1), 0, SPLIT - 1, SPLIT)
            b_seam = seam_band(
                meshed_vertices(b_chunk, 0, b_offset, 1), 0, SPLIT - 1, SPLIT
            )
            assert np.array_equal(a_seam, b_seam), f"halo={halo}"


class TestInsufficientOverlap:
    def test_one_voxel_overlap_leaves_the_seam_cell_layer_missing(self):
        """Why the halo is required, demonstrated rather than asserted.

        With chunks overlapping by only one voxel, no dual cell straddles the
        seam: the lower chunk's last cell ends where the upper chunk's first
        cell begins. Neither produces the layer of vertices in between, so the
        joined surface would have a ring-shaped hole.
        """
        array = tube_volume()
        a_chunk = array[0:SPLIT]  # owns [0, 32), no halo
        b_chunk = array[SPLIT - 1 :]  # one voxel of overlap only

        a_seam = seam_band(meshed_vertices(a_chunk, 0, 0, 1), 0, SPLIT - 1, SPLIT)
        b_seam = seam_band(
            meshed_vertices(b_chunk, 0, SPLIT - 1, 1), 0, SPLIT - 1, SPLIT
        )

        assert len(a_seam) == 0, "lower chunk should reach no further than the seam"
        assert len(b_seam) > 0
        assert not np.array_equal(a_seam, b_seam)

    def test_the_halo_is_what_fixes_it(self):
        """The same volume, same split, one extra voxel of halo."""
        array = tube_volume()
        a_chunk, b_chunk, b_offset = split_with_halo(array, 0, halo=1)
        a_seam = seam_band(meshed_vertices(a_chunk, 0, 0, 1), 0, SPLIT - 1, SPLIT)
        b_seam = seam_band(
            meshed_vertices(b_chunk, 0, b_offset, 1), 0, SPLIT - 1, SPLIT
        )
        assert len(a_seam) > 0
        assert np.array_equal(a_seam, b_seam)


class TestSeamIndependenceFromContext:
    def test_a_chunk_is_unaffected_by_data_beyond_its_halo(self):
        """Placement is local, so distant voxels cannot move a seam vertex.

        This is the property that makes the guarantee hold in a real pipeline,
        where neighbouring chunks contain wholly different objects.
        """
        array = tube_volume()
        a_chunk, _, _ = split_with_halo(array, 0, halo=1)
        baseline = seam_band(meshed_vertices(a_chunk, 0, 0, 1), 0, SPLIT - 1, SPLIT)

        # Add unrelated objects outside the chunk's halo entirely.
        noisy = array.copy()
        noisy[SPLIT + 4 : SPLIT + 12, 4:12, 4:12] = 99
        noisy_chunk, _, _ = split_with_halo(noisy, 0, halo=1)
        perturbed = seam_band(
            meshed_vertices(noisy_chunk, 0, 0, 1), 0, SPLIT - 1, SPLIT
        )

        assert np.array_equal(baseline, perturbed)


def relaxed_seam(chunk, offset, k, axis=0):
    """Seam-band vertices of a chunk meshed with `k` relaxation iterations."""
    mesher = serra_mesh.Mesher(relaxation=k).mesh(chunk, close=False)
    if 1 not in mesher:
        return np.zeros((0, 3), np.int64)
    v = mesher.get(1).vertices.astype(np.float64).copy()
    v[:, axis] += offset
    return seam_band(v, axis, SPLIT - 1, SPLIT)


class TestSeamsUnderRelaxation:
    """Relaxation must not widen the halo.

    Iterative smoothing normally propagates one cell per iteration, so k
    iterations would need k+1 voxels of halo for chunks to agree. serra instead
    pins the outermost layer of cells — precisely the vertices whose one-ring
    the chunk does not fully contain — so relaxation never reads past the halo
    and one voxel stays sufficient at any k.
    """

    @pytest.mark.parametrize("k", [0, 1, 3, 10, 50])
    def test_seam_stays_bit_identical_at_any_iteration_count(self, k):
        array = tube_volume()
        a_chunk, b_chunk, b_offset = split_with_halo(array, 0, halo=1)
        a_seam = relaxed_seam(a_chunk, 0, k)
        b_seam = relaxed_seam(b_chunk, b_offset, k)

        assert len(a_seam) > 0
        assert np.array_equal(a_seam, b_seam), f"seam diverged at relaxation={k}"

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_holds_on_every_axis(self, axis):
        array = np.moveaxis(tube_volume(), 0, axis).copy()
        a_chunk, b_chunk, b_offset = split_with_halo(array, axis, halo=1)
        a_seam = relaxed_seam(a_chunk, 0, 10, axis=axis)
        b_seam = relaxed_seam(b_chunk, b_offset, 10, axis=axis)
        assert len(a_seam) > 0
        assert np.array_equal(a_seam, b_seam)

    @pytest.mark.parametrize("k", [3, 25])
    def test_a_chunk_is_still_a_pure_function_of_its_own_array(self, k):
        """The property that makes the guarantee usable in a real pipeline.

        Objects sitting outside a chunk's halo must not perturb its mesh at all
        — not merely its seam. Without pinning, relaxation would pull in
        information from k cells away and this would fail.
        """
        array = tube_volume()
        a_chunk, _, _ = split_with_halo(array, 0, halo=1)
        baseline = serra_mesh.Mesher(relaxation=k).mesh(a_chunk, close=False).get(1)

        noisy = array.copy()
        noisy[SPLIT + 4 : SPLIT + 14, 3:12, 3:12] = 99
        noisy_chunk, _, _ = split_with_halo(noisy, 0, halo=1)
        perturbed = (
            serra_mesh.Mesher(relaxation=k).mesh(noisy_chunk, close=False).get(1)
        )

        np.testing.assert_array_equal(baseline.vertices, perturbed.vertices)
        np.testing.assert_array_equal(baseline.faces, perturbed.faces)

    def test_relaxation_does_change_the_interior(self):
        """Guards against the guarantee holding trivially.

        If relaxation were silently doing nothing, every test above would pass.
        """
        array = tube_volume()
        a_chunk, _, _ = split_with_halo(array, 0, halo=1)
        plain = serra_mesh.Mesher(relaxation=0).mesh(a_chunk, close=False).get(1)
        smooth = serra_mesh.Mesher(relaxation=10).mesh(a_chunk, close=False).get(1)
        # Same connectivity, but the surface is measurably smaller in area.
        assert smooth.area() < plain.area() * 0.99

    @pytest.mark.parametrize("k", [3, 25])
    def test_seam_vertices_are_exactly_the_unrelaxed_ones(self, k):
        """Pinned means pinned: the seam layer keeps its local placement."""
        array = tube_volume()
        a_chunk, _, _ = split_with_halo(array, 0, halo=1)
        assert np.array_equal(relaxed_seam(a_chunk, 0, k), relaxed_seam(a_chunk, 0, 0))
