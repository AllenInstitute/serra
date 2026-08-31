"""Joining chunk meshes, and the PyChunkedGraph convention in particular.

The contract serra has to satisfy for a chunked pipeline is stronger than "seam
vertices agree": the chunks must actually reassemble into the mesh you would
have got from meshing the volume in one piece, using nothing but vertex
deduplication. That means no missing faces and no duplicate faces.

Two things make it work, and both are tested here for failure as well as
success:

* **Two voxels of halo, on the positive side only.** A quad dual to a y- or
  z-edge is built from four cells spanning two cell layers, so with one voxel of
  halo the cells at a seam are split between neighbours and the faces there are
  produced by neither. Marching cubes needs only one voxel because it reads a
  single cell per quad, which is why PyChunkedGraph fetches one.
* **`owned_shape`**, which assigns each face to exactly one chunk. Without it
  both sides of a seam emit the wall between them.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import serra_mesh

S = 24  # chunk size in voxels
N = 3 * S  # three chunks per axis
HALO = 2  # positive-side halo required by dual contouring


# --------------------------------------------------------------------------
# volumes
# --------------------------------------------------------------------------


def sphere():
    g = np.indices((N, N, N))
    c = (N - 1) / 2
    return (((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= 30**2).astype(
        np.uint32
    )


def slab_ending_on_a_seam():
    """A wall lying exactly on a chunk boundary — the duplicate-face case.

    Both chunks can compute this wall, so it is what `owned_shape` has to
    arbitrate. A shape whose walls fall anywhere else will not exercise it.
    """
    a = np.zeros((N, N, N), np.uint32)
    a[4 : S + 1, 4 : N - 4, 4 : N - 4] = 1
    return a


def tube():
    """Runs off both ends of the volume, so it is legitimately open."""
    a = np.zeros((N, N, N), np.uint32)
    y, z = np.ogrid[:N, :N]
    a[:, ((y - 35.5) ** 2 + (z - 35.5) ** 2) <= 100] = 1
    return a


def two_labels_meeting_on_a_seam():
    a = np.zeros((N, N, N), np.uint32)
    a[4 : S + 1, 4 : N - 4, 4 : N - 4] = 1
    a[S + 1 : N - 4, 4 : N - 4, 4 : N - 4] = 2
    return a


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def decompose(volume, splits, halo=HALO, owned=True, threads=0, relaxation=0, label=1):
    """Mesh `volume` in chunks, splitting the axes named in `splits`.

    Follows the PyChunkedGraph convention: a chunk fetches its own voxels plus
    `halo` more in the +x, +y, +z directions only, clamped at the volume edge.
    """
    ranges = [list(range(0, N, S)) if k in splits else [0] for k in range(3)]
    pieces = []
    for origin in itertools.product(*ranges):
        window, owned_shape = [], []
        for k in range(3):
            extent = S if k in splits else N
            end = min(origin[k] + extent, N)
            owned_shape.append(end - origin[k])
            window.append(slice(origin[k], min(end + halo, N)))
        mesher = serra_mesh.Mesher(threads=threads, relaxation=relaxation).mesh(
            volume[tuple(window)],
            close=False,
            owned_shape=owned_shape if owned else None,
        )
        if label in mesher:
            pieces.append((mesher.get(label), np.array(origin, dtype=float)))
    return pieces


def whole(volume, relaxation=0, label=1):
    return serra_mesh.Mesher(relaxation=relaxation).mesh(volume, close=False).get(label)


def face_keys(mesh, offset=(0, 0, 0)):
    v = np.round((mesh.vertices.astype(np.float64) + np.asarray(offset)) * 256)
    v = v.astype(np.int64)
    return [t.tobytes() for t in np.sort(v[mesh.faces], axis=1)]


ALL_SPLITS = [({0}, "x"), ({0, 1}, "xy"), ({0, 1, 2}, "xyz")]


# --------------------------------------------------------------------------


class TestChunkedMatchesWhole:
    """With the halo and ownership right, chunks reassemble exactly."""

    @pytest.mark.parametrize("splits,name", ALL_SPLITS)
    @pytest.mark.parametrize(
        "volume", [sphere(), slab_ending_on_a_seam()], ids=["sphere", "slab-on-seam"]
    )
    def test_face_count_and_watertightness(self, volume, splits, name):
        pieces = decompose(volume, splits)
        # dedup_faces=False, because ownership should already guarantee it —
        # this is exactly what PyChunkedGraph's stitcher does.
        joined = serra_mesh.stitch(pieces, dedup_faces=False)
        reference = whole(volume)

        assert len(joined.faces) == len(reference.faces)
        assert joined.count_boundary_edges() == 0
        assert joined.is_closed()

    @pytest.mark.parametrize("splits,name", ALL_SPLITS)
    def test_geometry_is_identical_not_merely_equinumerous(self, splits, name):
        volume = sphere()
        pieces = decompose(volume, splits)
        got = set()
        for mesh, offset in pieces:
            got.update(face_keys(mesh, offset))
        assert got == set(face_keys(whole(volume)))

    def test_an_object_open_at_the_volume_edge_is_reproduced_faithfully(self):
        """A tube running off both ends stays open — and only that open."""
        volume = tube()
        joined = serra_mesh.stitch(decompose(volume, {0, 1, 2}), dedup_faces=False)
        reference = whole(volume)
        assert len(joined.faces) == len(reference.faces)
        # Open at the two volume faces, and nowhere else.
        assert joined.count_boundary_edges() == reference.count_boundary_edges()

    def test_every_label_survives_a_multi_label_volume(self):
        volume = two_labels_meeting_on_a_seam()
        for label in (1, 2):
            pieces = decompose(volume, {0, 1, 2}, label=label)
            joined = serra_mesh.stitch(pieces, dedup_faces=False, id=label)
            assert len(joined.faces) == len(whole(volume, label=label).faces)
            assert joined.is_closed(), f"label {label} not watertight"


class TestOwnershipIsWhatRemovesDuplicates:
    """Without `owned_shape`, both sides of a seam emit the wall between them."""

    def test_no_duplicates_when_ownership_is_declared(self):
        pieces = decompose(slab_ending_on_a_seam(), {0, 1, 2}, owned=True)
        total = sum(len(m.faces) for m, _ in pieces)
        joined = serra_mesh.stitch(pieces, dedup_faces=False)
        assert total == len(joined.faces), "ownership should leave no duplicates"

    def test_duplicates_appear_without_ownership(self):
        pieces = decompose(slab_ending_on_a_seam(), {0, 1, 2}, owned=False)
        total = sum(len(m.faces) for m, _ in pieces)
        vertex_only = serra_mesh.stitch(pieces, dedup_faces=False)
        assert total > len(whole(slab_ending_on_a_seam()).faces)
        assert vertex_only.count_boundary_edges() > 0, (
            "a wall counted twice should show up as non-manifold"
        )

    def test_face_dedup_repairs_the_unowned_case(self):
        volume = slab_ending_on_a_seam()
        pieces = decompose(volume, {0, 1, 2}, owned=False)
        repaired = serra_mesh.stitch(pieces, dedup_faces=True)
        assert len(repaired.faces) == len(whole(volume).faces)
        assert repaired.is_closed()


class TestHaloWidth:
    """One voxel of halo is enough for marching cubes and not for serra."""

    @pytest.mark.parametrize(
        "volume", [sphere(), slab_ending_on_a_seam()], ids=["sphere", "slab-on-seam"]
    )
    def test_one_voxel_of_halo_loses_faces(self, volume):
        pieces = decompose(volume, {0}, halo=1, owned=False)
        joined = serra_mesh.stitch(pieces, dedup_faces=True)
        reference = whole(volume)
        assert len(joined.faces) < len(reference.faces), (
            "expected a hole along the seam"
        )
        assert joined.count_boundary_edges() > 0

    @pytest.mark.parametrize(
        "volume", [sphere(), slab_ending_on_a_seam()], ids=["sphere", "slab-on-seam"]
    )
    def test_two_voxels_of_halo_is_sufficient(self, volume):
        joined = serra_mesh.stitch(decompose(volume, {0}, halo=2), dedup_faces=False)
        assert len(joined.faces) == len(whole(volume).faces)
        assert joined.is_closed()

    def test_a_wider_halo_is_harmless(self):
        volume = sphere()
        for halo in (2, 3, 5):
            joined = serra_mesh.stitch(
                decompose(volume, {0}, halo=halo), dedup_faces=False
            )
            assert len(joined.faces) == len(whole(volume).faces), f"halo={halo}"
            assert joined.is_closed(), f"halo={halo}"

    def test_owning_more_than_the_array_holds_is_rejected(self):
        with pytest.raises(ValueError, match="cannot own"):
            serra_mesh.Mesher().mesh(
                np.zeros((10, 10, 10), np.uint32), owned_shape=[20, 10, 10]
            )

    def test_owned_shape_must_have_three_entries(self):
        with pytest.raises(ValueError, match="3 entries"):
            serra_mesh.Mesher().mesh(
                np.zeros((10, 10, 10), np.uint32), owned_shape=[5, 5]
            )


class TestUnderRelaxation:
    """Relaxation pins the seam cell layer, so stitching is unaffected."""

    @pytest.mark.parametrize("k", [0, 3, 10])
    def test_stitch_stays_watertight(self, k):
        volume = sphere()
        joined = serra_mesh.stitch(
            decompose(volume, {0, 1, 2}, relaxation=k), dedup_faces=False
        )
        assert joined.is_closed(), f"relaxation={k} broke the seam"

    @pytest.mark.parametrize("k", [3, 10])
    def test_relaxation_still_changes_the_interior(self, k):
        """Guards against the seam holding only because nothing moved."""
        volume = sphere()
        plain = serra_mesh.stitch(decompose(volume, {0, 1, 2}, relaxation=0))
        smooth = serra_mesh.stitch(decompose(volume, {0, 1, 2}, relaxation=k))
        assert smooth.area() < plain.area() * 0.995


class TestThreadIndependence:
    @pytest.mark.parametrize("threads", [1, 2, 4, 0])
    def test_chunked_output_does_not_depend_on_thread_count(self, threads):
        volume = sphere()
        joined = serra_mesh.stitch(
            decompose(volume, {0, 1, 2}, threads=threads), dedup_faces=False
        )
        assert len(joined.faces) == len(whole(volume).faces)
        assert joined.is_closed()


class TestStitchHelper:
    def test_empty_input(self):
        result = serra_mesh.stitch([])
        assert len(result.faces) == 0 and len(result.vertices) == 0

    def test_single_piece_round_trips(self):
        mesh = whole(sphere())
        joined = serra_mesh.stitch([(mesh, (0, 0, 0))])
        assert len(joined.faces) == len(mesh.faces)
        assert len(joined.vertices) == len(mesh.vertices)

    def test_offsets_are_applied(self):
        mesh = whole(sphere())
        joined = serra_mesh.stitch([(mesh, (100.0, 0, 0))])
        assert joined.vertices[:, 0].min() == pytest.approx(
            mesh.vertices[:, 0].min() + 100.0
        )

    def test_id_is_recorded(self):
        assert serra_mesh.stitch([(whole(sphere()), (0, 0, 0))], id=7).id == 7
