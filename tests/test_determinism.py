"""Bit-exact reproducibility.

Identical input must give byte-identical output regardless of memory order,
dtype width, thread count, or how many times it is run. This is not a nicety:
a chunked pipeline joins meshes by exact vertex equality, and any dependence on
incidental execution details would break it.

zmesh fails this — its own suite skips a golden-file comparison off macOS
because `std::unordered_map` iterates in a different order there, which changes
the output. serra emits labels in sorted order for exactly that reason.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import serra_mesh


def multilabel_volume(shape=(24, 20, 18), labels=6, seed=0):
    """A volume with several interleaved objects and plenty of boundary."""
    rng = np.random.default_rng(seed)
    grid = np.indices(shape).astype(np.float64)
    a = np.zeros(shape, np.uint32)
    for n in range(labels):
        # Placed per axis, so the helper works for thin volumes too rather than
        # assuming every axis is comfortably larger than the radius.
        centre = np.array([rng.uniform(0.25 * s, 0.75 * s) for s in shape])
        radius = rng.uniform(3.0, 6.0)
        d = sum((grid[k] - centre[k]) ** 2 for k in range(3))
        a[d <= radius**2] = n + 1
    return a


def fingerprint(mesher):
    """A hash covering every label's full geometry."""
    digest = hashlib.sha256()
    for label in mesher.ids():
        mesh = mesher.get(int(label), normals=True)
        digest.update(np.uint64(label).tobytes())
        digest.update(mesh.vertices.tobytes())
        digest.update(mesh.faces.tobytes())
        digest.update(mesh.normals.tobytes())
    return digest.hexdigest()


def mesh_and_fingerprint(array, **kwargs):
    return fingerprint(serra_mesh.Mesher(**kwargs).mesh(array, close=True))


class TestRepeatability:
    def test_same_input_twice(self):
        a = multilabel_volume()
        assert mesh_and_fingerprint(a) == mesh_and_fingerprint(a)

    def test_reusing_one_mesher(self):
        a = multilabel_volume()
        mesher = serra_mesh.Mesher()
        first = fingerprint(mesher.mesh(a, close=True))
        second = fingerprint(mesher.mesh(a, close=True))
        assert first == second

    def test_get_is_pure(self):
        """Repeated retrieval of the same label must not drift."""
        mesher = serra_mesh.Mesher().mesh(multilabel_volume(), close=True)
        label = int(mesher.ids()[0])
        a = mesher.get(label, normals=True)
        b = mesher.get(label, normals=True)
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.faces, b.faces)
        np.testing.assert_array_equal(a.normals, b.normals)

    def test_erasing_one_label_does_not_change_the_others(self):
        a = multilabel_volume()
        mesher = serra_mesh.Mesher().mesh(a, close=True)
        ids = [int(i) for i in mesher.ids()]
        before = {i: mesher.get(i).vertices.copy() for i in ids[1:]}
        mesher.erase(ids[0])
        for i in ids[1:]:
            np.testing.assert_array_equal(before[i], mesher.get(i).vertices)


class TestMemoryLayout:
    def test_c_and_fortran_order_agree(self):
        a = multilabel_volume()
        c_order = np.ascontiguousarray(a)
        f_order = np.asfortranarray(a)
        assert f_order.flags.f_contiguous and not f_order.flags.c_contiguous
        assert mesh_and_fingerprint(c_order) == mesh_and_fingerprint(f_order)

    def test_a_non_contiguous_view_agrees(self):
        """A strided slice of a larger array, neither C- nor F-contiguous."""
        big = multilabel_volume(shape=(48, 40, 36))
        view = big[::2, ::2, ::2]
        assert not view.flags.c_contiguous and not view.flags.f_contiguous
        packed = np.ascontiguousarray(view)
        assert mesh_and_fingerprint(view) == mesh_and_fingerprint(packed)

    @pytest.mark.parametrize(
        "slicer",
        [
            lambda a: a[::-1],
            lambda a: a[:, ::-1],
            lambda a: a[::-1, ::-1, ::-1],
        ],
        ids=["axis0", "axis1", "all-axes"],
    )
    def test_a_reversed_view_agrees_with_its_copy(self, slicer):
        """Negative strides — where zero-copy array views usually break."""
        view = slicer(multilabel_volume())
        assert min(view.strides) < 0
        assert mesh_and_fingerprint(view) == mesh_and_fingerprint(
            np.ascontiguousarray(view)
        )

    def test_a_transposed_view_agrees_with_its_copy(self):
        a = multilabel_volume()
        view = a.transpose(2, 0, 1)
        assert mesh_and_fingerprint(view) == mesh_and_fingerprint(
            np.ascontiguousarray(view)
        )

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_dtype_width_does_not_change_geometry(self, dtype):
        # Keep labels inside uint8 so every width represents them exactly.
        a = multilabel_volume(labels=5)
        assert a.max() < 256
        reference = mesh_and_fingerprint(a.astype(np.uint32))
        assert mesh_and_fingerprint(a.astype(dtype)) == reference


class TestLabelValues:
    def test_geometry_is_independent_of_label_magnitude(self):
        """Renaming objects must move no vertex."""
        a = multilabel_volume(labels=4)
        remapped = np.zeros_like(a, dtype=np.uint64)
        mapping = {1: 10, 2: 98_340_797, 3: 4_000_000_000, 4: 2**40}
        for old, new in mapping.items():
            remapped[a == old] = new

        original = serra_mesh.Mesher().mesh(a, close=True)
        renamed = serra_mesh.Mesher().mesh(remapped, close=True)
        for old, new in mapping.items():
            np.testing.assert_array_equal(
                original.get(old).vertices, renamed.get(new).vertices
            )
            np.testing.assert_array_equal(
                original.get(old).faces, renamed.get(new).faces
            )

    def test_ids_are_always_ascending(self):
        a = multilabel_volume(labels=8, seed=3)
        ids = serra_mesh.Mesher().mesh(a, close=True).ids()
        assert np.all(np.diff(ids) > 0)


class TestThreadIndependence:
    """Extraction is split into bands along one axis and merged.

    The failure mode this guards against is subtle: geometry can be perfectly
    correct while *ordering* depends on how many bands were used, which would
    make output differ between machines. It happened during development — a
    band cannot emit its own first cell layer's quads, and appending those at
    the end instead of splicing them between the bands they sit between changed
    every label's face order.
    """

    @pytest.mark.parametrize("shape", [(40, 36, 32), (17, 23, 97), (8, 8, 9)])
    def test_every_thread_count_agrees(self, shape):
        a = multilabel_volume(shape=shape, labels=7, seed=5)
        prints = {
            t: mesh_and_fingerprint(a, threads=t) for t in (1, 2, 3, 4, 5, 8, 13, 0)
        }
        assert len(set(prints.values())) == 1, prints

    @pytest.mark.parametrize("k", [0, 3])
    def test_relaxation_is_also_thread_independent(self, k):
        a = multilabel_volume(shape=(40, 36, 32), labels=8, seed=2)
        one = mesh_and_fingerprint(a, threads=1, relaxation=k)
        many = mesh_and_fingerprint(a, threads=0, relaxation=k)
        assert one == many

    def test_a_volume_too_shallow_to_band_still_works(self):
        """Fewer layers than the minimum band depth must fall back cleanly."""
        a = multilabel_volume(shape=(20, 20, 3), labels=4, seed=7)
        assert mesh_and_fingerprint(a, threads=1) == mesh_and_fingerprint(a, threads=8)

    def test_thread_count_is_reported(self):
        assert serra_mesh.Mesher(threads=3).effective_threads == 3
        assert serra_mesh.Mesher(threads=1).effective_threads == 1
        assert serra_mesh.Mesher(threads=0).effective_threads >= 1

    def test_negative_thread_count_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            serra_mesh.Mesher(threads=-1)

    ENV_SCRIPT = textwrap.dedent(
        """
        import hashlib, sys
        import numpy as np
        sys.path.insert(0, {tests_dir!r})
        from test_determinism import multilabel_volume, fingerprint
        import serra_mesh
        a = multilabel_volume(shape=(40, 36, 32), labels=10, seed=1)
        print(fingerprint(serra_mesh.Mesher().mesh(a, close=True)))
        """
    )

    def _run(self, threads):
        env = dict(os.environ)
        env["RAYON_NUM_THREADS"] = str(threads)
        script = self.ENV_SCRIPT.format(tests_dir=os.path.dirname(__file__))
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return out.stdout.strip()

    def test_rayon_num_threads_does_not_change_output(self):
        """threads=0 defers to RAYON_NUM_THREADS, so that must not matter."""
        assert self._run(1) == self._run(8)
        assert self._run(2) == self._run(1)
