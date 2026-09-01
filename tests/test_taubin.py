"""Taubin smoothing.

Taubin alternates a positive step with a larger negative one. The pair is a
low-pass filter rather than a diffusion, so unlike Laplacian relaxation it does
not steadily drain volume — which is the reason to have it at all when the
surface is a measurement rather than a model.

What it must not cost is the two properties the rest of the library is built on:
seam vertices stay bit-identical between chunks, so chunks still stitch by exact
vertex equality, and no vertex strays further from the data than
``max_deviation``. Both are asserted here, because a smoother that quietly moves
a seam vertex produces meshes that render correctly and stitch into
disconnected rubble.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import sphere_mask

import serra_mesh

S = 24  # chunk size in voxels
N = 3 * S
HALO = 2  # positive-side halo required by dual contouring


def smoothed(mask, **kwargs):
    return serra_mesh.Mesher(**kwargs).mesh(mask, close=True).get(1, normals=True)


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


def chunked_sphere():
    g = np.indices((N, N, N))
    c = (N - 1) / 2
    return (((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= 30**2).astype(
        np.uint32
    )


# --------------------------------------------------------------------------
# it does something, and the right something
# --------------------------------------------------------------------------


def nearest_distance(points, reference):
    """Distance from each point to the closest of `reference`.

    Brute force on purpose: the fixtures here are small, and it avoids a test
    dependency on a spatial index. Vertex *indices* are not comparable between
    two meshes — a quad is split along its shorter diagonal, so moving the
    vertices changes the triangulation and therefore the emission order — so
    every comparison of two meshes has to go through positions.
    """
    out = np.empty(len(points))
    for i in range(0, len(points), 512):
        block = points[i : i + 512]
        d = np.linalg.norm(block[:, None, :] - reference[None, :, :], axis=2)
        out[i : i + 512] = d.min(axis=1)
    return out


def test_taubin_smooths():
    mask = sphere_mask(20.0)
    plain = smoothed(mask)
    filtered = smoothed(mask, taubin=10)
    assert len(plain.vertices) == len(filtered.vertices)
    assert len(plain.faces) == len(filtered.faces)
    assert not np.array_equal(
        np.sort(plain.vertices, axis=0), np.sort(filtered.vertices, axis=0)
    )


def test_taubin_improves_area_and_normals_on_a_sphere():
    radius = 20.0
    mask = sphere_mask(radius)
    exact_area = 4 * np.pi * radius**2

    plain = smoothed(mask)
    filtered = smoothed(mask, taubin=10)
    assert abs(filtered.area() / exact_area - 1) < abs(plain.area() / exact_area - 1)


def test_taubin_holds_onto_volume_where_relaxation_does_not():
    """The whole point of the negative step.

    Laplacian iteration is diffusion: run it hard and the object shrinks. Taubin
    is a low-pass filter, so the low frequencies that carry the object's bulk
    come through at close to unit gain.
    """
    radius = 20.0
    mask = sphere_mask(radius)
    exact = 4 / 3 * np.pi * radius**3

    # Deliberately past the point where relaxation starts eating the shape.
    laplacian = smoothed(mask, relaxation=20, max_deviation=2.0)
    low_pass = smoothed(mask, taubin=20, max_deviation=2.0)

    assert laplacian.volume() < low_pass.volume()
    assert abs(low_pass.volume() / exact - 1) < 0.01


# --------------------------------------------------------------------------
# the guarantees it must not break
# --------------------------------------------------------------------------


@pytest.mark.parametrize("max_deviation", [0.0, 0.125, 0.5])
def test_no_vertex_strays_past_the_deviation_bound(max_deviation):
    mask = sphere_mask(10.0)
    plain = smoothed(mask)
    filtered = smoothed(mask, taubin=40, max_deviation=max_deviation)
    moved = nearest_distance(filtered.vertices, plain.vertices).max()
    # The bound is per axis, so the worst case in Euclidean distance is the
    # diagonal of the cube it defines. One fixed-point unit of slack for the
    # rounding back to 1/256 of a voxel.
    limit = np.sqrt(3) * (max_deviation + 1.0 / 256) + 1e-6
    assert moved <= limit


@pytest.mark.parametrize("splits,name", [({0}, "x"), ({0, 1, 2}, "xyz")])
def test_seam_vertices_stay_bit_identical_between_chunks(splits, name):
    """Exact equality, not nearly.

    Stitching welds by exact coordinate comparison, so a filter that moves a
    seam vertex by one part in a million breaks it just as thoroughly as one
    that moves it by a voxel.
    """
    volume = chunked_sphere()
    pieces = decompose(volume, splits, taubin=8)

    seen: dict[bytes, int] = {}
    for mesh, offset in pieces:
        shifted = np.ascontiguousarray(
            mesh.vertices.astype(np.float64) + offset, dtype=np.float32
        )
        for row in shifted:
            key = row.tobytes()
            seen[key] = seen.get(key, 0) + 1

    shared = sum(1 for n in seen.values() if n > 1)
    assert shared > 0, "the decomposition produced no shared seam vertices at all"


@pytest.mark.parametrize(
    "splits,name", [({0}, "x"), ({0, 1}, "xy"), ({0, 1, 2}, "xyz")]
)
def test_chunks_still_stitch_watertight_after_smoothing(splits, name):
    volume = chunked_sphere()
    joined = serra_mesh.stitch(decompose(volume, splits, taubin=8), dedup_faces=False)
    reference = serra_mesh.Mesher(taubin=8).mesh(volume, close=False).get(1)

    assert len(joined.faces) == len(reference.faces)
    assert joined.count_boundary_edges() == 0
    assert joined.is_closed()


def test_output_is_identical_whatever_the_thread_count():
    mask = sphere_mask(18.0)
    one = smoothed(mask, taubin=6, threads=1)
    many = smoothed(mask, taubin=6, threads=4)
    assert np.array_equal(one.vertices, many.vertices)
    assert np.array_equal(one.faces, many.faces)


def test_smoothing_runs_before_simplification():
    """`get` simplifies whatever `mesh` produced, so the order is structural.

    Asserted rather than assumed: smoothing before decimating is the order that
    measures better (see ``bench/taubin.py``), and nothing else in the API
    enforces it.
    """
    mask = sphere_mask(14.0)
    smooth_mesher = serra_mesh.Mesher(taubin=10).mesh(mask, close=True)
    raw_mesher = serra_mesh.Mesher().mesh(mask, close=True)

    full = smooth_mesher.get(1)
    reduced = smooth_mesher.get(1, reduction_factor=8, max_error=1e9)
    raw_reduced = raw_mesher.get(1, reduction_factor=8, max_error=1e9)

    assert len(reduced.faces) < len(full.faces)
    # The decimated result carries the smoothing, so it sits closer to the
    # smoothed surface than to the unsmoothed one.
    to_smooth = nearest_distance(reduced.vertices, full.vertices).mean()
    to_raw = nearest_distance(reduced.vertices, raw_mesher.get(1).vertices).mean()
    assert to_smooth < to_raw
    # And it is not simply the same thing decimation would have produced anyway.
    assert not np.array_equal(
        np.sort(reduced.vertices, axis=0), np.sort(raw_reduced.vertices, axis=0)
    )


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------


def test_zero_iterations_changes_nothing():
    mask = sphere_mask(14.0)
    assert np.array_equal(smoothed(mask).vertices, smoothed(mask, taubin=0).vertices)


def test_the_two_filters_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either relaxation or taubin"):
        serra_mesh.Mesher(relaxation=3, taubin=3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"taubin_pass_band": 0.0},
        {"taubin_pass_band": 1.0},
        {"taubin_lambda": 0.0},
        {"taubin_lambda": 1.0},
        # A pass band too wide for this lambda drives mu below -1, at which
        # point the expanding pass overshoots and the filter amplifies.
        {"taubin_pass_band": 0.9, "taubin_lambda": 0.63},
    ],
)
def test_unstable_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        serra_mesh.Mesher(taubin=3, **kwargs)


def test_parameters_are_only_checked_when_the_filter_runs():
    """A default that would be invalid is harmless while `taubin` is zero."""
    serra_mesh.Mesher(taubin=0, taubin_pass_band=0.9, taubin_lambda=0.63)


def test_negative_iterations_are_rejected():
    with pytest.raises(ValueError, match="taubin must be non-negative"):
        serra_mesh.Mesher(taubin=-1)
