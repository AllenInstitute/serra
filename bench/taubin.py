"""Does Taubin smoothing help serra's meshes the way it helps marching cubes?

Raised as issue #1. Taubin (a windowed-sinc low-pass on the mesh Laplacian) is
the standard fix for marching-cubes staircasing, and serra already does its own
constrained smoothing pass (`relaxation=k`). The questions worth measuring are
whether Taubin adds anything on top, whether it can be run per chunk without
breaking stitching, and in which order it should be composed with quadric
simplification.

Five experiments:

1. Analytic shapes, where area, volume and every normal are known exactly.
2. Seam preservation: smooth two chunks separately, then stitch.
3. Order of operations against quadric simplification.
4. Real neuropil, where the objects are thin processes rather than spheres,
   and where unconstrained smoothing has somewhere to go wrong.
5. serra's own built-in filter (`Mesher(taubin=k)`) against all of the above --
   what the measurements in 1-4 were used to justify building.
6. The same filter on real neuropil, which is what settles the default.
7. Whether a fixed iteration count transfers across object size, and what the
   cost of smoothing per chunk actually is.

    uv sync --group bench
    python bench/taubin.py

VTK's window function is set directly: pyvista 0.48 does not expose it, and the
issue specifically recommends Blackman.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time

import numpy as np

RESOLUTION = np.array([32.0, 32.0, 40.0])


# --------------------------------------------------------------------------
# Taubin


def taubin(
    vertices,
    faces,
    n_iter=20,
    pass_band=0.1,
    window="blackman",
    boundary_smoothing=True,
):
    """Windowed-sinc smoothing, with the window chosen explicitly.

    `boundary_smoothing=False` holds every vertex on an open border fixed. That
    is the knob that decides whether chunked smoothing can be stitched: with
    `owned_shape`, a chunk's seam is exactly its open border.
    """
    import pyvista as pv
    import vtk

    poly = to_polydata(vertices, faces)

    filt = vtk.vtkWindowedSincPolyDataFilter()
    filt.SetInputData(poly)
    filt.SetNumberOfIterations(n_iter)
    filt.SetPassBand(pass_band)
    filt.SetBoundarySmoothing(boundary_smoothing)
    filt.SetFeatureEdgeSmoothing(False)
    filt.SetNonManifoldSmoothing(False)
    filt.SetNormalizeCoordinates(False)
    # VTK 9.6 ships a typo'd setter for Hanning ("SetWindowFunctionoHanning").
    setter = f"SetWindowFunctionTo{window.capitalize()}"
    if not hasattr(filt, setter):
        setter = setter.replace("FunctionTo", "Functiono")
    getattr(filt, setter)()
    filt.Update()

    out = pv.wrap(filt.GetOutput())
    points = np.asarray(out.points, dtype=np.float64)
    assert len(points) == len(vertices), "VTK renumbered points"
    return points, faces


def to_polydata(vertices, faces):
    import pyvista as pv

    padded = np.hstack(
        [np.full((len(faces), 1), 3, np.int64), np.asarray(faces, np.int64)]
    ).ravel()
    return pv.PolyData(np.asarray(vertices, np.float64), padded)


# --------------------------------------------------------------------------
# Measurements


def area(vertices, faces):
    v = np.asarray(vertices, np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)


def volume(vertices, faces):
    v = np.asarray(vertices, np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def normal_error_degrees(vertices, faces, exact_normal):
    """Mean angle between each face normal and the analytic surface normal."""
    v = np.asarray(vertices, np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1)
    keep = length > 1e-12
    n = n[keep] / length[keep, None]
    want = exact_normal((a + b + c)[keep] / 3.0)
    want = want / np.linalg.norm(want, axis=1)[:, None]
    cos = np.abs(np.einsum("ij,ij->i", n, want)).clip(0, 1)
    return float(np.degrees(np.arccos(cos)).mean())


def bad_edges(faces):
    """Edges not used by exactly two faces — a hole or a non-manifold join."""
    e = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts != 2).sum())


def distance_to(reference, points):
    import igl

    rv, rf = reference
    d2, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(points, np.float64),
        np.ascontiguousarray(rv, np.float64),
        np.ascontiguousarray(rf, np.int64),
    )
    return np.sqrt(d2)


# --------------------------------------------------------------------------
# 1. Analytic shapes


def sphere_mask(radius, pad=4):
    n = 2 * (radius + pad) + 1
    g = np.arange(n) - (n - 1) / 2
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return ((x * x + y * y + z * z) <= radius * radius).astype(np.uint32)


def experiment_analytic(zmesh_mod, serra_mesh, radius=32):
    print("=" * 78)
    print("1. Analytic sphere, radius 32 voxels (isotropic 1 nm grid)")
    print("=" * 78)
    print(
        "   Taubin: 20 iterations, pass band 0.1, Blackman window,"
        " boundary smoothing on\n"
    )

    mask = sphere_mask(radius)
    centre = (mask.shape[0] - 1) / 2
    exact_area = 4 * np.pi * radius**2
    exact_volume = 4 / 3 * np.pi * radius**3

    def radial(p):
        return p - centre

    zm = zmesh_mod.Mesher((1, 1, 1))
    zm.mesh(mask, close=True)
    zmesh_mesh = zm.get(1)

    variants = [("zmesh", zmesh_mesh.vertices, zmesh_mesh.faces)]
    for k in (0, 3):
        m = serra_mesh.Mesher(voxel_resolution=[1.0, 1.0, 1.0], relaxation=k)
        m.mesh(mask, close=True)
        got = m.get(1)
        variants.append((f"serra k={k}", got.vertices, got.faces))

    print(f"{'mesh':>22} {'area err':>9} {'volume err':>11} {'normal err':>11}")
    for name, v, f in variants:
        for suffix, (sv, sf) in (
            ("", (v, f)),
            (" + taubin", taubin(v, f)),
        ):
            print(
                f"{name + suffix:>22} "
                f"{area(sv, sf) / exact_area - 1:>+8.2%} "
                f"{volume(sv, sf) / exact_volume - 1:>+10.2%} "
                f"{normal_error_degrees(sv, sf, radial):>10.2f}d"
            )
    print()


# --------------------------------------------------------------------------
# 2. Seams


def chunk_sphere(serra_mesh, n=72, chunk=24, radius=30, halo=2):
    """The chunking convention from docs/chunked.md: +halo only, owned_shape set."""
    import itertools

    g = np.indices((n, n, n))
    c = (n - 1) / 2
    volume = (
        ((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= radius**2
    ).astype(np.uint32)

    pieces = []
    for origin in itertools.product(*[range(0, n, chunk)] * 3):
        window, owned = [], []
        for k in range(3):
            end = min(origin[k] + chunk, n)
            owned.append(end - origin[k])
            window.append(slice(origin[k], min(end + halo, n)))
        mesher = serra_mesh.Mesher().mesh(
            volume[tuple(window)], close=False, owned_shape=owned
        )
        if 1 in mesher:
            pieces.append((mesher.get(1), np.array(origin, dtype=float)))
    return volume, pieces


def experiment_seams(serra_mesh):
    print("=" * 78)
    print("2. Smoothing each chunk separately, then stitching")
    print("=" * 78)
    print(
        "   A sphere across 3x3x3 chunks of 24 voxels, meshed the way a chunked"
        "\n   pipeline would: positive-only halo of 2, owned_shape set so each"
        " face is\n   emitted once. serra welds seams by *exact* float equality,"
        " so the question\n   is not whether seam vertices move much -- it is"
        " whether they move at all.\n"
    )

    volume, pieces = chunk_sphere(serra_mesh)
    reference = serra_mesh.Mesher().mesh(volume, close=False).get(1)
    joined = serra_mesh.stitch(pieces, dedup_faces=False)
    print(
        f"   unsmoothed: {len(pieces)} chunks -> {len(joined.faces):,} faces, "
        f"{joined.count_boundary_edges()} open edges "
        f"(whole volume: {len(reference.faces):,})\n"
    )

    seen: dict[bytes, int] = {}
    for mesh, offset in pieces:
        for key in vertex_keys(mesh.vertices, offset):
            seen[key] = seen.get(key, 0) + 1
    seam_total = sum(1 for n in seen.values() if n > 1)
    on_seam = [
        np.array([seen[k] > 1 for k in vertex_keys(mesh.vertices, offset)])
        for mesh, offset in pieces
    ]

    one_piece = taubin(reference.vertices.astype(np.float64), reference.faces)

    print(
        f"{'window':>10} {'boundary':>9} {'seam moved':>11} {'max seam v':>9} "
        f"{'open edges':>11} {'vs one-piece':>13}"
    )
    for window in ("blackman", "hamming", "hanning", "nuttall"):
        for boundary in (True, False):
            smoothed, moved, worst = [], 0, 0.0
            for (mesh, offset), seam in zip(pieces, on_seam):
                v = mesh.vertices.astype(np.float64)
                sv, _ = taubin(
                    v, mesh.faces, window=window, boundary_smoothing=boundary
                )
                delta = np.abs(sv[seam] - v[seam])
                moved += int((delta > 1e-9).any(axis=1).sum())
                worst = max(worst, float(delta.max()) if len(delta) else 0.0)
                smoothed.append((MeshLike(sv, mesh.faces), offset))
            welded = serra_mesh.stitch(smoothed, dedup_faces=False)
            drift = distance_to(one_piece, welded.vertices).max()
            print(
                f"{window:>10} {str(boundary):>9} {moved:>11,} {worst:>9.1e} "
                f"{welded.count_boundary_edges():>11,} {drift:>12.3f}v"
            )

    # The repair: smooth each chunk, then put every seam vertex back where the
    # extractor left it. Exact equality is restored by construction, at the cost
    # of a ring of unsmoothed geometry one vertex wide around each chunk.
    smoothed = []
    for (mesh, offset), seam in zip(pieces, on_seam):
        v = mesh.vertices.astype(np.float64)
        sv, _ = taubin(v, mesh.faces, boundary_smoothing=False)
        sv[seam] = v[seam]
        smoothed.append((MeshLike(sv, mesh.faces), offset))
    welded = serra_mesh.stitch(smoothed, dedup_faces=False)
    drift = distance_to(one_piece, welded.vertices)
    print(
        f"\n{'snap seams back':>10} {'-':>9} {0:>11,} {0.0:>9.1e} "
        f"{welded.count_boundary_edges():>11,} {drift.max():>12.3f}v"
        f"   <- stitches"
    )
    print(
        f"\n   ({seam_total:,} distinct seam vertices across {len(pieces)} chunks;"
        f" snapping leaves them\n    a median {np.median(drift):.3f} and at worst"
        f" {drift.max():.3f} voxels from where one-piece\n    smoothing would have"
        f" put them)\n"
    )


def vertex_keys(vertices, offset):
    """Exact-equality keys, matching how `serra_mesh.stitch` welds."""
    v = np.ascontiguousarray(
        np.asarray(vertices, np.float64) + np.asarray(offset), np.float32
    )
    return [row.tobytes() for row in v]


class MeshLike:
    """Minimum surface `serra_mesh.stitch` needs from a mesh."""

    def __init__(self, vertices, faces):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.faces = faces


# --------------------------------------------------------------------------
# 3. Order of operations, and 4. real neuropil


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return np.load(handle)
    return np.load(path)


def sample_labels(volume, count, low, high):
    ids, counts = np.unique(volume, return_counts=True)
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]
    order = np.argsort(counts)
    picks = np.linspace(low, high, count) / 100.0 * (len(order) - 1)
    chosen = [order[int(round(p))] for p in picks]
    return [(int(ids[i]), int(counts[i])) for i in chosen]


def decimate(vertices, faces, reduction):
    """VTK quadric decimation, so both orders below use the same simplifier."""
    out = to_polydata(vertices, faces).decimate(reduction)
    f = np.asarray(out.faces).reshape(-1, 4)[:, 1:]
    return np.asarray(out.points, np.float64), f


def roughness_degrees(vertices, faces):
    """Mean dihedral angle across edges -- a scale-free measure of faceting."""
    v = np.asarray(vertices, np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]

    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    left, right = owner[:-1][same], owner[1:][same]
    cos = np.einsum("ij,ij->i", n[left], n[right]).clip(-1, 1)
    return float(np.degrees(np.arccos(cos)).mean())


def experiment_order(serra_mesh, volume_array, labels, reduce_to=10):
    print("=" * 78)
    print(f"3. Order of operations: Taubin and quadric simplification ({reduce_to}x)")
    print("=" * 78)
    print(
        "   Both orders use VTK's quadric decimation, so the order is the only"
        "\n   difference. Accuracy is distance from the result back to the"
        " full-resolution\n   serra k=0 surface; roughness is the mean dihedral"
        " angle -- a fair measure\n   between variants of the same extractor,"
        " but not between extractors (see 4).\n"
    )

    mesher = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION))
    mesher.mesh(volume_array, close=True)
    reduction = 1.0 - 1.0 / reduce_to

    rows: dict[str, list] = {}
    used = 0
    for label, _ in labels:
        full = mesher.get(label)
        if len(full.faces) < 2000 or not full.is_closed():
            continue
        used += 1
        rv, rf = full.vertices.astype(np.float64), full.faces
        reference = (rv, rf)

        smoothed = taubin(rv, rf)
        candidates = {
            "simplify only": decimate(rv, rf, reduction),
            "taubin then simplify": decimate(*smoothed, reduction),
            "simplify then taubin": taubin(*decimate(rv, rf, reduction)),
            "serra simplify only": (
                mesher.get(label, reduction_factor=reduce_to, max_error=1e9).vertices
            ),
        }
        got = mesher.get(label, reduction_factor=reduce_to, max_error=1e9)
        candidates["serra simplify only"] = (
            got.vertices.astype(np.float64),
            got.faces,
        )

        for name, (v, f) in candidates.items():
            d = distance_to(reference, v) / RESOLUTION.min()
            rows.setdefault(name, []).append(
                (
                    float(d.mean()),
                    float(d.max()),
                    volume(v, f) / volume(*reference),
                    roughness_degrees(v, f),
                    len(f),
                )
            )

    print(
        f"{'order':>22} {'faces':>8} {'mean dist':>10} {'worst':>7} "
        f"{'volume kept':>12} {'roughness':>10}"
    )
    for name, values in rows.items():
        arr = np.array(values)
        print(
            f"{name:>22} {arr[:, 4].sum():>8,.0f} {arr[:, 0].mean():>9.3f}v "
            f"{arr[:, 1].max():>6.2f}v {arr[:, 2].mean():>11.1%} "
            f"{arr[:, 3].mean():>9.1f}d"
        )
    print(f"\n   ({used} closed objects, distances in voxels)\n")


def experiment_neuropil(zmesh_mod, serra_mesh, volume_array, labels):
    print("=" * 78)
    print("4. Real neuropil: what smoothing costs a thin process")
    print("=" * 78)
    print(
        "   Voxel count is the truth. A sphere has nothing to lose to"
        " smoothing; a\n   200 nm spine neck does. Only objects clear of the"
        " array faces are used, since\n   an object running off the edge is"
        " closed by a cap rather than by its own surface.\n"
    )

    voxel_volume = float(np.prod(RESOLUTION))
    zm = zmesh_mod.Mesher(tuple(RESOLUTION))
    zm.mesh(volume_array, close=True)
    s0 = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION)).mesh(
        volume_array, close=True
    )
    s3 = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION), relaxation=3).mesh(
        volume_array, close=True
    )

    interior = [
        (label, voxels)
        for label, voxels in labels
        if label in s0 and s0.get(label).is_closed()
    ]

    # Mean dihedral angle understates marching-cubes staircasing: a terrace is
    # locally flat, so most of its edges score zero and the 90-degree steps are
    # a minority. Compare roughness within an extractor, not across them; for a
    # cross-extractor measure of surface fidelity use experiment 1's normal
    # error against an analytic shape.
    variants = [("zmesh", zm), ("serra k=0", s0), ("serra k=3", s3)]
    print(
        f"{'mesh':>22} {'volume/true':>12} {'area/zmesh':>11} "
        f"{'roughness':>10} {'time':>8}"
    )
    baseline_area = None
    for name, mesher in variants:
        for suffix, smooth in (("", False), (" + taubin", True)):
            total_v = total_a = total_truth = 0.0
            rough, elapsed = [], 0.0
            for label, voxels in interior:
                mesh = mesher.get(label)
                v, f = mesh.vertices.astype(np.float64), mesh.faces
                if smooth:
                    start = time.perf_counter()
                    v, f = taubin(v, f)
                    elapsed += time.perf_counter() - start
                total_v += volume(v, f)
                total_a += area(v, f)
                total_truth += voxels * voxel_volume
                rough.append(roughness_degrees(v, f))
            if baseline_area is None:
                baseline_area = total_a
            print(
                f"{name + suffix:>22} {total_v / total_truth:>11.1%} "
                f"{total_a / baseline_area:>10.1%} {np.mean(rough):>9.1f}d "
                + (f"{elapsed:>7.2f}s" if smooth else f"{'-':>8}")
            )

    # Where the loss lands. Smoothing removes a roughly fixed depth from every
    # surface, so the cost scales with surface-to-volume ratio, not with size.
    print("\n   volume relative to the voxel truth, by object size (serra k=0):\n")
    print(f"{'voxels':>12} {'plain':>10} {'+ taubin':>10} {'lost':>8}")
    for label, voxels in interior[:: max(1, len(interior) // 8)]:
        mesh = s0.get(label)
        v, f = mesh.vertices.astype(np.float64), mesh.faces
        sv, sf = taubin(v, f)
        truth = voxels * voxel_volume
        plain, smooth = volume(v, f) / truth, volume(sv, sf) / truth
        print(
            f"{voxels:>12,} {plain:>9.1%} {smooth:>9.1%} "
            f"{(smooth - plain) / plain:>+7.1%}"
        )
    print(f"\n   ({len(interior)} closed objects)\n")


def experiment_builtin(serra_mesh, radius=32):
    """serra's own filter, against the post-hoc one and against relaxation."""
    print("=" * 78)
    print("5. The built-in filter: Mesher(taubin=k)")
    print("=" * 78)
    print(
        "   lambda/mu Taubin in Rust, run inside mesh() with seam vertices"
        " pinned and\n   the same max_deviation bound relaxation uses. Compare"
        " the shape of the two\n   curves, not single rows: relaxation buys"
        " normals with volume, Taubin does\n   not.\n"
    )

    mask = sphere_mask(radius)
    centre = (mask.shape[0] - 1) / 2
    exact_area = 4 * np.pi * radius**2
    exact_volume = 4 / 3 * np.pi * radius**3

    print(f"{'setting':>24} {'area err':>9} {'volume err':>11} {'normal err':>11}")
    settings = [
        ("no smoothing", {}),
        ("relaxation=3", {"relaxation": 3}),
        ("relaxation=10", {"relaxation": 10}),
        ("relaxation=20", {"relaxation": 20}),
        ("taubin=3", {"taubin": 3}),
        ("taubin=10", {"taubin": 10}),
        ("taubin=20", {"taubin": 20}),
        ("taubin=40", {"taubin": 40}),
    ]
    for name, kwargs in settings:
        mesher = serra_mesh.Mesher(voxel_resolution=[1.0, 1.0, 1.0], **kwargs)
        mesh = mesher.mesh(mask, close=True).get(1)
        v, f = mesh.vertices.astype(np.float64), mesh.faces
        print(
            f"{name:>24} "
            f"{area(v, f) / exact_area - 1:>+8.2%} "
            f"{volume(v, f) / exact_volume - 1:>+10.2%} "
            f"{normal_error_degrees(v, f, lambda p: p - centre):>10.2f}d"
        )

    # And against the post-hoc VTK filter on the same surface, since that is
    # what the earlier experiments measured.
    plain = serra_mesh.Mesher(voxel_resolution=[1.0, 1.0, 1.0])
    mesh = plain.mesh(mask, close=True).get(1)
    sv, sf = taubin(mesh.vertices.astype(np.float64), mesh.faces)
    print(
        f"{'k=0 + vtk (20 iter)':>24} "
        f"{area(sv, sf) / exact_area - 1:>+8.2%} "
        f"{volume(sv, sf) / exact_volume - 1:>+10.2%} "
        f"{normal_error_degrees(sv, sf, lambda p: p - centre):>10.2f}d"
    )
    print()


def experiment_builtin_on_real_data(serra_mesh, volume_array, labels):
    """Where it actually matters: thin processes, not spheres."""
    print("=" * 78)
    print("6. The built-in filter on real neuropil")
    print("=" * 78)
    print(
        "   Voxel count is the truth, and only objects clear of the array faces"
        " count,\n   since one running off the edge is closed by a cap rather"
        " than by its own\n   surface. This is the table that decides the"
        " default: a spine neck has a\n   surface-to-volume ratio a sphere does"
        " not.\n"
    )

    voxel_volume = float(np.prod(RESOLUTION))
    base = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION))
    base.mesh(volume_array, close=True)
    interior = [
        (label, voxels)
        for label, voxels in labels
        if label in base and base.get(label).is_closed()
    ]

    print(f"{'setting':>24} {'volume / true':>14} {'area / unsmoothed':>18}")
    baseline_area = None
    for name, kwargs in [
        ("no smoothing", {}),
        ("relaxation=3", {"relaxation": 3}),
        ("relaxation=10", {"relaxation": 10}),
        ("taubin=3", {"taubin": 3}),
        ("taubin=10", {"taubin": 10}),
        ("taubin=20", {"taubin": 20}),
    ]:
        mesher = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION), **kwargs)
        mesher.mesh(volume_array, close=True)
        total_v = total_a = total_truth = 0.0
        for label, voxels in interior:
            got = mesher.get(label)
            total_v += got.volume()
            total_a += got.area()
            total_truth += voxels * voxel_volume
        if baseline_area is None:
            baseline_area = total_a
        print(
            f"{name:>24} {total_v / total_truth:>13.2%} "
            f"{total_a / baseline_area:>17.1%}"
        )
    print(f"\n   ({len(interior)} closed objects)\n")


def experiment_scale(serra_mesh):
    """Do fixed parameters mean the same thing on a big object as a small one?

    Raised in issue #1, and the answer splits: convergence transfers, shrinkage
    does not.
    """
    print("=" * 78)
    print("7a. Does a fixed iteration count transfer across object size?")
    print("=" * 78)
    print(
        "   Spheres from r=8 to r=64 -- a 65x range in vertex count -- with the"
        " same\n   settings throughout. Normal error is what the filter is for;"
        " area error is\n   what it costs.\n"
    )

    radii = [8, 16, 32, 64]
    settings = [
        ("no smoothing", {}),
        ("relaxation=3", {"relaxation": 3}),
        ("relaxation=10", {"relaxation": 10}),
        ("taubin=10", {"taubin": 10}),
        ("taubin=20", {"taubin": 20}),
    ]

    header = "".join(f"{'r=' + str(r):>18}" for r in radii)
    print(f"{'setting':>16}{header}")
    for name, kwargs in settings:
        row = ""
        for radius in radii:
            mask = sphere_mask(radius)
            centre = (mask.shape[0] - 1) / 2
            mesher = serra_mesh.Mesher(voxel_resolution=[1.0, 1.0, 1.0], **kwargs)
            mesh = mesher.mesh(mask, close=True).get(1)
            v, f = mesh.vertices.astype(np.float64), mesh.faces
            angle = normal_error_degrees(v, f, lambda p, c=centre: p - c)
            row += f"{angle:>10.2f}d{area(v, f) / (4 * np.pi * radius**2) - 1:>+7.2%}"
        print(f"{name:>16}{row}")

    print(
        "\n   Normal error is flat across the range, and structurally so: dual"
        " contouring\n   puts one vertex per cell, so edge length is ~1 voxel"
        " whatever the object's\n   size, and the staircase artefact sits in a"
        " fixed band of graph frequency.\n   That would not hold on a decimated"
        " mesh -- which is why smoothing belongs\n   before simplification, and"
        " in serra structurally is.\n"
    )
    print(
        "   Area error is *not* flat. Laplacian iteration removes a roughly"
        " fixed depth\n   from every surface, so the relative cost goes as 1/r:"
        " negligible on a cell\n   body, several percent on a small process."
        " Taubin's is far flatter, which is\n   the argument for it.\n"
    )


def experiment_chunk_cost(serra_mesh, size=144, radius=60, halo=2):
    """What smoothing per chunk costs, versus smoothing the assembled mesh."""
    import itertools

    print("=" * 78)
    print("7b. The price of smoothing per chunk: the pinned ring at each seam")
    print("=" * 78)
    print(
        "   Both filters hold the outermost cell layer fixed, so seam vertices"
        " never move\n   and chunks still weld by exact equality. The cost is"
        " that the ring around\n   each seam goes unsmoothed.\n"
    )

    g = np.indices((size, size, size))
    c = (size - 1) / 2
    volume_array = (
        ((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= radius**2
    ).astype(np.uint32)
    whole = serra_mesh.Mesher(taubin=10).mesh(volume_array, close=False).get(1)

    print(
        f"{'chunk':>7} {'chunks':>7} {'seam verts':>11} {'pinned':>8} "
        f"{'stitches':>9} {'faces ok':>9} {'median':>9} {'worst':>8}"
    )
    for chunk in (16, 24, 36, 48, 72):
        pieces = []
        for origin in itertools.product(*[range(0, size, chunk)] * 3):
            window, owned = [], []
            for k in range(3):
                end = min(origin[k] + chunk, size)
                owned.append(end - origin[k])
                window.append(slice(origin[k], min(end + halo, size)))
            mesher = serra_mesh.Mesher(taubin=10).mesh(
                volume_array[tuple(window)], close=False, owned_shape=owned
            )
            if 1 in mesher:
                pieces.append((mesher.get(1), np.array(origin, dtype=float)))

        seen: dict[bytes, int] = {}
        for mesh, offset in pieces:
            shifted = np.ascontiguousarray(
                mesh.vertices.astype(np.float64) + offset, dtype=np.float32
            )
            for row in shifted:
                key = row.tobytes()
                seen[key] = seen.get(key, 0) + 1
        shared = sum(1 for n in seen.values() if n > 1)

        joined = serra_mesh.stitch(pieces, dedup_faces=False)
        d = distance_to(
            (whole.vertices.astype(np.float64), whole.faces), joined.vertices
        )
        print(
            f"{chunk:>7} {len(pieces):>7} {shared:>11,} {shared / len(seen):>7.1%} "
            f"{str(joined.count_boundary_edges() == 0):>9} "
            f"{str(len(joined.faces) == len(whole.faces)):>9} "
            f"{np.median(d):>8.4f}v {d.max():>7.3f}v"
        )
    print(
        "\n   The pinned fraction goes as surface-to-volume, so at a 256^3"
        " chunk it is\n   under 1%. `median` and `worst` are distances from"
        " where smoothing the whole\n   volume in one piece would have put the"
        " surface.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="data/microns_neuropil.npy.gz")
    parser.add_argument("--zmesh", default="/Users/forrestc/ConnectomeStack/zmesh")
    parser.add_argument("--objects", type=int, default=24)
    parser.add_argument("--low", type=float, default=70.0)
    parser.add_argument("--high", type=float, default=99.0)
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.zmesh)
    import zmesh

    import serra_mesh

    experiment_analytic(zmesh, serra_mesh)
    experiment_seams(serra_mesh)
    experiment_builtin(serra_mesh)

    if not args.skip_real:
        volume_array = load(args.volume)
        labels = sample_labels(volume_array, args.objects, args.low, args.high)
        experiment_order(serra_mesh, volume_array, labels)
        experiment_neuropil(zmesh, serra_mesh, volume_array, labels)
        experiment_builtin_on_real_data(serra_mesh, volume_array, labels)

    experiment_scale(serra_mesh)
    experiment_chunk_cost(serra_mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
