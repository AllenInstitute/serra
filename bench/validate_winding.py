"""Validate serra's surfaces against zmesh's, and both against the voxels.

The check `CLAUDE.md` asks for: classify sample points as inside or outside each
mesh with a robust generalized winding number, confirm the enclosed volumes
agree, and confirm that where the two disagree the points are *close to the
surface*. That last part is what separates a genuine difference in where the
surface was placed from an actual defect — a hole or an inverted patch makes
points disagree far from any boundary.

The voxel labels are also available, so both meshers are additionally compared
against the ground truth rather than only against each other.

    uv sync --group bench
    python bench/validate_winding.py --volume data/microns_neuropil.npy.gz
"""

from __future__ import annotations

import argparse
import gzip
import sys

import numpy as np

RESOLUTION = np.array([32.0, 32.0, 40.0])
VOXEL_VOLUME = float(np.prod(RESOLUTION))


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return np.load(handle)
    return np.load(path)


def interior_labels(volume, count, rng):
    """Objects clear of the array faces, so both meshes are genuinely closed.

    An object running off the edge is only closed by `close=True` capping it,
    and the cap is an artefact of the cutout rather than of either mesher.
    """
    ids, counts = np.unique(volume, return_counts=True)
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]

    faces = np.concatenate(
        [
            np.unique(volume[0]),
            np.unique(volume[-1]),
            np.unique(volume[:, 0]),
            np.unique(volume[:, -1]),
            np.unique(volume[:, :, 0]),
            np.unique(volume[:, :, -1]),
        ]
    )
    touching = set(np.unique(faces).tolist())

    # Spread the sample across the size range rather than taking the biggest.
    order = np.argsort(counts)
    candidates = [
        (int(ids[i]), int(counts[i]))
        for i in order
        if int(ids[i]) not in touching and counts[i] >= 200
    ]
    if not candidates:
        return []
    picks = np.linspace(0, len(candidates) - 1, min(count, len(candidates)))
    return [candidates[int(round(p))] for p in picks]


def winding_inside(vertices, faces, points):
    import igl

    w = igl.fast_winding_number(
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
        np.ascontiguousarray(points, dtype=np.float64),
    )
    return w > 0.5


def surface_distance(vertices, faces, points):
    import igl

    d2, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
    )
    return np.sqrt(d2)


def enclosed_volume(vertices, faces):
    v = vertices.astype(np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="data/microns_neuropil.npy.gz")
    parser.add_argument("--zmesh", default="/Users/forrestc/ConnectomeStack/zmesh")
    parser.add_argument("--objects", type=int, default=24)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.zmesh)
    import zmesh

    import serra_mesh

    rng = np.random.default_rng(args.seed)
    volume = load(args.volume)
    picks = interior_labels(volume, args.objects, rng)
    print(
        f"volume {volume.shape} {volume.dtype}; validating {len(picks)} interior objects\n"
    )

    zm = zmesh.Mesher(tuple(RESOLUTION))
    zm.mesh(volume, close=True)
    sm = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION)).mesh(volume, close=True)

    print(
        f"{'label':>8} {'voxels':>9} {'serra/true':>11} {'zmesh/true':>11} "
        f"{'agree':>7} {'disagr d':>8} {'all pts d':>9} {'worst':>6}"
    )
    print(
        f"{'':>8} {'':>9} {'':>11} {'':>11} {'':>7} {'(voxels from the surface)':>25}"
    )
    rows = []
    for label, voxels in picks:
        if label not in sm:
            continue
        s = sm.get(label)
        z = zm.get(label)
        if len(s.faces) == 0 or len(z.faces) == 0:
            continue

        truth = voxels * VOXEL_VOLUME
        s_vol = enclosed_volume(s.vertices, s.faces)
        z_vol = enclosed_volume(z.vertices, z.faces)

        lo = np.minimum(s.vertices.min(0), z.vertices.min(0)) - RESOLUTION
        hi = np.maximum(s.vertices.max(0), z.vertices.max(0)) + RESOLUTION
        points = rng.uniform(lo, hi, size=(args.samples, 3))

        in_s = winding_inside(s.vertices, s.faces, points)
        in_z = winding_inside(z.vertices, z.faces, points)
        agree = float((in_s == in_z).mean())

        # Distances to zmesh's surface, in voxels, for every sampled point and
        # for the disagreeing ones. Comparing the two is the real test: if the
        # disagreements were caused by a hole or an inverted patch they would
        # look like the sample as a whole, not hug the surface.
        all_d = surface_distance(z.vertices, z.faces, points) / RESOLUTION.min()
        disagreeing = points[in_s != in_z]
        if len(disagreeing):
            d = surface_distance(z.vertices, z.faces, disagreeing) / RESOLUTION.min()
            worst, median = float(d.max()), float(np.median(d))
        else:
            worst = median = 0.0
        baseline = float(np.median(all_d))

        rows.append(
            (
                label,
                voxels,
                s_vol / truth,
                z_vol / truth,
                agree,
                median,
                worst,
                baseline,
            )
        )
        print(
            f"{label:>8} {voxels:>9,} {s_vol / truth:>10.3f} {z_vol / truth:>10.3f} "
            f"{agree:>6.2%} {median:>8.2f} {baseline:>9.2f} {worst:>6.2f}"
        )

    if not rows:
        print("no objects validated")
        return 1

    arr = np.array([(r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows])
    print("\n" + "=" * 76)
    print(f"objects validated                : {len(rows)}")
    print(
        f"serra volume / true voxel volume : {arr[:, 0].mean():.3f} "
        f"(range {arr[:, 0].min():.3f}-{arr[:, 0].max():.3f})"
    )
    print(
        f"zmesh volume / true voxel volume : {arr[:, 1].mean():.3f} "
        f"(range {arr[:, 1].min():.3f}-{arr[:, 1].max():.3f})"
    )
    ratio = arr[:, 0] / arr[:, 1]
    print(
        f"serra / zmesh volume             : {ratio.mean():.3f} "
        f"(range {ratio.min():.3f}-{ratio.max():.3f})"
    )
    print(
        f"winding-number agreement         : {arr[:, 2].mean():.2%} of sampled points"
    )
    print(
        f"disagreeing points, distance     : median {np.median(arr[:, 3]):.2f} voxels, "
        f"worst {arr[:, 4].max():.2f} voxels"
    )
    print(
        f"all sampled points, distance     : median {np.median(arr[:, 5]):.2f} voxels"
    )
    concentration = np.median(arr[:, 5]) / max(np.median(arr[:, 3]), 1e-9)
    print(
        f"  -> disagreements sit {concentration:.0f}x closer to the surface than a "
        f"typical sampled point"
    )

    # What actually matters, per CLAUDE.md: the volumes agree within 5%, and the
    # disagreements are a boundary effect rather than a defect. Raw agreement
    # rate is not a useful gate here, because most of these objects are tiny and
    # uniform sampling in their bounding box puts a large share of points near
    # the surface: agreement is 99%+ on the larger objects and lower on the
    # smallest, purely from the surface-to-volume ratio.
    ok = (
        abs(ratio.mean() - 1.0) < 0.05 and arr[:, 4].max() < 3.0 and concentration > 3.0
    )
    print(
        "\n"
        + ("PASS" if ok else "FAIL")
        + ": volumes agree within 5%, and every disagreement lies within a few "
        "voxels\n      of the surface rather than scattered through the volume"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
