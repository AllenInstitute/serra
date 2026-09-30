"""Edge flips: what `edge_flips` does to triangle quality, and what it costs.

`edge_flips=k` runs up to k passes of edge flipping when `get()` triangulates a
surface. Only connectivity changes, so volume is unchanged by construction; the
question is how much the triangles improve and how long it takes.

Quality metrics are those of `bench/tangential.py`, plus the fraction of
vertices with the ideal six neighbours ("deg 6"). Measured over closed meshes
of a 256^3 corner of the MICrONS chunk, and on a sphere.

    python bench/flips.py
"""

from __future__ import annotations

import gzip
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serra_mesh  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def per_face(vertices, faces):
    """Area, shape quality and smallest angle (degrees) of every triangle."""
    p = vertices[faces].astype(np.float64)
    e = [p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(e[0], -e[2]), axis=1)
    l2 = np.stack([(x * x).sum(1) for x in e], axis=1)
    q = 4.0 * np.sqrt(3.0) * area / np.maximum(l2.sum(1), 1e-30)
    side = np.sort(np.sqrt(l2), axis=1)
    sin_min = 2.0 * area / np.maximum(side[:, 1] * side[:, 2], 1e-30)
    theta = np.degrees(np.arcsin(np.clip(sin_min, 0, 1)))
    return area, q, theta


CONFIGS = [
    ("none", {}),
    ("edge_flips=3", dict(edge_flips=3)),
    ("tangential=5", dict(fairing_tangential=5)),
    ("  +edge_flips=3", dict(fairing_tangential=5, edge_flips=3)),
    ("fairing=20+taubin", dict(fairing=20, fairing_taubin=True)),
    ("  +edge_flips=3", dict(fairing=20, fairing_taubin=True, edge_flips=3)),
    (
        "  +tangential=5+flips=3",
        dict(fairing=20, fairing_taubin=True, fairing_tangential=5, edge_flips=3),
    ),
]


def measure(mesher, labels):
    """Pooled statistics over every triangle of `labels`, accumulated one object
    at a time: holding a whole chunk's triangles in float64 at once needs
    several GB."""
    n = s1 = s2 = thin = 0.0
    qs, n6, nv, elapsed = [], 0, 0, 0.0
    for label in labels:
        start = time.perf_counter()
        m = mesher.get(int(label))
        elapsed += time.perf_counter() - start
        if len(m.faces) == 0:
            continue
        area, q, theta = per_face(m.vertices, m.faces)
        n += len(area)
        s1 += area.sum()
        s2 += (area * area).sum()
        thin += (theta < 20).sum()
        qs.append(q.astype(np.float32))
        deg = np.bincount(m.faces.ravel(), minlength=len(m.vertices))
        n6 += int((deg == 6).sum())
        nv += len(m.vertices)
    mean = s1 / n
    q = np.concatenate(qs)
    stats = {
        "area_cv": np.sqrt(max(s2 / n - mean * mean, 0.0)) / mean,
        "q_mean": float(q.mean()),
        "q_p1": float(np.percentile(q, 1)),
        "lt20": thin / n,
        "deg6": n6 / nv,
    }
    return stats, elapsed


def row(name, s, extra=""):
    print(
        f"| {name} | {s['area_cv']:.3f} | {s['q_mean']:.3f} | {s['q_p1']:.3f} | "
        f"{100 * s['lt20']:.2f}% | {100 * s['deg6']:.1f}% |{extra}"
    )


def run_sphere(radius=20.0):
    n = int(2 * radius) + 9
    g = np.arange(n) - (n - 1) / 2
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    mask = ((x * x + y * y + z * z) <= radius * radius).astype(np.uint8)
    print(f"\n### Sphere r={radius}\n")
    print("| settings | area CV | q mean | q p1 | <20° | deg 6 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, kw in CONFIGS:
        mesher = serra_mesh.Mesher(**kw).mesh(mask, close=True)
        s, _ = measure(mesher, [1])
        row(name, s)


def run_neuropil(resolution):
    a = np.load(gzip.open(os.path.join(DATA, "microns_neuropil.npy.gz")))[
        :256, :256, :256
    ]
    a = np.ascontiguousarray(a)
    print(f"\n### MICrONS neuropil {a.shape}, voxel_resolution={resolution}\n")
    print("| settings | area CV | q mean | q p1 | <20° | deg 6 | get() all |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    labels = None
    for name, kw in CONFIGS:
        mesher = serra_mesh.Mesher(voxel_resolution=resolution, **kw).mesh(
            a, close=True
        )
        if labels is None:
            labels = mesher.ids()
        s, elapsed = measure(mesher, labels)
        row(name, s, f" {elapsed:.2f} s |")


if __name__ == "__main__":
    run_sphere()
    run_neuropil([32, 32, 40])
    run_neuropil([4, 4, 40])
