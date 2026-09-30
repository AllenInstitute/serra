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

from tangential import triangle_quality  # noqa: E402

import serra_mesh  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def degree6(faces, n):
    """Fraction of vertices with exactly six neighbours, on closed meshes."""
    deg = np.bincount(faces.ravel(), minlength=n)
    return (deg == 6).mean()


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
    vs, fs, n6, total = [], [], 0.0, 0
    offset = 0
    start = time.perf_counter()
    meshes = [mesher.get(int(label)) for label in labels]
    elapsed = time.perf_counter() - start
    for m in meshes:
        if len(m.faces) == 0:
            continue
        vs.append(m.vertices)
        fs.append(m.faces + offset)
        n6 += degree6(m.faces, len(m.vertices)) * len(m.vertices)
        total += len(m.vertices)
        offset += len(m.vertices)
    stats = triangle_quality(np.concatenate(vs), np.concatenate(fs))
    stats["deg6"] = n6 / total
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
