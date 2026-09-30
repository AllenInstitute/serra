"""Tangential fairing: does it even out the triangles without moving the surface?

`fairing_tangential=k` runs k extra sweeps after fairing. Each sweep moves every
cell toward its neighbour average, but only along the surface: sheet cells in
their tangent plane, junction-curve cells along the curve, and corners not at
all. This script measures what it buys (triangle quality) and what it costs
(time, and any change to the shape).

Triangle quality is measured in physical space, as the caller sees the mesh:

* **area CV** -- standard deviation of triangle area over its mean. Lower is a
  tighter distribution.
* **q** -- 4*sqrt(3)*area / (sum of squared edge lengths): 1 for an equilateral
  triangle, 0 for a degenerate one. Reported as the mean and 1st percentile.
* **<20 deg** -- fraction of triangles whose smallest angle is under 20 degrees.

Shape is measured against the same settings without the tangential sweeps: the
change in each object's enclosed volume (median and 99th percentile over
objects; the tail is objects of a voxel or two, where a few percent is a few
hundredths of a voxel), and on the sphere the error against the exact volume
and area.

    python bench/tangential.py
"""

from __future__ import annotations

import gzip
import os
import time

import numpy as np

import serra_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def triangle_quality(vertices, faces):
    """Area CV, mean and 1st-percentile q, and fraction under 20 degrees."""
    p = vertices[faces].astype(np.float64)
    e0 = p[:, 1] - p[:, 0]
    e1 = p[:, 2] - p[:, 1]
    e2 = p[:, 0] - p[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(e0, -e2), axis=1)
    l2 = np.stack([(e * e).sum(1) for e in (e0, e1, e2)], axis=1)
    q = 4.0 * np.sqrt(3.0) * area / np.maximum(l2.sum(1), 1e-30)
    # The smallest angle is opposite the shortest edge, so
    # sin(theta_min) = 2*area / (product of the two edges enclosing it)
    order = np.sort(np.sqrt(l2), axis=1)
    sin_min = 2.0 * area / np.maximum(order[:, 1] * order[:, 2], 1e-30)
    theta = np.degrees(np.arcsin(np.clip(sin_min, 0, 1)))
    return {
        "area_cv": area.std() / area.mean(),
        "q_mean": q.mean(),
        "q_p1": np.percentile(q, 1),
        "lt20": (theta < 20).mean(),
    }


def volume(vertices, faces):
    p = vertices[faces].astype(np.float64)
    return np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0


def area(vertices, faces):
    p = vertices[faces].astype(np.float64)
    return (
        0.5
        * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum()
    )


def pooled(mesher, labels):
    """Quality over every triangle of `labels`, and each label's volume."""
    vs, fs, vols = [], [], {}
    offset = 0
    for label in labels:
        m = mesher.get(int(label))
        if len(m.faces) == 0:
            continue
        vs.append(m.vertices)
        fs.append(m.faces + offset)
        offset += len(m.vertices)
        vols[int(label)] = volume(m.vertices, m.faces)
    return triangle_quality(np.concatenate(vs), np.concatenate(fs)), vols


# (name, settings, which earlier row the volume change is measured against)
CONFIGS = [
    ("none", {}, None),
    ("tangential=5", dict(fairing_tangential=5), "none"),
    ("fairing=20+taubin", dict(fairing=20, fairing_taubin=True), None),
    (
        "  +tangential=5",
        dict(fairing=20, fairing_taubin=True, fairing_tangential=5),
        "fairing=20+taubin",
    ),
    (
        "  +tangential=10",
        dict(fairing=20, fairing_taubin=True, fairing_tangential=10),
        "fairing=20+taubin",
    ),
    ("fairing=10", dict(fairing=10), None),
    ("  +tangential=5 ", dict(fairing=10, fairing_tangential=5), "fairing=10"),
]


def header():
    print(f"| {'settings':<20} | area CV | q mean | q p1 | <20° | vol vs base | time |")
    print("| --- | --- | --- | --- | --- | --- | --- |")


def run_neuropil(resolution):
    # A 256^3 corner of the 512^3 chunk: pooling every triangle of a full
    # chunk for the statistics would need several GB.
    a = np.load(gzip.open(os.path.join(DATA, "microns_neuropil.npy.gz")))[
        :256, :256, :256
    ]
    a = np.ascontiguousarray(a)
    print(f"\n### MICrONS neuropil {a.shape}, voxel_resolution={resolution}\n")
    header()
    labels = None
    volumes = {}
    for name, kw, against in CONFIGS:
        mesher = serra_mesh.Mesher(voxel_resolution=resolution, **kw)
        t = time.perf_counter()
        # Closed, because an object cut open by the crop has no enclosed volume
        # to compare: its signed volume depends on where the origin is.
        mesher.mesh(a, close=True)
        dt = time.perf_counter() - t
        if labels is None:
            ids = mesher.ids()
            # Every 5th object keeps the run short while spanning all sizes.
            labels = ids[ids != 0][::5]
        stats, vols = pooled(mesher, labels)
        volumes[name] = vols
        # Volume change against the same settings minus the tangential sweeps,
        # as |relative change| over objects.
        dv = "—"
        if against is not None:
            base = volumes[against]
            rel = [abs(vols[k] / base[k] - 1) for k in vols if base.get(k, 0) > 0]
            dv = f"{100 * np.median(rel):.3f}% med, {100 * np.percentile(rel, 99):.2f}% p99"
        print(
            f"| {name:<20} | {stats['area_cv']:.3f} | {stats['q_mean']:.3f} | "
            f"{stats['q_p1']:.3f} | {100 * stats['lt20']:.1f}% | {dv} | {dt:.2f} s |"
        )


def run_sphere(radius=20.0):
    n = int(2 * radius) + 9
    g = np.arange(n) - (n - 1) / 2
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    mask = ((x * x + y * y + z * z) <= radius * radius).astype(np.uint8)
    true_v = 4 / 3 * np.pi * radius**3
    true_a = 4 * np.pi * radius**2
    print(f"\n### Sphere r={radius}\n")
    print("| settings | area CV | q mean | q p1 | <20° | volume err | area err |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, kw, _ in CONFIGS:
        m = serra_mesh.Mesher(**kw).mesh(mask, close=True).get(1)
        s = triangle_quality(m.vertices, m.faces)
        print(
            f"| {name} | {s['area_cv']:.3f} | {s['q_mean']:.3f} | {s['q_p1']:.3f} | "
            f"{100 * s['lt20']:.1f}% | {100 * (volume(m.vertices, m.faces) / true_v - 1):+.2f}% | "
            f"{100 * (area(m.vertices, m.faces) / true_a - 1):+.2f}% |"
        )


if __name__ == "__main__":
    run_sphere()
    run_neuropil([32, 32, 40])
    run_neuropil([4, 4, 40])
