"""Render one MICrONS segment four ways, at the resolution the dataset serves.

Compares the mesh MICrONS actually publishes for a segment against zmesh and
serra remeshing the same voxels. The point of comparison is the *served* LOD-0
mesh: its face count inside the cutout sets the budget, and both meshers are
decimated to match it. Comparing at equal `max_error` instead would be
misleading, since the two decimate by different amounts at the same setting.

    uv sync --group bench
    python bench/render_segment.py

Defaults to a 5 micron cube around a point on segment 864691136144674612.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_comparison import (  # noqa: E402
    beside,
    label_strip,
    panel_shape,
    render_panel,
    save_png,
    stack,
    to_polydata,
)

CLOUDPATH = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
RESOLUTION = np.array([32.0, 32.0, 40.0])
DEFAULT_SEGMENT = 864691136144674612
DEFAULT_POINT_MIP0 = (87497, 59229, 21010)  # in 8 x 8 x 40 nm voxels
MIP0_RESOLUTION = np.array([8.0, 8.0, 40.0])


def fetch(segment, point_mip0, micron, cache):
    """Cutout plus the served LOD-0 mesh, cached so reruns are cheap."""
    from cloudvolume import CloudVolume

    centre_nm = np.asarray(point_mip0, dtype=float) * MIP0_RESOLUTION
    half = micron * 1000.0 / 2.0
    origin = np.round((centre_nm - half) / RESOLUTION).astype(int)
    size = np.round((2 * half) / RESOLUTION).astype(int)

    volume_path = f"{cache}.npy.gz"
    mesh_path = f"{cache}.served.npz"
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)

    if os.path.exists(volume_path):
        with gzip.open(volume_path, "rb") as handle:
            labels = np.load(handle)
    else:
        cv = CloudVolume(
            CLOUDPATH, mip=2, use_https=True, progress=False, fill_missing=True
        )
        labels = np.asarray(
            cv[
                origin[0] : origin[0] + size[0],
                origin[1] : origin[1] + size[1],
                origin[2] : origin[2] + size[2],
            ]
        ).squeeze()
        with gzip.open(volume_path, "wb", compresslevel=6) as handle:
            np.save(handle, labels)

    if os.path.exists(mesh_path):
        cached = np.load(mesh_path)
        served_v, served_f = cached["v"], cached["f"]
    else:
        cv = CloudVolume(
            CLOUDPATH, mip=2, use_https=True, progress=False, fill_missing=True
        )
        got = cv.mesh.get(segment, lod=0)
        served = got[segment] if isinstance(got, dict) else got
        served_v = np.asarray(served.vertices, dtype=np.float64)
        served_f = np.asarray(served.faces)
        np.savez_compressed(mesh_path, v=served_v, f=served_f)

    return labels, origin, served_v, served_f


def clip(vertices, faces, lo, hi):
    """Keep the faces whose centroid falls inside the box, and renumber."""
    centroid = vertices[faces].mean(axis=1)
    keep = np.all((centroid >= lo) & (centroid < hi), axis=1)
    kept = faces[keep]
    used, inverse = np.unique(kept.ravel(), return_inverse=True)
    return vertices[used], inverse.reshape(kept.shape).astype(np.uint32)


def decimate_to(mesher, label, target):
    """Drive a mesher to a face count rather than to an error bound."""
    full = len(mesher.get(label).faces)
    if target <= 0 or full <= target:
        return mesher.get(label)
    factor = max(2, int(round(full / target)))
    # A max_error large enough not to bind, so the face target decides.
    return mesher.get(label, reduction_factor=factor, max_error=1e9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=int, default=DEFAULT_SEGMENT)
    parser.add_argument(
        "--point",
        type=int,
        nargs=3,
        default=DEFAULT_POINT_MIP0,
        help="centre, in 8x8x40 nm voxels",
    )
    parser.add_argument("--micron", type=float, default=5.0)
    parser.add_argument("--cache", default="data/microns_segment_5um")
    parser.add_argument("--out", default="docs/images")
    parser.add_argument("--size", type=int, default=520)
    parser.add_argument("--detail-nm", type=float, default=700.0)
    parser.add_argument("--zoom-out", type=float, default=1.35)
    parser.add_argument(
        "--zmesh",
        default="/Users/forrestc/ConnectomeStack/zmesh",
        help="local zmesh checkout, so the comparison uses the build you have",
    )
    args = parser.parse_args()

    sys.path.insert(0, args.zmesh)

    import pyvista as pv
    import zmesh

    import serra_mesh

    pv.OFF_SCREEN = True
    os.makedirs(args.out, exist_ok=True)

    labels, origin, served_v, served_f = fetch(
        args.segment, args.point, args.micron, args.cache
    )
    lo_nm = origin * RESOLUTION
    hi_nm = lo_nm + np.array(labels.shape) * RESOLUTION
    voxels = int((labels == args.segment).sum())
    print(f"cutout {labels.shape} at {origin}, segment occupies {voxels:,} voxels")

    served_cv, served_cf = clip(served_v, served_f, lo_nm, hi_nm)
    # The served mesh is in absolute dataset nanometres; a mesher run on the
    # cutout returns coordinates relative to that cutout's corner. Without this
    # the panels are ~700 microns apart and the served one renders off-screen.
    served_cv = served_cv - lo_nm
    target = len(served_cf)
    print(f"served LOD-0 mesh in this box: {target:,} faces -> that is the budget")

    zm = zmesh.Mesher(tuple(RESOLUTION))
    zm.mesh(labels)
    s0 = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION)).mesh(labels)
    s3 = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION), relaxation=3).mesh(labels)

    panels = [
        ("MICrONS served LOD-0", served_cv.astype(np.float32), served_cf, "#b9a06a"),
    ]
    for name, mesher, colour in [
        ("zmesh (marching cubes)", zm, "#c98f7a"),
        ("serra relaxation=0", s0, "#8fa9d0"),
        ("serra relaxation=3", s3, "#8fd0ab"),
    ]:
        mesh = decimate_to(mesher, args.segment, target)
        panels.append((name, mesh.vertices, mesh.faces, colour))

    for name, v, f, _ in panels:
        print(f"  {name:26s} {len(v):>8,} verts {len(f):>8,} faces")

    # One framing for every panel, taken from the serra mesh.
    reference = panels[2][1].astype(np.float64)
    lo = np.percentile(reference, 0.1, axis=0)
    hi = np.percentile(reference, 99.9, axis=0)
    bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    centroid = reference.mean(0)
    nearest = int(np.argmin(((reference - centroid) ** 2).sum(1)))
    detail = (reference[nearest], args.detail_nm)

    over_shape = panel_shape(bounds, args.size)
    overview, close = [], []
    for name, v, f, colour in panels:
        poly = to_polydata(v, f)
        over = render_panel(poly, over_shape, colour, bounds, zoom_out=args.zoom_out)
        deep = render_panel(poly, (args.size, args.size), colour, bounds, detail=detail)
        overview.append(
            stack(
                [over, label_strip(over.shape[1], f"{name} - {len(f) // 1000}K faces")],
                gap=0,
            )
        )
        close.append(deep)

    width = args.size * len(panels) + 8 * (len(panels) - 1)
    figure = stack(
        [
            label_strip(
                width,
                f"segment {args.segment} - {args.micron:g} micron cutout - "
                f"{voxels // 1000}K voxels - all at the served LOD-0 face count",
            ),
            beside(overview),
            label_strip(width, f"close-up spanning {2 * args.detail_nm:.0f} nm"),
            beside(close),
        ]
    )
    path = os.path.join(args.out, f"segment_{args.segment}.png")
    save_png(path, figure)
    print(f"wrote {path} ({figure.shape[1]}x{figure.shape[0]})")


if __name__ == "__main__":
    main()
