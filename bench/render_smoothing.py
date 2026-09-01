"""Render one segment under each smoothing filter, from an identical camera.

The numbers in `bench/taubin.py` say serra's built-in lambda/mu filter lands
between VTK's Nuttall and Blackman windows on normal error while preserving
volume better than any of them. This is what that looks like.

Flat shading throughout, deliberately: it makes individual triangles visible,
which is what distinguishes a smoothed surface from a merely shaded one.

    uv sync --group bench
    python bench/render_smoothing.py

Defaults to the same 5 micron cube around segment 864691136144674612 that
`render_segment.py` uses, and shares its cache.
"""

from __future__ import annotations

import argparse
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
from render_segment import (  # noqa: E402
    DEFAULT_POINT_MIP0,
    DEFAULT_SEGMENT,
    RESOLUTION,
    fetch,
)
from taubin import taubin as vtk_taubin  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=int, default=DEFAULT_SEGMENT)
    parser.add_argument("--point", type=int, nargs=3, default=DEFAULT_POINT_MIP0)
    parser.add_argument("--micron", type=float, default=5.0)
    parser.add_argument("--cache", default="data/microns_segment_5um")
    parser.add_argument("--out", default="docs/images")
    parser.add_argument("--size", type=int, default=520)
    parser.add_argument("--detail-nm", type=float, default=500.0)
    parser.add_argument("--zoom-out", type=float, default=1.15)
    args = parser.parse_args()

    import pyvista as pv

    import serra_mesh

    pv.OFF_SCREEN = True
    os.makedirs(args.out, exist_ok=True)

    labels, origin, _, _ = fetch(args.segment, args.point, args.micron, args.cache)
    voxels = int((labels == args.segment).sum())
    print(f"cutout {labels.shape}, segment occupies {voxels:,} voxels")

    def meshed(**kwargs):
        mesher = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION), **kwargs)
        return mesher.mesh(labels).get(args.segment)

    plain = meshed()
    # VTK's post-hoc filter is applied to the unsmoothed surface, and one VTK
    # iteration is one pass — so 20 there is comparable with taubin=10 here,
    # which is 20 passes. Boundary smoothing off, the setting that holds an open
    # border still (it is documented backwards in pyvista; see #8860).
    nuttall = vtk_taubin(
        plain.vertices.astype(np.float64),
        plain.faces,
        n_iter=20,
        window="nuttall",
        boundary_smoothing=False,
    )

    panels = [
        ("no smoothing", plain.vertices.astype(np.float64), plain.faces, "#b9a06a"),
        ("vtk nuttall, 20 passes", nuttall[0], nuttall[1], "#c98f7a"),
        ("serra taubin=10, 20 passes", None, None, "#8fa9d0"),
        ("serra taubin=20, 40 passes", None, None, "#8fd0ab"),
    ]
    for index, iterations in ((2, 10), (3, 20)):
        got = meshed(taubin=iterations)
        name, _, _, colour = panels[index]
        panels[index] = (name, got.vertices.astype(np.float64), got.faces, colour)

    for name, v, f, _ in panels:
        print(f"  {name:28s} {len(v):>8,} verts {len(f):>8,} faces")

    # One framing for every panel, taken from the unsmoothed mesh so no filter
    # gets to choose its own crop.
    reference = panels[0][1]
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
        overview.append(stack([over, label_strip(over.shape[1], name)], gap=0))
        close.append(deep)

    width = args.size * len(panels) + 8 * (len(panels) - 1)
    figure = stack(
        [
            label_strip(
                width,
                f"segment {args.segment} - {args.micron:g} micron cutout - "
                f"{voxels // 1000}K voxels - same camera, flat shaded",
            ),
            beside(overview),
            label_strip(width, f"close-up spanning {2 * args.detail_nm:.0f} nm"),
            beside(close),
        ]
    )
    path = os.path.join(args.out, f"smoothing_{args.segment}.png")
    save_png(path, figure)
    print(f"wrote {path} ({figure.shape[1]}x{figure.shape[0]})")


if __name__ == "__main__":
    main()
