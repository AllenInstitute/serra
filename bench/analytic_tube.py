"""Meshes judged against an analytic solid, not against the voxels.

Every other accuracy measurement in this repository compares a mesh either to
the voxel grid it was built from or to a finer grid. Both are quantised, so both
confound "the mesh is wrong" with "the reference is blocky". This one has no
grid in the reference at all.

A tube is defined by a smooth closed space curve C(t) and a radius R(t). A point
is inside when its distance to the curve is less than the radius there, and that
question can be answered exactly for any floating-point coordinate. The
voxelisation is the usual thing -- a voxel is labelled by whether its *centre*
is inside -- and is deliberately the only quantised object in the experiment.

Two things are then measured against the analytic solid:

* **Surface accuracy.** Sample points on the mesh, area-weighted, and evaluate
  the analytic signed distance at each. A perfect mesh scores zero. The mean
  absolute value is how far the surface sits from the truth, in voxels; the mean
  signed value is whether it sits systematically inside or outside.
* **Classification.** Sample points in the box, ask the analytic solid and the
  mesh independently, and count disagreements.

This is the setting most favourable to smoothing that exists: the true surface
really is smooth, and everything that distinguishes it from the voxelisation is
quantisation noise. If smoothing cannot recover a surface here it cannot recover
one anywhere.

    uv sync --group bench
    python bench/analytic_tube.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


class Tube:
    """A smooth closed curve with a radius, and exact inside/outside queries.

    The curve winds around the z axis while oscillating along it, so the surface
    presents every orientation to the voxel grid rather than favouring the axis
    planes the way a cylinder or a sphere does.

        C(t) = ( (major + wobble*cos(lobes*t)) * cos t,
                 (major + wobble*cos(lobes*t)) * sin t,
                 climb * sin(lobes*t) )
        R(t) = radius * (1 + flare*sin(2t))

    Distance is evaluated against a densely sampled polyline. At the default
    sampling the spacing is under a thousandth of a voxel, so the discretisation
    of the *reference* is three orders of magnitude finer than anything being
    measured.
    """

    def __init__(
        self,
        major=20.0,
        wobble=3.0,
        climb=9.0,
        lobes=3,
        radius=4.0,
        flare=0.0,
        samples=400_000,
    ):
        from scipy.spatial import cKDTree

        self.radius, self.flare = radius, flare
        t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        rho = major + wobble * np.cos(lobes * t)
        self.curve = np.stack(
            [rho * np.cos(t), rho * np.sin(t), climb * np.sin(lobes * t)], axis=1
        )
        self.radii = radius * (1.0 + flare * np.sin(2.0 * t))
        self.tree = cKDTree(self.curve)

        step = np.linalg.norm(np.diff(self.curve, axis=0), axis=1)
        self.spacing = float(step.max())
        self.length = float(step.sum())

    def signed_distance(self, points, chunk=500_000):
        """Distance to the surface: negative inside, positive outside.

        With a varying radius this is the normal-projection tube — the radius is
        taken at the nearest point on the curve. For the union-of-balls reading
        to differ, the radius would have to change faster along the curve than
        the surface turns; `flare` is kept well below that.
        """
        out = np.empty(len(points))
        for i in range(0, len(points), chunk):
            block = np.ascontiguousarray(points[i : i + chunk], dtype=np.float64)
            d, j = self.tree.query(block, k=1, workers=-1)
            out[i : i + chunk] = d - self.radii[j]
        return out

    def inside(self, points):
        return self.signed_distance(points) < 0.0

    def voxelize(self, shape, origin):
        """Label a voxel by whether its centre is inside. The only quantisation."""
        grid = np.stack(
            np.meshgrid(
                *[np.arange(n, dtype=np.float64) for n in shape], indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, 3)
        inside = self.inside(grid + origin)
        return inside.reshape(shape).astype(np.uint32)

    def volume(self, rng, samples=4_000_000, lo=None, hi=None):
        """Monte Carlo volume, in cubic voxels, with its own standard error."""
        points = rng.uniform(lo, hi, size=(samples, 3))
        hits = self.inside(points)
        box = float(np.prod(np.asarray(hi) - np.asarray(lo)))
        p = float(hits.mean())
        return box * p, box * float(np.sqrt(p * (1 - p) / samples))


def contour(field, level):
    """Marching cubes on a continuous field, via VTK's flying edges.

    Points come out in index coordinates -- origin 0, unit spacing -- which is
    the same convention serra uses, so nothing needs shifting.
    """
    import pyvista as pv

    grid = pv.ImageData(dimensions=field.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.point_data["f"] = np.asarray(field, np.float32).flatten(order="F")
    surf = grid.contour([level], scalars="f").triangulate()
    faces = np.asarray(surf.faces).reshape(-1, 4)[:, 1:]
    return np.asarray(surf.points, np.float64), faces.astype(np.uint32)


def signed_edt(mask):
    """Signed Euclidean distance, positive inside.

    Marching cubes on a *binary* field always interpolates to the midpoint of a
    crossed edge, so every vertex lands on a half-voxel lattice however smooth
    the true surface is. On a distance field the crossing varies continuously
    and the vertex can land anywhere -- which is the cheapest route to sub-voxel
    placement from a binary mask, and the idea behind Gibson's distance maps.
    """
    from scipy.ndimage import distance_transform_edt

    inside = mask.astype(bool)
    return distance_transform_edt(inside) - distance_transform_edt(~inside)


def sign_constrained_blur(mask, iterations=12, sigma=0.8, margin=0.02, bound=0.5):
    """Gaussian blur of the embedding with the voxel signs pinned.

    This is the obvious thing to reach for, and it is NOT Whitaker's constrained
    anti-aliasing (VolVis 2000, shipped as ITK's AntiAliasBinaryImageFilter).
    That method moves the level set under *curvature* flow inside a narrow band;
    an isotropic Gaussian is a poor stand-in, because on a tube two voxels across
    it averages right through the object. It is included because the failure is
    instructive, not as a test of the published method -- see the note in
    docs/accuracy.md. Do not read the row below as evidence about Whitaker.
    """
    from scipy.ndimage import gaussian_filter

    inside = mask.astype(bool)
    phi = np.where(inside, bound, -bound).astype(np.float64)
    for _ in range(iterations):
        phi = gaussian_filter(phi, sigma)
        # Clamp BOTH ends, not just the sign. Pinning inside voxels to a tiny
        # positive value while outside voxels keep a large negative one drags
        # every crossing towards the inside voxel centre -- half a voxel of
        # inward bias, which is worse than not smoothing. Bounding the magnitude
        # symmetrically keeps the crossing near where the data put it.
        phi = np.where(
            inside,
            np.clip(phi, margin, bound),
            np.clip(phi, -bound, -margin),
        )
    return phi


def sample_surface(vertices, faces, rng, samples):
    """Points spread uniformly over the mesh surface, area-weighted."""
    v = np.asarray(vertices, np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    pick = rng.choice(len(faces), size=samples, p=areas / areas.sum())
    u = rng.random((samples, 1))
    w = rng.random((samples, 1))
    over = (u + w) > 1.0
    u[over], w[over] = 1.0 - u[over], 1.0 - w[over]
    return a[pick] + u * (b[pick] - a[pick]) + w * (c[pick] - a[pick])


def enclosed_volume(vertices, faces):
    v = np.asarray(vertices, np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def classification(vertices, faces, points, truth):
    import igl

    w = igl.fast_winding_number(
        np.ascontiguousarray(vertices, np.float64),
        np.ascontiguousarray(faces, np.int64),
        np.ascontiguousarray(points, np.float64),
    )
    inside = w > 0.5
    return {
        "iou": float((inside & truth).sum()) / max(float((inside | truth).sum()), 1.0),
        "wrong": float((inside != truth).mean()),
        "too_big": float((inside & ~truth).sum()) / max(float(truth.sum()), 1.0),
        "too_small": float((~inside & truth).sum()) / max(float(truth.sum()), 1.0),
    }


def evaluate(tube, vertices, faces, points, truth, rng, surface_samples):
    on = sample_surface(vertices, faces, rng, surface_samples)
    d = tube.signed_distance(on)
    stats = classification(vertices, faces, points, truth)
    stats.update(
        mean_abs=float(np.abs(d).mean()),
        bias=float(d.mean()),
        p95=float(np.percentile(np.abs(d), 95)),
        worst=float(np.abs(d).max()),
    )
    return stats


def analytic_surface(tube, n_sides=64, along=1500):
    """The tube itself as a surface, for the reference panel."""
    import pyvista as pv

    step = max(1, len(tube.curve) // along)
    pts = np.vstack([tube.curve[::step], tube.curve[:1]])
    rad = np.concatenate([tube.radii[::step], tube.radii[:1]])
    line = pv.PolyData(pts)
    line.lines = np.hstack([[len(pts)], np.arange(len(pts))])
    line["radius"] = rad
    return line.tube(scalars="radius", absolute=True, n_sides=n_sides, capping=False)


def colour_bar(width, limit, font, height=52, cmap="coolwarm"):
    """A labelled gradient, so "white is zero" is shown rather than asserted."""
    import matplotlib

    bar = np.full((height, width, 3), 22, np.uint8)
    band = 22
    ramp = np.linspace(0.0, 1.0, width)
    rgb = (np.asarray(matplotlib.colormaps[cmap](ramp))[:, :3] * 255).astype(np.uint8)
    bar[6 : 6 + band] = rgb[None, :, :]

    def write(x, text, scale=2):
        for ch in text.upper():
            pattern = font.get(ch)
            if pattern is None:
                x += 4 * scale
                continue
            for row, bits in enumerate(pattern):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        y0 = 6 + band + 6 + row * scale
                        bar[
                            y0 : y0 + scale, x + col * scale : x + (col + 1) * scale
                        ] = 210
            x += (len(pattern[0]) + 1) * scale

    left = f"{-limit:.2f} VOXEL INSIDE"
    right = f"{limit:.2f} VOXEL OUTSIDE"
    write(8, left)
    write(width // 2 - 14, "0")
    write(width - 12 * len(right) - 8, right)
    return bar


def render_figure(tube, panels, origin, path, size=460, detail_span=7.0):
    """Three rows: the whole loop, a close-up, and the close-up coloured by error."""
    import pyvista as pv

    from render_comparison import (
        FONT,
        beside,
        label_strip,
        panel_shape,
        render_panel,
        save_png,
        stack,
        to_polydata,
    )

    pv.OFF_SCREEN = True
    truth = analytic_surface(tube)
    everything = np.vstack([truth.points] + [v for _, v, _ in panels])
    lo, hi = everything.min(0), everything.max(0)
    bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

    # Aim the close-up at a real point on the curve, away from the box edge.
    focus = tube.curve[len(tube.curve) // 8]
    detail = (focus, detail_span)
    over_shape = panel_shape(bounds, size)

    colours = ["#b9a06a", "#c98f7a", "#8fa9d0", "#a98fd0", "#8fd0ab", "#d0c48f"]

    # One symmetric range for every panel, wide enough that nothing clips.
    # A fixed range looks tidy but saturates the worst mesh to a flat block of
    # colour, which reads as "uniformly a little wrong" when it is in fact
    # "uniformly a lot wrong" -- exactly the case the figure needs to show.
    pooled = np.concatenate(
        [
            np.abs(tube.signed_distance(np.asarray(v)))
            for _, v, f in panels
            if f is not None
        ]
    )
    # A pooled percentile rather than the worst panel's maximum: keyed to the
    # worst mesh, every good mesh washes out to white; keyed to the best, the
    # worst saturates to a flat block. The 98th percentile of everything shown
    # keeps structure visible in both.
    limit = float(np.ceil(np.percentile(pooled, 98) * 20.0) / 20.0)

    over, close, err = [], [], []
    entries = [("analytic tube", np.asarray(truth.points), None)] + list(panels)
    for i, (name, v, f) in enumerate(entries):
        poly = truth if f is None else to_polydata(v, f)
        colour = colours[i % len(colours)]
        over.append(
            stack(
                [
                    render_panel(poly, over_shape, colour, bounds, zoom_out=1.1),
                    label_strip(over_shape[0], name),
                ],
                gap=0,
            )
        )
        close.append(render_panel(poly, (size, size), colour, bounds, detail=detail))
        if f is None:
            err.append(np.full((size, size, 3), 22, np.uint8))
            continue
        d = tube.signed_distance(np.asarray(poly.points, np.float64))
        err.append(
            render_panel(
                poly,
                (size, size),
                colour,
                bounds,
                detail=detail,
                scalars=d,
                cmap="coolwarm",
                clim=(-limit, limit),
            )
        )

    width = size * len(entries) + 8 * (len(entries) - 1)
    figure = stack(
        [
            label_strip(
                width,
                f"analytic tube, radius {tube.radius:g} +/- {tube.flare:.0%} voxels "
                f"- voxelised by centre - meshed at 1 voxel",
            ),
            beside(over),
            label_strip(width, f"close-up spanning {2 * detail_span:.0f} voxels"),
            beside(close),
            label_strip(
                width,
                "same close-up, coloured by signed distance from the true surface",
            ),
            beside(err),
            colour_bar(width, limit, FONT),
        ]
    )
    save_png(path, figure)
    print(f"  wrote {path} ({figure.shape[1]}x{figure.shape[0]})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, nargs="*", default=[2.0, 4.0, 8.0])
    parser.add_argument("--flare", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=2_000_000)
    parser.add_argument("--surface-samples", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zmesh", default="/Users/forrestc/ConnectomeStack/zmesh")
    parser.add_argument("--render", action="store_true", help="also write a figure")
    parser.add_argument("--out", default="docs/images")
    args = parser.parse_args()

    sys.path.insert(0, args.zmesh)
    import zmesh

    import serra_mesh

    for radius in args.radius:
        tube = Tube(radius=radius, flare=args.flare)
        pad = radius + 4.0
        lo = tube.curve.min(0) - pad
        hi = tube.curve.max(0) + pad
        origin = np.floor(lo)
        shape = tuple(int(n) for n in np.ceil(hi - origin) + 1)

        labels = tube.voxelize(shape, origin)
        rng = np.random.default_rng(args.seed)
        truth_volume, err = tube.volume(rng, lo=lo, hi=hi)

        print("=" * 90)
        print(
            f"tube radius {radius} voxels, flare {args.flare:+.0%}, "
            f"curve length {tube.length:.1f} voxels"
        )
        print(
            f"  grid {shape}, {labels.sum():,} voxels inside; reference polyline "
            f"spacing {tube.spacing:.2e} voxels"
        )
        print(
            f"  analytic volume {truth_volume:,.0f} +/- {err:,.0f} cubic voxels; "
            f"voxel count says {labels.sum():,} "
            f"({labels.sum() / truth_volume - 1:+.2%})"
        )

        # One set of query points for every mesh, so the comparison is paired.
        points = np.random.default_rng(args.seed + 1).uniform(
            lo, hi, size=(args.samples, 3)
        )
        truth = tube.inside(points)

        variants = [("zmesh (marching cubes)", None)]
        variants += [(f"volume: {n}", ("field", fn)) for n, fn in [
            ("marching cubes on binary", lambda m: (m.astype(np.float32), 0.5)),
            ("gaussian blur, sigma 1", lambda m: (
                __import__("scipy.ndimage", fromlist=["g"]).gaussian_filter(
                    m.astype(np.float32), 1.0), 0.5)),
            ("signed distance field", lambda m: (signed_edt(m), 0.0)),
            ("gaussian + sign pin (naive)", lambda m: (sign_constrained_blur(m), 0.0)),
        ]]
        variants += [("serra, no smoothing", {})]
        variants += [
            (f"serra relaxation={k}", dict(relaxation=k)) for k in (1, 3, 5, 10)
        ]
        variants += [(f"serra taubin={k}", dict(taubin=k)) for k in (3, 10, 20, 40)]

        print(
            f"\n{'mesh':>26} {'faces':>9} {'|error|':>9} {'bias':>9} {'p95':>8} "
            f"{'volume':>9} {'IoU':>9} {'wrong':>8}"
        )
        print(f"{'':>26} {'':>9} {'distance from the true surface, voxels':>37}")
        meshes = {}
        for name, kwargs in variants:
            if isinstance(kwargs, tuple):
                field, level = kwargs[1](labels)
                v, f = contour(field, level)
            elif kwargs is None:
                m = zmesh.Mesher((1, 1, 1))
                m.mesh(labels, close=True)
                got = m.get(1)
                # zmesh's `close` pads the volume by one voxel and does not
                # subtract the pad again, so its output sits a whole voxel from
                # serra's. Measured on a sphere: centroid (33,33,33) against
                # (32,32,32). Left uncorrected it would report a coordinate
                # convention as an accuracy difference.
                v = np.asarray(got.vertices, np.float64) - 1.0
                f = got.faces
            else:
                try:
                    m = serra_mesh.Mesher(voxel_resolution=[1.0, 1.0, 1.0], **kwargs)
                except (TypeError, ValueError):
                    continue
                m.mesh(labels, close=True)
                got = m.get(1)
                v, f = np.asarray(got.vertices, np.float64), got.faces
            v = v + origin
            s = evaluate(
                tube,
                v,
                f,
                points,
                truth,
                np.random.default_rng(args.seed + 2),
                args.surface_samples,
            )
            print(
                f"{name:>26} {len(f):>9,} {s['mean_abs']:>8.4f} {s['bias']:>+8.4f} "
                f"{s['p95']:>7.4f} "
                f"{enclosed_volume(v, f) / truth_volume - 1:>+8.2%} "
                f"{s['iou']:>8.3%} {s['wrong']:>7.3%}"
            )
            meshes[name] = (v, f)

        if args.render:
            import os

            os.makedirs(args.out, exist_ok=True)
            wanted = [
                "zmesh (marching cubes)",
                "serra, no smoothing",
                "serra relaxation=10",
                "serra taubin=20",
            ]
            render_figure(
                tube,
                [(n, *meshes[n]) for n in wanted if n in meshes],
                origin,
                os.path.join(args.out, f"analytic_tube_r{radius:g}.png"),
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
