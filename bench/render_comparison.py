"""Render side-by-side comparison figures of zmesh against serra.

Meshes the three largest objects in the connectomics test volume with each
implementation and renders them from an identical camera, so any difference in
the image is a difference in the geometry.

Flat shading is deliberate: it makes individual triangles visible, which is what
distinguishes a staircased surface from a smooth one. Smooth (Phong) shading
would hide precisely the artefact under comparison.

zmesh is imported from a local checkout rather than PyPI so the comparison runs
against the build you actually have. Point --zmesh at it.

    uv sync --group bench          # pyvista is not a default dependency
    python bench/render_comparison.py --zmesh ../zmesh --out docs/images
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys

import numpy as np

RESOLUTION = (32.0, 32.0, 40.0)  # nanometres per voxel, per array axis
DEFAULT_VOLUME = "/Users/forrestc/ConnectomeStack/zmesh/connectomics.npy.gz"

# A single fixed viewing direction, so every panel is directly comparable.
VIEW_DIRECTION = np.array([0.55, -0.75, 0.38])
UP = (0.0, 0.0, 1.0)


def load(path: str) -> np.ndarray:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return np.load(handle)
    return np.load(path)


def largest_labels(volume: np.ndarray, count: int) -> list[tuple[int, int]]:
    """The `count` most voluminous objects, as (label, voxel count)."""
    labels, counts = np.unique(volume, return_counts=True)
    keep = labels != 0
    labels, counts = labels[keep], counts[keep]
    order = np.argsort(counts)[::-1][:count]
    return [(int(labels[i]), int(counts[i])) for i in order]


def to_polydata(vertices: np.ndarray, faces: np.ndarray):
    import pyvista as pv

    padded = np.hstack(
        [np.full((len(faces), 1), 3, np.int64), faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(vertices.astype(np.float64), padded)


def render_panel(mesh, size, colour, bounds, detail=None):
    """One panel.

    `bounds` frames the whole object and is shared by every panel in a row, so
    the three implementations are shown at identical scale. Passing `detail` as
    (focal_point, half_width_nm) instead renders a close-up of a fixed physical
    region, again identical across panels.
    """
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background("#16181c")
    plotter.add_mesh(
        mesh,
        color=colour,
        smooth_shading=False,
        specular=0.35,
        specular_power=18,
        ambient=0.22,
        diffuse=0.85,
    )
    plotter.enable_anti_aliasing("ssaa")
    plotter.camera.parallel_projection = True
    plotter.view_vector(VIEW_DIRECTION, viewup=UP)
    plotter.reset_camera(bounds=bounds)

    if detail is not None:
        focal, half_width = detail
        plotter.camera.focal_point = tuple(focal)
        plotter.camera.parallel_scale = half_width

    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def label_strip(width: int, text: str, height: int = 34) -> np.ndarray:
    """A caption bar drawn with a tiny built-in bitmap font."""
    strip = np.full((height, width, 3), 22, np.uint8)
    glyphs = FONT
    scale = 2
    x = 8
    for ch in text.upper():
        pattern = glyphs.get(ch)
        if pattern is None:
            x += 4 * scale
            continue
        for row, bits in enumerate(pattern):
            for col, bit in enumerate(bits):
                if bit == "1":
                    y0 = 8 + row * scale
                    x0 = x + col * scale
                    strip[y0 : y0 + scale, x0 : x0 + scale] = 225
        x += (len(pattern[0]) + 1) * scale
        if x > width - 20:
            break
    return strip


# 5x7 bitmap font, enough for the captions used here.
FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10011", "01111"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "00100", "01000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def stack(rows, gap=8, colour=22):
    width = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < width:
            pad = np.full((r.shape[0], width - r.shape[1], 3), colour, np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    spacer = np.full((gap, width, 3), colour, np.uint8)
    out = []
    for n, r in enumerate(padded):
        if n:
            out.append(spacer)
        out.append(r)
    return np.concatenate(out, axis=0)


def beside(images, gap=8, colour=22):
    height = max(i.shape[0] for i in images)
    padded = []
    for i in images:
        if i.shape[0] < height:
            pad = np.full((height - i.shape[0], i.shape[1], 3), colour, np.uint8)
            i = np.concatenate([i, pad], axis=0)
        padded.append(i)
    spacer = np.full((height, gap, 3), colour, np.uint8)
    out = []
    for n, i in enumerate(padded):
        if n:
            out.append(spacer)
        out.append(i)
    return np.concatenate(out, axis=1)


def save_png(path: str, image: np.ndarray) -> None:
    import struct
    import zlib

    rgb = np.ascontiguousarray(image[:, :, :3].astype(np.uint8))
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    blob = b"\x89PNG\r\n\x1a\n"
    blob += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(raw, 9))
    blob += chunk(b"IEND", b"")
    with open(path, "wb") as handle:
        handle.write(blob)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--zmesh", default="/Users/forrestc/ConnectomeStack/zmesh")
    parser.add_argument("--out", default="docs/images")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--size", type=int, default=520)
    parser.add_argument(
        "--detail-nm",
        type=float,
        default=700.0,
        help="half-width in nanometres of the close-up row",
    )
    args = parser.parse_args()

    sys.path.insert(0, args.zmesh)
    import pyvista as pv
    import zmesh

    import serra

    pv.OFF_SCREEN = True
    os.makedirs(args.out, exist_ok=True)

    volume = load(args.volume)
    targets = largest_labels(volume, args.count)
    print(f"volume {volume.shape} {volume.dtype}")
    for label, voxels in targets:
        print(f"  label {label}: {voxels:,} voxels")

    print("meshing with zmesh ...")
    zm = zmesh.Mesher(RESOLUTION)
    zm.mesh(volume)

    print("meshing with serra (relaxation=0) ...")
    s0 = serra.Mesher(voxel_resolution=list(RESOLUTION)).mesh(volume)

    print("meshing with serra (relaxation=3) ...")
    s3 = serra.Mesher(voxel_resolution=list(RESOLUTION), relaxation=3).mesh(volume)

    variants = [
        ("zmesh (marching cubes)", lambda lab: zm.get(lab), "#c98f7a"),
        ("serra relaxation=0", lambda lab: s0.get(lab), "#8fa9d0"),
        ("serra relaxation=3", lambda lab: s3.get(lab), "#8fd0ab"),
    ]

    for rank, (label, voxels) in enumerate(targets, start=1):
        meshes = []
        for name, getter, colour in variants:
            mesh = getter(label)
            meshes.append((name, mesh, colour))
            print(
                f"  label {label} {name}: "
                f"{len(mesh.vertices):,} verts {len(mesh.faces):,} faces"
            )

        # Framing is computed once from the serra mesh and reused for every
        # panel, so the three implementations are drawn at identical scale.
        reference = meshes[1][1].vertices.astype(np.float64)
        # Percentiles rather than min/max: these objects often have a small
        # disconnected fragment far from the main body, and framing to the full
        # bounding box would shrink the part worth looking at.
        lo = np.percentile(reference, 1.5, axis=0)
        hi = np.percentile(reference, 98.5, axis=0)
        bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

        # Aim the close-up at a real point on the surface: the vertex nearest
        # the object's centroid, which is reliably in a dense region.
        centroid = reference.mean(0)
        nearest = int(np.argmin(((reference - centroid) ** 2).sum(1)))
        detail = (reference[nearest], args.detail_nm)

        overview_row, detail_row = [], []
        for name, mesh, colour in meshes:
            poly = to_polydata(mesh.vertices, mesh.faces)
            # These objects are wide and flat, so the overview gets a wide panel
            # while the close-up stays square.
            over = render_panel(
                poly, (args.size, int(args.size * 0.60)), colour, bounds
            )
            deep = render_panel(
                poly, (args.size, args.size), colour, bounds, detail=detail
            )
            caption = f"{name} - {len(mesh.faces) // 1000}K faces"
            overview_row.append(
                stack([over, label_strip(over.shape[1], caption)], gap=0)
            )
            detail_row.append(deep)

        figure = stack(
            [
                label_strip(
                    args.size * 3 + 16,
                    f"object {label} - {voxels // 1000}K voxels - whole object",
                ),
                beside(overview_row),
                label_strip(
                    args.size * 3 + 16,
                    f"same object, close-up spanning {2 * args.detail_nm:.0f} nm",
                ),
                beside(detail_row),
            ]
        )
        path = os.path.join(args.out, f"compare_{rank}_label{label}.png")
        save_png(path, figure)
        print(f"  wrote {path}  ({figure.shape[1]}x{figure.shape[0]})")


if __name__ == "__main__":
    main()
