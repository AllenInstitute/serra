"""How well do the meshes compress, once Draco has had them?

The expectation going in was that serra should win: Draco predicts each vertex
from its neighbours with a parallelogram rule and entropy-codes the residual, so
a smoother surface should leave smaller residuals, and marching cubes bakes
axis-aligned terracing into the surface.

It does not work out that way, and the second table here is why. Terracing is
not noise to the predictor — marching-cubes vertices land on a small, highly
repetitive set of positions, which is exactly what an entropy coder is good at,
while serra's sit at continuous sub-voxel positions whose residuals look like
noise. At a fine quantization bin that costs serra up to 2x the bytes per face.

What rescues it is that PyChunkedGraph does not use a fine bin. Its lattice is
floor(min(resolution)/sqrt(2)) = 21 nm, two thirds of a voxel, which is coarser
than everything that distinguishes the two surfaces: both collapse onto the same
lattice and the difference all but disappears.

Meshes are compared at matched face counts, not at matched decimation settings:
at the same `max_error` the two decimate by different amounts, so bytes per mesh
would mostly be measuring who produced more triangles.

Encoding uses PyChunkedGraph's own Draco settings, computed the way
`meshgen.py:557-611` does, so the numbers reflect what the pipeline would store.

    uv sync --group bench
    python bench/draco_compression.py
"""

from __future__ import annotations

import argparse
import gzip
import sys

import numpy as np

RESOLUTION = np.array([32.0, 32.0, 40.0])


def draco_settings(block_shape, resolution, high_padding=2):
    """Reproduce `get_draco_encoding_settings_for_chunk` (meshgen.py:578)."""
    min_range = float(max((np.asarray(block_shape) + high_padding) * resolution))
    max_bin = float(np.floor(min(resolution) / np.sqrt(2)))

    bits = int(np.ceil(np.log2(min_range / max_bin + 1)))
    while True:
        num_bins = 2**bits - 1
        bin_size = float(np.ceil(min_range / num_bins))
        rng = bin_size * num_bins
        if rng >= min_range + bin_size:
            break
        if bin_size == max_bin:
            bits += 1
        else:
            bin_size += 1
            rng += num_bins
            break
    return {
        "quantization_bits": bits,
        "quantization_range": rng,
        "quantization_origin": [0.0, 0.0, 0.0],
    }, bin_size


def encode(vertices, faces, settings):
    import DracoPy

    return DracoPy.encode_mesh_to_buffer(
        np.ascontiguousarray(vertices, dtype=np.float32).flatten("C"),
        np.ascontiguousarray(faces, dtype=np.uint32).flatten("C"),
        **settings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="data/microns_neuropil.npy.gz")
    parser.add_argument("--zmesh", default="/Users/forrestc/ConnectomeStack/zmesh")
    parser.add_argument("--objects", type=int, default=24)
    parser.add_argument("--low", type=float, default=70.0)
    parser.add_argument("--high", type=float, default=99.0)
    args = parser.parse_args()

    sys.path.insert(0, args.zmesh)
    import zmesh

    import serra_mesh

    with gzip.open(args.volume, "rb") as handle:
        volume = np.load(handle)

    settings, bin_size = draco_settings(volume.shape, RESOLUTION)
    print(
        f"Draco settings (PyChunkedGraph's): {settings['quantization_bits']} bits, "
        f"range {settings['quantization_range']:.0f} nm, bin {bin_size:.0f} nm "
        f"({bin_size / RESOLUTION[0]:.2f} voxel)\n"
    )

    # Objects spread across the upper size range: big enough that decimation is
    # meaningful, but not the handful of giants, whose bytes would otherwise be
    # the whole answer.
    ids, counts = np.unique(volume, return_counts=True)
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]
    order = np.argsort(counts)
    picks = np.linspace(args.low, args.high, args.objects) / 100.0 * (len(order) - 1)
    labels = [int(ids[order[int(round(p))]]) for p in picks]
    sizes = [int(counts[order[int(round(p))]]) for p in picks]
    print(
        f"{len(labels)} objects between the {args.low:g}th and {args.high:g}th size "
        f"percentile: {min(sizes):,}-{max(sizes):,} voxels\n"
    )

    zm = zmesh.Mesher(tuple(RESOLUTION))
    zm.mesh(volume)
    s0 = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION)).mesh(volume)
    s3 = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION), relaxation=3).mesh(volume)

    variants = [
        (
            "zmesh",
            lambda lab, r: (
                zm.get(lab, reduction_factor=r, max_error=1e9) if r > 1 else zm.get(lab)
            ),
        ),
        ("serra k=0", lambda lab, r: s0.get(lab, reduction_factor=r, max_error=1e9)),
        ("serra k=3", lambda lab, r: s3.get(lab, reduction_factor=r, max_error=1e9)),
    ]

    print(
        f"{'reduction':>10} {'mesher':>10} {'faces':>12} {'draco KB':>10} "
        f"{'bytes/face':>11} {'vs zmesh':>9}"
    )
    for reduction in (1, 2, 5, 10, 20, 50):
        base = None
        for name, get in variants:
            total_faces = total_bytes = 0
            for label in labels:
                mesh = get(label, reduction)
                if len(mesh.faces) == 0:
                    continue
                total_faces += len(mesh.faces)
                total_bytes += len(encode(mesh.vertices, mesh.faces, settings))
            per_face = total_bytes / max(total_faces, 1)
            if base is None:
                base = per_face
            print(
                f"{reduction:>9}x {name:>10} {total_faces:>12,} "
                f"{total_bytes / 1000:>9.0f} {per_face:>10.2f} "
                f"{per_face / base:>8.3f}"
            )
        print()

    # The same objects, undecimated, across quantization bins. This is the
    # measurement that explains the table above: the bin PyChunkedGraph uses is
    # coarse enough to erase the difference, and a finer one reverses it.
    print("undecimated, against the quantization bin:\n")
    print(
        f"{'bits':>5} {'bin nm':>8} {'zmesh B/f':>10} {'serra k=0':>10} "
        f"{'serra k=3':>10} {'k=3 vs zmesh':>13}"
    )
    for bits in (10, 12, 14, 16):
        sweep = dict(settings)
        sweep["quantization_bits"] = bits
        bin_nm = sweep["quantization_range"] / (2**bits - 1)
        per_face = []
        for _, mesher in (("zmesh", zm), ("k=0", s0), ("k=3", s3)):
            total_faces = total_bytes = 0
            for label in labels:
                mesh = mesher.get(label)
                total_faces += len(mesh.faces)
                total_bytes += len(encode(mesh.vertices, mesh.faces, sweep))
            per_face.append(total_bytes / max(total_faces, 1))
        marker = "  <- PyChunkedGraph" if bits == settings["quantization_bits"] else ""
        print(
            f"{bits:>5} {bin_nm:>8.1f} {per_face[0]:>10.3f} {per_face[1]:>10.3f} "
            f"{per_face[2]:>10.3f} {per_face[2] / per_face[0]:>13.3f}{marker}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
