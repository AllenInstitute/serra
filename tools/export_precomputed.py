"""Write a cutout and its meshes as a local Neuroglancer precomputed datastore.

Produces one layer per mesher so they can be loaded side by side and compared in
the browser. The voxels are identical in each; only the meshes differ.

    uv sync --group bench
    python tools/export_precomputed.py --size 256
    python tools/serve_precomputed.py

The mesh format is the legacy single-resolution one, which for a single chunk is
three lines: for each segment a JSON manifest at `mesh/<id>:0` naming its
fragments, and a fragment at `mesh/<id>:0:1` holding a little-endian uint32
vertex count, then the vertices as float32 triples **in nanometres**, then the
triangles as uint32 triples. No sharding, no Draco, no level-of-detail tree —
none of which a single chunk needs.

Everything is written uncompressed on purpose. CloudFiles would happily gzip the
chunks, but then a plain static file server hands the browser gzip bytes without
a `Content-Encoding` header and Neuroglancer sees garbage. Uncompressed costs
disk and buys not having to explain that.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

RESOLUTION = (32, 32, 40)  # nanometres per voxel, matching the MICrONS mip-2 cutout


def load(path: str) -> np.ndarray:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return np.load(handle)
    return np.load(path)


def write_volume(labels: np.ndarray, path: str, resolution, offset) -> None:
    """The segmentation layer, with a `mesh` key pointing at the mesh directory."""
    from cloudvolume import CloudVolume

    cv = CloudVolume.from_numpy(
        labels,
        vol_path=f"file://{os.path.abspath(path)}",
        resolution=resolution,
        voxel_offset=tuple(int(v) for v in offset),
        layer_type="segmentation",
        # raw + uncompressed: see the module docstring. compressed_segmentation
        # would be smaller but adds a second thing to get wrong.
        encoding="raw",
        compress=False,
        progress=False,
    )
    cv.info["mesh"] = "mesh"
    cv.commit_info()


def encode(vertices: np.ndarray, faces: np.ndarray, shift_nm) -> bytes:
    """One precomputed mesh fragment, shifted into the global frame.

    A mesher only knows the cutout, so its vertices are nanometres from the
    cutout's own corner. Neuroglancer places a layer's voxel `i` at `i *
    resolution` including the layer's `voxel_offset`, and expects mesh vertices
    in that same global frame — so the cutout's origin has to be added back on
    or the meshes sit at the corner of the dataset while the voxels sit in the
    right place.
    """
    v = np.ascontiguousarray(np.asarray(vertices, np.float64) + shift_nm, dtype="<f4")
    f = np.ascontiguousarray(faces, dtype="<u4")
    return struct.pack("<I", len(v)) + v.tobytes() + f.tobytes()


def write_meshes(meshes, path: str, shift_nm) -> int:
    """One manifest and one fragment per segment. Returns bytes written."""
    directory = os.path.join(path, "mesh")
    os.makedirs(directory, exist_ok=True)
    total = 0
    for label, mesh in meshes:
        fragment = f"{label}:0:1"
        blob = encode(mesh.vertices, mesh.faces, shift_nm)
        with open(os.path.join(directory, fragment), "wb") as handle:
            handle.write(blob)
        with open(os.path.join(directory, f"{label}:0"), "w") as handle:
            json.dump({"fragments": [fragment]}, handle)
        total += len(blob)
    return total


def mesh_with_zmesh(labels, resolution, args):
    """Marching cubes at the simplification igneous and PyChunkedGraph use."""
    sys.path.insert(0, args.zmesh)
    import zmesh

    mesher = zmesh.Mesher(resolution)
    mesher.mesh(labels, close=False)
    for label in mesher.ids():
        got = mesher.get(
            label,
            reduction_factor=args.reduction_factor,
            max_error=args.max_error,
        )
        if len(got.faces):
            yield int(label), got


def mesh_with_serra(labels, resolution, args, **kwargs):
    import serra_mesh

    mesher = serra_mesh.Mesher(voxel_resolution=list(resolution), **kwargs)
    mesher.mesh(labels, close=False)
    for label in mesher.ids():
        got = mesher.get(int(label))
        if len(got.faces):
            yield int(label), got


VARIANTS = {
    "zmesh": (
        "marching cubes, simplified 100x at 40 nm as igneous does",
        mesh_with_zmesh,
        {},
    ),
    "serra_fairing": (
        "serra, cell-domain fairing with Taubin steps, no simplification",
        lambda labels, res, args: mesh_with_serra(
            labels, res, args, fairing=20, fairing_taubin=True
        ),
        {},
    ),
    "serra_raw": (
        "serra, no smoothing and no simplification",
        lambda labels, res, args: mesh_with_serra(labels, res, args),
        {},
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="data/microns_neuropil.npy.gz")
    parser.add_argument("--size", type=int, default=256, help="cube edge, in voxels")
    parser.add_argument("--origin", type=int, nargs=3, default=[0, 0, 0])
    parser.add_argument("--out", default="precomputed")
    parser.add_argument(
        "--variant",
        nargs="*",
        default=["zmesh", "serra_fairing"],
        choices=sorted(VARIANTS),
    )
    parser.add_argument("--min-voxels", type=int, default=0)
    parser.add_argument("--reduction-factor", type=int, default=100)
    parser.add_argument("--max-error", type=float, default=40.0)
    parser.add_argument("--zmesh", default="/Users/forrestc/ConnectomeStack/zmesh")
    parser.add_argument("--resolution", type=int, nargs=3, default=list(RESOLUTION))
    parser.add_argument(
        "--global-origin",
        type=int,
        nargs=3,
        default=None,
        help="cutout corner in dataset voxels; read from the fixture sidecar if omitted",
    )
    parser.add_argument(
        "--local-frame",
        action="store_true",
        help="place the cutout at the origin instead, so it stands alone",
    )
    parser.add_argument(
        "--real-ids",
        action="store_true",
        help="restore live segment ids from the fixture's .segids.npy",
    )
    args = parser.parse_args()

    volume = load(args.volume)

    # Where this cutout sits in the dataset, so the imagery lines up with it.
    origin = args.global_origin
    if origin is None and not args.local_frame:
        sidecar = args.volume.replace(".npy.gz", ".json").replace(".npy", ".json")
        if os.path.exists(sidecar):
            with open(sidecar) as handle:
                origin = json.load(handle).get("origin_voxels")
    if origin is None:
        origin = [0, 0, 0]

    o, n = args.origin, args.size
    labels = np.ascontiguousarray(
        volume[o[0] : o[0] + n, o[1] : o[1] + n, o[2] : o[2] + n]
    ).astype(np.uint32)

    if args.min_voxels > 0:
        ids, counts = np.unique(labels, return_counts=True)
        drop = ids[(counts < args.min_voxels) & (ids != 0)]
        if len(drop):
            labels[np.isin(labels, drop)] = 0

    if args.real_ids:
        # The fixture renumbers labels to keep the artifact small; this puts the
        # live ids back so a segment can be looked up in the real dataset.
        table = args.volume.replace(".npy.gz", ".segids.npy").replace(
            ".npy", ".segids.npy"
        )
        if not os.path.exists(table):
            raise SystemExit(f"--real-ids needs {table}")
        segids = np.load(table)
        labels = segids[labels].astype(np.uint64)

    corner = [int(origin[k]) + int(o[k]) for k in range(3)]
    shift_nm = np.asarray(corner, np.float64) * np.asarray(args.resolution, np.float64)

    present = int(len(np.unique(labels)) - 1)
    print(f"cutout {labels.shape} at {tuple(args.resolution)} nm, {present:,} objects")
    print(f"placed at dataset voxel {tuple(corner)} = {tuple(shift_nm.astype(int))} nm")

    for name in args.variant:
        description, mesher, _ = VARIANTS[name]
        path = os.path.join(args.out, name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

        print(f"\n{name}: {description}")
        start = time.perf_counter()
        write_volume(labels, path, args.resolution, corner)
        volume_done = time.perf_counter()

        produced = list(mesher(labels, tuple(args.resolution), args))
        meshed = time.perf_counter()
        written = write_meshes(produced, path, shift_nm)
        done = time.perf_counter()

        faces = sum(len(m.faces) for _, m in produced)
        print(
            f"  {len(produced):,} meshes, {faces:,} faces, {written / 1e6:.1f} MB\n"
            f"  volume {volume_done - start:.1f}s, mesh {meshed - volume_done:.1f}s, "
            f"write {done - meshed:.1f}s"
        )

    print(f"\nwrote {args.out}/. Serve it with:\n    python tools/serve_precomputed.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
