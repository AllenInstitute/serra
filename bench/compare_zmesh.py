"""Benchmark serra against zmesh on a real connectomics volume.

Each implementation is run in its own process, so peak RSS reflects only that
implementation and the two never share an allocator. Invoke once per backend:

    python bench/compare_zmesh.py serra --volume path/to/labels.npy.gz
    python bench/compare_zmesh.py zmesh --volume path/to/labels.npy.gz

Both print one JSON line, so a wrapper can collect and tabulate them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import time

import numpy as np

DEFAULT_VOLUME = "/Users/forrestc/ConnectomeStack/zmesh/connectomics.npy.gz"
RESOLUTION = (32, 32, 40)


def peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    scale = 1e9 if usage > 1e9 else 1e6
    return usage / scale


def load(path: str) -> np.ndarray:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return np.load(handle)
    return np.load(path)


def run_serra(
    volume: np.ndarray,
    extract_all: bool,
    simplify: int = 0,
    max_error: float = 40.0,
    threads: int = 0,
) -> dict:
    import serra_mesh

    mesher = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION), threads=threads)
    start = time.perf_counter()
    mesher.mesh(volume)
    march = time.perf_counter() - start
    after_march = peak_rss_gb()

    stats = {"objects": len(mesher), "march_s": march, "march_peak_gb": after_march}
    if extract_all:
        start = time.perf_counter()
        vertices = faces = nbytes = 0
        for label in mesher.ids():
            mesh = mesher.get(
                int(label), reduction_factor=simplify, max_error=max_error
            )
            vertices += len(mesh.vertices)
            faces += len(mesh.faces)
            nbytes += mesh.nbytes
        stats.update(
            get_all_s=time.perf_counter() - start,
            vertices=vertices,
            faces=faces,
            output_gb=nbytes / 1e9,
            peak_gb=peak_rss_gb(),
        )
    return stats


def run_zmesh(
    volume: np.ndarray,
    extract_all: bool,
    simplify: int = 0,
    max_error: float = 40.0,
    threads: int = 0,
) -> dict:
    from zmesh import Mesher

    mesher = Mesher(RESOLUTION)
    start = time.perf_counter()
    mesher.mesh(volume)
    march = time.perf_counter() - start
    after_march = peak_rss_gb()

    ids = mesher.ids()
    stats = {"objects": len(ids), "march_s": march, "march_peak_gb": after_march}
    if extract_all:
        start = time.perf_counter()
        vertices = faces = nbytes = 0
        for label in ids:
            mesh = (
                mesher.get(label, reduction_factor=simplify, max_error=max_error)
                if simplify > 1
                else mesher.get(label)
            )
            vertices += len(mesh.vertices)
            faces += len(mesh.faces)
            nbytes += mesh.nbytes
        stats.update(
            get_all_s=time.perf_counter() - start,
            vertices=vertices,
            faces=faces,
            output_gb=nbytes / 1e9,
            peak_gb=peak_rss_gb(),
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["serra", "zmesh"])
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument(
        "--simplify",
        type=int,
        default=0,
        help="reduction factor to pass to get(); 0 disables simplification",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=40.0,
        help=(
            "error bound in nm. Note the two libraries mean different things by "
            "it: serra enforces a displacement bound, zmesh thresholds the "
            "quadric, so serra reduces less at the same number."
        ),
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="serra only; 0 uses every core"
    )
    parser.add_argument(
        "--march-only",
        action="store_true",
        help="skip per-object extraction, timing only the volume traversal",
    )
    args = parser.parse_args()

    volume = load(args.volume)
    runner = run_serra if args.backend == "serra" else run_zmesh
    stats = runner(
        volume,
        extract_all=not args.march_only,
        simplify=args.simplify,
        max_error=args.max_error,
        threads=args.threads,
    )
    stats["simplify"] = args.simplify

    stats["backend"] = args.backend
    stats["shape"] = list(volume.shape)
    stats["dtype"] = str(volume.dtype)
    stats["order"] = "F" if volume.flags.f_contiguous else "C"
    stats["mvx_per_s"] = (volume.size / 1e6) / stats["march_s"]
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
