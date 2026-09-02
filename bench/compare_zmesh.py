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
import os
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

DEFAULT_VOLUME = "/Users/forrestc/ConnectomeStack/zmesh/connectomics.npy.gz"
RESOLUTION = (32, 32, 40)


def peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kibibytes. Decide from the platform, not from
    # the magnitude: guessing by value reports a 900 MB peak as 900 GB, and
    # every sub-gigabyte volume lands in exactly that range.
    scale = 1e9 if sys.platform == "darwin" else 1e9 / 1024
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
    fairing: int = 0,
    fairing_taubin: bool = False,
) -> dict:
    import serra_mesh

    # Smoothing happens inside mesh(), so timing it means meshing twice: once
    # plain, once with fairing, and taking the difference. Reusing one mesher
    # would fold the smoothing into a traversal that has already run.
    smooth_s = None
    if fairing:
        warm = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION), threads=threads)
        start = time.perf_counter()
        warm.mesh(volume)
        baseline = time.perf_counter() - start
        del warm

    mesher = serra_mesh.Mesher(
        voxel_resolution=list(RESOLUTION),
        threads=threads,
        fairing=fairing,
        fairing_taubin=fairing_taubin,
    )
    start = time.perf_counter()
    mesher.mesh(volume)
    march = time.perf_counter() - start
    after_march = peak_rss_gb()
    if fairing:
        smooth_s = march - baseline

    stats = {"objects": len(mesher), "march_s": march, "march_peak_gb": after_march}
    if fairing:
        stats.update(
            fairing=fairing,
            fairing_taubin=fairing_taubin,
            march_without_smoothing_s=baseline,
            smooth_s=smooth_s,
        )
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


FRISKEN_BIN = "bench/frisken/build/stock/sn_bench"


def run_frisken(
    volume: np.ndarray,
    extract_all: bool,
    simplify: int = 0,
    max_error: float = 40.0,
    threads: int = 0,
    binary: str = FRISKEN_BIN,
    relax: int = 0,
    sample: int = -1,
) -> dict:
    """Frisken's reference C++ implementation, via bench/frisken/sn_bench.

    Run out of process rather than bound into this one. That is not only for a
    clean peak RSS: the reference allocates a dense cell array over the padded
    volume, which is 15 GB at 512 cubed, and an allocator that has once held
    that is not a fair place to measure anything else.
    """
    if not os.path.exists(binary):
        raise SystemExit(f"{binary} not found -- run `make -C bench/frisken stock`")

    # MMSurfaceNet takes unsigned short, materials 0-65534, with 65535 reserved
    # for the padding label. Segment IDs are 64-bit and arbitrary, so densify
    # them; the mapping is order-preserving but the IDs themselves are not
    # carried through, which is fine because nothing downstream needs them.
    ids, dense = np.unique(volume, return_inverse=True)
    if len(ids) > 65535:
        raise SystemExit(
            f"{len(ids)} distinct labels exceeds the 65535 the reference supports"
        )
    dense = dense.astype(np.uint16).reshape(volume.shape)

    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "volume.u16")
        dense.tofile(path)
        command = [
            os.path.abspath(binary),
            "--volume",
            path,
            "--dims",
            ",".join(str(n) for n in volume.shape),
            "--voxel",
            ",".join(str(float(r)) for r in RESOLUTION),
            "--relax",
            str(relax),
            "--sample",
            str(sample),
        ]
        if not extract_all:
            command.append("--no-extract")
        done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"sn_bench failed:\n{done.stderr}")
    stats = json.loads(done.stdout)
    # The child measured its own peak RSS; this process only ever held the
    # volume, so reporting our own would understate it by an order of magnitude.
    stats["measured_in_subprocess"] = True
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["serra", "zmesh", "frisken"])
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
    parser.add_argument(
        "--fairing",
        type=int,
        default=0,
        help="serra only; cell-domain fairing passes",
    )
    parser.add_argument(
        "--fairing-taubin",
        action="store_true",
        help="serra only; use Taubin steps for fairing",
    )
    parser.add_argument(
        "--relax",
        type=int,
        default=0,
        help=(
            "frisken only; relaxation sweeps. Not comparable one-for-one with "
            "serra's fairing: the reference is Gauss-Seidel, serra is Jacobi, "
            "so a reference sweep does roughly twice the work of a serra pass."
        ),
    )
    parser.add_argument(
        "--frisken-bin",
        default=FRISKEN_BIN,
        help="frisken only; use build/patched/sn_bench for the patched variant",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=-1,
        help=(
            "frisken only; extract this many labels instead of all. objData() "
            "rescans every quad once per label, so 'all' is minutes."
        ),
    )
    args = parser.parse_args()

    volume = load(args.volume)
    runners = {"serra": run_serra, "zmesh": run_zmesh, "frisken": run_frisken}
    if args.backend == "frisken":
        extra = {
            "relax": args.relax,
            "sample": args.sample,
            "binary": args.frisken_bin,
        }
    elif args.backend == "serra":
        extra = {"fairing": args.fairing, "fairing_taubin": args.fairing_taubin}
    else:
        extra = {}
    stats = runners[args.backend](
        volume,
        extract_all=not args.march_only,
        simplify=args.simplify,
        max_error=args.max_error,
        threads=args.threads,
        **extra,
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
