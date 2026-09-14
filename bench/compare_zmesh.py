"""Benchmark serra against zmesh, Frisken's reference C++, and VTK.

Each implementation is run in its own process, so peak RSS reflects only that
implementation and no two ever share an allocator. Invoke once per backend:

    python bench/compare_zmesh.py serra --threads 1 --close
    python bench/compare_zmesh.py zmesh --close --zmesh ../zmesh

Every backend prints one JSON line, so a wrapper can collect and tabulate them.
zmesh comes from a local checkout rather than PyPI, so the comparison runs
against the build you actually have; point --zmesh at it.
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
    close: bool = False,
) -> dict:
    import serra_mesh

    # Smoothing happens inside mesh(), so timing it means meshing twice: once
    # plain, once with fairing, and taking the difference. Reusing one mesher
    # would fold the smoothing into a traversal that has already run.
    smooth_s = None
    if fairing:
        warm = serra_mesh.Mesher(voxel_resolution=list(RESOLUTION), threads=threads)
        start = time.perf_counter()
        warm.mesh(volume, close=close)
        baseline = time.perf_counter() - start
        del warm

    mesher = serra_mesh.Mesher(
        voxel_resolution=list(RESOLUTION),
        threads=threads,
        fairing=fairing,
        fairing_taubin=fairing_taubin,
    )
    start = time.perf_counter()
    mesher.mesh(volume, close=close)
    march = time.perf_counter() - start
    after_march = peak_rss_gb()
    if fairing:
        smooth_s = march - baseline

    stats = {
        "objects": len(mesher),
        "march_s": march,
        "march_peak_gb": after_march,
        # Recorded rather than assumed. Whether the volume was padded is the
        # single most load-bearing setting in the comparison, and without this
        # nothing in the output distinguishes a padded run from an unpadded one.
        "close": close,
    }
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
    close: bool = False,
) -> dict:
    """zmesh's marching cubes, one object at a time.

    A different algorithm from the other three backends, so it is here as the
    marching-cubes baseline rather than as a cross-check: it places vertices on
    voxel edges instead of inside cells, and a different count is the correct
    outcome, not a disagreement to be explained away.

    `threads` is accepted and ignored -- zmesh has no threading of any kind --
    which is why it belongs against serra's one-thread column.

    `close` is not the same price here as in serra. serra's is a virtual border
    (src/grid.rs) and costs nothing. zmesh's wrapper allocates a padded C-order
    copy of the volume, 0.54 GB at 512 cubed uint32, and copies a
    Fortran-ordered input into it -- which also moves the traversal off zmesh's
    Fortran-order loop onto its C-order one. So its padded and unpadded
    traversal times are not two measurements of the same thing. It also never
    subtracts the pad again, so vertices land +1 voxel from every other
    backend's: counts are unaffected, coordinates are not. Nothing here reads a
    coordinate, so nothing here corrects it; bench/analytic_tube.py does, where
    coordinates matter.
    """
    import zmesh

    mesher = zmesh.Mesher(RESOLUTION)
    start = time.perf_counter()
    mesher.mesh(volume, close=close)
    march = time.perf_counter() - start
    after_march = peak_rss_gb()

    ids = mesher.ids()
    stats = {
        "objects": len(ids),
        "march_s": march,
        "march_peak_gb": after_march,
        "close": close,
        # zmesh is single-threaded whatever --threads says.
        "threads": 1,
        # Which build produced the numbers. A zmesh is usually also installed in
        # the environment, so a --zmesh that does not resolve falls back to it
        # silently -- which is the one thing the flag exists to prevent. Recorded
        # so the answer is in the output rather than in the shell history.
        "zmesh_path": zmesh.__file__,
    }
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


def split_by_label(points, triangles, boundary_labels, background=0):
    """Partition VTK's single output mesh into one indexed mesh per label.

    vtkSurfaceNets3D returns the whole volume as one polydata, with a
    two-component `boundary_labels` cell array naming the materials on either
    side of each face. serra returns one mesh per object, and a connectomics
    pipeline needs the latter -- one file per segment -- so this is real work,
    not a step invented to slow VTK down.

    Bucketed in a single pass rather than filtered once per label. The naive
    version rescans every face for every label, which is the O(labels x faces)
    trap, and measuring against that would flatter serra for no reason.
    """
    n = len(triangles)
    label = boundary_labels.ravel()
    cell = np.repeat(np.arange(n, dtype=np.int64), 2)
    keep = label != background
    label, cell = label[keep], cell[keep]

    order = np.argsort(label, kind="stable")
    label, cell = label[order], cell[order]
    ids, starts = np.unique(label, return_index=True)
    bounds = np.append(starts, len(label))

    meshes = {}
    for k, ident in enumerate(ids):
        tris = triangles[cell[bounds[k] : bounds[k + 1]]]
        # Renumbered to this object's own vertices, which is what makes the
        # result an indexed mesh rather than a slice of a global one.
        used, inverse = np.unique(tris, return_inverse=True)
        meshes[int(ident)] = (points[used], inverse.reshape(tris.shape))
    return meshes


def run_vtk(
    volume: np.ndarray,
    extract_all: bool,
    simplify: int = 0,
    max_error: float = 40.0,
    threads: int = 0,
    smoothing_iterations: int = 0,
    boundary_style: str = "all",
) -> dict:
    """VTK's vtkSurfaceNets3D, through pyvista's ImageData.contour_labels.

    Same Frisken lineage as serra, so this is the closest comparison available:
    one pass over the volume with all materials at once, rather than marching
    cubes per object.

    Two settings here are not pyvista's defaults, and both matter.
    `boundary_style="all"` emits the walls between adjacent objects, where
    pyvista defaults to "external" and returns only the outside of the
    foreground -- on real neuropil that is a third of the faces, so the default
    would have VTK doing a fraction of the work. And smoothing is off unless
    asked for, because pyvista turns it on with 16 iterations, which would put
    a smoothed VTK mesh against an unsmoothed serra one.
    """
    import pyvista as pv
    import vtk

    # VTK's SMP backend is fixed at build time. The PyPI wheels ship
    # Sequential, so this filter is single-threaded whatever it is asked for --
    # recorded next to the timing rather than left for the reader to assume.
    backend = vtk.vtkSMPTools.GetBackend()
    smp_threads = vtk.vtkSMPTools.GetEstimatedNumberOfThreads()

    # pyvista wants scalars in Fortran order, matching ImageData's indexing. On
    # an F-contiguous volume that is a view; on a C-contiguous one it is a full
    # copy, so it is timed rather than hidden.
    start = time.perf_counter()
    grid = pv.ImageData(dimensions=volume.shape, spacing=RESOLUTION)
    grid.point_data["labels"] = volume.ravel(order="F")
    convert = time.perf_counter() - start

    start = time.perf_counter()
    mesh = grid.contour_labels(
        boundary_style=boundary_style,
        # pyvista's default, pinned: serra's --close is what matches it, and the
        # doc asserts padding is on rather than relying on a default holding.
        pad_background=True,
        smoothing=smoothing_iterations > 0,
        smoothing_iterations=max(smoothing_iterations, 1),
        output_mesh_type="triangles",
    )
    march = time.perf_counter() - start
    after_march = peak_rss_gb()

    stats = {
        "implementation": "vtk",
        "vtk_version": vtk.VTK_VERSION,
        "pyvista_version": pv.__version__,
        "smp_backend": backend,
        "smp_threads": smp_threads,
        "boundary_style": boundary_style,
        "smoothing_iterations": smoothing_iterations,
        "convert_s": convert,
        "march_s": march,
        "march_peak_gb": after_march,
        # One mesh for the whole volume, each shared wall stored once. serra
        # stores it once per label, so these two are not comparable until the
        # split below has run.
        "shared_points": mesh.n_points,
        "shared_faces": mesh.n_cells,
    }

    if extract_all:
        points = np.asarray(mesh.points, dtype=np.float64)
        triangles = mesh.faces.reshape(-1, 4)[:, 1:]
        labels = mesh.cell_data["boundary_labels"]
        start = time.perf_counter()
        meshes = split_by_label(points, triangles, labels)
        split = time.perf_counter() - start
        stats.update(
            objects=len(meshes),
            get_all_s=split,
            vertices=sum(len(v) for v, _ in meshes.values()),
            faces=sum(len(f) for _, f in meshes.values()),
            output_gb=sum(v.nbytes + f.nbytes for v, f in meshes.values()) / 1e9,
            peak_gb=peak_rss_gb(),
        )
    else:
        stats["objects"] = len(np.unique(mesh.cell_data["boundary_labels"]))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["serra", "zmesh", "frisken", "vtk"])
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument(
        "--zmesh",
        default="/Users/forrestc/ConnectomeStack/zmesh",
        help="local zmesh checkout, so the comparison uses the build you have",
    )
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
        "--close",
        action="store_true",
        help=(
            "serra and zmesh; close objects at the volume boundary. Needed to "
            "match VTK, whose pad_background defaults on, and Frisken, which "
            "always pads -- with it serra's face count matches VTK's exactly, "
            "without it serra emits fewer. zmesh pays for it with a padded copy "
            "of the volume and a one-voxel offset in its output; see run_zmesh."
        ),
    )
    parser.add_argument(
        "--smoothing-iterations",
        type=int,
        default=0,
        help="vtk only; vtkSurfaceNets3D constrained-smoothing iterations",
    )
    parser.add_argument(
        "--boundary-style",
        default="all",
        choices=["all", "external", "internal"],
        help=(
            "vtk only. 'all' includes the walls between adjacent objects, which "
            "is what serra emits; pyvista's own default of 'external' returns "
            "only the outside of the foreground."
        ),
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

    # run_zmesh imports lazily, so the checkout has to be on the path before the
    # runner is called rather than at module import. Only for that backend: the
    # checkout root holds modules of its own (build/, templates/, test.py,
    # perf.py) and there is no reason to let them shadow anything in a serra,
    # VTK or Frisken run.
    if args.backend == "zmesh":
        sys.path.insert(0, args.zmesh)

    volume = load(args.volume)
    runners = {
        "serra": run_serra,
        "zmesh": run_zmesh,
        "frisken": run_frisken,
        "vtk": run_vtk,
    }
    if args.backend == "frisken":
        extra = {
            "relax": args.relax,
            "sample": args.sample,
            "binary": args.frisken_bin,
        }
    elif args.backend == "vtk":
        extra = {
            "smoothing_iterations": args.smoothing_iterations,
            "boundary_style": args.boundary_style,
        }
    elif args.backend == "serra":
        extra = {
            "fairing": args.fairing,
            "fairing_taubin": args.fairing_taubin,
            "close": args.close,
        }
    elif args.backend == "zmesh":
        extra = {"close": args.close}
    else:
        # `choices` makes this unreachable while all four backends are covered.
        # Kept so a fifth one fails by producing a bare result rather than a
        # NameError partway through a twenty-minute run.
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
