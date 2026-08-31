"""Fetch a MICrONS segmentation cutout and store it as a test fixture.

    uv sync --group bench
    python bench/download_microns.py

The default region was chosen deliberately. serra is meant for volumes holding
very many small objects, so a cutout dominated by one cell body would exercise
the wrong thing. Candidate regions near the centre of the imaged column were
scored at low resolution for object count, for how much of the volume the
largest object occupies, and for how little background they contain; the
default is the one where no single object exceeds ~4%.

Labels are renumbered to a compact dtype, which changes nothing about the
geometry and makes the artifact small enough to commit. The mapping back to the
original segment ids is written alongside, so any object can still be traced to
the live dataset.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np

CLOUDPATH = "gs://iarpa_microns/minnie/minnie65/seg_m1300"
MIP = 2  # 32 x 32 x 40 nm
SIZE = 512

# Dense neuropil near the centre of the imaged column. See the module docstring
# for how this was picked; `--survey` reproduces the search.
DEFAULT_ORIGIN = (31872, 23728, 20396)


def survey(cv_factory, centre, probe=96):
    """Score candidate regions so the choice of fixture is reproducible."""
    cv = cv_factory(MIP)
    print(f"{'mip2 corner':>24} {'objects':>8} {'largest':>8} {'top5':>7} {'bg':>6}")
    rows = []
    for dx in (-2000, -700, 700, 2000):
        for dy in (-1400, 0, 1400):
            for dz in (-900, 0, 900):
                o = np.asarray(centre) + np.array([dx, dy, dz])
                a = np.asarray(
                    cv[o[0] : o[0] + probe, o[1] : o[1] + probe, o[2] : o[2] + probe]
                ).squeeze()
                ids, counts = np.unique(a, return_counts=True)
                background = (
                    counts[ids == 0].sum() / a.size if (ids == 0).any() else 0.0
                )
                sizes = np.sort(counts[ids != 0])[::-1]
                if sizes.size == 0:
                    continue
                largest = sizes[0] / a.size
                rows.append((largest, tuple(int(v) for v in o)))
                print(
                    f"{str(tuple(int(v) for v in o)):>24} {len(sizes):>8} "
                    f"{largest:>7.2%} {sizes[:5].sum() / a.size:>6.2%} {background:>5.1%}"
                )
    rows.sort()
    print(f"\nsmallest dominant object at: {rows[0][1]} ({rows[0][0]:.2%})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", type=int, nargs=3, default=DEFAULT_ORIGIN)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--out", default="data/microns_neuropil")
    parser.add_argument(
        "--survey", action="store_true", help="score candidate regions and exit"
    )
    args = parser.parse_args()

    import fastremap
    from cloudvolume import CloudVolume

    def cv_factory(mip):
        return CloudVolume(
            CLOUDPATH, mip=mip, use_https=True, progress=False, fill_missing=True
        )

    if args.survey:
        cv = cv_factory(10)
        occupied = np.argwhere(np.asarray(cv[:, :, :]).squeeze() != 0)
        lo, hi = occupied.min(0) * 256, (occupied.max(0) + 1) * 256
        print(f"imaged extent at mip2: {lo} .. {hi}")
        survey(cv_factory, (lo + hi) // 2)
        return

    o, n = np.asarray(args.origin), args.size
    cv = cv_factory(MIP)
    print(f"downloading {n}^3 at mip {MIP} from {o} ...")
    start = time.perf_counter()
    raw = np.asarray(cv[o[0] : o[0] + n, o[1] : o[1] + n, o[2] : o[2] + n]).squeeze()
    print(f"  {raw.nbytes / 1e9:.2f} GB in {time.perf_counter() - start:.0f}s")

    # Renumber so the fixture is small. `preserve_zero` keeps background as 0,
    # which every mesher treats specially.
    compact, mapping = fastremap.renumber(raw, in_place=False, preserve_zero=True)
    compact = fastremap.refit(compact)
    del raw

    segids = np.zeros(max(mapping.values()) + 1, dtype=np.uint64)
    for original, new in mapping.items():
        segids[new] = original

    ids, counts = np.unique(compact, return_counts=True)
    sizes = np.sort(counts[ids != 0])[::-1]
    stats = {
        "cloudpath": CLOUDPATH,
        "mip": MIP,
        "resolution_nm": [32, 32, 40],
        "origin_voxels": [int(v) for v in o],
        "shape": [int(v) for v in compact.shape],
        "dtype": str(compact.dtype),
        "objects": int(sizes.size),
        "background_fraction": float(
            counts[ids == 0].sum() / compact.size if (ids == 0).any() else 0.0
        ),
        "largest_object_fraction": float(sizes[0] / compact.size),
        "top10_fraction": float(sizes[:10].sum() / compact.size),
        "median_object_voxels": int(np.median(sizes)),
        "segid_example": {
            "fixture_label": 1,
            "live_segment_id": None,  # filled in below
        },
    }

    stats["segid_example"]["live_segment_id"] = (
        int(segids[1]) if len(segids) > 1 else None
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(f"{args.out}.npy.gz", "wb", compresslevel=6) as handle:
        np.save(handle, compact)
    # Indexed by the label in the fixture, yielding the live dataset's segment
    # id — the direction anyone tracing an object actually needs. (renumber
    # returns the opposite mapping.)
    np.save(f"{args.out}.segids.npy", segids)
    with open(f"{args.out}.json", "w") as handle:
        json.dump(stats, handle, indent=2)

    print(json.dumps(stats, indent=2))
    size_mb = os.path.getsize(f"{args.out}.npy.gz") / 1e6
    print(f"\nwrote {args.out}.npy.gz ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
