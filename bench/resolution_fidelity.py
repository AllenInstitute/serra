"""Does smoothing move a coarse mesh toward the fine-resolution truth?

Every accuracy number elsewhere in this repository treats the voxels being
meshed as ground truth. In a connectomics pipeline they usually are not. MICrONS
is segmented at 8x8x40 nm and meshed at 32x32x40 for speed and memory, so the
array serra sees is a 4x4x1 *downsample* of a segmentation that already knows
the boundary four times more precisely in x and y.

That changes what smoothing is. Measured against the coarse voxels, relaxation
"loses volume" and "loses area" -- it looks like damage. Measured against the
fine segmentation the coarse array was made from, the same displacement may be
*recovering* the boundary that downsampling threw away. This script decides
which, by fetching both resolutions of the same box and measuring the coarse
mesh against the fine one.

    uv sync --group bench
    python bench/resolution_fidelity.py

The fine mesh is itself a staircase, at 8 nm rather than 32, so it is a
reference and not truth. Its own quantisation is +/- 4 nm against the +/- 16 nm
of the coarse one, which is the margin the comparison rests on.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taubin import distance_to  # noqa: E402

CLOUDPATH = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
FINE = np.array([8.0, 8.0, 40.0])
COARSE = np.array([32.0, 32.0, 40.0])
FACTOR = np.rint(COARSE / FINE).astype(int)  # 4, 4, 1

DEFAULT_SEGMENT = 864691136144674612
DEFAULT_POINT_MIP0 = (87497, 59229, 21010)


def fetch(segment, point_mip0, micron, cache):
    """The same physical box at both resolutions, as boolean masks.

    Both arrays are cut on the *coarse* grid so that coarse voxel (i,j,k) covers
    exactly fine voxels (4i..4i+3, 4j..4j+3, k). That alignment is what makes
    the half-cell offset below a known constant rather than a fitted one.
    """
    from cloudvolume import CloudVolume

    centre_nm = np.asarray(point_mip0, float) * FINE
    half = micron * 1000.0 / 2.0
    origin_c = np.round((centre_nm - half) / COARSE).astype(int)
    size_c = np.round(2 * half / COARSE).astype(int)
    origin_f, size_f = origin_c * FACTOR, size_c * FACTOR

    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    paths = {"coarse": f"{cache}.coarse.npy.gz", "fine": f"{cache}.fine.npy.gz"}
    out = {}
    for name, (mip, origin, size) in {
        "coarse": (2, origin_c, size_c),
        "fine": (0, origin_f, size_f),
    }.items():
        if os.path.exists(paths[name]):
            with gzip.open(paths[name], "rb") as handle:
                out[name] = np.load(handle)
            continue
        cv = CloudVolume(
            CLOUDPATH, mip=mip, use_https=True, progress=True, fill_missing=True
        )
        print(f"  downloading {name} {tuple(size)} at mip {mip} ...")
        block = np.asarray(
            cv[
                origin[0] : origin[0] + size[0],
                origin[1] : origin[1] + size[1],
                origin[2] : origin[2] + size[2],
            ]
        ).squeeze()
        mask = (block == segment).astype(np.uint32)
        del block
        with gzip.open(paths[name], "wb", compresslevel=6) as handle:
            np.save(handle, mask)
        out[name] = mask
    return out["coarse"], out["fine"], origin_c


def meshed(serra_mesh, mask, resolution, **kwargs):
    mesher = serra_mesh.Mesher(voxel_resolution=list(resolution), **kwargs)
    mesher.mesh(mask, close=True)
    got = mesher.get(1)
    return np.asarray(got.vertices, np.float64), got.faces


def offset_between_grids():
    """Physical shift between the two grids' sample positions.

    serra places sample index i at i*resolution, so a coarse sample sits at
    32i nm while the block of fine samples it was built from -- 4i..4i+3 --
    is centred at (4i + 1.5)*8 = 32i + 12 nm. Without correcting that, the
    comparison would report a spurious 12 nm bias in x and y.
    """
    return (FACTOR - 1) / 2.0 * FINE


def surface_stats(vertices, faces, reference):
    d = distance_to(reference, vertices)
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
    }


def occupancy_agreement(vertices, faces, fine, shift, seed, samples):
    """Does the coarse mesh enclose the right set of FINE voxels?

    Distance to a reference surface is only as good as the reference, and the
    fine mesh is itself a staircase at 8 nm — a smoother coarse surface can be
    closer to the true boundary while measuring further from that staircase.
    This metric avoids the problem by never building a reference surface: sample
    points, ask the fine segmentation whether each is inside, ask the coarse
    mesh the same with a winding number, and count disagreements.
    """
    import igl

    # A fresh generator per call, so every variant is scored on identical
    # points. Differences here are fractions of a percent; with independent
    # draws they would be indistinguishable from noise.
    rng = np.random.default_rng(seed)
    occupied = np.argwhere(fine)
    lo = np.maximum(occupied.min(0) - 2, 0)
    hi = np.minimum(occupied.max(0) + 3, np.asarray(fine.shape))
    idx = rng.integers(lo, hi, size=(samples, 3))
    truth = fine[idx[:, 0], idx[:, 1], idx[:, 2]].astype(bool)

    # Fine voxel index -> physical nm, in the same frame the coarse mesh uses
    # after `shift` (both grids share an origin; see offset_between_grids).
    points = idx.astype(np.float64) * FINE
    w = igl.fast_winding_number(
        np.ascontiguousarray(vertices, np.float64),
        np.ascontiguousarray(faces, np.int64),
        np.ascontiguousarray(points, np.float64),
    )
    inside = w > 0.5
    intersection = float((inside & truth).sum())
    union = float((inside | truth).sum())
    return {
        "iou": intersection / max(union, 1.0),
        "wrong": float((inside != truth).mean()),
        "false_in": float((inside & ~truth).sum()) / max(float(truth.sum()), 1.0),
        "false_out": float((~inside & truth).sum()) / max(float(truth.sum()), 1.0),
    }


def enclosed_volume(vertices, faces):
    v = np.asarray(vertices, np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=int, default=DEFAULT_SEGMENT)
    parser.add_argument("--point", type=int, nargs=3, default=DEFAULT_POINT_MIP0)
    parser.add_argument("--micron", type=float, default=5.0)
    parser.add_argument("--cache", default="data/microns_two_resolution")
    parser.add_argument(
        "--iterations", type=int, nargs="*", default=[0, 1, 2, 3, 5, 10]
    )
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import serra_mesh

    coarse, fine, origin = fetch(args.segment, args.point, args.micron, args.cache)
    print(f"coarse {coarse.shape} at 32x32x40, {coarse.sum():,} voxels")
    print(f"fine   {fine.shape} at  8x8x40,  {fine.sum():,} voxels")

    fine_volume = float(fine.sum()) * float(np.prod(FINE))
    coarse_volume = float(coarse.sum()) * float(np.prod(COARSE))
    print(f"\nvolume from fine voxels   : {fine_volume / 1e9:.4f} um^3")
    print(
        f"volume from coarse voxels : {coarse_volume / 1e9:.4f} um^3 "
        f"({coarse_volume / fine_volume - 1:+.2%} vs fine)"
    )

    # How faithful is the served downsample to a plain majority vote? If these
    # disagree wildly the comparison is measuring the downsampler, not serra.
    trimmed = fine[
        : coarse.shape[0] * FACTOR[0],
        : coarse.shape[1] * FACTOR[1],
        : coarse.shape[2] * FACTOR[2],
    ]
    blocks = trimmed.reshape(
        coarse.shape[0],
        FACTOR[0],
        coarse.shape[1],
        FACTOR[1],
        coarse.shape[2],
        FACTOR[2],
    )
    majority = (blocks.sum(axis=(1, 3, 5)) * 2 > np.prod(FACTOR)).astype(np.uint32)
    agree = float((majority == coarse).mean())
    print(f"served coarse vs majority-downsampled fine: {agree:.3%} of voxels agree")

    shift = offset_between_grids()
    print(f"grid offset applied to the coarse mesh: {shift} nm\n")

    raw_reference = meshed(serra_mesh, fine, FINE)
    print(
        f"fine reference mesh: {len(raw_reference[0]):,} verts "
        f"{len(raw_reference[1]):,} faces"
    )
    # The fine mesh has its own 8 nm staircase. Lightly smoothing it is a better
    # estimate of where the boundary really is, and measuring against both says
    # how much of the result is an artefact of the reference.
    smooth_reference = meshed(serra_mesh, fine, FINE, relaxation=2)

    variants = [("no smoothing", {})]
    for k in args.iterations:
        if k:
            variants.append((f"relaxation={k}", dict(relaxation=k)))
    for k in (3, 10, 20):
        variants.append((f"taubin={k}", dict(taubin=k)))

    rows = []
    for name, kwargs in variants:
        try:
            v, f = meshed(serra_mesh, coarse, COARSE, **kwargs)
        except (TypeError, ValueError):
            continue  # taubin exists only on the branch that adds it
        v = v + shift
        rows.append(
            (
                name,
                f,
                surface_stats(v, f, raw_reference),
                surface_stats(v, f, smooth_reference),
                enclosed_volume(v, f) / fine_volume - 1,
                occupancy_agreement(v, f, fine, shift, args.seed, args.samples),
            )
        )

    print(f"\n{'coarse mesh':>26} {'to raw fine':>22} {'to smoothed fine':>22}")
    print(
        f"{'':>26} {'mean':>8} {'median':>7} {'p95':>6} {'mean':>8} {'median':>7} {'p95':>6}"
    )
    for name, _, raw, sm, _, _ in rows:
        print(
            f"{name:>26} {raw['mean']:>7.1f} {raw['median']:>6.1f} {raw['p95']:>5.1f} "
            f"{sm['mean']:>7.1f} {sm['median']:>6.1f} {sm['p95']:>5.1f}"
        )
    print("   (distance from the coarse mesh to the fine surface, nm)")

    # Coarse topology, fine geometry. Every vertex is projected onto the fine
    # surface and the displacement clamped to its own cell -- the invariant
    # serra's placement already guarantees. Connectivity, face count and mesh
    # memory stay exactly coarse; only the positions carry fine information.
    import igl as _igl

    base_v, base_f = meshed(serra_mesh, coarse, COARSE)
    base_v = base_v + shift
    _, _, closest = _igl.point_mesh_squared_distance(
        np.ascontiguousarray(base_v),
        np.ascontiguousarray(raw_reference[0]),
        np.ascontiguousarray(raw_reference[1], np.int64),
    )
    limit = COARSE / 2.0
    snapped = base_v + np.clip(closest - base_v, -limit, limit)
    rows.append(
        (
            "coarse topology, fine placement",
            base_f,
            surface_stats(snapped, base_f, raw_reference),
            surface_stats(snapped, base_f, smooth_reference),
            enclosed_volume(snapped, base_f) / fine_volume - 1,
            occupancy_agreement(snapped, base_f, fine, shift, args.seed, args.samples),
        )
    )

    # Two reference points that bound the question. The fine mesh is the
    # ceiling: it is what you get by paying the memory. The fine mesh decimated
    # to the coarse face count says how much of that ceiling is about the
    # *sampling* rather than the triangle budget -- if it stays near the
    # ceiling, then meshing coarse is what costs you, not the face count, and
    # no smoother can win it back.
    fine_full = raw_reference[0], raw_reference[1]
    target = len(rows[0][1])
    fine_mesher = serra_mesh.Mesher(voxel_resolution=list(FINE))
    fine_mesher.mesh(fine, close=True)
    factor = max(2, int(round(len(fine_full[1]) / target)))
    small = fine_mesher.get(1, reduction_factor=factor, max_error=1e9)
    for name, (v, f) in [
        ("FINE mesh (the ceiling)", fine_full),
        (
            f"FINE decimated {factor}x",
            (np.asarray(small.vertices, np.float64), small.faces),
        ),
    ]:
        rows.append(
            (
                name,
                f,
                None,
                None,
                enclosed_volume(v, f) / fine_volume - 1,
                occupancy_agreement(v, f, fine, shift, args.seed, args.samples),
            )
        )

    print(
        f"\n{'mesh':>26} {'faces':>9} {'volume vs fine':>15} {'IoU':>9} {'wrong':>8} "
        f"{'too big':>9} {'too small':>10}"
    )
    for name, f, _, _, vol, occ in rows:
        print(
            f"{name:>26} {len(f):>9,} {vol:>14.2%} {occ['iou']:>8.3%} "
            f"{occ['wrong']:>7.3%} {occ['false_in']:>8.2%} {occ['false_out']:>9.2%}"
        )
    print(f"   (agreement with the FINE voxels themselves, {args.samples:,} samples;")
    print("    'too big'/'too small' are relative to the fine object's own volume)")

    # The fine mesh's own quantisation, for scale: how far is a fine-resolution
    # mesh from the fine voxels it came from? Nothing coarse can beat this.
    print(
        f"\n   for scale, one coarse voxel is {COARSE[0]:.0f} nm across and the fine\n"
        f"   grid quantises to +/-{FINE[0] / 2:.0f} nm."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
