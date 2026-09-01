"""Smoothing in the cell domain, with GAMer's local structure tensor.

Two ideas measured together, both prompted by Yu et al. (2008) and GAMer 2
(Lee et al., PLoS Comput Biol 2020).

**Cell-domain smoothing.** serra places one vertex per cell as a pure function of
that cell's eight corner labels, so every label present in the cell gets the
*same* position. It then copies that position into each label's own mesh and
smooths each mesh independently — so the two copies of a wall shared by two
touching objects drift apart, and the segmentation stops being a partition of
space. Smoothing one shared position per cell instead fixes that by
construction, and costs less: on dense neuropil there are 2.06 label-copies per
boundary cell.

**The local structure tensor.** T = sum of n n^T over the incident face normals.
Its eigenvalues say what the surface is doing locally — three similar values is a
sphere or a saddle, two large and one small a ridge, one large and two small a
plane. Damping the smoothing step by 1/(1 + scale*lambda_k) along each
eigenvector suppresses motion across high-curvature directions while leaving it
free along flat ones. That replaces serra's isotropic `max_deviation` box, which
cannot tell the two apart. `n n^T` is sign-invariant, so the winding flip between
the two sides of a shared wall does not matter and one tensor serves every label.

This is a prototype: everything runs in numpy against serra's existing output,
with no Rust changes. It exists to decide whether the Rust work is worth doing.

    uv sync --group bench
    python bench/cell_smoothing.py                 # every experiment
    python bench/cell_smoothing.py --analytic-only # just the shape fixtures
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taubin import area, roughness_degrees, volume  # noqa: E402

SUBVOXEL = 256  # serra's fixed-point scale; positions are exact multiples of 1/256


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def sphere_mask(radius, pad=4):
    n = 2 * (radius + pad) + 1
    g = np.arange(n) - (n - 1) / 2
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return ((x * x + y * y + z * z) <= radius * radius).astype(np.uint32)


def box_mask(side, pad=4):
    n = side + 2 * pad
    a = np.zeros((n, n, n), np.uint32)
    a[pad : pad + side, pad : pad + side, pad : pad + side] = 1
    return a


def cylinder_mask(radius, height, pad=4):
    n = int(2 * (radius + pad)) + 1
    m = int(height) + 2 * pad
    a = np.zeros((m, n, n), np.uint32)
    c = (n - 1) / 2
    y, z = np.ogrid[:n, :n]
    disc = ((y - c) ** 2 + (z - c) ** 2) <= radius * radius
    a[pad : pad + int(height)] = disc.astype(np.uint32)
    return a


# --------------------------------------------------------------------------
# the cell domain
# --------------------------------------------------------------------------


class CellGraph:
    """Every label's surface, welded into one graph keyed by cell.

    Built from meshes produced at unit voxel resolution, so a vertex position is
    an exact multiple of 1/256 of a voxel and the cell key is exact integer
    arithmetic rather than a tolerance.
    """

    def __init__(self, meshes, shape, pin_border=True, share=True):
        """`meshes` is a list of (label, vertices, faces) at unit resolution.

        With `share=False` no vertex is welded to any other, so every label keeps
        its own private copy of a shared wall. That is what serra does today, and
        running the identical operator over it is the only controlled way to
        attribute a difference to the domain rather than to the smoother.
        """
        self.labels = [m[0] for m in meshes]
        self.faces = [np.asarray(m[2], np.int64) for m in meshes]

        counts = [len(m[1]) for m in meshes]
        offsets = np.concatenate([[0], np.cumsum(counts)])
        allv = np.concatenate([np.asarray(m[1], np.float64) for m in meshes])

        # Exact integer key. Positions are k/256 of a voxel by construction, and
        # a cell's offset lies strictly inside it, so rounding is exact.
        ints = np.rint(allv * SUBVOXEL).astype(np.int64)
        assert ints.min() >= 0
        span = int(max(shape)) * SUBVOXEL + 2 * SUBVOXEL
        key = (ints[:, 0] * span + ints[:, 1]) * span + ints[:, 2]

        node = np.full(len(allv), -1, np.int64)
        if share:
            uniq, inverse = np.unique(key, return_inverse=True)
            shared_n = len(uniq)
        else:
            inverse = np.arange(len(allv))
            shared_n = len(allv)

        # A label may contribute more than one vertex at the same position: the
        # extractor splits a cell's vertex per surface sheet where a label
        # touches itself only diagonally. Welding those would fuse sheets that
        # are deliberately separate, so each gets a private node instead. Rare
        # (well under 1% of cells) but counted, not assumed away.
        self.multi_sheet = 0
        private = shared_n
        node[:] = inverse
        for i in range(len(meshes)) if share else ():
            lo, hi = offsets[i], offsets[i + 1]
            sub = inverse[lo:hi]
            order = np.argsort(sub, kind="stable")
            s = sub[order]
            dup_start = np.flatnonzero(np.diff(s) == 0)
            if len(dup_start) == 0:
                continue
            dup_rows = np.unique(np.concatenate([dup_start, dup_start + 1]))
            rows = order[dup_rows]
            self.multi_sheet += len(rows)
            node[lo + rows] = private + np.arange(len(rows))
            private += len(rows)

        self.n_nodes = private
        self.n_shared = shared_n
        self.node_of = [node[offsets[i] : offsets[i + 1]] for i in range(len(meshes))]
        self.copies = len(allv)

        # One position per node, taken from the first contributor. Every
        # contributor holds the same value, which is the whole point.
        self.positions = np.zeros((self.n_nodes, 3), np.float64)
        self.positions[node] = allv

        self.shape = shape
        self.pinned = np.zeros(self.n_nodes, bool)
        if pin_border:
            # Cells in the outermost layer have an incomplete one-ring, the same
            # set serra pins so that a chunk stays reproducible from its own
            # array.
            p = self.positions
            hi = np.asarray(shape, np.float64) - 2.0
            self.pinned = (p <= 1.0).any(axis=1) | (p >= hi).any(axis=1)

    # -- adjacency ---------------------------------------------------------

    def adjacency(self):
        """Node ring neighbours in CSR form, unioned over every label.

        Built from triangle edges, so it includes each quad's split diagonal.
        The Rust relaxation uses quad ring edges and so excludes them; both
        arms of the comparison below use this same construction, so the
        comparison stays fair even though the operator differs slightly from
        the shipped one.
        """
        e = []
        for f, nodes in zip(self.faces, self.node_of):
            t = nodes[f]
            e.append(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]))
        e = np.concatenate(e)
        e = np.concatenate([e, e[:, ::-1]])
        e = e[e[:, 0] != e[:, 1]]
        e = np.unique(e, axis=0)
        counts = np.bincount(e[:, 0], minlength=self.n_nodes)
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return starts, np.ascontiguousarray(e[:, 1])

    def unique_faces(self):
        """Every wall once, as node triples.

        A wall between labels A and B is emitted into *both* meshes with
        reversed winding. `n n^T` is sign-invariant, so gathering faces per
        label does not cancel that duplication — it doubles it, weighting
        interior walls twice against walls facing background. Deduplicating in
        the node domain, on the sorted triple, counts each wall once.
        """
        tri = np.concatenate([nodes[f] for f, nodes in zip(self.faces, self.node_of)])
        return np.unique(np.sort(tri, axis=1), axis=0)

    # -- the structure tensor ---------------------------------------------

    def structure_tensor(self, resolution):
        """T = sum of n n^T over incident walls, normalised to unit trace.

        Normalising makes the eigenvalues independent of how many faces happen
        to meet at a node — valence here runs 3 to 6 — so the damping below is a
        statement about shape rather than about connectivity.

        Returned as the 6 unique components, not an (N,3,3) stack: the update
        needs a 3x3 solve, not an eigendecomposition, and the solve is 30x
        cheaper.
        """
        tri = self.unique_faces()
        v = self.positions * resolution
        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-12
        n = n[ok] / ln[ok, None]
        flat = tri[ok].ravel()

        comps = np.empty((self.n_nodes, 6))
        pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        for k, (i, j) in enumerate(pairs):
            comps[:, k] = np.bincount(
                flat, weights=np.repeat(n[:, i] * n[:, j], 3), minlength=self.n_nodes
            )
        return comps


def damped_step(comps, target, current, gamma):
    """GAMer's Eq. 5, as a 3x3 solve rather than an eigendecomposition.

    Because the eigenvectors are orthonormal and T = sum_k lambda_k e_k e_k^T,
    the per-eigendirection damping sum_k 1/(1 + gamma*lambda_k) ((x_bar-x).e_k) e_k
    is identically (I + gamma*T)^-1 (x_bar - x). T is a sum of outer products so
    it is positive semi-definite, which makes I + gamma*T symmetric positive
    definite with eigenvalues at least 1 — always invertible, well conditioned,
    and solvable in closed form.

    What this operator actually does is worth stating plainly, because it is not
    an anisotropic version of `max_deviation`. On a locally flat patch every
    incident normal is the same, so T is rank 1 with its large eigenvalue along
    the *surface normal*: the damping suppresses motion off the surface and
    leaves motion along it free. On a crease the normals span two directions and
    only motion along the crease line survives. At a corner nothing moves. It is
    a "slide along the surface, do not move off it" constraint — which bounds
    normal displacement, and bounds tangential displacement not at all.
    """
    trace = np.maximum(comps[:, 0] + comps[:, 3] + comps[:, 5], 1e-30)
    g = gamma / trace
    a00 = 1.0 + g * comps[:, 0]
    a01 = g * comps[:, 1]
    a02 = g * comps[:, 2]
    a11 = 1.0 + g * comps[:, 3]
    a12 = g * comps[:, 4]
    a22 = 1.0 + g * comps[:, 5]

    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c11 = a00 * a22 - a02 * a02
    c12 = a02 * a01 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02

    b = target - current
    out = np.empty_like(b)
    out[:, 0] = (c00 * b[:, 0] + c01 * b[:, 1] + c02 * b[:, 2]) / det
    out[:, 1] = (c01 * b[:, 0] + c11 * b[:, 1] + c12 * b[:, 2]) / det
    out[:, 2] = (c02 * b[:, 0] + c12 * b[:, 1] + c22 * b[:, 2]) / det
    return current + out


# --------------------------------------------------------------------------
# the two smoothers, sharing one Laplacian
# --------------------------------------------------------------------------


def smooth(
    graph,
    resolution,
    iterations,
    step=0.5,
    mode="isotropic",
    max_deviation=0.5,
    lst_scale=8.0,
    eigen=None,
):
    """Jacobi Laplacian smoothing in the cell domain.

    `mode="isotropic"` reproduces what serra does today: clamp each axis to
    `max_deviation` voxels of the placed position. `mode="lst"` instead damps the
    step per eigendirection of the local structure tensor, which is GAMer's
    Eq. 5 with an explicit scale on the eigenvalues.
    """
    starts, neighbours = graph.adjacency()
    resolution = np.asarray(resolution, np.float64)
    origin = graph.positions * resolution
    current = origin.copy()
    pinned = graph.pinned
    counts = np.diff(starts)
    free = (counts > 0) & ~pinned

    if mode == "lst":
        comps = eigen if eigen is not None else graph.structure_tensor(resolution)

    limit = max_deviation * resolution
    src = np.repeat(np.arange(graph.n_nodes), counts)

    for _ in range(iterations):
        acc = np.zeros_like(current)
        for k in range(3):
            acc[:, k] = np.bincount(
                src, weights=current[neighbours, k], minlength=graph.n_nodes
            )
        target = acc / np.maximum(counts, 1)[:, None]
        delta = step * (target - current)

        if mode == "lst":
            moved = damped_step(comps, current + delta, current, lst_scale)
        else:
            moved = np.clip(current + delta, origin - limit, origin + limit)

        current = np.where(free[:, None], moved, current)

    return current / resolution


def scatter(graph, positions):
    """Node positions back onto every label's own vertex array."""
    return [positions[nodes] for nodes in graph.node_of]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def mesh_all(serra_mesh, volume_array, close, labels=None):
    """Every label's mesh at unit voxel resolution, unsmoothed."""
    mesher = serra_mesh.Mesher().mesh(volume_array, close=close)
    ids = [int(i) for i in mesher.ids()] if labels is None else labels
    out = []
    for label in ids:
        got = mesher.get(label)
        if len(got.faces):
            out.append((label, np.asarray(got.vertices, np.float64), got.faces))
    return out


def label_domain_reference(serra_mesh, volume_array, close, iterations, max_deviation):
    """What serra does today: independent relaxation per label."""
    mesher = serra_mesh.Mesher(relaxation=iterations, max_deviation=max_deviation).mesh(
        volume_array, close=close
    )
    return [
        (
            int(i),
            np.asarray(mesher.get(int(i)).vertices, np.float64),
            mesher.get(int(i)).faces,
        )
        for i in mesher.ids()
        if len(mesher.get(int(i)).faces)
    ]


def shared_wall_drift(shared, verts):
    """Spread among the copies of each shared cell, in whatever units `verts` is.

    Correspondence comes from the UNSMOOTHED graph, so it survives whatever the
    smoother did. Re-welding the smoothed output by position would instead find
    nothing shared, which is precisely the defect being measured.
    """
    total = np.zeros((shared.n_nodes, 3))
    seen = np.zeros(shared.n_nodes, np.int64)
    for v, nodes in zip(verts, shared.node_of):
        np.add.at(total, nodes, v)
        np.add.at(seen, nodes, 1)
    mean = total / np.maximum(seen, 1)[:, None]

    spread = np.zeros(shared.n_nodes)
    for v, nodes in zip(verts, shared.node_of):
        np.maximum.at(spread, nodes, np.linalg.norm(v - mean[nodes], axis=1))
    return spread[seen > 1] * 2.0  # worst pair separation, not radius


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------


def sweep(graph, resolution, meshes, mode, raw, raw_area, truth, eigen, args, steps):
    """Run one smoother at increasing strength, recording what it cost."""
    rows = []
    for iterations in steps:
        kwargs = (
            dict(mode="isotropic", max_deviation=args.max_deviation)
            if mode == "isotropic"
            else dict(mode="lst", lst_scale=args.lst_scale, eigen=eigen)
        )
        pos = smooth(graph, resolution, iterations, **kwargs)
        verts = scatter(graph, pos)
        move = np.linalg.norm(np.concatenate(verts) - np.concatenate(raw), axis=1)
        rows.append(
            dict(
                iterations=iterations,
                move=float(move.mean()),
                volume=sum(volume(v, f) for v, (_, _, f) in zip(verts, meshes)) / truth,
                area=sum(area(v, f) for v, (_, _, f) in zip(verts, meshes)) / raw_area,
                rough=float(
                    np.mean(
                        [
                            roughness_degrees(v, f)
                            for v, (_, _, f) in zip(verts, meshes)
                            if len(f) > 4
                        ]
                    )
                ),
            )
        )
    return rows


def at_displacement(rows, target):
    """Linear interpolation of a sweep to a fixed mean displacement.

    Comparing two smoothers at equal parameters is meaningless when they move
    vertices by different amounts — the one that moves less will always look
    like it preserves more. Matching displacement first is what makes the
    comparison about the *shape* of the constraint rather than its strength.
    """
    move = np.array([r["move"] for r in rows])
    if target < move.min() or target > move.max():
        return None
    out = {"move": target}
    for key in ("volume", "area", "rough"):
        out[key] = float(np.interp(target, move, [r[key] for r in rows]))
    return out


def experiment_shapes(serra_mesh, args):
    """Where the structure tensor should and should not help."""
    print("=" * 78)
    print("1. Feature preservation: LST damping vs the isotropic box")
    print("=" * 78)
    print(
        "   Both smoothers are the same Jacobi Laplacian; only the constraint on\n"
        "   the step differs. Each is swept over iteration count and then read off\n"
        "   at a MATCHED mean displacement, because a smoother that simply moves\n"
        "   vertices less will always look like it preserves more.\n"
    )
    print(
        "   The sphere is the control: its structure tensor is near-isotropic, so\n"
        "   LST damping has no anisotropy to exploit and the two should tie. The\n"
        "   cylinder rim and the box edges are the actual test.\n"
    )

    fixtures = [
        ("sphere r=20", sphere_mask(20), None),
        ("cylinder r=12 h=24", cylinder_mask(12, 24), None),
        ("box 24^3", box_mask(24), 24),
    ]
    steps = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

    for name, mask, side in fixtures:
        truth = float(mask.sum())
        meshes = mesh_all(serra_mesh, mask, close=True)
        graph = CellGraph(meshes, mask.shape, pin_border=False)
        eigen = graph.structure_tensor(np.ones(3))
        raw = [v for _, v, _ in meshes]
        raw_area = sum(area(v, f) for _, v, f in meshes)
        raw_vol = sum(volume(v, f) for _, v, f in meshes) / truth
        raw_rough = float(
            np.mean([roughness_degrees(v, f) for _, v, f in meshes if len(f) > 4])
        )

        print(
            f"   {name}: unsmoothed volume {raw_vol:.2%}, dihedral {raw_rough:.1f}d",
            end="",
        )
        if side is not None:
            print(
                f", exact bevel loss {3 * side - 2} voxels ({(3 * side - 2) / truth:.2%})"
            )
        else:
            print()

        sweeps = {
            m: sweep(
                graph, np.ones(3), meshes, m, raw, raw_area, truth, eigen, args, steps
            )
            for m in ("isotropic", "lst")
        }
        lo = max(min(r["move"] for r in sweeps[m]) for m in sweeps)
        hi = min(max(r["move"] for r in sweeps[m]) for m in sweeps)

        print(
            f"{'mean move':>12} {'constraint':>12} {'volume/true':>12} "
            f"{'area/raw':>9} {'dihedral':>10}"
        )
        for target in np.linspace(lo, hi, 4):
            for m in ("isotropic", "lst"):
                r = at_displacement(sweeps[m], target)
                if r is None:
                    continue
                print(
                    f"{format(target, '.3f') + 'v' if m == 'isotropic' else '':>12} "
                    f"{m:>12} "
                    f"{r['volume']:>11.2%} {r['area']:>8.1%} {r['rough']:>9.1f}d"
                )
        print()


def experiment_neuropil(serra_mesh, args):
    """The two things cell-domain smoothing is actually for."""
    print("=" * 78)
    print("2. Real neuropil: drift, cost, and quality")
    print("=" * 78)

    with gzip.open(args.volume, "rb") as handle:
        vol = np.load(handle)
    n = args.subvolume
    sub = np.ascontiguousarray(vol[:n, :n, :n])
    resolution = np.asarray(args.resolution, np.float64)
    print(f"   subvolume {sub.shape}, {len(np.unique(sub)) - 1:,} objects\n")

    meshes = mesh_all(serra_mesh, sub, close=False)
    shared = CellGraph(meshes, sub.shape)
    private = CellGraph(meshes, sub.shape, share=False)

    print(f"   label-copies of a cell vertex : {shared.copies:,}")
    print(f"   distinct cells (shared nodes) : {shared.n_shared:,}")
    print(f"   -> cell domain is {shared.copies / shared.n_shared:.2f}x smaller")
    print(
        f"   multi-sheet copies given a private node : {shared.multi_sheet:,} "
        f"({shared.multi_sheet / shared.copies:.3%})\n"
    )

    t0 = time.perf_counter()
    eigen = shared.structure_tensor(resolution)
    t_lst = time.perf_counter() - t0
    print(
        f"   structure tensor for {shared.n_nodes:,} nodes: {t_lst:.2f}s "
        f"({t_lst / shared.n_nodes * 1e9:.0f} ns each, computed once)\n"
    )

    raw_verts = [v * resolution for _, v, _ in meshes]
    raw_area = sum(area(v, f) for v, (_, _, f) in zip(raw_verts, meshes))
    raw_rough = np.mean(
        [
            roughness_degrees(v, f)
            for v, (_, _, f) in zip(raw_verts, meshes)
            if len(f) > 4
        ]
    )
    raw_ang = np.concatenate(
        [triangle_angles(v, f) for v, (_, _, f) in zip(raw_verts, meshes)]
    )

    voxel_volume = float(np.prod(resolution))
    ids, counts = np.unique(sub, return_counts=True)
    truth = dict(zip(ids.tolist(), counts.tolist()))
    closed = [i for i, (label, _, f) in enumerate(meshes) if is_closed(f)]
    truth_volume = sum(truth[meshes[i][0]] * voxel_volume for i in closed)

    def measure(name, verts, elapsed=None):
        ar = sum(area(v, f) for v, (_, _, f) in zip(verts, meshes)) / raw_area
        rough = np.mean(
            [
                roughness_degrees(v, f)
                for v, (_, _, f) in zip(verts, meshes)
                if len(f) > 4
            ]
        )
        ang = np.concatenate(
            [triangle_angles(v, f) for v, (_, _, f) in zip(verts, meshes)]
        )
        vol = sum(volume(verts[i], meshes[i][2]) for i in closed) / truth_volume
        drift = shared_wall_drift(shared, verts) / resolution.min()
        print(
            f"{name:>26} {ar:>8.1%} {rough:>9.1f}d {((ang >= 40) & (ang <= 80)).mean():>10.1%} "
            f"{vol:>10.2%} {np.median(drift):>8.3f}v {drift.max():>8.3f}v"
            + (f" {elapsed:>6.1f}s" if elapsed else "")
        )

    print(
        f"{'smoothing':>26} {'area/raw':>9} {'dihedral':>10} {'in [40,80]':>11} "
        f"{'volume/true':>11} {'drift med':>9} {'drift max':>9}"
    )
    measure("none", raw_verts)

    runs = [
        (
            "per label, isotropic",
            private,
            dict(mode="isotropic", max_deviation=args.max_deviation),
        ),
        (
            "cell domain, isotropic",
            shared,
            dict(mode="isotropic", max_deviation=args.max_deviation),
        ),
        (
            "cell domain, lst",
            shared,
            dict(mode="lst", lst_scale=args.lst_scale, eigen=eigen),
        ),
    ]
    for name, graph, kwargs in runs:
        t0 = time.perf_counter()
        pos = smooth(graph, resolution, args.iterations, **kwargs)
        dt = time.perf_counter() - t0
        measure(name, [p * resolution for p in scatter(graph, pos)], dt)
    print(
        "\n   'drift' is the worst separation between two labels' copies of the same\n"
        "   wall. Zero in the cell domain by construction; the per-label row is what\n"
        "   serra does today, run through the identical operator so the only\n"
        "   difference is the domain.\n"
    )


def is_closed(faces):
    e = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    _, c = np.unique(e, axis=0, return_counts=True)
    return bool((c == 2).all())


def triangle_angles(v, f):
    v = np.asarray(v, np.float64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    out = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u, w = q - p, r - p
        nu, nw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
        ok = (nu > 1e-12) & (nw > 1e-12)
        cos = np.einsum("ij,ij->i", u[ok], w[ok]) / (nu[ok] * nw[ok])
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.concatenate(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="data/microns_neuropil.npy.gz")
    parser.add_argument("--subvolume", type=int, default=160)
    parser.add_argument("--resolution", type=float, nargs=3, default=[32.0, 32.0, 40.0])
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--max-deviation", type=float, default=0.5)
    parser.add_argument("--lst-scale", type=float, default=8.0)
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--neuropil-only", action="store_true")
    args = parser.parse_args()

    import serra_mesh

    if not args.neuropil_only:
        experiment_shapes(serra_mesh, args)
    if not args.analytic_only:
        experiment_neuropil(serra_mesh, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
