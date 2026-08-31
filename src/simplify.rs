//! Quadric error edge collapse (Garland & Heckbert).
//!
//! # What is guaranteed
//!
//! A collapse is applied only if it passes all of the following, so a closed
//! 2-manifold input comes out a closed 2-manifold:
//!
//! * **The link condition** (Dey et al.). Collapsing an edge preserves topology
//!   exactly when the vertices adjacent to *both* endpoints are precisely the
//!   vertices opposite that edge. Skipping this check is the usual way a naive
//!   QEM implementation silently produces a pinched vertex or an edge shared by
//!   three triangles.
//! * **No duplicate faces.** The link condition alone still permits collapsing
//!   a tetrahedron into two coincident triangles.
//! * **No normal flips.** A collapse can be topologically fine and still fold
//!   the surface back through itself.
//! * **Pinned endpoints are never collapsed.**
//!
//! # Why pinned vertices matter more than they appear to
//!
//! Holding seam vertices *in place* is not enough for chunked meshing. Two
//! chunks sharing a seam must also agree on the *edges* between seam vertices,
//! otherwise an edge is used once from one side and not at all from the other,
//! and the stitched surface leaks.
//!
//! Refusing any collapse with a pinned endpoint gives that for free. A face
//! carrying two pinned vertices can never die, because all three of its edges
//! touch a pinned vertex; and no collapse can create a new pinned-to-pinned
//! edge, because a merged vertex is never itself pinned. So the seam's
//! connectivity comes out exactly as it went in, in every chunk independently.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::mesh::TriangleMesh;

/// A symmetric 4x4 quadric as its upper triangle:
/// `[a², ab, ac, ad, b², bc, bd, c², cd, d²]` for the plane `ax + by + cz + d = 0`.
type Quadric = [f64; 10];

fn plane_quadric(n: [f64; 3], d: f64) -> Quadric {
    let [a, b, c] = n;
    [
        a * a,
        a * b,
        a * c,
        a * d,
        b * b,
        b * c,
        b * d,
        c * c,
        c * d,
        d * d,
    ]
}

#[inline]
fn add_quadric(into: &mut Quadric, other: &Quadric) {
    for k in 0..10 {
        into[k] += other[k];
    }
}

/// `vᵀ Q v` for `v = (x, y, z, 1)` — the sum of squared distances to the planes
/// that built the quadric.
fn quadric_error(q: &Quadric, p: [f64; 3]) -> f64 {
    let [x, y, z] = p;
    q[0] * x * x
        + 2.0 * q[1] * x * y
        + 2.0 * q[2] * x * z
        + 2.0 * q[3] * x
        + q[4] * y * y
        + 2.0 * q[5] * y * z
        + 2.0 * q[6] * y
        + q[7] * z * z
        + 2.0 * q[8] * z
        + q[9]
}

/// The position minimising the quadric, when the system is well conditioned.
fn optimal_position(q: &Quadric) -> Option<[f64; 3]> {
    let m = [[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]];
    let rhs = [-q[3], -q[6], -q[8]];

    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    // Scale-aware: a flat or symmetric neighbourhood leaves this singular, and
    // inverting it anyway flings the vertex to infinity.
    let scale = (q[0].abs() + q[4].abs() + q[7].abs()).max(1e-12);
    if !det.is_finite() || det.abs() <= 1e-10 * scale * scale * scale {
        return None;
    }

    let cofactor = |r: usize, c: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
        (m[c1][r1] * m[c2][r2] - m[c1][r2] * m[c2][r1]) / det
    };
    let mut out = [0.0f64; 3];
    for (r, slot) in out.iter_mut().enumerate() {
        *slot = (0..3).map(|c| cofactor(r, c) * rhs[c]).sum();
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// A candidate collapse, ordered cheapest-first.
///
/// The ordering is a strict total order — error, then the two vertex ids — so
/// the sequence of collapses is determined entirely by the mesh, never by heap
/// or hash iteration order.
#[derive(Debug)]
struct Candidate {
    error: f64,
    u: u32,
    v: u32,
    target: [f64; 3],
    stamp: (u64, u64),
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap is a max-heap, and the cheapest should pop first.
        other
            .error
            .total_cmp(&self.error)
            .then(other.u.cmp(&self.u))
            .then(other.v.cmp(&self.v))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// How far to simplify.
#[derive(Clone, Copy, Debug)]
pub struct SimplifyOptions {
    /// Stop once the mesh has at most this many faces.
    pub target_faces: usize,
    /// How far a vertex may end up from where it started, in the mesh's own
    /// units.
    ///
    /// This is enforced as a real geometric bound, not just as a threshold on
    /// the quadric: a collapse is rejected unless the merged position is within
    /// `max_error` of the original position of *both* endpoints. A quadric
    /// error is a sum of squared distances to planes and grows with vertex
    /// valence, so thresholding it alone lets vertices wander further the
    /// busier the neighbourhood — which is not what a caller passing "40 nm"
    /// expects.
    pub max_error: f64,
}

/// Reusable buffers, so the safety checks allocate nothing per candidate.
#[derive(Default)]
struct Scratch {
    of_u: Vec<u32>,
    of_v: Vec<u32>,
    shared: Vec<u32>,
    opposite: Vec<u32>,
    survivors: Vec<[u32; 3]>,
}

struct State {
    positions: Vec<[f64; 3]>,
    /// Where each vertex started. Never updated, so the displacement bound is
    /// measured against the input surface rather than the previous step.
    original: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
    face_alive: Vec<bool>,
    incident: Vec<Vec<u32>>,
    quadrics: Vec<Quadric>,
    pinned: Vec<bool>,
    stamp: Vec<u64>,
    dead_vertex: Vec<bool>,
    live_faces: usize,
}

impl State {
    /// Vertices sharing a live face with `v`, into a caller-owned buffer.
    ///
    /// A vertex has around six neighbours, so a short sorted vector beats a
    /// hash set comfortably — and reusing the caller's buffer keeps the hot
    /// path free of allocation, which is where most of the time went before.
    fn neighbours_into(&self, v: u32, out: &mut Vec<u32>) {
        out.clear();
        for &f in &self.incident[v as usize] {
            if !self.face_alive[f as usize] {
                continue;
            }
            for &w in &self.faces[f as usize] {
                if w != v {
                    out.push(w);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// The vertices opposite edge `(u, v)` — one per face carrying that edge.
    fn opposite_into(&self, u: u32, v: u32, out: &mut Vec<u32>) {
        out.clear();
        for &f in &self.incident[u as usize] {
            if !self.face_alive[f as usize] {
                continue;
            }
            let tri = self.faces[f as usize];
            if tri.contains(&v) {
                for &w in &tri {
                    if w != u && w != v {
                        out.push(w);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// Every condition that has to hold for this collapse to be safe.
    ///
    /// `scratch` carries three reusable buffers so the checks allocate nothing.
    fn is_safe(
        &self,
        u: u32,
        v: u32,
        target: [f64; 3],
        max_error: f64,
        scratch: &mut Scratch,
    ) -> bool {
        if self.pinned[u as usize] || self.pinned[v as usize] {
            return false;
        }

        // Hard displacement bound, against the original surface.
        let limit = max_error * max_error;
        for endpoint in [u, v] {
            let o = self.original[endpoint as usize];
            let d = [target[0] - o[0], target[1] - o[1], target[2] - o[2]];
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > limit {
                return false;
            }
        }

        // The link condition. Anything adjacent to both endpoints that is not
        // opposite the edge becomes a non-manifold junction after the merge.
        self.opposite_into(u, v, &mut scratch.opposite);
        if scratch.opposite.is_empty() {
            return false;
        }
        self.neighbours_into(u, &mut scratch.of_u);
        self.neighbours_into(v, &mut scratch.of_v);
        // Both lists are sorted, so the intersection is a merge walk.
        scratch.shared.clear();
        let (mut i, mut j) = (0usize, 0usize);
        while i < scratch.of_u.len() && j < scratch.of_v.len() {
            match scratch.of_u[i].cmp(&scratch.of_v[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    scratch.shared.push(scratch.of_u[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        if scratch.shared != scratch.opposite {
            return false;
        }

        // A closed component cannot shrink past a tetrahedron without
        // degenerating, and the link condition does not catch that.
        if self.live_faces <= 4 {
            return false;
        }

        scratch.survivors.clear();
        for &f in self.incident[u as usize]
            .iter()
            .chain(self.incident[v as usize].iter())
        {
            let f = f as usize;
            if !self.face_alive[f] {
                continue;
            }
            let tri = self.faces[f];
            if tri.contains(&u) && tri.contains(&v) {
                continue; // dies along with the edge
            }
            let moved = tri.map(|w| if w == v { u } else { w });
            if moved[0] == moved[1] || moved[1] == moved[2] || moved[0] == moved[2] {
                return false;
            }
            if self.flips(tri, u, v, target) {
                return false;
            }
            let mut key = moved;
            key.sort_unstable();
            if scratch.survivors.contains(&key) {
                return false; // two faces would end up coincident
            }
            scratch.survivors.push(key);
        }
        true
    }

    /// Whether moving `u`/`v` to `target` turns this triangle inside out.
    fn flips(&self, tri: [u32; 3], u: u32, v: u32, target: [f64; 3]) -> bool {
        let before = normal(tri.map(|w| self.positions[w as usize]));
        let after = normal(tri.map(|w| {
            if w == u || w == v {
                target
            } else {
                self.positions[w as usize]
            }
        }));
        match (before, after) {
            (Some(a), Some(b)) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] <= 0.0,
            // A triangle collapsing to zero area has no orientation left.
            (Some(_), None) => true,
            _ => false,
        }
    }

    fn apply(&mut self, u: u32, v: u32, target: [f64; 3]) {
        self.positions[u as usize] = target;
        let mut merged = self.quadrics[u as usize];
        add_quadric(&mut merged, &self.quadrics[v as usize]);
        self.quadrics[u as usize] = merged;

        let moving: Vec<u32> = std::mem::take(&mut self.incident[v as usize]);
        for f in moving {
            let fi = f as usize;
            if !self.face_alive[fi] {
                continue;
            }
            if self.faces[fi].contains(&u) {
                self.face_alive[fi] = false;
                self.live_faces -= 1;
            } else {
                for slot in self.faces[fi].iter_mut() {
                    if *slot == v {
                        *slot = u;
                    }
                }
                self.incident[u as usize].push(f);
            }
        }
        self.dead_vertex[v as usize] = true;
        // Only the two endpoints need invalidating. A neighbour's quadric is
        // untouched by this collapse, so any queued edge between two of them
        // still has the right cost; and anything that has become unsafe is
        // caught by the check at pop time. Bumping the whole neighbourhood
        // instead would churn the heap for nothing.
        self.stamp[u as usize] += 1;
        self.stamp[v as usize] += 1;
        let alive = &self.face_alive;
        self.incident[u as usize].retain(|&f| alive[f as usize]);
    }
}

fn normal(t: [[f64; 3]; 3]) -> Option<[f64; 3]> {
    let e1 = [t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]];
    let e2 = [t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    (len > 0.0).then(|| [n[0] / len, n[1] / len, n[2] / len])
}

/// Where the merged vertex should go, and what it costs.
fn best_target(state: &State, u: u32, v: u32) -> ([f64; 3], f64) {
    let mut q = state.quadrics[u as usize];
    add_quadric(&mut q, &state.quadrics[v as usize]);

    let pu = state.positions[u as usize];
    let pv = state.positions[v as usize];
    let mid = [
        (pu[0] + pv[0]) / 2.0,
        (pu[1] + pv[1]) / 2.0,
        (pu[2] + pv[2]) / 2.0,
    ];

    let mut best = mid;
    let mut cost = quadric_error(&q, mid);
    for candidate in [pu, pv].into_iter().chain(optimal_position(&q)) {
        let c = quadric_error(&q, candidate);
        if c < cost {
            cost = c;
            best = candidate;
        }
    }
    (best, cost.max(0.0))
}

/// Vertices on an open rim, which would otherwise erode inward.
fn boundary_vertices(faces: &[[u32; 3]], n: usize) -> Vec<bool> {
    let mut uses: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    for f in faces {
        for i in 0..3 {
            let (a, b) = (f[i], f[(i + 1) % 3]);
            *uses.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
        }
    }
    let mut out = vec![false; n];
    for (&(a, b), &count) in uses.iter() {
        if count != 2 {
            out[a as usize] = true;
            out[b as usize] = true;
        }
    }
    out
}

/// Simplify in place. Returns the number of collapses applied.
pub fn simplify(mesh: &mut TriangleMesh, opts: &SimplifyOptions) -> usize {
    if mesh.faces.len() <= opts.target_faces || mesh.faces.len() <= 4 {
        return 0;
    }

    let n = mesh.vertices.len();
    let positions: Vec<[f64; 3]> = mesh
        .vertices
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    let mut incident: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, f) in mesh.faces.iter().enumerate() {
        for &w in f {
            incident[w as usize].push(i as u32);
        }
    }

    // Fundamental quadrics: one plane per incident triangle.
    let mut quadrics: Vec<Quadric> = vec![[0.0; 10]; n];
    for f in mesh.faces.iter() {
        let p = f.map(|w| positions[w as usize]);
        let Some(nrm) = normal(p) else { continue };
        let d = -(nrm[0] * p[0][0] + nrm[1] * p[0][1] + nrm[2] * p[0][2]);
        let q = plane_quadric(nrm, d);
        for &w in f {
            add_quadric(&mut quadrics[w as usize], &q);
        }
    }

    // Pin the caller's set plus the mesh's own open rim.
    let mut pinned = boundary_vertices(&mesh.faces, n);
    for (slot, &p) in pinned.iter_mut().zip(mesh.pinned.iter()) {
        *slot |= p;
    }

    let live_faces = mesh.faces.len();
    let mut state = State {
        original: positions.clone(),
        positions,
        faces: mesh.faces.clone(),
        face_alive: vec![true; live_faces],
        incident,
        quadrics,
        pinned,
        stamp: vec![0; n],
        dead_vertex: vec![false; n],
        live_faces,
    };

    let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
    let queue = |heap: &mut BinaryHeap<Candidate>, state: &State, a: u32, b: u32| {
        let key = if a < b { (a, b) } else { (b, a) };
        if state.pinned[key.0 as usize] || state.pinned[key.1 as usize] {
            return;
        }
        let (target, error) = best_target(state, key.0, key.1);
        heap.push(Candidate {
            error,
            u: key.0,
            v: key.1,
            target,
            stamp: (state.stamp[key.0 as usize], state.stamp[key.1 as usize]),
        });
    };

    let mut seeded: FxHashSet<(u32, u32)> = FxHashSet::default();
    for f in mesh.faces.iter() {
        for i in 0..3 {
            let (a, b) = (f[i], f[(i + 1) % 3]);
            let key = if a < b { (a, b) } else { (b, a) };
            if seeded.insert(key) {
                queue(&mut heap, &state, key.0, key.1);
            }
        }
    }

    let mut collapses = 0usize;
    let mut scratch = Scratch::default();
    let mut touched: Vec<u32> = Vec::with_capacity(16);

    while state.live_faces > opts.target_faces {
        let Some(c) = heap.pop() else { break };
        if state.dead_vertex[c.u as usize] || state.dead_vertex[c.v as usize] {
            continue;
        }
        if c.stamp != (state.stamp[c.u as usize], state.stamp[c.v as usize]) {
            // Stale: the neighbourhood changed since this was queued.
            queue(&mut heap, &state, c.u, c.v);
            continue;
        }
        if !state.is_safe(c.u, c.v, c.target, opts.max_error, &mut scratch) {
            continue;
        }

        state.apply(c.u, c.v, c.target);
        collapses += 1;

        // Every edge at the merged vertex has a new cost.
        state.neighbours_into(c.u, &mut touched);
        for &w in touched.iter() {
            queue(&mut heap, &state, c.u, w);
        }
    }

    if collapses == 0 {
        return 0;
    }

    // Rebuild, dropping dead faces and now-unreferenced vertices.
    let mut remap = vec![u32::MAX; n];
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut kept_pinned: Vec<bool> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();
    for (i, f) in state.faces.iter().enumerate() {
        if !state.face_alive[i] {
            continue;
        }
        let mut tri = [0u32; 3];
        for (k, &w) in f.iter().enumerate() {
            if remap[w as usize] == u32::MAX {
                remap[w as usize] = vertices.len() as u32;
                let p = state.positions[w as usize];
                vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
                kept_pinned.push(state.pinned[w as usize]);
            }
            tri[k] = remap[w as usize];
        }
        faces.push(tri);
    }

    mesh.vertices = vertices;
    mesh.faces = faces;
    mesh.pinned = kept_pinned;
    if mesh.normals.is_some() {
        mesh.normals = Some(crate::mesh::vertex_normals(&mesh.vertices, &mesh.faces));
    }
    collapses
}
