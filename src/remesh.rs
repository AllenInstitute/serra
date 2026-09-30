//! Editing the shared wall mesh: the operations remeshing is built from.
//!
//! [`Remesh`] holds a [`WallMesh`] in a form that can be changed locally --
//! each vertex knows the triangles around it -- and offers the four operations
//! of isotropic remeshing: [`split`](Remesh::split),
//! [`collapse`](Remesh::collapse), [`flip`](Remesh::flip) and
//! [`relocate`](Remesh::relocate). Each checks that it is allowed and leaves
//! the mesh untouched if not, so a remeshing policy can simply try operations
//! and keep what succeeds.
//!
//! Because every wall is stored once, every operation is automatically the
//! same for both labels on it: two touching objects cannot come apart.
//!
//! # What each vertex may do
//!
//! Following Faraj et al.'s multi-material remesher, a vertex's freedom depends
//! on what meets there, read from the label pairs of the triangles around it
//! ([`Remesh::vertex_kind`]):
//!
//! * a **wall** vertex, whose triangles all lie on one wall between two labels,
//!   may move within that wall and collapse into any neighbour;
//! * a **curve** vertex, on a curve where three or more labels meet (exactly
//!   two junction edges), may move along its curve and collapse along it;
//! * a **corner**, where curves meet or end, never moves and is never
//!   collapsed away, though neighbours may collapse into it;
//! * a **locked** vertex keeps not only its position but every triangle around
//!   it exactly as it is: no operation may create, destroy or rewire a triangle
//!   touching it.
//!
//! Locked are: vertices the manifold repair may split (from a cell with an
//! ambiguous face, or a [`WallMesh::pinch`]), whose triangles it must see
//! unchanged; vertices on the chunk's outer cell layer, so that neighbouring
//! chunks still agree about everything near their seam; and vertices on any
//! label's open edge at the volume's boundary. Locking is conservative. It is the simplest rule under
//! which each of those still holds, and it can be relaxed case by case later.
//!
//! # Coordinates
//!
//! Positions are kept in voxel units, as the extraction's fixed-point values
//! divided exactly by 256, so an untouched vertex comes out bit-identical to
//! [`crate::mesh::build`]. Geometric tests -- orientation, area -- are made in
//! physical units, since that is the space the triangles live in.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::extract::CellField;
use crate::mesh::{finish, fixed_to_voxel, MeshOptions, TriangleMesh};
use crate::tables::AMBIGUOUS_CELL;
use crate::walls::{WallMesh, OUTSIDE};

/// What an edge is, from the triangles on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// One triangle: the surface's open edge.
    Boundary,
    /// Two triangles of the same wall, wound consistently.
    Wall,
    /// Anything else: where three or more labels meet.
    Junction,
}

/// What a vertex may do; see the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexKind {
    Locked,
    Wall,
    Curve,
    Corner,
}

/// A [`WallMesh`] that can be edited.
pub struct Remesh {
    /// Positions in voxel units.
    voxel: Vec<[f64; 3]>,
    /// Physical size of a voxel along each array axis.
    scale: [f64; 3],
    faces: Vec<[u32; 3]>,
    front: Vec<u32>,
    back: Vec<u32>,
    face_alive: Vec<bool>,
    /// Live faces around each vertex.
    incident: Vec<Vec<u32>>,
    vertex_alive: Vec<bool>,
    locked: Vec<bool>,
    /// Handed to the manifold repair when a label is extracted.
    suspect: Vec<bool>,
    pinned: Vec<bool>,
    /// Orders the repair; new vertices take a neighbour's, which is never
    /// consulted since they are never suspects.
    cell: Vec<u32>,
    pinch: Vec<bool>,
    alias: FxHashMap<u64, u32>,
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// How much smaller than before a triangle's area may become before an
/// operation is refused as making it degenerate.
const MIN_AREA_RATIO: f64 = 1e-6;

impl Remesh {
    /// Take a wall mesh for editing. `resolution` is the physical voxel size
    /// along each array axis, the space in which shapes are judged.
    pub fn new(walls: &WallMesh, cells: &CellField, resolution: [f64; 3]) -> Remesh {
        let n = walls.positions.len();
        let mut incident: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (f, t) in walls.faces.iter().enumerate() {
            for &v in t {
                incident[v as usize].push(f as u32);
            }
        }
        let suspect: Vec<bool> = (0..n)
            .map(|v| {
                walls.pinch[v] || AMBIGUOUS_CELL[cells.crossings[walls.cells[v] as usize] as usize]
            })
            .collect();
        let pinned: Vec<bool> = (0..n)
            .map(|v| cells.pinned[walls.cells[v] as usize])
            .collect();

        // Vertices on any label's open rim: an edge with only one of that
        // label's faces. Asked per label, not of the whole mesh: where a label's
        // surface ends at the volume's edge, other labels' walls can still meet
        // along the same edge, and the whole mesh shows no rim there.
        let mut edges: Vec<(u32, u64)> = Vec::with_capacity(walls.faces.len() * 6);
        for (f, t) in walls.faces.iter().enumerate() {
            for label in [walls.front[f], walls.back[f]] {
                if label == OUTSIDE {
                    continue;
                }
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    edges.push((label, ((lo as u64) << 32) | hi as u64));
                }
            }
        }
        edges.sort_unstable();
        let mut rim = vec![false; n];
        let mut i = 0;
        while i < edges.len() {
            let mut j = i + 1;
            while j < edges.len() && edges[j] == edges[i] {
                j += 1;
            }
            if j - i == 1 {
                let e = edges[i].1;
                rim[(e >> 32) as usize] = true;
                rim[(e & 0xffff_ffff) as usize] = true;
            }
            i = j;
        }

        let locked = (0..n).map(|v| suspect[v] || pinned[v] || rim[v]).collect();
        Remesh {
            voxel: walls.positions.iter().map(|&p| fixed_to_voxel(p)).collect(),
            scale: resolution,
            faces: walls.faces.clone(),
            front: walls.front.clone(),
            back: walls.back.clone(),
            face_alive: vec![true; walls.faces.len()],
            incident,
            vertex_alive: vec![true; n],
            locked,
            suspect,
            pinned,
            cell: walls.cells.clone(),
            pinch: walls.pinch.clone(),
            alias: walls.alias.clone(),
        }
    }

    /// Number of vertex ids, live or not.
    pub fn vertex_count(&self) -> usize {
        self.voxel.len()
    }

    /// Number of face ids, live or not.
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn face(&self, f: u32) -> Option<[u32; 3]> {
        self.face_alive[f as usize].then(|| self.faces[f as usize])
    }

    pub fn is_alive(&self, v: u32) -> bool {
        self.vertex_alive[v as usize]
    }

    /// Position in voxel units.
    pub fn position(&self, v: u32) -> [f64; 3] {
        self.voxel[v as usize]
    }

    fn physical(&self, p: [f64; 3]) -> [f64; 3] {
        [
            p[0] * self.scale[0],
            p[1] * self.scale[1],
            p[2] * self.scale[2],
        ]
    }

    /// Normal of face `f` in physical units, as if `moved` were at `to` (voxel
    /// units). Its length is twice the triangle's area.
    fn normal_with(&self, f: u32, moved: u32, to: [f64; 3]) -> [f64; 3] {
        let t = self.faces[f as usize];
        let at = |v: u32| {
            self.physical(if v == moved {
                to
            } else {
                self.voxel[v as usize]
            })
        };
        let (a, b, c) = (at(t[0]), at(t[1]), at(t[2]));
        cross(sub(b, a), sub(c, a))
    }

    fn face_locked(&self, f: u32) -> bool {
        self.faces[f as usize]
            .iter()
            .any(|&v| self.locked[v as usize])
    }

    /// Live faces holding both `u` and `v`.
    pub fn edge_faces(&self, u: u32, v: u32) -> Vec<u32> {
        self.incident[u as usize]
            .iter()
            .copied()
            .filter(|&f| self.faces[f as usize].contains(&v))
            .collect()
    }

    /// Whether face `f` runs from `a` to `b` (rather than `b` to `a`).
    fn runs(&self, f: u32, a: u32, b: u32) -> bool {
        let t = self.faces[f as usize];
        (0..3).any(|k| t[k] == a && t[(k + 1) % 3] == b)
    }

    pub fn edge_kind(&self, u: u32, v: u32) -> EdgeKind {
        let faces = self.edge_faces(u, v);
        match faces.as_slice() {
            [_] => EdgeKind::Boundary,
            &[f, g] => {
                let same_wall = self.front[f as usize] == self.front[g as usize]
                    && self.back[f as usize] == self.back[g as usize];
                if same_wall && self.runs(f, u, v) != self.runs(g, u, v) {
                    EdgeKind::Wall
                } else {
                    EdgeKind::Junction
                }
            }
            _ => EdgeKind::Junction,
        }
    }

    /// Distinct neighbours of `v`, ascending.
    pub fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut out: Vec<u32> = self.incident[v as usize]
            .iter()
            .flat_map(|&f| self.faces[f as usize])
            .filter(|&w| w != v)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The neighbours joined to `v` by a junction edge.
    pub fn curve_neighbours(&self, v: u32) -> Vec<u32> {
        self.neighbours(v)
            .into_iter()
            .filter(|&w| self.edge_kind(v, w) == EdgeKind::Junction)
            .collect()
    }

    pub fn vertex_kind(&self, v: u32) -> VertexKind {
        if self.locked[v as usize] {
            return VertexKind::Locked;
        }
        let mut junction = 0;
        for w in self.neighbours(v) {
            match self.edge_kind(v, w) {
                EdgeKind::Boundary => return VertexKind::Locked,
                EdgeKind::Junction => junction += 1,
                EdgeKind::Wall => {}
            }
        }
        match junction {
            0 => VertexKind::Wall,
            2 => VertexKind::Curve,
            _ => VertexKind::Corner,
        }
    }

    // --- split -------------------------------------------------------------

    /// Split edge `(u, v)` at its midpoint. Every face on the edge is split,
    /// so a junction edge stays a junction edge on both sides. Returns the new
    /// vertex.
    pub fn split(&mut self, u: u32, v: u32) -> Option<u32> {
        let faces = self.edge_faces(u, v);
        if faces.is_empty() || faces.iter().any(|&f| self.face_locked(f)) {
            return None;
        }
        let (pu, pv) = (self.voxel[u as usize], self.voxel[v as usize]);
        let m = self.voxel.len() as u32;
        self.voxel
            .push(std::array::from_fn(|k| 0.5 * (pu[k] + pv[k])));
        self.incident.push(Vec::new());
        self.vertex_alive.push(true);
        self.locked.push(false);
        self.suspect.push(false);
        self.pinned.push(false);
        self.cell.push(self.cell[u as usize]);
        self.pinch.push(false);

        for f in faces {
            let t = self.faces[f as usize];
            // Rotate so the edge runs t[k] -> t[k + 1].
            let k = (0..3)
                .find(|&k| {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    (a == u && b == v) || (a == v && b == u)
                })
                .expect("face holds the edge");
            let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
            let g = self.faces.len() as u32;
            self.faces[f as usize] = [a, m, c];
            self.faces.push([m, b, c]);
            self.front.push(self.front[f as usize]);
            self.back.push(self.back[f as usize]);
            self.face_alive.push(true);
            let inc = &mut self.incident[b as usize];
            let slot = inc.iter().position(|&x| x == f).expect("b was on f");
            inc[slot] = g;
            self.incident[c as usize].push(g);
            self.incident[m as usize].push(f);
            self.incident[m as usize].push(g);
        }
        Some(m)
    }

    // --- flip --------------------------------------------------------------

    /// Swap wall edge `(u, v)` for the other diagonal of its two faces.
    pub fn flip(&mut self, u: u32, v: u32) -> bool {
        if self.edge_kind(u, v) != EdgeKind::Wall {
            return false;
        }
        let faces = self.edge_faces(u, v);
        let (fa, fb) = if self.runs(faces[0], u, v) {
            (faces[0], faces[1])
        } else {
            (faces[1], faces[0])
        };
        if self.face_locked(fa) || self.face_locked(fb) {
            return false;
        }
        let opposite = |f: u32| {
            self.faces[f as usize]
                .into_iter()
                .find(|&w| w != u && w != v)
                .unwrap()
        };
        let (p, q) = (opposite(fa), opposite(fb));
        if p == q || self.neighbours(p).binary_search(&q).is_ok() {
            return false;
        }
        // fa = (u, v, p), fb = (v, u, q); after, (p, u, q) and (q, v, p).
        let at = |w: u32| self.physical(self.voxel[w as usize]);
        let (pu, pv, pp, pq) = (at(u), at(v), at(p), at(q));
        let old = {
            let a = cross(sub(pv, pu), sub(pp, pu));
            let b = cross(sub(pu, pv), sub(pq, pv));
            [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
        };
        let na = cross(sub(pu, pp), sub(pq, pp));
        let nb = cross(sub(pv, pq), sub(pp, pq));
        let tiny = MIN_AREA_RATIO * dot(old, old).sqrt();
        if dot(na, old) <= 0.0
            || dot(nb, old) <= 0.0
            || dot(na, na).sqrt() <= tiny
            || dot(nb, nb).sqrt() <= tiny
        {
            return false;
        }
        self.faces[fa as usize] = [p, u, q];
        self.faces[fb as usize] = [q, v, p];
        self.incident[u as usize].retain(|&f| f != fb);
        self.incident[v as usize].retain(|&f| f != fa);
        self.incident[p as usize].push(fb);
        self.incident[q as usize].push(fa);
        true
    }

    // --- relocate ----------------------------------------------------------

    /// Move `v` to `to` (voxel units), if no face around it would turn over or
    /// become degenerate. Direction is the caller's business: a policy keeps
    /// wall vertices in their wall and curve vertices on their curve.
    pub fn relocate(&mut self, v: u32, to: [f64; 3]) -> bool {
        match self.vertex_kind(v) {
            VertexKind::Wall | VertexKind::Curve => {}
            _ => return false,
        }
        if !self.keeps_orientation(v, to, &[]) {
            return false;
        }
        self.voxel[v as usize] = to;
        true
    }

    /// Whether moving `v` to `to` keeps every face around it (except those in
    /// `dying`) facing the same way and non-degenerate.
    fn keeps_orientation(&self, v: u32, to: [f64; 3], dying: &[u32]) -> bool {
        let here = self.voxel[v as usize];
        self.incident[v as usize].iter().all(|&f| {
            if dying.contains(&f) {
                return true;
            }
            let before = self.normal_with(f, v, here);
            let after = self.normal_with(f, v, to);
            dot(before, after) > 0.0
                && dot(after, after).sqrt() > MIN_AREA_RATIO * dot(before, before).sqrt()
        })
    }

    // --- collapse ----------------------------------------------------------

    /// Collapse `u` into `v`, leaving `v` at `to` (voxel units).
    ///
    /// Where `v` may not move -- a wall vertex collapsing onto a curve or a
    /// corner, a curve vertex onto a corner -- `to` is ignored and `v` stays
    /// put. Refused unless it keeps every label's surface manifold with the
    /// same topology; see the checks below.
    pub fn collapse(&mut self, u: u32, v: u32, to: [f64; 3]) -> bool {
        let (ku, kv) = (self.vertex_kind(u), self.vertex_kind(v));
        let to = match (ku, kv) {
            (VertexKind::Wall, VertexKind::Wall) => to,
            (VertexKind::Wall, VertexKind::Curve | VertexKind::Corner) => self.voxel[v as usize],
            (VertexKind::Curve, VertexKind::Curve)
                if self.edge_kind(u, v) == EdgeKind::Junction =>
            {
                to
            }
            (VertexKind::Curve, VertexKind::Corner)
                if self.edge_kind(u, v) == EdgeKind::Junction =>
            {
                self.voxel[v as usize]
            }
            _ => return false,
        };
        let dying = self.edge_faces(u, v);
        if dying.is_empty() {
            return false;
        }
        // Everything around `u` is rewired, so nothing there may touch a
        // locked vertex; `v` itself is known not to be locked.
        if self.incident[u as usize]
            .iter()
            .any(|&f| self.face_locked(f))
        {
            return false;
        }

        // --- the link condition, for the whole mesh and for every label ------
        // Collapsing (u, v) keeps the topology exactly when the vertices
        // joined to both are precisely those opposite the edge. Asked of the
        // whole mesh, and again of each label's surface alone: a vertex can be
        // joined to both through one label's faces while the face opposite it
        // on the edge belongs to another, and that label would come out
        // pinched.
        let opposite = |f: u32| {
            self.faces[f as usize]
                .into_iter()
                .find(|&w| w != u && w != v)
                .unwrap()
        };
        let mut opp_all: Vec<u32> = dying.iter().map(|&f| opposite(f)).collect();
        opp_all.sort_unstable();
        opp_all.dedup();
        let nu = self.neighbours(u);
        let nv = self.neighbours(v);
        let common: Vec<u32> = nu
            .iter()
            .copied()
            .filter(|w| *w != v && nv.binary_search(w).is_ok())
            .collect();
        if common != opp_all {
            return false;
        }
        let mut labels: FxHashSet<u32> = FxHashSet::default();
        for &f in self.incident[u as usize]
            .iter()
            .chain(&self.incident[v as usize])
        {
            labels.insert(self.front[f as usize]);
            if self.back[f as usize] != OUTSIDE {
                labels.insert(self.back[f as usize]);
            }
        }
        let has = |f: u32, l: u32| self.front[f as usize] == l || self.back[f as usize] == l;
        for &l in &labels {
            let around = |x: u32| {
                let mut n: Vec<u32> = self.incident[x as usize]
                    .iter()
                    .filter(|&&f| has(f, l))
                    .flat_map(|&f| self.faces[f as usize])
                    .filter(|&w| w != x)
                    .collect();
                n.sort_unstable();
                n.dedup();
                n
            };
            let (lu, lv) = (around(u), around(v));
            let common: Vec<u32> = lu
                .iter()
                .copied()
                .filter(|w| *w != v && lv.binary_search(w).is_ok())
                .collect();
            let mut opp: Vec<u32> = dying
                .iter()
                .filter(|&&f| has(f, l))
                .map(|&f| opposite(f))
                .collect();
            opp.sort_unstable();
            opp.dedup();
            if common != opp {
                return false;
            }
            // And a label with faces at both ends must have one on the edge
            // itself. Otherwise the edge belongs to other labels' walls, `u`
            // and `v` are two separate points of this label's surface, and
            // collapsing them glues it to itself -- which the link test above
            // cannot see, since both sides of it are then empty.
            if !lu.is_empty() && !lv.is_empty() && !dying.iter().any(|&f| has(f, l)) {
                return false;
            }
        }
        // A curve must not close up on itself: the curve neighbours of `u`
        // and `v`, other than each other, must differ.
        if ku == VertexKind::Curve {
            let (cu, cv) = (self.curve_neighbours(u), self.curve_neighbours(v));
            if cu.iter().any(|w| *w != v && cv.contains(w)) {
                return false;
            }
        }

        // No two faces may end up on the same three vertices. The link
        // condition alone still allows collapsing a tetrahedron into a pair of
        // coincident triangles wound opposite ways -- a closed surface of two
        // faces, which the next collapse deletes outright.
        let key = |t: [u32; 3]| {
            let mut t = t;
            t.sort_unstable();
            t
        };
        let mut after: Vec<[u32; 3]> = self.incident[u as usize]
            .iter()
            .chain(&self.incident[v as usize])
            .filter(|f| !dying.contains(f))
            .map(|&f| key(self.faces[f as usize].map(|w| if w == u { v } else { w })))
            .collect();
        after.sort_unstable();
        if after.windows(2).any(|w| w[0] == w[1]) {
            return false;
        }

        // --- geometry -------------------------------------------------------
        if !self.keeps_orientation(u, to, &dying) || !self.keeps_orientation(v, to, &dying) {
            return false;
        }

        // --- apply -----------------------------------------------------------
        for &f in &dying {
            self.face_alive[f as usize] = false;
            for w in self.faces[f as usize] {
                if w != u {
                    self.incident[w as usize].retain(|&g| g != f);
                }
            }
        }
        let moving = std::mem::take(&mut self.incident[u as usize]);
        for f in moving {
            if !self.face_alive[f as usize] {
                continue;
            }
            for slot in self.faces[f as usize].iter_mut() {
                if *slot == u {
                    *slot = v;
                }
            }
            self.incident[v as usize].push(f);
        }
        self.vertex_alive[u as usize] = false;
        self.voxel[v as usize] = to;
        true
    }

    // --- output --------------------------------------------------------------

    /// Label slot `slot`'s surface, as [`WallMesh::label_mesh`] gives it.
    pub fn label_mesh(&self, slot: u32, opts: &MeshOptions) -> TriangleMesh {
        let faces: Vec<[u32; 3]> = (0..self.faces.len())
            .filter(|&i| self.face_alive[i])
            .filter_map(|i| {
                let f = self.faces[i];
                let side = if self.front[i] == slot {
                    0
                } else if self.back[i] == slot {
                    1
                } else {
                    return None;
                };
                // Faces touching a pinch are locked, so they still have the
                // index their aliases were recorded under.
                let corner = |k: usize| {
                    let v = f[k];
                    if self.pinch[v as usize] {
                        let key = (3 * i as u64 + k as u64) * 2 + side;
                        self.alias.get(&key).copied().unwrap_or(v)
                    } else {
                        v
                    }
                };
                Some(if side == 0 {
                    [corner(0), corner(1), corner(2)]
                } else {
                    [corner(0), corner(2), corner(1)]
                })
            })
            .collect();
        if faces.is_empty() {
            return TriangleMesh::default();
        }
        finish(
            self.voxel.len(),
            |v| self.voxel[v as usize],
            &faces,
            |v| self.cell[v as usize],
            |v| self.suspect[v as usize],
            Some(&|v: u32| self.pinned[v as usize]),
            opts,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{extract, Extraction};
    use crate::grid::VolumeView;
    use ndarray::Array3;

    fn noisy(n: usize, labels: u32, seed: u64) -> Array3<u32> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut rand = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let mut a = Array3::<u32>::zeros((n, n, n));
        for l in 1..=labels {
            let c = [rand() * n as f64, rand() * n as f64, rand() * n as f64];
            let r = 3.0 + rand() * n as f64 / 4.0;
            for ((i, j, k), v) in a.indexed_iter_mut() {
                let d = (i as f64 - c[0]).powi(2)
                    + (j as f64 - c[1]).powi(2)
                    + (k as f64 - c[2]).powi(2);
                if d <= r * r {
                    *v = l;
                }
            }
        }
        for v in a.iter_mut() {
            if rand() < 0.02 {
                *v = 1 + (rand() * labels as f64) as u32 % labels;
            }
        }
        a
    }

    /// Per label: Euler characteristic and open-edge count, after checking the
    /// surface is a proper 2-manifold -- every directed edge used once, every
    /// edge by at most two faces wound oppositely, and every vertex's faces a
    /// single fan.
    fn topology(m: &TriangleMesh) -> (i64, usize) {
        let mut directed: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        for f in &m.faces {
            assert!(
                f[0] != f[1] && f[1] != f[2] && f[0] != f[2],
                "degenerate face {f:?}"
            );
            for k in 0..3 {
                *directed.entry((f[k], f[(k + 1) % 3])).or_default() += 1;
            }
        }
        assert!(
            directed.values().all(|&c| c == 1),
            "a directed edge is used twice"
        );
        let mut undirected: FxHashSet<(u32, u32)> = FxHashSet::default();
        let mut open = 0;
        for &(a, b) in directed.keys() {
            undirected.insert((a.min(b), a.max(b)));
            if !directed.contains_key(&(b, a)) {
                open += 1;
            }
        }
        // Each vertex's link: its faces' opposite edges must form one path or
        // cycle. Except on the open rim: a surface cut by the volume's edge can
        // touch that edge at a single vertex, which `get()` has always produced
        // for open volumes and stitching resolves. Rim vertices are locked, so
        // remeshing cannot change them either way.
        let mut on_rim: FxHashSet<u32> = FxHashSet::default();
        for &(a, b) in directed.keys() {
            if !directed.contains_key(&(b, a)) {
                on_rim.insert(a);
                on_rim.insert(b);
            }
        }
        let mut link: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        for f in &m.faces {
            for k in 0..3 {
                link.entry(f[k])
                    .or_default()
                    .push((f[(k + 1) % 3], f[(k + 2) % 3]));
            }
        }
        for (v, edges) in &link {
            if on_rim.contains(v) {
                continue;
            }
            let mut next: FxHashMap<u32, u32> = FxHashMap::default();
            for &(a, b) in edges {
                next.insert(a, b);
            }
            // Walk from a start with no predecessor if there is one.
            let targets: FxHashSet<u32> = next.values().copied().collect();
            let start = next
                .keys()
                .copied()
                .find(|k| !targets.contains(k))
                .unwrap_or(edges[0].0);
            let (mut at, mut seen) = (start, 0);
            while let Some(&n) = next.get(&at) {
                seen += 1;
                at = n;
                if at == start || seen > edges.len() {
                    break;
                }
            }
            assert_eq!(seen, edges.len(), "vertex {v} is non-manifold");
        }
        let chi = m.vertices.len() as i64 - undirected.len() as i64 + m.faces.len() as i64;
        (chi, open)
    }

    fn setup(n: usize, labels: u32, seed: u64, close: bool) -> (Extraction, Remesh, MeshOptions) {
        let a = noisy(n, labels, seed);
        let e = extract(&VolumeView::new(a.view(), close));
        let walls = WallMesh::build(&e).unwrap();
        let rm = Remesh::new(&walls, &e.cells, [1.0, 1.0, 1.0]);
        let opts = MeshOptions {
            shape: [n, n, n],
            ..Default::default()
        };
        (e, rm, opts)
    }

    #[test]
    fn untouched_it_is_the_wall_mesh() {
        let a = noisy(20, 6, 3);
        let e = extract(&VolumeView::new(a.view(), true));
        let walls = WallMesh::build(&e).unwrap();
        let rm = Remesh::new(&walls, &e.cells, [1.0, 1.0, 1.0]);
        let opts = MeshOptions {
            shape: [20, 20, 20],
            ..Default::default()
        };
        for s in 0..e.meshes.len() as u32 {
            let a = walls.label_mesh(s, &e.cells, &opts);
            let b = rm.label_mesh(s, &opts);
            assert_eq!(a.vertices, b.vertices);
            assert_eq!(a.faces, b.faces);
        }
    }

    #[test]
    fn it_finds_walls_curves_and_corners() {
        let (_, rm, _) = setup(24, 6, 1, true);
        let mut count = [0usize; 4];
        for v in 0..rm.vertex_count() as u32 {
            if rm.incident[v as usize].is_empty() {
                continue;
            }
            count[rm.vertex_kind(v) as usize] += 1;
        }
        let [locked, wall, curve, corner] = count;
        assert!(wall > curve && curve > corner && corner > 0, "{count:?}");
        assert!(locked > 0);
        // Every curve vertex has exactly two curve neighbours, by definition.
        for v in 0..rm.vertex_count() as u32 {
            if !rm.incident[v as usize].is_empty() && rm.vertex_kind(v) == VertexKind::Curve {
                assert_eq!(rm.curve_neighbours(v).len(), 2);
            }
        }
    }

    /// Random operations, thousands of them, and after every batch every
    /// label's surface must still be a closed 2-manifold of unchanged
    /// topology. The shared walls cannot disagree, since there is one of each.
    fn fuzz(seed: u64, close: bool, collapse_heavy: bool) -> [usize; 4] {
        let (e, mut rm, opts) = setup(24, 6, seed, close);
        let slots = e.meshes.len() as u32;
        let before: Vec<(i64, usize)> = (0..slots)
            .map(|s| topology(&rm.label_mesh(s, &opts)))
            .collect();
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut rand = move |n: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as usize) % n.max(1)
        };
        let mut applied = [0usize; 4];
        for round in 0..20 {
            for _ in 0..400 {
                let f = rand(rm.face_count()) as u32;
                let Some(t) = rm.face(f) else { continue };
                let k = rand(3);
                let (u, v) = (t[k], t[(k + 1) % 3]);
                let mid: [f64; 3] = {
                    let (a, b) = (rm.position(u), rm.position(v));
                    std::array::from_fn(|i| 0.5 * (a[i] + b[i]))
                };
                // Mostly collapses, in the heavy variant, to drive components
                // down to the small configurations the checks exist for.
                let op = if collapse_heavy && rand(10) < 7 {
                    1
                } else {
                    rand(4)
                };
                let ok = match op {
                    0 => rm.split(u, v).is_some(),
                    1 => rm.collapse(u, v, mid),
                    2 => rm.flip(u, v),
                    _ => {
                        let p = rm.position(u);
                        let d = [
                            rand(100) as f64 / 500.0 - 0.1,
                            rand(100) as f64 / 500.0 - 0.1,
                            rand(100) as f64 / 500.0 - 0.1,
                        ];
                        rm.relocate(u, std::array::from_fn(|i| p[i] + d[i]))
                    }
                };
                if ok {
                    applied[op] += 1;
                }
            }
            for s in 0..slots {
                let after = topology(&rm.label_mesh(s, &opts));
                assert_eq!(after, before[s as usize], "label slot {s}, round {round}");
            }
        }
        applied
    }

    /// The same, collapsing most of the time: small objects shrink to the
    /// fewest faces their topology allows, which is where a tetrahedron once
    /// collapsed into two coincident triangles.
    #[test]
    fn collapsing_to_the_minimum_keeps_every_label_manifold() {
        let seeds: u64 = std::env::var("SERRA_FUZZ_SEEDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        for seed in 0..seeds {
            let applied = fuzz(1000 + seed, seed % 2 == 1, true);
            assert!(applied[1] > 1000, "seed {seed}: {applied:?}");
        }
    }

    /// Four seeds by default; set `SERRA_FUZZ_SEEDS` for a longer run.
    #[test]
    fn random_operations_keep_every_label_manifold() {
        let seeds: u64 = std::env::var("SERRA_FUZZ_SEEDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        for seed in 0..seeds {
            let applied = fuzz(seed, seed % 2 == 0, false);
            if std::env::var_os("SERRA_FUZZ_SEEDS").is_some() {
                eprintln!("seed {seed}: split/collapse/flip/relocate applied {applied:?}");
            }
            // Each kind must actually have happened a good number of times,
            // or the test proves nothing.
            assert!(applied.iter().all(|&n| n > 50), "seed {seed}: {applied:?}");
        }
    }
}
