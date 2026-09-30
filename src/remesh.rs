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
//! * a **fixed** vertex is one that different labels see as different numbers
//!   of vertices (below). It never moves and nothing collapses into it, but
//!   the triangles around it may change;
//! * a **locked** vertex keeps every triangle around it exactly as it is: the
//!   chunk's outer cell layer, so neighbouring chunks still agree about
//!   everything near their seam, and any label's open rim at the volume's
//!   boundary.
//!
//! # One point, several vertices
//!
//! At a cell with an ambiguous face, one label's triangles can form two fans
//! meeting at a point, and extraction's manifold repair gives that label two
//! vertices there. Where the labels agree, [`Remesh::new`] makes the split
//! itself, once, in the shared mesh. Where they do not -- one label needs two
//! vertices where another, whose single fan touches both, needs one -- no
//! split can serve both. The shared vertex is then kept, and each extra fan of
//! the label that needs it gets an *alias*: a vertex at the same place, which
//! that label's triangles there refer to instead. [`WallMesh::pinch`] groups
//! are the same situation, found while building, and handled the same way.
//!
//! So every label's identities are explicit, the repair never has to run on a
//! remeshed surface, and a vertex with aliases is merely fixed rather than
//! frozen with everything around it. Aliases are keyed by face, vertex and
//! side, so they survive operations that reorder a triangle's corners, and
//! each operation carries them onto the triangles it creates.
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
    Fixed,
    Wall,
    Curve,
    Corner,
}

/// Which side of a face a label is on: 0 the front, 1 the back.
type Side = u8;

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
    /// No triangle around a locked vertex may change.
    locked: Vec<bool>,
    /// A fixed vertex never moves and nothing collapses into it.
    fixed: Vec<bool>,
    pinned: Vec<bool>,
    /// What label `side` of `face` means by `vertex`, where that is not
    /// `vertex` itself: `(face, vertex, side) -> alias`.
    alias: FxHashMap<(u32, u32, Side), u32>,
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

/// Disjoint sets over small local indices.
struct Sets(Vec<usize>);

impl Sets {
    fn new(n: usize) -> Self {
        Sets((0..n).collect())
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a.max(b)] = a.min(b);
        }
    }
}

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
        // Re-key the pinch aliases by vertex instead of corner.
        let mut alias = FxHashMap::default();
        for (&key, &a) in &walls.alias {
            let side = (key % 2) as Side;
            let (face, corner) = ((key / 2) / 3, (key / 2) % 3);
            let v = walls.faces[face as usize][corner as usize];
            alias.insert((face as u32, v, side), a);
        }
        let mut rm = Remesh {
            voxel: walls.positions.iter().map(|&p| fixed_to_voxel(p)).collect(),
            scale: resolution,
            faces: walls.faces.clone(),
            front: walls.front.clone(),
            back: walls.back.clone(),
            face_alive: vec![true; walls.faces.len()],
            incident,
            vertex_alive: vec![true; n],
            locked: Vec::new(),
            fixed: walls.pinch.clone(),
            pinned: (0..n)
                .map(|v| cells.pinned[walls.cells[v] as usize])
                .collect(),
            alias,
        };
        rm.resolve(&walls.label_suspect, &walls.cells);
        rm.lock();
        rm
    }

    /// What label `side` of face `f` means by vertex `v`.
    #[inline]
    fn identity(&self, f: u32, v: u32, side: Side) -> u32 {
        if self.fixed[v as usize] {
            self.alias.get(&(f, v, side)).copied().unwrap_or(v)
        } else {
            v
        }
    }

    /// Which side of face `f` label `l` is on, if either.
    #[inline]
    fn side_of(&self, f: u32, l: u32) -> Option<Side> {
        if self.front[f as usize] == l {
            Some(0)
        } else if self.back[f as usize] == l {
            Some(1)
        } else {
            None
        }
    }

    /// A vertex at the same place as `v`, with no faces of its own, for a label
    /// that needs another vertex there.
    fn new_alias(&mut self, v: u32) -> u32 {
        let a = self.voxel.len() as u32;
        self.voxel.push(self.voxel[v as usize]);
        self.incident.push(Vec::new());
        self.vertex_alive.push(true);
        self.fixed.push(true);
        self.pinned.push(self.pinned[v as usize]);
        a
    }

    /// Do each label's manifold repair once, up front; see the module docs.
    ///
    /// In order of cell, as the repair goes: splitting one vertex changes the
    /// edges its neighbours see, and fans are found from each label's own view
    /// of its neighbours, so each label's sequence of splits is the one its own
    /// repair would make.
    fn resolve(&mut self, label_suspect: &[(u32, u32)], cells: &[u32]) {
        let mut order: Vec<u32> = label_suspect.iter().map(|&(v, _)| v).collect();
        order.sort_unstable_by_key(|&v| (cells[v as usize], v));
        order.dedup();
        for s in order {
            let lo = label_suspect.partition_point(|&(v, _)| v < s);
            let hi = label_suspect.partition_point(|&(v, _)| v <= s);
            let suspects: Vec<u32> = label_suspect[lo..hi].iter().map(|&(_, l)| l).collect();
            self.resolve_one(s, &suspects);
        }
    }

    fn resolve_one(&mut self, s: u32, suspect_labels: &[u32]) {
        let faces = self.incident[s as usize].clone();
        if faces.is_empty() {
            return;
        }
        let mut labels: Vec<u32> = faces
            .iter()
            .flat_map(|&f| [self.front[f as usize], self.back[f as usize]])
            .filter(|&l| l != OUTSIDE)
            .collect();
        labels.sort_unstable();
        labels.dedup();

        // Each label's fans at `s`, as lists of local face indices. Faces the
        // label already sees through different identities of `s` are never in
        // one fan; for a label its repair would split here, fans are further
        // divided the repair's way, joined only across an edge exactly two of
        // the label's faces use.
        let mut merged = Sets::new(faces.len());
        let mut fans_of: Vec<(u32, Vec<Vec<usize>>)> = Vec::with_capacity(labels.len());
        for &l in &labels {
            let mine: Vec<(usize, Side)> = (0..faces.len())
                .filter_map(|i| self.side_of(faces[i], l).map(|side| (i, side)))
                .collect();
            let mut fan = Sets::new(faces.len());
            // Same identity of `s`: provisionally one fan...
            let mut by_identity: FxHashMap<u32, usize> = FxHashMap::default();
            for &(i, side) in &mine {
                let id = self.identity(faces[i], s, side);
                let first = *by_identity.entry(id).or_insert(i);
                if !suspect_labels.contains(&l) {
                    fan.union(first, i);
                }
            }
            if suspect_labels.contains(&l) {
                // ...divided by the repair's rule.
                let mut users: FxHashMap<(u32, u32), Vec<usize>> = FxHashMap::default();
                for &(i, side) in &mine {
                    let f = faces[i];
                    let id = self.identity(f, s, side);
                    for w in self.faces[f as usize] {
                        if w != s {
                            users
                                .entry((id, self.identity(f, w, side)))
                                .or_default()
                                .push(i);
                        }
                    }
                }
                for list in users.values() {
                    if list.len() == 2 {
                        fan.union(list[0], list[1]);
                    }
                }
            }
            let mut groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
            for &(i, _) in &mine {
                groups.entry(fan.find(i)).or_default().push(i);
            }
            let mut groups: Vec<Vec<usize>> = groups.into_values().collect();
            groups.sort_unstable();
            for g in &groups {
                for &i in &g[1..] {
                    merged.union(g[0], i);
                }
            }
            fans_of.push((l, groups));
        }

        let consistent = fans_of.iter().all(|(_, groups)| {
            let mut roots: Vec<usize> = groups.iter().map(|g| merged.find(g[0])).collect();
            roots.sort_unstable();
            roots.dedup();
            roots.len() == groups.len()
        });
        let has_alias = self.fixed[s as usize];

        if consistent && !has_alias {
            // Every label agrees: one real vertex per merged group.
            let first = merged.find(0);
            let mut copy_of: FxHashMap<usize, u32> = FxHashMap::default();
            for (i, &f) in faces.iter().enumerate() {
                let root = merged.find(i);
                if root == first {
                    continue;
                }
                let target = *copy_of.entry(root).or_insert_with(|| {
                    let v = self.voxel.len() as u32;
                    self.voxel.push(self.voxel[s as usize]);
                    self.incident.push(Vec::new());
                    self.vertex_alive.push(true);
                    self.fixed.push(false);
                    self.pinned.push(self.pinned[s as usize]);
                    v
                });
                for slot in self.faces[f as usize].iter_mut() {
                    if *slot == s {
                        *slot = target;
                    }
                }
                self.incident[s as usize].retain(|&g| g != f);
                self.incident[target as usize].push(f);
            }
            return;
        }

        // They do not: keep `s`, and give each label's extra fans aliases. A
        // fan keeps the identity its first face already has, unless an earlier
        // fan of the same label has claimed it.
        for (l, groups) in fans_of {
            let mut claimed: FxHashSet<u32> = FxHashSet::default();
            for g in groups {
                let f0 = faces[g[0]];
                let side = self.side_of(f0, l).unwrap();
                let current = self.identity(f0, s, side);
                let id = if claimed.insert(current) {
                    current
                } else {
                    let a = self.new_alias(s);
                    claimed.insert(a);
                    a
                };
                for &i in &g {
                    let f = faces[i];
                    let side = self.side_of(f, l).unwrap();
                    if id == s {
                        self.alias.remove(&(f, s, side));
                    } else {
                        self.alias.insert((f, s, side), id);
                    }
                }
            }
        }
        self.fixed[s as usize] = true;
    }

    /// Work out which vertices are locked; see the module docs.
    fn lock(&mut self) {
        let n = self.voxel.len();
        // Vertices on any label's open rim: an edge with only one of that
        // label's faces, counted with the label's own identities. Asked per
        // label, not of the whole mesh: where a label's surface ends at the
        // volume's edge, other labels' walls can still meet along the same
        // edge, and the whole mesh shows no rim there.
        let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(self.faces.len() * 6);
        for (f, t) in self.faces.iter().enumerate() {
            if !self.face_alive[f] {
                continue;
            }
            for (side, label) in [(0, self.front[f]), (1, self.back[f])] {
                if label == OUTSIDE {
                    continue;
                }
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    let (ia, ib) = (
                        self.identity(f as u32, a, side),
                        self.identity(f as u32, b, side),
                    );
                    edges.push((label, ia.min(ib), ia.max(ib)));
                }
            }
        }
        edges.sort_unstable();
        let mut rim = vec![false; n];
        // Identities map back to their shared vertex through the faces; mark
        // both the identity and, below, the vertices of faces touching it.
        let mut i = 0;
        while i < edges.len() {
            let mut j = i + 1;
            while j < edges.len() && edges[j] == edges[i] {
                j += 1;
            }
            if j - i == 1 {
                rim[edges[i].1 as usize] = true;
                rim[edges[i].2 as usize] = true;
            }
            i = j;
        }
        // An alias on the rim locks the shared vertex it stands for.
        for (&(_, v, _), &a) in &self.alias {
            if rim[a as usize] {
                rim[v as usize] = true;
            }
        }
        self.locked = (0..n).map(|v| self.pinned[v] || rim[v]).collect();
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

    /// Whether any live face uses `v`.
    pub fn is_used(&self, v: u32) -> bool {
        !self.incident[v as usize].is_empty()
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
        if self.fixed[v as usize] {
            return VertexKind::Fixed;
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

    /// Copy every alias `face` holds for `v` onto `onto`.
    fn copy_aliases(&mut self, face: u32, v: u32, onto: u32) {
        for side in [0, 1] {
            if let Some(&a) = self.alias.get(&(face, v, side)) {
                self.alias.insert((onto, v, side), a);
            }
        }
    }

    fn drop_aliases(&mut self, face: u32, v: u32) {
        for side in [0, 1] {
            self.alias.remove(&(face, v, side));
        }
    }

    // --- split -------------------------------------------------------------

    /// Split edge `(u, v)` at its midpoint. Every face on the edge is split,
    /// so a junction edge stays a junction edge on both sides. Returns the new
    /// vertex.
    pub fn split(&mut self, u: u32, v: u32) -> Option<u32> {
        // An edge ending at a fixed vertex may be several edges to a label that
        // sees that vertex as several; one new vertex on all of them would join
        // that label's fans through it.
        if self.fixed[u as usize] || self.fixed[v as usize] {
            return None;
        }
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
        self.fixed.push(false);
        self.pinned.push(false);

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
            // `b` moves from f to g; `c` is on both.
            self.copy_aliases(f, b, g);
            self.drop_aliases(f, b);
            self.copy_aliases(f, c, g);
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
        // fa loses v and gains q; fb loses u and gains p. The two faces are one
        // wall, so a corner means the same to each label on either.
        self.drop_aliases(fa, v);
        self.copy_aliases(fb, q, fa);
        self.drop_aliases(fb, u);
        self.copy_aliases(fa, p, fb);
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
        // whole mesh, and again of each label's surface alone, in that label's
        // own identities: a vertex can be joined to both through one label's
        // faces while the face opposite it on the edge belongs to another, and
        // that label would come out pinched.
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
        for &l in &labels {
            // `u` and `v` are neither fixed nor aliased, so only their
            // neighbours' identities can differ from the shared ids.
            let around = |x: u32| {
                let mut n: Vec<u32> = Vec::new();
                for &f in &self.incident[x as usize] {
                    if let Some(side) = self.side_of(f, l) {
                        for w in self.faces[f as usize] {
                            if w != x {
                                n.push(self.identity(f, w, side));
                            }
                        }
                    }
                }
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
                .filter_map(|&f| {
                    self.side_of(f, l)
                        .map(|side| self.identity(f, opposite(f), side))
                })
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
            if !lu.is_empty()
                && !lv.is_empty()
                && !dying.iter().any(|&f| self.side_of(f, l).is_some())
            {
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
                self.drop_aliases(f, w);
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
    ///
    /// Every label's identities are explicit, so the manifold repair never runs
    /// here.
    pub fn label_mesh(&self, slot: u32, opts: &MeshOptions) -> TriangleMesh {
        let faces: Vec<[u32; 3]> = (0..self.faces.len())
            .filter(|&i| self.face_alive[i])
            .filter_map(|i| {
                let side = self.side_of(i as u32, slot)?;
                let f = self.faces[i].map(|v| self.identity(i as u32, v, side));
                Some(if side == 0 { f } else { [f[0], f[2], f[1]] })
            })
            .collect();
        if faces.is_empty() {
            return TriangleMesh::default();
        }
        finish(
            self.voxel.len(),
            |v| self.voxel[v as usize],
            &faces,
            |_| 0,
            |_| false,
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

    /// Each face as its sorted corner positions and whether sorting kept the
    /// winding, so meshes can be compared whatever order they store things in.
    fn signed_triangles(m: &TriangleMesh) -> Vec<([[u32; 3]; 3], bool)> {
        let mut out: Vec<([[u32; 3]; 3], bool)> = m
            .faces
            .iter()
            .map(|f| {
                let c = f.map(|v| m.vertices[v as usize]);
                let mut order = [0usize, 1, 2];
                order.sort_by_key(|&k| c[k].map(f32::to_bits));
                let even = matches!(order, [0, 1, 2] | [1, 2, 0] | [2, 0, 1]);
                (order.map(|k| c[k].map(f32::to_bits)), even)
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Untouched, the editable mesh gives back exactly what `get()` does --
    /// including at the ambiguous-cell vertices it split up front, which the
    /// manifold repair no longer sees.
    #[test]
    fn untouched_it_is_what_get_returns() {
        for seed in 0..10 {
            let (n, labels) = [(16, 4), (24, 8), (32, 12)][seed as usize % 3];
            for close in [true, false] {
                let a = noisy(n, labels, seed);
                let e = extract(&VolumeView::new(a.view(), close));
                let walls = WallMesh::build(&e).unwrap();
                let rm = Remesh::new(&walls, &e.cells, [1.0, 1.0, 1.0]);
                let opts = MeshOptions {
                    shape: [n, n, n],
                    ..Default::default()
                };
                for (slot, raw) in e.meshes.iter().enumerate() {
                    let want = crate::mesh::build(raw, &opts);
                    let got = rm.label_mesh(slot as u32, &opts);
                    assert_eq!(
                        got.vertices.len(),
                        want.vertices.len(),
                        "seed {seed} slot {slot}"
                    );
                    assert_eq!(
                        signed_triangles(&got),
                        signed_triangles(&want),
                        "seed {seed} slot {slot}"
                    );
                }
            }
        }
    }

    /// Every ambiguous-cell vertex is resolved, by a split or by aliases, and
    /// only a small share of vertices end up fixed.
    #[test]
    fn only_a_few_vertices_are_fixed() {
        let a = noisy(32, 12, 5);
        let e = extract(&VolumeView::new(a.view(), true));
        let walls = WallMesh::build(&e).unwrap();
        let rm = Remesh::new(&walls, &e.cells, [1.0, 1.0, 1.0]);
        let used = (0..rm.vertex_count() as u32)
            .filter(|&v| rm.is_used(v))
            .count();
        let fixed = (0..rm.vertex_count() as u32)
            .filter(|&v| rm.is_used(v) && rm.vertex_kind(v) == VertexKind::Fixed)
            .count();
        assert!(fixed > 0, "fixture has nothing to fix");
        assert!(fixed * 50 < used, "{fixed} of {used} fixed");
    }

    #[test]
    fn it_finds_walls_curves_and_corners() {
        let (_, rm, _) = setup(24, 6, 1, true);
        let mut count = [0usize; 5];
        for v in 0..rm.vertex_count() as u32 {
            if !rm.is_used(v) {
                continue;
            }
            count[rm.vertex_kind(v) as usize] += 1;
        }
        let [locked, _fixed, wall, curve, corner] = count;
        assert!(wall > curve && curve > corner && corner > 0, "{count:?}");
        // A closed volume has no seam and no open rim to lock.
        assert_eq!(locked, 0, "{count:?}");
        // Every curve vertex has exactly two curve neighbours, by definition.
        for v in 0..rm.vertex_count() as u32 {
            if rm.is_used(v) && rm.vertex_kind(v) == VertexKind::Curve {
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
