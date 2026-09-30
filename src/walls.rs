//! Every wall between two labels, stored once.
//!
//! The extractor builds each label's surface separately, so a wall between two
//! touching objects exists twice: once in each label's mesh, wound in opposite
//! directions. Anything that edits a wall -- simplification, remeshing -- has
//! to edit both copies identically, or the objects stop meeting exactly. The
//! simplest way to guarantee that is to have only one copy.
//!
//! A [`WallMesh`] is that single copy: the union of every label's surface, with
//! each triangle stored once and tagged with the label on each side of it
//! ([`WallMesh::front`], which it is wound outward for, and
//! [`WallMesh::back`]). Vertices are shared between labels too. Where three or
//! more labels meet, an edge simply has three or more triangles; the mesh is
//! not a 2-manifold, but each label's part of it is, and
//! [`WallMesh::label_mesh`] recovers exactly the surface [`crate::mesh::build`]
//! would have produced.
//!
//! # How it is built
//!
//! From an [`Extraction`], without touching the volume again. Each voxel edge
//! between two labels produced one quad in each label's mesh, both from the
//! same four cells, so matching quads on their cells pairs the two copies of
//! every wall. Merging the paired quads' vertices, cell by cell, gives the
//! shared vertices, and one triangulated copy of each quad gives the faces --
//! split along the same diagonal both labels chose, so nothing changes shape.
//!
//! # Where merging goes further than one label would
//!
//! A shared vertex is a group of per-label vertices joined through walls. That
//! can join two separate sheets of one label: label `a` at two corners at
//! opposite ends of a cell's body diagonal has two sheets there, and a label
//! `b` lying along an edge between them touches both, so `b`'s single vertex
//! ties them together. The cell has no ambiguous face, so the per-label mesh
//! would never have needed repair. Such groups are marked [`WallMesh::pinch`],
//! and each of the label's members gets an alias vertex at the same place;
//! [`WallMesh::alias`] records which one each face corner means to that label.
//! [`WallMesh::label_mesh`] therefore sees exactly the label's own vertices,
//! and hands the result to the same manifold repair as always. Rewiring the
//! label's vertices through the group instead made that repair decide
//! differently, since its choices at an ambiguous vertex depend on the
//! identities of the vertices around it. Remeshing must never move a pinch.
//!
//! Positions must agree between the copies for any of this to make sense. They
//! always do without smoothing and with cell-domain `fairing`, which moves one
//! position per cell; per-label relaxation and Taubin smoothing move the copies
//! apart, and building from them is refused.

use rustc_hash::FxHashMap;

use crate::extract::{CellField, Extraction};
use crate::mesh::{finish, fixed_to_voxel, split_along_first_diagonal, MeshOptions, TriangleMesh};
use crate::tables::AMBIGUOUS_CELL;

/// The label slot recorded on the far side of a wall facing background.
pub const OUTSIDE: u32 = u32::MAX;

/// The union of every label's surface, each wall stored once.
#[derive(Default, Clone)]
pub struct WallMesh {
    /// Vertex positions in 1/256-voxel units, as in [`crate::extract::LabelMesh`].
    pub positions: Vec<[i32; 3]>,
    /// The cell each vertex came from.
    pub cells: Vec<u32>,
    /// Vertices that join two sheets of one label; see the module docs.
    pub pinch: Vec<bool>,
    /// Triangles, wound outward for their [`front`](Self::front) label.
    pub faces: Vec<[u32; 3]>,
    /// Per face: the label slot (index into [`Extraction::labels`]) the face is
    /// wound outward for.
    pub front: Vec<u32>,
    /// Per face: the label slot on the other side, or [`OUTSIDE`].
    pub back: Vec<u32>,
    /// Where a label has several vertices in one merged group, which of them a
    /// face corner means to that label: keyed by `(3 * face + corner) * 2 +
    /// side`, side 0 the front label and 1 the back, giving an alias vertex.
    /// Aliases sit after the group's own vertices in `positions`, at the same
    /// place, so a label's surface keeps exactly its own vertex identities.
    pub alias: FxHashMap<u64, u32>,
    /// `(vertex, label slot)` pairs, ascending: the vertices each label's own
    /// mesh would hand to the manifold repair. Kept per label because the
    /// repair splits a vertex for one label and not another.
    pub label_suspect: Vec<(u32, u32)>,
}

/// Why a [`WallMesh`] could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WallError {
    /// Two labels' copies of a wall are not at the same place, because they
    /// were smoothed separately.
    NotCoincident { labels: (u64, u64) },
}

impl std::fmt::Display for WallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WallError::NotCoincident { labels } => write!(
                f,
                "the walls between labels {} and {} do not coincide; build from an \
                 unsmoothed or cell-domain (fairing) extraction",
                labels.0, labels.1
            ),
        }
    }
}

impl std::error::Error for WallError {}

/// Disjoint sets over `u32` ids, with path halving.
struct Sets(Vec<u32>);

impl Sets {
    fn new(n: usize) -> Self {
        Sets((0..n as u32).collect())
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            let up = self.0[self.0[x as usize] as usize];
            self.0[x as usize] = up;
            x = up;
        }
        x
    }

    /// Join the sets of `a` and `b`, keeping the smaller root so the result
    /// does not depend on the order of calls.
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.0[hi as usize] = lo;
        }
    }
}

impl WallMesh {
    /// Build from an extraction. Fails if two labels' copies of a wall are not
    /// at the same place; see the module docs.
    pub fn build(e: &Extraction) -> Result<WallMesh, WallError> {
        // Every label's vertices in one numbering: label `l`'s vertex `v` is
        // `offset[l] + v`.
        let mut offset = Vec::with_capacity(e.meshes.len() + 1);
        offset.push(0u32);
        for m in &e.meshes {
            offset.push(offset.last().unwrap() + m.positions.len() as u32);
        }
        let total = *offset.last().unwrap() as usize;

        // --- pair the two copies of each wall -------------------------------
        // A quad is identified by its four cells, which only the voxel edge it
        // is dual to has. Sorting on that brings the copies together.
        let mut quads: Vec<(u128, u32, u32)> = Vec::new();
        for (l, m) in e.meshes.iter().enumerate() {
            for (i, q) in m.quads.iter().enumerate() {
                let mut c = q.map(|v| m.cells[v as usize]);
                c.sort_unstable();
                let key = c.iter().fold(0u128, |k, &x| (k << 32) | x as u128);
                quads.push((key, l as u32, i as u32));
            }
        }
        quads.sort_unstable();

        // --- merge the vertices the copies share ----------------------------
        let mut sets = Sets::new(total);
        let mut i = 0;
        while i < quads.len() {
            let j = if i + 1 < quads.len() && quads[i + 1].0 == quads[i].0 {
                i + 2
            } else {
                i + 1
            };
            debug_assert!(
                j >= quads.len() || quads[j].0 != quads[i].0,
                "a voxel edge produced more than two quads"
            );
            if j == i + 2 {
                let (la, qa) = (quads[i].1 as usize, quads[i].2 as usize);
                let (lb, qb) = (quads[i + 1].1 as usize, quads[i + 1].2 as usize);
                let (ma, mb) = (&e.meshes[la], &e.meshes[lb]);
                for &va in &ma.quads[qa] {
                    let cell = ma.cells[va as usize];
                    let vb = *mb.quads[qb]
                        .iter()
                        .find(|&&vb| mb.cells[vb as usize] == cell)
                        .expect("paired quads share all four cells");
                    if ma.positions[va as usize] != mb.positions[vb as usize] {
                        return Err(WallError::NotCoincident {
                            labels: (e.labels[la], e.labels[lb]),
                        });
                    }
                    sets.union(offset[la] + va, offset[lb] + vb);
                }
            }
            i = j;
        }

        // --- number the shared vertices -------------------------------------
        // In order of first appearance, label by label, so the numbering is a
        // pure function of the extraction. Along the way, note any group that
        // takes two vertices from one label.
        let mut id = vec![u32::MAX; total];
        let mut walls = WallMesh::default();
        let mut last_label: Vec<u32> = Vec::new();
        for (l, m) in e.meshes.iter().enumerate() {
            for v in 0..m.positions.len() {
                let root = sets.find(offset[l] + v as u32) as usize;
                if id[root] == u32::MAX {
                    id[root] = walls.positions.len() as u32;
                    walls.positions.push(m.positions[v]);
                    walls.cells.push(m.cells[v]);
                    walls.pinch.push(false);
                    last_label.push(l as u32);
                } else {
                    let w = id[root] as usize;
                    if last_label[w] == l as u32 {
                        walls.pinch[w] = true;
                    }
                    last_label[w] = l as u32;
                }
            }
        }
        let shared = |sets: &mut Sets, l: usize, v: u32| id[sets.find(offset[l] + v) as usize];

        // --- aliases for labels with several members in one group ------------
        let mut members: Vec<(u32, u32, u32)> = Vec::new(); // (group, label, vertex)
        for (l, m) in e.meshes.iter().enumerate() {
            for v in 0..m.positions.len() as u32 {
                let w = shared(&mut sets, l, v);
                if walls.pinch[w as usize] {
                    members.push((w, l as u32, v));
                }
            }
        }
        members.sort_unstable();
        let mut member_alias: FxHashMap<u32, u32> = FxHashMap::default();
        let mut i = 0;
        while i < members.len() {
            let mut j = i + 1;
            while j < members.len() && members[j].0 == members[i].0 && members[j].1 == members[i].1
            {
                j += 1;
            }
            if j - i > 1 {
                for &(w, l, v) in &members[i..j] {
                    let alias = walls.positions.len() as u32;
                    walls.positions.push(walls.positions[w as usize]);
                    walls.cells.push(walls.cells[w as usize]);
                    walls.pinch.push(true);
                    member_alias.insert(offset[l as usize] + v, alias);
                }
            }
            i = j;
        }

        for (l, m) in e.meshes.iter().enumerate() {
            for &v in &m.suspects {
                walls
                    .label_suspect
                    .push((shared(&mut sets, l, v), l as u32));
            }
        }
        walls.label_suspect.sort_unstable();
        walls.label_suspect.dedup();

        // --- one triangulated copy of each wall -----------------------------
        let mut i = 0;
        while i < quads.len() {
            let paired = i + 1 < quads.len() && quads[i + 1].0 == quads[i].0;
            let (l, q) = (quads[i].1 as usize, quads[i].2 as usize);
            let m = &e.meshes[l];
            let ring = m.quads[q];
            let p = ring.map(|v| &m.positions[v as usize]);
            let tris = if split_along_first_diagonal(p) {
                [[ring[0], ring[1], ring[2]], [ring[0], ring[2], ring[3]]]
            } else {
                [[ring[1], ring[2], ring[3]], [ring[1], ring[3], ring[0]]]
            };
            let partner = paired.then(|| (quads[i + 1].1 as usize, quads[i + 1].2 as usize));
            let back = partner.map_or(OUTSIDE, |(lb, _)| lb as u32);
            for t in tris {
                let face = walls.faces.len() as u64;
                for (k, &v) in t.iter().enumerate() {
                    if let Some(&alias) = member_alias.get(&(offset[l] + v)) {
                        walls.alias.insert((3 * face + k as u64) * 2, alias);
                    }
                    if let Some((lb, qb)) = partner {
                        let mb = &e.meshes[lb];
                        let cell = m.cells[v as usize];
                        let vb = *mb.quads[qb]
                            .iter()
                            .find(|&&vb| mb.cells[vb as usize] == cell)
                            .expect("paired quads share all four cells");
                        if let Some(&alias) = member_alias.get(&(offset[lb] + vb)) {
                            walls.alias.insert((3 * face + k as u64) * 2 + 1, alias);
                        }
                    }
                }
                walls.faces.push(t.map(|v| shared(&mut sets, l, v)));
                walls.front.push(l as u32);
                walls.back.push(back);
            }
            i += if paired { 2 } else { 1 };
        }
        Ok(walls)
    }

    /// Label slot `slot`'s surface, as [`crate::mesh::build`] would give it.
    ///
    /// The same triangles and the same vertex positions; the order of both may
    /// differ, since faces here are in wall order rather than the label's own.
    pub fn label_mesh(&self, slot: u32, cells: &CellField, opts: &MeshOptions) -> TriangleMesh {
        let faces: Vec<[u32; 3]> = self
            .faces
            .iter()
            .zip(self.front.iter().zip(&self.back))
            .enumerate()
            .filter_map(|(i, (f, (&front, &back)))| {
                let side = if front == slot {
                    0
                } else if back == slot {
                    1
                } else {
                    return None;
                };
                // Rare, so looked up only at a pinch.
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
        let suspect = |v: u32| {
            self.pinch[v as usize]
                || AMBIGUOUS_CELL[cells.crossings[self.cells[v as usize] as usize] as usize]
        };
        let pinned = |v: u32| cells.pinned[self.cells[v as usize] as usize];
        let cell_of = |v: u32| self.cells[v as usize];
        finish(
            self.positions.len(),
            |v| fixed_to_voxel(self.positions[v as usize]),
            &faces,
            cell_of,
            suspect,
            Some(&pinned),
            opts,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::extract;
    use crate::grid::VolumeView;
    use crate::mesh::build;
    use ndarray::Array3;

    /// Each triangle as its three corner positions, sorted, so two meshes can
    /// be compared whatever order they store things in.
    fn triangles(m: &TriangleMesh) -> Vec<[[u32; 3]; 3]> {
        let mut out: Vec<[[u32; 3]; 3]> = m
            .faces
            .iter()
            .map(|f| {
                let mut t = f.map(|v| m.vertices[v as usize].map(f32::to_bits));
                t.sort_unstable();
                t
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Winding survives the comparison above only through orientation, so check
    /// it separately: every face's normal, by sorted corner triple.
    fn signed_triangles(m: &TriangleMesh) -> Vec<([[u32; 3]; 3], bool)> {
        let mut out: Vec<([[u32; 3]; 3], bool)> = m
            .faces
            .iter()
            .map(|f| {
                let c = f.map(|v| m.vertices[v as usize]);
                let mut order = [0usize, 1, 2];
                order.sort_by_key(|&k| c[k].map(f32::to_bits));
                // An even permutation keeps the winding.
                let even = matches!(order, [0, 1, 2] | [1, 2, 0] | [2, 0, 1]);
                let t = order.map(|k| c[k].map(f32::to_bits));
                (t, even)
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Several overlapping blobs with voxel-scale noise, so there are junctions,
    /// thin pieces, ambiguous cells and multi-sheet cells to get right.
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
            let r = 2.0 + rand() * n as f64 / 4.0;
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
            if rand() < 0.08 {
                *v = 1 + (rand() * labels as f64) as u32 % labels;
            }
        }
        a
    }

    fn check_matches_build(a: &Array3<u32>, close: bool) {
        let e = extract(&VolumeView::new(a.view(), close));
        let walls = WallMesh::build(&e).expect("unsmoothed walls coincide");
        let opts = MeshOptions {
            shape: [a.shape()[0], a.shape()[1], a.shape()[2]],
            ..Default::default()
        };
        for (slot, raw) in e.meshes.iter().enumerate() {
            let want = build(raw, &opts);
            let got = walls.label_mesh(slot as u32, &e.cells, &opts);
            assert_eq!(
                got.vertices.len(),
                want.vertices.len(),
                "label {}",
                e.labels[slot]
            );
            assert_eq!(
                signed_triangles(&got),
                signed_triangles(&want),
                "label {}",
                e.labels[slot]
            );
        }
    }

    #[test]
    fn every_label_comes_back_exactly() {
        for seed in 0..12 {
            let (n, labels) = [(16, 4), (24, 8), (32, 16)][seed as usize % 3];
            check_matches_build(&noisy(n, labels, seed), true);
            check_matches_build(&noisy(n, labels, seed), false);
        }
    }

    /// Cell-domain fairing moves one position per cell, so the copies still
    /// coincide and the walls can still be built -- and still come back exact.
    #[test]
    fn faired_extractions_come_back_exactly_too() {
        use crate::smooth::{fair, scatter, Fairing};
        for seed in 0..4 {
            let a = noisy(24, 8, 100 + seed);
            let mut e = extract(&VolumeView::new(a.view(), true));
            let params = Fairing {
                iterations: 10,
                pass_band: Some(0.1),
                tangential: 3,
                ..Default::default()
            };
            fair(&mut e.cells, &params, false);
            for m in e.meshes.iter_mut() {
                scatter(&e.cells, m);
            }
            let walls = WallMesh::build(&e).expect("faired walls coincide");
            let opts = MeshOptions {
                shape: [24, 24, 24],
                ..Default::default()
            };
            for (slot, raw) in e.meshes.iter().enumerate() {
                let want = build(raw, &opts);
                let got = walls.label_mesh(slot as u32, &e.cells, &opts);
                assert_eq!(got.vertices.len(), want.vertices.len());
                assert_eq!(signed_triangles(&got), signed_triangles(&want));
            }
        }
    }

    /// The configuration that needs the pinch flag: label 1 at the two ends of
    /// a body diagonal, label 2 along an edge touching both, the rest label 3.
    #[test]
    fn a_label_joined_through_its_neighbour_is_split_back_apart() {
        let mut a = Array3::<u32>::from_elem((4, 4, 4), 3);
        // Corner index c = x + 2y + 4z within the cell at (1, 1, 1).
        let at = |c: usize| (1 + (c & 1), 1 + ((c >> 1) & 1), 1 + ((c >> 2) & 1));
        for (c, l) in [(3, 1), (4, 1), (6, 2), (7, 2)] {
            let (i, j, k) = at(c);
            a[[i, j, k]] = l;
        }
        let e = extract(&VolumeView::new(a.view(), true));
        let walls = WallMesh::build(&e).unwrap();
        assert!(
            walls.pinch.iter().any(|&p| p),
            "fixture no longer exercises the pinch"
        );
        check_matches_build(&a, true);
    }

    #[test]
    fn each_wall_is_stored_once() {
        let a = noisy(24, 8, 7);
        let e = extract(&VolumeView::new(a.view(), true));
        let walls = WallMesh::build(&e).unwrap();
        let per_label: usize = e.meshes.iter().map(|m| 2 * m.quads.len()).sum();
        let shared = walls.back.iter().filter(|&&b| b != OUTSIDE).count();
        // A face with a label behind it stands for two per-label faces.
        assert_eq!(walls.faces.len() + shared, per_label);
        assert!(shared > 0);
        // And the triangles are the same ones.
        let opts = MeshOptions {
            shape: [24, 24, 24],
            ..Default::default()
        };
        let mut all: Vec<[[u32; 3]; 3]> = (0..e.meshes.len() as u32)
            .flat_map(|s| triangles(&walls.label_mesh(s, &e.cells, &opts)))
            .collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), walls.faces.len());
    }

    #[test]
    fn separately_smoothed_walls_are_refused() {
        let a = noisy(16, 4, 1);
        let mut e = extract(&VolumeView::new(a.view(), true));
        // Nudge one label's copy of everything, as per-label smoothing would.
        for p in e.meshes[0].positions.iter_mut() {
            p[0] += 1;
        }
        assert!(matches!(
            WallMesh::build(&e),
            Err(WallError::NotCoincident { .. })
        ));
    }
}
