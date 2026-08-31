//! The single pass over the volume.
//!
//! # Structure
//!
//! A *cell* is a 2x2x2 block of voxels; its dual vertex is the surface point
//! contributed by that cell. A *quad* is dual to a voxel edge whose two voxels
//! carry different labels, and is built from the four cells surrounding that
//! edge.
//!
//! Every cell is visited exactly once, no matter how many labels the volume
//! contains — the cost is driven by boundary area, not by label count. On a
//! real 512^3 connectomics volume with 2524 labels only 16.7% of cells sit on a
//! boundary, and 84.8% of those carry just two labels, so the early-out on
//! uniform cells does most of the work.
//!
//! # Why one traversal suffices
//!
//! The four cells around the voxel edge running from voxel `p` along axis `a`
//! all have minimum corners at `p - du*e_u - dv*e_v` for `du, dv` in {0,1}.
//! None of them exceeds `p`, so by the time the loop reaches the cell at `p`
//! every cell it needs has already been built. Only two slabs of cell data have
//! to stay live: the current one and the one before it.
//!
//! # Manifoldness
//!
//! A cell contributes one vertex per 6-connected component of each label's
//! corners (see [`crate::tables`]). Recording, per cell, which vertex each
//! corner belongs to makes the quad step a direct lookup: the corner index of
//! `p` inside each surrounding cell is known from `(du, dv, a)`, and that
//! corner necessarily carries `p`'s label.

use rustc_hash::FxHashMap;

use crate::grid::{Label, VolumeView};
use crate::place::{cell_vertex, crossing_mask};
use crate::tables::TABLES;

/// Sentinel for "this corner contributes no vertex" (a background corner).
const NO_VERTEX: u32 = u32::MAX;

/// The four cells around a voxel edge, as offsets `(du, dv)` along the two axes
/// perpendicular to the edge.
///
/// This cycle is what orients the quad: traversed in this order, its normal
/// points along `+a`, which is out of the object occupying the lower voxel.
const RING: [(usize, usize); 4] = [(0, 0), (1, 0), (1, 1), (0, 1)];

/// One label's raw surface, still in fixed-point index space.
#[derive(Default, Clone)]
pub struct LabelMesh {
    /// Vertex positions in 1/256-voxel units, absolute in sample-index space.
    pub positions: Vec<[i32; 3]>,
    /// Quads as indices into `positions`, wound so normals point outward.
    pub quads: Vec<[u32; 4]>,
}

/// Everything a pass over the volume produced.
pub struct Extraction {
    /// Label ids, ascending. Sorted so output never depends on hash order —
    /// the defect that makes zmesh's results differ across platforms.
    pub labels: Vec<u64>,
    /// Per-label surfaces, parallel to `labels`.
    pub meshes: Vec<LabelMesh>,
}

impl Extraction {
    /// Index of `label`, if it was present in the volume.
    pub fn position_of(&self, label: u64) -> Option<usize> {
        self.labels.binary_search(&label).ok()
    }

    /// The surface for `label`, if any.
    pub fn mesh(&self, label: u64) -> Option<&LabelMesh> {
        self.position_of(label).map(|i| &self.meshes[i])
    }

    /// Drop one label's surface, freeing its memory. Returns whether it was
    /// there to begin with.
    pub fn remove(&mut self, label: u64) -> bool {
        match self.position_of(label) {
            Some(i) => {
                self.labels.remove(i);
                self.meshes.remove(i);
                true
            }
            None => false,
        }
    }
}

/// Per-cell record for one slab: for each of the 8 corners, the vertex it
/// belongs to within its own label's vertex list.
type CornerVertices = [u32; 8];

/// One plane of cells.
struct Slab {
    /// Index into `data` for each cell in the plane, or `NO_VERTEX` when the
    /// cell is uniform and so contributes nothing.
    slot: Vec<u32>,
    data: Vec<CornerVertices>,
}

impl Slab {
    fn new(cells: usize) -> Self {
        Slab {
            slot: vec![NO_VERTEX; cells],
            data: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.slot.fill(NO_VERTEX);
        self.data.clear();
    }

    #[inline]
    fn get(&self, index: usize) -> Option<&CornerVertices> {
        let s = self.slot[index];
        if s == NO_VERTEX {
            None
        } else {
            Some(&self.data[s as usize])
        }
    }
}

#[inline]
fn all_equal<T: PartialEq>(c: &[T; 8]) -> bool {
    c[1] == c[0]
        && c[2] == c[0]
        && c[3] == c[0]
        && c[4] == c[0]
        && c[5] == c[0]
        && c[6] == c[0]
        && c[7] == c[0]
}

/// Extract every label's surface in a single traversal.
pub fn extract<T: Label>(view: &VolumeView<T>) -> Extraction {
    let nc = view.cell_counts();
    let lo = view.sample_lo();
    if nc[0] == 0 || nc[1] == 0 || nc[2] == 0 {
        return Extraction {
            labels: Vec::new(),
            meshes: Vec::new(),
        };
    }
    let plane = nc[0] * nc[1];

    let mut label_index: FxHashMap<u64, u32> = FxHashMap::default();
    let mut meshes: Vec<LabelMesh> = Vec::new();

    // Slabs are planes of cells spanning array axes 0 and 1, stepped along
    // axis 2. Only the current and previous plane are ever needed.
    let mut prev = Slab::new(plane);
    let mut cur = Slab::new(plane);

    for cz in 0..nc[2] {
        std::mem::swap(&mut prev, &mut cur);
        cur.reset();

        for cy in 0..nc[1] {
            for cx in 0..nc[0] {
                let origin = [
                    lo[0] + cx as isize,
                    lo[1] + cy as isize,
                    lo[2] + cz as isize,
                ];
                let corners = view.cell_corners(origin);
                if all_equal(&corners) {
                    continue;
                }

                // --- vertices ------------------------------------------------
                let position = cell_vertex(origin, crossing_mask(&corners));
                let mut corner_vertex: CornerVertices = [NO_VERTEX; 8];

                for c in 0..8 {
                    let label = corners[c];
                    if label == T::BACKGROUND || corner_vertex[c] != NO_VERTEX {
                        continue;
                    }
                    // All corners carrying this label, as an 8-bit mask.
                    let mut mask = 0usize;
                    for (k, &other) in corners.iter().enumerate() {
                        if other == label {
                            mask |= 1 << k;
                        }
                    }

                    let li = label_slot(&mut label_index, &mut meshes, label.as_u64());
                    let mesh = &mut meshes[li];
                    let base = mesh.positions.len() as u32;
                    // One vertex per connected component keeps this label's
                    // surface 2-manifold even where it touches itself only
                    // diagonally. All components share the cell's position.
                    for _ in 0..TABLES.ncomp[mask] {
                        mesh.positions.push(position);
                    }
                    let split = &TABLES.split[mask];
                    for k in 0..8 {
                        if mask & (1 << k) != 0 {
                            corner_vertex[k] = base + split[k] as u32;
                        }
                    }
                }

                let slot = cur.data.len() as u32;
                cur.data.push(corner_vertex);
                cur.slot[cy * nc[0] + cx] = slot;

                // --- quads ---------------------------------------------------
                // Only the three edges leaving the cell's own minimum corner are
                // considered here; every other voxel edge is owned by some other
                // cell, so each is emitted exactly once.
                for axis in 0..3 {
                    let far = 1usize << axis;
                    let lower = corners[0];
                    let upper = corners[far];
                    if lower == upper {
                        continue;
                    }
                    if lower == T::BACKGROUND && upper == T::BACKGROUND {
                        continue;
                    }

                    let u = (axis + 1) % 3;
                    let v = (axis + 2) % 3;
                    let cell = [cx, cy, cz];

                    let mut ring_lower = [NO_VERTEX; 4];
                    let mut ring_upper = [NO_VERTEX; 4];
                    let mut complete = true;

                    for (i, &(du, dv)) in RING.iter().enumerate() {
                        let mut nb = [cell[0] as isize, cell[1] as isize, cell[2] as isize];
                        nb[u] -= du as isize;
                        nb[v] -= dv as isize;
                        if nb[u] < 0 || nb[v] < 0 {
                            // The surrounding cell lies outside the volume, so
                            // the surface is left open here.
                            complete = false;
                            break;
                        }
                        let (nx, ny, nz) = (nb[0] as usize, nb[1] as usize, nb[2] as usize);
                        let flat = ny * nc[0] + nx;
                        let slab = if nz == cz { &cur } else { &prev };
                        let Some(entry) = slab.get(flat) else {
                            complete = false;
                            break;
                        };
                        // The voxel at this cell's own corner 0 sits at offset
                        // `du` along u and `dv` along v inside the neighbour.
                        let near_corner = (du << u) | (dv << v);
                        ring_lower[i] = entry[near_corner];
                        ring_upper[i] = entry[near_corner | far];
                    }
                    if !complete {
                        continue;
                    }

                    if lower != T::BACKGROUND {
                        let li = label_slot(&mut label_index, &mut meshes, lower.as_u64());
                        meshes[li].quads.push(ring_lower);
                    }
                    if upper != T::BACKGROUND {
                        // Seen from the upper voxel the outward direction is
                        // reversed, so the ring is traversed the other way.
                        let li = label_slot(&mut label_index, &mut meshes, upper.as_u64());
                        meshes[li].quads.push([
                            ring_upper[3],
                            ring_upper[2],
                            ring_upper[1],
                            ring_upper[0],
                        ]);
                    }
                }
            }
        }
    }

    finish(label_index, meshes)
}

#[inline]
fn label_slot(index: &mut FxHashMap<u64, u32>, meshes: &mut Vec<LabelMesh>, label: u64) -> usize {
    match index.get(&label) {
        Some(&i) => i as usize,
        None => {
            let i = meshes.len() as u32;
            index.insert(label, i);
            meshes.push(LabelMesh::default());
            i as usize
        }
    }
}

/// Reorder into ascending label order, so results never depend on hash order.
fn finish(index: FxHashMap<u64, u32>, meshes: Vec<LabelMesh>) -> Extraction {
    let mut pairs: Vec<(u64, u32)> = index.into_iter().collect();
    pairs.sort_unstable_by_key(|&(label, _)| label);

    let mut slots: Vec<Option<LabelMesh>> = meshes.into_iter().map(Some).collect();
    let labels = pairs.iter().map(|&(label, _)| label).collect();
    let ordered = pairs
        .iter()
        .map(|&(_, i)| slots[i as usize].take().expect("each label appears once"))
        .collect();

    Extraction {
        labels,
        meshes: ordered,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, ShapeBuilder};

    fn view_of(a: &Array3<u32>, close: bool) -> VolumeView<'_, u32> {
        VolumeView::new(a.view(), close)
    }

    /// Every undirected edge of a closed surface must be used exactly twice.
    fn boundary_edge_count(mesh: &LabelMesh) -> usize {
        let mut counts: FxHashMap<(u32, u32), i32> = FxHashMap::default();
        for q in &mesh.quads {
            for i in 0..4 {
                let (a, b) = (q[i], q[(i + 1) % 4]);
                let key = if a < b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts.values().filter(|&&c| c != 2).count()
    }

    #[test]
    fn an_empty_volume_produces_nothing() {
        let a = Array3::<u32>::zeros((5, 5, 5));
        let e = extract(&view_of(&a, true));
        assert!(e.labels.is_empty());
    }

    #[test]
    fn background_is_never_meshed() {
        let mut a = Array3::<u32>::zeros((5, 5, 5));
        a[[2, 2, 2]] = 9;
        let e = extract(&view_of(&a, true));
        assert_eq!(e.labels, vec![9]);
    }

    #[test]
    fn a_single_voxel_is_a_closed_cube() {
        let mut a = Array3::<u32>::zeros((5, 5, 5));
        a[[2, 2, 2]] = 1;
        let e = extract(&view_of(&a, true));
        let m = e.mesh(1).unwrap();
        // Eight surrounding cells each contribute one vertex; six faces.
        assert_eq!(m.positions.len(), 8);
        assert_eq!(m.quads.len(), 6);
        assert_eq!(boundary_edge_count(m), 0, "cube must be closed");
    }

    #[test]
    fn two_touching_labels_share_a_wall_and_both_stay_closed() {
        let mut a = Array3::<u32>::zeros((8, 5, 5));
        for i in 1..4 {
            for j in 1..4 {
                for k in 1..4 {
                    a[[i, j, k]] = 1;
                    a[[i + 3, j, k]] = 2;
                }
            }
        }
        let e = extract(&view_of(&a, true));
        assert_eq!(e.labels, vec![1, 2]);
        for label in [1u64, 2] {
            let m = e.mesh(label).unwrap();
            assert_eq!(boundary_edge_count(m), 0, "label {label} must be closed");
        }
        // The shared wall is meshed once for each side.
        assert_eq!(
            e.mesh(1).unwrap().quads.len(),
            e.mesh(2).unwrap().quads.len()
        );
    }

    /// Two cubes meeting at a single corner would give a non-manifold vertex if
    /// the cell vertex were not split per connected component.
    #[test]
    fn a_corner_contact_is_split_into_two_manifold_sheets() {
        // Two single voxels meeting only diagonally, at the cell spanning
        // voxels (1,1,1)..(2,2,2).
        let mut a = Array3::<u32>::zeros((7, 7, 7));
        a[[1, 1, 1]] = 1;
        a[[2, 2, 2]] = 1;

        let e = extract(&view_of(&a, true));
        let m = e.mesh(1).unwrap();
        assert_eq!(boundary_edge_count(m), 0, "both sheets must be closed");

        // Fifteen cells touch the two voxels, and the one they share carries
        // corners 0 and 7 — diagonal, so not 6-connected. Splitting it yields
        // two vertices there and 16 in total: two separate closed cubes. A
        // single shared vertex would give 15 and a non-manifold pinch point.
        assert_eq!(m.positions.len(), 16);
        assert_eq!(m.quads.len(), 12);
    }

    #[test]
    fn c_and_fortran_order_give_identical_output() {
        let mut c = Array3::<u32>::zeros((9, 8, 7));
        for i in 1..8 {
            for j in 1..7 {
                for k in 1..6 {
                    c[[i, j, k]] = ((i + j + k) % 3 + 1) as u32;
                }
            }
        }
        let mut f = Array3::<u32>::zeros((9, 8, 7).f());
        f.assign(&c);
        assert!(!f.is_standard_layout());

        let ec = extract(&view_of(&c, true));
        let ef = extract(&view_of(&f, true));
        assert_eq!(ec.labels, ef.labels);
        for (a, b) in ec.meshes.iter().zip(ef.meshes.iter()) {
            assert_eq!(a.positions, b.positions);
            assert_eq!(a.quads, b.quads);
        }
    }

    #[test]
    fn without_close_a_touching_object_is_left_open() {
        let mut a = Array3::<u32>::zeros((5, 5, 5));
        for j in 0..5 {
            for k in 0..5 {
                a[[0, j, k]] = 1;
                a[[1, j, k]] = 1;
            }
        }
        let open = extract(&view_of(&a, false));
        assert!(
            boundary_edge_count(open.mesh(1).unwrap()) > 0,
            "surface should be open at the volume face"
        );
        let closed = extract(&view_of(&a, true));
        assert_eq!(
            boundary_edge_count(closed.mesh(1).unwrap()),
            0,
            "close=true should seal it"
        );
    }

    #[test]
    fn labels_come_back_sorted() {
        let mut a = Array3::<u32>::zeros((9, 5, 5));
        let ids = [98_340_797u32, 7, 4_000_000_000, 12];
        for (n, &id) in ids.iter().enumerate() {
            a[[2 * n + 1, 2, 2]] = id;
        }
        let e = extract(&view_of(&a, true));
        assert_eq!(e.labels, vec![7, 12, 98_340_797, 4_000_000_000]);
    }
}
