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

use rayon::prelude::*;
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
    /// Parallel to `positions`: vertices that relaxation must not move.
    ///
    /// Empty unless [`ExtractOptions::mark_boundary`] was set. See that field
    /// for why these particular vertices are the ones to hold fixed.
    pub pinned: Vec<bool>,
}

/// How to run a pass.
#[derive(Clone, Copy, Default)]
pub struct ExtractOptions {
    /// Mark vertices in the outermost layer of cells as pinned.
    ///
    /// Those are exactly the vertices whose one-ring is incomplete: a cell in
    /// the last layer has neighbours in the layer beyond, which this chunk does
    /// not contain. Holding them fixed lets relaxation run on the interior
    /// without ever consulting data the chunk does not have, so a chunk's mesh
    /// stays a pure function of its own array however many iterations are used
    /// — the halo never has to grow past one voxel.
    ///
    /// Leave unset when meshing a whole volume with `close`, where the outermost
    /// cell layer is the sealing cap and has nothing beyond it to be missing.
    pub mark_boundary: bool,

    /// How many cells along each axis this chunk owns, counted from index 0.
    ///
    /// Set this when meshing one chunk of a larger volume. Cells beyond the
    /// owned range are halo: they are still built, because quads need them, but
    /// the quads *belonging* to them are left to the chunk that owns them. That
    /// is what stops two neighbouring chunks from both emitting the wall on
    /// their shared seam, which would leave duplicate faces after stitching.
    ///
    /// One halo cell per axis is required, so the array must extend two voxels
    /// past the owned region — see [`owns_quad`].
    ///
    /// `None` means the array is the whole volume and every quad is emitted.
    pub owned_cells: Option<[usize; 3]>,
}

/// Whether this chunk should emit the quad dual to the edge leaving `cell`
/// along `axis`.
///
/// The quad is built from four cells: the same index along `axis`, and one
/// lower or equal along the other two. Assigning it to the chunk owning the
/// *minimum* of those four gives every quad exactly one owner across a
/// decomposition, with no gaps and no duplicates.
///
/// The chunk therefore needs cells one past its owned range, hence two voxels
/// of halo. With only one voxel — enough for marching cubes, which reads a
/// single cell per quad — the four cells straddling a seam are split between
/// two chunks and neither can emit, leaving a hole.
#[inline]
fn owns_quad(owned: &Option<[usize; 3]>, cell: [usize; 3], axis: usize) -> bool {
    let Some(owned) = owned else {
        return true;
    };
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;
    let mut lowest = [cell[0] as isize, cell[1] as isize, cell[2] as isize];
    lowest[u] -= 1;
    lowest[v] -= 1;
    (0..3).all(|k| lowest[k] >= 0 && (lowest[k] as usize) < owned[k])
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
#[derive(Clone)]
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

/// Extract every label's surface in a single traversal, with default options.
pub fn extract<T: Label + Sync>(view: &VolumeView<T>) -> Extraction {
    extract_with(view, &ExtractOptions::default())
}

/// Extract every label's surface, single-threaded.
pub fn extract_with<T: Label + Sync>(view: &VolumeView<T>, opts: &ExtractOptions) -> Extraction {
    extract_parallel(view, opts, 1)
}

/// Smallest number of cell layers worth giving a worker. Below this the merge
/// and seam work costs more than the traversal it saves.
const MIN_BAND_DEPTH: usize = 8;

/// Extract every label's surface, optionally across several threads.
///
/// # How the work is split
///
/// The volume is cut into bands of cell layers along axis 2, one per worker.
/// A band builds its own vertices and quads with a private label table, so
/// workers never contend.
///
/// Quads are the awkward part: a quad owned by the cell at `cz` reads the cell
/// layer at `cz - 1`, which for the first layer of a band belongs to the
/// previous band. Rather than duplicating that layer, each band simply skips
/// its own first layer's quads, and a short serial pass afterwards emits them
/// using the two neighbouring bands' exported layers. That pass touches one
/// cell layer per seam — a few thousandths of the volume.
///
/// # Determinism
///
/// Bands are merged in band order and each band's labels are sorted before
/// merging, so a label's vertices always appear in the same sequence no matter
/// which worker finished first or how many workers there were. Output is
/// therefore byte-identical across thread counts.
pub fn extract_parallel<T: Label + Sync>(
    view: &VolumeView<T>,
    opts: &ExtractOptions,
    threads: usize,
) -> Extraction {
    let nc = view.cell_counts();
    let lo = view.sample_lo();
    if nc[0] == 0 || nc[1] == 0 || nc[2] == 0 {
        return Extraction {
            labels: Vec::new(),
            meshes: Vec::new(),
        };
    }

    let wanted = if threads == 0 {
        rayon::current_num_threads()
    } else {
        threads
    };
    let bands = wanted.clamp(1, nc[2].div_ceil(MIN_BAND_DEPTH).max(1));

    if bands <= 1 {
        let band = extract_band(view, opts, nc, lo, 0, nc[2], true);
        return finish_single(band);
    }

    let ranges = split_range(nc[2], bands);
    let outputs: Vec<Band> = ranges
        .par_iter()
        .map(|&(start, end)| extract_band(view, opts, nc, lo, start, end, start == 0))
        .collect();

    merge(view, nc, lo, outputs, &opts.owned_cells)
}

/// Divide `total` layers into `parts` contiguous ranges, largest first.
fn split_range(total: usize, parts: usize) -> Vec<(usize, usize)> {
    let base = total / parts;
    let extra = total % parts;
    let mut out = Vec::with_capacity(parts);
    let mut at = 0;
    for i in 0..parts {
        let depth = base + usize::from(i < extra);
        if depth == 0 {
            continue;
        }
        out.push((at, at + depth));
        at += depth;
    }
    out
}

/// One worker's share of the volume.
struct Band {
    /// Ascending label ids, parallel to `meshes`.
    labels: Vec<u64>,
    meshes: Vec<LabelMesh>,
    /// Cell layer at `start`, needed by the seam below this band.
    first: Slab,
    /// Cell layer at `end - 1`, needed by the seam above this band.
    last: Slab,
    start: usize,
}

fn extract_band<T: Label>(
    view: &VolumeView<T>,
    opts: &ExtractOptions,
    nc: [usize; 3],
    lo: [isize; 3],
    start: usize,
    end: usize,
    emit_first_layer_quads: bool,
) -> Band {
    let plane = nc[0] * nc[1];
    let mut label_index: FxHashMap<u64, u32> = FxHashMap::default();
    let mut meshes: Vec<LabelMesh> = Vec::new();

    // Only the current and previous cell layer are ever needed.
    let mut prev = Slab::new(plane);
    let mut cur = Slab::new(plane);
    let mut first = Slab::new(plane);
    let mut last = Slab::new(plane);

    for cz in start..end {
        std::mem::swap(&mut prev, &mut cur);
        cur.reset();
        // The first layer's quads read the layer below, which belongs to the
        // previous band, so they are left to the seam pass.
        let emit_quads = emit_first_layer_quads || cz > start;

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
                // Computed from the global cell grid, so banding cannot change
                // which cells count as the volume's boundary.
                let on_boundary = opts.mark_boundary
                    && (cx == 0
                        || cy == 0
                        || cz == 0
                        || cx + 1 == nc[0]
                        || cy + 1 == nc[1]
                        || cz + 1 == nc[2]);
                let position = cell_vertex(origin, crossing_mask(&corners));
                let mut corner_vertex: CornerVertices = [NO_VERTEX; 8];

                for c in 0..8 {
                    let label = corners[c];
                    if label == T::BACKGROUND || corner_vertex[c] != NO_VERTEX {
                        continue;
                    }
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
                        if opts.mark_boundary {
                            mesh.pinned.push(on_boundary);
                        }
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

                if !emit_quads {
                    continue;
                }

                // --- quads ---------------------------------------------------
                // Only the three edges leaving this cell's own minimum corner
                // are considered; every other voxel edge belongs to some other
                // cell, so each is emitted exactly once.
                for axis in 0..3 {
                    let far = 1usize << axis;
                    let lower = corners[0];
                    let upper = corners[far];
                    if lower == upper
                        || (lower == T::BACKGROUND && upper == T::BACKGROUND)
                        || !owns_quad(&opts.owned_cells, [cx, cy, cz], axis)
                    {
                        continue;
                    }

                    let u = (axis + 1) % 3;
                    let v = (axis + 2) % 3;
                    let cell = [cx as isize, cy as isize, cz as isize];

                    let mut ring_lower = [NO_VERTEX; 4];
                    let mut ring_upper = [NO_VERTEX; 4];
                    let mut complete = true;

                    for (i, &(du, dv)) in RING.iter().enumerate() {
                        let mut nb = cell;
                        nb[u] -= du as isize;
                        nb[v] -= dv as isize;
                        if nb[u] < 0 || nb[v] < 0 {
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

        if cz == start {
            first = cur.clone();
        }
        if cz + 1 == end {
            last = cur.clone();
        }
    }

    let (labels, meshes) = sort_by_label(label_index, meshes);
    Band {
        labels,
        meshes,
        first,
        last,
        start,
    }
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

/// Put labels in ascending order, so nothing downstream depends on hash order.
fn sort_by_label(index: FxHashMap<u64, u32>, meshes: Vec<LabelMesh>) -> (Vec<u64>, Vec<LabelMesh>) {
    let mut pairs: Vec<(u64, u32)> = index.into_iter().collect();
    pairs.sort_unstable_by_key(|&(label, _)| label);
    let mut slots: Vec<Option<LabelMesh>> = meshes.into_iter().map(Some).collect();
    let labels = pairs.iter().map(|&(label, _)| label).collect();
    let ordered = pairs
        .iter()
        .map(|&(_, i)| slots[i as usize].take().expect("each label appears once"))
        .collect();
    (labels, ordered)
}

fn finish_single(band: Band) -> Extraction {
    Extraction {
        labels: band.labels,
        meshes: band.meshes,
    }
}

/// Concatenate the bands and add the quads that straddle their seams.
///
/// The subtlety is ordering. A band cannot emit its own first layer's quads, so
/// those are produced separately — but simply appending them afterwards would
/// leave each label's faces in a different order than a single-threaded run
/// produces, and the output would no longer be byte-identical across thread
/// counts. Instead each seam's quads are bucketed by label and spliced in
/// between the bands they sit between, which reproduces exact cell order.
fn merge<T: Label + Sync>(
    view: &VolumeView<T>,
    nc: [usize; 3],
    lo: [isize; 3],
    mut bands: Vec<Band>,
    owned: &Option<[usize; 3]>,
) -> Extraction {
    let mut all: Vec<u64> = bands
        .iter()
        .flat_map(|b| b.labels.iter().copied())
        .collect();
    all.sort_unstable();
    all.dedup();
    let count = all.len();
    let global: FxHashMap<u64, usize> = all
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    let mut meshes: Vec<LabelMesh> = vec![LabelMesh::default(); count];

    // Vertices first: bands in order, so a label's vertices end up in the same
    // sequence a single traversal would have produced. Record where each band's
    // block starts so quad indices can be shifted onto it.
    let mut offsets: Vec<Vec<u32>> = Vec::with_capacity(bands.len());
    let mut local_of: Vec<Vec<u32>> = Vec::with_capacity(bands.len());
    for band in bands.iter_mut() {
        let mut per_label = Vec::with_capacity(band.labels.len());
        let mut lookup = vec![u32::MAX; count];
        for (local, &label) in band.labels.iter().enumerate() {
            let g = global[&label];
            lookup[g] = local as u32;
            let target = &mut meshes[g];
            per_label.push(target.positions.len() as u32);
            // Move rather than copy, and let the band's copy go as soon as it
            // has been appended. Only the quads are needed after this, so the
            // duplicated vertex data never exceeds one label's worth instead of
            // every band's at once.
            let positions = std::mem::take(&mut band.meshes[local].positions);
            target.positions.extend_from_slice(&positions);
            drop(positions);
            let pinned = std::mem::take(&mut band.meshes[local].pinned);
            target.pinned.extend_from_slice(&pinned);
        }
        offsets.push(per_label);
        local_of.push(lookup);
    }

    // Each seam's quads, bucketed by global label. Seams are independent.
    let seams: Vec<Vec<Vec<[u32; 4]>>> = (1..bands.len())
        .into_par_iter()
        .map(|b| seam_quads(view, nc, lo, &bands, &offsets, &global, count, b, owned))
        .collect();

    // Now the quads, in cell order: band 0, then seam 1, then band 1, and so on.
    for (g, mesh) in meshes.iter_mut().enumerate() {
        let mut total = 0usize;
        for b in 0..bands.len() {
            if b > 0 {
                total += seams[b - 1][g].len();
            }
            let local = local_of[b][g];
            if local != u32::MAX {
                total += bands[b].meshes[local as usize].quads.len();
            }
        }
        mesh.quads.reserve_exact(total);

        for b in 0..bands.len() {
            if b > 0 {
                mesh.quads.extend_from_slice(&seams[b - 1][g]);
            }
            let local = local_of[b][g];
            if local != u32::MAX {
                let base = offsets[b][local as usize];
                let source = std::mem::take(&mut bands[b].meshes[local as usize].quads);
                mesh.quads
                    .extend(source.into_iter().map(|q| q.map(|v| v + base)));
            }
        }
    }

    Extraction {
        labels: all,
        meshes,
    }
}

/// Quads on the first cell layer of band `b`, bucketed by global label index.
///
/// These are the only quads no band could produce alone, because they read the
/// cell layer below, which belongs to the band beneath. Both layers were
/// exported by their owners, so this just looks them up and shifts the
/// band-local vertex ids onto their merged positions.
#[allow(clippy::too_many_arguments)]
fn seam_quads<T: Label>(
    view: &VolumeView<T>,
    nc: [usize; 3],
    lo: [isize; 3],
    bands: &[Band],
    offsets: &[Vec<u32>],
    global: &FxHashMap<u64, usize>,
    count: usize,
    b: usize,
    owned: &Option<[usize; 3]>,
) -> Vec<Vec<[u32; 4]>> {
    let mut out: Vec<Vec<[u32; 4]>> = vec![Vec::new(); count];

    let base_of = |band: usize, label: u64| -> u32 {
        let local = bands[band]
            .labels
            .binary_search(&label)
            .expect("a cell's label must exist in the band that meshed it");
        offsets[band][local]
    };

    let cz = bands[b].start;
    let below = &bands[b - 1].last;
    let above = &bands[b].first;

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

            for axis in 0..3 {
                let far = 1usize << axis;
                let lower = corners[0];
                let upper = corners[far];
                if lower == upper
                    || (lower == T::BACKGROUND && upper == T::BACKGROUND)
                    || !owns_quad(owned, [cx, cy, cz], axis)
                {
                    continue;
                }

                let u = (axis + 1) % 3;
                let v = (axis + 2) % 3;
                let cell = [cx as isize, cy as isize, cz as isize];

                let mut ring_lower = [NO_VERTEX; 4];
                let mut ring_upper = [NO_VERTEX; 4];
                let mut complete = true;

                for (i, &(du, dv)) in RING.iter().enumerate() {
                    let mut nb = cell;
                    nb[u] -= du as isize;
                    nb[v] -= dv as isize;
                    if nb[u] < 0 || nb[v] < 0 {
                        complete = false;
                        break;
                    }
                    let (nx, ny, nz) = (nb[0] as usize, nb[1] as usize, nb[2] as usize);
                    let flat = ny * nc[0] + nx;
                    // Only two layers can be involved: this one, and the one
                    // below it which is the previous band's last.
                    let (slab, owner) = if nz == cz { (above, b) } else { (below, b - 1) };
                    let Some(entry) = slab.get(flat) else {
                        complete = false;
                        break;
                    };
                    let near_corner = (du << u) | (dv << v);
                    // Shift only labels that are actually meshed; a background
                    // corner holds no vertex to shift.
                    if lower != T::BACKGROUND {
                        ring_lower[i] = entry[near_corner] + base_of(owner, lower.as_u64());
                    }
                    if upper != T::BACKGROUND {
                        ring_upper[i] = entry[near_corner | far] + base_of(owner, upper.as_u64());
                    }
                }
                if !complete {
                    continue;
                }

                if lower != T::BACKGROUND {
                    out[global[&lower.as_u64()]].push(ring_lower);
                }
                if upper != T::BACKGROUND {
                    out[global[&upper.as_u64()]].push([
                        ring_upper[3],
                        ring_upper[2],
                        ring_upper[1],
                        ring_upper[0],
                    ]);
                }
            }
        }
    }
    out
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
