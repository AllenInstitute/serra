//! Where a cell's dual vertex goes.
//!
//! Placement is deliberately *local*: a cell's vertex position depends on
//! nothing but that cell's own eight corner labels. That is what makes chunked
//! meshing exact — two neighbouring chunks that both contain a seam cell run the
//! identical table lookup on the identical corners and get the identical
//! integer, with no dependence on what else is in either chunk.
//!
//! The optional relaxation pass lives here too. It smooths the interior while
//! holding the outermost layer of cells fixed, so it costs no extra halo: a
//! chunk's mesh stays reproducible from that chunk's own array however many
//! iterations run. The price is that a chunk's interior is smoothed slightly
//! more than the band around its seams, so a stitched surface is not identical
//! to one meshed in a single piece — only self-consistent and watertight.

use crate::extract::LabelMesh;
use crate::grid::Label;
use crate::tables::{CENTROID, EDGES, SUBVOXEL};

/// Which of the 12 cube edges have differing labels at their two ends.
///
/// Bit `e` corresponds to `tables::EDGES[e]`.
#[inline]
pub fn crossing_mask<T: Label>(corners: &[T; 8]) -> u16 {
    let mut mask = 0u16;
    for (e, &(a, b)) in EDGES.iter().enumerate() {
        if corners[a as usize] != corners[b as usize] {
            mask |= 1 << e;
        }
    }
    mask
}

/// The cell's dual vertex, in 1/256-voxel fixed point, absolute in
/// sample-index space.
///
/// `origin` is the cell's minimum corner. The whole computation is integer, so
/// the result is exactly reproducible.
#[inline]
pub fn cell_vertex(origin: [isize; 3], crossings: u16) -> [i32; 3] {
    let offset = &CENTROID[crossings as usize];
    [
        origin[0] as i32 * SUBVOXEL + offset[0],
        origin[1] as i32 * SUBVOXEL + offset[1],
        origin[2] as i32 * SUBVOXEL + offset[2],
    ]
}

/// Settings for the optional relaxation pass.
#[derive(Clone, Copy, Debug)]
pub struct Relaxation {
    /// Number of smoothing iterations. Zero disables the pass.
    pub iterations: u32,
    /// How far a vertex may drift from where local placement put it, in voxels.
    ///
    /// This is what bounds the surface's deviation from the data, and what stops
    /// Laplacian smoothing from shrinking an object away.
    pub max_deviation: f64,
    /// Fraction of the way to the neighbour average each iteration, in (0, 1].
    pub step: f64,
}

impl Default for Relaxation {
    fn default() -> Self {
        Relaxation {
            iterations: 0,
            max_deviation: 0.5,
            step: 0.5,
        }
    }
}

/// Smooth a label's surface, holding pinned vertices fixed.
///
/// Iterations are Jacobi, not Gauss-Seidel: every vertex is updated from the
/// previous iteration's positions, so the result cannot depend on the order
/// vertices happen to be visited in. That is what keeps the output identical
/// across thread counts and platforms.
///
/// Pinned vertices never move. When [`crate::extract::ExtractOptions::
/// mark_boundary`] marked the outermost cell layer, this makes the whole pass
/// depend only on data the chunk already holds — so a chunk's mesh is
/// reproducible from that chunk alone, whatever `iterations` is set to, and the
/// halo stays at one voxel.
pub fn relax(mesh: &mut LabelMesh, params: &Relaxation) {
    if params.iterations == 0 || mesh.quads.is_empty() {
        return;
    }
    let n = mesh.positions.len();
    let adjacency = Adjacency::of(mesh, n);

    let start: Vec<[f64; 3]> = mesh
        .positions
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let mut current = start.clone();
    let mut next = start.clone();

    let limit = params.max_deviation * SUBVOXEL as f64;
    let pinned = |v: usize| mesh.pinned.get(v).copied().unwrap_or(false);

    for _ in 0..params.iterations {
        for v in 0..n {
            let neighbours = adjacency.of_vertex(v);
            if pinned(v) || neighbours.is_empty() {
                next[v] = current[v];
                continue;
            }
            let mut sum = [0.0f64; 3];
            for &w in neighbours {
                let p = current[w as usize];
                for k in 0..3 {
                    sum[k] += p[k];
                }
            }
            let count = neighbours.len() as f64;
            for k in 0..3 {
                let average = sum[k] / count;
                let moved = current[v][k] + params.step * (average - current[v][k]);
                // Clamp against the *original* position, so the bound is on
                // total deviation from the data rather than per step.
                next[v][k] = moved.clamp(start[v][k] - limit, start[v][k] + limit);
            }
        }
        std::mem::swap(&mut current, &mut next);
    }

    for (slot, relaxed) in mesh.positions.iter_mut().zip(current.iter()) {
        for k in 0..3 {
            slot[k] = relaxed[k].round() as i32;
        }
    }
}

/// Vertex neighbours in CSR form, built from the quad ring edges.
struct Adjacency {
    offsets: Vec<u32>,
    neighbours: Vec<u32>,
}

impl Adjacency {
    fn of(mesh: &LabelMesh, n: usize) -> Self {
        // Each quad contributes its four ring edges, in both directions. An
        // interior edge is shared by two quads and so appears twice; the
        // duplicates are removed rather than left to double-weight a neighbour,
        // which would otherwise bias vertices along an open boundary.
        let mut degree = vec![0u32; n + 1];
        for q in &mesh.quads {
            for i in 0..4 {
                degree[q[i] as usize + 1] += 2;
            }
        }
        for i in 1..=n {
            degree[i] += degree[i - 1];
        }
        let offsets = degree;
        let mut cursor = offsets.clone();
        let mut neighbours = vec![0u32; *offsets.last().unwrap() as usize];
        for q in &mesh.quads {
            for i in 0..4 {
                let a = q[i];
                let b = q[(i + 1) % 4];
                for (from, to) in [(a, b), (b, a)] {
                    let slot = &mut cursor[from as usize];
                    neighbours[*slot as usize] = to;
                    *slot += 1;
                }
            }
        }

        // Compact each vertex's list in place.
        let mut compacted = Vec::with_capacity(neighbours.len());
        let mut new_offsets = Vec::with_capacity(n + 1);
        new_offsets.push(0u32);
        for v in 0..n {
            let span = &mut neighbours[offsets[v] as usize..offsets[v + 1] as usize];
            span.sort_unstable();
            let mut last: Option<u32> = None;
            for &w in span.iter() {
                if w as usize != v && last != Some(w) {
                    compacted.push(w);
                    last = Some(w);
                }
            }
            new_offsets.push(compacted.len() as u32);
        }

        Adjacency {
            offsets: new_offsets,
            neighbours: compacted,
        }
    }

    #[inline]
    fn of_vertex(&self, v: usize) -> &[u32] {
        &self.neighbours[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{extract_with, ExtractOptions};
    use crate::grid::VolumeView;
    use ndarray::Array3;

    /// A tube spanning the array along axis 0, with the outermost cell layer
    /// marked as pinned.
    ///
    /// The object deliberately runs off both ends, which is what a chunk cut
    /// through the middle of a structure looks like. A shape floating clear of
    /// the array edge would leave the outermost cells uniform, so they would
    /// contribute no vertices and nothing would be pinned at all.
    fn marked_tube(radius: f64) -> LabelMesh {
        let n = (2.0 * radius) as usize + 8;
        let c = (n as f64 - 1.0) / 2.0;
        let a = Array3::from_shape_fn((n, n, n), |(_, j, k)| {
            let d = (j as f64 - c).powi(2) + (k as f64 - c).powi(2);
            u32::from(d <= radius * radius)
        });
        let opts = ExtractOptions {
            mark_boundary: true,
        };
        let mut e = extract_with(&VolumeView::new(a.view(), false), &opts);
        e.meshes.remove(0)
    }

    #[test]
    fn pinned_vertices_never_move() {
        let before = marked_tube(9.0);
        let mut after = before.clone();
        relax(
            &mut after,
            &Relaxation {
                iterations: 40,
                max_deviation: 0.5,
                step: 0.6,
            },
        );
        let mut pinned_seen = 0;
        for (i, &is_pinned) in before.pinned.iter().enumerate() {
            if is_pinned {
                pinned_seen += 1;
                assert_eq!(
                    before.positions[i], after.positions[i],
                    "pinned vertex {i} moved"
                );
            }
        }
        assert!(pinned_seen > 0, "the fixture produced no pinned vertices");
    }

    #[test]
    fn unpinned_vertices_do_move() {
        let before = marked_tube(9.0);
        let mut after = before.clone();
        relax(
            &mut after,
            &Relaxation {
                iterations: 10,
                max_deviation: 0.5,
                step: 0.5,
            },
        );
        let moved = before
            .positions
            .iter()
            .zip(after.positions.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(moved > 0, "relaxation did nothing");
    }

    #[test]
    fn no_vertex_exceeds_the_deviation_bound() {
        let before = marked_tube(9.0);
        for &max_deviation in &[0.0f64, 0.125, 0.5] {
            let mut after = before.clone();
            relax(
                &mut after,
                &Relaxation {
                    iterations: 100,
                    max_deviation,
                    step: 1.0,
                },
            );
            // One unit of slack for the final rounding back to fixed point.
            let limit = (max_deviation * SUBVOXEL as f64).round() as i32 + 1;
            for (i, (a, b)) in before
                .positions
                .iter()
                .zip(after.positions.iter())
                .enumerate()
            {
                for k in 0..3 {
                    assert!(
                        (a[k] - b[k]).abs() <= limit,
                        "vertex {i} axis {k} moved {} units, limit {limit}",
                        (a[k] - b[k]).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn zero_iterations_is_a_no_op() {
        let before = marked_tube(6.0);
        let mut after = before.clone();
        relax(&mut after, &Relaxation::default());
        assert_eq!(before.positions, after.positions);
    }

    #[test]
    fn relaxation_is_order_independent() {
        // Jacobi iteration reads only the previous state, so running it twice
        // must give the same answer bit for bit.
        let base = marked_tube(7.0);
        let params = Relaxation {
            iterations: 7,
            max_deviation: 0.4,
            step: 0.5,
        };
        let mut first = base.clone();
        let mut second = base.clone();
        relax(&mut first, &params);
        relax(&mut second, &params);
        assert_eq!(first.positions, second.positions);
    }

    #[test]
    fn a_uniform_cell_has_no_crossings() {
        assert_eq!(crossing_mask(&[7u32; 8]), 0);
    }

    #[test]
    fn a_wall_perpendicular_to_x_crosses_the_four_x_edges() {
        // Corners with bit 0 clear are one label, the rest another.
        let mut corners = [0u32; 8];
        for (c, slot) in corners.iter_mut().enumerate() {
            *slot = if c & 1 == 0 { 1 } else { 2 };
        }
        let mask = crossing_mask(&corners);
        assert_eq!(mask.count_ones(), 4);
        // Which puts the vertex on the cell's mid-plane in x, centred in y and z.
        let v = cell_vertex([0, 0, 0], mask);
        assert_eq!(v, [SUBVOXEL / 2, SUBVOXEL / 2, SUBVOXEL / 2]);
    }

    #[test]
    fn position_is_offset_by_the_cell_origin() {
        let mut corners = [1u32; 8];
        corners[7] = 2;
        let mask = crossing_mask(&corners);
        let at_origin = cell_vertex([0, 0, 0], mask);
        let shifted = cell_vertex([3, -1, 5], mask);
        assert_eq!(shifted[0], at_origin[0] + 3 * SUBVOXEL);
        assert_eq!(shifted[1], at_origin[1] - SUBVOXEL);
        assert_eq!(shifted[2], at_origin[2] + 5 * SUBVOXEL);
    }

    /// A single differing corner is cut off by three edges, so its vertex sits
    /// near that corner rather than at the cell centre.
    #[test]
    fn one_odd_corner_pulls_the_vertex_toward_it() {
        let mut corners = [1u32; 8];
        corners[0] = 2; // corner (0,0,0)
        let v = cell_vertex([0, 0, 0], crossing_mask(&corners));
        for k in 0..3 {
            assert!(v[k] < SUBVOXEL / 2, "axis {k} should be pulled low: {v:?}");
        }
    }
}
