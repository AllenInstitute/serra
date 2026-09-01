//! Where a cell's dual vertex goes.
//!
//! Placement is deliberately *local*: a cell's vertex position depends on
//! nothing but that cell's own eight corner labels. That is what makes chunked
//! meshing exact — two neighbouring chunks that both contain a seam cell run the
//! identical table lookup on the identical corners and get the identical
//! integer, with no dependence on what else is in either chunk.
//!
//! Whatever smoothing runs afterwards lives in [`crate::smooth`].

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

#[cfg(test)]
mod tests {
    use super::*;

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
