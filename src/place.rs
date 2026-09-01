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
use crate::tables::{CENTROID, EDGES, FACE_CORNERS, FACE_JUNCTION, FACE_KIND, NCORNERS, SUBVOXEL};

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

/// How each of the cell's six faces is crossed, packed two bits per face.
///
/// Face `f` occupies bits `2f` and `2f+1`, holding [`FACE_UNIFORM`],
/// [`FACE_SURFACE`] or [`FACE_JUNCTION`]. Like [`crossing_mask`] this reads the
/// cell's eight corners and nothing else, so neighbouring chunks classify a
/// shared face identically.
///
/// Only the *pattern* of equalities among a face's four corner labels matters,
/// never the label values, which is what lets a 64-entry table decide it.
///
/// [`FACE_UNIFORM`]: crate::tables::FACE_UNIFORM
/// [`FACE_SURFACE`]: crate::tables::FACE_SURFACE
/// [`FACE_JUNCTION`]: crate::tables::FACE_JUNCTION
#[inline]
pub fn face_kinds<T: Label>(corners: &[T; NCORNERS]) -> u16 {
    let mut packed = 0u16;
    for (f, face) in FACE_CORNERS.iter().enumerate() {
        let l = [
            corners[face[0] as usize],
            corners[face[1] as usize],
            corners[face[2] as usize],
            corners[face[3] as usize],
        ];
        let mut code = 0usize;
        let mut bit = 0;
        for i in 0..4 {
            for j in (i + 1)..4 {
                if l[i] == l[j] {
                    code |= 1 << bit;
                }
                bit += 1;
            }
        }
        packed |= (FACE_KIND[code] as u16) << (2 * f);
    }
    packed
}

/// Whether this cell's vertex sits on a junction, in Frisken's sense: some face
/// has more than one surface crossing it.
///
/// Such a vertex is faired against its junction neighbours alone, so that it
/// slides along the junction curve instead of being dragged off it by the
/// ordinary walls that also meet there.
#[inline]
pub fn is_edge_vertex(face_kinds: u16) -> bool {
    (0..6).any(|f| (face_kinds >> (2 * f)) & 0b11 == FACE_JUNCTION as u16)
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
