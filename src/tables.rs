//! Precomputed cube tables, all built at compile time by `const fn`.
//!
//! A *cell* is a 2x2x2 block of voxels. Its 8 corners are indexed so that the
//! bits of the index are the offsets along each axis:
//!
//! ```text
//! corner c  ->  (x, y, z) = (c & 1, (c >> 1) & 1, (c >> 2) & 1)
//! ```
//!
//! Two tables matter:
//!
//! * [`CENTROID`] — where a cell's dual vertex goes, as a pure function of which
//!   of the 12 cube edges have differing labels at their two ends. Because the
//!   crossing midpoints are at fixed positions, the centroid depends on nothing
//!   but that 12-bit pattern, so vertex placement is a single table lookup in
//!   exact integer arithmetic.
//!
//! * [`TABLES.split`] — how to split a cell's vertex so each object's surface
//!   stays 2-manifold. A label's dual surface is the boundary of its voxel set;
//!   non-manifold edges and vertices appear exactly where that set touches itself
//!   only diagonally. Splitting the cell vertex per 6-connected component of the
//!   label's corners separates those sheets.

/// Number of corners in a cell.
pub const NCORNERS: usize = 8;

/// Number of edges in a cell.
pub const NEDGES: usize = 12;

/// Marker for "this corner does not carry the label in question".
pub const NO_COMPONENT: u8 = u8::MAX;

/// Fixed-point scale for vertex positions: 1 voxel = 256 units.
///
/// Positions are kept as integers so that two neighbouring chunks meshing the
/// same seam cell produce bit-identical results, with no dependence on
/// floating-point association order.
pub const SUBVOXEL: i32 = 256;

/// Offset of a corner within its cell, in {0, 1} per axis.
#[inline]
pub const fn corner_offset(c: usize) -> [i32; 3] {
    [(c & 1) as i32, ((c >> 1) & 1) as i32, ((c >> 2) & 1) as i32]
}

/// The 12 edges of the cube, as `(lower_corner, upper_corner)` index pairs.
///
/// Two corners are joined by an edge exactly when their indices differ in a
/// single bit — that is, they are face-adjacent voxels (6-connectivity).
pub const EDGES: [(u8, u8); NEDGES] = build_edges();

const fn build_edges() -> [(u8, u8); NEDGES] {
    let mut out = [(0u8, 0u8); NEDGES];
    let mut n = 0;
    let mut a = 0;
    while a < NCORNERS {
        let mut axis = 0;
        while axis < 3 {
            let b = a ^ (1 << axis);
            if b > a {
                out[n] = (a as u8, b as u8);
                n += 1;
            }
            axis += 1;
        }
        a += 1;
    }
    out
}

/// Dual-vertex position per 12-bit edge-crossing pattern, in 1/256 voxel units
/// relative to the cell's minimum corner.
///
/// Index by a bitmask whose bit `e` is set when the two corners of `EDGES[e]`
/// carry different labels. Entry 0 (no crossings) is unused.
pub static CENTROID: [[i32; 3]; 1 << NEDGES] = build_centroid();

const fn build_centroid() -> [[i32; 3]; 1 << NEDGES] {
    let mut table = [[0i32; 3]; 1 << NEDGES];
    let mut mask = 0usize;
    while mask < (1 << NEDGES) {
        let mut sum = [0i32; 3];
        let mut count = 0i32;
        let mut e = 0;
        while e < NEDGES {
            if mask & (1 << e) != 0 {
                let a = corner_offset(EDGES[e].0 as usize);
                let b = corner_offset(EDGES[e].1 as usize);
                // Midpoint of the edge, scaled to fixed point. Each component of
                // (a + b) is 0, 1 or 2, so this is exact.
                let mut k = 0;
                while k < 3 {
                    sum[k] += (a[k] + b[k]) * (SUBVOXEL / 2);
                    k += 1;
                }
                count += 1;
            }
            e += 1;
        }
        if count > 0 {
            let mut k = 0;
            while k < 3 {
                // Round half up. Every quantity here is non-negative.
                table[mask][k] = (sum[k] + count / 2) / count;
                k += 1;
            }
        }
        mask += 1;
    }
    table
}

/// Per-mask component splits, indexed by the 8-bit "which corners carry this
/// label" mask.
pub struct CubeTables {
    /// `split[mask][corner]` is the component id of that corner, or
    /// [`NO_COMPONENT`] when the corner does not carry the label.
    pub split: [[u8; NCORNERS]; 256],
    /// `ncomp[mask]` is how many distinct components `mask` decomposes into,
    /// and therefore how many vertices the cell contributes for that label.
    pub ncomp: [u8; 256],
}

/// The component-split tables.
pub static TABLES: CubeTables = build_tables();

const fn build_tables() -> CubeTables {
    let mut split = [[NO_COMPONENT; NCORNERS]; 256];
    let mut ncomp = [0u8; 256];

    let mut mask = 0usize;
    while mask < 256 {
        // Label each present corner with itself, then repeatedly pull adjacent
        // corners down to the smaller representative. The cube's diameter is 3,
        // so a handful of rounds always reaches a fixed point; 8 is ample.
        let mut rep = [NO_COMPONENT; NCORNERS];
        let mut c = 0;
        while c < NCORNERS {
            if mask & (1 << c) != 0 {
                rep[c] = c as u8;
            }
            c += 1;
        }
        let mut round = 0;
        while round < NCORNERS {
            let mut e = 0;
            while e < NEDGES {
                let a = EDGES[e].0 as usize;
                let b = EDGES[e].1 as usize;
                if rep[a] != NO_COMPONENT && rep[b] != NO_COMPONENT {
                    let m = if rep[a] < rep[b] { rep[a] } else { rep[b] };
                    rep[a] = m;
                    rep[b] = m;
                }
                e += 1;
            }
            round += 1;
        }

        // Renumber representatives densely, in increasing corner order, so the
        // component ids are a deterministic function of the mask alone.
        let mut dense = [NO_COMPONENT; NCORNERS];
        let mut next = 0u8;
        let mut c = 0;
        while c < NCORNERS {
            if rep[c] != NO_COMPONENT {
                let r = rep[c] as usize;
                if dense[r] == NO_COMPONENT {
                    dense[r] = next;
                    next += 1;
                }
                split[mask][c] = dense[r];
            }
            c += 1;
        }
        ncomp[mask] = next;
        mask += 1;
    }

    CubeTables { split, ncomp }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_the_twelve_cube_edges() {
        assert_eq!(EDGES.len(), 12);
        for &(a, b) in EDGES.iter() {
            assert!(a < b, "edges are stored low-to-high");
            let diff = (a ^ b) as u32;
            assert_eq!(diff.count_ones(), 1, "edge corners differ in one axis");
        }
        // No duplicates.
        for (i, a) in EDGES.iter().enumerate() {
            for b in &EDGES[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    /// Reference implementation of the split, written the obvious way, to check
    /// the const-fn table against.
    fn reference_components(mask: u8) -> (Vec<Option<u8>>, u8) {
        let mut comp: Vec<Option<u8>> = vec![None; NCORNERS];
        let mut next = 0u8;
        for start in 0..NCORNERS {
            if mask & (1 << start) == 0 || comp[start].is_some() {
                continue;
            }
            // Breadth-first flood fill from `start` over face-adjacent corners.
            let mut stack = vec![start];
            comp[start] = Some(next);
            while let Some(c) = stack.pop() {
                for axis in 0..3 {
                    let n = c ^ (1 << axis);
                    if mask & (1 << n) != 0 && comp[n].is_none() {
                        comp[n] = Some(next);
                        stack.push(n);
                    }
                }
            }
            next += 1;
        }
        (comp, next)
    }

    #[test]
    fn split_matches_reference_for_all_256_masks() {
        for mask in 0..256usize {
            let (want, want_n) = reference_components(mask as u8);
            assert_eq!(
                TABLES.ncomp[mask], want_n,
                "component count for mask {mask:08b}"
            );
            for (c, expected) in want.iter().enumerate() {
                let got = TABLES.split[mask][c];
                match expected {
                    None => assert_eq!(got, NO_COMPONENT, "mask {mask:08b} corner {c}"),
                    Some(w) => assert_eq!(got, *w, "mask {mask:08b} corner {c}"),
                }
            }
        }
    }

    #[test]
    fn at_most_four_components_and_checkerboards_hit_that_bound() {
        for mask in 0..256usize {
            assert!(
                TABLES.ncomp[mask] <= 4,
                "mask {mask:08b} split into {} components",
                TABLES.ncomp[mask]
            );
        }
        // The two checkerboard masks are the worst case: four mutually
        // non-adjacent corners, so four separate sheets.
        let even = (0..8usize)
            .filter(|c| (*c as u32).count_ones() % 2 == 0)
            .fold(0usize, |m, c| m | 1 << c);
        let odd = 0xFF & !even;
        assert_eq!(TABLES.ncomp[even], 4);
        assert_eq!(TABLES.ncomp[odd], 4);
    }

    #[test]
    fn empty_and_full_masks_are_degenerate() {
        assert_eq!(TABLES.ncomp[0], 0);
        assert_eq!(TABLES.split[0], [NO_COMPONENT; NCORNERS]);
        assert_eq!(TABLES.ncomp[0xFF], 1);
        assert_eq!(TABLES.split[0xFF], [0; NCORNERS]);
    }

    #[test]
    fn component_ids_are_dense_and_ordered() {
        for mask in 0..256usize {
            let n = TABLES.ncomp[mask];
            let mut seen = vec![false; n as usize];
            let mut highest_so_far: i32 = -1;
            for c in 0..NCORNERS {
                let v = TABLES.split[mask][c];
                if v == NO_COMPONENT {
                    continue;
                }
                assert!(v < n, "mask {mask:08b}: id {v} out of range");
                if !seen[v as usize] {
                    // First appearance of each id must be in increasing order,
                    // which is what makes the numbering canonical.
                    assert_eq!(v as i32, highest_so_far + 1, "mask {mask:08b}");
                    highest_so_far = v as i32;
                    seen[v as usize] = true;
                }
            }
            assert!(seen.iter().all(|&s| s), "mask {mask:08b}: sparse ids");
        }
    }

    #[test]
    fn centroid_of_a_single_crossing_is_that_edge_midpoint() {
        for e in 0..NEDGES {
            let a = corner_offset(EDGES[e].0 as usize);
            let b = corner_offset(EDGES[e].1 as usize);
            let got = CENTROID[1 << e];
            for (k, &g) in got.iter().enumerate() {
                assert_eq!(g, (a[k] + b[k]) * (SUBVOXEL / 2), "edge {e} axis {k}");
            }
        }
    }

    #[test]
    fn centroid_stays_inside_the_cell() {
        for (mask, entry) in CENTROID.iter().enumerate().skip(1) {
            for (k, &v) in entry.iter().enumerate() {
                assert!(
                    (0..=SUBVOXEL).contains(&v),
                    "mask {mask:012b} axis {k} = {v}"
                );
            }
        }
    }

    /// A flat wall through the middle of the cell should put the vertex exactly
    /// on that plane, not merely near it.
    #[test]
    fn centroid_of_an_axis_aligned_wall_is_the_face_centre() {
        // Split the cube by x: corners with x = 0 are one label, x = 1 another.
        // The crossing edges are exactly the four running along x.
        let mut mask = 0usize;
        for (e, &(a, b)) in EDGES.iter().enumerate() {
            if (a ^ b) == 1 {
                mask |= 1 << e;
            }
        }
        assert_eq!(CENTROID[mask], [SUBVOXEL / 2, SUBVOXEL / 2, SUBVOXEL / 2]);
    }
}
