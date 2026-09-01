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

/// Index of the edge joining `corner` and `corner ^ (1 << axis)`.
pub const EDGE_INDEX: [[u8; 3]; NCORNERS] = build_edge_index();

const fn build_edge_index() -> [[u8; 3]; NCORNERS] {
    let mut out = [[0u8; 3]; NCORNERS];
    let mut c = 0;
    while c < NCORNERS {
        let mut axis = 0;
        while axis < 3 {
            let other = c ^ (1 << axis);
            let (lo, hi) = if c < other { (c, other) } else { (other, c) };
            let mut e = 0;
            while e < NEDGES {
                if EDGES[e].0 as usize == lo && EDGES[e].1 as usize == hi {
                    out[c][axis] = e as u8;
                }
                e += 1;
            }
            axis += 1;
        }
        c += 1;
    }
    out
}

/// The six faces of the cube, as the four edges lying in each.
pub const FACE_EDGES: [[u8; 4]; 6] = build_face_edges();

const fn build_face_edges() -> [[u8; 4]; 6] {
    let mut out = [[0u8; 4]; 6];
    let mut face = 0;
    while face < 6 {
        let axis = face / 2;
        let value = face % 2;
        let mut n = 0;
        let mut e = 0;
        while e < NEDGES {
            let a = EDGES[e].0 as usize;
            let b = EDGES[e].1 as usize;
            // An edge lies in the face when both ends sit on it.
            if (a >> axis) & 1 == value && (b >> axis) & 1 == value {
                out[face][n] = e as u8;
                n += 1;
            }
            e += 1;
        }
        face += 1;
    }
    out
}

/// How a label's surface passes through a cell.
///
/// A cell contributes one vertex per connected component of the label's
/// *crossing edges* — the cube edges with exactly one end carrying the label.
/// That is the number of separate sheets of surface passing through the cell.
///
/// Splitting by connected components of the label's *corners* instead is not
/// enough, and gets it wrong in a case that occurs regularly in real data: six
/// label corners whose two background corners are body-diagonal (mask
/// `0b11100111`). The label is 6-connected, so a corner-based split yields one
/// vertex, but there is a sheet of surface around each background corner and
/// they would share it — a pinch point.
pub struct SurfaceTables {
    /// `component[mask][edge]` is which sheet that crossing edge belongs to,
    /// or [`NO_COMPONENT`] when the edge does not cross.
    pub component: [[u8; NEDGES]; 256],
    /// How many sheets, and so how many vertices, `mask` needs.
    pub sheets: [u8; 256],
}

/// Surface-sheet decomposition for every possible label mask.
pub static SURFACE: SurfaceTables = build_surface();

/// Whether a mask has a face where the label sits at two diagonally opposite
/// corners.
///
/// Such a face is the only place the surface can pinch: its four crossings can
/// be paired two ways, and whichever is chosen, some configurations are left
/// with a shared vertex or an over-used edge. Only vertices from these cells
/// need the repair pass, which keeps it off the hot path for the ~99% of cells
/// that cannot be affected.
pub static AMBIGUOUS: [bool; 256] = build_ambiguous();

const fn build_ambiguous() -> [bool; 256] {
    let mut out = [false; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut face = 0;
        while face < 6 {
            let mut crossings = 0;
            let mut k = 0;
            while k < 4 {
                let e = FACE_EDGES[face][k] as usize;
                if ((mask >> EDGES[e].0) & 1) != ((mask >> EDGES[e].1) & 1) {
                    crossings += 1;
                }
                k += 1;
            }
            // Four crossings on one face means two diagonal label corners.
            if crossings == 4 {
                out[mask] = true;
            }
            face += 1;
        }
        mask += 1;
    }
    out
}

/// Faces of a cell that carry any surface at all, as a bitmask over the 6
/// faces, indexed by the 12-bit crossing mask.
///
/// A face with no crossing separates nothing: the two cells either side of it
/// are not on a common sheet, and no quad joins them. Fairing must skip those
/// neighbours. Averaging across them instead collapses a one-voxel sheet to
/// zero thickness — the two sides are face-adjacent, so they meet in the middle
/// — which no displacement bound prevents, because the sides start exactly one
/// voxel apart.
pub static SURFACE_FACES: [u8; 1 << NEDGES] = build_face_mask(2);

/// Faces crossed by two or more surfaces: Frisken's *JunctionFace*, where three
/// or more materials meet, or two meet on opposite diagonals.
///
/// A cell with one of these carries an *EdgeVertex*, which is faired against
/// its junction neighbours alone so that it slides along the junction curve
/// rather than being dragged off it by the ordinary walls that also meet there.
pub static JUNCTION_FACES: [u8; 1 << NEDGES] = build_face_mask(3);

/// Bit `f` set when face `f` has at least `threshold` of its four perimeter
/// edges crossing.
///
/// The count is the whole classification. Around a face's four-cycle of edges,
/// zero changes means one material, two means a single boundary, and three or
/// four means two boundaries — either three materials, or two on opposite
/// diagonals. Exactly one change is impossible: a cyclic sequence cannot return
/// to itself after a single change. Verified against the corner-partition
/// definition over all 4140 distinguishable cell labellings; see the tests.
const fn build_face_mask(threshold: u32) -> [u8; 1 << NEDGES] {
    let mut out = [0u8; 1 << NEDGES];
    let mut mask = 0usize;
    while mask < (1 << NEDGES) {
        let mut face = 0;
        while face < 6 {
            let mut n = 0;
            let mut k = 0;
            while k < 4 {
                if mask & (1 << FACE_EDGES[face][k]) != 0 {
                    n += 1;
                }
                k += 1;
            }
            if n >= threshold {
                out[mask] |= 1 << face;
            }
            face += 1;
        }
        mask += 1;
    }
    out
}

/// The four corners of each cell face, in cyclic order, so entries 0/2 and 1/3
/// are the two diagonals. Ordered like [`crate::extract`]'s `RING`.
///
/// Only the tests use this: it expresses Frisken's classification directly, in
/// terms of how many materials sit on the face, so the cheap crossing-mask form
/// above can be checked against the definition rather than against itself.
pub const FACE_CORNERS: [[u8; 4]; 6] = build_face_corners();

const fn build_face_corners() -> [[u8; 4]; 6] {
    const CYCLE: [(usize, usize); 4] = [(0, 0), (1, 0), (1, 1), (0, 1)];
    let mut out = [[0u8; 4]; 6];
    let mut face = 0;
    while face < 6 {
        let axis = face / 2;
        let value = face % 2;
        let u = (axis + 1) % 3;
        let w = (axis + 2) % 3;
        let mut k = 0;
        while k < 4 {
            let (du, dv) = CYCLE[k];
            out[face][k] = ((value << axis) | (du << u) | (dv << w)) as u8;
            k += 1;
        }
        face += 1;
    }
    out
}

const fn build_surface() -> SurfaceTables {
    let mut component = [[NO_COMPONENT; NEDGES]; 256];
    let mut sheets = [0u8; 256];

    let mut mask = 0usize;
    while mask < 256 {
        // A cube edge crosses the surface when exactly one end carries the label.
        let mut rep = [NO_COMPONENT; NEDGES];
        let mut e = 0;
        while e < NEDGES {
            let a = (mask >> EDGES[e].0) & 1;
            let b = (mask >> EDGES[e].1) & 1;
            if a != b {
                rep[e] = e as u8;
            }
            e += 1;
        }

        // Join crossing edges that meet within a face. Repeated relaxation
        // rather than union-find, which const fn cannot express as tidily; the
        // graph has at most 12 nodes so this converges immediately.
        let mut round = 0;
        while round < NEDGES {
            let mut face = 0;
            while face < 6 {
                // Count this face's crossings.
                let mut present = [NO_COMPONENT; 4];
                let mut n = 0;
                let mut k = 0;
                while k < 4 {
                    let e = FACE_EDGES[face][k] as usize;
                    if rep[e] != NO_COMPONENT {
                        present[n] = e as u8;
                        n += 1;
                    }
                    k += 1;
                }

                if n == 2 {
                    // Unambiguous: the surface crosses the face once.
                    let (x, y) = (present[0] as usize, present[1] as usize);
                    let m = if rep[x] < rep[y] { rep[x] } else { rep[y] };
                    rep[x] = m;
                    rep[y] = m;
                } else if n == 4 {
                    // Ambiguous: the label occupies two diagonally opposite
                    // corners. Keeping them apart is the 6-connected reading,
                    // matching how the label's own corners are treated. Both
                    // cells sharing this face see the same corners, so they
                    // resolve it the same way.
                    let mut c = 0;
                    while c < NCORNERS {
                        if (mask >> c) & 1 == 1 && (c >> (face / 2)) & 1 == face % 2 {
                            // Join the crossing edges meeting at this corner.
                            let mut first = NO_COMPONENT;
                            let mut k = 0;
                            while k < 4 {
                                let e = FACE_EDGES[face][k] as usize;
                                if rep[e] != NO_COMPONENT
                                    && (EDGES[e].0 as usize == c || EDGES[e].1 as usize == c)
                                {
                                    if first == NO_COMPONENT {
                                        first = e as u8;
                                    } else {
                                        let x = first as usize;
                                        let m = if rep[x] < rep[e] { rep[x] } else { rep[e] };
                                        rep[x] = m;
                                        rep[e] = m;
                                    }
                                }
                                k += 1;
                            }
                        }
                        c += 1;
                    }
                }
                face += 1;
            }
            round += 1;
        }

        // Renumber densely in increasing edge order, so ids depend only on the
        // mask.
        let mut dense = [NO_COMPONENT; NEDGES];
        let mut next = 0u8;
        let mut e = 0;
        while e < NEDGES {
            if rep[e] != NO_COMPONENT {
                let r = rep[e] as usize;
                if dense[r] == NO_COMPONENT {
                    dense[r] = next;
                    next += 1;
                }
                component[mask][e] = dense[r];
            }
            e += 1;
        }
        sheets[mask] = next;
        mask += 1;
    }

    SurfaceTables { component, sheets }
}

#[cfg(test)]
mod surface_tests {
    // Faces and edges are indexed by their geometric number throughout, which
    // is the natural formulation here.
    #![allow(clippy::needless_range_loop)]

    use super::*;

    /// Reference: flood fill over crossing edges, joined within each face.
    fn reference(mask: usize) -> (Vec<Option<u8>>, u8) {
        let crossing: Vec<bool> = (0..NEDGES)
            .map(|e| ((mask >> EDGES[e].0) & 1) != ((mask >> EDGES[e].1) & 1))
            .collect();

        let mut links: Vec<Vec<usize>> = vec![Vec::new(); NEDGES];
        for face in 0..6 {
            let on_face: Vec<usize> = FACE_EDGES[face]
                .iter()
                .map(|&e| e as usize)
                .filter(|&e| crossing[e])
                .collect();
            if on_face.len() == 2 {
                links[on_face[0]].push(on_face[1]);
                links[on_face[1]].push(on_face[0]);
            } else if on_face.len() == 4 {
                for c in 0..NCORNERS {
                    if (mask >> c) & 1 == 1 && (c >> (face / 2)) & 1 == face % 2 {
                        let at_corner: Vec<usize> = on_face
                            .iter()
                            .copied()
                            .filter(|&e| EDGES[e].0 as usize == c || EDGES[e].1 as usize == c)
                            .collect();
                        for i in 0..at_corner.len() {
                            for j in (i + 1)..at_corner.len() {
                                links[at_corner[i]].push(at_corner[j]);
                                links[at_corner[j]].push(at_corner[i]);
                            }
                        }
                    }
                }
            }
        }

        let mut comp: Vec<Option<u8>> = vec![None; NEDGES];
        let mut next = 0u8;
        for start in 0..NEDGES {
            if !crossing[start] || comp[start].is_some() {
                continue;
            }
            let mut stack = vec![start];
            comp[start] = Some(next);
            while let Some(e) = stack.pop() {
                for &w in &links[e] {
                    if comp[w].is_none() {
                        comp[w] = Some(next);
                        stack.push(w);
                    }
                }
            }
            next += 1;
        }
        (comp, next)
    }

    #[test]
    fn surface_split_matches_reference_for_all_256_masks() {
        for mask in 0..256usize {
            let (want, want_n) = reference(mask);
            assert_eq!(SURFACE.sheets[mask], want_n, "sheet count for {mask:08b}");
            for e in 0..NEDGES {
                let got = SURFACE.component[mask][e];
                match want[e] {
                    None => assert_eq!(got, NO_COMPONENT, "mask {mask:08b} edge {e}"),
                    Some(w) => assert_eq!(got, w, "mask {mask:08b} edge {e}"),
                }
            }
        }
    }

    /// The configuration that motivated this table.
    #[test]
    fn six_corners_with_a_diagonal_pair_missing_needs_two_sheets() {
        let mask = 0b1110_0111usize;
        assert_eq!(TABLES.ncomp[mask], 1, "the label's corners are 6-connected");
        assert_eq!(
            SURFACE.sheets[mask], 2,
            "but there is a sheet around each background corner"
        );
    }

    #[test]
    fn a_flat_wall_is_a_single_sheet() {
        // Corners with bit 0 clear are the label: one planar crossing.
        let mask = (0..NCORNERS)
            .filter(|c| c & 1 == 0)
            .fold(0, |m, c| m | 1 << c);
        assert_eq!(SURFACE.sheets[mask], 1);
        assert_eq!(
            SURFACE.component[mask]
                .iter()
                .filter(|&&c| c != NO_COMPONENT)
                .count(),
            4
        );
    }

    #[test]
    fn an_isolated_corner_is_a_single_sheet_of_three_edges() {
        for c in 0..NCORNERS {
            let mask = 1usize << c;
            assert_eq!(SURFACE.sheets[mask], 1, "corner {c}");
            assert_eq!(
                SURFACE.component[mask]
                    .iter()
                    .filter(|&&x| x != NO_COMPONENT)
                    .count(),
                3
            );
        }
    }

    #[test]
    fn the_checkerboard_still_gives_four_sheets() {
        let even = (0..NCORNERS)
            .filter(|c| (*c as u32).count_ones() % 2 == 0)
            .fold(0usize, |m, c| m | 1 << c);
        assert_eq!(SURFACE.sheets[even], 4);
    }

    #[test]
    fn empty_and_full_masks_have_no_surface() {
        assert_eq!(SURFACE.sheets[0], 0);
        assert_eq!(SURFACE.sheets[0xFF], 0);
    }

    #[test]
    fn every_crossing_edge_is_assigned_and_ids_are_dense() {
        for mask in 0..256usize {
            let n = SURFACE.sheets[mask];
            let mut seen = vec![false; n as usize];
            for e in 0..NEDGES {
                let crossing = ((mask >> EDGES[e].0) & 1) != ((mask >> EDGES[e].1) & 1);
                let got = SURFACE.component[mask][e];
                assert_eq!(crossing, got != NO_COMPONENT, "mask {mask:08b} edge {e}");
                if got != NO_COMPONENT {
                    assert!(got < n);
                    seen[got as usize] = true;
                }
            }
            assert!(seen.iter().all(|&s| s), "mask {mask:08b} has a gap in ids");
        }
    }

    #[test]
    fn sheets_never_exceed_four() {
        for mask in 0..256usize {
            assert!(SURFACE.sheets[mask] <= 4, "mask {mask:08b}");
        }
    }

    #[test]
    fn edge_index_is_consistent_with_the_edge_list() {
        for c in 0..NCORNERS {
            for axis in 0..3 {
                let e = EDGE_INDEX[c][axis] as usize;
                let other = c ^ (1 << axis);
                let (lo, hi) = if c < other { (c, other) } else { (other, c) };
                assert_eq!((EDGES[e].0 as usize, EDGES[e].1 as usize), (lo, hi));
            }
        }
    }
}

#[cfg(test)]
mod face_kind_tests {
    use super::*;

    /// Frisken's classification stated directly: how many materials sit on the
    /// face, and are two of them diagonally opposed.
    fn from_corners(l: [u32; 4]) -> u32 {
        let mut distinct = 0;
        for i in 0..4 {
            if (0..i).all(|j| l[j] != l[i]) {
                distinct += 1;
            }
        }
        let diagonal = l[0] == l[2] && l[1] == l[3] && l[0] != l[1];
        if distinct == 1 {
            0 // uniform
        } else if distinct >= 3 || diagonal {
            2 // junction
        } else {
            1 // surface
        }
    }

    /// Every distinguishable labelling of a cell's 8 corners: the 4140 set
    /// partitions, generated as restricted growth strings.
    fn each_partition(mut f: impl FnMut(&[u32; NCORNERS])) {
        fn rec(a: &mut [u32; NCORNERS], i: usize, max: i32, f: &mut impl FnMut(&[u32; 8])) {
            if i == NCORNERS {
                f(a);
                return;
            }
            for v in 0..=(max + 1) {
                a[i] = v as u32;
                rec(a, i + 1, max.max(v), f);
            }
        }
        let mut a = [0u32; NCORNERS];
        rec(&mut a, 0, -1, &mut f);
    }

    fn crossings_of(corners: &[u32; NCORNERS]) -> usize {
        let mut mask = 0usize;
        for (e, &(a, b)) in EDGES.iter().enumerate() {
            if corners[a as usize] != corners[b as usize] {
                mask |= 1 << e;
            }
        }
        mask
    }

    /// The crossing-mask tables must agree with the corner definition on every
    /// configuration a cell can actually take.
    #[test]
    fn crossing_mask_classification_matches_the_definition() {
        let mut seen = [0usize; 5];
        let mut cells = 0;
        each_partition(|corners| {
            cells += 1;
            let mask = crossings_of(corners);
            for (f, face) in FACE_CORNERS.iter().enumerate() {
                let want = from_corners([
                    corners[face[0] as usize],
                    corners[face[1] as usize],
                    corners[face[2] as usize],
                    corners[face[3] as usize],
                ]);
                let crossings = FACE_EDGES[f]
                    .iter()
                    .filter(|&&e| mask & (1 << e) != 0)
                    .count();
                seen[crossings] += 1;
                let got = if SURFACE_FACES[mask] >> f & 1 == 0 {
                    0
                } else if JUNCTION_FACES[mask] >> f & 1 == 1 {
                    2
                } else {
                    1
                };
                assert_eq!(got, want, "corners {corners:?} face {f}");
            }
        });
        assert_eq!(cells, 4140, "set partitions of 8 corners");
        // One crossing on a face is impossible: a cyclic sequence cannot return
        // to itself after a single change. That is why "no crossings" and "some
        // crossings" can be told apart by a threshold of 2.
        assert_eq!(seen[1], 0, "a face with exactly one crossing occurred");
        assert!(seen[0] > 0 && seen[2] > 0 && seen[3] > 0 && seen[4] > 0);
    }

    #[test]
    fn junction_implies_surface() {
        // A junction face is a face, so the stencil mask must be a superset.
        for mask in 0..(1usize << NEDGES) {
            assert_eq!(
                JUNCTION_FACES[mask] & !SURFACE_FACES[mask],
                0,
                "mask {mask:#x} has a junction on a face with no surface"
            );
        }
    }

    #[test]
    fn a_uniform_cell_has_no_faces_and_no_junction() {
        assert_eq!(SURFACE_FACES[0], 0);
        assert_eq!(JUNCTION_FACES[0], 0);
    }

    #[test]
    fn face_corners_are_cyclic_and_cover_each_face() {
        for (f, face) in FACE_CORNERS.iter().enumerate() {
            let (axis, value) = (f / 2, f % 2);
            let mut seen = [false; NCORNERS];
            for &c in face {
                assert_eq!(
                    (c as usize >> axis) & 1,
                    value,
                    "face {f} corner {c} off-face"
                );
                assert!(!seen[c as usize], "face {f} repeats corner {c}");
                seen[c as usize] = true;
            }
            for k in 0..4 {
                let (a, b) = (face[k] as usize, face[(k + 1) % 4] as usize);
                assert_eq!((a ^ b).count_ones(), 1, "face {f}: {a},{b} not an edge");
            }
            assert_eq!((face[0] ^ face[2]).count_ones(), 2);
            assert_eq!((face[1] ^ face[3]).count_ones(), 2);
        }
    }
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
