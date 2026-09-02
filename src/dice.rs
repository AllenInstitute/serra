//! Cutting a finished mesh into a regular grid of cells.
//!
//! Neuroglancer's multi-resolution mesh format stores each level of detail as
//! fragments on an octree grid, and quantizes each fragment's vertices over the
//! cell it is filed under. So a fragment must contain nothing outside its own
//! cell, and two fragments that meet must agree exactly on the geometry along
//! the plane between them — a tolerance is no use, because the format has
//! already thrown the original coordinates away by the time anything could weld
//! them.
//!
//! Both properties come from the same two decisions.
//!
//! **Classification is exact — there is no epsilon.** A vertex is below a
//! plane, above it, or on it, by exact comparison. That is what makes a
//! T-junction impossible: a plane crosses the interior of a shared edge only if
//! one endpoint is strictly below and the other strictly above, and both
//! triangles holding that edge see the same two coordinates, so either both
//! split it or neither does. Treating a vertex within some epsilon of a plane
//! as lying on it is what breaks this — the triangle whose corner got snapped
//! is no longer split, while its neighbour across the far edge still is, and the
//! vertex that neighbour inserted sits in the middle of an edge the first
//! triangle spans. The surface is then torn along an edge no weld can find,
//! because the two sides genuinely disagree about how many vertices are there.
//! Exactness costs a sliver triangle where a vertex lies a hair off a plane,
//! which is harmless: quantization collapses it and the decoder drops it, which
//! leaves the edge shared by exactly the two triangles either side.
//!
//! **A cut point is computed from the edge, in a canonical order.** Two
//! triangles sharing an edge each clip it themselves, so the arithmetic has to
//! land on the same bits both times. Ordering the endpoints by coordinate makes
//! the interpolation identical whichever triangle asks for it, whichever way
//! round that triangle happens to wind. This is the same argument
//! [`crate::mesh`] makes for choosing a quad's diagonal.
//!
//! Cell membership is *tracked through the splitting* rather than recovered
//! afterwards from the geometry. When a piece is cut at plane `p` on an axis,
//! the far half is known to live at index `p` on that axis — no rounding, no
//! centroid, no reciprocal to be inexact. Recovering it from coordinates
//! afterwards is what puts geometry in the wrong fragment.

use rustc_hash::FxHashMap;

use crate::mesh::TriangleMesh;

/// A cell under construction: its vertices, its faces, and the index that
/// shares a vertex between the triangles meeting at it.
///
/// Vertices are keyed on their float32 bit patterns, so two triangles share one
/// only when they agree exactly — the same standard the fragments are later
/// joined by.
type Bucket = (Vec<[f32; 3]>, Vec<[u32; 3]>, FxHashMap<[u32; 3], u32>);

/// One cell of the grid and the piece of the mesh that belongs to it.
pub struct DicedCell {
    /// Cell index along each axis.
    pub index: [i64; 3],
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

/// Where the cutting planes are.
#[derive(Clone, Copy)]
pub struct DiceOptions {
    /// Cell size along each axis, in the mesh's own units.
    pub chunk_shape: [f64; 3],
    /// Position of the grid's lower corner.
    pub grid_origin: [f64; 3],
    /// How many cells the grid holds along each axis. Indices are clamped into
    /// it, so a vertex sitting exactly on the far boundary of the last cell
    /// stays in that cell instead of starting a new one.
    pub grid_size: [i64; 3],
    /// Emit vertices already on the octree format's integer lattice, as
    /// integers in `[0, 2^bits - 1]` measured across the cell they belong to.
    ///
    /// This is the same place `to_stored_model_space` would put them, arrived at
    /// one step earlier — and that matters for more than saving a pass. Two
    /// cells sharing a vertex agree on it here only as closely as float32 can
    /// represent it, and at coordinates of a thousand voxels one float32 step is
    /// about a nanometre. Rounding onto the lattice *before* vertices are shared
    /// within a cell collapses that difference instead of preserving it, so the
    /// fragments agree exactly rather than very nearly.
    ///
    /// The rounding is done on a single grid spanning the whole volume and the
    /// cell's own offset subtracted afterwards, so the value a shared vertex
    /// gets in one cell and in its neighbour come from the same arithmetic:
    /// `Q` in the lower cell and `0` in the upper one, exactly.
    pub quantization_bits: Option<u32>,
}

/// Which side of a plane a vertex lies on. `On` is exact equality.
#[derive(PartialEq, Eq, Clone, Copy)]
enum Side {
    Below,
    On,
    Above,
}

#[inline]
fn side(value: f64, plane: f64) -> Side {
    if value < plane {
        Side::Below
    } else if value > plane {
        Side::Above
    } else {
        Side::On
    }
}

/// True if `a` sorts before `b`. Any total order on coordinates will do; this
/// one only has to be the same for both triangles holding an edge.
#[inline]
fn precedes(a: &[f64; 3], b: &[f64; 3]) -> bool {
    (a[0], a[1], a[2]) < (b[0], b[1], b[2])
}

/// Where the segment `a`–`b` crosses `plane` on `axis`.
///
/// The endpoints are put in a canonical order first, so this returns the same
/// bits for an edge however it is wound, and the coordinate on the split axis is
/// set to the plane rather than interpolated onto it.
#[inline]
fn cut(a: &[f64; 3], b: &[f64; 3], axis: usize, plane: f64) -> [f64; 3] {
    let (p, q) = if precedes(a, b) { (a, b) } else { (b, a) };
    let t = (plane - p[axis]) / (q[axis] - p[axis]);
    let mut r = [0.0f64; 3];
    for k in 0..3 {
        r[k] = p[k] + (q[k] - p[k]) * t;
    }
    r[axis] = plane;
    r
}

/// A convex polygon being cut up, and the cell it currently belongs to.
struct Piece {
    points: Vec<[f64; 3]>,
    index: [i64; 3],
}

/// Keep the part of `points` on one side of the plane, in order.
fn keep(points: &[[f64; 3]], axis: usize, plane: f64, want: Side, out: &mut Vec<[f64; 3]>) {
    out.clear();
    let n = points.len();
    for i in 0..n {
        let a = &points[i];
        let b = &points[(i + 1) % n];
        let sa = side(a[axis], plane);
        let sb = side(b[axis], plane);

        if sa == want || sa == Side::On {
            out.push(*a);
        }
        // A cut point is needed only where the plane separates the two ends
        // strictly; an endpoint already on the plane has just been emitted.
        if (sa == Side::Below && sb == Side::Above) || (sa == Side::Above && sb == Side::Below) {
            out.push(cut(a, b, axis, plane));
        }
    }
}

/// Cut a mesh into the cells of a regular grid.
///
/// Every triangle ends up in exactly one cell, wholly inside it, and the
/// fragments share their boundary geometry bit for bit. Concatenating them and
/// deduplicating vertices therefore reproduces the input mesh.
pub fn dice(mesh: &TriangleMesh, options: &DiceOptions) -> Vec<DicedCell> {
    let mut buckets: FxHashMap<[i64; 3], Bucket> = FxHashMap::default();

    let limit = [
        options.grid_size[0].max(1) - 1,
        options.grid_size[1].max(1) - 1,
        options.grid_size[2].max(1) - 1,
    ];

    let cell_of = |value: f64, axis: usize| -> i64 {
        let raw = (value - options.grid_origin[axis]) / options.chunk_shape[axis];
        (raw.floor() as i64).clamp(0, limit[axis])
    };

    // The lattice, if one was asked for: `step` converts a coordinate to a
    // global lattice index and `span` is how many of those a cell covers.
    let lattice = options.quantization_bits.map(|bits| {
        let span = ((1u64 << bits) - 1) as f64;
        (span, options.chunk_shape.map(|cs| span / cs))
    });

    let mut pieces: Vec<Piece> = Vec::with_capacity(8);
    let mut next: Vec<Piece> = Vec::with_capacity(8);
    let mut scratch: Vec<[f64; 3]> = Vec::with_capacity(8);

    for face in &mesh.faces {
        let corners: [[f64; 3]; 3] = [
            to_f64(&mesh.vertices[face[0] as usize]),
            to_f64(&mesh.vertices[face[1] as usize]),
            to_f64(&mesh.vertices[face[2] as usize]),
        ];

        let mut lo = [0i64; 3];
        let mut hi = [0i64; 3];
        for axis in 0..3 {
            let a = corners[0][axis].min(corners[1][axis]).min(corners[2][axis]);
            let b = corners[0][axis].max(corners[1][axis]).max(corners[2][axis]);
            lo[axis] = cell_of(a, axis);
            hi[axis] = cell_of(b, axis);
        }

        // The common case by a wide margin: the triangle is inside one cell.
        if lo == hi {
            emit(&mut buckets, lo, &corners, &options.grid_origin, lattice);
            continue;
        }

        pieces.clear();
        pieces.push(Piece {
            points: corners.to_vec(),
            index: lo,
        });

        for axis in 0..3 {
            for plane_index in (lo[axis] + 1)..=hi[axis] {
                let plane =
                    options.grid_origin[axis] + plane_index as f64 * options.chunk_shape[axis];
                next.clear();
                for piece in pieces.drain(..) {
                    let mut any_below = false;
                    let mut any_above = false;
                    for point in &piece.points {
                        match side(point[axis], plane) {
                            Side::Below => any_below = true,
                            Side::Above => any_above = true,
                            Side::On => {}
                        }
                    }

                    if !any_above {
                        next.push(piece);
                        continue;
                    }
                    if !any_below {
                        let mut moved = piece;
                        moved.index[axis] = plane_index;
                        next.push(moved);
                        continue;
                    }

                    keep(&piece.points, axis, plane, Side::Below, &mut scratch);
                    if scratch.len() >= 3 {
                        next.push(Piece {
                            points: scratch.clone(),
                            index: piece.index,
                        });
                    }
                    keep(&piece.points, axis, plane, Side::Above, &mut scratch);
                    if scratch.len() >= 3 {
                        let mut index = piece.index;
                        index[axis] = plane_index;
                        next.push(Piece {
                            points: scratch.clone(),
                            index,
                        });
                    }
                }
                std::mem::swap(&mut pieces, &mut next);
            }
        }

        // Fan-triangulate. Clipping a triangle by planes leaves a convex
        // polygon with the winding it started with, so a fan preserves the
        // orientation of the surface.
        for piece in pieces.drain(..) {
            for i in 1..piece.points.len() - 1 {
                emit(
                    &mut buckets,
                    piece.index,
                    &[piece.points[0], piece.points[i], piece.points[i + 1]],
                    &options.grid_origin,
                    lattice,
                );
            }
        }
    }

    let mut cells: Vec<DicedCell> = buckets
        .into_iter()
        .map(|(index, (vertices, faces, _))| DicedCell {
            index,
            vertices,
            faces,
        })
        .collect();
    cells.sort_by_key(|c| c.index);
    cells
}

#[inline]
fn to_f64(v: &[f32; 3]) -> [f64; 3] {
    [v[0] as f64, v[1] as f64, v[2] as f64]
}

/// Add one triangle to a cell, sharing vertices within that cell.
fn emit(
    buckets: &mut FxHashMap<[i64; 3], Bucket>,
    index: [i64; 3],
    triangle: &[[f64; 3]; 3],
    grid_origin: &[f64; 3],
    lattice: Option<(f64, [f64; 3])>,
) {
    let mut corner = [[0.0f32; 3]; 3];
    match lattice {
        None => {
            for i in 0..3 {
                for k in 0..3 {
                    corner[i][k] = triangle[i][k] as f32;
                }
            }
        }
        Some((span, step)) => {
            // Round on the grid spanning the whole volume, then subtract this
            // cell's offset, so a shared vertex comes out of identical
            // arithmetic on both sides of the plane between two cells.
            for i in 0..3 {
                for k in 0..3 {
                    let global = ((triangle[i][k] - grid_origin[k]) * step[k]).round();
                    let local = global - index[k] as f64 * span;
                    corner[i][k] = local.clamp(0.0, span) as f32;
                }
            }
        }
    }
    // A triangle that lost two of its corners to rounding encloses nothing.
    // Dropping it leaves the edge it shared with each neighbour held by exactly
    // those two neighbours, which is what an unrepaired surface needs.
    if corner[0] == corner[1] || corner[1] == corner[2] || corner[0] == corner[2] {
        return;
    }

    let bucket = buckets.entry(index).or_default();
    let mut face = [0u32; 3];
    for i in 0..3 {
        let key = [
            corner[i][0].to_bits(),
            corner[i][1].to_bits(),
            corner[i][2].to_bits(),
        ];
        face[i] = *bucket.2.entry(key).or_insert_with(|| {
            bucket.0.push(corner[i]);
            (bucket.0.len() - 1) as u32
        });
    }
    bucket.1.push(face);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_mesh() -> TriangleMesh {
        // A flat sheet spanning x in [0, 4], y in [0, 4], at z = 0, cut into a
        // 4x4 grid of quads so plenty of triangles straddle the planes below.
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let n = 5;
        for i in 0..n {
            for j in 0..n {
                vertices.push([i as f32, j as f32, 0.0]);
            }
        }
        let at = |i: usize, j: usize| (i * n + j) as u32;
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                faces.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                faces.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        TriangleMesh {
            vertices,
            faces,
            normals: None,
            pinned: Vec::new(),
        }
    }

    fn options(chunk: f64, size: i64) -> DiceOptions {
        DiceOptions {
            chunk_shape: [chunk; 3],
            grid_origin: [0.0; 3],
            grid_size: [size, size, 1],
            quantization_bits: None,
        }
    }

    fn area(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> f64 {
        let mut total = 0.0;
        for f in faces {
            let a = to_f64(&vertices[f[0] as usize]);
            let b = to_f64(&vertices[f[1] as usize]);
            let c = to_f64(&vertices[f[2] as usize]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            total += 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        }
        total
    }

    #[test]
    fn area_is_preserved() {
        let mesh = quad_mesh();
        let before = area(&mesh.vertices, &mesh.faces);
        // 1.5 is deliberately not a whole number of the mesh's own spacing, so
        // the planes fall inside triangles rather than along their edges.
        let cells = dice(&mesh, &options(1.5, 3));
        let after: f64 = cells.iter().map(|c| area(&c.vertices, &c.faces)).sum();
        assert!((before - after).abs() < 1e-9, "{before} vs {after}");
    }

    #[test]
    fn nothing_escapes_its_cell() {
        let options = options(1.5, 3);
        for cell in dice(&quad_mesh(), &options) {
            for axis in 0..3 {
                let lo =
                    options.grid_origin[axis] + cell.index[axis] as f64 * options.chunk_shape[axis];
                let hi = lo + options.chunk_shape[axis];
                for v in &cell.vertices {
                    let value = v[axis] as f64;
                    assert!(
                        value >= lo - 1e-6 && value <= hi + 1e-6,
                        "{value} not in [{lo}, {hi}]"
                    );
                }
            }
        }
    }

    #[test]
    fn the_union_is_watertight() {
        // Every edge of the reassembled surface must be shared by exactly two
        // triangles, except the ones on the outer rim of the sheet, which the
        // input has too. A T-junction shows up here as an extra count of one.
        let mesh = quad_mesh();
        let expected = open_edges(&mesh.vertices, &mesh.faces);

        let cells = dice(&mesh, &options(1.5, 3));
        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();
        let mut index: FxHashMap<[u32; 3], u32> = FxHashMap::default();
        for cell in &cells {
            for f in &cell.faces {
                let mut nf = [0u32; 3];
                for i in 0..3 {
                    let v = cell.vertices[f[i] as usize];
                    let key = [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
                    nf[i] = *index.entry(key).or_insert_with(|| {
                        vertices.push(v);
                        (vertices.len() - 1) as u32
                    });
                }
                faces.push(nf);
            }
        }
        // The rim gains vertices where the planes cross it, so count length
        // rather than edges.
        let got = open_edges(&vertices, &faces);
        assert!(
            (got - expected).abs() < 1e-9,
            "open boundary length {got} != {expected}"
        );
    }

    #[test]
    fn the_lattice_makes_neighbours_agree_exactly() {
        // On the lattice, a vertex two cells share must come out as the top of
        // one cell's range and the bottom of the next's, exactly -- that is what
        // makes the fragments weld with no tolerance at all.
        let mut options = options(1.5, 3);
        options.quantization_bits = Some(16);
        let span = ((1u64 << 16) - 1) as f32;

        let cells = dice(&quad_mesh(), &options);
        assert!(cells.len() > 1);
        let mut seam_pairs = 0;
        for cell in &cells {
            for v in &cell.vertices {
                for k in 0..3 {
                    // every coordinate is an integer in range
                    assert_eq!(v[k], v[k].round(), "{:?} is not on the lattice", v);
                    assert!(v[k] >= 0.0 && v[k] <= span, "{:?} out of range", v);
                    if v[k] == 0.0 || v[k] == span {
                        seam_pairs += 1;
                    }
                }
            }
        }
        assert!(seam_pairs > 0, "expected vertices on the cell faces");
    }

    #[test]
    fn the_lattice_union_is_still_watertight() {
        let mesh = quad_mesh();
        let expected = open_edges(&mesh.vertices, &mesh.faces);
        let mut options = options(1.5, 3);
        options.quantization_bits = Some(16);
        let cells = dice(&mesh, &options);

        // Put every fragment back into world coordinates before welding.
        let span = ((1u64 << 16) - 1) as f64;
        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();
        let mut index: FxHashMap<[u32; 3], u32> = FxHashMap::default();
        for cell in &cells {
            for f in &cell.faces {
                let mut nf = [0u32; 3];
                for i in 0..3 {
                    let stored = cell.vertices[f[i] as usize];
                    let mut world = [0.0f32; 3];
                    for k in 0..3 {
                        let global = stored[k] as f64 + cell.index[k] as f64 * span;
                        world[k] = (options.grid_origin[k] + global / span * options.chunk_shape[k])
                            as f32;
                    }
                    let key = [world[0].to_bits(), world[1].to_bits(), world[2].to_bits()];
                    nf[i] = *index.entry(key).or_insert_with(|| {
                        vertices.push(world);
                        (vertices.len() - 1) as u32
                    });
                }
                faces.push(nf);
            }
        }
        let got = open_edges(&vertices, &faces);
        assert!(
            (got - expected).abs() < 1e-3,
            "open boundary length {got} != {expected}"
        );
    }

    /// Total length of edges not shared by exactly two triangles.
    fn open_edges(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> f64 {
        let mut counts: FxHashMap<(u32, u32), i32> = FxHashMap::default();
        for f in faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        let mut total = 0.0;
        for ((a, b), n) in counts {
            if n != 2 {
                let p = to_f64(&vertices[a as usize]);
                let q = to_f64(&vertices[b as usize]);
                total +=
                    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            }
        }
        total
    }
}
