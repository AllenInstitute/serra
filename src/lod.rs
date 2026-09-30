//! Adaptive remeshing of the shared wall mesh, bounded by `max_error`.
//!
//! One level of detail is produced by [`remesh`]: Botsch and Kobbelt's isotropic
//! remeshing loop -- split long edges, collapse short ones, flip toward regular
//! vertex degree, relax vertices within the surface and project them back onto
//! it -- driven by a target edge length that varies with curvature, after
//! Dunyach et al. (2013), and with every operation bounded by `max_error`
//! against the fine surface.
//!
//! # The error bound is exact at every fine vertex
//!
//! Each vertex of the fine surface is assigned to the coarse triangle it lies
//! on. Before an operation is applied, the triangles it would produce are
//! worked out, and it goes ahead only if every fine vertex in the region stays
//! within `max_error` of one of them; afterwards those vertices are assigned to
//! their nearest new triangle. So no fine vertex is ever further than
//! `max_error` from the coarse surface. (Points *between* fine vertices are
//! not checked; at the fine mesh's density that is well below the bound.) The
//! assignment is also the fine-to-coarse map between levels.
//!
//! # Size
//!
//! The target edge length at a fine vertex is `sqrt(6 e / k - 3 e^2)`, where `e`
//! is `max_error` and `k` the curvature there: the longest edge whose chord
//! stays within `e` of a circle of curvature `k`. It is clamped between the
//! level's minimum and maximum. Wall curvature comes from the angle between
//! neighbouring triangles; curve curvature from how sharply the curve turns.
//!
//! # What stays where it is
//!
//! Wall vertices stay on their wall and curve vertices on their curve: they
//! are projected back onto the fine surface restricted to the same wall, or
//! onto the fine curve. Corners, fixed and locked vertices do not move at all
//! (see [`crate::remesh`]).
//!
//! # Small contacts
//!
//! Left exact, a coarse level keeps every contact between labels however
//! small, and the junction curves round a patch a voxel or two across hold
//! the triangles there to that size. With
//! [`LevelParams::drop_small_contacts`], [`level`] first separates each wall
//! patch no more than `2 * max_error` across that a single other material
//! surrounds ([`Remesh::separate_small_contacts`]): the two labels each keep
//! their own copy of the patch, at the same place, with that material between
//! them. Every label's surface keeps its shape, so the error bound is
//! unchanged; what changes is that the two no longer touch there, and the
//! material between them has one hole fewer. Pinches, where one label meets
//! itself, are never separated, so no object's own connectivity changes.
//!
//! The two copies are then moved a quarter of `max_error` apart, each into its
//! own label, and are never projected back onto the fine surface, where they
//! would coincide again. From there they are remeshed like any other walls:
//! the check below keeps them from crossing, so they can only move apart.
//!
//! # No crossings
//!
//! Each label's error bound says nothing about the walls on the far side of a
//! thin gap: across extracellular space thinner than `max_error`, two
//! objects' coarse surfaces could each stay within the bound and still pass
//! through each other. So no operation may add a crossing -- a triangle
//! passing through another that shares no vertex with it. Every split,
//! collapse, flip and move is checked against the faces near it, found
//! through a uniform grid, and refused if its triangles would cross more
//! faces than the ones they replace did. Touching is not crossing, and where
//! the fine surface already crosses itself, operations there are not all
//! refused, only kept from making it worse.
//!
//! Single-threaded and visited in id order, so the result is a pure function of
//! the input.

use rustc_hash::FxHashMap;

use crate::extract::CellField;
use crate::remesh::{EdgeKind, Remesh, Separated, VertexKind};
use crate::walls::{WallMesh, OUTSIDE};

/// What one level asks for.
#[derive(Clone, Copy, Debug)]
pub struct LevelParams {
    /// Longest edge wanted, in physical units.
    pub max_length: f64,
    /// Shortest edge wanted, in physical units.
    pub min_length: f64,
    /// How far any fine vertex may be from the result, in physical units.
    pub max_error: f64,
    /// Rounds of split, collapse, flip and relax.
    pub iterations: u32,
    /// Let [`level`] drop contacts between different materials that are too
    /// small for this level: see "Small contacts" in the module docs. Off,
    /// every label's surface keeps exactly the fine surface's topology.
    pub drop_small_contacts: bool,
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The point of triangle `(a, b, c)` nearest `p` (Ericson, *Real-Time
/// Collision Detection*, 5.1.5).
pub(crate) fn closest_on_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add(a, scale(ab, d1 / (d1 - d3)));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, scale(ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let bc = sub(c, b);
        return add(b, scale(bc, (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    add(a, add(scale(ab, v), scale(ac, w)))
}

fn closest_on_segment(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    if len2 <= 0.0 {
        return a;
    }
    let t = (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0);
    add(a, scale(ab, t))
}

/// Whether segment `p`-`q` passes through triangle `t`: its ends strictly on
/// opposite sides of the plane (by more than `tol`), and the crossing point
/// strictly inside. Touching -- an end on the plane, coplanar overlap -- is not
/// passing through.
fn pierces(p: [f64; 3], q: [f64; 3], t: &[[f64; 3]; 3], tol: f64) -> bool {
    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
    let l = norm(n);
    if l == 0.0 {
        return false;
    }
    let (dp, dq) = (dot(n, sub(p, t[0])) / l, dot(n, sub(q, t[0])) / l);
    if !((dp > tol && dq < -tol) || (dp < -tol && dq > tol)) {
        return false;
    }
    let x = add(p, scale(sub(q, p), dp / (dp - dq)));
    (0..3).all(|k| dot(cross(sub(t[(k + 1) % 3], t[k]), sub(x, t[k])), n) > 0.0)
}

/// Whether two triangles cross: some edge of one passes through the other,
/// which is how any two non-coplanar triangles that intersect do.
fn crosses(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], tol: f64) -> bool {
    // Wholly on one side of the other's plane: cannot cross.
    let apart = |s: &[[f64; 3]; 3], t: &[[f64; 3]; 3]| {
        let n = cross(sub(s[1], s[0]), sub(s[2], s[0]));
        let l = norm(n);
        if l == 0.0 {
            return true;
        }
        let d = t.map(|p| dot(n, sub(p, s[0])) / l);
        d.iter().all(|&x| x > -tol) || d.iter().all(|&x| x < tol)
    };
    if apart(a, b) || apart(b, a) {
        return false;
    }
    (0..3).any(|k| pierces(a[k], a[(k + 1) % 3], b, tol))
        || (0..3).any(|k| pierces(b[k], b[(k + 1) % 3], a, tol))
}

fn bounds(t: &[[f64; 3]; 3]) -> ([f64; 3], [f64; 3]) {
    let mut lo = t[0];
    let mut hi = t[0];
    for p in &t[1..] {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

/// `x` in single precision, moved a little further in direction `sign`, so
/// boxes compared in single precision never shrink by the conversion.
fn widen(sign: f64) -> impl Fn(f64) -> f32 {
    move |x| (x + sign * (1e-6 * x.abs() + 1e-9)) as f32
}

/// Coarse faces by where they are: a uniform grid of cells, each listing the
/// faces whose bounding box touches it. Entries go stale as faces change; a
/// query checks every candidate against the face as it is now, and the grid
/// is rebuilt at the start of each pass.
#[derive(Default)]
struct Grid {
    cell: f64,
    /// Per cell, each face with its bounding box when it was inserted, so a
    /// query can pass over most entries without looking at the mesh.
    cells: FxHashMap<[i64; 3], Vec<(u32, [f32; 6])>>,
    /// Entries in all cells, stale ones included.
    entries: usize,
    /// Faces inserted spanning more than a few cells along some axis.
    oversized: usize,
    /// Per face, the cells it is listed in, so moving it drops the old entries.
    span: Vec<Option<([i64; 3], [i64; 3])>>,
}

impl Grid {
    fn key(&self, p: [f64; 3]) -> [i64; 3] {
        p.map(|x| (x / self.cell).floor() as i64)
    }

    fn insert(&mut self, f: u32, t: &[[f64; 3]; 3]) {
        let (lo, hi) = bounds(t);
        let (a, b) = (self.key(lo), self.key(hi));
        let (down, up) = (widen(-1.0), widen(1.0));
        let bbox = [
            down(lo[0]),
            down(lo[1]),
            down(lo[2]),
            up(hi[0]),
            up(hi[1]),
            up(hi[2]),
        ];
        if (0..3).any(|k| b[k] - a[k] > 2) {
            self.oversized += 1;
        }
        if self.span.len() <= f as usize {
            self.span.resize(f as usize + 1, None);
        }
        if let Some((oa, ob)) = self.span[f as usize].take() {
            for i in oa[0]..=ob[0] {
                for j in oa[1]..=ob[1] {
                    for k in oa[2]..=ob[2] {
                        if let Some(list) = self.cells.get_mut(&[i, j, k]) {
                            let before = list.len();
                            list.retain(|&(g, _)| g != f);
                            self.entries -= before - list.len();
                        }
                    }
                }
            }
        }
        self.span[f as usize] = Some((a, b));
        for i in a[0]..=b[0] {
            for j in a[1]..=b[1] {
                for k in a[2]..=b[2] {
                    self.cells.entry([i, j, k]).or_default().push((f, bbox));
                    self.entries += 1;
                }
            }
        }
    }

    /// Faces whose box when inserted touched the box `lo`-`hi`, possibly
    /// repeated. A face that has moved since may be missing from its old
    /// place's list, but is always listed where it is now.
    fn near(&self, lo: [f64; 3], hi: [f64; 3], out: &mut Vec<u32>) {
        let (a, b) = (self.key(lo), self.key(hi));
        let (l, h) = (lo.map(widen(-1.0)), hi.map(widen(1.0)));
        for i in a[0]..=b[0] {
            for j in a[1]..=b[1] {
                for k in a[2]..=b[2] {
                    if let Some(list) = self.cells.get(&[i, j, k]) {
                        out.extend(list.iter().filter_map(|&(f, bb)| {
                            let apart = (0..3).any(|k| bb[k + 3] < l[k] || bb[k] > h[k]);
                            (!apart).then_some(f)
                        }));
                    }
                }
            }
        }
    }
}

/// Shape quality of a triangle: 1 equilateral, 0 degenerate.
fn tri_quality(t: &[[f64; 3]; 3]) -> f64 {
    let (ab, bc, ca) = (sub(t[1], t[0]), sub(t[2], t[1]), sub(t[0], t[2]));
    let l2 = dot(ab, ab) + dot(bc, bc) + dot(ca, ca);
    if l2 > 0.0 {
        2.0 * 3f64.sqrt() * norm(cross(ab, sub(t[2], t[0]))) / l2
    } else {
        0.0
    }
}

/// No operation may leave a triangle worse than this, unless the region
/// already had one at least as bad.
const QUALITY_FLOOR: f64 = 0.3;

/// Whether triangles `after` are acceptable in place of `before`.
fn acceptable(before: &[[[f64; 3]; 3]], after: &[[[f64; 3]; 3]]) -> bool {
    let worst = |ts: &[[[f64; 3]; 3]]| ts.iter().map(tri_quality).fold(f64::INFINITY, f64::min);
    let (b, a) = (worst(before), worst(after));
    a >= QUALITY_FLOOR || a >= b
}

/// The fine surface, frozen as it was before remeshing.
pub struct Reference {
    /// Physical positions of the fine vertices, by the ids they had.
    position: Vec<[f64; 3]>,
    /// Fine faces, with their labels.
    faces: Vec<[u32; 3]>,
    labels: Vec<(u32, u32)>,
    /// Fine faces around each fine vertex.
    incident: Vec<Vec<u32>>,
    /// Each fine curve vertex's neighbours along its curve.
    curve: Vec<Vec<u32>>,
    /// Curvature at each fine vertex, in 1 / physical units.
    curvature: Vec<f64>,
    /// One sample per fine vertex per label whose surface it is on: the unit
    /// the error bound is kept for. A vertex on a wall between two labels is
    /// two samples, one on each label's surface; one on a junction curve, one
    /// per label meeting there.
    samples: Vec<(u32, u32)>,
}

impl Reference {
    pub fn new(rm: &Remesh) -> Reference {
        let n = rm.vertex_count();
        let position: Vec<[f64; 3]> = (0..n as u32).map(|v| rm.physical_position(v)).collect();
        let mut faces = Vec::new();
        let mut labels = Vec::new();
        let mut incident: Vec<Vec<u32>> = vec![Vec::new(); n];
        for f in 0..rm.face_count() as u32 {
            if let Some(t) = rm.face(f) {
                let id = faces.len() as u32;
                faces.push(t);
                labels.push(rm.labels(f));
                for &v in &t {
                    incident[v as usize].push(id);
                }
            }
        }
        let used: Vec<bool> = incident.iter().map(|i| !i.is_empty()).collect();
        let curve: Vec<Vec<u32>> = (0..n as u32)
            .map(|v| {
                if used[v as usize] {
                    rm.curve_neighbours(v)
                } else {
                    Vec::new()
                }
            })
            .collect();

        let normal = |f: u32| {
            let t = faces[f as usize];
            let (a, b, c) = (
                position[t[0] as usize],
                position[t[1] as usize],
                position[t[2] as usize],
            );
            let n = cross(sub(b, a), sub(c, a));
            let l = norm(n);
            if l > 0.0 {
                scale(n, 1.0 / l)
            } else {
                n
            }
        };
        let mut curvature = vec![0.0f64; n];
        for v in 0..n {
            if !used[v] {
                continue;
            }
            let mut k = 0.0f64;
            if curve[v].len() == 2 {
                // Turning angle along the curve over the mean segment length.
                let (a, b) = (
                    position[curve[v][0] as usize],
                    position[curve[v][1] as usize],
                );
                let (d0, d1) = (sub(position[v], a), sub(b, position[v]));
                let (l0, l1) = (norm(d0), norm(d1));
                if l0 > 0.0 && l1 > 0.0 {
                    let c = (dot(d0, d1) / (l0 * l1)).clamp(-1.0, 1.0);
                    k = c.acos() / (0.5 * (l0 + l1));
                }
            } else {
                // Dihedral angle across each wall edge at the vertex, over the
                // edge's length.
                let around = &incident[v];
                for (i, &f) in around.iter().enumerate() {
                    for &g in &around[i + 1..] {
                        if labels[f as usize] != labels[g as usize] {
                            continue;
                        }
                        let tf = faces[f as usize];
                        let shared: Vec<u32> = tf
                            .iter()
                            .copied()
                            .filter(|&w| w != v as u32 && faces[g as usize].contains(&w))
                            .collect();
                        if shared.len() != 1 {
                            continue;
                        }
                        let len = norm(sub(position[shared[0] as usize], position[v]));
                        if len > 0.0 {
                            let c = dot(normal(f), normal(g)).clamp(-1.0, 1.0);
                            k = k.max(c.acos() / len);
                        }
                    }
                }
            }
            curvature[v] = k;
        }
        // Averaged twice over each vertex and its neighbours, so a single
        // noisy dihedral left by the voxel staircase does not set the size.
        for _ in 0..2 {
            let before = curvature.clone();
            for v in 0..n {
                if !used[v] {
                    continue;
                }
                let mut ring: Vec<u32> = incident[v]
                    .iter()
                    .flat_map(|&f| faces[f as usize])
                    .collect();
                ring.sort_unstable();
                ring.dedup();
                let sum: f64 = ring.iter().map(|&w| before[w as usize]).sum();
                curvature[v] = sum / ring.len() as f64;
            }
        }
        let mut samples = Vec::new();
        for v in 0..n as u32 {
            let mut mine: Vec<u32> = incident[v as usize]
                .iter()
                .flat_map(|&f| {
                    let (a, b) = labels[f as usize];
                    [a, b]
                })
                .filter(|&l| l != OUTSIDE)
                .collect();
            mine.sort_unstable();
            mine.dedup();
            samples.extend(mine.into_iter().map(|l| (v, l)));
        }
        Reference {
            position,
            faces,
            labels,
            incident,
            curve,
            curvature,
            samples,
        }
    }

    /// Target edge length at fine vertex `v`.
    fn size(&self, v: u32, p: &LevelParams) -> f64 {
        let (k, e) = (self.curvature[v as usize], p.max_error);
        let h = if k > 0.0 {
            let x = 6.0 * e / k - 3.0 * e * e;
            if x > 0.0 {
                x.sqrt()
            } else {
                p.min_length
            }
        } else {
            p.max_length
        };
        h.clamp(p.min_length, p.max_length)
    }

    fn triangle(&self, f: u32) -> [[f64; 3]; 3] {
        self.faces[f as usize].map(|v| self.position[v as usize])
    }
}

/// Where each fine vertex sits on the coarse surface.
struct Tracking {
    /// Coarse face per sample (see [`Reference::samples`]): always a face
    /// carrying the sample's label.
    face: Vec<u32>,
    /// Samples on each coarse face.
    on_face: Vec<Vec<u32>>,
}

/// What one call did, for tests and benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub splits: usize,
    pub collapses: usize,
    pub flips: usize,
    pub moves: usize,
    /// Operations refused because they would have broken `max_error`.
    pub refused_for_error: usize,
    /// Operations refused because a triangle would have crossed another.
    pub refused_for_crossing: usize,
    /// Small contacts [`level`] separated before remeshing.
    pub separated: usize,
}

struct State<'a> {
    rm: &'a mut Remesh,
    reference: &'a Reference,
    params: LevelParams,
    track: Tracking,
    /// Target length at each coarse vertex, refreshed each round.
    size: Vec<f64>,
    grid: Grid,
    /// Per face, the last query that saw it, to skip repeats without sorting.
    seen: std::cell::RefCell<(u32, Vec<u32>)>,
    /// Vertices of separated copies, and whatever collapses them in: never
    /// projected back onto the fine surface, where the copies coincide.
    parted: Vec<bool>,
    /// How far past a plane a point must be to count as through it.
    tol: f64,
    counts: Counts,
}

/// One level of detail, from the fine wall mesh: [`Remesh::new`], the level's
/// small contacts separated if it asks for that, then [`Reference::new`] and
/// [`remesh`].
pub fn level(
    walls: &WallMesh,
    cells: &CellField,
    resolution: [f64; 3],
    params: &LevelParams,
) -> (Remesh, Counts) {
    let mut rm = Remesh::new(walls, cells, resolution);
    let separated = if params.drop_small_contacts {
        rm.separate_small_contacts(2.0 * params.max_error)
    } else {
        Vec::new()
    };
    let reference = Reference::new(&rm);
    let mut counts = run(&mut rm, &reference, params, &separated);
    counts.separated = separated.len();
    (rm, counts)
}

/// Remesh `rm` in place to one level of detail; see the module docs.
///
/// `reference` must have been made from `rm` before any editing. Small
/// contacts are left alone here whatever the parameters say: separating them
/// changes what the reference is, so [`level`] does it before making one.
pub fn remesh(rm: &mut Remesh, reference: &Reference, params: &LevelParams) -> Counts {
    run(rm, reference, params, &[])
}

fn run(
    rm: &mut Remesh,
    reference: &Reference,
    params: &LevelParams,
    separated: &[Separated],
) -> Counts {
    let mut on_face: Vec<Vec<u32>> = vec![Vec::new(); rm.face_count()];
    let mut face = vec![u32::MAX; reference.samples.len()];
    for (i, &(v, l)) in reference.samples.iter().enumerate() {
        let f = *rm
            .faces_around(v)
            .iter()
            .find(|&&f| {
                let (a, b) = rm.labels(f);
                a == l || b == l
            })
            .expect("a sample's label has a face at it");
        face[i] = f;
        on_face[f as usize].push(i as u32);
    }
    let mut state = State {
        rm,
        reference,
        params: *params,
        track: Tracking { face, on_face },
        size: Vec::new(),
        grid: Grid::default(),
        seen: Default::default(),
        parted: Vec::new(),
        tol: 1e-6 * params.max_length,
        counts: Counts::default(),
    };
    state.rebuild_grid();
    state.part(separated);
    for _ in 0..params.iterations {
        state.refresh_sizes();
        state.split_pass();
        state.refresh_sizes();
        // Edges are longest after the splits and grow with the collapses:
        // cells sized to them here keep queries small for the rest.
        state.rebuild_grid();
        state.collapse_pass();
        state.flip_pass(false);
        state.relax_pass();
        state.flip_pass(true);
    }
    state.counts
}

impl State<'_> {
    fn pos(&self, v: u32) -> [f64; 3] {
        self.rm.physical_position(v)
    }

    fn triangle_with(&self, f: u32, moved: Option<(u32, u32, [f64; 3])>) -> [[f64; 3]; 3] {
        let t = self.rm.face(f).expect("live face");
        t.map(|w| match moved {
            Some((from, _, at)) if w == from => at,
            Some((_, to, at)) if w == to => at,
            _ => self.pos(w),
        })
    }

    /// Index every live face, in cells twice the mean edge length.
    fn rebuild_grid(&mut self) {
        let (mut sum, mut n) = (0.0, 0usize);
        for f in 0..self.rm.face_count() as u32 {
            if let Some(t) = self.rm.face(f) {
                sum += norm(sub(self.pos(t[0]), self.pos(t[1])));
                n += 1;
            }
        }
        let cell = if n > 0 {
            (2.0 * sum / n as f64).max(self.params.min_length)
        } else {
            self.params.max_length
        };
        self.grid = Grid {
            cell,
            ..Default::default()
        };
        for f in 0..self.rm.face_count() as u32 {
            if self.rm.face(f).is_some() {
                let t = self.triangle_with(f, None);
                self.grid.insert(f, &t);
            }
        }
    }

    fn is_parted(&self, v: u32) -> bool {
        self.parted.get(v as usize).copied().unwrap_or(false)
    }

    fn mark_parted(&mut self, v: u32) {
        if self.parted.len() <= v as usize {
            self.parted.resize(v as usize + 1, false);
        }
        self.parted[v as usize] = true;
    }

    /// Put the faces around `v` in the grid where they are now, rebuilding it
    /// once stale entries outnumber live ones a few times over, or once the
    /// faces have grown well past its cells, as they do while collapsing.
    fn reindex(&mut self, v: u32) {
        for f in self.rm.faces_around(v).to_vec() {
            let t = self.triangle_with(f, None);
            self.grid.insert(f, &t);
        }
        let n = self.rm.face_count();
        if self.grid.entries > 3 * n || self.grid.oversized > n / 64 {
            self.rebuild_grid();
        }
    }

    /// How many (triangle, live face) pairs cross, over triangles `tris` (each
    /// with its vertex ids) and every live face other than `skip` sharing no
    /// vertex with it.
    fn crossings(&self, tris: &[([u32; 3], [[f64; 3]; 3])], skip: &[u32]) -> usize {
        if tris.is_empty() {
            return 0;
        }
        // One query for the lot: an operation's triangles overlap, and asking
        // for each separately scans the same cells over and over.
        let boxes: Vec<([f64; 3], [f64; 3])> = tris.iter().map(|(_, t)| bounds(t)).collect();
        let (mut lo, mut hi) = boxes[0];
        for (l, h) in &boxes[1..] {
            for k in 0..3 {
                lo[k] = lo[k].min(l[k]);
                hi[k] = hi[k].max(h[k]);
            }
        }
        let mut near = Vec::new();
        self.grid.near(lo, hi, &mut near);
        let mut seen = self.seen.borrow_mut();
        let (stamp, marks) = &mut *seen;
        if marks.len() < self.rm.face_count() {
            marks.resize(self.rm.face_count(), 0);
        }
        *stamp = stamp.wrapping_add(1);
        if *stamp == 0 {
            marks.iter_mut().for_each(|m| *m = 0);
            *stamp = 1;
        }
        let mut n = 0;
        for &g in &near {
            if marks[g as usize] == *stamp || skip.contains(&g) {
                continue;
            }
            marks[g as usize] = *stamp;
            let Some(gt) = self.rm.face(g) else {
                continue;
            };
            let other = gt.map(|w| self.pos(w));
            let (olo, ohi) = bounds(&other);
            if (0..3).any(|k| ohi[k] < lo[k] || olo[k] > hi[k]) {
                continue;
            }
            for ((ids, t), (l, h)) in tris.iter().zip(&boxes) {
                if (0..3).any(|k| ohi[k] < l[k] || olo[k] > h[k])
                    || gt.iter().any(|w| ids.contains(w))
                {
                    continue;
                }
                if crosses(t, &other, self.tol) {
                    n += 1;
                }
            }
        }
        n
    }

    /// Whether replacing faces `region` with `after` would make more crossings
    /// than there are: no operation may add one, but where the fine surface
    /// already had some, operations there are not all refused.
    fn adds_crossing(&self, region: &[u32], after: &[([u32; 3], [[f64; 3]; 3])]) -> bool {
        let new = self.crossings(after, region);
        if new == 0 {
            return false;
        }
        let before: Vec<([u32; 3], [[f64; 3]; 3])> = region
            .iter()
            .map(|&f| (self.rm.face(f).unwrap(), self.triangle_with(f, None)))
            .collect();
        new > self.crossings(&before, region)
    }

    /// Move each separated patch's two copies a quarter of `max_error` apart,
    /// each into its own label, so the copies no longer coincide and any later
    /// operation that would push one through the other shows as a crossing.
    fn part(&mut self, separated: &[Separated]) {
        let gap = 0.25 * self.params.max_error;
        self.parted = vec![false; self.rm.vertex_count()];
        for s in separated {
            for &(va, vx) in &s.twins {
                self.parted[va as usize] = true;
                self.parted[vx as usize] = true;
            }
        }
        for s in separated {
            let (a, x) = s.labels;
            for &(va, vx) in &s.twins {
                for (v, label) in [(va, a), (vx, x)] {
                    // Outward for `label`, over its faces here.
                    let mut n = [0.0; 3];
                    for &f in self.rm.faces_around(v) {
                        let t = self.triangle_with(f, None);
                        let fnorm = cross(sub(t[1], t[0]), sub(t[2], t[0]));
                        let front = self.rm.labels(f).0 == label;
                        n = add(n, if front { fnorm } else { scale(fnorm, -1.0) });
                    }
                    let l = norm(n);
                    if l > 0.0 && self.try_move(v, sub(self.pos(v), scale(n, gap / l))) {
                        self.counts.moves += 1;
                    }
                }
            }
        }
    }

    /// Target length at each coarse vertex: the mean over the fine vertices on
    /// its faces, or the level's maximum where there are none.
    fn refresh_sizes(&mut self) {
        let n = self.rm.vertex_count();
        self.size = vec![self.params.max_length; n];
        for v in 0..n as u32 {
            if !self.rm.is_used(v) {
                continue;
            }
            // The mean, not the minimum: at a coarse level a vertex's faces
            // cover many fine vertices, and the smallest of them would set
            // every edge there to the finest size anywhere nearby.
            let (mut sum, mut count) = (0.0, 0usize);
            for &f in self.rm.faces_around(v) {
                for &sample in &self.track.on_face[f as usize] {
                    let fine = self.reference.samples[sample as usize].0;
                    sum += self.reference.size(fine, &self.params);
                    count += 1;
                }
            }
            if count > 0 {
                self.size[v as usize] = sum / count as f64;
            }
        }
    }

    fn target(&self, u: u32, v: u32) -> f64 {
        let at = |w: u32| {
            self.size
                .get(w as usize)
                .copied()
                .unwrap_or(self.params.max_length)
        };
        at(u).min(at(v))
    }

    /// Every edge once, from its lower end, in id order.
    fn edges(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for f in 0..self.rm.face_count() as u32 {
            if let Some(t) = self.rm.face(f) {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    out.push((a.min(b), a.max(b)));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Assign the fine vertices on `region` to the nearest of `faces` as they
    /// would be, if all stay within `max_error`. `moved` substitutes a new
    /// position (and, for a collapse, a new vertex) into those faces.
    fn fit(
        &self,
        region: &[u32],
        faces: &[u32],
        moved: Option<(u32, u32, [f64; 3])>,
        tris: Option<&[[[f64; 3]; 3]]>,
    ) -> Option<Vec<(u32, u32)>> {
        let tris: Vec<[[f64; 3]; 3]> = match tris {
            Some(t) => t.to_vec(),
            None => faces
                .iter()
                .map(|&f| self.triangle_with(f, moved))
                .collect(),
        };
        let limit = self.params.max_error * self.params.max_error;
        let labels: Vec<(u32, u32)> = faces.iter().map(|&f| self.rm.labels(f)).collect();
        let mut out = Vec::new();
        for &f in region {
            for &sample in &self.track.on_face[f as usize] {
                let (fine, label) = self.reference.samples[sample as usize];
                let p = self.reference.position[fine as usize];
                // Only this label's triangles count: the bound is on each
                // label's own surface.
                let mut best = (f64::INFINITY, u32::MAX);
                for (i, t) in tris.iter().enumerate() {
                    if labels[i].0 != label && labels[i].1 != label {
                        continue;
                    }
                    let c = closest_on_triangle(p, t[0], t[1], t[2]);
                    let d = sub(c, p);
                    let d2 = dot(d, d);
                    if d2 < best.0 {
                        best = (d2, faces[i]);
                    }
                }
                if best.0 > limit {
                    return None;
                }
                out.push((sample, best.1));
            }
        }
        Some(out)
    }

    fn commit(&mut self, region: &[u32], assignment: Vec<(u32, u32)>) {
        for &f in region {
            if (f as usize) < self.track.on_face.len() {
                self.track.on_face[f as usize].clear();
            }
        }
        if self.track.on_face.len() < self.rm.face_count() {
            self.track.on_face.resize(self.rm.face_count(), Vec::new());
        }
        for (sample, f) in assignment {
            self.track.face[sample as usize] = f;
            self.track.on_face[f as usize].push(sample);
        }
    }

    /// Nearest point on the fine surface to `p`, among the fine faces around
    /// the fine vertices on `region` that lie on the wall `labels`; or on the
    /// fine curve through them, for a curve vertex.
    fn project(&self, p: [f64; 3], region: &[u32], labels: Option<(u32, u32)>) -> [f64; 3] {
        let mut best = (f64::INFINITY, p);
        let mut consider = |c: [f64; 3]| {
            let d = sub(c, p);
            let d2 = dot(d, d);
            if d2 < best.0 {
                best = (d2, c);
            }
        };
        for &f in region {
            for &sample in &self.track.on_face[f as usize] {
                let fine = self.reference.samples[sample as usize].0;
                match labels {
                    Some(wall) => {
                        for &g in &self.reference.incident[fine as usize] {
                            if self.reference.labels[g as usize] == wall {
                                let t = self.reference.triangle(g);
                                consider(closest_on_triangle(p, t[0], t[1], t[2]));
                            }
                        }
                    }
                    None => {
                        let a = self.reference.position[fine as usize];
                        for &w in &self.reference.curve[fine as usize] {
                            consider(closest_on_segment(
                                p,
                                a,
                                self.reference.position[w as usize],
                            ));
                        }
                    }
                }
            }
        }
        best.1
    }

    /// The wall a wall vertex lies on.
    fn wall_of(&self, v: u32) -> (u32, u32) {
        self.rm.labels(self.rm.faces_around(v)[0])
    }

    // --- split ---------------------------------------------------------------

    fn split_pass(&mut self) {
        for (u, v) in self.edges() {
            if !self.rm.is_used(u) || !self.rm.is_used(v) {
                continue;
            }
            let len = norm(sub(self.pos(u), self.pos(v)));
            if len <= 4.0 / 3.0 * self.target(u, v) {
                continue;
            }
            let kind = self.rm.edge_kind(u, v);
            let before = self.rm.edge_faces(u, v);
            let first_new = self.rm.face_count() as u32;
            let Some(m) = self.rm.split(u, v) else {
                continue;
            };
            self.counts.splits += 1;
            self.track.on_face.resize(self.rm.face_count(), Vec::new());
            if self.size.len() < self.rm.vertex_count() {
                self.size
                    .resize(self.rm.vertex_count(), self.params.max_length);
            }
            self.size[m as usize] = self.target(u, v);
            self.reindex(m);
            // The surface has not changed shape: redistribute each split face's
            // fine vertices between its two halves.
            let mut region = before.clone();
            region.extend(first_new..self.rm.face_count() as u32);
            if let Some(a) = self.fit(&region, &region, None, None) {
                self.commit(&region, a);
            }
            if self.is_parted(u) || self.is_parted(v) {
                self.mark_parted(m);
                continue;
            }
            // Pull the new vertex onto the fine surface or curve.
            let mid = self.pos(m);
            let target = match kind {
                EdgeKind::Junction => self.project(mid, &region, None),
                _ => self.project(mid, &region, Some(self.wall_of(m))),
            };
            self.try_move(m, target);
        }
    }

    // --- collapse ------------------------------------------------------------

    fn collapse_pass(&mut self) {
        for (a, b) in self.edges() {
            if !self.rm.is_used(a) || !self.rm.is_used(b) {
                continue;
            }
            let len = norm(sub(self.pos(a), self.pos(b)));
            let target = self.target(a, b);
            if len >= 4.0 / 5.0 * target {
                continue;
            }
            // Collapse the freer vertex into the more constrained one.
            let rank = |k: VertexKind| match k {
                VertexKind::Wall => 0,
                VertexKind::Curve => 1,
                VertexKind::Corner => 2,
                VertexKind::Fixed => 3,
                VertexKind::Locked => 4,
            };
            let (ka, kb) = (self.rm.vertex_kind(a), self.rm.vertex_kind(b));
            let (u, v, ku, kv) = if rank(ka) <= rank(kb) {
                (a, b, ka, kb)
            } else {
                (b, a, kb, ka)
            };
            // Corners may merge into another corner or a fixed vertex along the
            // junction edge between them: clusters of both, joined by edges
            // shorter than a voxel, form where several labels meet inside one
            // voxel, and would otherwise pin every coarser level.
            let movable = matches!(ku, VertexKind::Wall | VertexKind::Curve)
                || (ku == VertexKind::Corner
                    && matches!(kv, VertexKind::Corner | VertexKind::Fixed));
            if !movable {
                continue;
            }
            // Where both may move, meet in the middle, on the surface.
            let mut region: Vec<u32> = self.rm.faces_around(u).to_vec();
            region.extend_from_slice(self.rm.faces_around(v));
            region.sort_unstable();
            region.dedup();
            // (A corner merging into another stays where the other is.)
            let parted = self.is_parted(u) || self.is_parted(v);
            let at = if ku == kv && ku != VertexKind::Corner {
                let mid = scale(add(self.pos(u), self.pos(v)), 0.5);
                if parted {
                    mid
                } else if ku == VertexKind::Wall {
                    self.project(mid, &region, Some(self.wall_of(u)))
                } else {
                    self.project(mid, &region, None)
                }
            } else {
                self.pos(v)
            };
            // Botsch & Kobbelt: never create an edge that would itself be split.
            // Except to remove an edge already shorter than the level's
            // minimum, which should go whatever it leaves.
            let long =
                len >= self.params.min_length
                    && self.rm.neighbours(u).into_iter().any(|w| {
                        w != v && norm(sub(at, self.pos(w))) > 4.0 / 3.0 * self.target(v, w)
                    });
            if long {
                continue;
            }
            let dying = self.rm.edge_faces(u, v);
            let survivors: Vec<u32> = region
                .iter()
                .copied()
                .filter(|f| !dying.contains(f))
                .collect();
            let before: Vec<_> = region
                .iter()
                .map(|&f| self.triangle_with(f, None))
                .collect();
            let after: Vec<_> = survivors
                .iter()
                .map(|&f| self.triangle_with(f, Some((u, v, at))))
                .collect();
            if !acceptable(&before, &after) {
                continue;
            }
            let Some(assignment) = self.fit(&region, &survivors, Some((u, v, at)), None) else {
                self.counts.refused_for_error += 1;
                continue;
            };
            let replaced: Vec<([u32; 3], [[f64; 3]; 3])> = survivors
                .iter()
                .zip(&after)
                .map(|(&f, t)| {
                    let ids = self.rm.face(f).unwrap().map(|w| if w == u { v } else { w });
                    (ids, *t)
                })
                .collect();
            if self.adds_crossing(&region, &replaced) {
                self.counts.refused_for_crossing += 1;
                continue;
            }
            if self.rm.collapse(u, v, self.rm.to_voxel(at)) {
                self.counts.collapses += 1;
                self.commit(&region, assignment);
                self.reindex(v);
                if parted {
                    self.mark_parted(v);
                }
            }
        }
    }

    // --- flip ----------------------------------------------------------------

    /// Flip wall edges toward degree 6 or, with `for_quality`, where that
    /// makes the worse of the two triangles better.
    fn flip_pass(&mut self, for_quality: bool) {
        let ideal = |rm: &Remesh, w: u32| -> Option<i64> {
            (rm.vertex_kind(w) == VertexKind::Wall).then_some(6)
        };
        for (u, v) in self.edges() {
            if !self.rm.is_used(u)
                || !self.rm.is_used(v)
                || self.rm.edge_kind(u, v) != EdgeKind::Wall
            {
                continue;
            }
            let faces = self.rm.edge_faces(u, v);
            let opposite = |f: u32| {
                self.rm
                    .face(f)
                    .unwrap()
                    .into_iter()
                    .find(|&w| w != u && w != v)
                    .unwrap()
            };
            let (p, q) = (opposite(faces[0]), opposite(faces[1]));
            // Valence before and after, over the vertices that have an ideal.
            let deg = |w: u32| self.rm.neighbours(w).len() as i64;
            let cost = |d: [i64; 4]| -> i64 {
                [u, v, p, q]
                    .iter()
                    .zip(d)
                    .filter_map(|(&w, d)| ideal(self.rm, w).map(|i| (d - i) * (d - i)))
                    .sum()
            };
            let (du, dv, dp, dq) = (deg(u), deg(v), deg(p), deg(q));
            if !for_quality && cost([du - 1, dv - 1, dp + 1, dq + 1]) >= cost([du, dv, dp, dq]) {
                continue;
            }
            // Only nearly flat pairs, so the flip barely changes the shape.
            let (pu, pv, pp, pq) = (self.pos(u), self.pos(v), self.pos(p), self.pos(q));
            let na = cross(sub(pv, pu), sub(pp, pu));
            let nb = cross(sub(pu, pv), sub(pq, pv));
            if dot(na, nb) < (20f64).to_radians().cos() * norm(na) * norm(nb) {
                continue;
            }
            let tris = [[pp, pu, pq], [pq, pv, pp]];
            let old = [[pu, pv, pp], [pv, pu, pq]];
            let worst =
                |ts: &[[[f64; 3]; 3]]| ts.iter().map(tri_quality).fold(f64::INFINITY, f64::min);
            if for_quality {
                if worst(&tris) <= worst(&old) * 1.001 {
                    continue;
                }
            } else if !acceptable(&old, &tris) {
                continue;
            }
            let (fa, fb) = if self.rm.face(faces[0]).unwrap().contains(&p) {
                (faces[0], faces[1])
            } else {
                (faces[1], faces[0])
            };
            // Checked against the pair as it would be. Which of the two face
            // ids ends up holding which triangle is the flip's own choice, so the
            // samples are assigned again from the faces as they actually are.
            // The two cover the same ground either way, so this cannot fail.
            if self.fit(&[fa, fb], &[fa, fb], None, Some(&tris)).is_none() {
                self.counts.refused_for_error += 1;
                continue;
            }
            if self.adds_crossing(&[fa, fb], &[([p, u, q], tris[0]), ([q, v, p], tris[1])]) {
                self.counts.refused_for_crossing += 1;
                continue;
            }
            if self.rm.flip(u, v) {
                self.counts.flips += 1;
                for f in [fa, fb] {
                    let t = self.triangle_with(f, None);
                    self.grid.insert(f, &t);
                }
                let assignment = self
                    .fit(&[fa, fb], &[fa, fb], None, None)
                    .expect("a flip keeps the pair's ground");
                self.commit(&[fa, fb], assignment);
            }
        }
    }

    // --- relax ---------------------------------------------------------------

    fn relax_pass(&mut self) {
        for v in 0..self.rm.vertex_count() as u32 {
            if !self.rm.is_used(v) {
                continue;
            }
            let kind = self.rm.vertex_kind(v);
            let p = self.pos(v);
            let target = match kind {
                VertexKind::Wall => {
                    // Area-weighted centroid of the neighbours' faces, moved
                    // along the tangent plane only.
                    let mut c = [0.0; 3];
                    let mut n = [0.0; 3];
                    let mut total = 0.0;
                    for &f in self.rm.faces_around(v) {
                        let t = self.triangle_with(f, None);
                        let fnorm = cross(sub(t[1], t[0]), sub(t[2], t[0]));
                        let area = 0.5 * norm(fnorm);
                        let centroid = scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0);
                        c = add(c, scale(centroid, area));
                        n = add(n, fnorm);
                        total += area;
                    }
                    if total <= 0.0 {
                        continue;
                    }
                    let c = scale(c, 1.0 / total);
                    let ln = norm(n);
                    if ln <= 0.0 {
                        continue;
                    }
                    let n = scale(n, 1.0 / ln);
                    let d = sub(c, p);
                    let tangent = sub(d, scale(n, dot(d, n)));
                    let moved = add(p, scale(tangent, 0.5));
                    if self.is_parted(v) {
                        moved
                    } else {
                        let region = self.rm.faces_around(v).to_vec();
                        self.project(moved, &region, Some(self.wall_of(v)))
                    }
                }
                VertexKind::Curve => {
                    let ends = self.rm.curve_neighbours(v);
                    if ends.len() != 2 {
                        continue;
                    }
                    let mid = scale(add(self.pos(ends[0]), self.pos(ends[1])), 0.5);
                    let moved = add(p, scale(sub(mid, p), 0.5));
                    let region = self.rm.faces_around(v).to_vec();
                    self.project(moved, &region, None)
                }
                _ => continue,
            };
            if self.try_move(v, target) {
                self.counts.moves += 1;
            }
        }
    }

    fn try_move(&mut self, v: u32, to: [f64; 3]) -> bool {
        let region = self.rm.faces_around(v).to_vec();
        let before: Vec<_> = region
            .iter()
            .map(|&f| self.triangle_with(f, None))
            .collect();
        let after: Vec<_> = region
            .iter()
            .map(|&f| self.triangle_with(f, Some((v, v, to))))
            .collect();
        if !acceptable(&before, &after) {
            return false;
        }
        let Some(assignment) = self.fit(&region, &region, Some((v, v, to)), None) else {
            self.counts.refused_for_error += 1;
            return false;
        };
        let moved: Vec<([u32; 3], [[f64; 3]; 3])> = region
            .iter()
            .zip(&after)
            .map(|(&f, t)| (self.rm.face(f).unwrap(), *t))
            .collect();
        if self.adds_crossing(&region, &moved) {
            self.counts.refused_for_crossing += 1;
            return false;
        }
        if self.rm.relocate(v, self.rm.to_voxel(to)) {
            self.commit(&region, assignment);
            self.reindex(v);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{extract, Extraction};
    use crate::grid::VolumeView;
    use crate::mesh::{MeshOptions, TriangleMesh};
    use crate::smooth::{fair, scatter, Fairing};
    use crate::walls::WallMesh;
    use ndarray::Array3;
    use rustc_hash::{FxHashMap, FxHashSet};

    fn faired(a: &Array3<u32>, close: bool) -> Extraction {
        let mut e = extract(&VolumeView::new(a.view(), close));
        let p = Fairing {
            iterations: 20,
            pass_band: Some(0.1),
            ..Default::default()
        };
        fair(&mut e.cells, &p, false);
        for m in e.meshes.iter_mut() {
            scatter(&e.cells, m);
        }
        e
    }

    fn sphere(r: f64) -> Array3<u32> {
        let n = (2.0 * r) as usize + 8;
        let c = (n - 1) as f64 / 2.0;
        Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            let d = (i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2);
            u32::from(d <= r * r)
        })
    }

    /// A sphere split by a slanted plane into labels 1 and 2, inside label 3.
    fn three_labels(r: f64) -> Array3<u32> {
        let n = (2.0 * r) as usize + 8;
        let c = (n - 1) as f64 / 2.0;
        Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
            if x * x + y * y + z * z > r * r {
                3
            } else if x + 0.6 * y + 0.3 * z < 0.0 {
                1
            } else {
                2
            }
        })
    }

    /// Euler characteristic, after checking every directed edge is used once.
    fn euler(m: &TriangleMesh) -> i64 {
        let mut directed: FxHashSet<(u32, u32)> = FxHashSet::default();
        for f in &m.faces {
            assert!(
                f[0] != f[1] && f[1] != f[2] && f[0] != f[2],
                "degenerate face"
            );
            for k in 0..3 {
                assert!(
                    directed.insert((f[k], f[(k + 1) % 3])),
                    "directed edge used twice"
                );
            }
        }
        for &(a, b) in &directed {
            assert!(directed.contains(&(b, a)), "open edge in a closed surface");
        }
        let edges = directed.len() / 2;
        m.vertices.len() as i64 - edges as i64 + m.faces.len() as i64
    }

    fn quality(m: &TriangleMesh) -> (f64, f64) {
        let mut sum = 0.0;
        let mut worst = f64::INFINITY;
        for f in &m.faces {
            let p = f.map(|v| m.vertices[v as usize].map(|x| x as f64));
            let (ab, bc, ca) = (sub(p[1], p[0]), sub(p[2], p[1]), sub(p[0], p[2]));
            let area2 = norm(cross(ab, sub(p[2], p[0])));
            let l2 = dot(ab, ab) + dot(bc, bc) + dot(ca, ca);
            let q = 2.0 * 3f64.sqrt() * area2 / l2;
            sum += q;
            worst = worst.min(q);
        }
        (sum / m.faces.len() as f64, worst)
    }

    /// Largest distance from any fine vertex to the coarse surface, by brute
    /// force over every coarse triangle.
    fn max_distance(fine: &TriangleMesh, coarse: &TriangleMesh) -> f64 {
        let tris: Vec<[[f64; 3]; 3]> = coarse
            .faces
            .iter()
            .map(|f| f.map(|v| coarse.vertices[v as usize].map(|x| x as f64)))
            .collect();
        fine.vertices
            .iter()
            .map(|v| {
                let p = v.map(|x| x as f64);
                tris.iter()
                    .map(|t| norm(sub(closest_on_triangle(p, t[0], t[1], t[2]), p)))
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max)
    }

    fn run(
        a: &Array3<u32>,
        close: bool,
        params: &LevelParams,
    ) -> (Extraction, Remesh, Remesh, Counts) {
        let e = faired(a, close);
        let walls = WallMesh::build(&e).unwrap();
        let fine = Remesh::new(&walls, &e.cells, [1.0; 3]);
        let mut rm = Remesh::new(&walls, &e.cells, [1.0; 3]);
        let reference = Reference::new(&rm);
        let counts = remesh(&mut rm, &reference, params);
        (e, fine, rm, counts)
    }

    fn opts(a: &Array3<u32>) -> MeshOptions {
        MeshOptions {
            shape: [a.shape()[0], a.shape()[1], a.shape()[2]],
            ..Default::default()
        }
    }

    const COARSE: LevelParams = LevelParams {
        max_length: 4.0,
        min_length: 0.5,
        max_error: 0.25,
        iterations: 5,
        drop_small_contacts: false,
    };

    #[test]
    fn a_sphere_gets_coarser_within_the_error_bound() {
        let a = sphere(14.0);
        let (_, fine, rm, counts) = run(&a, true, &COARSE);
        let o = opts(&a);
        let (before, after) = (fine.label_mesh(0, &o), rm.label_mesh(0, &o));
        assert!(counts.collapses > 0 && counts.moves > 0, "{counts:?}");
        assert!(
            after.faces.len() * 3 < before.faces.len(),
            "{} -> {} faces",
            before.faces.len(),
            after.faces.len()
        );
        assert_eq!(euler(&after), 2);
        let err = max_distance(&before, &after);
        assert!(
            err <= COARSE.max_error + 1e-4,
            "fine vertex {err} from the coarse surface"
        );
        let (mean, worst) = quality(&after);
        assert!(mean > 0.85, "mean quality {mean}");
        assert!(worst > 0.3, "worst quality {worst}");
    }

    #[test]
    fn every_label_keeps_its_topology_and_walls_still_meet() {
        let a = three_labels(12.0);
        let (e, fine, rm, _) = run(&a, true, &COARSE);
        let o = opts(&a);
        let keys = |m: &TriangleMesh| -> FxHashSet<[[u32; 3]; 3]> {
            m.faces
                .iter()
                .map(|f| {
                    let mut t = f.map(|v| m.vertices[v as usize].map(f32::to_bits));
                    t.sort_unstable();
                    t
                })
                .collect()
        };
        let mut meshes = Vec::new();
        for s in 0..e.meshes.len() as u32 {
            let (before, after) = (fine.label_mesh(s, &o), rm.label_mesh(s, &o));
            assert_eq!(euler(&after), euler(&before), "label slot {s}");
            assert!(after.faces.len() < before.faces.len());
            let err = max_distance(&before, &after);
            assert!(err <= COARSE.max_error + 1e-4, "slot {s}: {err}");
            meshes.push(after);
        }
        // Labels 1 and 2 share a wall: some triangles in both, and every
        // triangle of each is shared with one of the other two labels.
        let k: Vec<_> = meshes.iter().map(keys).collect();
        assert!(!k[0].is_disjoint(&k[1]));
        for (i, ki) in k.iter().enumerate() {
            let others: FxHashSet<_> = k
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .flat_map(|(_, kj)| kj.iter().copied())
                .collect();
            let outer = i == 2; // label 3's outside faces the volume's edge
            if !outer {
                assert!(
                    ki.is_subset(&others),
                    "label slot {i} has a triangle nobody shares"
                );
            }
        }
    }

    #[test]
    fn locked_vertices_and_their_triangles_do_not_change() {
        // Cut in half, so the volume's edge crosses every label.
        let whole = three_labels(10.0);
        let half = whole.shape()[0] / 2;
        let a = whole.slice(ndarray::s![..half, .., ..]).to_owned();
        let (_, fine, rm, _) = run(&a, false, &COARSE);
        let mut checked = 0;
        for v in 0..fine.vertex_count() as u32 {
            if fine.is_used(v) && fine.vertex_kind(v) == VertexKind::Locked {
                assert_eq!(rm.position(v), fine.position(v));
                let tri = |m: &Remesh| -> Vec<[u32; 3]> {
                    let mut t: Vec<[u32; 3]> = m
                        .faces_around(v)
                        .iter()
                        .map(|&f| m.face(f).unwrap())
                        .collect();
                    t.sort_unstable();
                    t
                };
                assert_eq!(tri(&rm), tri(&fine));
                checked += 1;
            }
        }
        assert!(checked > 20, "only {checked} locked vertices");
    }

    /// Overlapping blobs with a little voxel noise: many labels, junctions and
    /// thin pieces, which is where a bookkeeping slip shows as a broken bound.
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

    /// The bound is on each label's own surface: every fine vertex of every
    /// label within `max_error` of that label's coarse surface, measured by
    /// brute force. Swapped face ids after a flip once broke this at four
    /// times the bound, on real data, while the smooth fixtures above passed.
    #[test]
    fn the_bound_holds_for_every_label_of_a_noisy_volume() {
        let params = LevelParams {
            max_length: 3.0,
            min_length: 0.5,
            max_error: 0.15,
            iterations: 4,
            drop_small_contacts: false,
        };
        for seed in 0..3 {
            let a = noisy(22, 8, seed);
            let (e, fine, rm, counts) = run(&a, true, &params);
            assert!(counts.flips > 100 && counts.collapses > 100, "{counts:?}");
            let o = opts(&a);
            for s in 0..e.meshes.len() as u32 {
                let (before, after) = (fine.label_mesh(s, &o), rm.label_mesh(s, &o));
                if after.faces.is_empty() {
                    continue;
                }
                let err = max_distance(&before, &after);
                assert!(
                    err <= params.max_error + 1e-4,
                    "seed {seed} slot {s}: {err}"
                );
                assert_eq!(euler(&after), euler(&before), "seed {seed} slot {s}");
            }
        }
    }

    /// Dropping small contacts changes which labels touch, but no label's own
    /// surface loses its bound, and only the material left between a
    /// separated pair changes topology (a hole through it closes).
    #[test]
    fn dropping_small_contacts_keeps_every_bound() {
        let params = LevelParams {
            max_length: 3.0,
            min_length: 0.5,
            max_error: 0.6,
            iterations: 3,
            drop_small_contacts: true,
        };
        let mut separated = 0;
        for seed in 0..4 {
            let a = noisy(22, 8, seed);
            let e = faired(&a, true);
            let walls = WallMesh::build(&e).unwrap();
            let fine = Remesh::new(&walls, &e.cells, [1.0; 3]);
            let mut rm = Remesh::new(&walls, &e.cells, [1.0; 3]);
            let done = rm.separate_small_contacts(2.0 * params.max_error);
            separated += done.len();
            let reference = Reference::new(&rm);
            let before = crossing_pairs(&fine);
            super::run(&mut rm, &reference, &params, &done);
            let after = crossing_pairs(&rm);
            assert!(
                after <= before,
                "seed {seed}: {before} -> {after} crossings"
            );
            let o = opts(&a);
            for s in 0..e.meshes.len() as u32 {
                let (before, after) = (fine.label_mesh(s, &o), rm.label_mesh(s, &o));
                if after.faces.is_empty() {
                    continue;
                }
                let err = max_distance(&before, &after);
                assert!(
                    err <= params.max_error + 1e-4,
                    "seed {seed} slot {s}: {err}"
                );
                let between = done.iter().filter(|d| d.between == s).count() as i64;
                assert_eq!(
                    euler(&after),
                    euler(&before) + 2 * between,
                    "seed {seed} slot {s}"
                );
            }
        }
        assert!(separated > 0, "nothing was separated");
    }

    /// Pairs of live faces sharing no vertex that cross, through the grid.
    fn crossing_pairs(rm: &Remesh) -> usize {
        let pos = |f: u32| rm.face(f).unwrap().map(|v| rm.physical_position(v));
        let mut grid = Grid {
            cell: 2.0,
            ..Default::default()
        };
        let live: Vec<u32> = (0..rm.face_count() as u32)
            .filter(|&f| rm.face(f).is_some())
            .collect();
        for &f in &live {
            grid.insert(f, &pos(f));
        }
        let mut n = 0;
        let mut near = Vec::new();
        for &f in &live {
            let (t, ids) = (pos(f), rm.face(f).unwrap());
            let (lo, hi) = bounds(&t);
            near.clear();
            grid.near(lo, hi, &mut near);
            near.sort_unstable();
            near.dedup();
            for &g in &near {
                if g <= f || rm.face(g).unwrap().iter().any(|w| ids.contains(w)) {
                    continue;
                }
                if crosses(&t, &pos(g), 1e-6) {
                    n += 1;
                }
            }
        }
        n
    }

    /// No level adds a crossing: a coarse triangle passing through another
    /// wall, which each label's error bound alone would allow across any gap
    /// thinner than `max_error`.
    #[test]
    fn remeshing_adds_no_crossings() {
        let params = LevelParams {
            max_length: 4.0,
            min_length: 0.5,
            max_error: 0.6,
            iterations: 4,
            drop_small_contacts: false,
        };
        let mut refused = 0;
        for seed in 0..4 {
            let a = noisy(22, 8, seed);
            let (_, fine, rm, counts) = run(&a, true, &params);
            refused += counts.refused_for_crossing;
            let (before, after) = (crossing_pairs(&fine), crossing_pairs(&rm));
            assert!(
                after <= before,
                "seed {seed}: {before} -> {after} crossings"
            );
        }
        assert!(refused > 0, "the check never had anything to refuse");
    }

    #[test]
    fn it_is_deterministic() {
        let a = three_labels(10.0);
        let (e, _, x, _) = run(&a, true, &COARSE);
        let (_, _, y, _) = run(&a, true, &COARSE);
        let o = opts(&a);
        for s in 0..e.meshes.len() as u32 {
            let (mx, my) = (x.label_mesh(s, &o), y.label_mesh(s, &o));
            assert_eq!(mx.vertices, my.vertices);
            assert_eq!(mx.faces, my.faces);
        }
    }

    #[test]
    fn a_tighter_bound_keeps_more_faces() {
        let a = sphere(14.0);
        let loose = run(&a, true, &COARSE).2.live_faces();
        let tight = run(
            &a,
            true,
            &LevelParams {
                max_error: 0.05,
                ..COARSE
            },
        )
        .2
        .live_faces();
        assert!(tight > loose, "{tight} vs {loose}");
        let _ = FxHashMap::<u32, u32>::default();
    }
}
