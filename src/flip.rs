//! Edge flips on one label's triangulated surface, consistent across labels.
//!
//! A quad is split into two triangles along its shorter diagonal (see
//! [`crate::mesh`]). That choice is local to the quad: it never looks across a
//! quad edge, so it cannot fix a thin triangle whose better partner sits in the
//! next quad over. Flipping edges can. An edge shared by triangles `(u, v, p)`
//! and `(v, u, q)` is replaced by `(p, q)` when that makes the worse of the two
//! triangles better. This is the flip step of isotropic remeshing (Botsch &
//! Kobbelt 2004). It moves no vertex, so positions, seams and bounds are
//! untouched: only connectivity changes.
//!
//! # Staying consistent across labels
//!
//! A wall between labels `a` and `b` appears in both labels' meshes, and each
//! mesh is flipped on its own, possibly on different threads, possibly never
//! both. The two copies must still come out identical, or the surfaces stop
//! being coincident and the segmentation stops being a partition of space.
//! Three rules give that, following Faraj et al.'s (2016) multi-material rule
//! that an operation must not change which materials an element separates:
//!
//! * **Only edges inside one wall are flipped.** An edge is a candidate only if
//!   one of its endpoints is a *sheet* vertex ([`crate::tables::SHEET_CELL`]):
//!   one wall between two labels passes through its cell, so every triangle
//!   around it belongs to that wall, in both labels' meshes. An edge along a
//!   junction curve, where the two triangles belong to different walls, never
//!   qualifies, because junction cells are not sheet cells.
//! * **Every decision reads only that wall.** The quality test uses the four
//!   positions, which both copies share bit for bit, and is evaluated with the
//!   four vertices in a canonical order, so rounding cannot tell the copies
//!   apart either. The test that the new edge `(p, q)` does not already exist
//!   also requires a sheet vertex among `p`, `q`: every edge at a sheet vertex
//!   is in its wall, so both copies give the same answer.
//! * **Each pass is independent of storage order.** Every edge is judged on
//!   the same surface, and the edges flipped are those that beat every
//!   neighbour they share a triangle with, ties broken by their cells. The
//!   walls of a mesh do not interact under these rules, so each wall's flips are
//!   the same in every mesh containing it, whatever else that mesh holds and
//!   however its faces are numbered.
//!
//! Vertices that are pinned (the chunk's outermost cell layer), or that come
//! from a cell with an ambiguous face (the only ones the manifold repair in
//! [`crate::mesh`] can split), never take part in a flip. Pinning keeps every
//! triangle a neighbouring chunk also sees exactly as that chunk sees it, so
//! chunks still stitch. Both are properties of the cell, not the label, so both
//! copies of a wall agree on them.
//!
//! # Keeping the shape
//!
//! Flipping a non-planar pair of triangles changes the surface between them by
//! the tetrahedron the four points span. Only pairs within [`MAX_HINGE_DEGREES`]
//! of planar, before and after, are flipped. The pair must also be convex,
//! with `p`, `q` on opposite sides of `(u, v)` and `u`, `v` on opposite sides
//! of `(p, q)`, so the flip cannot fold the surface over.

/// How far from planar a pair of triangles may be, measured as the angle
/// between their normals, for the edge between them to be flipped. Measured
/// before and after the flip.
pub const MAX_HINGE_DEGREES: f64 = 20.0;

/// Relative improvement in the worse triangle's quality a flip must buy.
///
/// A margin rather than zero, so a pair already as good either way does not
/// flip back and forth with rounding.
const MIN_GAIN: f64 = 1e-3;

/// Shape quality above which a pair of triangles is not considered for a flip:
/// if the worse of the two is already this good, the flip test is skipped.
const GOOD_QUALITY: f32 = 0.7;

/// Vertex flag: on a single wall between two labels.
const SHEET: u8 = 1;
/// Vertex flag: must not take part in any flip.
const FROZEN: u8 = 2;

/// Twin of a half-edge with no twin: the surface's edge.
const NONE: u32 = u32::MAX;
/// Twin of a half-edge on an edge used by more than two faces, which is never
/// flipped. Below [`NONE`], so `>= MANY` catches both.
const MANY: u32 = u32::MAX - 1;

#[cfg(test)]
#[inline]
fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Shape quality of a triangle: 4*sqrt(3)*area / (sum of squared edges), which
/// is 1 for an equilateral triangle and 0 for a degenerate one. The flip test
/// works with its square; this is for the tests.
#[cfg(test)]
fn quality(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let (ab, bc, ca) = (sub(b, a), sub(c, b), sub(a, c));
    let n = cross(ab, sub(c, a));
    let l2 = dot(ab, ab) + dot(bc, bc) + dot(ca, ca);
    if l2 > 0.0 {
        2.0 * 3f64.sqrt() * dot(n, n).sqrt() / l2
    } else {
        0.0
    }
}

/// How much swapping the diagonal `(u, v)` of the quad `u, p, v, q` for `(p, q)`
/// improves the worse triangle, as a fraction; zero if the swap is unsafe or
/// not worth making.
///
/// Symmetric in `u` and `v`, and in `p` and `q`: the caller passes them in a
/// canonical order, so every mesh holding this pair computes the same bits.
fn improvement(u: [f64; 3], v: [f64; 3], p: [f64; 3], q: [f64; 3], cos_hinge: f64) -> f64 {
    // Normals of the two triangles on each diagonal, both taken with the
    // diagonal as the first edge. For a flat, convex quad each pair points in
    // opposite directions, which is what the sign tests check. Each normal's
    // length is twice its triangle's area, so they give the qualities too.
    //
    // Everything is compared squared: this runs for every edge of every pass,
    // and the square roots were most of its cost. One is taken at the end,
    // and only for a flip that is going ahead.
    let uv = sub(v, u);
    let (up, uq) = (sub(p, u), sub(q, u));
    let (n_p, n_q) = (cross(uv, up), cross(uv, uq));
    let pq = sub(q, p);
    let pv = sub(v, p);
    let (m_u, m_v) = (cross(pq, sub(u, p)), cross(pq, pv));
    let cos2 = cos_hinge * cos_hinge;
    let near_flat = |a: [f64; 3], b: [f64; 3], aa: f64, bb: f64| {
        let ab = dot(a, b);
        aa > 0.0 && bb > 0.0 && ab < 0.0 && ab * ab >= cos2 * aa * bb
    };
    let (np2, nq2, mu2, mv2) = (dot(n_p, n_p), dot(n_q, n_q), dot(m_u, m_u), dot(m_v, m_v));
    if !near_flat(n_p, n_q, np2, nq2) || !near_flat(m_u, m_v, mu2, mv2) {
        return 0.0;
    }
    // Squared edge lengths, each shared by two of the four triangles.
    let vq = sub(q, v);
    let (e_uv, e_up, e_uq) = (dot(uv, uv), dot(up, up), dot(uq, uq));
    let (e_vp, e_vq, e_pq) = (dot(pv, pv), dot(vq, vq), dot(pq, pq));
    // quality^2 = 12 |n|^2 / (sum of squared edges)^2; the 12 cancels.
    let q2 = |n2: f64, l2: f64| n2 / (l2 * l2);
    let before = q2(np2, e_uv + e_up + e_vp).min(q2(nq2, e_uv + e_uq + e_vq));
    let after = q2(mu2, e_pq + e_up + e_uq).min(q2(mv2, e_pq + e_vp + e_vq));
    let margin = (1.0 + MIN_GAIN) * (1.0 + MIN_GAIN);
    if after > before * margin {
        (after / before).sqrt() - 1.0
    } else {
        0.0
    }
}

/// What flipping needs to know about each vertex.
///
/// A trait rather than arrays, so nothing has to be materialised per vertex:
/// [`crate::mesh`] answers each question from the cell tables as it is asked.
pub trait VertexInfo {
    /// How many vertices there are; ids run from zero.
    fn count(&self) -> usize;
    /// Position in physical space.
    fn position(&self, v: u32) -> [f64; 3];
    /// A canonical id shared by every copy of this vertex across labels: its
    /// cell.
    fn key(&self, v: u32) -> u32;
    /// On a single wall between two labels.
    fn sheet(&self, v: u32) -> bool;
    /// Must not take part in any flip.
    fn frozen(&self, v: u32) -> bool;
}

/// What judging an edge needs of each vertex, in one place.
#[derive(Clone, Copy, Default)]
struct Vertex {
    position: [f64; 3],
    key: u32,
    /// [`SHEET`] and [`FROZEN`] bits.
    flags: u8,
}

/// Working memory, kept per thread across calls.
///
/// Flipping allocates a few words per half-edge, and each call is one object.
/// Freed and reallocated per call, those pages are returned to the system and
/// faulted back in every time, and zeroing them was the largest single cost of
/// flipping. Kept, they are faulted once per thread.
#[derive(Default)]
struct Scratch {
    twin: Vec<u32>,
    gain: Vec<f32>,
    out: Vec<u32>,
    start: Vec<u32>,
    /// Half-edges bucketed by origin, as (destination, half-edge).
    by_origin: Vec<(u32, u32)>,
    /// Faces rewritten by the last pass, whose edges need judging again.
    dirty: Vec<bool>,
    vertex: Vec<Vertex>,
    /// Per face: squared shape quality.
    quality: Vec<f32>,
    chosen: Vec<(u64, u32)>,
    keep: Vec<bool>,
    stamp: Vec<u32>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// Past this many half-edges the scratch is released after use rather than
/// kept, so one huge object does not pin its memory for the thread's lifetime.
const KEEP_SCRATCH_UP_TO: usize = 1 << 25;

/// Flip edges in `faces` for up to `passes` passes. Returns how many flips
/// were made.
///
/// Each pass is Jacobi-style, so its result cannot depend on the order faces
/// happen to be stored in, which differs between the labels sharing a wall:
///
/// 1. every interior edge's gain is evaluated on the current surface;
/// 2. an edge is chosen if its gain beats that of every edge it shares a
///    triangle with, so the chosen edges share no triangle;
/// 3. the chosen flips are applied, in any order, since none touches another's
///    triangles.
///
/// Two chosen flips could still create the same new edge, from two quads that
/// share both opposite vertices; both are then dropped. Stops early once a pass
/// flips nothing.
pub fn flip_edges<V: VertexInfo>(faces: &mut [[u32; 3]], verts: &V, passes: u32) -> usize {
    if passes == 0 || faces.is_empty() {
        return 0;
    }
    SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let flipped = run(faces, verts, passes, &mut scratch);
        if 3 * faces.len() > KEEP_SCRATCH_UP_TO {
            *scratch = Scratch::default();
        }
        flipped
    })
}

fn run<V: VertexInfo>(faces: &mut [[u32; 3]], verts: &V, passes: u32, s: &mut Scratch) -> usize {
    let cos_hinge = MAX_HINGE_DEGREES.to_radians().cos();
    let nh = faces.len() * 3;
    let nv = verts.count();

    let head = |faces: &[[u32; 3]], h: usize| faces[h / 3][h % 3];
    let next = |h: usize| h - h % 3 + (h + 1) % 3;
    let prev = |h: usize| h - h % 3 + (h + 2) % 3;

    // --- twins ---------------------------------------------------------------
    // Half-edge `3f + k` runs from `faces[f][k]` to `faces[f][(k + 1) % 3]`.
    // Bucket the half-edges by origin with their destination alongside (a
    // counting sort, linear), then find each one's twin in the short list
    // leaving its far end -- without touching `faces` again, which is what
    // made this slow. An edge used other than once in each direction, by
    // three faces or twice the same way, gets no twin and is never flipped.
    s.start.clear();
    s.start.resize(nv + 1, 0);
    for f in faces.iter() {
        for &v in f {
            s.start[v as usize + 1] += 1;
        }
    }
    for v in 0..nv {
        s.start[v + 1] += s.start[v];
    }
    s.by_origin.clear();
    s.by_origin.resize(nh, (0, 0));
    {
        // `out` doubles as the fill cursor here, then is reset below.
        s.out.clear();
        s.out.extend_from_slice(&s.start[..nv]);
        for (f, tri) in faces.iter().enumerate() {
            for k in 0..3 {
                let v = tri[k] as usize;
                s.by_origin[s.out[v] as usize] = (tri[(k + 1) % 3], (3 * f + k) as u32);
                s.out[v] += 1;
            }
        }
    }
    s.twin.clear();
    s.twin.resize(nh, NONE);
    for a in 0..nv {
        let mine = &s.by_origin[s.start[a] as usize..s.start[a + 1] as usize];
        for (i, &(b, h)) in mine.iter().enumerate() {
            // Each edge once, from its lower end.
            if (b as usize) < a {
                continue;
            }
            let dup = mine.iter().enumerate().any(|(j, &(c, _))| j != i && c == b);
            let theirs =
                &s.by_origin[s.start[b as usize] as usize..s.start[b as usize + 1] as usize];
            let mut back = theirs.iter().filter(|&&(c, _)| c == a as u32);
            let (first, second) = (back.next(), back.next());
            match (dup, first, second) {
                (false, Some(&(_, t)), None) => {
                    s.twin[h as usize] = t;
                    s.twin[t as usize] = h;
                }
                (false, None, _) => {}
                _ => {
                    s.twin[h as usize] = MANY;
                    for &(_, t) in theirs.iter().filter(|&&(c, _)| c == a as u32) {
                        s.twin[t as usize] = MANY;
                    }
                }
            }
        }
    }
    // A half-edge running from its higher end whose edge exists only that way
    // round was skipped above and is correctly left without a twin.
    // One outgoing half-edge per vertex, to start walks around it from.
    s.out.clear();
    s.out.resize(nv, NONE);
    for h in 0..nh {
        s.out[head(faces, h) as usize] = h as u32;
    }

    // Whether `a` and `b` are joined, by walking the fan of `a`. Only called
    // with `a` a sheet vertex, whose fan is a disc (or half of one on the
    // volume's edge) with no over-used edges.
    let joined = |faces: &[[u32; 3]], twin: &[u32], out: &[u32], a: u32, b: u32| -> bool {
        let start = out[a as usize];
        if start == NONE {
            return false;
        }
        let tail_of = |h: usize| faces[h / 3][(h + 1) % 3];
        // Rotate one way: the half-edge into `a` in this face, then across.
        let mut h = start as usize;
        loop {
            if tail_of(h) == b || head(faces, prev(h)) == b {
                return true;
            }
            let t = twin[prev(h)];
            if t >= MANY {
                break;
            }
            h = t as usize;
            if h == start as usize {
                return false;
            }
        }
        // Hit the edge of the surface: rotate the other way from the start.
        let mut h = start as usize;
        loop {
            let t = twin[h];
            if t >= MANY {
                return false;
            }
            h = next(t as usize);
            if tail_of(h) == b {
                return true;
            }
            if h == start as usize {
                return false;
            }
        }
    };

    // Everything judging an edge asks of a vertex, gathered once, in order,
    // into one record. Asked per edge instead, each of four vertices cost a
    // scattered read of its flags, its cell and its position -- three chains
    // of dependent lookups -- and that, not the arithmetic, was most of the
    // cost of judging an edge.
    s.vertex.clear();
    s.vertex.extend((0..nv as u32).map(|v| Vertex {
        position: verts.position(v),
        key: verts.key(v),
        flags: (if verts.sheet(v) { SHEET } else { 0 })
            | (if verts.frozen(v) { FROZEN } else { 0 }),
    }));
    let vertex = std::mem::take(&mut s.vertex);
    let at = |v: u32| &vertex[v as usize];
    let movable = |v: u32| at(v).flags & FROZEN == 0;
    let sheet = |v: u32| at(v).flags & SHEET != 0;
    // Canonical order of two vertices: by cell, then by position. Two
    // vertices of one cell share a position, so a pair they would tie on is
    // degenerate and never flipped.
    let before = |a: u32, b: u32| {
        let (va, vb) = (at(a), at(b));
        if va.key != vb.key {
            va.key < vb.key
        } else {
            va.position < vb.position
        }
    };
    let ordered = |a: u32, b: u32| if before(a, b) { (a, b) } else { (b, a) };
    // An edge's identity shared by every copy of it: its cells.
    let cell_pair = |a: u32, b: u32| {
        let (a, b) = ordered(a, b);
        ((at(a).key as u64) << 32) | at(b).key as u64
    };

    // The gain from flipping the edge with halves `h` and `t`, or zero when it
    // may not be flipped. A function of the two triangles alone.
    // Squared quality of a face, with its corners in canonical order so every
    // copy of it rounds the same way.
    let face_quality = |f: [u32; 3]| -> f32 {
        let mut c = f;
        if before(c[1], c[0]) {
            c.swap(0, 1);
        }
        if before(c[2], c[1]) {
            c.swap(1, 2);
        }
        if before(c[1], c[0]) {
            c.swap(0, 1);
        }
        let (a, b, d) = (at(c[0]).position, at(c[1]).position, at(c[2]).position);
        let (ab, ad, bd) = (sub(b, a), sub(d, a), sub(d, b));
        let n = cross(ab, ad);
        let l2 = dot(ab, ab) + dot(ad, ad) + dot(bd, bd);
        if l2 > 0.0 {
            (12.0 * dot(n, n) / (l2 * l2)) as f32
        } else {
            0.0
        }
    };
    s.quality.clear();
    s.quality.extend(faces.iter().map(|&f| face_quality(f)));
    let mut quality = std::mem::take(&mut s.quality);

    let judge = |faces: &[[u32; 3]], quality: &[f32], h: usize, t: usize| -> f32 {
        // A pair whose worse triangle is already good is left alone, without
        // the full test: on a well-shaped surface that is most of them.
        if quality[h / 3].min(quality[t / 3]) >= GOOD_QUALITY * GOOD_QUALITY {
            return 0.0;
        }
        let (a, b) = (head(faces, h), head(faces, next(h)));
        let (c, d) = (head(faces, prev(h)), head(faces, prev(t)));
        if !(sheet(a) || sheet(b))
            || !movable(a)
            || !movable(b)
            || c == d
            || !movable(c)
            || !movable(d)
            || !(sheet(c) || sheet(d))
        {
            return 0.0;
        }
        let (u, v) = ordered(a, b);
        let (p, q) = ordered(c, d);
        let pos = |w: u32| at(w).position;
        // Rounded to f32 for storage. Every copy of the wall rounds the same
        // bits, so comparisons still agree.
        improvement(pos(u), pos(v), pos(p), pos(q), cos_hinge) as f32
    };

    s.gain.clear();
    s.gain.resize(nh, 0.0);
    // The pass that last rewrote each face, to check the scheduling promise
    // that no two flips in a pass share a triangle. Checked in debug builds
    // only; the tests run there.
    s.stamp.clear();
    if cfg!(debug_assertions) {
        s.stamp.resize(faces.len(), u32::MAX);
    }
    let mut total = 0;
    for pass in 0..passes {
        // --- 1. gains --------------------------------------------------------
        // An edge's gain depends only on its two triangles, and positions never
        // change, so after the first pass only edges touching a triangle the
        // last pass rewrote are evaluated again. Both halves are written, so
        // an edge whose lower half sits in an untouched face is still caught:
        // the test is on either face.
        for h in 0..nh {
            let t = s.twin[h];
            if t >= MANY {
                // Possibly an edge that had a twin until a neighbouring flip:
                // its old gain must not survive.
                s.gain[h] = 0.0;
                continue;
            }
            if (t as usize) < h {
                continue;
            }
            let t = t as usize;
            if pass > 0 && !s.dirty[h / 3] && !s.dirty[t / 3] {
                continue;
            }
            let g = judge(faces, &quality, h, t);
            s.gain[h] = g;
            s.gain[t] = g;
        }
        // The incremental evaluation must agree with a full one. Checked in
        // debug builds, where the tests run.
        if cfg!(debug_assertions) {
            for h in 0..nh {
                let t = s.twin[h];
                if t < MANY && (t as usize) > h {
                    assert_eq!(
                        s.gain[h],
                        judge(faces, &quality, h, t as usize),
                        "stale gain"
                    );
                }
            }
        }
        s.dirty.clear();
        s.dirty.resize(faces.len(), false);

        // The new edge must not exist already. Checked afresh every pass for
        // every edge still in the running, since a flip anywhere can create
        // the edge another flip would: that is the one thing an edge's gain
        // depends on beyond its own two triangles. Asked of a sheet vertex,
        // whose edges all lie in this wall, so every copy agrees.
        for h in 0..nh {
            let t = s.twin[h];
            if s.gain[h] <= 0.0 || (t as usize) < h {
                continue;
            }
            let (c, d) = (head(faces, prev(h)), head(faces, prev(t as usize)));
            let (sv, o) = if sheet(c) { (c, d) } else { (d, c) };
            if joined(faces, &s.twin, &s.out, sv, o) {
                s.gain[h] = 0.0;
                s.gain[t as usize] = 0.0;
            }
        }

        // --- 2. choose -------------------------------------------------------
        s.chosen.clear();
        for h in 0..nh {
            let t = s.twin[h];
            if s.gain[h] <= 0.0 || (t as usize) < h {
                continue;
            }
            let t = t as usize;
            let own = (s.gain[h], cell_pair(head(faces, h), head(faces, next(h))));
            let beaten = [next(h), prev(h), next(t), prev(t)].into_iter().any(|e| {
                s.gain[e] > 0.0 && {
                    let other = (s.gain[e], cell_pair(head(faces, e), head(faces, next(e))));
                    other > own
                }
            });
            if !beaten {
                let (c, d) = (head(faces, prev(h)), head(faces, prev(t)));
                s.chosen.push((cell_pair(c, d), h as u32));
            }
        }
        // Two flips creating one edge: drop both.
        s.chosen.sort_unstable();
        s.keep.clear();
        s.keep.resize(s.chosen.len(), true);
        for i in 1..s.chosen.len() {
            if s.chosen[i].0 == s.chosen[i - 1].0 {
                s.keep[i] = false;
                s.keep[i - 1] = false;
            }
        }

        // --- 3. apply --------------------------------------------------------
        let mut flipped = 0;
        for i in 0..s.chosen.len() {
            if !s.keep[i] {
                continue;
            }
            let h = s.chosen[i].1 as usize;
            let t = s.twin[h] as usize;
            let (fa, fb) = (h / 3, t / 3);
            if cfg!(debug_assertions) {
                assert!(
                    s.stamp[fa] != pass && s.stamp[fb] != pass,
                    "two flips in one pass share a triangle"
                );
                s.stamp[fa] = pass;
                s.stamp[fb] = pass;
            }
            let (u, v, p) = (head(faces, h), head(faces, next(h)), head(faces, prev(h)));
            let q = head(faces, prev(t));
            // Outer half-edges and their twins, before anything moves:
            // v->p and p->u in fa, u->q and q->v in fb.
            let twin = &mut s.twin;
            let outer = [twin[next(h)], twin[prev(h)], twin[next(t)], twin[prev(t)]];
            // fa becomes (p, u, q) and fb (q, v, p): the quad u, q, v, p
            // split along p-q, with the winding kept.
            faces[fa] = [p, u, q];
            faces[fb] = [q, v, p];
            let (a0, a1, a2) = (3 * fa, 3 * fa + 1, 3 * fa + 2);
            let (b0, b1, b2) = (3 * fb, 3 * fb + 1, 3 * fb + 2);
            let link = |twin: &mut [u32], x: usize, y: u32| {
                twin[x] = y;
                if y < MANY {
                    twin[y as usize] = x as u32;
                }
            };
            link(twin, a0, outer[1]); // p->u
            link(twin, a1, outer[2]); // u->q
            link(twin, b0, outer[3]); // q->v
            link(twin, b1, outer[0]); // v->p
            twin[a2] = b2 as u32; // q->p
            twin[b2] = a2 as u32; // p->q
            s.out[p as usize] = a0 as u32;
            s.out[u as usize] = a1 as u32;
            s.out[q as usize] = b0 as u32;
            s.out[v as usize] = b1 as u32;
            s.dirty[fa] = true;
            s.dirty[fb] = true;
            quality[fa] = face_quality(faces[fa]);
            quality[fb] = face_quality(faces[fb]);
            flipped += 1;
        }
        total += flipped;
        if flipped == 0 {
            break;
        }
    }
    s.vertex = vertex;
    s.quality = quality;
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit square split along its long way round: two thin triangles
    /// either side of a stretched rhombus's long diagonal.
    fn rhombus() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let pos = vec![
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
        ];
        // Edge (0, 1) is the long diagonal.
        (pos, vec![[0, 1, 2], [1, 0, 3]])
    }

    /// Every vertex alike: keyed by its own id, all sheet or none, all frozen
    /// or none.
    struct Uniform<'a> {
        pos: &'a [[f64; 3]],
        sheet: bool,
        frozen: bool,
    }

    impl VertexInfo for Uniform<'_> {
        fn count(&self) -> usize {
            self.pos.len()
        }
        fn position(&self, v: u32) -> [f64; 3] {
            self.pos[v as usize]
        }
        fn key(&self, v: u32) -> u32 {
            v
        }
        fn sheet(&self, _: u32) -> bool {
            self.sheet
        }
        fn frozen(&self, _: u32) -> bool {
            self.frozen
        }
    }

    fn run_passes(
        pos: &[[f64; 3]],
        faces: &mut [[u32; 3]],
        sheet: bool,
        frozen: bool,
        passes: u32,
    ) -> usize {
        flip_edges(faces, &Uniform { pos, sheet, frozen }, passes)
    }

    fn run(pos: &[[f64; 3]], faces: &mut [[u32; 3]], sheet: bool, frozen: bool) -> usize {
        run_passes(pos, faces, sheet, frozen, 4)
    }

    #[test]
    fn the_long_diagonal_is_flipped() {
        let (pos, mut faces) = rhombus();
        assert_eq!(run(&pos, &mut faces, true, false), 1);
        let mut used: Vec<u64> = faces
            .iter()
            .flat_map(|f| (0..3).map(move |k| edge_key(f[k], f[(k + 1) % 3])))
            .collect();
        used.sort_unstable();
        assert!(used.contains(&edge_key(2, 3)));
        assert!(!used.contains(&edge_key(0, 1)));
    }

    #[test]
    fn winding_is_kept() {
        let (pos, mut faces) = rhombus();
        run(&pos, &mut faces, true, false);
        for f in &faces {
            let (a, b, c) = (pos[f[0] as usize], pos[f[1] as usize], pos[f[2] as usize]);
            // Both input faces wind counter-clockwise seen from +z.
            let n = cross(sub(b, a), sub(c, a));
            let n0 = {
                let (a, b, c) = (pos[0], pos[1], pos[2]);
                cross(sub(b, a), sub(c, a))
            };
            assert!(dot(n, n0) > 0.0, "a flip reversed a face");
        }
    }

    #[test]
    fn nothing_moves_without_a_sheet_vertex_or_when_frozen() {
        let (pos, mut faces) = rhombus();
        assert_eq!(run(&pos, &mut faces, false, false), 0);
        assert_eq!(run(&pos, &mut faces, true, true), 0);
    }

    #[test]
    fn a_creased_pair_is_left_alone() {
        let (mut pos, mut faces) = rhombus();
        // Fold the two triangles 60 degrees out of plane about the long edge.
        pos[2] = [2.0, 0.5, 0.866];
        assert_eq!(run(&pos, &mut faces, true, false), 0);
    }

    #[test]
    fn a_non_convex_quad_is_left_alone() {
        let (mut pos, mut faces) = rhombus();
        // Pull q over to p's side of the edge: flipping would fold the surface.
        pos[3] = [2.0, 0.2, 0.0];
        assert_eq!(run(&pos, &mut faces, true, false), 0);
    }

    /// A bumpy grid, each square split along its worse diagonal.
    fn grid(n: usize) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let mut pos = Vec::new();
        for j in 0..n {
            for i in 0..n {
                // Deterministic pseudo-random jitter, irregular enough that
                // neighbouring edges compete for the same triangles.
                let mut x = (i * 7919 + j * 104_729 + 12_345) as u64;
                let mut noise = || {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    ((x >> 33) as f64 / (1u64 << 31) as f64) - 0.5
                };
                let (a, b, c) = (noise(), noise(), noise());
                pos.push([i as f64 * 1.6 + 0.6 * a, j as f64 + 0.4 * b, 0.05 * c]);
            }
        }
        let id = |i: usize, j: usize| (j * n + i) as u32;
        let mut faces = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
                faces.push([a, b, d]);
                faces.push([b, c, d]);
            }
        }
        (pos, faces)
    }

    fn canonical(faces: &[[u32; 3]]) -> Vec<[u32; 3]> {
        let mut out: Vec<[u32; 3]> = faces
            .iter()
            .map(|f| {
                let mut t = *f;
                t.sort_unstable();
                t
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Two labels' copies of a wall store its faces in different orders and
    /// with opposite winding. The flips must come out the same regardless.
    #[test]
    fn the_result_does_not_depend_on_face_order_or_winding() {
        let (pos, faces) = grid(12);
        let mut a = faces.clone();
        let flipped = run(&pos, &mut a, true, false);
        assert!(flipped > 10, "only {flipped} flips");

        let mut b: Vec<[u32; 3]> = faces.iter().rev().map(|f| [f[0], f[2], f[1]]).collect();
        b.rotate_left(17);
        run(&pos, &mut b, true, false);
        assert_eq!(canonical(&a), canonical(&b));
    }

    /// Each flip improves the worse of its own two triangles, and flips chosen
    /// in one pass share no triangle, so the mesh's worst triangle can never
    /// get worse. It did once: a bookkeeping slip let two neighbouring flips
    /// both claim one triangle, and the second undid the first's reasoning.
    #[test]
    fn the_worst_triangle_never_gets_worse() {
        let (pos, mut faces) = grid(30);
        let worst = |faces: &[[u32; 3]]| {
            faces
                .iter()
                .map(|f| quality(pos[f[0] as usize], pos[f[1] as usize], pos[f[2] as usize]))
                .fold(f64::INFINITY, f64::min)
        };
        let before = worst(&faces);
        for _ in 0..4 {
            let last = worst(&faces);
            // One pass at a time, so every pass is checked.
            run_passes(&pos, &mut faces, true, false, 1);
            assert!(worst(&faces) >= last);
        }
        assert!(worst(&faces) > before);
    }

    /// A polygon, stretched 3:1, triangulated as a fan from one corner. Its
    /// interior edges share triangles with their neighbours, and several of
    /// them want flipping at once, so the flips chosen in a pass have to be
    /// picked from rivals.
    fn fan(n: usize) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let pos = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [3.0 * a.cos(), a.sin(), 0.0]
            })
            .collect();
        let faces = (1..n as u32 - 1).map(|i| [0, i, i + 1]).collect();
        (pos, faces)
    }

    #[test]
    fn rival_flips_are_never_applied_together() {
        // The debug-build check in `flip_edges` panics if two flips in one
        // pass share a triangle; this is the fixture that would trip it.
        let (pos, mut faces) = fan(11);
        let before = faces.clone();
        let flipped = run(&pos, &mut faces, true, false);
        assert!(flipped >= 3, "only {flipped} flips");
        let worst = |faces: &[[u32; 3]]| {
            faces
                .iter()
                .map(|f| quality(pos[f[0] as usize], pos[f[1] as usize], pos[f[2] as usize]))
                .fold(f64::INFINITY, f64::min)
        };
        assert!(worst(&faces) > worst(&before));

        // And the same answer from reversed storage and winding.
        let mut other: Vec<[u32; 3]> = before.iter().rev().map(|f| [f[0], f[2], f[1]]).collect();
        run(&pos, &mut other, true, false);
        assert_eq!(canonical(&faces), canonical(&other));
    }

    /// Every edge still has at most two faces, and interior edges exactly two.
    #[test]
    fn the_surface_stays_manifold() {
        let (pos, mut faces) = grid(12);
        run(&pos, &mut faces, true, false);
        let mut uses: std::collections::HashMap<u64, usize> = Default::default();
        for f in &faces {
            for k in 0..3 {
                *uses.entry(edge_key(f[k], f[(k + 1) % 3])).or_default() += 1;
            }
        }
        assert!(uses.values().all(|&n| n <= 2));
        let boundary = uses.values().filter(|&&n| n == 1).count();
        // A 12 x 12 grid has 4 * 11 boundary edges, and flips never touch them.
        assert_eq!(boundary, 44);
    }

    #[test]
    fn improvement_is_the_same_whichever_way_round() {
        let (pos, _) = rhombus();
        let cos = MAX_HINGE_DEGREES.to_radians().cos();
        let a = improvement(pos[0], pos[1], pos[2], pos[3], cos);
        assert!(a > 0.0);
        assert_eq!(a, improvement(pos[1], pos[0], pos[3], pos[2], cos));
        assert_eq!(a, improvement(pos[0], pos[1], pos[3], pos[2], cos));
    }
}
