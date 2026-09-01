//! Smoothing a label's surface once its vertices have been placed.
//!
//! Two filters live here. Both are Jacobi — every vertex is updated from the
//! previous iteration's positions — so neither can depend on the order vertices
//! happen to be visited in, and both hold pinned vertices fixed.
//!
//! * [`Relaxation`] is a constrained Laplacian: each pass moves a vertex a
//!   fraction of the way to the average of its neighbours. It is diffusion, so
//!   it shrinks, and the shrinking is what [`Relaxation::max_deviation`] exists
//!   to bound.
//!
//! * [`Taubin`] alternates a positive step `lambda` with a larger negative step
//!   `mu`. The pair is a low-pass filter rather than a diffusion: frequencies
//!   below the pass band are left alone while the rest are attenuated, so the
//!   surface can be smoothed much harder without the volume draining away. It
//!   costs two passes per iteration instead of one, and shares everything else.
//!
//! Both keep the deviation bound, which the usual formulations do not. Taubin's
//! shrink compensation is statistical — it holds in aggregate, not per vertex —
//! so a bound is still worth having when the surface is a measurement rather
//! than a model. It also preserves what the accuracy documentation claims: no
//! vertex is ever further than `max_deviation` from where the data put it.
//!
//! Pinning is what makes any of this safe to run per chunk. The outermost layer
//! of cells is held fixed, so a chunk's mesh stays reproducible from that
//! chunk's own array however many iterations run, and seam vertices stay
//! bit-identical between neighbours. The price is that a chunk's interior
//! smooths slightly more than the band around its seams, so a stitched surface
//! is self-consistent and watertight but not identical to the same volume
//! smoothed in one piece.

use crate::extract::{CellField, LabelMesh};
use crate::tables::{CENTROID, JUNCTION_FACES, SUBVOXEL, SURFACE_FACES};
use rayon::prelude::*;

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

/// Settings for the optional Taubin pass.
///
/// The two steps are tied together by the pass-band frequency
/// `k = 1/lambda + 1/mu`, which is the graph frequency the filter leaves at
/// unit gain. Everything below it passes; everything above is attenuated. Fixing
/// `k` and `lambda` fixes `mu`, so those are the two knobs exposed and `mu`
/// follows.
#[derive(Clone, Copy, Debug)]
pub struct Taubin {
    /// Number of lambda/mu pairs. Zero disables the pass. Each costs two passes
    /// over the surface.
    pub iterations: u32,
    /// Pass-band frequency, in (0, 1). Smaller smooths more.
    pub pass_band: f64,
    /// The positive step, in (0, 1).
    pub lambda: f64,
    /// How far a vertex may drift from where local placement put it, in voxels.
    pub max_deviation: f64,
}

impl Default for Taubin {
    fn default() -> Self {
        Taubin {
            iterations: 0,
            // 0.1 is the usual default, and what VTK's windowed-sinc filter
            // uses, so the two are comparable at matched iteration counts.
            pass_band: 0.1,
            lambda: 0.63,
            max_deviation: 0.5,
        }
    }
}

impl Taubin {
    /// The negative step implied by `pass_band` and `lambda`.
    ///
    /// Negative and larger in magnitude than `lambda` for any sane pair, which
    /// is precisely the condition that makes the filter low-pass rather than a
    /// slower diffusion.
    #[inline]
    pub fn mu(&self) -> f64 {
        1.0 / (self.pass_band - 1.0 / self.lambda)
    }

    /// Whether the parameters give a stable low-pass filter.
    ///
    /// With both knobs in (0, 1), `mu` is always negative and always larger in
    /// magnitude than `lambda`, so the only way to get this wrong is to make
    /// the pass band too wide for the chosen `lambda`: once `mu <= -1` the
    /// expanding pass overshoots and the filter amplifies rather than smooths.
    pub fn is_valid(&self) -> bool {
        if !(self.pass_band.is_finite() && self.lambda.is_finite()) {
            return false;
        }
        if self.pass_band <= 0.0 || self.pass_band >= 1.0 {
            return false;
        }
        if self.lambda <= 0.0 || self.lambda >= 1.0 {
            return false;
        }
        let mu = self.mu();
        mu.is_finite() && mu < 0.0 && mu.abs() > self.lambda && mu > -1.0
    }
}

/// Which smoothing filter to run, if any.
#[derive(Clone, Copy, Debug, Default)]
pub enum Smoothing {
    #[default]
    None,
    Laplacian(Relaxation),
    Taubin(Taubin),
    /// Frisken's fairing, in the cell domain. Handled separately from the other
    /// two because it runs once over the shared cell field rather than once per
    /// label; see [`fair`].
    Fairing(Fairing),
}

impl Smoothing {
    /// Whether anything will actually run. Used to decide whether the extractor
    /// needs to mark the chunk's outermost cell layer as pinned.
    pub fn is_active(&self) -> bool {
        match self {
            Smoothing::None => false,
            Smoothing::Laplacian(r) => r.iterations > 0,
            Smoothing::Taubin(t) => t.iterations > 0,
            Smoothing::Fairing(f) => f.iterations > 0,
        }
    }

    /// Whether this runs over the shared cell field rather than per label.
    pub fn is_cell_domain(&self) -> bool {
        matches!(self, Smoothing::Fairing(_))
    }
}

/// Settings for fairing in the cell domain.
///
/// This is Frisken's surface fairing rather than serra's per-label relaxation:
/// one position per cell, shared by every material there, so the copies of a
/// wall between two touching objects cannot drift apart.
#[derive(Clone, Copy, Debug)]
pub struct Fairing {
    /// Number of Jacobi sweeps. Zero disables the pass.
    pub iterations: u32,
    /// Fraction of the way to the neighbour average each sweep, in (0, 1].
    pub step: f64,
    /// How far a vertex may drift from where placement put it, in voxels.
    ///
    /// Intersected with the cell, never replaced by it. The two bounds are not
    /// nested: a placed vertex sits between 43/256 and 213/256 of the way
    /// across its cell, so the cell is the tighter bound on one side and the
    /// looser on the other. Taking both keeps the documented `max_deviation`
    /// guarantee true and adds Frisken's containment on top.
    pub max_deviation: f64,
    /// Restrict junction cells to their junction neighbours.
    pub junction_rule: bool,
    /// Alternate Taubin's shrink and unshrink steps instead of repeating
    /// `step`, with this pass band. `None` is the plain Laplacian.
    ///
    /// Worth having because the cell domain and the shrinkage are independent
    /// problems. Sharing a cell's vertex stops adjacent objects drifting apart;
    /// it does nothing about volume loss, because Frisken's fairing is a plain
    /// Laplacian and shrinks exactly as serra's relaxation does. Alternating a
    /// larger negative step fixes that, and the two compose.
    pub pass_band: Option<f64>,
    /// The positive step of the Taubin pair, in (0, 1).
    pub lambda: f64,
}

impl Default for Fairing {
    fn default() -> Self {
        Fairing {
            iterations: 0,
            step: 0.5,
            max_deviation: 0.5,
            junction_rule: true,
            pass_band: None,
            lambda: 0.63,
        }
    }
}

impl Fairing {
    /// The steps one iteration cycles through: one for a Laplacian, two for
    /// Taubin.
    fn steps(&self) -> Vec<f32> {
        match self.pass_band {
            None => vec![self.step as f32],
            Some(pass_band) => {
                let mu = 1.0 / (pass_band - 1.0 / self.lambda);
                vec![self.lambda as f32, mu as f32]
            }
        }
    }
}

/// Fair the cell field in place.
///
/// Neighbours are the cells across the six faces, which is Frisken's stencil
/// with one deliberate exception: **faces with no crossing are skipped**. A face
/// no surface passes through separates two cells that are not on a common
/// sheet, and averaging across it drags them together. On a sheet one voxel
/// thick the two sides are face-adjacent, so the literal rule collapses it to
/// zero thickness, and no displacement bound prevents that because the sides
/// begin exactly one voxel apart. Excluding uniform faces makes the stencil
/// exactly the union over labels of the quad adjacency serra already builds.
///
/// Jacobi, and every output depends only on the previous sweep and on a fixed
/// six-entry stencil summed in face order, so the result is identical for any
/// chunking or thread count. `parallel` therefore changes only the speed — and
/// it has to be passed rather than assumed, because rayon's global pool would
/// otherwise spread the sweep across every core even when the caller asked for
/// a single thread.
pub fn fair(cells: &mut CellField, params: &Fairing, parallel: bool) {
    let n = cells.positions.len();
    if params.iterations == 0 || n == 0 {
        return;
    }
    let [nx, ny, nz] = cells.nc;
    let plane = nx * ny;
    let offset: [i64; 6] = [
        -1,
        1,
        -(nx as i64),
        nx as i64,
        -(plane as i64),
        plane as i64,
    ];

    let placed = &cells.positions;
    let linear = &cells.linear;
    let crossings = &cells.crossings;
    let pinned = &cells.pinned;
    let limit = (params.max_deviation * SUBVOXEL as f64) as f32;
    let steps = params.steps();

    let mut current: Vec<[f32; 3]> = placed
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let mut next = current.clone();

    // Which faces each cell averages over, decided once for all sweeps.
    //
    // None of it varies between sweeps, and recomputing it was most of the
    // cost. Locating a cell in the grid takes three integer divisions, and the
    // compiler cannot strength-reduce them because the extents are runtime
    // values -- so at twenty sweeps that was sixty divisions per cell spent
    // rediscovering one byte.
    //
    // Folded in here too: whether a face has a neighbour at all, without which
    // the low face of column zero would name the last column of the previous
    // row -- a real cell, silently averaged against. And the junction rule: a
    // junction cell slides along its curve, and seven in eight have exactly two
    // junction faces, a curve entering and leaving. The rest fall back to the
    // full stencil rather than be dragged onto a single neighbour.
    let stencil: Vec<u8> = (0..n)
        .map(|i| {
            let l = linear[i] as usize;
            let (cx, cy, cz) = (l % nx, (l / nx) % ny, l / plane);
            let mut in_grid = 0u8;
            for (d, inside) in [
                cx > 0,
                cx + 1 < nx,
                cy > 0,
                cy + 1 < ny,
                cz > 0,
                cz + 1 < nz,
            ]
            .into_iter()
            .enumerate()
            {
                if inside {
                    in_grid |= 1 << d;
                }
            }
            let mask = crossings[i] as usize;
            let junction = JUNCTION_FACES[mask] & in_grid;
            if params.junction_rule && junction.count_ones() >= 2 {
                junction
            } else {
                SURFACE_FACES[mask] & in_grid
            }
        })
        .collect();

    // Big enough that the six binary searches at the head of each chunk are
    // noise, small enough to keep every core fed.
    const CHUNK: usize = 1 << 16;

    for _ in 0..params.iterations {
        for &step in &steps {
            let sweep = |(block, out): (usize, &mut [[f32; 3]])| {
                let start = block * CHUNK;
                // Seek the cursors to this chunk. Targets are monotone in `i`,
                // so from here each one only ever moves forward.
                let mut cursor = [0usize; 6];
                for (d, c) in cursor.iter_mut().enumerate() {
                    let target = linear[start] as i64 + offset[d];
                    *c = linear.partition_point(|&x| (x as i64) < target);
                }

                for (k, slot) in out.iter_mut().enumerate() {
                    let i = start + k;
                    let use_mask = stencil[i];

                    let mut sum = [0.0f32; 3];
                    let mut count = 0u32;
                    for (d, c) in cursor.iter_mut().enumerate() {
                        // Advance every cursor, used or not, or the ones that
                        // are skipped fall behind and desynchronise.
                        let target = linear[i] as i64 + offset[d];
                        while *c < n && (linear[*c] as i64) < target {
                            *c += 1;
                        }
                        if use_mask & (1 << d) == 0 {
                            continue;
                        }
                        if *c < n && linear[*c] as i64 == target {
                            let p = current[*c];
                            for a in 0..3 {
                                sum[a] += p[a];
                            }
                            count += 1;
                        }
                    }

                    if pinned[i] || count == 0 {
                        *slot = current[i];
                        continue;
                    }

                    let scale = 1.0 / count as f32;
                    let offsets = &CENTROID[crossings[i] as usize];
                    for a in 0..3 {
                        let average = sum[a] * scale;
                        let moved = current[i][a] + step * (average - current[i][a]);
                        // The cell, and the deviation bound, intersected.
                        let base = (placed[i][a] - offsets[a]) as f32;
                        let anchor = placed[i][a] as f32;
                        let lo = base.max(anchor - limit);
                        let hi = (base + SUBVOXEL as f32).min(anchor + limit);
                        slot[a] = moved.clamp(lo, hi);
                    }
                }
            };
            if parallel {
                next.par_chunks_mut(CHUNK).enumerate().for_each(sweep);
            } else {
                next.chunks_mut(CHUNK).enumerate().for_each(sweep);
            }
            std::mem::swap(&mut current, &mut next);
        }
    }

    for (slot, moved) in cells.positions.iter_mut().zip(current.iter()) {
        for a in 0..3 {
            slot[a] = moved[a].round() as i32;
        }
    }
}

/// Copy faired cell positions onto every vertex that came from that cell.
///
/// After this the two copies of a shared wall are bit-identical, because they
/// are reads of the same number rather than two independently faired values.
pub fn scatter(cells: &CellField, mesh: &mut LabelMesh) {
    for (position, &cell) in mesh.positions.iter_mut().zip(mesh.cells.iter()) {
        *position = cells.positions[cell as usize];
    }
}

/// Reusable working memory for [`smooth_with`].
///
/// Smoothing a chunk means smoothing thousands of separate surfaces, and the
/// buffers it needs are large: the adjacency alone is eight entries per quad
/// corner. Allocated per mesh, the first touch of each fresh page dominates the
/// whole pass — it is the single largest cost in smoothing a chunk, ahead of
/// the arithmetic. Carried across meshes, the pages are faulted once per thread
/// and then reused, and the cost disappears.
#[derive(Default)]
pub struct Scratch {
    /// Start of each vertex's slot range in `neighbours`, before dedup.
    slots: Vec<u32>,
    /// Write cursor per vertex, which after the scatter is the range end.
    cursor: Vec<u32>,
    neighbours: Vec<u32>,
    current: Vec<[f32; 3]>,
    next: Vec<[f32; 3]>,
    pins: Vec<bool>,
}

impl Scratch {
    /// Grow a buffer to `len` without paying to rewrite what is already there.
    ///
    /// Used only for `neighbours`, whose every slot is written before it is
    /// read; the zeroing `resize` would do is pure waste at gigabyte sizes.
    fn ensure(buffer: &mut Vec<u32>, len: usize) {
        if buffer.len() < len {
            buffer.resize(len, 0);
        }
    }
}

/// Smooth a label's surface in place, holding pinned vertices fixed.
///
/// Allocates its own working memory. Prefer [`smooth_with`] when smoothing many
/// meshes in a row.
pub fn smooth(mesh: &mut LabelMesh, smoothing: &Smoothing) {
    smooth_with(mesh, smoothing, &mut Scratch::default());
}

/// Smooth a label's surface in place, reusing `scratch` across calls.
pub fn smooth_with(mesh: &mut LabelMesh, smoothing: &Smoothing, scratch: &mut Scratch) {
    match smoothing {
        Smoothing::None => {}
        Smoothing::Laplacian(p) => run(mesh, p.iterations, p.max_deviation, &[p.step], scratch),
        Smoothing::Taubin(p) => run(
            mesh,
            p.iterations,
            p.max_deviation,
            &[p.lambda, p.mu()],
            scratch,
        ),
        // Ran over the whole cell field before this point; each mesh only has
        // to read its positions back.
        Smoothing::Fairing(_) => {}
    }
}

/// Constrained Laplacian smoothing.
///
/// Pinned vertices never move. When [`crate::extract::ExtractOptions::
/// mark_boundary`] marked the outermost cell layer, this makes the whole pass
/// depend only on data the chunk already holds — so a chunk's mesh is
/// reproducible from that chunk alone, whatever `iterations` is set to, and the
/// halo does not grow with it. (The extractor needs two voxels of halo either
/// way, because a quad spans two cell layers; the point here is that smoothing
/// adds nothing to that.)
pub fn relax(mesh: &mut LabelMesh, params: &Relaxation) {
    run(
        mesh,
        params.iterations,
        params.max_deviation,
        &[params.step],
        &mut Scratch::default(),
    );
}

/// Taubin's lambda/mu low-pass filter.
///
/// Each iteration is a shrinking pass followed by a larger expanding one. The
/// net effect on low graph frequencies is close to unity, which is why this
/// smooths far harder than Laplacian iteration for the same volume loss.
pub fn taubin(mesh: &mut LabelMesh, params: &Taubin) {
    run(
        mesh,
        params.iterations,
        params.max_deviation,
        &[params.lambda, params.mu()],
        &mut Scratch::default(),
    );
}

/// The shared engine: `iterations` rounds of one Jacobi pass per entry in
/// `steps`.
///
/// Both filters differ only in the step sizes they alternate through, so the
/// adjacency structure, the buffers, the pinning and the deviation clamp are
/// built once here rather than twice.
fn run(
    mesh: &mut LabelMesh,
    iterations: u32,
    max_deviation: f64,
    steps: &[f64],
    scratch: &mut Scratch,
) {
    if iterations == 0 || steps.is_empty() || mesh.quads.is_empty() {
        return;
    }
    let n = mesh.positions.len();
    build_adjacency(mesh, n, scratch);

    // Two buffers, not three. The clamp is against `mesh.positions`, which is
    // not written until the very end, so the extra copy of the starting state
    // that an explicit `start` buffer would hold is already on hand.
    //
    // f32, not f64. Smoothing is memory-bound — a pass is one random gather per
    // neighbour — so halving the width of the position buffers halves the
    // traffic. The precision is ample: positions are fixed-point units of 1/256
    // of a voxel, a 1024-voxel chunk reaches 262144, and f32 resolves integers
    // exactly to 16.7 million. The result is rounded back to whole units
    // regardless. IEEE f32 is as deterministic as f64, so output is still
    // identical across threads and platforms.
    scratch.current.clear();
    scratch.current.extend(
        mesh.positions
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]),
    );
    scratch.next.clear();
    scratch.next.extend_from_slice(&scratch.current);

    // `pinned` is left empty by the extractor when nothing was marked, so give
    // the inner loop a slice it can index without a length check.
    scratch.pins.clear();
    if mesh.pinned.len() == n {
        scratch.pins.extend_from_slice(&mesh.pinned);
    } else {
        scratch.pins.resize(n, false);
    }

    let limit = (max_deviation * SUBVOXEL as f64) as f32;
    let origin: &[[i32; 3]] = &mesh.positions;

    for _ in 0..iterations {
        for &step in steps {
            pass(
                Ring {
                    starts: &scratch.slots,
                    ends: &scratch.cursor,
                    neighbours: &scratch.neighbours,
                },
                origin,
                &scratch.current,
                &mut scratch.next,
                &scratch.pins,
                step as f32,
                limit,
            );
            std::mem::swap(&mut scratch.current, &mut scratch.next);
        }
    }

    for (slot, moved) in mesh.positions.iter_mut().zip(scratch.current.iter()) {
        for k in 0..3 {
            slot[k] = moved[k].round() as i32;
        }
    }
}

/// A vertex-to-ring-neighbour map in CSR form, with gaps: vertex `v`'s
/// neighbours are `neighbours[starts[v]..ends[v]]`. The gaps are what the dedup
/// left behind, and closing them would cost more than skipping them.
struct Ring<'a> {
    starts: &'a [u32],
    ends: &'a [u32],
    neighbours: &'a [u32],
}

impl Ring<'_> {
    #[inline]
    fn of(&self, v: usize) -> &[u32] {
        &self.neighbours[self.starts[v] as usize..self.ends[v] as usize]
    }
}

/// One Jacobi pass. `step` may be negative, which is what makes Taubin work.
fn pass(
    ring: Ring<'_>,
    origin: &[[i32; 3]],
    current: &[[f32; 3]],
    next: &mut [[f32; 3]],
    pinned: &[bool],
    step: f32,
    limit: f32,
) {
    for v in 0..current.len() {
        let ring = ring.of(v);
        if pinned[v] || ring.is_empty() {
            next[v] = current[v];
            continue;
        }
        let mut sum = [0.0f32; 3];
        for &w in ring {
            let p = current[w as usize];
            for k in 0..3 {
                sum[k] += p[k];
            }
        }
        let scale = 1.0 / ring.len() as f32;
        for k in 0..3 {
            let average = sum[k] * scale;
            let moved = current[v][k] + step * (average - current[v][k]);
            // Clamp against the *placed* position, so the bound is on total
            // deviation from the data rather than per step. For Taubin this
            // also keeps the expanding pass from overshooting on a surface
            // whose shrink compensation happens to be imperfect locally.
            let anchor = origin[v][k] as f32;
            next[v][k] = moved.clamp(anchor - limit, anchor + limit);
        }
    }
}

/// Build vertex neighbours in CSR form into `scratch`, from the quad ring
/// edges.
fn build_adjacency(mesh: &LabelMesh, n: usize, scratch: &mut Scratch) {
    // Each quad contributes its four ring edges, in both directions. An
    // interior edge is shared by two quads and so appears twice; the duplicates
    // are removed rather than left to double-weight a neighbour, which would
    // otherwise bias vertices along an open boundary.
    scratch.slots.clear();
    scratch.slots.resize(n + 1, 0);
    for q in &mesh.quads {
        for &v in q.iter() {
            scratch.slots[v as usize + 1] += 2;
        }
    }
    for i in 1..=n {
        scratch.slots[i] += scratch.slots[i - 1];
    }
    let total = scratch.slots[n] as usize;

    scratch.cursor.clear();
    scratch.cursor.extend_from_slice(&scratch.slots);
    Scratch::ensure(&mut scratch.neighbours, total);

    for q in &mesh.quads {
        for (i, &a) in q.iter().enumerate() {
            let b = q[(i + 1) % 4];
            for (from, to) in [(a, b), (b, a)] {
                let slot = &mut scratch.cursor[from as usize];
                scratch.neighbours[*slot as usize] = to;
                *slot += 1;
            }
        }
    }

    // Now drop the duplicates, within each vertex's own span. An interior edge
    // belongs to two quads and so was written twice; leaving it would weight
    // that neighbour double, which biases any vertex whose ring is not closed.
    // Compacting inside the span rather than across the whole array keeps this
    // a local operation on a handful of entries.
    for v in 0..n {
        let lo = scratch.slots[v] as usize;
        let hi = scratch.cursor[v] as usize;
        let span = &mut scratch.neighbours[lo..hi];
        span.sort_unstable();
        let mut write = 0usize;
        let mut last: Option<u32> = None;
        for i in 0..span.len() {
            let w = span[i];
            if w as usize != v && last != Some(w) {
                span[write] = w;
                write += 1;
                last = Some(w);
            }
        }
        scratch.cursor[v] = (lo + write) as u32;
    }

    // The gaps the dedup left are not closed up. Compacting them would mean
    // reading and rewriting the whole neighbour array — a gigabyte of traffic
    // on a large chunk — to save nothing: a vertex's list is
    // `neighbours[slots[v]..cursor[v]]` either way, and both index arrays are
    // walked in order.
}

#[cfg(test)]
mod fairing_tests {
    use super::*;
    use crate::extract::{extract_parallel, extract_with, ExtractOptions, Extraction};
    use crate::grid::VolumeView;
    use ndarray::Array3;
    use std::collections::HashMap;

    fn extract(a: &Array3<u32>, close: bool, threads: usize) -> Extraction {
        let opts = ExtractOptions {
            mark_boundary: !close,
            ..Default::default()
        };
        let view = VolumeView::new(a.view(), close);
        if threads == 0 {
            extract_with(&view, &opts)
        } else {
            extract_parallel(&view, &opts, threads)
        }
    }

    /// Three labels stacked so cells carry two and three materials at once.
    fn stacked(n: usize) -> Array3<u32> {
        Array3::from_shape_fn((n, n, n), |(x, y, z)| {
            let c = (n - 1) as f64 / 2.0;
            let r =
                ((x as f64 - c).powi(2) + (y as f64 - c).powi(2) + (z as f64 - c).powi(2)).sqrt();
            if r > c * 0.8 {
                0
            } else if z * 3 < n {
                1
            } else if z * 3 < 2 * n {
                2
            } else {
                3
            }
        })
    }

    fn params() -> Fairing {
        Fairing {
            iterations: 6,
            ..Default::default()
        }
    }

    /// The same update, written the slow obvious way: a map from linear index to
    /// array position, no cursors.
    ///
    /// This is the test that earns the cursor sweep. It catches both ways the
    /// clever version can be wrong — a cursor left behind because its face was
    /// skipped, and the low face of column zero wrapping onto the last column of
    /// the previous row.
    fn fair_reference(cells: &CellField, p: &Fairing, skip_uniform: bool) -> Vec<[i32; 3]> {
        let n = cells.positions.len();
        let [nx, ny, nz] = cells.nc;
        let index: HashMap<u32, usize> = cells
            .linear
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, i))
            .collect();
        let limit = (p.max_deviation * SUBVOXEL as f64) as f32;
        let step = p.step as f32;
        let mut current: Vec<[f32; 3]> = cells
            .positions
            .iter()
            .map(|q| [q[0] as f32, q[1] as f32, q[2] as f32])
            .collect();

        for _ in 0..p.iterations {
            let mut next = current.clone();
            for i in 0..n {
                let l = cells.linear[i] as usize;
                let (cx, cy, cz) = (l % nx, (l / nx) % ny, l / (nx * ny));
                let mut in_grid = 0u8;
                for (d, bit) in [
                    cx > 0,
                    cx + 1 < nx,
                    cy > 0,
                    cy + 1 < ny,
                    cz > 0,
                    cz + 1 < nz,
                ]
                .into_iter()
                .enumerate()
                {
                    if bit {
                        in_grid |= 1 << d;
                    }
                }
                let mask = cells.crossings[i] as usize;
                // `skip_uniform == false` is Frisken's rule taken literally:
                // every face neighbour, whether or not a surface passes between
                // them. Kept so the counterexample below can be run.
                let surface = if skip_uniform {
                    SURFACE_FACES[mask] & in_grid
                } else {
                    in_grid
                };
                let junction = JUNCTION_FACES[mask] & in_grid;
                let use_mask = if p.junction_rule && junction.count_ones() >= 2 {
                    junction
                } else {
                    surface
                };

                let mut sum = [0.0f32; 3];
                let mut count = 0u32;
                for d in 0..6 {
                    if use_mask & (1 << d) == 0 {
                        continue;
                    }
                    let target = cells.neighbour_linear(cells.linear[i], d);
                    if let Some(j) = target.and_then(|t| index.get(&t)) {
                        for a in 0..3 {
                            sum[a] += current[*j][a];
                        }
                        count += 1;
                    }
                }
                if cells.pinned[i] || count == 0 {
                    continue;
                }
                let scale = 1.0 / count as f32;
                for a in 0..3 {
                    let average = sum[a] * scale;
                    let moved = current[i][a] + step * (average - current[i][a]);
                    let base = (cells.positions[i][a] - CENTROID[mask][a]) as f32;
                    let anchor = cells.positions[i][a] as f32;
                    let lo = base.max(anchor - limit);
                    let hi = (base + SUBVOXEL as f32).min(anchor + limit);
                    next[i][a] = moved.clamp(lo, hi);
                }
            }
            current = next;
        }
        current
            .iter()
            .map(|q| {
                [
                    q[0].round() as i32,
                    q[1].round() as i32,
                    q[2].round() as i32,
                ]
            })
            .collect()
    }

    #[test]
    fn the_cursor_sweep_matches_a_direct_lookup() {
        for n in [17usize, 24, 48] {
            let a = stacked(n);
            for close in [false, true] {
                let mut e = extract(&a, close, 0);
                let want = fair_reference(&e.cells, &params(), true);
                fair(&mut e.cells, &params(), true);
                assert_eq!(e.cells.positions, want, "n={n} close={close}");
            }
        }
    }

    /// The reason for all of this: two labels' copies of a shared wall must come
    /// out bit-identical, because they are reads of one number.
    #[test]
    fn shared_walls_do_not_drift() {
        let a = stacked(32);
        let mut e = extract(&a, true, 0);
        fair(&mut e.cells, &params(), true);
        for mesh in e.meshes.iter_mut() {
            scatter(&e.cells, mesh);
        }

        // Group every vertex by the cell it came from, across all labels.
        let mut by_cell: HashMap<u32, [i32; 3]> = HashMap::new();
        let mut shared = 0;
        for mesh in &e.meshes {
            for (&cell, &position) in mesh.cells.iter().zip(mesh.positions.iter()) {
                match by_cell.get(&cell) {
                    None => {
                        by_cell.insert(cell, position);
                    }
                    Some(&first) => {
                        assert_eq!(first, position, "cell {cell} disagrees between labels");
                        shared += 1;
                    }
                }
            }
        }
        assert!(shared > 0, "no cell was shared, so nothing was tested");
    }

    fn one_voxel_sheet() -> Array3<u32> {
        let mut a = Array3::<u32>::zeros((20, 20, 20));
        a.slice_mut(ndarray::s![10, 4..16, 4..16]).fill(1);
        a
    }

    /// The sheet must still have two sides afterwards.
    #[test]
    fn a_one_voxel_sheet_survives() {
        let a = one_voxel_sheet();
        let mut e = extract(&a, true, 0);
        let before = spread_x(&e.cells.positions);
        fair(
            &mut e.cells,
            &Fairing {
                iterations: 80,
                max_deviation: 10.0, // deliberately no help from the bound
                ..Default::default()
            },
            true,
        );
        let after = spread_x(&e.cells.positions);
        // It thins -- a thin slab is joined around its rim, so curvature flow
        // pulls the sides together, and `max_deviation` is deliberately not
        // helping here. What matters is that it keeps a thickness at all.
        assert!(
            after > before * 0.35,
            "sheet thinned too far: {before} -> {after} fixed-point units across"
        );
    }

    /// Why uniform faces are excluded, kept executable so the exclusion cannot
    /// be quietly removed as a simplification.
    ///
    /// Averaging across a face no surface passes through joins the two sides of
    /// a one-voxel sheet, which are face-adjacent, and they meet in the middle.
    /// The displacement bound does not save it: the sides start exactly one
    /// voxel apart, so half of that is within any sane bound.
    #[test]
    fn the_literal_stencil_would_collapse_that_sheet() {
        let a = one_voxel_sheet();
        let e = extract(&a, true, 0);
        let p = Fairing {
            iterations: 80,
            max_deviation: 10.0,
            ..Default::default()
        };
        let before = spread_x(&e.cells.positions);
        let literal = fair_reference(&e.cells, &p, false);
        let ours = fair_reference(&e.cells, &p, true);
        assert!(
            spread_x(&literal) < before * 0.05,
            "expected the literal stencil to collapse the sheet, got {} of {before}",
            spread_x(&literal)
        );
        assert!(
            spread_x(&ours) > before * 0.35,
            "skipping uniform faces should have preserved it, got {} of {before}",
            spread_x(&ours)
        );
    }

    fn spread_x(positions: &[[i32; 3]]) -> f32 {
        let lo = positions.iter().map(|p| p[0]).min().unwrap();
        let hi = positions.iter().map(|p| p[0]).max().unwrap();
        (hi - lo) as f32
    }

    #[test]
    fn no_vertex_leaves_its_cell_or_the_deviation_bound() {
        let a = stacked(24);
        for &max_deviation in &[0.0f64, 0.125, 0.5, 4.0] {
            let mut e = extract(&a, true, 0);
            let placed = e.cells.positions.clone();
            fair(
                &mut e.cells,
                &Fairing {
                    iterations: 40,
                    max_deviation,
                    ..Default::default()
                },
                true,
            );
            let limit = (max_deviation * SUBVOXEL as f64).round() as i32 + 1;
            for (i, (before, after)) in placed.iter().zip(e.cells.positions.iter()).enumerate() {
                let mask = e.cells.crossings[i] as usize;
                for a in 0..3 {
                    assert!(
                        (before[a] - after[a]).abs() <= limit,
                        "cell {i} axis {a} moved past max_deviation"
                    );
                    let base = before[a] - CENTROID[mask][a];
                    assert!(
                        after[a] >= base - 1 && after[a] <= base + SUBVOXEL + 1,
                        "cell {i} axis {a} left its cell: {} not in [{base}, {}]",
                        after[a],
                        base + SUBVOXEL
                    );
                }
            }
        }
    }

    /// Chunking splits the sweep and each worker re-seeks its own cursors, so
    /// this is really a test that the seek agrees with the walk.
    #[test]
    fn fairing_is_identical_however_the_work_is_split() {
        let a = stacked(40);
        let single = {
            let mut e = extract(&a, false, 1);
            fair(&mut e.cells, &params(), true);
            e.cells.positions
        };
        for threads in [2usize, 4, 8] {
            let mut e = extract(&a, false, threads);
            fair(&mut e.cells, &params(), true);
            assert_eq!(e.cells.positions, single, "{threads} threads differed");
        }
    }

    #[test]
    fn zero_iterations_changes_nothing() {
        let a = stacked(16);
        let mut e = extract(&a, true, 0);
        let before = e.cells.positions.clone();
        fair(&mut e.cells, &Fairing::default(), true);
        assert_eq!(before, e.cells.positions);
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
            ..Default::default()
        };
        let mut e = extract_with(&VolumeView::new(a.view(), false), &opts);
        e.meshes.remove(0)
    }

    /// A sphere clear of the array edge, so nothing is pinned and the whole
    /// surface is free to move. Volume loss is only meaningful on a closed one.
    fn free_sphere(radius: f64) -> LabelMesh {
        let n = (2.0 * radius) as usize + 8;
        let c = (n as f64 - 1.0) / 2.0;
        let a = Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            let d = (i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2);
            u32::from(d <= radius * radius)
        });
        let mut e = extract_with(&VolumeView::new(a.view(), true), &ExtractOptions::default());
        e.meshes.remove(0)
    }

    /// Signed volume of the quad surface, in cubic fixed-point units. Only
    /// ratios of this are used, so the units do not matter.
    fn quad_volume(mesh: &LabelMesh) -> f64 {
        let p = |i: u32| {
            let v = mesh.positions[i as usize];
            [v[0] as f64, v[1] as f64, v[2] as f64]
        };
        let mut total = 0.0;
        for q in &mesh.quads {
            for (a, b, c) in [(q[0], q[1], q[2]), (q[0], q[2], q[3])] {
                let (a, b, c) = (p(a), p(b), p(c));
                total += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]);
            }
        }
        (total / 6.0).abs()
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
    fn taubin_never_moves_a_pinned_vertex() {
        let before = marked_tube(9.0);
        let mut after = before.clone();
        taubin(
            &mut after,
            &Taubin {
                iterations: 20,
                ..Default::default()
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

    /// The bound is the reason Taubin is safe to expose on measured data, so it
    /// is asserted at an iteration count well past where the filter converges.
    #[test]
    fn taubin_respects_the_deviation_bound() {
        let before = marked_tube(9.0);
        for &max_deviation in &[0.0f64, 0.125, 0.5] {
            let mut after = before.clone();
            taubin(
                &mut after,
                &Taubin {
                    iterations: 60,
                    max_deviation,
                    ..Default::default()
                },
            );
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

        let mut untouched = before.clone();
        taubin(&mut untouched, &Taubin::default());
        assert_eq!(before.positions, untouched.positions);

        let mut by_enum = before.clone();
        smooth(&mut by_enum, &Smoothing::None);
        assert_eq!(before.positions, by_enum.positions);
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
    fn taubin_is_order_independent() {
        let base = marked_tube(7.0);
        let params = Taubin {
            iterations: 9,
            ..Default::default()
        };
        let mut first = base.clone();
        let mut second = base.clone();
        taubin(&mut first, &params);
        taubin(&mut second, &params);
        assert_eq!(first.positions, second.positions);
    }

    /// The whole point of the negative step. Laplacian iteration is diffusion
    /// and drains volume; Taubin holds onto it, and at equal pass counts should
    /// lose distinctly less.
    #[test]
    fn taubin_loses_less_volume_than_laplacian_at_equal_passes() {
        let base = free_sphere(12.0);
        let start = quad_volume(&base);

        // A deviation bound loose enough that the filters, not the clamp, are
        // what decide the outcome.
        let mut laplacian = base.clone();
        relax(
            &mut laplacian,
            &Relaxation {
                iterations: 40,
                max_deviation: 10.0,
                step: 0.63,
            },
        );
        let mut low_pass = base.clone();
        taubin(
            &mut low_pass,
            &Taubin {
                iterations: 20, // two passes each, so 40 in total
                max_deviation: 10.0,
                ..Default::default()
            },
        );

        let laplacian_kept = quad_volume(&laplacian) / start;
        let taubin_kept = quad_volume(&low_pass) / start;
        assert!(
            taubin_kept > laplacian_kept + 0.05,
            "taubin kept {taubin_kept:.3} of the volume, laplacian {laplacian_kept:.3}"
        );
        assert!(
            taubin_kept > 0.95,
            "taubin should barely shrink, kept {taubin_kept:.3}"
        );
    }

    #[test]
    fn mu_is_negative_and_larger_than_lambda() {
        let t = Taubin::default();
        let mu = t.mu();
        assert!(mu < 0.0, "mu {mu}");
        assert!(
            mu.abs() > t.lambda,
            "|mu| {} vs lambda {}",
            mu.abs(),
            t.lambda
        );
        assert!(t.is_valid());
    }

    #[test]
    fn implausible_parameters_are_rejected() {
        for bad in [
            Taubin {
                pass_band: 0.0,
                ..Default::default()
            },
            Taubin {
                pass_band: 1.5,
                ..Default::default()
            },
            Taubin {
                lambda: 0.0,
                ..Default::default()
            },
            Taubin {
                lambda: 1.5,
                ..Default::default()
            },
            // A pass band too wide for this lambda: mu comes out below -1,
            // so the expanding pass overshoots and the filter amplifies
            // instead of smoothing.
            Taubin {
                pass_band: 0.9,
                lambda: 0.63,
                ..Default::default()
            },
        ] {
            assert!(!bad.is_valid(), "should have been rejected: {bad:?}");
        }
    }
}
