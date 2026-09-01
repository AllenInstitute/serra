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

use crate::extract::LabelMesh;
use crate::tables::SUBVOXEL;

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
}

impl Smoothing {
    /// Whether anything will actually run. Used to decide whether the extractor
    /// needs to mark the chunk's outermost cell layer as pinned.
    pub fn is_active(&self) -> bool {
        match self {
            Smoothing::None => false,
            Smoothing::Laplacian(r) => r.iterations > 0,
            Smoothing::Taubin(t) => t.iterations > 0,
        }
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
