//! Turning one label's raw surface into a triangle mesh for the caller.
//!
//! This is where fixed-point index coordinates become physical floating-point
//! ones, quads become triangles, and axis order is applied.

use crate::extract::LabelMesh;
use crate::orient::Layout;
use crate::tables::SUBVOXEL;

/// A triangle mesh in physical coordinates.
#[derive(Default, Clone)]
pub struct TriangleMesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
}

/// How to interpret the extracted surface.
#[derive(Clone, Copy)]
pub struct MeshOptions {
    /// Physical size of a voxel along each **array** axis.
    pub resolution: [f64; 3],
    /// Array-axis to physical-axis mapping.
    pub layout: Layout,
    /// Extent of the input array, needed only when an axis is flipped.
    pub shape: [usize; 3],
    /// Whether to compute per-vertex normals.
    pub normals: bool,
}

impl Default for MeshOptions {
    fn default() -> Self {
        MeshOptions {
            resolution: [1.0; 3],
            layout: Layout::default(),
            shape: [0; 3],
            normals: false,
        }
    }
}

#[inline]
fn dist2(a: &[i32; 3], b: &[i32; 3]) -> i64 {
    let mut s = 0i64;
    for k in 0..3 {
        let d = a[k] as i64 - b[k] as i64;
        s += d * d;
    }
    s
}

/// Which diagonal to split a quad along.
///
/// The shorter diagonal wins, which keeps triangles well shaped. The subtlety
/// is that a wall between two labels is emitted twice, once per label, with the
/// ring reversed — and reversing swaps which index pair *names* each diagonal.
/// The rule below is stated purely in terms of the two diagonals as unordered
/// point pairs, so both copies of the wall pick the same one geometrically and
/// the two surfaces stay exactly coincident.
#[inline]
fn split_along_first_diagonal(p: [&[i32; 3]; 4]) -> bool {
    let da = dist2(p[0], p[2]);
    let db = dist2(p[1], p[3]);
    if da != db {
        return da < db;
    }
    // Exact tie: order the two diagonals canonically and take the smaller.
    let canon = |x: &[i32; 3], y: &[i32; 3]| if x <= y { (*x, *y) } else { (*y, *x) };
    canon(p[0], p[2]) < canon(p[1], p[3])
}

/// Build a triangle mesh from one label's extracted surface.
///
/// Vertices that no quad references — which happens where the surface was left
/// open at the volume boundary — are dropped.
pub fn build(raw: &LabelMesh, opts: &MeshOptions) -> TriangleMesh {
    if raw.quads.is_empty() {
        return TriangleMesh::default();
    }

    // --- triangulate, in index space so the choice is exact ------------------
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(raw.quads.len() * 2);
    for q in &raw.quads {
        let p = [
            &raw.positions[q[0] as usize],
            &raw.positions[q[1] as usize],
            &raw.positions[q[2] as usize],
            &raw.positions[q[3] as usize],
        ];
        if split_along_first_diagonal(p) {
            faces.push([q[0], q[1], q[2]]);
            faces.push([q[0], q[2], q[3]]);
        } else {
            faces.push([q[1], q[2], q[3]]);
            faces.push([q[1], q[3], q[0]]);
        }
    }

    // --- drop unreferenced vertices, preserving order ------------------------
    let mut remap = vec![u32::MAX; raw.positions.len()];
    let mut kept: Vec<u32> = Vec::new();
    for f in &faces {
        for &v in f {
            if remap[v as usize] == u32::MAX {
                remap[v as usize] = kept.len() as u32;
                kept.push(v);
            }
        }
    }

    // --- index space -> physical space ---------------------------------------
    let scale = SUBVOXEL as f64;
    let vertices: Vec<[f32; 3]> = kept
        .iter()
        .map(|&v| {
            let p = &raw.positions[v as usize];
            let voxel = [
                p[0] as f64 / scale,
                p[1] as f64 / scale,
                p[2] as f64 / scale,
            ];
            let phys = opts.layout.to_physical(voxel, opts.resolution, opts.shape);
            [phys[0] as f32, phys[1] as f32, phys[2] as f32]
        })
        .collect();

    // A handedness-reversing axis mapping would leave every normal pointing
    // inward, so the winding is reversed to compensate.
    let flip = opts.layout.inverts_orientation();
    let faces: Vec<[u32; 3]> = faces
        .iter()
        .map(|f| {
            let (a, b, c) = (
                remap[f[0] as usize],
                remap[f[1] as usize],
                remap[f[2] as usize],
            );
            if flip {
                [a, c, b]
            } else {
                [a, b, c]
            }
        })
        .collect();

    let normals = if opts.normals {
        Some(vertex_normals(&vertices, &faces))
    } else {
        None
    };

    TriangleMesh {
        vertices,
        faces,
        normals,
    }
}

/// Area-weighted vertex normals.
///
/// The cross product of two triangle edges already has magnitude proportional
/// to twice the triangle's area, so accumulating it unnormalised weights each
/// face by its area for free.
pub fn vertex_normals(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0f64; 3]; vertices.len()];
    for f in faces {
        let a = vertices[f[0] as usize];
        let b = vertices[f[1] as usize];
        let c = vertices[f[2] as usize];
        let ab = [
            b[0] as f64 - a[0] as f64,
            b[1] as f64 - a[1] as f64,
            b[2] as f64 - a[2] as f64,
        ];
        let ac = [
            c[0] as f64 - a[0] as f64,
            c[1] as f64 - a[1] as f64,
            c[2] as f64 - a[2] as f64,
        ];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for &vi in f {
            let slot = &mut acc[vi as usize];
            for k in 0..3 {
                slot[k] += n[k];
            }
        }
    }
    acc.into_iter()
        .map(|n| {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 0.0 {
                [
                    (n[0] / len) as f32,
                    (n[1] / len) as f32,
                    (n[2] / len) as f32,
                ]
            } else {
                [0.0; 3]
            }
        })
        .collect()
}

/// Signed volume enclosed by a closed mesh, via the divergence theorem.
pub fn enclosed_volume(m: &TriangleMesh) -> f64 {
    let mut total = 0f64;
    for f in &m.faces {
        let a = m.vertices[f[0] as usize];
        let b = m.vertices[f[1] as usize];
        let c = m.vertices[f[2] as usize];
        let cross = [
            b[1] as f64 * c[2] as f64 - b[2] as f64 * c[1] as f64,
            b[2] as f64 * c[0] as f64 - b[0] as f64 * c[2] as f64,
            b[0] as f64 * c[1] as f64 - b[1] as f64 * c[0] as f64,
        ];
        total += a[0] as f64 * cross[0] + a[1] as f64 * cross[1] + a[2] as f64 * cross[2];
    }
    total / 6.0
}

/// Total surface area.
pub fn surface_area(m: &TriangleMesh) -> f64 {
    let mut total = 0f64;
    for f in &m.faces {
        let a = m.vertices[f[0] as usize];
        let b = m.vertices[f[1] as usize];
        let c = m.vertices[f[2] as usize];
        let ab = [
            b[0] as f64 - a[0] as f64,
            b[1] as f64 - a[1] as f64,
            b[2] as f64 - a[2] as f64,
        ];
        let ac = [
            c[0] as f64 - a[0] as f64,
            c[1] as f64 - a[1] as f64,
            c[2] as f64 - a[2] as f64,
        ];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        total += 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::extract;
    use crate::grid::VolumeView;
    use ndarray::Array3;
    use rustc_hash::FxHashMap;

    /// How many undirected edges are not shared by exactly two triangles.
    fn non_manifold_edges(m: &TriangleMesh) -> usize {
        let mut counts: FxHashMap<(u32, u32), i32> = FxHashMap::default();
        for f in &m.faces {
            for i in 0..3 {
                let (a, b) = (f[i], f[(i + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts.values().filter(|&&c| c != 2).count()
    }

    fn sphere(radius: f64) -> Array3<u32> {
        let n = (2.0 * radius) as usize + 8;
        let centre = (n as f64 - 1.0) / 2.0;
        Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            let d = (i as f64 - centre).powi(2)
                + (j as f64 - centre).powi(2)
                + (k as f64 - centre).powi(2);
            u32::from(d <= radius * radius)
        })
    }

    fn mesh_sphere(radius: f64) -> TriangleMesh {
        let a = sphere(radius);
        let view = VolumeView::new(a.view(), true);
        let e = extract(&view);
        let opts = MeshOptions {
            shape: [a.shape()[0], a.shape()[1], a.shape()[2]],
            ..Default::default()
        };
        build(e.mesh(1).unwrap(), &opts)
    }

    /// The headline claim. Marching cubes overstates a sphere's area by ~9%
    /// and does not improve with resolution; local dual placement should land
    /// near 3% and stay there, with volume essentially exact.
    #[test]
    fn sphere_volume_and_area_match_analytic() {
        for &r in &[10.0f64, 20.0] {
            let m = mesh_sphere(r);
            assert_eq!(non_manifold_edges(&m), 0, "r={r} must be closed");

            let vol = enclosed_volume(&m);
            let area = surface_area(&m);
            let vol_exact = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
            let area_exact = 4.0 * std::f64::consts::PI * r * r;

            let vol_err = vol / vol_exact - 1.0;
            let area_err = area / area_exact - 1.0;
            assert!(vol_err.abs() < 0.01, "r={r}: volume error {vol_err:+.4}");
            assert!(
                (0.0..0.04).contains(&area_err),
                "r={r}: area error {area_err:+.4}"
            );
        }
    }

    /// Prints the accuracy table used to compare against zmesh and against the
    /// numpy prototype the design was validated with. Run with:
    /// `cargo test --no-default-features accuracy_table -- --ignored --nocapture`
    #[test]
    #[ignore = "reporting only"]
    fn accuracy_table() {
        println!(
            "{:>3}  {:>10}  {:>10}  {:>9}  {:>8}",
            "R", "vol err", "area err", "tris", "tri-CV"
        );
        for &r in &[10.0f64, 20.0, 40.0] {
            let m = mesh_sphere(r);
            let vol_err =
                enclosed_volume(&m) / (4.0 / 3.0 * std::f64::consts::PI * r.powi(3)) - 1.0;
            let area_err = surface_area(&m) / (4.0 * std::f64::consts::PI * r * r) - 1.0;

            let areas: Vec<f64> = m
                .faces
                .iter()
                .map(|f| {
                    let one = TriangleMesh {
                        vertices: m.vertices.clone(),
                        faces: vec![*f],
                        normals: None,
                    };
                    surface_area(&one)
                })
                .collect();
            let mean = areas.iter().sum::<f64>() / areas.len() as f64;
            let var = areas.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / areas.len() as f64;
            println!(
                "{:>3}  {:>+9.2}%  {:>+9.2}%  {:>9}  {:>8.3}",
                r,
                vol_err * 100.0,
                area_err * 100.0,
                m.faces.len(),
                var.sqrt() / mean
            );
        }
    }

    /// Volume must come out positive, which it only does if quads are wound so
    /// their normals point out of the object.
    #[test]
    fn winding_puts_normals_outward() {
        let m = mesh_sphere(10.0);
        assert!(enclosed_volume(&m) > 0.0);

        let normals = vertex_normals(&m.vertices, &m.faces);
        // The sphere is centred in its array, so an outward normal agrees with
        // the direction from the centre to the vertex.
        let c = m
            .vertices
            .iter()
            .fold([0f64; 3], |mut acc, v| {
                for k in 0..3 {
                    acc[k] += v[k] as f64;
                }
                acc
            })
            .map(|s| s / m.vertices.len() as f64);
        let mut agree = 0usize;
        for (v, n) in m.vertices.iter().zip(normals.iter()) {
            let radial = [v[0] as f64 - c[0], v[1] as f64 - c[1], v[2] as f64 - c[2]];
            let dot = radial[0] * n[0] as f64 + radial[1] * n[1] as f64 + radial[2] * n[2] as f64;
            if dot > 0.0 {
                agree += 1;
            }
        }
        assert_eq!(agree, m.vertices.len(), "every normal should point outward");
    }

    /// An orientation-reversing axis order must not turn the object inside out.
    #[test]
    fn reversed_axis_order_keeps_volume_positive() {
        let a = sphere(10.0);
        let shape = [a.shape()[0], a.shape()[1], a.shape()[2]];
        let e = extract(&VolumeView::new(a.view(), true));
        let raw = e.mesh(1).unwrap();

        let plain = build(
            raw,
            &MeshOptions {
                shape,
                ..Default::default()
            },
        );
        let swapped = build(
            raw,
            &MeshOptions {
                layout: Layout::from_axis_order("ZYX").unwrap(),
                shape,
                ..Default::default()
            },
        );
        // Same geometry, so the same enclosed volume.
        let (a_vol, b_vol) = (enclosed_volume(&plain), enclosed_volume(&swapped));
        assert!(b_vol > 0.0, "volume flipped sign: {b_vol}");
        assert!((a_vol - b_vol).abs() / a_vol < 1e-4, "{a_vol} vs {b_vol}");
    }

    #[test]
    fn anisotropic_resolution_scales_volume() {
        let a = sphere(10.0);
        let shape = [a.shape()[0], a.shape()[1], a.shape()[2]];
        let e = extract(&VolumeView::new(a.view(), true));
        let raw = e.mesh(1).unwrap();

        let iso = build(
            raw,
            &MeshOptions {
                shape,
                ..Default::default()
            },
        );
        let aniso = build(
            raw,
            &MeshOptions {
                resolution: [4.0, 4.0, 40.0],
                shape,
                ..Default::default()
            },
        );
        // Scaling the axes scales the volume by the product of the factors.
        let expected = enclosed_volume(&iso) * 4.0 * 4.0 * 40.0;
        let got = enclosed_volume(&aniso);
        assert!(
            (got - expected).abs() / expected < 1e-4,
            "{got} vs {expected}"
        );
    }

    /// Both sides of a shared wall must be triangulated along the same
    /// diagonal, or the two surfaces would not be geometrically coincident.
    #[test]
    fn a_shared_wall_is_triangulated_identically_from_both_sides() {
        let mut a = Array3::<u32>::zeros((10, 7, 7));
        for i in 1..4 {
            for j in 1..6 {
                for k in 1..6 {
                    a[[i, j, k]] = 1;
                    a[[i + 3, j, k]] = 2;
                }
            }
        }
        let e = extract(&VolumeView::new(a.view(), true));
        let shape = [a.shape()[0], a.shape()[1], a.shape()[2]];
        let opts = MeshOptions {
            shape,
            ..Default::default()
        };
        let m1 = build(e.mesh(1).unwrap(), &opts);
        let m2 = build(e.mesh(2).unwrap(), &opts);

        // Collect each mesh's triangles as sorted vertex-position triples and
        // check that the wall's triangles appear in both.
        let tris = |m: &TriangleMesh| {
            let mut out: Vec<[[i64; 3]; 3]> = m
                .faces
                .iter()
                .map(|f| {
                    let mut t = f.map(|v| {
                        let p = m.vertices[v as usize];
                        [
                            (p[0] as f64 * 256.0).round() as i64,
                            (p[1] as f64 * 256.0).round() as i64,
                            (p[2] as f64 * 256.0).round() as i64,
                        ]
                    });
                    t.sort();
                    t
                })
                .collect::<Vec<_>>();
            out.sort();
            out
        };
        let t1 = tris(&m1);
        let t2 = tris(&m2);
        let shared = t1.iter().filter(|t| t2.binary_search(t).is_ok()).count();
        // The interface spans j, k in 1..6, so 5x5 = 25 voxel faces, giving
        // 25 quads and 50 triangles. Every one must appear in both meshes.
        assert_eq!(shared, 50, "shared wall triangles must coincide exactly");
    }
}
