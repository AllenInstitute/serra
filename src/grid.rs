//! A zero-copy view over the caller's label array.
//!
//! The view never copies or transposes the input. It handles C-order,
//! Fortran-order and any other strided layout, because it is backed by an
//! [`ndarray::ArrayView3`].
//!
//! It also provides the *virtual* background border used by `close=True`:
//! rather than allocating a padded copy of the volume (which for a 512^3 uint32
//! array would cost an extra 540 MB), out-of-bounds samples simply read as
//! background.

use ndarray::ArrayView3;

/// A voxel label.
///
/// Only unsigned integer types are supported. Segmentation labels are unsigned
/// in practice, and admitting signed types would force a choice about what a
/// negative label means in the `uint64` id space that no caller actually wants.
pub trait Label: Copy + Eq + Ord + std::hash::Hash + Send + Sync + 'static {
    /// The background label, which is never meshed.
    const BACKGROUND: Self;
    /// Widen to the common id type used throughout the crate.
    fn as_u64(self) -> u64;
}

macro_rules! impl_label {
    ($($t:ty),* $(,)?) => {
        $(
            impl Label for $t {
                const BACKGROUND: Self = 0;
                #[inline(always)]
                fn as_u64(self) -> u64 { self as u64 }
            }
        )*
    };
}
impl_label!(u8, u16, u32, u64);

/// A borrowed 3-D label volume, addressed in array-index space.
pub struct VolumeView<'a, T: Label> {
    view: ArrayView3<'a, T>,
    close: bool,
}

impl<'a, T: Label> VolumeView<'a, T> {
    /// Wrap an array view. When `close` is set, the volume behaves as though it
    /// were surrounded by a one-voxel shell of background, so objects touching
    /// the array edge come back closed.
    pub fn new(view: ArrayView3<'a, T>, close: bool) -> Self {
        VolumeView { view, close }
    }

    /// Extent of the underlying array, in array-axis order.
    #[inline]
    pub fn shape(&self) -> [usize; 3] {
        let s = self.view.shape();
        [s[0], s[1], s[2]]
    }

    /// Whether the virtual background border is in effect.
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.close
    }

    /// Inclusive lower bound of the sampling domain along each axis.
    ///
    /// With `close`, sampling starts one voxel outside the array.
    #[inline]
    pub fn sample_lo(&self) -> [isize; 3] {
        let p = if self.close { -1 } else { 0 };
        [p; 3]
    }

    /// Exclusive upper bound of the sampling domain along each axis.
    #[inline]
    pub fn sample_hi(&self) -> [isize; 3] {
        let s = self.shape();
        let p = if self.close { 1 } else { 0 };
        [s[0] as isize + p, s[1] as isize + p, s[2] as isize + p]
    }

    /// Number of cells (2x2x2 corner blocks) along each axis.
    ///
    /// A cell exists between every pair of adjacent samples, so there is one
    /// fewer cell than samples along each axis.
    #[inline]
    pub fn cell_counts(&self) -> [usize; 3] {
        let lo = self.sample_lo();
        let hi = self.sample_hi();
        let mut out = [0usize; 3];
        for k in 0..3 {
            out[k] = (hi[k] - lo[k] - 1).max(0) as usize;
        }
        out
    }

    /// Read a voxel, returning background outside the array.
    #[inline(always)]
    pub fn sample(&self, p: [isize; 3]) -> T {
        let s = self.view.shape();
        for k in 0..3 {
            if p[k] < 0 || p[k] >= s[k] as isize {
                return T::BACKGROUND;
            }
        }
        // Safety: every index was just bounds-checked against the view's shape.
        unsafe {
            *self
                .view
                .uget([p[0] as usize, p[1] as usize, p[2] as usize])
        }
    }

    /// Gather the 8 corner labels of the cell whose minimum corner is `origin`.
    ///
    /// Corners are ordered so that the bits of the index are the per-axis
    /// offsets, matching [`crate::tables`].
    #[inline]
    pub fn cell_corners(&self, origin: [isize; 3]) -> [T; 8] {
        let mut out = [T::BACKGROUND; 8];
        for (c, slot) in out.iter_mut().enumerate() {
            *slot = self.sample([
                origin[0] + (c & 1) as isize,
                origin[1] + ((c >> 1) & 1) as isize,
                origin[2] + ((c >> 2) & 1) as isize,
            ]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, ShapeBuilder};

    fn ramp_c_order() -> Array3<u32> {
        Array3::from_shape_fn((3, 4, 5), |(i, j, k)| (i * 100 + j * 10 + k) as u32)
    }

    #[test]
    fn sampling_matches_the_array() {
        let a = ramp_c_order();
        let v = VolumeView::new(a.view(), false);
        assert_eq!(v.shape(), [3, 4, 5]);
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(
                        v.sample([i as isize, j as isize, k as isize]),
                        (i * 100 + j * 10 + k) as u32
                    );
                }
            }
        }
    }

    /// The whole point of the view: C- and Fortran-ordered arrays holding the
    /// same logical data must sample identically.
    #[test]
    fn c_and_fortran_order_sample_identically() {
        let c = ramp_c_order();
        let mut f = Array3::<u32>::zeros((3, 4, 5).f());
        f.assign(&c);
        assert!(c.is_standard_layout());
        assert!(!f.is_standard_layout());

        let vc = VolumeView::new(c.view(), false);
        let vf = VolumeView::new(f.view(), false);
        for i in 0..3isize {
            for j in 0..4isize {
                for k in 0..5isize {
                    assert_eq!(vc.sample([i, j, k]), vf.sample([i, j, k]));
                }
            }
        }
    }

    #[test]
    fn out_of_bounds_reads_as_background() {
        let a = ramp_c_order();
        let v = VolumeView::new(a.view(), false);
        assert_eq!(v.sample([-1, 0, 0]), 0);
        assert_eq!(v.sample([3, 0, 0]), 0);
        assert_eq!(v.sample([0, -1, 0]), 0);
        assert_eq!(v.sample([0, 0, 5]), 0);
    }

    #[test]
    fn close_widens_the_sampling_domain_by_one_voxel() {
        let a = ramp_c_order();

        let open = VolumeView::new(a.view(), false);
        assert_eq!(open.sample_lo(), [0, 0, 0]);
        assert_eq!(open.sample_hi(), [3, 4, 5]);
        assert_eq!(open.cell_counts(), [2, 3, 4]);

        let closed = VolumeView::new(a.view(), true);
        assert_eq!(closed.sample_lo(), [-1, -1, -1]);
        assert_eq!(closed.sample_hi(), [4, 5, 6]);
        // Two extra samples per axis means two extra cells per axis.
        assert_eq!(closed.cell_counts(), [4, 5, 6]);
    }

    #[test]
    fn cell_corners_are_indexed_by_axis_bits() {
        let a = ramp_c_order();
        let v = VolumeView::new(a.view(), false);
        let corners = v.cell_corners([1, 1, 1]);
        // Corner c sits at offset (c&1, (c>>1)&1, (c>>2)&1) from the origin.
        assert_eq!(corners[0], 111); // (1,1,1)
        assert_eq!(corners[1], 211); // (2,1,1)
        assert_eq!(corners[2], 121); // (1,2,1)
        assert_eq!(corners[4], 112); // (1,1,2)
        assert_eq!(corners[7], 222); // (2,2,2)
    }

    #[test]
    fn a_degenerate_axis_yields_no_cells() {
        let a = Array3::<u32>::zeros((1, 4, 5));
        let v = VolumeView::new(a.view(), false);
        assert_eq!(v.cell_counts(), [0, 3, 4]);
    }
}
