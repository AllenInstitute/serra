//! Axis order and handedness.
//!
//! The extractor works entirely in array-index space. Orientation is applied
//! once, when a label's mesh is handed back to the caller: vertex coordinates
//! are permuted and scaled, and — if the mapping inverts handedness — triangle
//! winding is reversed so that normals still point out of the object.
//!
//! Doing it this way keeps the hot loop free of any orientation logic.

use std::fmt;

/// How the caller's array axes map onto physical X/Y/Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// `axis_of[i]` is the physical axis (0 = X, 1 = Y, 2 = Z) that array axis
    /// `i` corresponds to.
    pub axis_of: [usize; 3],
    /// `flip[i]` marks an array axis that runs opposite to its physical axis.
    /// The flip is taken about the array's extent, so coordinates stay
    /// non-negative.
    pub flip: [bool; 3],
}

/// Reasons an axis-order string can be rejected.
#[derive(Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// The string was not three characters long.
    WrongLength(usize),
    /// The string contained something other than X, Y or Z.
    BadAxis(char),
    /// An axis appeared more than once.
    RepeatedAxis(char),
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::WrongLength(n) => {
                write!(f, "axis_order must name exactly 3 axes, got {n}")
            }
            LayoutError::BadAxis(c) => {
                write!(f, "axis_order may only contain X, Y and Z, got {c:?}")
            }
            LayoutError::RepeatedAxis(c) => {
                write!(f, "axis_order repeats the axis {c:?}")
            }
        }
    }
}

impl Default for Layout {
    fn default() -> Self {
        Layout {
            axis_of: [0, 1, 2],
            flip: [false; 3],
        }
    }
}

impl Layout {
    /// Parse an axis order such as `"XYZ"` or `"ZYX"`, case-insensitively.
    ///
    /// The i'th character names the physical axis that array axis `i` runs
    /// along, so `"ZYX"` describes an array indexed `[z, y, x]`.
    pub fn from_axis_order(order: &str) -> Result<Self, LayoutError> {
        let chars: Vec<char> = order.chars().collect();
        if chars.len() != 3 {
            return Err(LayoutError::WrongLength(chars.len()));
        }
        let mut axis_of = [0usize; 3];
        let mut seen = [false; 3];
        for (i, c) in chars.iter().enumerate() {
            let a = match c.to_ascii_uppercase() {
                'X' => 0,
                'Y' => 1,
                'Z' => 2,
                other => return Err(LayoutError::BadAxis(other)),
            };
            if seen[a] {
                return Err(LayoutError::RepeatedAxis(c.to_ascii_uppercase()));
            }
            seen[a] = true;
            axis_of[i] = a;
        }
        Ok(Layout {
            axis_of,
            flip: [false; 3],
        })
    }

    /// Mark the array axis that maps to physical `axis` as running backwards.
    /// Used for the "Y increases downward" image convention.
    pub fn with_flipped_physical_axis(mut self, axis: usize) -> Self {
        for i in 0..3 {
            if self.axis_of[i] == axis {
                self.flip[i] = !self.flip[i];
            }
        }
        self
    }

    /// Whether this mapping reverses handedness, in which case triangle winding
    /// must be reversed to keep normals pointing outward.
    ///
    /// The determinant of the mapping is the sign of the axis permutation times
    /// one factor of -1 per flipped axis.
    pub fn inverts_orientation(&self) -> bool {
        let mut inversions = 0;
        for i in 0..3 {
            for j in (i + 1)..3 {
                if self.axis_of[i] > self.axis_of[j] {
                    inversions += 1;
                }
            }
        }
        let flips = self.flip.iter().filter(|&&f| f).count();
        (inversions + flips) % 2 == 1
    }

    /// Map a position given in array-index space to physical space.
    ///
    /// `coord` is in array-axis order and may be fractional; `resolution` is the
    /// physical size of a voxel along each **array** axis; `shape` is the array's
    /// extent, needed only to flip about the correct centre.
    pub fn to_physical(
        &self,
        coord: [f64; 3],
        resolution: [f64; 3],
        shape: [usize; 3],
    ) -> [f64; 3] {
        let mut out = [0.0f64; 3];
        for i in 0..3 {
            let c = if self.flip[i] {
                (shape[i] as f64 - 1.0) - coord[i]
            } else {
                coord[i]
            };
            out[self.axis_of[i]] = c * resolution[i];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xyz_is_the_identity() {
        let l = Layout::from_axis_order("XYZ").unwrap();
        assert_eq!(l, Layout::default());
        assert!(!l.inverts_orientation());
        assert_eq!(
            l.to_physical([1.0, 2.0, 3.0], [4.0, 4.0, 40.0], [10, 10, 10]),
            [4.0, 8.0, 120.0]
        );
    }

    #[test]
    fn zyx_swaps_x_and_z_and_inverts_handedness() {
        let l = Layout::from_axis_order("ZYX").unwrap();
        assert_eq!(l.axis_of, [2, 1, 0]);
        // A single transposition is an odd permutation.
        assert!(l.inverts_orientation());
        // Array coord (1,2,3) with per-array-axis resolution (40,4,4) lands at
        // physical X=3*4, Y=2*4, Z=1*40.
        assert_eq!(
            l.to_physical([1.0, 2.0, 3.0], [40.0, 4.0, 4.0], [10, 10, 10]),
            [12.0, 8.0, 40.0]
        );
    }

    #[test]
    fn cyclic_permutations_preserve_handedness() {
        for order in ["XYZ", "YZX", "ZXY"] {
            assert!(
                !Layout::from_axis_order(order)
                    .unwrap()
                    .inverts_orientation(),
                "{order} is an even permutation"
            );
        }
        for order in ["ZYX", "XZY", "YXZ"] {
            assert!(
                Layout::from_axis_order(order)
                    .unwrap()
                    .inverts_orientation(),
                "{order} is an odd permutation"
            );
        }
    }

    #[test]
    fn a_flip_toggles_handedness() {
        let l = Layout::from_axis_order("XYZ").unwrap();
        assert!(!l.inverts_orientation());
        let f = l.with_flipped_physical_axis(1);
        assert!(f.inverts_orientation());
        // Flipping twice returns to the original handedness.
        assert!(!f.with_flipped_physical_axis(1).inverts_orientation());
    }

    #[test]
    fn flip_mirrors_about_the_array_extent() {
        let l = Layout::from_axis_order("XYZ")
            .unwrap()
            .with_flipped_physical_axis(1);
        // With extent 10 along y, index 0 maps to the far end and index 9 to 0.
        assert_eq!(
            l.to_physical([0.0, 0.0, 0.0], [1.0; 3], [10, 10, 10])[1],
            9.0
        );
        assert_eq!(
            l.to_physical([0.0, 9.0, 0.0], [1.0; 3], [10, 10, 10])[1],
            0.0
        );
    }

    #[test]
    fn bad_axis_orders_are_rejected() {
        assert_eq!(
            Layout::from_axis_order("XY"),
            Err(LayoutError::WrongLength(2))
        );
        assert_eq!(
            Layout::from_axis_order("XYW"),
            Err(LayoutError::BadAxis('W'))
        );
        assert_eq!(
            Layout::from_axis_order("XXY"),
            Err(LayoutError::RepeatedAxis('X'))
        );
    }

    #[test]
    fn lowercase_is_accepted() {
        assert_eq!(
            Layout::from_axis_order("zyx").unwrap(),
            Layout::from_axis_order("ZYX").unwrap()
        );
    }
}
