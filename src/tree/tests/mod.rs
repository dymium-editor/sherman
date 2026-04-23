//! Various collections of unit tests
//!
//! The bulk of our confidence in correctness comes from fuzzing, but each failure from fuzzing is
//! used as a new unit tests, so that there's a moderate corpus here as well.

#[cfg(feature = "fuzz")]
use arbitrary::{Arbitrary, Unstructured};
#[cfg(feature = "fuzz")]
use std::fmt;

#[cfg(any(test, feature = "fuzz"))]
use crate::Slice;
#[cfg(feature = "fuzz")]
use crate::fuzz::{RustExpr, RustType};

/// Manually written test cases
#[cfg(test)]
mod manual;

#[cfg(test)]
mod fuzz_basic;
#[cfg(test)]
mod fuzz_basic_range;
#[cfg(test)]
mod fuzz_checked;
#[cfg(test)]
mod fuzz_multi_cow;
#[cfg(test)]
mod fuzz_slice_ref;
#[cfg(test)]
mod fuzz_slice_ref_range;

/// Helper type for fuzzing - a [`Slice`] implementation that joins continuous character ranges
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(any(test, feature = "fuzz"))]
pub struct CharRange(pub(crate) std::ops::Range<char>);

#[cfg(any(test, feature = "fuzz"))]
impl<I> Slice<I> for CharRange
where
    I: Into<u64>,
{
    fn split_at(self, idx: I) -> (Self, Self) {
        let idx = idx.into() as u8;
        let mid = (self.0.start as u8 + idx) as char;
        assert!(mid < self.0.end);
        (CharRange(self.0.start..mid), CharRange(mid..self.0.end))
    }

    fn try_join(self, other: Self) -> Result<Self, (Self, Self)> {
        if self.0.end == other.0.start {
            Ok(CharRange(self.0.start..other.0.end))
        } else {
            Err((self, other))
        }
    }
}

#[cfg(feature = "fuzz")]
impl<'d> Arbitrary<'d> for CharRange {
    fn arbitrary(u: &mut Unstructured<'d>) -> arbitrary::Result<Self> {
        let start_idx = u.int_in_range(0_u8..=24)?;
        let start = (b'A' + start_idx) as char;
        let end_idx = u.int_in_range(start_idx + 1..=25)?;
        let end = (b'A' + end_idx) as char;
        Ok(CharRange(start..end))
    }
}

#[cfg(feature = "fuzz")]
impl RustType for CharRange {
    fn write_rust_type(f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("CharRange")
    }
}

#[cfg(feature = "fuzz")]
impl RustExpr for CharRange {
    fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("CharRange({})", self.0.display_rust_expr()))
    }
}
