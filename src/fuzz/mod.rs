//! Fuzzing related utilities

#[cfg(feature = "fuzz")]
use arbitrary::{Arbitrary, Unstructured};
use std::fmt::{self, Debug};

#[cfg(feature = "fuzz")]
mod arbitrary_operation;
mod as_rust;
mod fake;
mod tracked_index;

pub use crate::tree::tests::CharRange;
#[cfg(feature = "fuzz")]
pub use arbitrary_operation::{
    ArbitraryOp, BasicOperation, CheckedIndexOperation, MultiCowOperation, OpSequence,
    SliceRefOperation,
};
pub use as_rust::{RustExpr, RustType};
pub use fake::{Fake, FakeSliceRef};
pub use tracked_index::{IndexInfo, TrackedIndex, TrackedSlice};

/// Helper type for fuzzing - restricted character set that pretends it's `char`
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct UpperLetter(char);

impl Debug for UpperLetter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

#[cfg(feature = "fuzz")]
impl<'d> Arbitrary<'d> for UpperLetter {
    fn arbitrary(u: &mut Unstructured<'d>) -> arbitrary::Result<Self> {
        let idx = u.int_in_range(0_u8..=25)?;
        Ok(UpperLetter((b'A' + idx) as char))
    }
}

impl RustType for UpperLetter {
    fn write_rust_type(f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("char")
    }
}

impl RustExpr for UpperLetter {
    fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result {
        RustExpr::write_rust_expr(&self.0, f)
    }
}
