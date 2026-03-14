//! # Sherman — a truly monstrous tree type
//!
//! It turns out that in the production of an editor, sometimes specialized data structures are
//! used. Sometimes data structures are similar enough that they can be united into a single
//! abstract type, with parameterizations to suit the original needs. This crate primarily exports
//! a single type -- [`RleTree`] -- which represents the collection of all the features we needed.
//!
//! ### Notable features
//!
//! * Values are retrieved by global index
//! * [`RleTree`] is named such for its run-length encoding -- individual entries in the tree
//!     represent a uniform range of indexes
//! * Efficient "shift" operations -- the details of the run-length encoding allow new ranges to be
//!     inserted in the middle, shifting everything after them, in O(log n) time
//! * Slice references -- the current position and value of a prior insertion can be fetched in
//!     O(log n) time, with relatively little overhead (*conflicts with COW*)
//! * Wait-free concurrent clone-on-write -- [`RleTree`]s can be shared across threads, with
//!     concurrent writes cloning only the path down to the changed node(s). (*conflicts with slice
//!     references*)
//!
//! And of course, all of these features are zero-cost when not in use: the tree is constructed in
//! such a way so that only the instances that actually *do* use these extra feature (like node
//! references or concurrent COW) have to pay the cost of them.
//!
//! ### Feature flags
//!
//! This crate provides the following _public_ feature flags:
//!
//! * `serde` — *opt-in*, enables [`serde`] support
//! * `nightly` — *opt-in*, enables some minor improvements (notably: implementing `Drop` for
//!   [`RleTree`] with `#[may_dangle]`). Requires nightly rustc.
//!
//! We also use the `fuzz` feature flag, just for testing.
//!
//! ### Naming
//!
//! This library is named after [General Sherman], a tree in Sequoia National Park that's the
//! current largest tree on Earth by volume.
//!
//! [`serde`]: https://docs.rs/serde
//! [General Sherman]: https://en.wikipedia.org/wiki/General_Sherman_(tree)

#![deny(unsafe_op_in_unsafe_fn, rustdoc::broken_intra_doc_links)]
#![cfg_attr(
    feature = "nightly",
    allow(incomplete_features),
    feature(dropck_eyepatch, specialization)
)]

use std::fmt::{self, Debug, Formatter};

#[macro_use]
mod macros;

#[cfg(feature = "fuzz")]
pub mod fuzz;

mod public_traits;
mod tree;

pub use public_traits::{DirectionalAdd, DirectionalSub, Index, Slice, Zero};
pub use tree::RleTree;

/// Helper implementation of [`Slice`] for *actual* run-length encoding - a run of identical values
///
/// Many usages of [`RleTree`] implement [`Slice`] in ways that go beyond traditional run-length
/// encoding -- only really using it as an easy way to manage splitting and joining of similar
/// nodes.
///
/// `Constant` exists purely for this "plain" run-length encoding usage. It splits by cloning the
/// inner value, and joins when the two values are the same.
///
/// All the derivable traits are provided for `Constant`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "fuzz", derive(arbitrary::Arbitrary))]
pub struct Constant<T>(pub T);

impl<Idx, T: Clone + PartialEq> Slice<Idx> for Constant<T> {
    fn split_at(self, _idx: Idx) -> (Self, Self) {
        (self.clone(), self)
    }

    fn try_join(self, other: Self) -> Result<Self, (Self, Self)> {
        if self == other {
            Ok(self)
        } else {
            Err((self, other))
        }
    }
}

#[cfg(feature = "fuzz")]
impl<T: fuzz::RustType> fuzz::RustType for Constant<T> {
    fn write_rust_type(f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("Constant<{T}>", T = T::display_rust_type()))
    }
}

#[cfg(feature = "fuzz")]
impl<T: fuzz::RustExpr> fuzz::RustExpr for Constant<T> {
    fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("Constant({})", self.0.display_rust_expr()))
    }
}

#[cold]
#[track_caller]
pub(crate) fn panic_internal_error_or_bad_index<I: Index>(msg: &str) -> ! {
    if I::TRUSTED {
        panic!("internal error: {msg}");
    } else {
        panic!("internal error or bad `Index` implementation: {msg}");
    }
}

/// Crate-internal trait for allowing introspection into debuggable types without introducing new
/// bounds.
///
/// This is only usefully implemented during tests with `feature = "nightly"`.
#[allow(unused)]
trait MaybeDebug {
    /// If `Self` implements `Debug`, provides access to the `Debug` implementation
    fn try_debug(&self) -> Option<&dyn Debug>;

    fn fallible_debug(&self) -> &dyn Debug {
        self.try_debug().unwrap_or(&NoDebugImpl)
    }
}

#[cfg(not(all(feature = "nightly", test)))]
impl<T> MaybeDebug for T {
    fn try_debug(&self) -> Option<&dyn Debug> {
        None
    }
}
#[cfg(all(feature = "nightly", test))]
impl<T> MaybeDebug for T {
    default fn try_debug(&self) -> Option<&dyn Debug> {
        None
    }
}

#[cfg(all(feature = "nightly", test))]
impl<T: Debug> MaybeDebug for T {
    fn try_debug(&self) -> Option<&dyn Debug> {
        Some(self)
    }
}

struct NoDebugImpl;

impl Debug for NoDebugImpl {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str("<No Debug impl>")
    }
}
