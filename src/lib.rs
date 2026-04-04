//! # Sherman — a big tree type
//!
//! This crate provides the [`RleTree`] type, a generalized run-length encoded binary search tree
//! with `O(log(n))` point lookup, range insertion, and range deletion that shifts the indexes of
//! all following ranges on update.
//!
//! One way of thinking about [`RleTree`] is that it's like a [rope] that's generic over the value
//! type and the index, which opens up a handful of other opportunities (See "Motivation" below).
//!
//! [rope]: https://en.wikipedia.org/wiki/Rope_(data_structure)
//!
//! ## `RleTree` feature summary
//!
//! * Values of an `RleTree` are retrieved by global index
//! * Individual entries in the `RleTree` represent a uniform range of indexes
//! * Efficient "shift" operations — ranges may be inserted or removed in the middle, shifting the
//!   positions of everything after them, in `O(log(n))` time.
//! * Customizable index types; see [`Index`]
//! * Customizable value types, so long as they can be split and maybe merged; see [`Slice`]
//
// TODO: When (re-)implemented:
//
// * Slice references -- the current position and value of a prior insertion can be fetched in
//   O(log n) time, with relatively little overhead (*conflicts with COW*)
// * Wait-free concurrent clone-on-write -- [`RleTree`]s can be shared across threads, with
//   concurrent writes cloning only the path down to the changed node(s). (*conflicts with slice
//   references*)
//
// And of course, all of these features are zero-cost when not in use: the tree is constructed in
// such a way so that only the instances that actually *do* use these extra feature (like node
// references or concurrent COW) have to pay the cost of them.
//
//!
//! ## Feature flags
//!
//! This crate provides the following public feature flags:
//!
//! * `nightly` — *opt-in*, enables some minor improvements available behind nightly features
//!   (notably: implementing `Drop` for [`RleTree`] with `#[may_dangle]`).
//!
//! Internally, the `fuzz` feature flag is used for testing.
//!
//! ## Testing
//!
//! `RleTree` is a very complicated data structure that makes judicious use of unsafe Rust. It's
//! reasonable to ask how we ensure correctness. We generally achieve this through two mechanisms:
//!
//! 1. Fuzz-based testing against a simpler, less efficient implementation
//! 2. Testing with [`miri`] (under `-Zmiri-tree-borrows`)
//!
//! [`miri`]: https://github.com/rust-lang/miri
//!
//! The fuzz testing uses a specialized type adapted to [`cargo-fuzz`] that represents a sequence
//! of operations against an `RleTree`. Each fuzz target runs compares the results of the
//! operations (e.g., [`get`], [`iter`], [`drain`]) against a simpler implementation of the same
//! interface, backed by a `Vec` of the value ranges. The sequence of operations also implements
//! `Debug` to produce a runnable unit test, which is where most of our tests come from.
//!
//! [`cargo-fuzz`]: https://github.com/rust-fuzz/cargo-fuzz
//! [`get`]: RleTree::get
//! [`iter`]: RleTree::iter
//! [`drain`]: RleTree::drain
//!
//! ## Motivation
//!
//! The `RleTree` data structure arose in the context of building a text editor, where there are
//! many natural needs to represent ranges of values over the span of a file's contents.
//!
//! For example, we might want to tag each byte in a file with a unique identifier for the last
//! edit that touched it. Or we might want to periodically cache the syntax highlighting state at
//! various points in the file, so that we can quickly re-validate the cache when a small change is
//! made. Or we can even represent the file content itself, using a `Slice` that is itself a chunk
//! of bytes with limited size (effectively just a rope).
//!
//! It's not just limited to text editors, though! Any index type that naturally forms ranges is a
//! good candidate here — for example, IP addresses / CIDR blocks, or the key space of a database.
//!
//! ## Naming
//!
//! This library is named after [General Sherman], the largest tree by volume (on Earth, at time of
//! writing).
//!
//! [General Sherman]: https://en.wikipedia.org/wiki/General_Sherman_(tree)

#![deny(unsafe_op_in_unsafe_fn, rustdoc::broken_intra_doc_links)]
#![cfg_attr(feature = "nightly", feature(dropck_eyepatch))]
#![cfg_attr(
    all(feature = "nightly", test),
    allow(incomplete_features),
    feature(specialization)
)]

use std::fmt::{self, Debug, Formatter};

#[macro_use]
mod macros;

#[cfg(feature = "fuzz")]
pub mod fuzz;

mod public_traits;
mod tree;

pub use public_traits::{DirectionalAdd, DirectionalSub, Index, Slice, Zero};
pub use tree::{Drain, IntoIter, Iter, RleTree, SliceEntry};

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
