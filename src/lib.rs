//! # Sherman -- a truly monstrous tree type
//!
//! It turns out that in the production of an editor, sometimes specialized data structures are
//! used.  Sometimes data structures are similar enough that they can be united into a single
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
//! * Cursors -- operations can reuse prior paths through the tree, resulting in large speedups for
//!     typical access patterns.
//!
//! And of course, all of these features are zero-cost when not in use: the tree is constructed in
//! such a way so that only the instances that actually *do* use these extra feature (like node
//! references or concurrent COW) have to pay the cost of them. And the cost of each individual
//! feature has been minimized as much as is possible.
//!
//! For a more detailed explanation on how we're able to pull this off,
//! [ARCHITECTURE.md] gives a high-level overview of all the moving pieces to this
//! crate.
//!
//! [ARCHITECTURE.md]: https://github.com/dymium-editor/sherman/blob/main/ARCHITECTURE.md
//!
//! ### Feature flags
//!
//! There is currently just one feature flag -- `nightly`. This opt-in feature enables some minor
//! improvements (notably: implementing `Drop` for [`RleTree`] with `#[may_dangle]`), but requires
//! nightly rustc.
//!
//! ### Naming
//!
//! This library is named after [General Sherman], a tree in Sequoia National Park that's the
//! current largest tree on Earth by volume.
//!
//! [General Sherman]: https://en.wikipedia.org/wiki/General_Sherman_(tree)

#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(feature = "nightly", feature(dropck_eyepatch))]
#![cfg_attr(all(feature = "nightly", test), feature(specialization))]

use std::fmt::{self, Debug, Formatter};

#[macro_use]
mod macros;

pub mod param;
pub mod range;

mod const_math_hack;
mod cursor;
mod public_traits;
mod recycle;
mod tree;

pub use cursor::{BoundedCursor, Cursor, NoCursor, PathComponent};
pub use public_traits::{DirectionalAdd, DirectionalSub, Index, Slice, Zero};
pub use tree::{Drain, Iter, RleTree, SliceEntry, SliceRef, DEFAULT_MIN_KEYS};

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
pub struct Constant<T>(pub T);

impl<Idx, T: Clone + PartialEq> Slice<Idx> for Constant<T> {
    const MAY_JOIN: bool = true;

    fn split_at(&mut self, _idx: Idx) -> Self {
        Constant(self.0.clone())
    }

    fn try_join(self, other: Self) -> Result<Self, (Self, Self)> {
        if self == other {
            Ok(self)
        } else {
            Err((self, other))
        }
    }
}

/// Crate-internal trait for allowing introspection into debuggable types without introducing new
/// bounds.
///
/// This is only usefully implemented during tests with `feature = "nightly"`.
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
