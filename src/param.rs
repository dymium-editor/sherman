//! Parameterization for [`RleTree`], see [`RleTreeConfig`] for more.

use std::marker::PhantomData;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::AtomicUsize;

use crate::RleTree;
use crate::tree::{node, rc};

/// Marker type: Don't enable any additional features for an [`RleTree`]
#[cfg_attr(test, derive(Debug))]
pub struct NoFeatures(PhantomData<()>);

/// Marker type: Enable copy-on-write capabilities for an [`RleTree`]
#[cfg_attr(test, derive(Debug))]
pub struct EnableCow(PhantomData<std::sync::Arc<()>>);

/// Marker type: Enable stable slice references for an [`RleTree`]
#[cfg_attr(test, derive(Debug))]
pub struct EnableRefs(PhantomData<std::rc::Rc<()>>);

pub(crate) mod sealed {
    /// Marker trait - this trait is not implementable outside this crate
    pub trait Sealed {}

    impl Sealed for super::NoFeatures {}
    impl Sealed for super::EnableCow {}
    impl Sealed for super::EnableRefs {}
}

/// Sealed trait for [`RleTree`] parameterizations
///
/// There are three parameterization options for an [`RleTree`]:
///
///  1. [`NoFeatures`] - no special functionality
///  2. [`EnableCow`] - copy-on-write enabled, requires `S: Clone` and adds atomic reference
///     counters and makes the tree `Send`/`Sync` only if `I` and `S` are `Send + Sync`.
///  3. [`EnableRefs`] - stable slice references enabled, makes the tree `!Send` and `!Sync`
///
/// Where applicable, the documentation on [`RleTree`] explains the algorithmic complexity.
pub trait RleTreeConfig<I, S>: 'static + Sized + sealed::Sealed {
    /// Is this type [`EnableCow`]?
    const COW: bool;
    /// Is this type [`EnableRefs`]?
    const REFS: bool;

    /// If `EnableCow` or `EnableRefs`, the number of owning references to the node
    ///
    /// With `EnableCow`, this value ranges from zero to many.
    /// With `EnableRefs` it is either zero or one.
    type StrongCount: 'static + rc::RefCount + UnwindSafe + RefUnwindSafe;

    /// If `EnableRefs`, the number of weak references to the node
    type WeakCount: 'static + rc::RefCount + UnwindSafe + RefUnwindSafe;

    /// If `EnableRefs`, inner state that allows tracking borrows.
    type BorrowState: 'static + rc::BorrowState + UnwindSafe + RefUnwindSafe;

    /// If `EnableRefs`, an optional pointer to a node that the value was joined with
    type Redirect: rc::Redirect<I, S, Self>;

    /// If `EnableCow`, a stack for managing disjoint parent pointers during tree traversal.
    /// Otherwise, just a borrow on a single node.
    type BorrowStack<'a>: node::BorrowStack<'a, Index = I, Slice = S, Param = Self>
    where
        I: 'a,
        S: 'a;
}

impl<I, S> RleTreeConfig<I, S> for NoFeatures {
    const COW: bool = false;
    const REFS: bool = false;

    type StrongCount = ();
    type WeakCount = ();
    type BorrowState = ();
    type Redirect = ();
    type BorrowStack<'a>
        = node::HandleImmut<'a, I, S, Self>
    where
        I: 'a,
        S: 'a;
}

impl<I, S> RleTreeConfig<I, S> for EnableCow {
    const COW: bool = true;
    const REFS: bool = false;

    type StrongCount = AtomicUsize;
    type WeakCount = ();
    type BorrowState = ();
    type Redirect = ();
    type BorrowStack<'a>
        = node::ImmutStack<'a, I, S, Self>
    where
        I: 'a,
        S: 'a;
}

impl<I, S> RleTreeConfig<I, S> for EnableRefs {
    const COW: bool = false;
    const REFS: bool = true;

    type StrongCount = ();
    type WeakCount = rc::UsizeCell;
    type BorrowState = rc::BorrowStateCell;
    type Redirect = rc::RedirectCell<I, S>;
    type BorrowStack<'a>
        = node::HandleImmut<'a, I, S, Self>
    where
        I: 'a,
        S: 'a;
}

/// Sealed marker trait: Can an [`RleTree`] be modified with the given [`RleTreeConfig`]?
///
/// This is how we manage the `Clone` requirement for [`EnableCow`] without requiring it for the
/// other parameterizations.
pub trait SupportsUpdate<I, S>: 'static + sealed::Sealed {
    /// Copies the index `I`. This is only implemented for [`EnableCow`].
    fn copy_index(_: &I) -> I {
        unreachable!();
    }

    /// Clones the slice `S`. This is only implemented for [`EnableCow`].
    fn clone_slice(_: &S) -> S {
        unreachable!();
    }
}

impl<I, S> SupportsUpdate<I, S> for NoFeatures {}

impl<I, S> SupportsUpdate<I, S> for EnableRefs {}

impl<I: Copy, S: Clone> SupportsUpdate<I, S> for EnableCow {
    fn copy_index(i: &I) -> I {
        *i
    }

    fn clone_slice(s: &S) -> S {
        s.clone()
    }
}

/// Sealed marker trait: Should an [`RleTree`] implement [`Send`] for a given [`RleTreeConfig`]?
///
/// This trait exists for cleaner error messages (which would otherwise reference the internal
/// contents of node pointers and such, which obscures the real reason).
///
/// # Which configurations implement `Send`?
///
/// The short version is:
///
/// * With [`NoFeatures`], `RleTree<I, S>: Send` if `I: Send` and `S: Send`
/// * With [`EnableCow`], `RleTree<I, S>: Send` if `I: Send + Sync` and `S: Send + Sync`
///   (roughly matching the behavior of [`Arc<(I, S)>`](std::sync::Arc))
/// * With [`EnableRefs`], `RleTree` is never `Send`
///   (roughly matching the behavior of [`Rc<(I, S)>`](std::rc::Rc))
///
/// If you are building an abstraction on top of `RleTree` with `EnableRefs`, it *is* possible to
/// implement `Send` for that abstraction, provided that whenver the `RleTree` is sent across
/// threads, all stable references are sent with it (and vice versa, with respect to any stable
/// reference).
#[allow(clippy::missing_safety_doc)]
pub unsafe trait RleTreeIsSend<I, S>: sealed::Sealed {}

// SAFETY: With no extra features, it is safe to send RleTree<I, S> across threads if I and S are
// both Send because there is no shared mutable state. All mutations are encapsulated in methods
// that take &mut RleTree, and there are no returned types that can mutate the same state (unless
// the shared state comes from I or S, but we know it shouldn't if they are both Send).
unsafe impl<I: Send, S: Send> RleTreeIsSend<I, S> for NoFeatures {}

// SAFETY: With COW enabled, RleTree<I, S> has roughly the same behavior as Arc<(I, S)>.
// When it is sent across threads, we are implicitly also sending references to the values across
// threads as well, because multithreaded reads are allowed, so I and S must both be Send + Sync.
unsafe impl<I: Send + Sync, S: Send + Sync> RleTreeIsSend<I, S> for EnableCow {}

// NOTE: RleTree<I, S, EnableRefs> is never Send, because stable node references might be
// concurrently cloned or dropped, which would perform an unsynchronized mutation of the ref count.

unsafe impl<I, S, P> Send for RleTree<I, S, P> where P: RleTreeConfig<I, S> + RleTreeIsSend<I, S> {}

/// Sealed marker trait: Should an [`RleTree`] implement [`Sync`] for a given [`RleTreeConfig`]?
///
/// This trait exists for cleaner error messages (which would otherwise reference the internal
/// contents of node pointers and such, which obscures the real reason).
///
/// # Which configurations implement `Sync`?
///
/// The short version is:
///
/// * With [`NoFeatures`], `RleTree<I, S>: Sync` if `I: Sync` and `S: Sync`
/// * With [`EnableCow`], `RleTree<I, S>: Sync` if `I: Send + Sync` and `S: Send + Sync`
///   (roughly matching the behavior of [`Arc<(I, S)>`](std::sync::Arc))
/// * With [`EnableRefs`], `RleTree` is never `Sync`
///   (roughly matching the behavior of [`Rc<(I, S)>`](std::rc::Rc))
///
/// If you are building an abstraction on top of `RleTree` with `EnableRefs`, it *is* possible to
/// implement `Sync` for that abstraction, provided that (a) any stable references cannot be
/// created/cloned/mutated/dropped concurrently with reads to the `RleTree` or any other stable
/// references; and (b) the `RleTree` cannot be modified concurrently with any operation on any
/// stable reference.
#[allow(clippy::missing_safety_doc)]
pub unsafe trait RleTreeIsSync<I, S>: sealed::Sealed {}

// SAFETY: With no extra features, it is safe to send &RleTree<I, S> if I: Sync and S: Sync because
// all mutations are done through methods that take &mut RleTree, so no mutations can happen unless
// they, themselves allow unsynchronized mutation from an immutable borrow (in which case, they
// must not be Sync).
unsafe impl<I: Sync, S: Sync> RleTreeIsSync<I, S> for NoFeatures {}
// SAFETY: With COW enabled, a reference to RleTree<I, S> can be turned into a shallow clone.
// Sending a reference across threads has the same requirements as sending the full thing, namely
// that both I and S must be Send + Sync.
unsafe impl<I: Send + Sync, S: Send + Sync> RleTreeIsSync<I, S> for EnableCow {}

// NOTE: RleTree<I, S, EnableRefs> is never Sync, because immutable access across threads still
// allows modification by incrementing & decrementing the non-atomic ref counters on nodes.

unsafe impl<I, S, P> Sync for RleTree<I, S, P> where P: RleTreeConfig<I, S> + RleTreeIsSync<I, S> {}
