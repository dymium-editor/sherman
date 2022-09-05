//! Parameterization for [`RleTree`]
//!
//! There are only three parameterization options for an [`RleTree`]: no features, with slice
//! references, or with clone-on-write. We unfortunately don't have both slice references *and* COW
//! capabilities because any implementation would be fairly costly.
//!
//! This module exports the types [`NoFeatures`], [`AllowCow`], and [`AllowSliceRefs`] -- each of
//! which can be used as parameterization for the [`RleTree`], like:
//!
//! ```
//! use sherman::RleTree;
//! use sherman::param::AllowCow;
//!
//! type MyTree<I, V> = RleTree<I, V, AllowCow>;
//! ```
//!
//! The default configuration is [`NoFeatures`]. All configuration types implement
//! [`RleTreeConfig`] -- the trait can't be implemented outside of this crate, but it's there to
//! help with errors and documentation.
//!
//! [`RleTree`]: crate::RleTree

use crate::tree::slice_ref::{self, BorrowFailure, ShouldDrop};
use std::ptr::NonNull;

/// Marker type to not enable any features for the [`RleTree`](crate::RleTree) (*default*)
pub enum NoFeatures {}

/// Marker type to allow clone-on-write capabilities for an [`RleTree`](crate::RleTree)
pub enum AllowCow {}

/// Marker type to allow slice references in the [`RleTree`](crate::RleTree)
pub enum AllowSliceRefs {}

/// *Internal-only* trait that parameterizations are required to implement
///
/// This trait is made public to help with error messages and for your curiosity. It cannot be
/// implemented outside of this crate.
pub trait RleTreeConfig<I, S> {
    /// A type bound you can't satisfy, here to make sure you can't implement this trait
    #[doc(hidden)]
    type YouCantImplementThis: sealed::TraitYouCantReach;

    /// Marker for whether `Self` is `AllowCow`.
    const COW: bool;

    /// If clone-on-write capabilities are enabled, we need a way to track the number of pointers
    /// to an individual node. If COW isn't enabled, then there's no point in taking up that space
    ///
    /// This is still called the "strong" count, because it essentially has the same meaning as in
    /// [`Rc`]/[`Arc`], even though there aren't "weak" references in the same way here.
    ///
    /// [`Rc`]: std::rc::Rc
    /// [`Arc`]: std::sync::Arc
    type StrongCount: StrongCount;

    /// A "strong count" that can be shared between multiple objects -- an `Arc<()>` (if provided),
    /// with the strong count taken directly from the `Arc`
    type SharedStrongCount: StrongCount;

    /// If slice references are enabled, we keep a big vector mapping from `SliceRefId`s to
    /// pointers to the nodes (i.e. allocations) containing the slice
    type SliceRefStore: Default + BorrowState + SliceRefStore<I, S>;
}

/// *Internal-only* trait that marks configurations as supporting insertion
///
/// This trait is implemented for everything except `AllowCow` when `S` doesn't implement `Clone`.
pub trait SupportsInsert<I, S>: sealed::YouCantImplementThis {
    /// Shim to access the implementation of `Clone` for `S` *only when used with `AllowCow`*. All
    /// other implementations are equivalent to [`std::hint::unreachable_unchecked`].
    ///
    /// ## Safety
    ///
    /// The caller must ensure that `Self = AllowCow`, else the result will be
    /// instantaneous UB.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn clone_slice(_slice: &S) -> S {
        weak_unreachable!();
    }
}

pub(crate) mod sealed {
    /// It's a trait you (the user of this crate) can't reach
    pub trait TraitYouCantReach {}
    impl TraitYouCantReach for TypeYouCantSee {}

    /// The trait you can't reach is only implemented by a type you can't see
    pub struct TypeYouCantSee;

    /// It's a supertrait you (the user of this crate) can't implement
    pub trait YouCantImplementThis {}

    impl YouCantImplementThis for () {}
    impl YouCantImplementThis for super::NoFeatures {}
    impl YouCantImplementThis for super::AllowCow {}
    impl YouCantImplementThis for super::AllowSliceRefs {}
}

/// *Internal-only* abstraction trait required for types used as [`RleTreeConfig::StrongCount`] or
/// [`RleTreeConfig::SharedStrongCount`].
pub trait StrongCount: sealed::YouCantImplementThis {
    /// Returns the associated count
    ///
    /// For `Arc<()>`, this returns `Arc::strong_count(self)`. For `AtomicUsize`, this returns
    /// `self.load(...)`.
    fn count(&self) -> usize;

    /// Creates a new value with a count of one
    fn one() -> Self;

    /// Returns whether the owner of the strong count is a unique owner
    ///
    /// Because of how multithreading works, this method is can only ever say "yes" or
    /// "probably not" -- another thread might decrement the strong count, making this instance
    /// unqiue between the method's result being calculated vs returned.
    ///
    /// However, if this instance *is* unique, it's guaranteed to stay that way until outside
    /// action changes that fact.
    ///
    /// This method exists as an optimization for not copying COW-enabled trees when there's no
    /// other references.
    fn is_unique(&self) -> bool;

    /// Increments the strong count, returning the old value
    ///
    /// For an `AtomicUsize`, this just returns the previous strong count. For `Arc<()>` though,
    /// the cloned value *is* the way that incrementing is done, so it's kept around.
    fn increment(&self) -> Self;

    /// Manually decrements the strong count, if that applies to this type (only `AtomicUsize`),
    /// returning `true` if the value it guards should be dropped
    fn decrement(&self) -> bool;
}

/// *Internal-only* abstraction trait required for types used as [`RleTreeConfig::SliceRefStore`]
pub trait SliceRefStore<I, S>: sealed::YouCantImplementThis {
    /// An `Option<RefId>` for slices to point back to their own references, if these are being
    /// tracked.
    ///
    /// If slice references aren't being tracked, then this is an empty tuple: `()`.
    type OptionRefId: Default;

    /// Record that the slices with the two `Option<RefId>`s have joined, and return whatever's now
    /// being used
    fn join_refs(&mut self, rx: Self::OptionRefId, ry: Self::OptionRefId) -> Self::OptionRefId;

    /// Record that the slice with this `Option<RefId>` has been removed, so any references
    /// pointing to it should be dropped
    fn remove(&mut self, x: Self::OptionRefId);

    /// Updates the existing `Option<RefId>` (if applicable), to the given node pointer with index
    /// within the node
    ///
    /// ## Safety
    ///
    /// The node pointed to by `ptr` must be valid, containing `x` as a reference at index `idx`.
    /// Callers may *assume* (in `unsafe` code, without checking) that the node at `ptr` is not
    /// accessed during any call to this function.
    unsafe fn update(&mut self, x: &Self::OptionRefId, ptr: NonNull<(I, S)>, idx: u8);

    /// Marks that the slice with this `RefId` has been temporarily removed from its node, but will
    /// be added back later
    ///
    /// Slice references cannot access suspended contents, and so `suspend` should be accompanied
    /// by a call to [`update`] to "resume" access to the slice, *before* releasing any mutable
    /// borrow on the contents of the tree. If this is not upheld, then future interactions with
    /// any [`SliceRef`] for the slice *will* panic.
    ///
    /// [`update`]: Self::update
    /// [`SliceRef`]: crate::SliceRef
    fn suspend(&mut self, x: &Self::OptionRefId);
}

/// *Internal-only* abstraction trait required for types used as [`RleTreeConfig::SliceRefStore`]
pub trait BorrowState: sealed::YouCantImplementThis {
    /// Attempt to acquire an immutable reference, returning an appropriate error if it wasn't
    /// possible
    fn acquire_immutable(&self) -> Result<(), BorrowFailure>;
    /// Release a single previously-held immutable reference
    ///
    /// It's possible for a lone immutable reference to keep the tree alive, so dropping an
    /// immutable reference must handle the possibility that it is now responsible for dropping
    /// everything else as well.
    fn release_immutable(&self) -> ShouldDrop;

    /// Attempt to acquire a mutable reference, returning an appropriate error if it wasn't possible
    fn acquire_mutable(&self) -> Result<(), BorrowFailure>;
    /// Release the previously-held mutable reference
    fn release_mutable(&self);

    /// Try to acquire a "dropping" borrow, failing only if there's currently another mutable
    /// borrow
    ///
    /// A dropping borrow is mutable, but crucially can fail softly if there's existing immutable
    /// borrows, instead instructing the last immutable borrow to perform the drop instead.
    ///
    /// The caller may only assume unique access if `ShouldDrop::Yes` is returned.
    fn try_acquire_drop(&self) -> ShouldDrop;
}

impl<I, S> RleTreeConfig<I, S> for NoFeatures {
    type YouCantImplementThis = sealed::TypeYouCantSee;
    const COW: bool = false;
    type StrongCount = ();
    type SharedStrongCount = ();
    type SliceRefStore = ();
}

impl<I, S> RleTreeConfig<I, S> for AllowCow {
    type YouCantImplementThis = <NoFeatures as RleTreeConfig<I, S>>::YouCantImplementThis;

    const COW: bool = true;
    type StrongCount = std::sync::atomic::AtomicUsize;
    type SharedStrongCount = std::sync::Arc<()>;

    type SliceRefStore = ();
}

impl<I, S> RleTreeConfig<I, S> for AllowSliceRefs {
    type YouCantImplementThis = <NoFeatures as RleTreeConfig<I, S>>::YouCantImplementThis;

    const COW: bool = false;
    type StrongCount = ();
    type SharedStrongCount = ();

    type SliceRefStore = slice_ref::SliceRefStore<I, S>;
}

impl<I, S> SupportsInsert<I, S> for NoFeatures {}
impl<I, S> SupportsInsert<I, S> for AllowSliceRefs {}

impl<I: Clone, S: Clone> SupportsInsert<I, S> for AllowCow {
    unsafe fn clone_slice(slice: &S) -> S {
        slice.clone()
    }
}
