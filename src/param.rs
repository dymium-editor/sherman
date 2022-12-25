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
//! use sherman::param::AllowCow;
//! use sherman::RleTree;
//!
//! type MyTree<I, V> = RleTree<I, V, AllowCow>;
//! ```
//!
//! The default configuration is [`NoFeatures`]. All configuration types implement
//! [`RleTreeConfig`] -- the trait can't be implemented outside of this crate, but it's there to
//! help with errors and documentation.
//!
//! [`RleTree`]: crate::RleTree

use std::marker::PhantomData;

use crate::tree::slice_ref::{self, BorrowFailure, RawRoot, RawSliceRef, ShouldDrop};

/// Marker type to not enable any features for the [`RleTree`](crate::RleTree) (*default*)
pub struct NoFeatures(PhantomData<()>);

/// Marker type to allow clone-on-write capabilities for an [`RleTree`](crate::RleTree)
pub struct AllowCow(PhantomData<()>);

/// Marker type to allow slice references in the [`RleTree`](crate::RleTree)
///
/// **Note:** Trees parameterized by this type do not implement `Send` or `Sync`; supporting
/// multi-threaded slice references would be significantly more costly and we have not yet found a
/// use case for them.
pub struct AllowSliceRefs(PhantomData<std::rc::Rc<()>>);

/// Trait that [`RleTree`] parameterizations are required to implement
///
/// This trait is made public to help with error messages and for your curiosity. It cannot be
/// implemented outside of this crate.
///
/// [`RleTree`]: crate::RleTree
pub trait RleTreeConfig<I, S, const M: usize>: Sized {
    /// A type bound you can't satisfy, here to make sure you can't implement this trait
    #[doc(hidden)]
    type YouCantImplementThis: sealed::TraitYouCantReach;

    /// Marker for whether `Self` is `AllowCow`.
    const COW: bool;

    /// Marker for whether `Self` is `AllowSliceRefs`.
    const SLICE_REFS: bool;

    /// If clone-on-write capabilities are enabled, we need a way to track the number of pointers
    /// to an individual node. If COW isn't enabled, then there's no point in taking up that space
    ///
    /// This is still called the "strong" count, because it essentially has the same meaning as in
    /// [`Rc`]/[`Arc`], even though there aren't "weak" references in the same way here.
    ///
    /// [`Rc`]: std::rc::Rc
    /// [`Arc`]: std::sync::Arc
    type StrongCount: StrongCount;

    /// If slice references are enabled, we keep a big vector mapping from `SliceRefId`s to
    /// pointers to the nodes (i.e. allocations) containing the slice
    type SliceRefStore: BorrowState + SliceRefStore<I, S, Self, M>;
}

/// Trait that marks configurations as supporting insertion
///
/// Like [`RleTreeConfig`], this trait is made public to hep with error messages. It cannot be
/// implemented outside this crate.
///
/// This trait is implemented for everything except `AllowCow` when `S` doesn't implement `Clone`.
pub trait SupportsInsert<I, S, const M: usize>: sealed::YouCantImplementThis {
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

/// Marker trait for which [`RleTree`]s can implement [`Send`], based on their [`RleTreeConfig`]
///
/// This trait exists because determining whether an [`RleTree`] should implement [`Send`] is a bit
/// annoying, and this is the only solution that allows us to retain error messages that don't
/// involve implementation internals.
///
/// The table below lays out which [`RleTreeConfig`] implement [`Send`], and when:
///
/// | Parameter | Requirements |
/// |-----------|--------------|
/// | [`NoFeatures`] | [`RleTree`]s with this parameter can be thought of as analogous to `Box`es, `Vec`s, or `BTreeMap`s -- they aren't doing any funky stuff with concurrency, so they implement `Send` if the values they contain do. This requires only `I: Send` and `S: Send`. |
/// | [`AllowCow`] | [`RleTree`]s with this parameter are roughly analogous to `Arc`s. Because sending a COW-enabled tree across threads doesn't clone the value, we can view a reference to a value from multiple threads, meaning that we must have `I: Send + Sync` **and** `S: Send + Sync`. |
/// | [`AllowSliceRefs`] | [`RleTree`]s with this enabled will never implement `Send`. Our mechanisms for handling [`SliceRef`]s are not synchronized (they internally use `Cell` and others), so it is never safe to implement `Send` for a slice ref-enabled [`RleTree`]. |
///
/// ## When can I implement `Send` for `AllowSliceRefs`?
///
/// Ok, I know the table above says "don't do it, ever". However: there are _some_ guarantees that
/// we can provide about our internal implementation that mean it's possible to have higher-level
/// datastructures built with [`SliceRef`]s that _do_ implement [`Send`].
///
/// In short: if your datastructure always ensures that the [`SliceRef`]s are in the same thread as
/// the [`RleTree`] itself, and moving the [`RleTree`] between threads is appropriately
/// synchronized (either with atomics or locks), then the outer type containing both the
/// [`RleTree`] and the [`SliceRef`]s _may_ implement [`Send`]. The typical requirements that `I:
/// Send` and `S: Send` will of course still apply.
///
/// ## Actually though, why does this trait exist?
///
/// To prevent [`AllowSliceRefs`] from implementing [`Send`], we basically have three options:
///
/// 1. Use separate `impl`s for [`NoFeatures`] and [`AllowCow`], and don't have one for
///    [`AllowSliceRefs`]
/// 2. Same as (1), but add `impl !Send` for [`AllowSliceRefs`]
/// 3. The current solution: implement [`Send`] for all `P`, and restrict with another trait bound
///
/// The second doesn't work right now, because [negative impls aren't stable yet]. So let's briefly
/// talk about why we didn't use the first solution.
///
/// [negative impls aren't stable yet]: https://github.com/rust-lang/rust/issues/68318
///
/// If we just didn't implement [`Send`] for slice ref-enabled trees, the error message would refer
/// to the raw pointers we use internally as why it doesn't implement [`Send`], which is actively
/// unhelpful:
///
/// ```text
/// `NonNull<AbstractNode<I, S, P, M>>` cannot be sent between threads safely
/// ```
///
/// By providing an implementation of [`Send`] that covers all [`RleTree`]s, where instead the
/// trait bound isn't satisfied, we instead get an error that reads more like:
///
/// ```text
/// the trait bound `AllowSliceRefs: RleTreeIsSend<I, S>` is not satisfied
/// ```
///
/// [`RleTree`]: crate::RleTree
/// [`SliceRef`]: crate::SliceRef
pub trait RleTreeIsSend<I, S> {}

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

/// *Internal-only* abstraction trait required for types used as [`RleTreeConfig::StrongCount`]
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
    /// unique between the method's result being calculated vs returned.
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
pub trait SliceRefStore<I, S, P, const M: usize>: sealed::YouCantImplementThis
where
    P: RleTreeConfig<I, S, M>,
{
    /// An `Option<RefId>` for slices to point back to their own references, if these are being
    /// tracked.
    ///
    /// If slice references aren't being tracked, then this is an empty tuple: `()`.
    type OptionRefId: Default;

    /// Constructs a new `SliceRefStore` with the given root node
    fn new(root: RawRoot<I, S, P, M>) -> Self;

    /// Overwrites the reference for the root of the tree
    fn set_root(&mut self, new_root: Option<RawRoot<I, S, P, M>>);

    /// Record that `from` has joined into `to`, and that references to `from` should now point to
    /// `to`
    fn redirect(&mut self, from: &mut Self::OptionRefId, to: &mut Self::OptionRefId);

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
    unsafe fn update(&mut self, x: &Self::OptionRefId, handle: RawSliceRef<I, S, P, M>);

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

impl<I, S, const M: usize> RleTreeConfig<I, S, M> for NoFeatures {
    type YouCantImplementThis = sealed::TypeYouCantSee;
    const COW: bool = false;
    const SLICE_REFS: bool = false;
    type StrongCount = ();
    type SliceRefStore = ();
}

impl<I, S, const M: usize> RleTreeConfig<I, S, M> for AllowCow {
    type YouCantImplementThis = <NoFeatures as RleTreeConfig<I, S, M>>::YouCantImplementThis;

    const COW: bool = true;
    const SLICE_REFS: bool = false;
    type StrongCount = std::sync::atomic::AtomicUsize;

    type SliceRefStore = ();
}

impl<I, S, const M: usize> RleTreeConfig<I, S, M> for AllowSliceRefs {
    type YouCantImplementThis = <NoFeatures as RleTreeConfig<I, S, M>>::YouCantImplementThis;

    const COW: bool = false;
    const SLICE_REFS: bool = true;
    type StrongCount = ();

    type SliceRefStore = slice_ref::SliceRefStore<I, S, M>;
}

impl<I, S, const M: usize> SupportsInsert<I, S, M> for NoFeatures {}
impl<I, S, const M: usize> SupportsInsert<I, S, M> for AllowSliceRefs {}

impl<I: Clone, S: Clone, const M: usize> SupportsInsert<I, S, M> for AllowCow {
    unsafe fn clone_slice(slice: &S) -> S {
        slice.clone()
    }
}

// Refer to `RleTreeIsSend` for more information about impelmenting `Send`
impl<I: Send, S: Send> RleTreeIsSend<I, S> for NoFeatures {}
impl<I: Send + Sync, S: Send + Sync> RleTreeIsSend<I, S> for AllowCow {}
