//! Public [`SliceRef`] type and internal handling
//!
//! One of the difficulties in the implementation here is that our reference counting is
//! centralized -- there isn't an easy or efficient way around this, but it has the unfortunate
//! side effect that we essentially need to *always* be ready to record `Drop`s or `Clone`s of
//! existing [`SliceRef`]s, even if the tree is currently mutably borrowed.

use std::alloc::{self, Layout};
use std::cell::{self, Cell, RefCell};
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::ops::Range;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::ptr::NonNull;

#[cfg(test)]
use std::fmt::Debug;

use super::node::{borrow, ty, NodeHandle, SliceHandle};
use super::DEFAULT_MIN_KEYS;
use crate::param::{self, AllowSliceRefs};
use crate::recycle::{self, RecycleVec};
use crate::Index;

pub type RawRoot<I, S, P, const M: usize> = NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>;

pub type RawSliceRef<I, S, P, const M: usize> =
    SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>;

/// A stable reference to an individual slice in an [`RleTree`], persistent across changes to the
/// tree
///
/// `SliceRef`s are typically created by the [`insert_ref`] method on [`RleTree`], or by the
/// [`SliceEntry::make_ref`]s (returned by an iterator over the tree).
///
/// For more information, refer to the documentation on [`RleTree`] itself.
///
/// [`RleTree`]: crate::RleTree
/// [`insert_ref`]: crate::RleTree::insert_ref
/// [`SliceEntry::make_ref`]: crate::SliceEntry::make_ref
pub struct SliceRef<I, S, const M: usize = DEFAULT_MIN_KEYS> {
    inner: NonNull<InnerStore<I, S, M>>,
    id: Cell<Option<RefId>>,
    marker: PhantomData<InnerStore<I, S, M>>,
}

#[cfg(test)]
impl<I, S, const M: usize> Debug for SliceRef<I, S, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let id = self.id.replace(None);
        let res = f.debug_struct("SliceRef").field("id", &id).finish();
        self.id.set(id);
        res
    }
}

impl<I: UnwindSafe + RefUnwindSafe, S: RefUnwindSafe, const M: usize> UnwindSafe
    for SliceRef<I, S, M>
{
}

impl<I: UnwindSafe + RefUnwindSafe, S: RefUnwindSafe, const M: usize> RefUnwindSafe
    for SliceRef<I, S, M>
{
}

/// Unique identifier for a slice reference, corresponding to an index in the internal vector
pub type RefId = recycle::EntryId;

/// Tree-wide store redirecting slice references
pub struct SliceRefStore<I, S, const M: usize> {
    inner: NonNull<InnerStore<I, S, M>>,
    marker: PhantomData<InnerStore<I, S, M>>,
}

/// Inner, shared component of the [`SliceRefStore`] and [`SliceRef`]
struct InnerStore<I, S, const M: usize> {
    /// The state of the borrow on the whole tree
    borrow: Cell<BorrowState>,
    /// Mapping of [`RefId`]s to the [`StoredSliceRef`] describing the value they reference
    refs: RefCell<RecycleVec<StoredSliceRef<I, S, M>>>,
    /// The root of the tree, if it's present
    root: Cell<Option<NodeHandle<ty::Unknown, borrow::Owned, I, S, AllowSliceRefs, M>>>,
    /// The current number of `SliceRef`s relying on this allocation. This has no impact on when
    /// we destruct `refs` and `root`, but *does* ensure that the allocation stays live until
    /// there's no longer reference to it.
    weak_count: Cell<usize>,
}

/// The referent of a particular [`RefId`]
///
/// Typical references are represented by a `SliceHandle` to the value itself, with the invariant
/// that any `RefId` with `StoredSliceRef::Handle` is the same as that slice's stored `RefId`. In
/// other words, any slice with an assigned `RefId` will have one that immediately refers back to
/// itself.
///
/// In cases where two slices are joined, we introduce `Redirect`s for one of the original
/// [`RefId`]s so that they now point to the same.
///
/// When a slice is permanently removed, the corresponding stored value is replaced with `Removed`.
/// When one is temporarily removed, the handle is set to `None`.
enum StoredSliceRef<I, S, const M: usize> {
    Handle(Option<SliceHandle<ty::Unknown, borrow::SliceRef, I, S, AllowSliceRefs, M>>),
    Redirect(RefId),
    Removed,
}

/// Information about the whether and how an `RleTree` is currently borrowed
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub enum BorrowState {
    /// The inner tree is not borrowed
    #[default]
    NotBorrowed,
    /// The tree is currently mutably borrowed by a method on the main `RleTree` type
    Mutable,
    /// A number of [`SliceRef`]s have immutable borrows on the tree
    ///
    /// If the main reference to the tree has been dropped but there are existing [`SliceRef`]s active,
    /// then `drop_if_zero` will be true. In this case, the contents of the [`InnerStore`]'s `refs`
    /// and `root` must be dropped when the final `Ref` is.
    ///
    /// *Note*: We don't track immutable borrows from methods on `RleTree` because they are safe by
    /// default.
    Immutable {
        count: NonZeroUsize,
        drop_if_zero: bool,
    },
    /// The tree has started dropping
    Dropping,
    /// The tree has been fully dropped, along with the [`SliceRefStore`]
    Dropped,
}

#[cfg(test)]
impl<I, S, const M: usize> Debug for SliceRefStore<I, S, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let inner = self.inner();

        let root = match inner.root.take() {
            Some(handle) => {
                let ptr = handle.ptr();
                inner.root.set(Some(handle));
                Some(ptr)
            }
            None => None,
        };

        let refs_guard = inner.refs.borrow();

        f.debug_struct("SliceRefStore")
            .field("borrow", &inner.borrow.get())
            .field("weak_count", &inner.weak_count.get())
            .field("root", &root)
            .field("refs", &refs_guard)
            .finish()
    }
}

#[cfg(test)]
impl<I, S, const M: usize> Debug for StoredSliceRef<I, S, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            StoredSliceRef::Removed => f.write_str("<Removed>"),
            StoredSliceRef::Handle(h) => match h {
                Some(handle) => {
                    let ptr = handle.node.ptr();
                    let idx = handle.idx;
                    f.write_fmt(format_args!("Handle(Some(@{ptr:p}, idx = {idx}))"))
                }
                None => f.write_str("Handle(None)"),
            },
            StoredSliceRef::Redirect(id) => f.write_fmt(format_args!("Redirect -> {id:?}")),
        }
    }
}

/// Error from failing to acquire a borrow
pub enum BorrowFailure {
    /// "cannot mutably borrow: already immutably borrowed"
    CannotGetMutAlreadyBorrowed,
    /// "cannot immutably borrow: already mutably borrowed"
    CannotGetImmutAlreadyMutablyBorrowed,
    /// "cannot immutably borrow: object has been dropped"
    CannotGetImmutAlreadyDropped,
}

impl Display for BorrowFailure {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use BorrowFailure::*;

        let s = match self {
            CannotGetMutAlreadyBorrowed => "cannot mutably borrow: already immutably borrowed",
            CannotGetImmutAlreadyMutablyBorrowed => {
                "cannot immutably borrow: already mutably borrowed"
            }
            CannotGetImmutAlreadyDropped => "cannot immutably borrow: object has been dropped",
        };

        f.write_str(s)
    }
}

/// Binary flag for whether the [`RleTree`] should be dropped on releasing a borrow
///
/// [`RleTree`]: crate::RleTree
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ShouldDrop {
    No,
    Yes,
}

/// Reference to a particular slice, returned by [`SliceRef::borrow_slice`]
///
/// This type is essentially identical to [`std::cell::Ref`], and similarly allows mapping to
/// components of the borrowed value via the [`map`] method.
///
/// ## Drop behavior
///
/// If any existing `Borrow`s on it are live, dropping an `RleTree` will not actually take effect
/// until the last `Borrow` itself is dropped.
///
/// [`map`]: Self::map
//
// Existence of a `Ref` guarantees that the `borrow` is `Immutable { .. }`, except for certain
// conditions *created* during its destructor
pub struct Borrow<'a, T> {
    /// The original `Slice`, or a reference produced from it
    val: &'a T,
    /// The state of the original borrow
    ///
    /// Existence of this `Ref` guarantees the borrow is `Immutable { .. }`, except for the few
    /// cases where we set it to `BorrowState::Dropped` during this type's destructor.
    borrow: &'a Cell<BorrowState>,
    /// Type-erased pointer to the `InnerStore<?, ?, ?>`, which we'll provide to `drop_fn` if we
    /// find that this borrow was the last thing keeping the tree alive
    data_ptr: NonNull<()>,
    /// Function to call with `data_ptr` if we find that the immutable borrow state has
    /// `drop_if_zero = true`, so we need to drop the tree.
    drop_fn: unsafe fn(NonNull<()>),
}

impl<'a, T: RefUnwindSafe> UnwindSafe for Borrow<'a, T> {}
impl<'a, T: RefUnwindSafe> RefUnwindSafe for Borrow<'a, T> {}

///////////////////////////////////////////////////////
// Internal `SliceRefStore` API (plus `param` impls) //
///////////////////////////////////////////////////////

#[rustfmt::skip]
impl param::BorrowState for () {
    fn acquire_mutable(&self) -> Result<(), BorrowFailure> { Ok(()) }
    fn release_mutable(&self) {}
    fn try_acquire_drop(&self) -> ShouldDrop { ShouldDrop::Yes }
}

impl<I, S, const M: usize> param::sealed::YouCantImplementThis for SliceRefStore<I, S, M> {}

impl<I, S, const M: usize> param::BorrowState for SliceRefStore<I, S, M> {
    fn acquire_mutable(&self) -> Result<(), BorrowFailure> {
        let borrow = &self.inner().borrow;
        match borrow.get() {
            BorrowState::NotBorrowed => {
                borrow.set(BorrowState::Mutable);
                Ok(())
            }
            BorrowState::Immutable { .. } => Err(BorrowFailure::CannotGetMutAlreadyBorrowed),
            BorrowState::Mutable => unreachable!("double mutable borrow attempted"),
            BorrowState::Dropped | BorrowState::Dropping => {
                unreachable!("mutable borrow attempted after drop")
            }
        }
    }

    fn release_mutable(&self) {
        assert!(self.inner().borrow.get() == BorrowState::Mutable);
        self.inner().borrow.set(BorrowState::NotBorrowed);
    }

    fn try_acquire_drop(&self) -> ShouldDrop {
        let borrow = &self.inner().borrow;
        match borrow.get() {
            // On the face of it, we shouldn't ever encounter an attempt to drop while we're
            // mutably borrowed. But because mutable borrows of the `SliceRefStore` aren't guarded,
            // and instead have to just rely on it being properly reset at the end of the
            // insertion, when panics occur, we can end up with a tree that's still *marked* as
            // mutably borrowed when it's dropped, even though the mutable borrow has expired.
            //
            // Unfortunately, we can't just check `std::thread::panicking()` either, because that
            // wouldn't handle cases where the panic has been recovered from, but it's still marked
            // as mutably borrowed. So we have to just blanket-accept that it might still be marked
            // as mutably borrowed when we go to drop it.
            BorrowState::Mutable => {
                borrow.set(BorrowState::Dropping);
                ShouldDrop::Yes
            }
            BorrowState::Dropped
            | BorrowState::Dropping
            | BorrowState::Immutable { drop_if_zero: true, .. } => {
                unreachable!("double-drop attempted")
            }
            BorrowState::NotBorrowed => {
                borrow.set(BorrowState::Dropping);
                ShouldDrop::Yes
            }
            BorrowState::Immutable { count, .. } => {
                borrow.set(BorrowState::Immutable { count, drop_if_zero: true });
                ShouldDrop::No
            }
        }
    }
}

impl<I, S, P: param::RleTreeConfig<I, S, M>, const M: usize> param::SliceRefStore<I, S, P, M>
    for ()
{
    type OptionRefId = ();

    fn new(_: RawRoot<I, S, P, M>) {}
    fn set_root(&mut self, _: Option<RawRoot<I, S, P, M>>) {}
    fn redirect(&mut self, _: &mut (), _: &mut ()) {}
    fn remove(&mut self, _: ()) {}
    unsafe fn update(&mut self, _: &(), _: RawSliceRef<I, S, P, M>) {}
    fn suspend(&mut self, _: &()) {}
}

impl<I, S, const M: usize> SliceRefStore<I, S, M> {
    fn inner(&self) -> &InnerStore<I, S, M> {
        // SAFETY: The existence of the `SliceRefStore` guarantees that the `InnerStore` is valid;
        // we only ever construct immutable references to it, so it's always valid to access
        // through a `SliceRefStore`.
        unsafe { &*self.inner.as_ptr() }
    }

    pub(super) fn make_ref(
        &self,
        handle: SliceHandle<ty::Unknown, borrow::SliceRef, I, S, AllowSliceRefs, M>,
    ) -> SliceRef<I, S, M> {
        let inner = self.inner();
        let mut refs = inner.refs.borrow_mut();

        let id = if let Some(id) = handle.take_refid() {
            let new_id = refs.clone(&id);
            assert!(handle.replace_refid(Some(id)).is_none());
            new_id
        } else {
            let handle_copy = unsafe { handle.clone_slice_ref() };
            let new_id = refs.push(StoredSliceRef::Handle(Some(handle_copy)));
            assert!(handle.replace_refid(Some(refs.clone(&new_id))).is_none());
            new_id
        };

        bump_weak_count(&inner.weak_count);

        SliceRef {
            inner: self.inner,
            id: Cell::new(Some(id)),
            marker: PhantomData,
        }
    }
}

impl<I, S, const M: usize> param::SliceRefStore<I, S, AllowSliceRefs, M>
    for SliceRefStore<I, S, M>
{
    type OptionRefId = Option<RefId>;

    fn new(root: RawRoot<I, S, AllowSliceRefs, M>) -> Self {
        SliceRefStore {
            inner: InnerStore::alloc(InnerStore {
                borrow: Cell::new(BorrowState::default()),
                refs: RefCell::new(RecycleVec::default()),
                root: Cell::new(Some(root)),
                weak_count: Cell::new(0),
            }),
            marker: PhantomData,
        }
    }

    fn set_root(&mut self, new_root: Option<RawRoot<I, S, AllowSliceRefs, M>>) {
        self.inner().root.set(new_root);
    }

    fn redirect(&mut self, from: &mut Option<RefId>, to: &mut Option<RefId>) {
        match (from, &to) {
            // Nothing to do
            (None, Some(_) | None) => return,
            // We don't currently have a reference in use for `to`, so we'll assign the same one as
            // is being used by `from`:
            (Some(r), None) => *to = Some(self.inner().refs.borrow().clone(r)),
            (Some(f), Some(t)) => {
                let mut refs = self.inner().refs.borrow_mut();
                let old_from = mem::replace(f, refs.clone(t));
                match refs.recycle(old_from) {
                    Some(StoredSliceRef::Handle(_)) | None => (),
                    Some(StoredSliceRef::Removed | StoredSliceRef::Redirect(_)) => unreachable!(),
                }
            }
        }
    }

    fn remove(&mut self, r: Option<RefId>) {
        let id = match r {
            Some(id) => id,
            None => return,
        };

        let mut refs = self.inner().refs.borrow_mut();
        let old = refs.replace(&id, StoredSliceRef::Removed);
        assert!(matches!(old, StoredSliceRef::Handle(_)));
        if let Some(old) = refs.recycle(id) {
            assert!(matches!(old, StoredSliceRef::Removed));
        }
    }

    unsafe fn update(
        &mut self,
        r: &Option<RefId>,
        handle: SliceHandle<ty::Unknown, borrow::SliceRef, I, S, AllowSliceRefs, M>,
    ) {
        if let Some(id) = r.as_ref() {
            let mut refs = self.inner().refs.borrow_mut();

            let old = refs.replace(id, StoredSliceRef::Handle(Some(handle)));
            assert!(matches!(old, StoredSliceRef::Handle(_)));
        }
    }

    fn suspend(&mut self, r: &Option<RefId>) {
        if let Some(id) = r.as_ref() {
            let old = self.inner().refs.borrow_mut().replace(id, StoredSliceRef::Handle(None));

            assert!(matches!(old, StoredSliceRef::Handle(Some(_))))
        }
    }
}

impl<I, S, const M: usize> Drop for SliceRefStore<I, S, M> {
    fn drop(&mut self) {
        let inner = self.inner();

        let new_state = match inner.borrow.get() {
            BorrowState::NotBorrowed
            | BorrowState::Mutable
            | BorrowState::Immutable { drop_if_zero: false, .. }
            | BorrowState::Dropped => unreachable!("invalid state"),

            BorrowState::Immutable { count, .. } => {
                BorrowState::Immutable { count, drop_if_zero: true }
            }
            BorrowState::Dropping => BorrowState::Dropped,
        };

        inner.borrow.set(new_state);

        // drop(ish) the inner contents. `false` because the tree's already handled by our caller,
        // in `destruct_root`
        inner.drop(false);

        // If everything has been *completely* dropped, and there's no weak references left (in the
        // form of `SliceRef`s), then we can deallocate the backing allocation.
        if matches!(new_state, BorrowState::Dropped if inner.weak_count.get() == 0) {
            // SAFETY: this is being called only once we know no other usages of the value can
            // exist
            unsafe { InnerStore::dealloc(self.inner) }
        }
    }
}

///////////////////////////
// Public `SliceRef` API //
///////////////////////////

#[track_caller]
fn bump_weak_count(count: &Cell<usize>) {
    let new_count = count
        .get()
        .checked_add(1)
        .expect("there should be fewer than usize::MAX concurrent `SliceRef`s");
    count.set(new_count);
}

impl<I, S, const M: usize> SliceRef<I, S, M> {
    /// Produces a reference to the `InnerStore`
    fn inner(&self) -> &InnerStore<I, S, M> {
        // SAFETY: Existence of a `SliceRef` guarantees that it's valid to construct a reference to
        // the `InnerStore`. The places where we "drop" the store aren't leaving behind initialized
        // memory
        unsafe { &*self.inner.as_ptr() }
    }

    fn borrow_refs(&self) -> Result<&RefCell<RecycleVec<StoredSliceRef<I, S, M>>>, BorrowFailure> {
        let inner = self.inner();
        let res = match inner.borrow.get() {
            BorrowState::NotBorrowed => Ok(BorrowState::Immutable {
                count: NonZeroUsize::new(1).unwrap(),
                drop_if_zero: false,
            }),
            BorrowState::Mutable => Err(BorrowFailure::CannotGetImmutAlreadyMutablyBorrowed),
            BorrowState::Immutable { mut count, drop_if_zero } => {
                count = count
                    .checked_add(1)
                    .expect("there should be fewer than usize::MAX concurrent borrows");

                Ok(BorrowState::Immutable { count, drop_if_zero })
            }
            BorrowState::Dropping | BorrowState::Dropped => {
                Err(BorrowFailure::CannotGetImmutAlreadyDropped)
            }
        };

        res.map(|new_state| {
            inner.borrow.set(new_state);
            &inner.refs
        })
    }

    // Like `borrow_refs`, but doesn't check that we have permissions to access the tree
    fn get_refs_without_borrow(&self) -> Option<&RefCell<RecycleVec<StoredSliceRef<I, S, M>>>> {
        let inner = self.inner();
        match inner.borrow.get() {
            BorrowState::NotBorrowed | BorrowState::Immutable { .. } | BorrowState::Mutable => {
                Some(&inner.refs)
            }
            BorrowState::Dropping | BorrowState::Dropped => None,
        }
    }

    fn collapse_refs<'r>(
        &self,
        refs: &'r RefCell<RecycleVec<StoredSliceRef<I, S, M>>>,
    ) -> Option<
        cell::Ref<'r, Option<SliceHandle<ty::Unknown, borrow::SliceRef, I, S, AllowSliceRefs, M>>>,
    > {
        let this_id = self.id.take()?;
        let mut redirected = false;

        // Fast path: immediately return the borrow, if there's no redirections. We only have one
        // way to fail here, so we set `redirected = true` if we encounter a redirection.
        //
        // The borrow checker failed to accept a version of this function that instead set
        // `redirected` to the `RefId` of the redirection, so we have a couple usages of
        // `unreachable!` later on instead.
        let initial_result = cell::Ref::filter_map(refs.borrow(), |r| match r.get(&this_id) {
            StoredSliceRef::Handle(handle) => Some(handle),
            StoredSliceRef::Redirect(_) => {
                redirected = true;
                None
            }
            StoredSliceRef::Removed => None,
        });

        match initial_result {
            Ok(handle_ref) => {
                self.id.set(Some(this_id));
                return Some(handle_ref);
            }
            Err(_) if redirected => (),
            Err(_) => {
                // The value was removed, but we still have a `RefId` pointing to its slot. We
                // should remove our `RefId` from the `RecycleVec`, and maybe free up that space.
                let _ = refs.borrow_mut().recycle(this_id);
                return None;
            }
        }

        // Slow path: `this_id` is redirected, so we need to (a) find the final `RefId` referred
        // to by the series of `Redirect`s and (b) overwrite all of the `RefId`s along the path. We
        // do this in two passes.

        let mut next_id = None;
        let mut refs_mut = refs.borrow_mut();

        let final_id = loop {
            let id = next_id.as_ref().unwrap_or(&this_id);
            let next = match refs_mut.get(id) {
                StoredSliceRef::Removed => break None,
                StoredSliceRef::Handle(_) => break Some(next_id.unwrap()),
                StoredSliceRef::Redirect(id) => id,
            };

            if let Some(old) = next_id.replace(refs_mut.clone(next)) {
                let entry = refs_mut.recycle(old);
                debug_assert!(entry.is_none());
            }
        };

        // Now that we have a final `RefId`, we start from `this_id` again and set all of the
        // values as we traverse along

        let mut next_id = refs.borrow().clone(&this_id);

        loop {
            let new_val = match final_id.as_ref() {
                Some(id) => StoredSliceRef::Redirect(refs_mut.clone(id)),
                None => StoredSliceRef::Removed,
            };

            match refs_mut.replace(&next_id, new_val) {
                StoredSliceRef::Redirect(next) => next_id = next,
                StoredSliceRef::Removed | StoredSliceRef::Handle(_) => {
                    let entry = refs_mut.recycle(next_id);
                    assert!(entry.is_none());
                    break;
                }
            }
        }

        drop(refs_mut);
        let r = final_id.as_ref().map(|id| {
            cell::Ref::map(refs.borrow(), |r| match r.get(id) {
                StoredSliceRef::Handle(h) => h,
                StoredSliceRef::Removed | StoredSliceRef::Redirect(_) => unreachable!(),
            })
        });
        self.id.set(final_id);
        r
    }

    /// Returns whether the contents referred to by this reference are still available to be
    /// accessed
    ///
    /// This method primarily exists to allow checking that a series of calls to other methods
    /// won't panic, or for debug assertions. It should be used alongside [`can_borrow`].
    ///
    /// **Note:** If `slice_ref.is_valid()` returns `true`, it is only guaranteed to still be true
    /// until the next mutation to the tree. If `is_valid` returns `false` once, then it will not
    /// ever be found valid again.
    ///
    /// **See also:** [`can_borrow`]
    ///
    /// [`can_borrow`]: Self::can_borrow
    /// [`RleTree`]: crate::RleTree
    /// [`Slice::try_join`]: crate::Slice::try_join
    pub fn is_valid(&self) -> bool {
        let refs = match self.get_refs_without_borrow() {
            Some(r) => r,
            None => return false,
        };

        self.collapse_refs(refs).is_some()
    }

    /// Returns whether this reference is currently able to access anything in the tree it refers
    /// to
    ///
    /// This method returns `false` if the tree is mutably borrowed or has been dropped, and `true`
    /// otherwise. It exists to allow checking that a series of calls to infallible methods won't
    /// panic, and should be used alongside [`is_valid`]
    ///
    /// **See also:** [`is_valid`].
    ///
    /// [`is_valid`]: Self::is_valid
    pub fn can_borrow(&self) -> bool {
        match self.inner().borrow.get() {
            BorrowState::NotBorrowed | BorrowState::Immutable { .. } => true,
            BorrowState::Mutable | BorrowState::Dropping | BorrowState::Dropped => false,
        }
    }

    /// Helper function to produce the panic message for failure with one of our infalliblle
    /// methods
    fn panic_msg(failure: Option<BorrowFailure>) -> &'static str {
        match failure {
            None => "cannot borrow: slice is no longer present in the tree",
            Some(e) => match e {
                BorrowFailure::CannotGetMutAlreadyBorrowed => unreachable!(),
                BorrowFailure::CannotGetImmutAlreadyMutablyBorrowed => {
                    "cannot borrow: tree is already mutably borrowed"
                }
                BorrowFailure::CannotGetImmutAlreadyDropped => {
                    "cannot borrow: tree has been dropped"
                }
            },
        }
    }

    /// Internal function to create a temporary borrow on the `SliceHandle` that `self` references
    ///
    /// This method returns `Err(_)` on failure when the tree cannot be accessed, and sets
    /// `borrow.handle = None` when the slice has been removed.
    fn try_temporary_borrow(&self) -> Result<TemporaryBorrow<I, S, M>, BorrowFailure> {
        let refs = self.borrow_refs()?;

        let unwrap_msg =
            "slice handles should not be suspended while the tree is not mutably borrowed";

        Ok(TemporaryBorrow {
            handle: self
                .collapse_refs(refs)
                .map(|r| cell::Ref::map(r, |opt| opt.as_ref().expect(unwrap_msg))),
            inner: self.inner(),
            ownership_transferred: false,
        })
    }

    /// Returns a borrow on the slice referred to by this `SliceRef`
    ///
    /// ## Panics
    ///
    /// This method will panic if the slice reference is no longer [valid] or the tree
    /// [cannot be borrowed] right now. For a fallible version of this method, see:
    /// [`try_borrow_slice`].
    ///
    /// [valid]: Self::is_valid
    /// [cannot be borrowed]: Self::can_borrow
    /// [`try_borrow_slice`]: Self::try_borrow_slice
    pub fn borrow_slice(&self) -> Borrow<S> {
        let panic_msg = match self.borrow_slice_internal() {
            Ok(Some(b)) => return b,
            Ok(None) => Self::panic_msg(None),
            Err(e) => Self::panic_msg(Some(e)),
        };

        panic!("{panic_msg}");
    }

    /// Returns a borrow on the slice referred to by this `SliceRef`, returning `None` if the
    /// reference isn't valid or can't be borrowed right now
    ///
    /// This is a fallible version of [`borrow_slice`](Self::borrow_slice).
    pub fn try_borrow_slice(&self) -> Option<Borrow<S>> {
        match self.borrow_slice_internal() {
            Ok(Some(b)) => Some(b),
            Ok(None) | Err(_) => None,
        }
    }

    /// Internal function with shared logic between [`borrow_slice`] and [`try_borrow_slice`]
    ///
    /// This method returns `Err(_)` on failure when the tree cannot be accessed, and `Ok(None)`
    /// when the slice has been removed.
    ///
    /// [`borrow_slice`]: Self::borrow_slice
    /// [`try_borrow_slice`]: Self::try_borrow_slice
    fn borrow_slice_internal<'r>(&'r self) -> Result<Option<Borrow<'r, S>>, BorrowFailure> {
        match self.try_temporary_borrow() {
            Ok(b) if b.handle.is_some() => Ok(Some(b.into_borrow())),
            Ok(_) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Returns the range of values covered by this slice
    ///
    /// ## Panics
    ///
    /// This method will panic if the slice reference is no longer [valid] or the tree
    /// [cannot be borrowed] right now. For a fallible version of this method, see:
    /// [`try_range`].
    ///
    /// [valid]: Self::is_valid
    /// [cannot be borrowed]: Self::can_borrow
    /// [`try_range`]: Self::try_range
    pub fn range(&self) -> Range<I>
    where
        I: Index,
    {
        let panic_msg = match self.range_internal() {
            Ok(Some(b)) => return b,
            Ok(None) => Self::panic_msg(None),
            Err(e) => Self::panic_msg(Some(e)),
        };

        panic!("{panic_msg}");
    }

    /// Returns the range of values covered by this slice, or `None` if the reference isn't valid
    /// or can't be borrowed right now
    ///
    /// This is a fallible version of [`range`](Self::range).
    pub fn try_range(&self) -> Option<Range<I>>
    where
        I: Index,
    {
        match self.range_internal() {
            Ok(Some(range)) => Some(range),
            Ok(None) | Err(_) => None,
        }
    }

    /// Internal function with shared logic between [`range`] and [`try_range`]
    ///
    /// This method returns `Err(_)` on failure when the tree cannot be accessed, and `Ok(None)`
    /// when the slice has been removed.
    ///
    /// [`range`]: Self::range
    /// [`try_range`]: Self::try_range
    fn range_internal(&self) -> Result<Option<Range<I>>, BorrowFailure>
    where
        I: Index,
    {
        let b = self.try_temporary_borrow()?;
        Ok(b.handle.as_ref().map(|h| h.range()))
    }
}

/// Cloning a `SliceRef` -- even an invalid one -- is always possible, and will never panic
impl<I, S, const M: usize> Clone for SliceRef<I, S, M> {
    fn clone(&self) -> Self {
        let id = self.id.take().and_then(|this_id| match self.get_refs_without_borrow() {
            None => None,
            Some(r) => {
                let refs = r.borrow();
                bump_weak_count(&self.inner().weak_count);
                let new_id = refs.clone(&this_id);
                self.id.set(Some(this_id));
                Some(new_id)
            }
        });

        SliceRef {
            inner: self.inner,
            id: Cell::new(id),
            marker: PhantomData,
        }
    }
}

struct TemporaryBorrow<'r, I, S, const M: usize> {
    // the `cell::Ref` needs to be behind a `ManuallyDrop` so that we don't try to drop the
    // contents of the `InnerStore` while it's still borrowed; we need to drop the `Ref` first.
    handle:
        Option<cell::Ref<'r, SliceHandle<ty::Unknown, borrow::SliceRef, I, S, AllowSliceRefs, M>>>,
    inner: &'r InnerStore<I, S, M>,
    ownership_transferred: bool,
}

#[cfg(test)]
impl<'r, I, S, const M: usize> Debug for TemporaryBorrow<'r, I, S, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        struct H(NonNull<()>, u8);

        impl Debug for H {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                f.write_fmt(format_args!("@{:p}, idx = {:?}", self.0, self.1))
            }
        }

        let handle = self.handle.as_ref().map(|h| H(h.node.ptr().cast(), h.idx));

        f.debug_struct("TemporaryBorrow")
            .field("handle", &handle)
            .field("ownership_transferred", &self.ownership_transferred)
            .finish()
    }
}

impl<'r, I, S, const M: usize> TemporaryBorrow<'r, I, S, M> {
    /// Converts the temporary borrow into a longer-term one
    ///
    /// ## Panics
    ///
    /// This method panics if `self.handle` is `None`
    fn into_borrow(mut self) -> Borrow<'r, S> {
        let handle = self
            .handle
            .as_ref()
            .expect("`into_borrow` should not be called with `self.handle = None`");

        // SAFETY: `borrow_slice` returns any lifetime we like; it's up to us to ensure that we
        // don't use it improperly. We know that it'll be sound because: (a) the `BorrowState`
        // ensures that the tree won't be modified out from underneath us, and (b) the tree is
        // not dropped until *at least* the `Borrow` is dropped.
        let val = unsafe { handle.borrow_slice() };

        self.ownership_transferred = true;
        Borrow {
            val,
            borrow: &self.inner.borrow,
            data_ptr: NonNull::from(self.inner).cast::<()>(),
            drop_fn: Self::drop_tree_once_no_borrows,
        }
    }

    unsafe fn drop_tree_once_no_borrows(data_ptr: NonNull<()>) {
        let store = data_ptr.cast::<InnerStore<I, S, M>>();
        unsafe { store.as_ref().drop(true) }
    }
}

impl<'r, I, S, const M: usize> Drop for TemporaryBorrow<'r, I, S, M> {
    fn drop(&mut self) {
        // Explicitly drop the `cell::Ref` early so that we don't run into borrow conflicts if we
        // need to drop the `InnerStore` as well.
        if let Some(handle_ref) = self.handle.take() {
            drop(handle_ref);
        }
        if self.ownership_transferred {
            return;
        }

        let data_ptr = NonNull::from(self.inner);

        // SAFETY: `drop_borrow` requires that it's called as the only operation in the destructor
        // of `Borrow` or `TemporaryBorrow`. Strictly speaking, that's not *quite* true here, but
        // the other stuff we're doing is both necessary and doesn't affect its soundness.
        unsafe {
            drop_borrow(&self.inner.borrow, data_ptr.cast(), Self::drop_tree_once_no_borrows)
        };
    }
}

#[cfg(not(feature = "nightly"))]
impl<I, S, const M: usize> Drop for SliceRef<I, S, M> {
    fn drop(&mut self) {
        // SAFETY: `do_drop` requires only that it's called as the implementation of `Drop`
        unsafe { self.do_drop() }
    }
}

#[cfg(feature = "nightly")]
impl<I, S, const M: usize> Drop for SliceRef<I, S, M> {
    fn drop(&mut self) {
        // SAFETY: `do_drop` requires only that it's called as the implementation of `Drop`
        unsafe { self.do_drop() }
    }
}

impl<I, S, const M: usize> SliceRef<I, S, M> {
    /// Actual `Drop` implementation
    ///
    /// ## Safety
    ///
    /// This method can only be called as the only operation inside the implementation of `Drop`
    /// for `SliceRef`.
    unsafe fn do_drop(&mut self) {
        // A dropping `SliceRef` is never responsible for calling `InnerStore::drop`, although it
        // is sometimes responsible for deallocating the `InnerStore`, if the contents have been
        // dropped and `self` is the last "weak" reference to the allocation.

        let old_weak_count = self.inner().weak_count.get();
        // SAFETY: the weak count cannot currently be zero because there's an existing `SliceRef`.
        // If this happens to be true, it's probably already a use-after-free to read `weak_count`
        unsafe { weak_assert!(old_weak_count != 0) };

        let weak_count = old_weak_count - 1;
        self.inner().weak_count.set(weak_count);
        // If this was the last weak reference and the main tree has been dropped, then we need to
        // deallocate the backing allocation for the `InnerStore`:
        if weak_count == 0 && matches!(self.inner().borrow.get(), BorrowState::Dropped) {
            // SAFETY: `do_drop` is only ever called in the destructor for `SliceRef` and we know
            // there are no longer any other references to the `InnerStore`, so it's safe to
            // deallcoate.
            unsafe { InnerStore::dealloc(self.inner) }
        }
    }
}

/////////////////////////
// Public `Borrow` API //
/////////////////////////

impl<'a, T> Deref for Borrow<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.val
    }
}

impl<'a, T> Borrow<'a, T> {
    /// Maps the reference using a component value
    ///
    /// This function cannot be a method, as it would conflict with possible methods on `&T`
    /// itself. As such, it must be called as `Borrow::map(...)`.
    pub fn map<U>(this: Self, f: impl FnOnce(&T) -> &U) -> Borrow<'a, U> {
        let new = Borrow {
            val: f(this.val),
            borrow: this.borrow,
            data_ptr: this.data_ptr,
            drop_fn: this.drop_fn,
        };
        mem::forget(this);
        new
    }
}

impl<'a, T> Clone for Borrow<'a, T> {
    fn clone(&self) -> Self {
        // bump the active borrow count
        match self.borrow.get() {
            BorrowState::Immutable { mut count, drop_if_zero } => {
                count = count
                    .checked_add(1)
                    .expect("should not have more than usize::MAX immutable borrows");
                self.borrow.set(BorrowState::Immutable { count, drop_if_zero });
            }
            _ => unreachable!(),
        }

        Borrow {
            val: self.val,
            borrow: self.borrow,
            data_ptr: self.data_ptr,
            drop_fn: self.drop_fn,
        }
    }
}

#[cfg(not(feature = "nightly"))]
impl<'a, T> Drop for Borrow<'a, T> {
    fn drop(&mut self) {
        // SAFETY: `drop_borrow` requires that we're calling it from `Borrow`'s destructor
        unsafe { drop_borrow(self.borrow, self.data_ptr, self.drop_fn) };
    }
}

// SAFETY: we're clearly not using our reference to `T` here, so we're all good. The safety around
// whether `I` or `S` can dangle is handled separately, *marked* at the destructor for `SliceRef`,
// even though it actually occurs here -- they can dangle here, so they can still dangle later.
#[cfg(feature = "nightly")]
unsafe impl<'a, #[may_dangle] T> Drop for Borrow<'a, T> {
    fn drop(&mut self) {
        // SAFETY: `drop_borrow` requires that we're calling it from `Borrow`'s destructor
        unsafe { drop_borrow(self.borrow, self.data_ptr, self.drop_fn) };
    }
}

/// Actually performs the destructor for a [`Borrow`]
///
/// This is extracted out so that we can have separate `Drop` implementations, with or without
/// `#[may_dangle]`. It's also a freestanding function because it doesn't need to be generic over
/// the type of the [`Borrow`].
///
/// ## Safety
///
/// This method can only be called from the destructor of [`Borrow`] or [`TemporaryBorrow`], and
/// must be called as the only operation.
unsafe fn drop_borrow(
    state: &Cell<BorrowState>,
    data_ptr: NonNull<()>,
    drop_fn: unsafe fn(NonNull<()>),
) {
    match state.get() {
        BorrowState::Immutable { count, drop_if_zero } => {
            match NonZeroUsize::new(count.get() - 1) {
                Some(count) => {
                    state.set(BorrowState::Immutable { count, drop_if_zero });
                    return;
                }
                None if drop_if_zero => state.set(BorrowState::Dropped),
                None => {
                    state.set(BorrowState::NotBorrowed);
                    return;
                }
            }
        }
        // SAFETY: existence of the borrow guarantees that the
        _ => unsafe { weak_unreachable!() },
    }

    // SAFETY: calling `drop_fn` is only legal if `drop_if_zero` is true *and* we've decreased the
    // count to zero
    unsafe { (drop_fn)(data_ptr) }
}

//////////////////////////////////////////
// Internal `InnerStore` helper methods //
//////////////////////////////////////////

impl<I, S, const M: usize> InnerStore<I, S, M> {
    /// Allocates a new `InnerStore`
    fn alloc(self) -> NonNull<Self> {
        let layout = Layout::new::<Self>();

        // SAFETY: `alloc` requires that the layout is non-zero sized, which is true here.
        let maybe_null_ptr = unsafe { alloc::alloc(layout) } as *mut Self;
        match NonNull::new(maybe_null_ptr) {
            Some(p) => {
                // SAFETY: `write` requires that `p.as_ptr()` is valid for writes and properly
                // aligned. This is guaranteed in the case of a non-null return from `alloc`
                unsafe { p.as_ptr().write(self) };
                p
            }
            None => alloc::handle_alloc_error(layout),
        }
    }

    /// Deallocates the inner component of the store, without calling the destructors of any inner
    /// values
    ///
    /// ## Safety
    ///
    /// The typical safety requirements for deallocation apply -- that `this` corresponds to a
    /// previous return from `Self::alloc` and that it hasn't already been deallocated.
    unsafe fn dealloc(this: NonNull<Self>) {
        let layout = Layout::new::<Self>();
        // SAFETY: guaranteed by caller
        unsafe { alloc::dealloc(this.as_ptr() as *mut u8, layout) };
    }

    /// Empties the contents of the `InnerStore`, removing the allocation for `refs` and then
    /// destructing the `RleTre`
    fn drop(&self, drop_tree: bool) {
        drop(self.refs.take());

        if drop_tree {
            let root = self.root.take();
            if let Some(handle) = root {
                handle.try_drop().unwrap().do_drop()
            }
        }
    }
}
