//! Public [`SliceRef`] type and internal handling
//!
//! One of the difficulties in the implementation here is that our reference counting is
//! centralized -- there isn't an easy or efficient way around this, but it has the unfortunate
//! side effect that we essentially need to *always* be ready to record `Drop`s or `Clone`s of
//! existing [`SliceRef`]s, even if the tree is currently mutably borrowed.

use std::cell::{Cell, RefCell};
use std::fmt::{self, Display, Formatter};
use std::mem::{self, ManuallyDrop};
use std::num::NonZeroUsize;
use std::ptr::NonNull;
use std::rc::{self, Rc};

use super::node::SliceHandle;
use super::DEFAULT_MIN_KEYS;
use crate::param::{self, AllowSliceRefs};
use crate::recycle::{self, RecycleVec};

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
    store: rc::Weak<SliceRefStore<I, S>>,
    id: ManuallyDrop<RefId>,
}

/// Unique identifier for a slice reference, corresponding to an index in the internal vector
pub type RefId = recycle::EntryId;

/// Tree-wide store redirecting slice references
pub struct SliceRefStore<I, S> {
    inner: Rc<SliceRefStoreInner<I, S>>,
}

/// The inner, shared component of the `SliceRefStore`
struct SliceRefStoreInner<I, S> {
    /// The state of the borrow on the whole tree
    ///
    /// Note: this is separate from the borrow on `refs`, which controls temporary access to
    /// `refs`, irrespective of the current borrow on the rest of the tree. `refs` is cleared when
    /// `borrow` is changed to `BorrowState::Dropped`
    borrow: Cell<BorrowState>,
    refs: RefCell<RecycleVec<StoredSliceRef<I, S>>>,
    root: Cell<Option<NonNull<(I, S)>>>,
}

// TODO: We can probably make this type smaller, although a substantial result would probably
// depend on cooperation from `RecycleVec` for merging enum variants with this particular type.
enum StoredSliceRef<I, S> {
    // The original pointee of this reference was joined with something else; go there for more
    // info.
    Redirect(RefId),
    // A pointer to the node, plus the height of the node, plus the index of the slice in it
    Ptr(NonNull<(I, S)>, u8, u8),
    // The slice no longer exists (it was probably removed)
    Removed,
}

impl<I, S> Default for SliceRefStore<I, S> {
    fn default() -> Self {
        SliceRefStore {
            inner: Rc::new(SliceRefStoreInner {
                borrow: Cell::default(),
                refs: RefCell::new(RecycleVec::default()),
                root: Cell::new(None),
            }),
        }
    }
}

//////////////////////////////////////////////////////
// param::SliceRefStore trait and implementation(s) //
//////////////////////////////////////////////////////

impl<I, S> param::SliceRefStore<I, S> for () {
    type OptionRefId = ();

    fn join_refs(&mut self, _: (), _: ()) {}
    fn remove(&mut self, _: ()) {}
    unsafe fn update(&mut self, _: &(), _: NonNull<(I, S)>, _: u8) {}
    fn suspend(&mut self, _: &()) {}
}

impl<I, S> param::SliceRefStore<I, S> for SliceRefStore<I, S> {
    type OptionRefId = Option<RefId>;

    fn join_refs(&mut self, rx: Option<RefId>, ry: Option<RefId>) -> Option<RefId> {
        // Simple cases:
        let (mut x, mut y) = match (rx, ry) {
            (None, None) => return None,
            (r @ Some(_), None) | (None, r @ Some(_)) => return r,
            (Some(x), Some(y)) => (x, y),
        };

        // When we join the two `RefId`s, one of them turns into a `Redirect` to the other. One
        // potential problem we'd like to avoid is inadvertently creating long "redirect" chains.
        //
        // If, for example, we always prefered the left-hand or right-hand option, then certain
        // access patterns on the `RleTree` could create long redirect chains. So we'd like to have
        // something that *somewhat* randomly picks between `x` or `y`.
        //
        // In order to make this testable (and sufficiently efficient) it can't *actually* be
        // random. So we take the second-to-last bit of `x ^ y` and if it's one, keep `y`.
        // Otherwise use `x`.
        if ((x.idx() ^ y.idx()) & 0x02) >> 1 == 1 {
            mem::swap(&mut x, &mut y);
        }

        // Join y into x:
        let mut refs = self.inner.refs.borrow_mut();
        let x_ref = refs.clone(&x);
        refs.set(&y, StoredSliceRef::Redirect(x_ref));
        if refs.recycle(y).is_some() {
            // Recycling `y` should never return a value; our invariant maintains that slices only
            // store a `RefId` for themselves if there's at least one other reference.
            unreachable!("slices should never hold the only reference to themselves");
        }
        Some(x)
    }

    fn remove(&mut self, r: Option<RefId>) {
        let id = match r {
            None => return,
            Some(id) => id,
        };

        let mut refs = self.inner.refs.borrow_mut();
        refs.set(&id, StoredSliceRef::Removed);
        if refs.recycle(id).is_some() {
            // Recycling this id should never return a value; our invariant maintains that slices
            // only store a `RefId` for themselves if there's at least one other reference.
            unreachable!("slices should never hold the only reference to themselves");
        }
    }

    unsafe fn update(&mut self, r: &Option<RefId>, ptr: NonNull<(I, S)>, idx: u8) {
        todo!()
    }

    fn suspend(&mut self, r: &Option<RefId>) {
        todo!()
    }
}

impl<I, S> SliceRefStore<I, S> {
    fn drop_id<const M: usize>(&self, mut id: RefId) {
        // This is a loop because we can end up with cascading drops from `Redirect`s
        let mut refs = self.inner.refs.borrow_mut();
        loop {
            match refs.get(&id) {
                // If there's only two references left to a pointer, the node's copy is the only
                // other reference. It should be removed first.
                StoredSliceRef::Ptr(p, height, k_idx) if refs.ref_count(&id).get() == 2 => {
                    // SAFETY: The args are valid because they exist in the `SliceRefStore`. The
                    // handle isn't being used beyond the end of this loop iteration, so it
                    // definitely won't outlive any part of the tree.
                    let mut handle = unsafe {
                        <SliceHandle<_, _, I, S, AllowSliceRefs, M>>::slice_ref_from_parts(
                            p.cast(),
                            *height,
                            *k_idx,
                        )
                    };

                    match handle.take_refid() {
                        Some(id) => match refs.recycle(id) {
                            None => (),
                            // SAFETY: we already know that the ref count is 2, so recycling this
                            // reference shouldn't give us back the original value. Explicitly
                            // marking this unreachable is only helpful because it avoids drop glue
                            Some(_) => unsafe { weak_unreachable!() },
                        },
                        // SAFETY: The existence of the `StoredSliceRef::Ptr` guarantees that the
                        // slice it refers to *does* have a reference.
                        None => unsafe { weak_unreachable!() },
                    }
                }
                _ => (),
            }

            // Actually recycle this reference
            match self.inner.refs.borrow_mut().recycle(id) {
                // If this was the last reference to a `Redirect`, then drop that RefId as well
                Some(StoredSliceRef::Redirect(i)) => id = i,
                _ => break,
            }
        }
    }

    fn clone_id(&self, id: &RefId) -> RefId {
        self.inner.refs.borrow_mut().clone(id)
    }
}

////////////////////////////////
// SliceRef implementation(s) //
////////////////////////////////

impl<I, S, const M: usize> SliceRef<I, S, M> {
    pub(super) fn new(store: &SliceRefStore<I, S>, id: RefId) -> Self {
        todo!()
    }
}

impl<I, S, const M: usize> Drop for SliceRef<I, S, M> {
    fn drop(&mut self) {
        // SAFETY: This is inside the destructor for `SliceRef`, so it's safe to a value out of a
        // `ManuallyDrop` only if we don't access the field again. We don't.
        let id = unsafe { ManuallyDrop::take(&mut self.id) };
        if let Some(store) = self.store.upgrade() {
            store.drop_id::<M>(id);
        }
    }
}

impl<I, S, const M: usize> Clone for SliceRef<I, S, M> {
    fn clone(&self) -> Self {
        SliceRef {
            id: match self.store.upgrade() {
                Some(store) => ManuallyDrop::new(store.clone_id(&self.id)),
                // SAFETY: we know that the backing `RecycleVec` has been dropped becuse we
                // couldn't get an `Rc` to it.
                None => ManuallyDrop::new(unsafe { self.id.clone_because_vec_is_dropped() }),
            },
            store: self.store.clone(),
        }
    }
}

//////////////////
// Borrow stuff //
//////////////////

/// Information about the whether and how an `RleTree` is currently borrowed
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub enum BorrowState {
    #[default]
    NotBorrowed,
    Mutable,
    Immutable {
        count: NonZeroUsize,
        drop_if_zero: bool,
    },
    Dropped,
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

/// Binary flag for whether the [`RleTree`] should be dropped on releasing a borrow
///
/// [`RleTree`]: crate::RleTree
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ShouldDrop {
    No,
    Yes,
}

#[rustfmt::skip]
impl param::BorrowState for () {
    fn acquire_immutable(&self) -> Result<(), BorrowFailure> { Ok(()) }
    fn release_immutable(&self) -> ShouldDrop { ShouldDrop::No }
    fn acquire_mutable(&self) -> Result<(), BorrowFailure> { Ok(()) }
    fn release_mutable(&self) {}
    fn try_acquire_drop(&self) -> ShouldDrop { ShouldDrop::Yes }
}

impl param::sealed::YouCantImplementThis for Cell<BorrowState> {}

impl param::BorrowState for Cell<BorrowState> {
    fn acquire_immutable(&self) -> Result<(), BorrowFailure> {
        let mut drop_if_zero = false;

        let new_count = match self.get() {
            BorrowState::Mutable => {
                return Err(BorrowFailure::CannotGetImmutAlreadyMutablyBorrowed)
            }
            BorrowState::Dropped => return Err(BorrowFailure::CannotGetImmutAlreadyDropped),
            BorrowState::Immutable {
                count,
                drop_if_zero: d,
            } => {
                drop_if_zero = d;
                count
                    .checked_add(1)
                    .expect("there should be fewer than usize::MAX concurrent borrows")
            }
            BorrowState::NotBorrowed => NonZeroUsize::new(1).unwrap(),
        };

        self.set(BorrowState::Immutable {
            count: new_count,
            drop_if_zero,
        });
        Ok(())
    }

    fn release_immutable(&self) -> ShouldDrop {
        match self.get() {
            BorrowState::Immutable {
                count,
                drop_if_zero,
            } => {
                let new_count = NonZeroUsize::new(count.get() - 1);

                if matches!(new_count, None if drop_if_zero) {
                    self.set(BorrowState::Dropped);
                    ShouldDrop::Yes
                } else {
                    match new_count {
                        Some(c) => self.set(BorrowState::Immutable {
                            count: c,
                            drop_if_zero,
                        }),
                        None => self.set(BorrowState::NotBorrowed),
                    }
                    ShouldDrop::No
                }
            }
            b => unreachable!("tried to release immutable borrow on {:?}", b),
        }
    }

    fn acquire_mutable(&self) -> Result<(), BorrowFailure> {
        match self.get() {
            BorrowState::NotBorrowed => {
                self.set(BorrowState::Mutable);
                Ok(())
            }
            BorrowState::Immutable { .. } => Err(BorrowFailure::CannotGetMutAlreadyBorrowed),
            BorrowState::Mutable => unreachable!("double mutable borrow attempted"),
            BorrowState::Dropped => unreachable!("mutable borrow attempted after drop"),
        }
    }

    fn release_mutable(&self) {
        assert!(
            self.get() == BorrowState::Mutable,
            "tried to release mutable borrow on {:?}",
            self.get(),
        );
        self.set(BorrowState::NotBorrowed);
    }

    fn try_acquire_drop(&self) -> ShouldDrop {
        match self.get() {
            BorrowState::Mutable => unreachable!("cannot drop while mutably borrowed"),
            BorrowState::Dropped
            | BorrowState::Immutable {
                drop_if_zero: true, ..
            } => {
                unreachable!("double-drop attempted")
            }
            BorrowState::NotBorrowed => {
                self.set(BorrowState::Dropped);
                ShouldDrop::Yes
            }
            BorrowState::Immutable { count, .. } => {
                self.set(BorrowState::Immutable {
                    count,
                    drop_if_zero: true,
                });
                ShouldDrop::No
            }
        }
    }
}

impl<I, S> param::sealed::YouCantImplementThis for SliceRefStore<I, S> {}

impl<I, S> param::BorrowState for SliceRefStore<I, S> {
    fn acquire_immutable(&self) -> Result<(), BorrowFailure> {
        self.inner.borrow.acquire_immutable()
    }

    fn release_immutable(&self) -> ShouldDrop {
        self.inner.borrow.release_immutable()
    }

    fn acquire_mutable(&self) -> Result<(), BorrowFailure> {
        self.inner.borrow.acquire_mutable()
    }

    fn release_mutable(&self) {
        self.inner.borrow.release_mutable()
    }

    fn try_acquire_drop(&self) -> ShouldDrop {
        self.inner.borrow.try_acquire_drop()
    }
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
