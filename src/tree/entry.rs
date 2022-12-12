//! Wrapper module for [`SliceEntry`]

use super::node::{borrow, ty, SliceHandle};
use super::{SliceRef, DEFAULT_MIN_KEYS};
use crate::param::{AllowSliceRefs, NoFeatures, RleTreeConfig};
use crate::{Cursor, Index, PathComponent};
use std::ops::Range;
use std::panic::{RefUnwindSafe, UnwindSafe};

/// Information about a single slice in an [`RleTree`], returned by [`get`] or yielded by [`iter`]
///
/// Conceptually, this type is not too different from `(Range<I>, &'t S)`, but the methods provide
/// some additional functionality that wouldn't be available with a simpler type.
///
/// More information is available in the methods themselves.
///
/// [`RleTree`]: crate::RleTree
/// [`get`]: crate::RleTree::get
/// [`iter`]: crate::RleTree::iter
pub struct SliceEntry<'t, I, S, P, const M: usize = DEFAULT_MIN_KEYS>
where
    P: RleTreeConfig<I, S, M>,
{
    // This is just a `Range<I>`, but extracted into separate fields so what we can implement
    // `Copy` for this type.
    pub(super) range_start: I,
    pub(super) range_end: I,
    pub(super) slice: SliceHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
    pub(super) store: &'t P::SliceRefStore,
}

impl<'t, I: Clone, S, P: RleTreeConfig<I, S, M>, const M: usize> Clone
    for SliceEntry<'t, I, S, P, M>
{
    fn clone(&self) -> Self {
        SliceEntry {
            range_start: self.range_start.clone(),
            range_end: self.range_end.clone(),
            slice: self.slice,
            store: self.store,
        }
    }
}

#[rustfmt::skip]
mod marker_impls {
    use super::*;

    impl<'t, I: Copy, S, P, const M: usize> Copy for SliceEntry<'t, I, S, P, M>
    where
        P: RleTreeConfig<I, S, M>,
    {}

    impl<'t, I: UnwindSafe + RefUnwindSafe, S: RefUnwindSafe, P, const M: usize> UnwindSafe
        for SliceEntry<'t, I, S, P, M>
    where
        P: RleTreeConfig<I, S, M>,
    {}

    impl<'t, I: RefUnwindSafe, S: RefUnwindSafe, P, const M: usize> RefUnwindSafe
        for SliceEntry<'t, I, S, P, M>
    where
        P: RleTreeConfig<I, S, M>,
    {}

    unsafe impl<'t, I: Send + Sync, S: Sync, P: Sync, const M: usize> Send
        for SliceEntry<'t, I, S, P, M>
    where
        P: RleTreeConfig<I, S, M>,
    {}

    unsafe impl<'t, I: Sync, S: Sync, P: Sync, const M: usize> Sync for SliceEntry<'t, I, S, P, M>
    where
        P: RleTreeConfig<I, S, M>
    {}
}

impl<'t, I, S, P, const M: usize> SliceEntry<'t, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
    I: Index,
{
    /// Returns the range of values covered by this entry
    pub fn range(&self) -> Range<I> {
        self.range_start..self.range_end
    }

    /// Returns the size of the range of values covered by this entry
    ///
    /// This is essentially a convenience method roughly equivalent to `self.range().len()`.
    pub fn size(&self) -> I {
        self.range_end.sub_right(self.range_start)
    }

    /// Returns a reference to the slice for this entry
    pub fn slice(&self) -> &'t S {
        self.slice.into_ref()
    }
}

/// Shared helper function to produce a cursor to a [`SliceHandle`]
///
/// ## Panics
///
/// This method will panic if called with `P = AllowCow`.
pub(super) fn cursor_to_handle<C: Cursor, I, S, P: RleTreeConfig<I, S, M>, const M: usize>(
    handle: SliceHandle<ty::Unknown, borrow::Immut, I, S, P, M>,
) -> C {
    assert!(!P::COW);

    let mut node = handle.node;
    let mut cursor = C::new_empty();

    // Short-circuit for no-op cursors
    if C::IS_NOP {
        return cursor;
    }

    while let Ok((parent, child_idx)) = node.into_parent() {
        cursor.prepend_to_path(PathComponent { child_idx });
        node = parent.erase_type();
    }

    cursor
}

macro_rules! fn_cursor {
    () => {
        /// Returns a [`Cursor`] to this `SliceEntry`
        ///
        /// For now, this method is not available on COW-enabled trees, because retrieving the
        /// cursor is actually non-trivial. For those, you should use the [`RleTree::cursor_to`]
        /// method.
        ///
        /// [`RleTree::cursor_to`]: crate::RleTree::cursor_to
        pub fn cursor<C: Cursor>(&self) -> C {
            cursor_to_handle(self.slice)
        }
    };
}

impl<'t, I, S, const M: usize> SliceEntry<'t, I, S, NoFeatures, M> {
    fn_cursor!();
}

impl<'t, I, S, const M: usize> SliceEntry<'t, I, S, AllowSliceRefs, M> {
    /// Creates a new [`SliceRef`] pointing to this slice
    pub fn make_ref(&self) -> SliceRef<I, S, M> {
        // SAFETY: the `SliceHandle` here is being provided directly to the `SliceRefStore`.
        let handle = unsafe { self.slice.clone_slice_ref() };
        self.store.make_ref(handle)
    }

    fn_cursor!();
}
