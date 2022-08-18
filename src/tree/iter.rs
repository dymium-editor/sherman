//! Wrapper module for [`RleTree`](crate::RleTree) iterator types -- [`Iter`], [`Drain`], and
//! related types

use crate::param::{AllowSliceRefs, RleTreeConfig};
use crate::public_traits::{Index, Slice};
use crate::range::{EndBound, RangeBounds, StartBound};
use crate::{Cursor, SliceRef};
use std::ops::Range;

use super::node::{borrow, ty, NodeHandle, SliceHandle};
use super::{Root, DEFAULT_MIN_KEYS};

/// An iterator over a range of slices and their positions in an [`RleTree`]
///
/// This iterator is double-ended, and yields [`SliceEntry`]s. It *may* allocate if COW features
/// are enabled.
///
/// This type is produced by the [`iter`] or [`iter_with_cursor`] methods on [`RleTree`]. Please
/// refer to either of those methods for more information.
///
/// [`RleTree`]: crate::RleTree
/// [`iter`]: crate::RleTree::iter
/// [`iter_with_cursor`]: crate::RleTree::iter_with_cursor
pub struct Iter<'t, C, I, S, P, const M: usize = DEFAULT_MIN_KEYS>
where
    P: RleTreeConfig<I, S>,
{
    start: I,
    end: IncludedOrExcludedBound<I>,
    root: Option<&'t NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>>,
    initial_cursor: Option<C>,
}

enum IncludedOrExcludedBound<I> {
    Included(I),
    Excluded(I),
}

/// Information about a single slice in an [`RleTree`], yeilded by [`Iter`]
///
/// Conceptually, this type is not too different from `(Range<I>, &'t S)`, but the methods provide
/// some additional functionality that wouldn't be provided by a simpler type.
///
/// More information is available in the methods themselves.
///
/// [`RleTree`]: crate::RleTree
pub struct SliceEntry<'t, I, S, P, const M: usize = DEFAULT_MIN_KEYS>
where
    P: RleTreeConfig<I, S>,
{
    range: Range<I>,
    slice: &'t SliceHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
}

impl<'t, I, S, const M: usize> SliceEntry<'t, I, S, AllowSliceRefs, M> {
    /// Creates a new [`SliceRef`] pointing to this slice
    pub fn make_ref(&self) -> SliceRef<I, S, M> {
        todo!()
    }
}

impl<'t, C, I, S, P, const M: usize> Iter<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: 't + Slice<I>,
    P: RleTreeConfig<I, S>,
{
    /// Creates a new iterator
    pub(super) fn new(
        range: impl RangeBounds<I>,
        tree_size: I,
        cursor: C,
        root: Option<&'t NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>>,
    ) -> Self {
        let start = match range.start_bound() {
            StartBound::Included(i) => *i,
            StartBound::Unbounded => I::ZERO,
        };

        let end = match range.end_bound() {
            EndBound::Included(i) => IncludedOrExcludedBound::Included(*i),
            EndBound::Excluded(i) => IncludedOrExcludedBound::Excluded(*i),
            EndBound::Unbounded => IncludedOrExcludedBound::Excluded(tree_size),
        };

        Iter {
            start,
            end,
            root,
            initial_cursor: Some(cursor),
        }
    }
}

impl<'t, C, I, S, P, const M: usize> Iterator for Iter<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: 't + Slice<I>,
    P: RleTreeConfig<I, S>,
{
    type Item = SliceEntry<'t, I, S, P, M>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'t, C, I, S, P, const M: usize> DoubleEndedIterator for Iter<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: 't + Slice<I>,
    P: RleTreeConfig<I, S>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

/// A destructive iterator over a range in an [`RleTree`]
///
/// This iterator is double-ended, and yields `(Range<I>, S)` tuples. On drop, the iterator
/// finishes removing all remaining items in the range.
///
/// This type is produced by the [`drain`] or [`drain_with_cursor`] methods on [`RleTree`]. Please
/// refer to either of those methods for more information.
///
/// [`RleTree`]: crate::RleTree
/// [`drain`]: crate::RleTree::drain
/// [`drain_with_cursor`]: crate::RleTree::drain_with_cursor
pub struct Drain<'t, C, I, S, P, const M: usize = DEFAULT_MIN_KEYS>
where
    P: RleTreeConfig<I, S>,
{
    range: Range<I>,
    root: Option<Root<I, S, P, M>>,
    initial_cursor: Option<C>,
    // The entire tree gets stored here, so that the
    tree_ref: &'t mut Option<Root<I, S, P, M>>,
}

impl<'t, C, I, S, P, const M: usize> Drain<'t, C, I, S, P, M>
where
    P: RleTreeConfig<I, S>,
{
    pub(super) fn new(root: &'t mut Option<Root<I, S, P, M>>, range: Range<I>, cursor: C) -> Self {
        let (tree_ref, root) = {
            let r = root.take();
            (root, r)
        };

        Drain {
            range,
            root,
            initial_cursor: Some(cursor),
            tree_ref,
        }
    }
}

impl<'t, C, I, S, P, const M: usize> Iterator for Drain<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: 't + Slice<I>,
    P: RleTreeConfig<I, S>,
{
    type Item = (Range<I>, S);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'t, C, I, S, P, const M: usize> DoubleEndedIterator for Drain<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: 't + Slice<I>,
    P: RleTreeConfig<I, S>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
