//! Wrapper module for [`RleTree`](crate::RleTree)'s destructive iterator -- [`Drain`] and related
//! types

use crate::param::RleTreeConfig;
use crate::{Cursor, Index, Slice};
use std::ops::Range;

use super::{Root, DEFAULT_MIN_KEYS};

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
    P: RleTreeConfig<I, S, M>,
{
    range: Range<I>,
    root: Option<Root<I, S, P, M>>,
    initial_cursor: Option<C>,
    // The entire tree gets stored here, so that the
    tree_ref: &'t mut Option<Root<I, S, P, M>>,
}

impl<'t, C, I, S, P, const M: usize> Drain<'t, C, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
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
    P: RleTreeConfig<I, S, M>,
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
    P: RleTreeConfig<I, S, M>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
