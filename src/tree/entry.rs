use std::ops::Range;

use super::{borrow, node};
use crate::{Index, Slice};

/// Information about a single slice in an [`RleTree`], returned by [`RleTree::get`] or yielded by
/// [`RleTree::iter`].
///
/// Conceptually, this type is basically just `(Range<I>, &'t S)`.
///
/// [`RleTree`]: crate::RleTree
/// [`RleTree::iter`]: crate::RleTree::iter
/// [`RleTree::get`]: crate::RleTree::get
pub struct SliceEntry<'t, I, S> {
    pub(super) range: Range<I>,
    pub(super) slice: node::NodeHandle<borrow::Immut<'t, node::Node<I, S>>>,
}

impl<'t, I, S> SliceEntry<'t, I, S>
where
    I: Index,
    S: Slice<I>,
{
    /// Returns the range of values covered by this entry
    pub fn range(&self) -> Range<I> {
        self.range.start..self.range.end
    }

    /// Returns the length of the range covered by this entry
    ///
    /// This is roughly equivalent to `self.range().len()`, but respects the potentially
    /// directional aspect of [`Index`] arithmetic.
    pub fn size(&self) -> I {
        self.range.end.sub_right(self.range.start)
    }

    /// Returns a reference to the slice for this entry
    pub fn slice(&self) -> &'t S {
        self.slice.value()
    }
}
