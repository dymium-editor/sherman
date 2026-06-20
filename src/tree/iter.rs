//! [`Iter`], [`IntoIter`], and related types

#[cfg(test)]
use std::fmt::{self, Debug};
use std::mem;
use std::ops::{Bound, Range};

use crate::param::{self, RleTreeConfig, SupportsUpdate};
use crate::{Index, RleTree, Slice};

use super::drain::Drain;
use super::entry::SliceEntry;
use super::replace::Removed;
use super::{SearchBound as EndBound, Side, node};

/// Borrowing iterator over a range of values in an [`RleTree`], returned by [`RleTree::iter`]
///
/// See [`RleTree::iter`] for more information, or [`SliceEntry`] for the values returned.
pub struct Iter<'t, I: 't, S: 't, P: RleTreeConfig<I, S> = param::NoFeatures> {
    start: I,
    end: EndBound<I>,
    root: Option<node::HandleImmut<'t, I, S, P>>,
    last_front: Edge<'t, I, S, P>,
    last_back: Edge<'t, I, S, P>,
}

enum Edge<'t, I: 't, S: 't, P: RleTreeConfig<I, S>> {
    Ongoing(Range<I>, node::StackHandleImmut<'t, I, S, P>),
    // Note: Ideally we would either have:
    //
    // * `Option<Edge>` with `Edge::Ongoing` and `Edge::Done`; or
    // * `Edge` with `Edge::NotStarted`, `Edge::Ongoing`, and `Edge::Done`
    //
    // But! Generally `StackHandleImmut` contains a `NonNull` transitively in one of its fields,
    // and we can do the same scalar optimization as `Option<NonNull>` only if there's a *single*
    // enum with two variants (rather than nesting with `Option` using three variants on `Edge`)
    //
    // So - it's easy enough to make the happy path (existing Ongoing stack) just require a single
    // "is the pointer/discriminant zero?" comparison, so let's do that.
    Empty { finished: bool },
}

#[cfg(test)]
impl<'t, I, S, P> Debug for Edge<'t, I, S, P>
where
    I: 't + Debug,
    S: 't + Debug,
    P: RleTreeConfig<I, S>,
    <P as RleTreeConfig<I, S>>::BorrowStack<'t>: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Edge::Ongoing(range, stack) => {
                f.debug_tuple("Ongoing").field(range).field(stack).finish()
            }
            Edge::Empty { finished } => {
                f.debug_struct("Empty").field("finished", finished).finish()
            }
        }
    }
}

impl<'t, I: 't, S: 't, P: RleTreeConfig<I, S>> Edge<'t, I, S, P> {
    fn take(&mut self) -> Edge<'t, I, S, P> {
        mem::replace(self, Edge::Empty { finished: true })
    }
}

impl<'t, I, S, P> Iter<'t, I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    #[track_caller]
    pub(super) fn new(
        tree: &'t RleTree<I, S, P>,
        start_bound: Bound<&I>,
        end_bound: Bound<&I>,
    ) -> Self {
        let start = match start_bound {
            Bound::Excluded(_) => panic!("cannot create iterator with exclusive start bound"),

            Bound::Included(idx) if *idx < I::ZERO => {
                panic!("start bound {idx:?} out of bounds, less than zero");
            }

            Bound::Unbounded => I::ZERO,
            Bound::Included(&idx) => idx,
        };

        let size = tree.size();
        let end = match end_bound {
            Bound::Excluded(idx) if *idx > size => {
                panic!("exclusive end bound {idx:?} out of bounds, greater than size {size:?}")
            }
            Bound::Included(idx) if *idx >= size => panic!(
                "inclusive end bound {idx:?} out of bounds, greater than or equal to size {size:?}"
            ),

            Bound::Unbounded => EndBound::Excluded(size),
            Bound::Included(&idx) => EndBound::Included(idx),
            Bound::Excluded(&idx) => EndBound::Excluded(idx),
        };

        // Check for invalid ranges:
        match end {
            EndBound::Included(e) | EndBound::Excluded(e) if e < start => {
                panic!("bad range: end bound {e:?} less than start {start:?}")
            }
            _ => (),
        }

        Iter {
            start,
            end,
            root: tree.root().map(|r| r.handle.borrow()),
            last_front: Edge::Empty { finished: false },
            last_back: Edge::Empty { finished: false },
        }
    }

    fn search(
        stack: node::StackHandleImmut<'t, I, S, P>,
        target: EndBound<I>,
    ) -> (node::StackHandleImmut<'t, I, S, P>, Range<I>, I) {
        super::search(stack, target, |s| s.reborrow(), |s, n, side| s.into_child(n, side))
    }

    fn get_next_front(&mut self) -> Option<(Range<I>, node::StackHandleImmut<'t, I, S, P>)> {
        match self.last_front.take() {
            Edge::Ongoing(range, stack) => return Self::step_next_front(range, stack),
            Edge::Empty { finished: true } => return None,
            Edge::Empty { finished: false } => (),
        }

        let root = self.root.as_ref()?;
        // Special case: skip search if we have an empty range at the end of the tree
        if self.start == root.subtree_size() {
            return None;
        }

        let stack = node::StackHandleImmut::new_root(root.reborrow());
        let (stack, range, offset_in_range) = Self::search(stack, EndBound::Included(self.start));

        let abs_start = self.start.sub_right(offset_in_range);
        let abs_end = abs_start.add_right(range.end.sub_left(range.start));
        Some((abs_start..abs_end, stack))
    }

    fn get_next_back(&mut self) -> Option<(Range<I>, node::StackHandleImmut<'t, I, S, P>)> {
        match self.last_back.take() {
            Edge::Ongoing(range, stack) => return Self::step_next_back(range, stack),
            Edge::Empty { finished: true } => return None,
            Edge::Empty { finished: false } => (),
        }

        let root = self.root.as_ref()?;
        // Special case: skip search if we have an empty range at the start of the tree
        if matches!(self.end, EndBound::Excluded(i) if i == I::ZERO) {
            return None;
        }

        let stack = node::StackHandleImmut::new_root(root.reborrow());
        let (stack, range, offset_in_range) = Self::search(stack, self.end);

        let self_end = match self.end {
            EndBound::Included(e) | EndBound::Excluded(e) => e,
        };
        let abs_start = self_end.sub_right(offset_in_range);
        let abs_end = abs_start.add_right(range.end.sub_left(range.start));
        Some((abs_start..abs_end, stack))
    }

    fn step_next_front(
        range: Range<I>,
        mut stack: node::StackHandleImmut<'t, I, S, P>,
    ) -> Option<(Range<I>, node::StackHandleImmut<'t, I, S, P>)> {
        let next_start_idx = range.end;

        if let Some(rhs) = stack.reborrow().into_rhs() {
            // If this node has a right-hand child, pursue that route, traversing into the
            // left-most transitive child of THIS node's right-hand child:
            stack = stack.into_child(rhs, Side::Rhs);
            while let Some(lhs) = stack.reborrow().into_lhs() {
                stack = stack.into_child(lhs, Side::Lhs);
            }
        } else {
            // ... otherwise (no right-hand child), we should traverse back up the tree until we
            // find a parent that's to the right of THIS node (i.e. this node is lhs).
            loop {
                let (n, side) = stack.into_parent()?;
                stack = n;
                if side == Side::Lhs {
                    break;
                }
            }
        }

        let local_range = stack.reborrow().value_range();
        let end_idx = next_start_idx.add_right(local_range.end.sub_left(local_range.start));

        Some((next_start_idx..end_idx, stack))
    }

    fn step_next_back(
        range: Range<I>,
        mut stack: node::StackHandleImmut<'t, I, S, P>,
    ) -> Option<(Range<I>, node::StackHandleImmut<'t, I, S, P>)> {
        let next_end_idx = range.start;

        if let Some(lhs) = stack.reborrow().into_lhs() {
            // If this node has a left-hand child, pursue that route, traversing into the
            // right-most transitive child of THIS node's left-hand child:
            stack = stack.into_child(lhs, Side::Lhs);
            while let Some(rhs) = stack.reborrow().into_rhs() {
                stack = stack.into_child(rhs, Side::Rhs);
            }
        } else {
            // ... otherwise (no left-hand child), we should traverse back up the tree until we
            // find a parent that's to the left of THIS node (i.e. this node is rhs)
            loop {
                let (n, side) = stack.into_parent()?;
                stack = n;
                if side == Side::Rhs {
                    break;
                }
            }
        }

        let local_range = stack.reborrow().value_range();
        let start_idx = next_end_idx.sub_right(local_range.end.sub_left(local_range.start));
        Some((start_idx..next_end_idx, stack))
    }
}

impl<'t, I, S, P> Iterator for Iter<'t, I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    type Item = SliceEntry<'t, I, S, P>;

    fn next(&mut self) -> Option<Self::Item> {
        let (range, next_front_stack) = self.get_next_front()?;

        match &self.last_back {
            // If there's already been calls to `next_back()`, check if we've caught up to the
            // backwards iteration.
            Edge::Ongoing(_, last_back_stack) => {
                if last_back_stack.reborrow().addr() == next_front_stack.reborrow().addr() {
                    return None;
                }
            }
            // `next_back()` actually already finished; we can't return anything.
            Edge::Empty { finished: true } => return None,
            // There haven't been any calls to `next_back()`. Check if what we're going to return
            // is still within the end bound.
            Edge::Empty { finished: false } => match self.end {
                EndBound::Included(end) if end < range.start => return None,
                EndBound::Excluded(end) if end <= range.start => return None,
                _ => (),
            },
        }

        let slice = next_front_stack.reborrow();
        self.last_front = Edge::Ongoing(range.clone(), next_front_stack);
        Some(SliceEntry { range, slice })
    }
}

impl<'t, I, S, P> DoubleEndedIterator for Iter<'t, I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let (range, next_back_stack) = self.get_next_back()?;

        match &self.last_front {
            // If there's already been calls to `next()`, check if we've caught up to the forwards
            // iteration.
            Edge::Ongoing(_, last_front_stack) => {
                if last_front_stack.reborrow().addr() == next_back_stack.reborrow().addr() {
                    return None;
                }
            }
            // `next()` actually already finished; we can't return anything.
            Edge::Empty { finished: true } => return None,
            // There haven't been any calls to `next()`. Check if what we're going to return is
            // still within the start bound.
            Edge::Empty { finished: false } => {
                if range.end <= self.start {
                    return None;
                }
            }
        }

        let slice = next_back_stack.reborrow();
        self.last_back = Edge::Ongoing(range.clone(), next_back_stack);
        Some(SliceEntry { range, slice })
    }
}

/// A destructive iterator over the entirety of an [`RleTree`], returned by [`RleTree::into_iter`]
///
/// [`RleTree::into_iter`]: IntoIterator
pub struct IntoIter<I, S, P: RleTreeConfig<I, S>> {
    // Internally, just use the `Drain` interface - it's slightly more generic.
    inner: Drain<I, S, P>,
}

impl<I, S, P> IntoIterator for RleTree<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    type Item = (Range<I>, S);
    type IntoIter = IntoIter<I, S, P>;

    fn into_iter(self) -> Self::IntoIter {
        let removed = Removed::from_tree(self);
        let drain = Drain::new(removed);
        IntoIter { inner: drain }
    }
}

impl<I, S, P> Iterator for IntoIter<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    type Item = (Range<I>, S);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<I, S, P> DoubleEndedIterator for IntoIter<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}
