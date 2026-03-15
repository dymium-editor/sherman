//! [`Iter`] and related types

use std::ops::{Bound, Range};

use crate::{Index, RleTree, Slice};

use super::SearchBound as EndBound;
use super::entry::SliceEntry;
use super::node;

/// An iterator over ranges of slices and their positions in an [`RleTree`]
pub struct Iter<'t, I, S> {
    start: I,
    end: EndBound<I>,
    root: Option<node::HandleImmut<'t, I, S>>,
    last_front: Option<(Range<I>, node::HandleImmut<'t, I, S>)>,
    last_back: Option<(Range<I>, node::HandleImmut<'t, I, S>)>,
}

impl<'t, I, S> Iter<'t, I, S>
where
    I: Index,
    S: Slice<I>,
{
    #[track_caller]
    pub(super) fn new(
        tree: &'t RleTree<I, S>,
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
            last_front: None,
            last_back: None,
        }
    }

    fn get_next_front(&self) -> Option<(Range<I>, node::HandleImmut<'t, I, S>)> {
        if let Some((range, node)) = self.last_front.as_ref() {
            return Self::step_next_front(range.clone(), node.reborrow());
        }

        let root = self.root.as_ref()?;
        // Special case: skip search if we have an empty range at the end of the tree
        if self.start == root.subtree_size() {
            return None;
        }

        let (node, range, offset_in_range) =
            super::search(root.reborrow(), EndBound::Included(self.start));

        let abs_start = self.start.sub_right(offset_in_range);
        let abs_end = abs_start.add_right(range.end.sub_left(range.start));
        Some((abs_start..abs_end, node))
    }

    fn get_next_back(&self) -> Option<(Range<I>, node::HandleImmut<'t, I, S>)> {
        if let Some((range, node)) = self.last_back.as_ref() {
            return Self::step_next_back(range.clone(), node.reborrow());
        }

        let root = self.root.as_ref()?;
        let (node, range, offset_in_range) = super::search(root.reborrow(), self.end);

        let self_end = match self.end {
            EndBound::Included(e) | EndBound::Excluded(e) => e,
        };
        let abs_start = self_end.sub_right(offset_in_range);
        let abs_end = abs_start.add_right(range.end.sub_left(range.start));
        Some((abs_start..abs_end, node))
    }

    fn step_next_front(
        range: Range<I>,
        mut node: node::HandleImmut<'t, I, S>,
    ) -> Option<(Range<I>, node::HandleImmut<'t, I, S>)> {
        let next_start_idx = range.end;

        // Traverse up the tree until the node has an rhs child
        loop {
            match node.reborrow().into_rhs() {
                Some(n) => {
                    node = n;
                    break;
                }
                None => (node, _) = node.into_parent()?,
            }
        }

        // From the next rhs node, find the left-most child node
        loop {
            match node.reborrow().into_lhs() {
                Some(n) => node = n,
                None => break,
            }
        }

        let local_range = node.value_range();
        let end_idx = next_start_idx.add_right(local_range.end.sub_left(local_range.start));

        Some((next_start_idx..end_idx, node))
    }

    fn step_next_back(
        range: Range<I>,
        mut node: node::HandleImmut<'t, I, S>,
    ) -> Option<(Range<I>, node::HandleImmut<'t, I, S>)> {
        let next_end_idx = range.start;

        // Traverse up the tree until the node has an lhs child
        loop {
            match node.reborrow().into_lhs() {
                Some(n) => {
                    node = n;
                    break;
                }
                None => (node, _) = node.into_parent()?,
            }
        }

        // From the next lhs node, find the right-most child node
        loop {
            match node.reborrow().into_rhs() {
                Some(n) => node = n,
                None => break,
            }
        }

        let local_range = node.value_range();
        let start_idx = next_end_idx.sub_right(local_range.end.sub_left(local_range.start));
        Some((start_idx..next_end_idx, node))
    }
}

impl<'t, I, S> Iterator for Iter<'t, I, S>
where
    I: Index,
    S: Slice<I>,
{
    type Item = SliceEntry<'t, I, S>;

    fn next(&mut self) -> Option<Self::Item> {
        let (range, next_front_node) = self.get_next_front()?;

        if let Some((_, last_back_node)) = self.last_back.as_ref() {
            // If there's already been calls to `next_back()`, check if we've caught up to the
            // backwards iteration.
            if last_back_node.addr() == next_front_node.addr() {
                return None;
            }
        } else {
            // There haven't been any calls to `next_back()`. Check if what we're going to return
            // is still within the end bound.
            match self.end {
                EndBound::Included(end) if end < range.start => return None,
                EndBound::Excluded(end) if end <= range.start => return None,
                _ => (),
            }
        }

        self.last_front = Some((range.clone(), next_front_node.reborrow()));
        Some(SliceEntry { range, slice: next_front_node })
    }
}

impl<'t, I, S> DoubleEndedIterator for Iter<'t, I, S>
where
    I: Index,
    S: Slice<I>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let (range, next_back_node) = self.get_next_back()?;

        if let Some((_, last_front_node)) = self.last_front.as_ref() {
            // If there's already been calls to `next()`, check if we've caught up to the forwards
            // iteration.
            if last_front_node.addr() == next_back_node.addr() {
                return None;
            }
        } else {
            // There haven't been any calls to `next()`. Check if what we're going to return is
            // still within the start bound.
            if range.end <= self.start {
                return None;
            }
        }

        self.last_back = Some((range.clone(), next_back_node.reborrow()));
        Some(SliceEntry { range, slice: next_back_node })
    }
}
