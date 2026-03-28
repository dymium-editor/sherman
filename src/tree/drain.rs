//! [`Drain`] and related types

use std::ops::Range;

use crate::{Index, Slice};

use super::remove::Removed;
use super::{Side, node};

/// A destructive iterator over a range of values in an [`RleTree`], returned by [`RleTree::drain`]
///
/// See [`RleTree::drain`] for more information.
///
/// [`RleTree`]: crate::RleTree
/// [`RleTree::drain`]: crate::RleTree::drain
pub struct Drain<I, S> {
    state: Option<DrainState<I, S>>,
}

enum DrainState<I, S> {
    Single(Option<(Range<I>, S)>),
    Tree(DrainTreeState<I, S>),
}

struct DrainTreeState<I, S> {
    // SAFETY INVARIANT: Must not be accessed. The tree is effectively borrowed by the edge ptrs.
    _root: node::HandleUniqueOwned<I, S>,

    // SAFETY INVARIANT: `front_edge` and `back_edge` will EXCLUSIVELY be used for traversing the
    // tree and removing nodes' values. In all other respects, the pointers are treated as
    // borrowing the tree present at `root`.
    front_edge: (Range<I>, node::Pointer<I, S>),
    back_edge: (Range<I>, node::Pointer<I, S>),

    closed: bool,
}

// SAFETY: This is the same condition as `RleTree`; see there for more.
unsafe impl<I: Send, S: Send> Send for Drain<I, S> {}
// SAFETY: This is the same condition as `RleTree`; see there for more.
unsafe impl<I: Sync, S: Sync> Sync for Drain<I, S> {}

impl<I, S> Drain<I, S>
where
    I: Index,
    S: Slice<I>,
{
    pub(super) fn new(start: I, end: I, removed: Option<Removed<I, S>>) -> Self {
        Drain {
            state: match removed {
                None => None,
                Some(Removed::Slice(value)) => Some(DrainState::Single(Some((start..end, value)))),
                Some(Removed::Tree(mut root)) => {
                    let mut leftmost_node = root.borrow_mut();
                    loop {
                        match leftmost_node.into_lhs() {
                            Ok(n) => leftmost_node = n,
                            Err(n) => {
                                leftmost_node = n;
                                break;
                            }
                        }
                    }
                    let leftmost_ptr = leftmost_node.ptr();
                    let leftmost_range = {
                        let r = leftmost_node.value_range();
                        let size = r.end.sub_left(r.start);
                        start..start.add_right(size)
                    };
                    drop(leftmost_node);

                    let mut rightmost_node = root.borrow_mut();
                    loop {
                        match rightmost_node.into_rhs() {
                            Ok(n) => rightmost_node = n,
                            Err(n) => {
                                rightmost_node = n;
                                break;
                            }
                        }
                    }
                    let rightmost_ptr = rightmost_node.ptr();
                    let rightmost_range = {
                        let r = rightmost_node.value_range();
                        let size = r.end.sub_left(r.start);
                        end.sub_left(size)..end
                    };
                    drop(rightmost_node);

                    Some(DrainState::Tree(DrainTreeState {
                        _root: root,
                        front_edge: (leftmost_range, leftmost_ptr),
                        back_edge: (rightmost_range, rightmost_ptr),
                        closed: false,
                    }))
                }
            },
        }
    }
}

impl<I, S> DrainTreeState<I, S>
where
    I: Index,
    S: Slice<I>,
{
    fn next(&mut self) -> Option<(Range<I>, S)> {
        if self.closed {
            return None;
        }

        let (front_range, front_ptr) = self.front_edge.clone();
        let (_, back_ptr) = self.back_edge;

        // Fetch the value at the edge, and then move it along.
        //
        // SAFETY: Invariants on the `DrainTreeState` guarantee that `front_ptr` is valid and that
        // our usage does not violate aliasing requirements.
        let mut node = unsafe { node::HandleMut::from_ptr(front_ptr) };
        let value = node.take_value();

        // We can progress the iteration as long as the two edges haven't already met.
        // If they have, we've actually exhausted everything.
        if front_ptr == back_ptr {
            self.closed = true;
        } else {
            let (next_range, next_node) = Self::step_edge_forward(front_range.clone(), node);
            self.front_edge = (next_range, next_node.ptr());
        }

        Some((front_range, value))
    }

    fn next_back(&mut self) -> Option<(Range<I>, S)> {
        if self.closed {
            return None;
        }

        let (_, front_ptr) = self.front_edge;
        let (back_range, back_ptr) = self.back_edge.clone();

        // Fetch the value at the edge, and then move it along.
        //
        // SAFETY: Invariants on the `DrainTreeState` guarantee that `back_ptr` is valid and that
        // our usage does not violate aliasing requirements.
        let mut node = unsafe { node::HandleMut::from_ptr(back_ptr) };
        let value = node.take_value();

        // We can progress the iteration as long as the two edges haven't already met.
        // If they have, we've actually exhausted everything.
        if front_ptr == back_ptr {
            self.closed = true;
        } else {
            let (next_range, next_node) = Self::step_edge_backward(back_range.clone(), node);
            self.back_edge = (next_range, next_node.ptr());
        }

        Some((back_range, value))
    }

    fn step_edge_forward(
        range: Range<I>,
        node: node::HandleMut<'_, I, S>,
    ) -> (Range<I>, node::HandleMut<'_, I, S>) {
        let next_node = match node.into_rhs() {
            // If the node has a right-hand child, we can step into that, and then find *that*
            // node's leftmost transitive child.
            Ok(rhs) => {
                let mut leftmost_rhs = rhs;
                loop {
                    match leftmost_rhs.into_lhs() {
                        Ok(n) => leftmost_rhs = n,
                        Err(n) => break n,
                    }
                }
            }
            // Otherwise, we'll have to traverse up the tree until we find a parent to the right of
            // this node (i.e. where this node is the left-hand child).
            Err(n) => {
                let mut parent = n;
                loop {
                    match parent.into_parent() {
                        Some((p, Side::Lhs)) => break p,
                        Some((p, Side::Rhs)) => parent = p,
                        None => panic!(
                            "internal error: tried to `step_edge_forward` without any nodes to the right"
                        ),
                    }
                }
            }
        };

        let r = next_node.value_range();
        let start = range.end;
        let end = start.add_right(r.end.sub_left(r.start));

        (start..end, next_node)
    }

    fn step_edge_backward(
        range: Range<I>,
        node: node::HandleMut<'_, I, S>,
    ) -> (Range<I>, node::HandleMut<'_, I, S>) {
        let next_node = match node.into_lhs() {
            // If this node has a left-hand child, we can step into that, and then find *that*
            // node's rightmost transitive child.
            Ok(lhs) => {
                let mut rightmost_lhs = lhs;
                loop {
                    match rightmost_lhs.into_rhs() {
                        Ok(n) => rightmost_lhs = n,
                        Err(n) => break n,
                    }
                }
            }
            // Otherwise, we'll have to traverse up the tree until we find a parent to the left of
            // this node (i.e. where this node is the right-hand child)
            Err(n) => {
                let mut parent = n;
                loop {
                    match parent.into_parent() {
                        Some((p, Side::Lhs)) => parent = p,
                        Some((p, Side::Rhs)) => break p,
                        None => panic!(
                            "internal error: tried to `step_edge_backward` without any nodes to the left"
                        ),
                    }
                }
            }
        };

        let r = next_node.value_range();
        let end = range.start;
        let start = end.sub_right(r.end.sub_left(r.start));

        (start..end, next_node)
    }
}

impl<I, S> Iterator for Drain<I, S>
where
    I: Index,
    S: Slice<I>,
{
    type Item = (Range<I>, S);

    fn next(&mut self) -> Option<Self::Item> {
        match self.state.as_mut()? {
            DrainState::Single(pair) => pair.take(),
            DrainState::Tree(t) => t.next(),
        }
    }
}

impl<I, S> DoubleEndedIterator for Drain<I, S>
where
    I: Index,
    S: Slice<I>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.state.as_mut()? {
            DrainState::Single(pair) => pair.take(),
            DrainState::Tree(t) => t.next_back(),
        }
    }
}
