//! Wrapper module for [`RleTree`](crate::RleTree) iterator types -- [`Iter`], [`Drain`], and
//! related types

use crate::param::{AllowSliceRefs, RleTreeConfig};
use crate::public_traits::{Index, Slice};
use crate::range::{EndBound, RangeBounds, StartBound};
use crate::{Cursor, SliceRef};
use std::any::TypeId;
use std::fmt::Debug;
use std::ops::Range;

use super::node::{borrow, ty, NodeHandle, SliceHandle, Type};
use super::{search_step, ChildOrKey, Root, DEFAULT_MIN_KEYS};

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
    state: Option<IterState<'t, C, I, S, P, M>>,
}

struct IterState<'t, C, I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S>,
{
    store: &'t P::SliceRefStore,
    root: NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
    cursor: Option<C>,
    fwd: Option<IterStack<'t, I, S, P, M>>,
    bkwd: Option<IterStack<'t, I, S, P, M>>,
}

struct IterStack<'t, I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S>,
{
    /// Stack of all the nodes leading to `head` where the child containing `head` has a different
    /// node as its parent
    ///
    /// This can only occur when COW is enabled, and so control flow around `stack` typically
    /// checks `P::COW` first.
    stack: Vec<(NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>, u8)>,
    head: SliceHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
    /// The absolute end position of `head`
    end_pos: I,
}

#[derive(Debug, Copy, Clone)]
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
    slice: SliceHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
    store: &'t P::SliceRefStore,
}

#[track_caller]
fn panic_internal_error_or_bad_index<I: Index>() -> ! {
    if crate::public_traits::perfect_index_impls().contains(&TypeId::of::<I>()) {
        panic!("internal error")
    } else {
        panic!("internal error or bad `Index` implementation")
    }
}

impl<'t, I, S, P, const M: usize> SliceEntry<'t, I, S, P, M>
where
    P: RleTreeConfig<I, S>,
    I: Index,
{
    /// Returns the range of values covered by this entry
    pub fn range(&self) -> Range<I> {
        self.range.clone()
    }

    /// Returns the size of the range of vales covered by this entry
    ///
    /// This is essentially a convenience method roughly equivalent to `self.range().len()`.
    pub fn size(&self) -> I {
        self.range.end.sub_right(self.range.start)
    }

    /// Returns a reference to the slice for this entry
    pub fn slice(&self) -> &'t S {
        self.slice.into_ref()
    }
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
    ///
    /// This function panics if the arguments are invalid, as described in [`RleTree::iter`].
    #[track_caller]
    pub(super) fn new(
        range: impl Debug + RangeBounds<I>,
        tree_size: I,
        cursor: C,
        root: Option<(
            NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
            &'t P::SliceRefStore,
        )>,
    ) -> Self {
        let start_bound = range.start_bound().cloned();
        let end_bound = range.end_bound().cloned();

        let start = match start_bound {
            StartBound::Included(i) => i,
            StartBound::Unbounded => I::ZERO,
        };

        let (end, strict_end) = match end_bound {
            EndBound::Included(i) => (IncludedOrExcludedBound::Included(i), EndBound::Included(i)),
            EndBound::Excluded(i) => (IncludedOrExcludedBound::Excluded(i), EndBound::Excluded(i)),
            EndBound::Unbounded => (
                IncludedOrExcludedBound::Excluded(tree_size),
                EndBound::Excluded(tree_size),
            ),
        };

        if range.starts_after_end() {
            panic!("invalid range `{range:?}`");
        } else if start < I::ZERO {
            panic!("range `{range:?}` out of bounds below zero");
        } else if matches!(start_bound, StartBound::Included(i) if i >= tree_size)
            || (StartBound::Unbounded, strict_end).overlaps_naive(tree_size..)
        {
            panic!("range `{range:?}` out of bounds for tree size {tree_size:?}");
        }

        Iter {
            start,
            end,
            state: root.map(|(root, store)| IterState {
                store,
                root,
                cursor: Some(cursor),
                fwd: None,
                bkwd: None,
            }),
        }
    }

    fn make_stack(
        root: NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
        cursor: Option<C>,
        target: IncludedOrExcludedBound<I>,
    ) -> IterStack<'t, I, S, P, M> {
        let mut stack = Vec::new();
        let mut cursor = cursor.map(|c| c.into_path());
        let mut head_node = root;
        let mut target = match target {
            IncludedOrExcludedBound::Included(i) => i,
            IncludedOrExcludedBound::Excluded(i) => i,
        };

        // Search downward with the cursor
        loop {
            let hint = cursor.as_mut().and_then(|c| c.next());

            let result = if target != head_node.leaf().subtree_size() {
                search_step(head_node, hint, target)
            } else {
                let len = head_node.leaf().len();

                // Use the key or child immediately to the left.
                match head_node.into_typed() {
                    Type::Leaf(_) => {
                        let i = len - 1;
                        // SAFETY: `len` is guaranteed to be `>= 1`, so `len - 1` can't overflow,
                        // and key indexes must be `< len`.
                        let p = unsafe { head_node.leaf().key_pos(i) };
                        ChildOrKey::Key((i, p))
                    }
                    Type::Internal(node) => {
                        // The child immediately left of the key has the same index
                        let c_idx = len;
                        // SAFETY: the index `len` is always a valid child index
                        let c_pos = unsafe { node.child_pos(c_idx) };
                        ChildOrKey::Child((c_idx, c_pos))
                    }
                }
            };

            match result {
                ChildOrKey::Key((k_idx, k_pos)) => {
                    let slice_handle = unsafe { head_node.into_slice_handle(k_idx) };
                    let slice_end = k_pos.add_right(slice_handle.slice_size());
                    return IterStack {
                        stack,
                        end_pos: slice_end,
                        head: slice_handle,
                    };
                }
                ChildOrKey::Child((c_idx, c_pos)) => {
                    let node = match head_node.into_typed() {
                        // Error resulting from our failure or a bad `Index` impl
                        Type::Leaf(_) => panic_internal_error_or_bad_index::<I>(),
                        Type::Internal(n) => n,
                    };

                    // SAFETY: `search_step` and our processing above guarantee that `c_idx` is a
                    // valid child index.
                    let child = unsafe { node.into_child(c_idx) };
                    // If we have COW enabled, and the child's parent is not this node, we need to
                    // add `head_node` to the stack.
                    if P::COW {
                        match child.leaf().parent().map(|p| p.ptr) {
                            // Different child pointer:
                            Some(p) if p != node.ptr() => stack.push((head_node, c_idx)),
                            _ => (),
                        }
                    }

                    target = target.sub_left(c_pos);
                    head_node = child;
                }
            }
        }
    }
}

impl<'t, I, S, P, const M: usize> IterStack<'t, I, S, P, M>
where
    I: Index,
    P: RleTreeConfig<I, S>,
{
    fn fwd_step(&mut self) {
        match self.head.node.into_typed() {
            // For an internal node, we recurse down to the next leftmost key
            Type::Internal(node) => {
                let mut parent = node;
                let mut c_idx = self.head.idx + 1;
                loop {
                    // SAFETY: `c_idx` is either 0 or equal to `self.head.idx + 1`, only on the
                    // first iteration. All internal nodes have a first child, and `self.head.idx`
                    // is a valid key index, so +1 is a valid child index.
                    let child = unsafe { parent.into_child(c_idx) };
                    if P::COW {
                        match child.leaf().parent().map(|p| p.ptr) {
                            Some(p) if p != node.ptr() => {
                                self.stack.push((parent.erase_type(), c_idx))
                            }
                            _ => (),
                        }
                    }

                    match child.into_typed() {
                        Type::Leaf(_) => {
                            assert!(child.leaf().len() >= 1);

                            // SAFETY: the assertion above guarantees that 0 is a valid key index
                            self.head = unsafe { child.into_slice_handle(0) };
                            self.end_pos = self.end_pos.add_right(self.head.slice_size());
                            return;
                        }
                        Type::Internal(p) => {
                            parent = p;
                            c_idx = 0;
                            continue;
                        }
                    }
                }
            }
            // For a leaf node where there's more keys, we can just progress to the next key
            Type::Leaf(node) if self.head.idx + 1 < node.leaf().len() => {
                // SAFETY: the condition above guarantees that `self.head.idx` is a valid key index
                self.head = unsafe { node.into_slice_handle(self.head.idx + 1).erase_type() };
                self.end_pos = self.end_pos.add_right(self.head.slice_size());
            }
            // But if we're at the end of this leaf node, we have to go upwards until we find the
            // next key.
            Type::Leaf(node) => {
                let mut child = node.erase_type();

                loop {
                    // If there's an entry in the stack for this child, use that instead of the
                    // child's stored parent
                    let parent_override = match P::COW {
                        false => None,
                        true => match self.stack.last().copied() {
                            Some((h, c_idx)) if h.height() == child.height() + 1 => {
                                self.stack.pop();
                                Some((h, c_idx))
                            }
                            _ => None,
                        },
                    };

                    let (parent, child_idx) = match parent_override {
                        Some(tuple) => tuple,
                        None => match child.into_parent() {
                            Ok((p, c_idx)) => (p.erase_type(), c_idx),
                            // If this node has no parent, then we're at the end of the root node,
                            // so no more iteration possible. This should have already been caught.
                            Err(_) => panic_internal_error_or_bad_index::<I>(),
                        },
                    };

                    // The key immediately after a child has the same index. If there *is* a key
                    // after this child, then we should just return that
                    if child_idx < parent.leaf().len() {
                        // SAFETY: we just checked that `child_idx` is a valid key index
                        self.head = unsafe { parent.into_slice_handle(child_idx).erase_type() };
                        self.end_pos = self.end_pos.add_right(self.head.slice_size());
                        return;
                    }

                    // Otherwise, we continue going upwards:
                    child = parent.erase_type();
                }
            }
        }
    }

    fn bkwd_step(&mut self) {
        match self.head.node.into_typed() {
            // For an internal node, recurse down to the rightmost key to the left of `self.head`
            Type::Internal(node) => {
                let mut parent = node;
                let mut c_idx = self.head.idx + 1;
                loop {
                    // SAFETY: `c_idx` is either set to `parent.leaf().len()` (which is a valid
                    // child index) or, on the first iteration only, `self.head.idx + 1`.
                    // `self.head.idx` is a valid key for `node`, so `self.head.idx + 1` is a valid
                    // child index.
                    let child = unsafe { parent.into_child(c_idx) };
                    if P::COW {
                        match child.leaf().parent().map(|p| p.ptr) {
                            Some(p) if p != node.ptr() => {
                                self.stack.push((parent.erase_type(), c_idx));
                            }
                            _ => (),
                        }
                    }

                    match child.into_typed() {
                        Type::Leaf(_) => {
                            assert!(child.leaf().len() != 0);

                            let k_idx = child.leaf().len() - 1;
                            self.end_pos = self.end_pos.sub_right(self.head.slice_size());
                            // SAFETY: the assertion above guarantees that `leaf.len() - 1` won't
                            // underflow, and produces a valid key index.
                            self.head = unsafe { child.into_slice_handle(k_idx) };
                            return;
                        }
                        Type::Internal(p) => {
                            parent = p;
                            c_idx = p.leaf().len();
                        }
                    }
                }
            }
            // For a leaf node where there's more keys, progress to the key immediately to the left
            Type::Leaf(node) if self.head.idx != 0 => {
                self.end_pos = self.end_pos.sub_right(self.head.slice_size());
                // SAFETY: because `self.head.idx` is a valid key index, so is `self.head.idx - 1`
                self.head = unsafe { node.into_slice_handle(self.head.idx - 1).erase_type() };
            }
            // If we're at the start of this leaf node, go upwards untli we find a node with a key
            // to the left of this child node
            Type::Leaf(node) => {
                let mut child = node.erase_type();

                loop {
                    // If there's an entry in the stack for this child, use that instead of the
                    // child's stored parent
                    let parent_override = match P::COW {
                        false => None,
                        true => match self.stack.last().copied() {
                            Some((h, c_idx)) if h.height() == child.height() + 1 => {
                                self.stack.pop();
                                Some((h, c_idx))
                            }
                            _ => None,
                        },
                    };

                    let (parent, child_idx) = match parent_override {
                        Some(tuple) => tuple,
                        None => match child.into_parent() {
                            Ok((p, c_idx)) => (p.erase_type(), c_idx),
                            // If this node has no parent, then we're at the end of the root node,
                            // so no more iteration possible. This should have already been caught.
                            Err(_) => panic_internal_error_or_bad_index::<I>(),
                        },
                    };

                    // The key immediately before a child is at index - 1. If there is a key before
                    // this child, we should just return that
                    if let Some(k_idx) = child_idx.checked_sub(1) {
                        self.end_pos = self.end_pos.sub_right(self.head.slice_size());
                        // SAFETY: `child_idx` is guaranteed to be a valid child index for
                        // `parent`, so `child_idx - 1` is a valid key index.
                        self.head = unsafe { parent.into_slice_handle(k_idx).erase_type() };
                        return;
                    }

                    // Otherwise, continue going upwards:
                    child = parent.erase_type();
                }
            }
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
        let state = self.state.as_mut()?;

        let fwd = match state.fwd.as_mut() {
            Some(s) => s,
            None => {
                let s = Self::make_stack(
                    state.root,
                    state.cursor.take(),
                    IncludedOrExcludedBound::Included(self.start),
                );

                let size = s.head.slice_size();
                let start = s.end_pos.sub_right(size);

                let entry = SliceEntry {
                    range: start..s.end_pos,
                    slice: s.head,
                    store: state.store,
                };

                state.fwd = Some(s);
                return Some(entry);
            }
        };

        // Update `fwd` to the next entry:
        let next_start = fwd.end_pos;
        // ... but first, check that it'll be within bounds:
        match self.end {
            IncludedOrExcludedBound::Included(i) if i < next_start => {
                self.state = None;
                return None;
            }
            IncludedOrExcludedBound::Excluded(i) if i <= next_start => {
                self.state = None;
                return None;
            }
            _ => (),
        }

        fwd.fwd_step();

        // If the current head is already the head of the backward stack, then it's already been
        // returned, which would mean that the iterator has yielded all of its items.
        let is_done = matches!(state.bkwd.as_ref(), Some(s) if s.head == fwd.head);

        if is_done {
            self.state = None;
            None
        } else {
            Some(SliceEntry {
                range: next_start..fwd.end_pos,
                slice: fwd.head,
                store: state.store,
            })
        }
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
        let state = self.state.as_mut()?;

        let bkwd = match state.bkwd.as_mut() {
            Some(s) => s,
            None => {
                let s = Self::make_stack(state.root, state.cursor.take(), self.end);

                let size = s.head.slice_size();
                let start = s.end_pos.sub_right(size);

                let entry = SliceEntry {
                    range: start..s.end_pos,
                    slice: s.head,
                    store: state.store,
                };

                state.bkwd = Some(s);
                return Some(entry);
            }
        };

        let next_end = bkwd.end_pos.sub_right(bkwd.head.slice_size());
        if next_end <= self.start {
            self.state = None;
            return None;
        }

        bkwd.bkwd_step();

        let is_done = matches!(state.fwd.as_ref(), Some(s) if s.head == bkwd.head);
        if is_done {
            self.state = None;
            None
        } else {
            let start = bkwd.end_pos.sub_right(bkwd.head.slice_size());

            Some(SliceEntry {
                range: start..bkwd.end_pos,
                slice: bkwd.head,
                store: state.store,
            })
        }
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
