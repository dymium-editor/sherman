//! Wrapper module for [`RleTree`](crate::RleTree)'s borrowing iterator -- [`Iter`] and related
//! types

use crate::param::RleTreeConfig;
use crate::public_traits::{Index, Slice};
use crate::range::{EndBound, RangeBounds, StartBound};
use crate::Cursor;
use std::fmt::Debug;
use std::panic::{RefUnwindSafe, UnwindSafe};

#[cfg(test)]
use crate::MaybeDebug;
#[cfg(test)]
use std::fmt::{self, Formatter};

use super::node::{borrow, ty, NodeHandle, SliceHandle, Type};
use super::{search_step, ChildOrKey, SliceEntry, DEFAULT_MIN_KEYS};

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
    P: RleTreeConfig<I, S, M>,
{
    start: I,
    end: IncludedOrExcludedBound<I>,
    state: Option<IterState<'t, C, I, S, P, M>>,
}

impl<'t, C: UnwindSafe, I: UnwindSafe + RefUnwindSafe, S: RefUnwindSafe, P, const M: usize>
    UnwindSafe for Iter<'t, C, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
}

// Bounds for `RefUnwindSafe` are much less strict, and represent only the things that we might
// *theoretically* provide access to. I.e. not `S`, but perhaps `C` or `I` (from the bounds)
impl<'t, C: RefUnwindSafe, I: RefUnwindSafe, S, P, const M: usize> RefUnwindSafe
    for Iter<'t, C, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
}

// Bounds for `Send`/`Sync` are a little complicated. We have to consider that this is *kind of*
// like a `&RleTree`, but also stores some copied `I` from it.
unsafe impl<'t, C: Send, I: Send + Sync, S: Sync, P: Send, const M: usize> Send
    for Iter<'t, C, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
}

unsafe impl<'t, C: Sync, I: Sync, S: Sync, P: Sync, const M: usize> Sync for Iter<'t, C, I, S, P, M> where
    P: RleTreeConfig<I, S, M>
{
}

struct IterState<'t, C, I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S, M>,
{
    store: &'t P::SliceRefStore,
    root: NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
    cursor: Option<C>,
    fwd: Option<IterStack<'t, I, S, P, M>>,
    bkwd: Option<IterStack<'t, I, S, P, M>>,
}

struct IterStack<'t, I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S, M>,
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

#[cfg(test)]
impl<'t, C, I, S, P: RleTreeConfig<I, S, M>, const M: usize> Debug for Iter<'t, C, I, S, P, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("Iter")
            .field("start", self.start.fallible_debug())
            .field("end", self.end.fallible_debug())
            .field("state", &self.state)
            .finish()
    }
}

#[cfg(test)]
impl<'t, C, I, S, P: RleTreeConfig<I, S, M>, const M: usize> Debug
    for IterState<'t, C, I, S, P, M>
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut s = f.debug_struct("IterState");
        s.field("root", &self.root.ptr());

        // Only show the cursor if there's possibly anything there
        if std::mem::size_of::<C>() != 0 {
            // match on `self.cursor` so that we display `Some(<No debug impl>)` if the cursor is
            // present but doesn't have a debug implementation; or `None` if the cursor isn't
            // present, regardless of whether it has a debug impl.
            let _ = match self.cursor.as_ref() {
                Some(c) => s.field("cursor", &Some(c.fallible_debug())),
                None => s.field("cursor", &None as &Option<()>),
            };
        }

        s.field("fwd", &self.fwd).field("bkwd", &self.bkwd).finish()
    }
}

#[cfg(test)]
impl<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> Debug for IterStack<'t, I, S, P, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        struct JustTheStack<'a, 't, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
            this: &'a IterStack<'t, I, S, P, M>,
        }

        impl<'a, 't, I, S, P: RleTreeConfig<I, S, M>, const M: usize> Debug
            for JustTheStack<'a, 't, I, S, P, M>
        {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                f.debug_list()
                    .entries(
                        self.this.stack.iter().map(|(node, child_idx)| (node.ptr(), *child_idx)),
                    )
                    .finish()
            }
        }

        struct HeadNode<T> {
            ptr: std::ptr::NonNull<T>,
            idx: u8,
        }

        impl<T> Debug for HeadNode<T> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                f.debug_struct("").field("ptr", &self.ptr).field("idx", &self.idx).finish()
            }
        }

        f.debug_struct("IterStack")
            .field("stack", &JustTheStack { this: self })
            .field("head", &HeadNode { ptr: self.head.node.ptr(), idx: self.head.idx })
            .field("end_pos", self.end_pos.fallible_debug())
            .finish()
    }
}

#[derive(Debug, Copy, Clone)]
enum IncludedOrExcludedBound<I> {
    Included(I),
    Excluded(I),
}

#[track_caller]
fn panic_internal_error_or_bad_index<I: Index>() -> ! {
    if I::TRUSTED {
        panic!("internal error");
    } else {
        panic!("internal error or bad `Index` implementation");
    }
}

impl<'t, C, I, S, P, const M: usize> Iter<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: 't + Slice<I>,
    P: RleTreeConfig<I, S, M>,
{
    /// Creates a new iterator
    ///
    /// This function panics if the arguments are invalid, as described in [`RleTree::iter`].
    ///
    /// [`RleTree::iter`]: crate::RleTree::iter
    #[track_caller]
    pub(super) fn new(
        range: impl Debug + RangeBounds<I>,
        tree_size: I,
        cursor: C,
        mut root: Option<(
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
            EndBound::Unbounded => {
                (IncludedOrExcludedBound::Excluded(tree_size), EndBound::Excluded(tree_size))
            }
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

        // Special case for empty iterators, so we don't attempt to traverse the tree when we won't
        // end up with anything.
        let bounded_end_bound = match end {
            IncludedOrExcludedBound::Included(i) => EndBound::Included(i),
            IncludedOrExcludedBound::Excluded(i) => EndBound::Excluded(i),
        };

        if (StartBound::Included(start), bounded_end_bound).is_empty_naive() {
            root = None;
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
        let mut head_pos = I::ZERO;
        let (mut target, excluded) = match target {
            IncludedOrExcludedBound::Included(i) => (i, false),
            IncludedOrExcludedBound::Excluded(i) => (i, true),
        };

        // Search downward with the cursor
        loop {
            let hint = cursor.as_mut().and_then(|c| c.next());

            let result = if target != head_node.leaf().subtree_size() {
                match search_step(head_node, hint, target) {
                    ChildOrKey::Key((k_idx, k_pos)) if excluded && k_pos == target => {
                        match head_node.into_typed() {
                            Type::Leaf(n) => {
                                if k_idx == 0 {
                                    panic_internal_error_or_bad_index::<I>();
                                }

                                // SAFETY: key indexes returned from `search_step` are guaranteed
                                // to be in bounds; we checked above that `k_idx != 0`, so
                                // subtracting one won't overlflow: `k_idx - 1` is still in bounds.
                                let lhs_k_pos = unsafe { n.leaf().key_pos(k_idx - 1) };
                                ChildOrKey::Key((k_idx - 1, lhs_k_pos))
                            }
                            Type::Internal(n) => {
                                let c_idx = k_idx;
                                // SAFETY: key indexes returned from `search_step` are guaranteed
                                // to be in bounds, so a child at the same index (i.e. to its left)
                                // will also be valid.
                                let c_pos = k_pos.sub_right(unsafe { n.child_size(c_idx) });
                                ChildOrKey::Child((c_idx, c_pos))
                            }
                        }
                    }
                    ChildOrKey::Child((c_idx, c_pos)) if excluded && c_pos == target => {
                        if c_idx == 0 {
                            panic_internal_error_or_bad_index::<I>();
                        }

                        let k_idx = c_idx - 1;
                        // SAFETY: child indexes returned from `search_step` are guaranteed to be
                        // valid children, so the key to their left (at `c_idx - 1`) is a valid key
                        let k_pos = unsafe { head_node.leaf().key_pos(k_idx) };
                        ChildOrKey::Key((k_idx, k_pos))
                    }
                    res => res,
                }
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
                    let slice_end = head_pos.add_right(k_pos).add_right(slice_handle.slice_size());
                    return IterStack { stack, end_pos: slice_end, head: slice_handle };
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
                        match child.leaf().parent().map(|p| (p.ptr, p.idx_in_parent)) {
                            // Different child pointer:
                            Some(pair) if pair != (node.ptr(), c_idx) => {
                                stack.push((head_node, c_idx))
                            }
                            _ => (),
                        }
                    }

                    target = target.sub_left(c_pos);
                    head_node = child;
                    head_pos = head_pos.add_right(c_pos);
                }
            }
        }
    }
}

impl<'t, I, S, P, const M: usize> IterStack<'t, I, S, P, M>
where
    I: Index,
    P: RleTreeConfig<I, S, M>,
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
                let mut c_idx = self.head.idx;
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
                            // If this node has no parent, then we're at the end of the root node;
                            // no more iteration is possible. This should have already been caught
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
    P: RleTreeConfig<I, S, M>,
{
    type Item = SliceEntry<'t, I, S, P, M>;

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.state.as_mut()?;

        let (fwd, next_start) = match state.fwd.as_mut() {
            Some(fwd) => {
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

                (fwd, next_start)
            }
            None => {
                let s = Self::make_stack(
                    state.root,
                    state.cursor.take(),
                    IncludedOrExcludedBound::Included(self.start),
                );

                // We can't use `self.start` for the start position because it might be in the
                // middle of a slice
                let next_start = s.end_pos.sub_right(s.head.slice_size());
                (state.fwd.insert(s), next_start)
            }
        };

        // If the current head is already the head of the backward stack, then it's already been
        // returned, which would mean that the iterator has yielded all of its items.
        let is_done = matches!(state.bkwd.as_ref(), Some(s) if s.head == fwd.head);

        if is_done {
            self.state = None;
            None
        } else {
            Some(SliceEntry {
                range_start: next_start,
                range_end: fwd.end_pos,
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
    P: RleTreeConfig<I, S, M>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let state = self.state.as_mut()?;

        let bkwd = match state.bkwd.as_mut() {
            Some(bkwd) => {
                let next_end = bkwd.end_pos.sub_right(bkwd.head.slice_size());
                if next_end <= self.start {
                    self.state = None;
                    return None;
                }

                bkwd.bkwd_step();
                bkwd
            }
            None => {
                let s = Self::make_stack(state.root, state.cursor.take(), self.end);

                state.bkwd.insert(s)
            }
        };

        let is_done = matches!(state.fwd.as_ref(), Some(s) if s.head == bkwd.head);
        if is_done {
            self.state = None;
            None
        } else {
            let start = bkwd.end_pos.sub_right(bkwd.head.slice_size());

            Some(SliceEntry {
                range_start: start,
                range_end: bkwd.end_pos,
                slice: bkwd.head,
                store: state.store,
            })
        }
    }
}
