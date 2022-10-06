//! Wrapper module for [`RleTree`](crate::RleTree)'s destructive iterator -- [`Drain`] and related
//! types

use crate::param::{BorrowState, RleTreeConfig, SliceRefStore, SupportsInsert};
use crate::public_traits::{Index, Slice};
use crate::range::RangeBounds;
use crate::Cursor;
use std::mem::{self, ManuallyDrop};
use std::ops::Range;

use super::node::{borrow, ty, NodeHandle, SliceHandle, Type};
use super::{
    bounded_search_step, panic_internal_error_or_bad_index, ChildOrKey, Root, Side,
    DEFAULT_MIN_KEYS,
};

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
    /// Total range of values returned by the iterator
    range: Range<I>,
    /// Stored reference to the tree, held so that we can put the root back once we're done
    ///
    /// ## Safety
    ///
    /// `tree_ref` must not be accessed before a later access to `state` or a value derived from
    /// it.
    tree_ref: &'t mut Option<Root<I, S, P, M>>,
    /// Copy of the tree's root, so that if the `Drain` is dropped, we don't leave the tree in an
    /// invalid state
    root: Option<Root<I, S, P, M>>,
    /// The current state of iteration, if it has been started. Else, the initial cursor. The full
    /// type will be `None` when `range` is empty.
    ///
    /// The lifetime `'t` here is technically invalid; we're actually borrowing `root`, and we have
    /// to be careful in the destructor that we don't access `state` after dropping `root` or
    /// moving it back into `tree_ref`.
    state: ManuallyDrop<Option<Result<DrainState<'t, I, S, P, M>, C>>>,
}

/// (*Internal*) Result of a call to [`Drain::initial_search`]
///
/// In general, the caller (`next` or `next_back`) is responsible for processing this search
/// result, adding to an existing or initial [`DrainState`]
struct DrainInitialSearch<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    slice: SliceHandle<ty::Unknown, borrow::DrainMut<'t>, I, S, P, M>,
    /// Absolute position of `slice`
    slice_pos: I,
}

/// (*Internal*) Current state of the iterator, *after* it has started, and with a non-empty range
struct DrainState<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    start: Option<DrainSide<'t, I, S, P, M>>,
    end: Option<DrainSide<'t, I, S, P, M>>,
}

impl<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> DrainState<'t, I, S, P, M> {
    /// Returns `true` iff iteration has been completed
    ///
    /// Iteration has only been completed when the cursors in `self.start` and `self.end` are
    /// equal, an invariant that we guarantee during calls to `next` and `next_back`.
    fn done(&self) -> bool {
        if let (Some(start), Some(end)) = (&self.start, &self.end) {
            start.cursor.slice == end.cursor.slice
        } else {
            false
        }
    }
}

/// (*Internal*) One side of a [`DrainState`], encompassing both drained range bound
/// ([`DrainEdge`]) and current iteration position ([`DrainSideCursor`])
struct DrainSide<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    edge: DrainEdge<'t, I, S, P, M>,
    cursor: DrainSideCursor<'t, I, S, P, M>,
}

/// (*Internal*) Static state associated with one side of the drained range
///
/// The state tracking the progress of the iteration itself is given by [`DrainSideCursor`].
struct DrainEdge<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    /// The slice containing the outermost key (for this edge) overapping with the drained edge
    outermost_removed: SliceHandle<ty::Unknown, borrow::DrainMut<'t>, I, S, P, M>,
    /// Absolute position of `outermost_removed` within the tree
    slice_pos: I,
    /// If the matching bound on `Drain.range` is in the middle of `outermost_removed`, `writeback`
    /// stores the *size* and *value* of the key extracted from the outer side of the slice as it was
    /// split to produce the outermost value for this side
    ///
    /// The value will be written back into `outermost_removed` (if possible; some edge cases
    /// make this harder) when the tree is dropped
    writeback: Option<(I, S)>,
}

/// (*Internal*) Dynamic state (tracks iteration progress) of one side of the drained range
///
/// The static state is stored in a [`DrainEdge`].
///
/// Broadly speaking, a `DrainSideCursor` stores a reference to the current position in our
/// iteration. Semantics are slightly different for the "start" vs "end" cursor: When a
/// `DrainSideCursor` is part of [`DrainState.start`]
///
/// [`DrainState.start`]: DrainState#field.start
struct DrainSideCursor<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    /// Cursor's current position in the tree. Refer to top-level docs for semantics
    slice: SliceHandle<ty::Unknown, borrow::DrainMut<'t>, I, S, P, M>,

    /// Absolute start position of `slice`
    slice_pos: I,
}

impl<'t, C, I, S, P, const M: usize> Drain<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    /// (*Internal*) Internal constructor for a `Drain`, panicking if the range is out of bounds
    #[track_caller]
    pub(super) fn new(root: &'t mut Option<Root<I, S, P, M>>, range: Range<I>, cursor: C) -> Self {
        if range.starts_after_end() {
            panic!("invalid range `{range:?}`");
        } else if range.start < I::ZERO {
            panic!("range `{range:?}` out of bounds below zero");
        }

        let size = root
            .as_ref()
            .map(|r| r.handle.leaf().subtree_size())
            .unwrap_or(I::ZERO);
        if range.end > size || range.start >= size {
            panic!("range `{range:?}` out of bounds for tree size {size:?}");
        }

        let (tree_ref, root) = {
            let r = root.take();
            (root, r)
        };

        // If the range is empty, then we shouldn't remove anything.
        let state = match range.start == range.end {
            true => None,
            false => Some(Err(cursor)),
        };

        // Mutably borrow the root. We'll release this borrow on drop.
        if let Some(r) = root.as_ref() {
            if let Err(conflict) = r.refs_store.acquire_mutable() {
                panic!("{conflict}");
            }
        }

        Drain {
            range,
            root,
            tree_ref,
            state: ManuallyDrop::new(state),
        }
    }

    /// (*Internal*) Traverses the tree to produce a handle on the slice containing(-ish) the
    /// target
    ///
    /// `excluded` is true for the upper bound only.
    fn initial_search(
        root: &mut Root<I, S, P, M>,
        cursor: Option<C>,
        other_side: Option<(Side, &DrainSideCursor<'t, I, S, P, M>)>,
        mut target: I,
        excluded: bool,
    ) -> DrainInitialSearch<'t, I, S, P, M> {
        let node: NodeHandle<ty::Unknown, borrow::DrainMut<'_>, I, S, P, M> =
            root.handle.make_unique().borrow_mut().into_drain();
        // SAFETY: We're extending the lifetime on the root node in a way that is not valid by
        // itself. We need to do this in order for the compiler to let us store references to
        // `root` in `Drain` itself. We don't need the `Drain` to be pinned because the borrowed
        // contents are of heap-allocated nodes, which won't move when `Drain` does.
        let mut node: NodeHandle<ty::Unknown, borrow::DrainMut<'t>, I, S, P, M> =
            unsafe { mem::transmute(node) };

        let mut node_pos = I::ZERO;
        let mut cursor = cursor.map(|c| c.into_path());

        loop {
            let hint = cursor.as_mut().and_then(|c| c.next());
            // SAFETY: `borrow_unchecked` requires that we never use the handle to produce a
            // reference to a `Slice`, which `bounded_search_step` doesn't do.
            let borrowed = unsafe { node.borrow_unchecked() };
            let result = bounded_search_step(borrowed, hint, target, excluded);

            match result {
                ChildOrKey::Key((k_idx, k_pos)) => {
                    // SAFETY: `bounded_search_step` guarantees that `k_idx` is within the bounds
                    // of the node.
                    let slice = unsafe { node.into_slice_handle(k_idx) };
                    if let Some((side, drain_cursor)) = other_side {
                        drain_check_valid_search_result(slice.copy_handle(), side, drain_cursor);
                    }

                    let slice_pos = node_pos.add_right(k_pos);
                    return DrainInitialSearch { slice, slice_pos };
                }
                ChildOrKey::Child((c_idx, c_pos)) => {
                    let this = match node.into_typed() {
                        Type::Leaf(_) => panic_internal_error_or_bad_index::<I>(),
                        Type::Internal(n) => n,
                    };

                    // SAFETY: `bounded_search_step` guarantees that `c_idx` is a valid child
                    // index.
                    let child = unsafe { this.into_child(c_idx) };
                    target = target.sub_left(c_pos);
                    node = child;
                    node_pos = node_pos.add_right(c_pos);
                }
            }
        }
    }

    /// Return the [`SliceHandle`] corresponding to the next (position-wise) slice in the tree
    ///
    /// ## Panics
    ///
    /// This methd panics by calling [`panic_internal_error_or_bad_index`] if this is the last
    /// `SliceHandle` in the tree.
    fn traverse_next(
        slice: SliceHandle<ty::Unknown, borrow::DrainMut<'t>, I, S, P, M>,
    ) -> SliceHandle<ty::Unknown, borrow::DrainMut<'t>, I, S, P, M> {
        match slice.into_typed() {
            Type::Leaf(mut h) if h.idx + 1 < h.node.leaf().len() => {
                h.idx += 1;
                return h.erase_type();
            }
            Type::Leaf(h) => {
                let mut node = h.node.erase_type();

                // Recurse upwards to the right
                loop {
                    let (parent, child_idx) = match node.into_parent() {
                        Err(_) => panic_internal_error_or_bad_index::<I>(),
                        Ok(p) => p,
                    };
                    node = parent.erase_type();

                    // If this is the last child, continue recursing
                    if node.leaf().len() == child_idx {
                        continue;
                    }

                    // Otherwise, we pick the key immediately after the child, which has the
                    // same index as the child.
                    //
                    // SAFETY: `into_slice_handle` requires `child_idx <= parent.leaf().len()`,
                    // which is guaranteed by the check above (it cannot be greater).
                    return unsafe { node.into_slice_handle(child_idx) };
                }
            }
            Type::Internal(h) => {
                let mut node = h.node;
                let mut c_idx = h.idx + 1; // Child to the right is key index + 1

                // Recurse downwards to the right
                loop {
                    // SAFETY: `into_child` requires `c_idx <= node.leaf().len()`, which is
                    // guaranteed on every iteration after the first (`c_idx = 0`). For the
                    // first iteration, we're using `h.idx + 1`, which is ok because `h.idx` is
                    // guaranteed `< node.leaf().len()`.
                    let child = unsafe { node.into_child(c_idx) };
                    match child.into_typed() {
                        Type::Leaf(h) => {
                            let n = h.erase_type();
                            // SAFETY: valid trees are guaranteed to have len > 0 on each node,
                            // so getting key index 0 is ok.
                            return unsafe { n.into_slice_handle(0) };
                        }
                        Type::Internal(n) => {
                            node = n;
                            c_idx = 0;
                        }
                    }
                }
            }
        }
    }

    /// Return the [`SliceHandle`] corresponding to the previous (position-wise) slice in the tree
    ///
    /// ## Panics
    ///
    /// This methd panics by calling [`panic_internal_error_or_bad_index`] if this is the last
    /// `SliceHandle` in the tree.
    fn traverse_next_back(
        slice: SliceHandle<ty::Unknown, borrow::DrainMut<'t>, I, S, P, M>,
    ) -> SliceHandle<ty::Unknown, borrow::DrainMut<'t>, I, S, P, M> {
        match slice.into_typed() {
            Type::Leaf(mut h) if h.idx > 0 => {
                h.idx -= 1;
                return h.erase_type();
            }
            Type::Leaf(h) => {
                // Note: h.idx == 0.
                let mut node = h.node.erase_type();

                // Recurse upwards to the left
                loop {
                    let (parent, child_idx) = match node.into_parent() {
                        Err(_) => panic_internal_error_or_bad_index::<I>(),
                        Ok(p) => p,
                    };
                    node = parent.erase_type();

                    // If this is the first child, continue recursing
                    if child_idx == 0 {
                        continue;
                    }

                    // Otherwise, we pick the key immediately before the child, which is at the
                    // child's index minus one
                    //
                    // SAFETY: `into_slice_handle` requires `child_idx <= parent.leaf().len()`,
                    // which is guaranteed because `child_idx != 0` means the subtraction won't
                    // overflow, and `child_idx <= parent.leaf().len()` already.
                    return unsafe { node.into_slice_handle(child_idx - 1) };
                }
            }
            Type::Internal(h) => {
                let mut node = h.node;
                let mut c_idx = h.idx; // Child to the left is equal to key index

                // Recurse downwards to the left
                loop {
                    // SAFETY: `into_child` requires `c_idx <= node.leaf().len()`, which is
                    // guaranteed on every iteration after the first (`c_idx = node.leaf().len()`).
                    // For the first iteration, we're using `h.idx`, which is ok because `h.idx` is
                    // guaranteed `< node.leaf().len()`.
                    let child = unsafe { node.into_child(c_idx) };
                    match child.into_typed() {
                        Type::Leaf(h) => {
                            let n = h.erase_type();
                            let len = n.leaf().len();
                            // SAFETY: `len != 0` is guaranteed for any vaild node
                            unsafe { weak_assert!(len != 0) };
                            // SAFETY: `into_slice_handle` requires `idx < len`, which is true for
                            // `len - 1` as long as it doesn't overflow (which we checked above).
                            return unsafe { n.into_slice_handle(len - 1) };
                        }
                        Type::Internal(n) => {
                            node = n;
                            c_idx = node.leaf().len();
                        }
                    }
                }
            }
        }
    }
}

/// (*Internal*) Check that a [`SliceHandle`] returned by [`initial_search`] is valid, given the
/// other side's iteration
///
/// This method is necessary because it's possible for a bad `Index` implementation to yield a
/// second search result inside of or before what we've already returned from the iterator.
///
/// This method is called before any return from [`initial_search`].
///
/// [`initial_search`]: Drain::initial_search
// We'd like this to be an associated function on `Drain`, but due to rust-lang/rust#102611 this
// has to be separated out into a standalone function.
//
// (link: https://github.com/rust-lang/rust/issues/102611)
fn drain_check_valid_search_result<'t, I, S, P, const M: usize>(
    slice: SliceHandle<ty::Unknown, borrow::DrainMut<'t>, I, S, P, M>,
    cursor_side: Side,
    drain_cursor: &DrainSideCursor<'t, I, S, P, M>,
) where
    I: Index,
    P: RleTreeConfig<I, S, M>,
{
    // There's two possibilities to be concerned about here: (a) `slice` is positionally within a
    // range that's already been returned, or (b) `slice` is positionally in the opposite direction
    // to the other bound of the drained range.
    //
    // These are both handled by checking if `slice` is on the "wrong" side of `drain_cursor`,
    // basically: if its direction relative to `drain_cursor` is equal to `cursor_side`.
    //
    // ---
    //
    // Algorithm:
    //
    // We traverse up the tree from `slice` and `drain_cursor.node` until we get to a common
    // ancestor, at which point we make the comparison with their positions and `cursor_side` to
    // determine if `slice` is on the correct side.

    let mut slice_node = slice.node.copy_handle();
    let mut cursor_node = drain_cursor.slice.node.copy_handle();

    let mut slice_idx = ChildOrKey::Key(slice.idx);
    let mut cursor_idx = ChildOrKey::Key(drain_cursor.slice.idx);

    // Traverse up the tree:
    loop {
        if slice_node.ptr() == cursor_node.ptr() {
            // SAFETY: Guaranteed by tree invariants.
            unsafe { weak_assert!(slice_node.height() == cursor_node.height()) };
            break;
        }

        // Move `slice_node` or `cursor_node` up the tree, depending on which one is further down.
        // If they're at the same height, move both up because we know they're not the same node
        // (from the check above).

        if slice_node.height() <= cursor_node.height() {
            match slice_node.into_parent() {
                Ok((h, c_idx)) => {
                    slice_node = h.erase_type();
                    slice_idx = ChildOrKey::Child(c_idx);
                }
                // SAFETY: If this node has no parent, then either `cursor_node` is at a higher
                // height than the root of the tree (tree invariant has been broken) or it is at
                // the same height but must be a different node (invariant is still broken). So
                // because we rely on tree invariants for soundness, it is no less sound to create
                // UB here on failure. We're guaranteed to have "correct" parents from the downward
                // traversal that created the `DrainMut`
                Err(_) => unsafe { weak_unreachable!() },
            }
        }

        if cursor_node.height() <= slice_node.height() {
            match cursor_node.into_parent() {
                Ok((h, c_idx)) => {
                    cursor_node = h.erase_type();
                    cursor_idx = ChildOrKey::Child(c_idx);
                }
                // SAFETY: see not above for `slice_node.into_parent()`.
                Err(_) => unsafe { weak_unreachable!() },
            }
        }
    }

    macro_rules! check {
        ($cond:expr) => {{
            if $cond {
                return;
            } else {
                panic_internal_error_or_bad_index::<I>();
            }
        }};
    }

    // Check for validity
    match cursor_side {
        // Cursor's to the left, so `slice` must be to its right.
        Side::Left => match (slice_idx, cursor_idx) {
            (ChildOrKey::Key(s), ChildOrKey::Key(c)) => check!(s >= c),
            (ChildOrKey::Key(s), ChildOrKey::Child(c)) => check!(s >= c),
            (ChildOrKey::Child(s), ChildOrKey::Key(c)) => check!(s > c),
            (ChildOrKey::Child(s), ChildOrKey::Child(c)) => check!(s >= c),
        },
        // Cursor's to the right, so `slice` must be to its left.
        Side::Right => match (slice_idx, cursor_idx) {
            (ChildOrKey::Key(s), ChildOrKey::Key(c)) => check!(s < c),
            (ChildOrKey::Key(s), ChildOrKey::Child(c)) => check!(s < c),
            (ChildOrKey::Child(s), ChildOrKey::Key(c)) => check!(s <= c),
            (ChildOrKey::Child(s), ChildOrKey::Child(c)) => check!(s < c),
        },
    }
}

impl<'t, C, I, S, P, const M: usize> Iterator for Drain<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: 't + Slice<I>,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    type Item = (Range<I>, S);

    fn next(&mut self) -> Option<Self::Item> {
        let root = self.root.as_mut()?;
        let state_result = self.state.as_mut()?;

        // Helper value to abbreviate logic below.
        let empty_state = DrainState {
            start: None,
            end: None,
        };

        let (start_side, end_side) = 'state: loop {
            // ^ treat the loop as a way to short-circuit needing to find the `start` side of the
            // range we're iterating over (i.e. skip it if we don't need to).
            let (maybe_cursor, state) = match state_result.as_mut() {
                Ok(state) if state.done() => return None,
                Ok(state) => match state.start.as_mut() {
                    Some(s) => break 'state (s, &mut state.end),
                    None => {
                        assert!(state.end.is_some(), "internal error");
                        (None, state)
                    }
                },
                // If we haven't started yet, temporarily replace `state_result` with a fully-empty
                // version of `DrainState` and then extract references to it.
                Err(_) => match mem::replace(state_result, Ok(empty_state)) {
                    Err(cursor) => match state_result.as_mut() {
                        Ok(s) => (Some(cursor), s),
                        // SAFETY: we just set this to `Ok(_)`
                        Err(_) => unsafe { weak_unreachable!() },
                    },
                    // SAFETY: we just observed this as an `Err(_)`
                    Ok(_) => unsafe { weak_unreachable!() },
                },
            };

            // Either we still have the cursor (nothing's been done yet) or `state.start` is None.
            // However, `state` always needs to have at least one non-None side, so `state.end`
            // must be `Some`.            ( `!=` is xor )
            debug_assert!(maybe_cursor.is_some() != state.end.is_some());

            let other_side = state.end.as_ref().map(|s| (Side::Right, &s.cursor));
            let mut search =
                Self::initial_search(root, maybe_cursor, other_side, self.range.start, false);
            let split_pos = search.slice_pos.sub_left(self.range.start);

            let mut writeback = None;
            if split_pos != I::ZERO {
                // We don't need to worry here about the special case where the two ends of the
                // drained range are within the same slice; that's handled further in the call to
                // `next`, where we limit the range of what we're returning to be bounded by the
                // end of the drained range.
                //
                // SAFETY: `with_slice_unchecked` requires that the slice hasn't already been
                // removed by a call to `remove_slice_unchecked`, which is guaranteed by our
                // pre-return call to `check_valid_search_result` in `initial_search`.
                unsafe {
                    search.slice.with_slice_unchecked(|s| {
                        // The left-hand side is the writeback. In order to return that, we have to
                        // put the split right-hand side back into the slice.
                        let rhs = s.split_at(split_pos);
                        let lhs = mem::replace(s, rhs);
                        writeback = Some(lhs);
                    });
                }
            }

            let start_side = DrainSide {
                edge: DrainEdge {
                    outermost_removed: search.slice.copy_handle(),
                    slice_pos: search.slice_pos,
                    writeback: writeback.map(|s| (split_pos, s)),
                },
                cursor: DrainSideCursor {
                    slice: search.slice,
                    slice_pos: search.slice_pos,
                },
            };
            break (state.start.insert(start_side), &mut state.end);
        };

        // At this point, we've been given `start_side` and `end_side`, and are free to produce the
        // next item in the iteration (with certain checks to ensure we don't double-yield).
        //
        // ---
        //
        // Algorithm:
        //
        // Fetch immediate slice information from `start_side.cursor`. Define:
        //  * `range: Range<I>` as the slice's range
        //  * `value: S` as the slice's value
        // If `end_side` is `None` and `range.end >= self.range.end`, then (approx.) set `end_side`
        // to `Some(start_side.cursor)` and appropriately hande its writeback. Otherwise, progress
        // `start_side.cursor` to the next slice in the tree (performing whatever traversal is
        // necessary).

        // slice_range: full range of the slice
        let slice_range = {
            let start = start_side.cursor.slice_pos;
            let size = start_side.cursor.slice.slice_size();
            start..start.add_right(size)
        };
        // range: range of the slice, with bounds clamped to `self.range`
        let range = Range {
            start: slice_range.start.max(self.range.start),
            end: slice_range.end.min(self.range.end),
        };

        // SAFETY: `remove_slice_unchecked` requires that we haven't previously called that method
        // for the same slice. This is guaranteed by the checks we use to ensure we don't
        // double-yield a value, with `if state.is_done() => return None` above and everything in
        // the remainder of this function.
        let mut value = unsafe { start_side.cursor.slice.remove_slice_unchecked() };
        // SAFETY: `remove_refid_unchecked` has the same restrictions as `remove_slice_unchecked`,
        // and our call here is sound for the same reason.
        let ref_id = unsafe { start_side.cursor.slice.remove_refid_unchecked() };
        root.refs_store.remove(ref_id);

        if end_side.is_none() && slice_range.end >= self.range.end {
            // From above: "If `end_side` is `None` and `slice_range.end >= self.range.end`, then
            // set `end_side` to `Some(start_side.cursor)` and appropriately handle its writeback."

            let end_ref = end_side.insert(DrainSide {
                edge: DrainEdge {
                    outermost_removed: start_side.cursor.slice.copy_handle(),
                    slice_pos: slice_range.start,
                    writeback: None, // <- we'll set this below if we need to
                },
                cursor: DrainSideCursor {
                    slice: start_side.cursor.slice.copy_handle(),
                    slice_pos: slice_range.start,
                },
            });

            // equal to: slice_range.end > self.range.end
            if slice_range.end != range.end {
                // [diagram note: area marked as '~' will be removed.]
                //
                // Removal not contained within the slice:
                //
                //   slice:  |~~~~~~~~~~~~~~~~~~~~~~~~~/-----------|
                //           |                  self.range.end
                //    slice_range.start
                //
                // Removal contained within the slice:
                //
                //   slice:  |----------/~~~~~~~~~~~~~~/-----------|
                //           |          |       self.range.end
                //           |  self.range.start
                //    slice_range.start
                //
                // With the slice ranging between the two '|'s, we can see that the split position
                // to use *after* any necessary left-hand split has been done will be given by
                // `range.end - range.start`, because: `range.start` is the maximum of `slice_pos`
                // and `self.range.start`; and `range.end` is currently equal to `self.range.end`.

                let splitpoint = range.end.sub_left(range.start);
                let split_size = slice_range.end.sub_left(range.end);
                let split_val = value.split_at(splitpoint);
                end_ref.edge.writeback = Some((split_size, split_val));
            }
        } else if cfg!(debug_assertions) && slice_range.end != range.end {
            // Above condition implies end_side.is_some() and range.end > self.range.end, which
            // shouldn't be true.
            panic_internal_error_or_bad_index::<I>();
        } else {
            // From above: "... Otherwise, progress `start_side.cursor` to the next slice in the
            // tree (performing whatever traversal is necessary)."
            //
            // For concerns about double-yielding a value: We already extracted `value`, which is
            // guaranteed to be fine because of the invariants of `DrainSideCursor`. The traversal
            // here exists to ensure that we appropriately update `start_side` so that a call to
            // `next_back` doesn't find the same value (if it otherwise would have, then
            // `start_side.cursor == end_side.cursor` by the end of this).
            //
            // For concerns about walking off the end of the tree: If we get to the end of the
            // tree, something went wrong -- either our fault or a bad index impl (we panic).

            start_side.cursor.slice = Self::traverse_next(start_side.cursor.slice.copy_handle());
            start_side.cursor.slice_pos = slice_range.end;
        }

        Some((range, value))
    }
}

impl<'t, C, I, S, P, const M: usize> DoubleEndedIterator for Drain<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: 't + Slice<I>,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let root = self.root.as_mut()?;
        let state_result = self.state.as_mut()?;

        // Helper value to abbreviate logic below.
        let empty_state = DrainState {
            start: None,
            end: None,
        };

        let (needs_traversal, start_side, end_side) = 'state: loop {
            // ^ treat the loop as a way to short-circuit needing to find the `start` side of the
            // range we're iterating over (i.e. skip it if we don't need to).
            let (maybe_cursor, state) = match state_result.as_mut() {
                Ok(state) if state.done() => return None,
                Ok(state) => match state.end.as_mut() {
                    Some(s) => break 'state (true, &mut state.start, s),
                    None => {
                        assert!(state.start.is_some(), "internal error");
                        (None, state)
                    }
                },
                // If we haven't started yet, temporarily replace `state_result` with a fully-empty
                // version of `DrainState` and then extract references to it.
                Err(_) => match mem::replace(state_result, Ok(empty_state)) {
                    Err(cursor) => match state_result.as_mut() {
                        Ok(s) => (Some(cursor), s),
                        // SAFETY: we just set this to `Ok(_)`
                        Err(_) => unsafe { weak_unreachable!() },
                    },
                    // SAFETY: we just observed this as an `Err(_)`
                    Ok(_) => unsafe { weak_unreachable!() },
                },
            };

            // Either we still have the cursor (nothing's been done yet) or `state.end` is None.
            // However, `state` always needs to have at least one non-None side, so `state.start`
            // must be `Some`.            ( `!=` is xor )
            debug_assert!(maybe_cursor.is_some() != state.start.is_some());

            let other_side = state.start.as_ref().map(|s| (Side::Left, &s.cursor));
            let mut search =
                Self::initial_search(root, maybe_cursor, other_side, self.range.end, true);
            let split_pos = self.range.end.sub_left(search.slice_pos);
            let slice_size = search.slice.slice_size();

            let mut writeback = None;
            if split_pos != slice_size {
                // We don't need to worry here about the special case where the two ends of the
                // drained range are within the same slice; that's handled further in the call to
                // `next_back`, where we limit the range of what we're returning to be bounded by
                // the end of the drained range.
                //
                // SAFETY: `with_slice_unchecked` requires that the slice hasn't already been
                // removed by a call to `remove_slice_unchecked`, which is guaranteed by our
                // pre-return call to `check_valid_search_result` in `initial_search`.
                let rhs = unsafe { search.slice.with_slice_unchecked(|s| s.split_at(split_pos)) };
                let writeback_size = slice_size.sub_left(split_pos);
                writeback = Some((split_pos, rhs));
            }

            let end_side = DrainSide {
                edge: DrainEdge {
                    outermost_removed: search.slice.copy_handle(),
                    slice_pos: search.slice_pos,
                    writeback,
                },
                cursor: DrainSideCursor {
                    slice: search.slice,
                    slice_pos: search.slice_pos,
                },
            };

            break (false, &mut state.start, state.end.insert(end_side));
        };

        // At this point, we've been given `start_side` and `end_side`, and are free to produce the
        // next item in the iteration (with certain checks to ensure we don't double-yield).
        //
        // ---
        //
        // Algorithm:
        //
        // If we didn't just create `end_side`, progress `end_side.cursor` to the previous slice in
        // the tree (performing whatever traversal is necessary). Update `end_side.cursor.slice_pos`
        // Fetch immediate slice information from `end_side.cursor`. Define:
        //  * `range: Range<I>` as the slice's range
        //  * `value: S` as the slice's value
        // If `start_side` is `None` and `slice_range.start <= self.range.start`, then (approx.)
        // set `start_side` to `Some(end_side.cursor)` and appropriately handle its writeback.

        // slice_range: full range of the slice. Also handle traversal here. Also update
        // `end_side.cursor.slice_pos`
        let slice_range = if needs_traversal {
            end_side.cursor.slice = Self::traverse_next_back(end_side.cursor.slice.copy_handle());

            let end = end_side.cursor.slice_pos;
            let size = end_side.cursor.slice.slice_size();
            let start = end.sub_right(size);

            end_side.cursor.slice_pos = start;
            start..end
        } else {
            let start = end_side.cursor.slice_pos;
            let size = end_side.cursor.slice.slice_size();
            start..start.add_right(size)
        };

        // range: range of the slice, with bounds clamped to `self.range`
        let range = Range {
            start: slice_range.start.max(self.range.start),
            end: slice_range.end.min(self.range.end),
        };

        // SAFETY: `remove_slice_unchecked` requires that we haven't previously called that method
        // for the same slice. This is guaranteed by the checks we use to ensure we don't
        // double-yield a value, with `if state.is_done() => return None` above and everything in
        // the remainder of this function (plus traversal above).
        let mut value = unsafe { end_side.cursor.slice.remove_slice_unchecked() };
        // SAFETY: `remove_refid_unchecked` has the same restrictions as `remove_slice_unchecked`,
        // and our call here is sound for the same reason.
        let ref_id = unsafe { end_side.cursor.slice.remove_refid_unchecked() };
        root.refs_store.remove(ref_id);

        if start_side.is_none() && slice_range.start <= self.range.start {
            // From above: "If `start_side` is `None` and `slice_range.start <= self.range.start`,
            // then set `start_side` to `Some(end_side.cursor)` and appropriately handle its
            // writeback."

            let start_ref = start_side.insert(DrainSide {
                edge: DrainEdge {
                    outermost_removed: end_side.cursor.slice.copy_handle(),
                    slice_pos: slice_range.start,
                    writeback: None, // <- we'll set this below if we need to
                },
                cursor: DrainSideCursor {
                    slice: end_side.cursor.slice.copy_handle(),
                    slice_pos: slice_range.start,
                },
            });

            // equal to: slice_range.start < self.range.start
            if slice_range.start != range.start {
                // [diagram note: area marked as '~' will be removed]
                //
                // Removal not contained within the slice:
                //
                //   slice:  |----------/~~~~~~~~~~~~~~~~~~~~~~~~~~|
                //              self.range.start                   |
                //                                         slice_range.end
                //
                // Removal contained within the slice:
                //
                //   slice:  |----------/~~~~~~~~~~~~~~/-----------|
                //              self.range.start       |           |
                //                              self.range.end     |
                //                                         slice_range.end
                //
                // With the slice ranging between teh two '|'s, we can see that the split position
                // to use *after* any necessary right-hand split has been done will be equal to
                // `range.start - slice_range.start`, because `range.start` is the maximum of
                // `slice_pos` and `self.range.start` (and `slice_pos < self.range.start`).
                let splitpoint = range.start;
                let rhs = value.split_at(splitpoint);
                let split_val = mem::replace(&mut value, rhs);
                start_ref.edge.writeback = Some((splitpoint, split_val));
            }
        } else if cfg!(debug_assertions) && slice_range.start != range.start {
            // Above condition implies start_side.is_some() and range.start < self.range.start,
            // which shouldn't be true.
            panic_internal_error_or_bad_index::<I>();
        }

        // All done :)
        Some((range, value))
    }
}

#[cfg(not(feature = "nightly"))]
impl<'t, C, I, S, P, const M: usize> Drop for Drain<'t, C, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn drop(&mut self) {
        // SAFETY: `do_drop` requires that the function is called as the the destructor
        unsafe { self.do_drop() }
    }
}

// Note: `I` cannot dangle because we fix the tree in the destructor, which requires operations
// using `I`.
#[cfg(feature = "nightly")]
unsafe impl<'t, #[may_dangle] C, I, #[may_dangle] S, P, const M: usize> Drop
    for Drain<'t, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn drop(&mut self) {
        // SAFETY: `do_drop` requires that the function is called as the the destructor
        unsafe { self.do_drop() }
    }
}

impl<'t, C, I, S, P: RleTreeConfig<I, S, M>, const M: usize> Drain<'t, C, I, S, P, M> {
    /// Implementation of `Drain`'s destructor, factored out so that we can have different
    /// attributes for `#[cfg(feature = "nightly")]`
    ///
    /// ## Safety
    ///
    /// `do_drop` can only be called as the implementation of `Drain`'s destructor
    unsafe fn do_drop(&mut self) {
        // SAFETY: Guaranteed by the caller that this is called only during the destructor, so
        // this is called exactly once. We don't drop `self.state` elsewhere in this function, so
        // we're good to `take` from the ManuallyDrop.
        let state = unsafe { ManuallyDrop::take(&mut self.state) };

        match state {
            None => (),
            Some(Err(cursor)) => drop(cursor),
            Some(Ok(_)) => todo!(),
        }

        // Release the mutable borrow initially acquired in `Drain::new`
        if let Some(root) = self.root.as_ref() {
            root.refs_store.release_mutable();
        }

        // SAFETY: We're not allowed to do anything with `tree_ref` until we're absolutely done
        // with `state`, which we are.
        *self.tree_ref = self.root.take();
    }
}
