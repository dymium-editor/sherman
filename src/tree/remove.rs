//! Internal removal implementation
//!
//! Unlike e.g., [`mod insert`], there's no singular public method that uses the types/methods from
//! this module. Instead, there's a handful of separate functions that need to perform removal.
//! Because of this, we don't make any assumptions about whether the total size of the tree
//! increased or decreased.
//!
//! [`mod insert`]: super::insert

use crate::param::{RleTreeConfig, SupportsInsert};
use crate::{Cursor, Index, PathComponent, RleTree, Slice};
use std::ops::{ControlFlow, Range};

use super::node::{borrow, ty, ChildOrKey, NodeHandle, NodePtr, SliceHandle, Type};
use super::{shift_keys_auto, ShiftKeys};

impl<I, S, P, const M: usize> RleTree<I, S, P, M>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    /// Performs the removal of a value from a node where the slice's value has already been
    /// removed (i.e., it is a hole)
    ///
    /// For more information on the usage of `move_notify`, refer to the
    /// documentation for [`TraverseUpdate`]. `map_state` is called after each `DeleteStepState` is
    /// created or modified. It is called exactly once at each level of the tree (deletion is
    /// always guaranteed to change a leaf node).
    pub(super) fn remove_hole<'t, Ty: ty::TypeHint, C: Cursor>(
        store: &mut P::SliceRefStore,
        slice: SliceHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
        move_notify: impl FnMut(TraverseUpdate<I, S, P, M>),
        map_state: impl FnMut(DeleteStepState<'t, C, I, S, P, M>) -> DeleteStepState<'t, C, I, S, P, M>,
    ) -> DeleteStepState<'t, C, I, S, P, M> {
        assert!(slice.is_hole());

        // We dispatch to the two different functions here because we can directly inline
        // `remove_hole_{leaf,internal}` when the slice's type is already known.
        match slice.into_typed() {
            Type::Leaf(h) => Self::remove_hole_leaf(store, h, move_notify, map_state),
            Type::Internal(h) => Self::remove_hole_internal(store, h, move_notify, map_state),
        }
    }

    /// (*Internal*) [`remove_hole`], but specialized for `Leaf` handles
    ///
    /// **General strategy**: we perform the removal directly in the leaf, and if it's underfull we
    /// carry the change up the tree, and then update the tree appropriately to account for the
    /// change. `move_notify` and `map_state` are called as specified in [`remove_hole`].
    ///
    /// [`remove_hole`]: Self::remove_hole
    fn remove_hole_leaf<'t, C: Cursor>(
        store: &mut P::SliceRefStore,
        slice: SliceHandle<ty::Leaf, borrow::Mut<'t>, I, S, P, M>,
        move_notify: impl FnMut(TraverseUpdate<I, S, P, M>),
        map_state: impl FnMut(DeleteStepState<'t, C, I, S, P, M>) -> DeleteStepState<'t, C, I, S, P, M>,
    ) -> DeleteStepState<'t, C, I, S, P, M> {
        todo!()
    }

    /// (*Internal*) [`remove_hole`](Self::remove_hole), but specialized for `Internal` handles
    fn remove_hole_internal<'t, C: Cursor>(
        store: &mut P::SliceRefStore,
        slice: SliceHandle<ty::Internal, borrow::Mut<'t>, I, S, P, M>,
        move_notify: impl FnMut(TraverseUpdate<I, S, P, M>),
        map_state: impl FnMut(DeleteStepState<'t, C, I, S, P, M>) -> DeleteStepState<'t, C, I, S, P, M>,
    ) -> DeleteStepState<'t, C, I, S, P, M> {
        todo!()
    }
}

/// Update type passed to tree restructuring callbacks
///
/// This type exists so that things like slice references from `insert_ref` can be appropriately
/// updated by algorithms that don't need to be aware of their existence.
///
/// Note: The slice references stored in the tree's [`SliceRefStore`] are already updated by the
/// methods on the [`NodeHandle`]s and [`SliceHandle`]s; they do not need to be accounted for here.
///
/// [`SliceRefStore`]: super::slice_ref::SliceRefStore
pub(super) enum TraverseUpdate<I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    /// Records that a range of slices have moved -- either within or between nodes
    ///
    /// It is *guaranteed* that `new_ptr` is a valid node in the tree, that `old_range` and
    /// `new_range` have the same length, and that the slices `(*new_ptr)[new_range]` are all
    /// initialized. It is allowed for `old_ptr` to have been deallocated by the time the callback
    /// receives the `Move` update.
    Move {
        old_ptr: NodePtr<I, S, P, M>,
        old_range: Range<u8>,
        new_ptr: NodePtr<I, S, P, M>,
        new_range: Range<u8>,
    },
    /// Marks the slice as being temporarily removed from the tree
    Remove { ptr: NodePtr<I, S, P, M>, idx: u8 },
    /// Returns a previously removed (via `Remove`) slice to a new position in the tree
    ///
    /// It is *guaranteed* by the caller of a callback using `TraverseUpdate` that `PutBack` will
    /// be provided exactly once after each `Remove`, without more than one `Remove` before the
    /// appropriate `PutBack`.
    PutBack { ptr: NodePtr<I, S, P, M>, idx: u8 },
}

/// Type returned by `map_state` callbacks for [`remove_hole`](RleTree::remove_hole).
///
/// Callbacks returning this type can be expected to return
pub(super) type MapResult<'t, C, I, S, P, const M: usize> =
    ControlFlow<C, DeleteStepState<'t, C, I, S, P, M>>;

/// End result of performing a deletion
pub(super) enum PostDeleteResult {
    /// The root node should be freed and replaced with its first child, if it has one
    ///
    /// If `DeleteRoot` is returned, it is *guaranteed* that the length of the root node is zero.
    /// It may or may not be an internal node -- if it is a leaf, the entire `Root` must be
    /// dropped.
    DeleteRoot,
    /// The deletion was successfully performed, with nothing interesting to report.
    Nominal,
}

/// The state during deletion after performing a single step
pub(super) enum DeleteStepState<'t, C, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    /// The deletion was successfully performed. The remaining processing is simply to propagate
    /// the change in size up the tree
    Propagate(OpUpdateState<'t, C, I, S, P, M>),
    /// The deletion was performed in the node from [`OpUpdateState`], but it left the node
    /// underfull (i.e., with a length less than `M` for non-root nodes, or a length of zero for
    /// root nodes)
    ///
    /// An underfull root node will be marked with the [`DeleteRoot`] result for [`Done`] instead,
    /// as it must be removed and replaced with its first child, if it has one.
    Underfull(DeleteUnderfull<'t, C, I, S, P, M>),
    /// Deletion was successfully completed, with `result` indicating whether further changes to
    /// the tree are necessary
    Done { cursor: C, result: PostDeleteResult },
}

/// State recording how to propagate an update to the size of the tree as we traverse upwards
pub(super) struct OpUpdateState<'t, C, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    /// The key or child containing the change as we're traversing upwards, alongside the typed
    /// node containing it
    ///
    /// For keys that are positioned after this child or key (i.e. at a greater index), their
    /// positions within the node can be updated with:
    ///
    /// ```text
    /// |- ... -------------- key --------------|   (original position of a key after `child_or_key`)
    /// |- ... --- pos ---|                         (node-relative `child_or_key` position)
    ///                   |- key.sub_left(pos) -|
    ///                   |---- new_size ----|
    ///                                      |- key.sub_left(pos) -|  (same size as above, shifted)
    ///                   |- key.sub_left(pos).add_left(new_size) -|
    /// |- ... - key.sub_left(pos).add_left(new_size).add_left(pos) -|  (new key position)
    /// ```
    ///
    /// The final result, `key.sub_left(pos).add_left(new_size).add_left(pos)` can largely be
    /// tracked by a running total of the distance from each key back to the changed child/key,
    /// adding the distance between keys each time. So the value is then
    /// `key.sub_left(pos).add_left(sum)`.
    ///
    /// This careful dance is only there because we have to make sure that we uphold the guarantees
    /// we provide for directional arithmetic -- we can't just `add_left(size)` because that's not
    /// where the slice is.
    ///
    /// If the usage around this is written in *just* the right way, the compiler should realize
    /// what's going on and optimize it to (approximately) `key += size` or `key -= size`,
    /// depending on the operation.
    child_or_key: ChildOrKey<
        // Child: must be internal
        (u8, NodeHandle<ty::Internal, borrow::Mut<'t>, I, S, P, M>),
        // Key: unknown type
        (u8, NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>),
    >,

    /// Optional override in case the start position of the size change is not equal to the start
    /// position given by `child_or_key`
    override_pos: Option<I>,

    /// The old size of the child or key containing the change
    old_size: I,
    /// The new size of the child or key containing the change. This is provided separately so that
    /// it can be updated *after* the original size has been recorded
    new_size: I,

    /// Cursor representing the path to the deepest node containing the change
    ///
    /// This cursor is often empty.
    ///
    /// ---
    ///
    /// In theory, it's entirely possible to use `inserted_slice` to build a cursor after the fact
    /// and get the exact path to the insertion. In practice, this involves doing a second upwards
    /// tree traversal, so we opt for an imperfect "good enough" solution that minimizes additional
    /// costs.
    partial_cursor: C,
}

/// State recording update propagation for a node that's underfull
pub(super) struct DeleteUnderfull<'t, C, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    child_or_key: ChildOrKey<
        (u8, NodeHandle<ty::Internal, borrow::Mut<'t>, I, S, P, M>),
        (u8, NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>),
    >,
    override_pos: Option<I>,
    old_size: I,
    new_size: I,
    partial_cursor: C,
}

impl<'t, C, I, S, P, const M: usize> DeleteStepState<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    P: RleTreeConfig<I, S, M>,
{
}

impl<'t, C, I, S, P, const M: usize> OpUpdateState<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    /// Performs an upward step, propagating the change in size up the tree
    ///
    /// If the propagation reaches the root node, then `Ok(C)` is returned with the completed
    /// cursor. Otherwise, the parent node is returned for the change to propagate through.
    pub(super) fn upward_step_propagate(mut self) -> Result<C, Self> {
        let (mut node, first_key_after, maybe_child_idx, pos) = match self.child_or_key {
            ChildOrKey::Key((k_idx, node)) => {
                let pos = self
                    .override_pos
                    // SAFETY: the documentation for `OpUpdateState` guarantees that `k_idx` is
                    // within bounds
                    .unwrap_or_else(|| unsafe { node.leaf().key_pos(k_idx) });

                (node, k_idx + 1, None, pos)
            }
            ChildOrKey::Child((c_idx, node)) => {
                let pos = self.override_pos.unwrap_or_else(|| {
                    let next_key_pos = node
                        .leaf()
                        .try_key_pos(c_idx)
                        .unwrap_or_else(|| node.leaf().subtree_size());
                    next_key_pos.sub_left(self.old_size)
                });

                (node.erase_type(), c_idx, Some(c_idx), pos)
            }
        };

        let old_size = node.leaf().subtree_size();

        // SAFETY: `shift_keys` only requires that `from` is less than or equal to
        // `node.leaf().len()`. This can be verified by looking at the code above; `from` is either
        // set to `k_idx + 1` or to `c_idx`, both of which exactly meet the required bound.
        unsafe {
            shift_keys_auto(
                &mut node,
                ShiftKeys {
                    from: first_key_after,
                    pos,
                    old_size: self.old_size,
                    new_size: self.new_size,
                },
            );
        }

        let new_size = node.leaf().subtree_size();

        // Update the cursor
        //
        // Technically speaking, this relies on this node having `child_or_key =
        // ChildOrKey::Child(_)` as long as the cursor is non-empty, but that should always be
        // true.
        if let Some(child_idx) = maybe_child_idx {
            self.partial_cursor
                .prepend_to_path(PathComponent { child_idx });
        }

        // Update `self` to the parent, if there is one. Otherwise, this is the root node so we
        // should return `Ok` with the cursor.
        match node.into_parent().ok() {
            None => Ok(self.partial_cursor),
            Some((parent_handle, c_idx)) => Err(OpUpdateState {
                child_or_key: ChildOrKey::Child((c_idx, parent_handle)),
                override_pos: None,
                old_size,
                new_size,
                partial_cursor: self.partial_cursor,
            }),
        }
    }
}
