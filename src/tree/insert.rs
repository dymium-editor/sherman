//! Internal insertion implementation

use crate::cursor::CursorBuilder;
use crate::param::{BorrowState, RleTreeConfig, SliceRefStore as _, SupportsInsert};
use crate::{Index, PathComponent, RleTree, Slice};
use std::cell::Cell;
use std::cmp::Ordering;
use std::ops::ControlFlow;

use super::fix::{
    self, MapOpUpdateState, MapState, OpUpdateState, TraverseUpdate, UpdatesCallback,
};
use super::fix::{shift_keys_auto, shift_keys_decrease, shift_keys_increase, ShiftKeys};
use super::node::{self, borrow, ty, ChildOrKey, NodeHandle, SliceHandle, Type};
use super::{search_step, NodeBox, SharedNodeBox, Side, SliceSize};

#[cfg(test)]
use crate::MaybeDebug;
#[cfg(test)]
use std::fmt::{self, Debug, Formatter};

///////////////////////////////////////////////
// High-level entrypoint (lower-level below) //
///////////////////////////////////////////////

impl<I, S, P, const M: usize> RleTree<I, S, P, M>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    /// (*Internal*) Shared "outer core" implementation of the public insertion APIs
    ///
    /// ## Safety
    ///
    /// The returned [`SliceHandle`] is `Some` *if and only if* type `P` is [`AllowSliceRefs`].
    /// This may be relied upon in unsafe code.
    ///
    /// [`AllowSliceRefs`]: crate::param::AllowSliceRefs
    #[track_caller] // so that panics are on the function that was called.
    pub(super) fn insert_internal<'t, 'c>(
        &'t mut self,
        cursor_builder: CursorBuilder<'c>,
        cursor_path: &mut dyn Iterator<Item = PathComponent>,
        idx: I,
        slice: S,
        size: I,
    ) -> Option<SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>> {
        // All non-trivial operations on this tree are a pain in the ass, and insertion is no
        // exception. We'll try to have comments explaining some of the more hairy stuff.
        //
        // This function is mostly high-level edge-case handling; all of the tree traversal &
        // shifting stuff is done in other functions. The other lower-level functions are also
        // exposed so that algorithms that - in their own edge cases - require insertion can use
        // them.

        if idx > self.size() {
            panic!(
                "cannot insert new slice, index {idx:?} out of bounds for size {:?}",
                self.size()
            );
        } else if size <= I::ZERO {
            panic!("cannot insert new slice, size {size:?} is not greater than zero");
        }

        let root = match &mut self.root {
            Some(r) => r,
            // If there's no root, then we don't need to worry about any shared references. We can just
            // create a fresh tree and return:
            None => {
                *self = Self::new(slice, size);
                if !P::SLICE_REFS {
                    return None;
                }

                let root_ptr = &self.root.as_mut().unwrap().handle;
                // SAFETY: We just created the root, so there can't be any conflicts to worry
                // about. The returned handle is already known to have the same lifetime as `self`.
                // `key_idx = 1` is also safe to pass, because the root is guaranteed to have a
                // single entry.
                let value_handle =
                    unsafe { root_ptr.borrow().into_slice_handle(0).clone_slice_ref() };
                return Some(value_handle);
            }
        };

        // Because we're modifying an existing tree, we need to acquire borrowing rights:
        if let Err(conflict) = root.refs_store.acquire_mutable() {
            panic!("{conflict}");
        }

        // If COW is enabled, we need to make sure we have a fresh reference to each part of the
        // tree, as we're going down. This is handled by each successive call to `into_child`
        // for the mutable handles, so all we *actually* have to do is make sure that the
        // uniqueness starts at the root.
        let owned_root = root.handle.make_unique();

        // With clone on write, we often have unreliable parent pointers. One piece of this is that
        // drain can cause our root node to have a parent pointer.
        if P::COW {
            owned_root.borrow_mut().remove_parent();
        }

        /////////////////////////////////////
        // Body of the insertion algorithm //
        /////////////////////////////////////
        //
        // This is where the juicy bits happen
        //
        // (ok not *actually* here, but kinda nearby)
        //
        let insertion_point = Self::find_insertion_point(owned_root.borrow_mut(), cursor_path, idx);

        // Inseriton algorithm, Part 2: do the thing
        let (post_insert_result, mut insertion) = if P::SLICE_REFS {
            let mut insertion = None;
            let mut u = unsafe { TraverseUpdate::apply_to_ref(&mut insertion) };
            // SAFETY: `do_insert_full` requires that, if `insertion_point` is a `LeafInsert`, its
            // `new_k_idx` is less than or equal to its `node`'s length, which is guaranteed by
            // `find_insertion_point`.
            let result = unsafe {
                Self::do_insert_full(
                    &mut root.refs_store,
                    cursor_builder,
                    &mut root.len,
                    insertion_point,
                    slice,
                    size,
                    &mut u,
                )
            };
            drop(u); // `u` borrows `insertion`, and would otherwise drop *after* we yield here
            (result, insertion)
        } else {
            let mut u = TraverseUpdate::do_nothing;
            // SAFETY: `do_insert_full` requires that, `insertion_point` is a `LeafInsert`, its
            // `new_k_idx` is less than or equal to its `node`'s length, which is guaranteed by
            // `find_insertion_point`.
            let result = unsafe {
                Self::do_insert_full(
                    &mut root.refs_store,
                    cursor_builder,
                    &mut root.len,
                    insertion_point,
                    slice,
                    size,
                    &mut u,
                )
            };
            (result, None)
        };

        match post_insert_result {
            PostInsertResult::Done => (),
            PostInsertResult::NewRoot { lhs, key, key_size, rhs } => {
                // we need to get rid of `lhs` first, so that we've released the borrow when we
                // access the root
                let lhs_ptr = lhs.ptr();
                debug_assert!(owned_root.ptr() == lhs_ptr);

                // SAFETY: `make_new_parent` requires that `owned_root` and `rhs` not have any
                // parent already and are at the same height, which is guaranteed by `NewRoot`
                unsafe {
                    owned_root.make_new_parent(&mut root.refs_store, key, key_size, rhs.take());
                }

                if P::SLICE_REFS && insertion.is_none() {
                    insertion =
                        Some(unsafe { root.handle.borrow().into_slice_handle(0).clone_slice_ref() })
                }
            }
        };

        if P::SLICE_REFS && insertion.is_none() {
            panic!("internal error: final insertion handle is `None` but P is `AllowSliceRefs`");
        }

        // Release the mutable borrow we originally acquired. It doesn't matter whether we do this
        // before or after replacing from a shallow copy because shallow copies only exist with COW
        // functionality, and explicit borrow state only exists with slice references (which is
        // incompatible with COW stuff)
        match self.root.as_mut() {
            Some(r) => {
                // Make sure that the `SliceRefStore`'s root is up to date:
                //
                // SAFETY: we're only using `copied_handle` for the `SliceRefStore`
                r.refs_store.set_root(Some(unsafe { r.handle.clone_root_for_refs_store() }));
                r.refs_store.release_mutable();
            }
            // SAFETY: We originally checekd that `self.root` is `Some(_)` up above, and while
            // insertion *can* remove up to one value, that value can never be the only value in
            // the tree -- so `self.root` must still be `Some(_)`
            None => unsafe { weak_unreachable!() },
        }

        insertion
    }
}

/////////////////////////////////////
// Helper / State type definitions //
/////////////////////////////////////

/// (*Internal*) The result of a successful downward search for an insertion point, returned by
/// [`RleTree::find_insertion_point`]
type InsertionPoint<'t, I, S, P, const M: usize> =
    ChildOrKey<EdgeInsert<'t, I, S, P, M>, SplitKeyInsert<'t, I, S, P, M>>;

/// (*Internal*) Component of an [`InsertionPoint`] specific to inserting *between* keys in a
/// leaf
struct EdgeInsert<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    /// The leaf node that the value should be inserted into
    node: NodeHandle<ty::Leaf, borrow::Mut<'t>, I, S, P, M>,
    /// The keys to the immediate left and right of the insertion point
    adjacent_keys: AdjacentKeys<'t, I, S, P, M>,
    /// The index that the newly inserted value should have
    ///
    /// `new_k_idx` is guaranteed to be no greater than `containing_node.len_keys()`, but *may* be
    /// equal to the maximum number of keys; i.e. one more than can fit. If this is the case, the
    /// leaf node will need to be split (or redistributed to its neighbors).
    new_k_idx: u8,
}

/// (*Internal*) Component of an [`InsertionPoint`] for when an insertion point is found in the
/// middle of an existing key
///
/// In this case, the key must be split in order to perform the insertion.
struct SplitKeyInsert<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    /// A handle on the key to insert the value into
    handle: SliceHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
    /// The index *within* the key to perform the insertion
    pos_in_key: I,
}

/// (*Internal*) The keys to either side of an insertion point
///
/// Both of the keys will be `None` if [`S::MAY_JOIN`] is false, and they will be discarded if the
/// insertion point is found to be in the middle of an existing key (instead of between two keys).
///
/// However, storing these as we search downwards is still useful in the cases where the insertion
/// point is at either end of a leaf node's keys -- it prevents an additional upwards search to
/// find the next left or next right key.
///
/// [`S::MAY_JOIN`]: Slice::MAY_JOIN
struct AdjacentKeys<'b, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    lhs: Option<SliceHandle<ty::Unknown, borrow::Mut<'b>, I, S, P, M>>,
    rhs: Option<SliceHandle<ty::Unknown, borrow::Mut<'b>, I, S, P, M>>,
}

/// (*Internal*) Helper struct to carry the information we use when an insertion overflowed a node,
/// and we need to "bubble" the new midpoint and right-hand side up to the parent
struct BubbledInsertState<'t, 'c, I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S, M>,
{
    /// The existing node that `rhs` was split off from
    lhs: NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
    /// The key between `lhs` and `rhs`
    key: node::Key<I, S, P, M>,
    /// Size of slice in `key`
    key_size: I,
    /// The newly-created node, to the right of `key`
    ///
    /// This will be inserted in `lhs`'s parent as its sibling to the right, with `key` in between
    /// them.
    ///
    /// The height of `rhs` is *always* equal to the height of `lhs`.
    rhs: NodeBox<ty::Unknown, I, S, P, M>,

    /// Total size of the node that was split into `lhs` and `rhs`, before we inserted anything or
    /// split it
    old_size: I,

    /// If provided, `shift_lhs` overrides the position and size of `lhs` during size calculations
    /// for [`do_upward_step`]
    ///
    /// Setting `shift_lhs` allows the default size calculation of the left-hand node's size
    /// (usually done using `self.old_size`) to be overridden
    ///
    /// [`do_upward_step`]: Self::do_upward_step
    shift_lhs: Option<BubbleLhs<I>>,

    /// If the slice we're building a cursor to a value that has already been inserted, this is a
    /// marker for which one of `lhs` or `rhs` it's in
    ///
    /// Without tampering from state mapping, this value is `None` exactly when `inserted` is
    /// false. State mapping allows us to redirect the cursor to a different value instead.
    cursor_side: Option<Side>,

    /// Marker for whether the original insertion has been made. If false, the next placement of
    /// the current `key` will be made with an [`Insertion`] update, instead of [`PutBack`]
    ///
    /// [`Insertion`]: TraverseUpdate::Insertion
    /// [`PutBack`]: TraverseUpdate::PutBack
    inserted: bool,

    /// Cursor representing the path to the insertion
    ///
    /// The cursor does not not yet include the path segment through `lhs` or `rhs`; we'd need to
    /// have already taken into account the shifts from inserting `key` at this level.
    cursor_builder: CursorBuilder<'c>,
}

/// (*Internal*) The end result of running the "full" insertion algorithm, minus the cleanup that
/// happens afterwards
///
/// This type is primarily returned by [`RleTree::do_insert_full`].
enum PostInsertResult<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    /// No more changes to the tree are necessary. The [`Cursor`](crate::Cursor) has been fully
    /// created
    Done,
    /// A new root node is required. The [`Cursor`](crate::Cursor) has already taken into account
    /// the locations that `lhs` and `rhs` will have.
    NewRoot {
        /// Current root node, which will become the left-hand child of the new root
        lhs: NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
        /// Key to be inserted between `lhs` and `rhs`
        key: node::Key<I, S, P, M>,
        /// Size of `key`
        key_size: I,
        /// A new node, which will become the right-hand child of the new root
        rhs: NodeBox<ty::Unknown, I, S, P, M>,
    },
}

/////////////////////////////
// `Debug` implementations //
/////////////////////////////

#[cfg(test)]
impl<'t, I, S, P, const M: usize> Debug for EdgeInsert<'t, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("EdgeInsert")
            .field("node", &self.node)
            .field("adjacent_keys", &self.adjacent_keys)
            .field("new_k_idx", &self.new_k_idx)
            .finish()
    }
}

#[cfg(test)]
impl<'t, I, S, P, const M: usize> Debug for SplitKeyInsert<'t, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("SplitKeyInsert")
            .field("handle", &self.handle)
            .field("pos_in_key", self.pos_in_key.fallible_debug())
            .finish()
    }
}

#[cfg(test)]
impl<'b, I, S, P, const M: usize> Debug for AdjacentKeys<'b, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("AdjacentKeys")
            .field("lhs", &self.lhs)
            .field("rhs", &self.rhs)
            .finish()
    }
}

#[cfg(test)]
impl<'t, 'c, I, S, P, const M: usize> Debug for BubbledInsertState<'t, 'c, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("BubbledInsertState")
            .field("lhs", &self.lhs)
            .field("key", &self.key)
            .field("key_size", self.key_size.fallible_debug())
            .field("rhs", &self.rhs)
            .field("old_size", &self.old_size.fallible_debug())
            .field("cursor_side", &self.cursor_side)
            .field("inserted", &self.inserted)
            .finish()
    }
}

#[cfg(test)]
impl<'t, I, S, P, const M: usize> Debug for PostInsertResult<'t, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            PostInsertResult::Done => f.write_str("Done"),
            PostInsertResult::NewRoot { lhs, key, key_size, rhs } => f
                .debug_struct("NewRoot")
                .field("lhs", lhs)
                .field("key", key)
                .field("key_size", key_size.fallible_debug())
                .field("rhs", rhs)
                .finish(),
        }
    }
}

/////////////////////////////////
// Private(ish) implementation //
/////////////////////////////////

/// Single-use flag indicating whether [`insert_rhs`] should produce a cursor to `right_of` or `fst`
///
/// [`insert_rhs`]: RleTree::insert_rhs
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CursorMode {
    Lhs,
    Fst,
}

#[allow(rustdoc::private_intra_doc_links)]
/// Internal-only helper functions for implementing [`insert_internal`](RleTree::insert_internal)
/// and machinery for other algorithms that require insertion.
impl<I, S, P, const M: usize> RleTree<I, S, P, M>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    ///////////////////////////////////////////////////////
    // High-level functions                              //
    // ---                                               //
    // Purely serves to dispatch based on enum variants. //
    ///////////////////////////////////////////////////////

    /// (*Internal*) Performs an insertion at the given `insertion_point`, returning the result of
    /// completing the insertion
    ///
    /// ## Safety
    ///
    /// If `insertion_point` is a `Child`, then the [`EdgeInsert`]'s `new_k_idx` must be less than
    /// or equal to its `node`'s length.
    unsafe fn do_insert_full<'t, 'c>(
        store: &mut P::SliceRefStore,
        cb: CursorBuilder<'c>,
        tree_len: &mut usize,
        insertion_point: InsertionPoint<'t, I, S, P, M>,
        slice: S,
        size: I,
        updates_callback: UpdatesCallback<I, S, P, M>,
    ) -> PostInsertResult<'t, I, S, P, M> {
        match insertion_point {
            // SAFETY: `do_insert_full_edge` requires that `edge_ins.new_k_idx` is less than or
            // equal to `edge_ins.node.leaf().len()`, which is guaranteed by the safety
            // requirements for this function.
            #[rustfmt::skip]
            ChildOrKey::Child(edge_ins) => unsafe {
                Self::do_insert_full_edge(
                    store, cb, tree_len,
                    edge_ins,
                    slice, size,
                    updates_callback,
                )
            },
            #[rustfmt::skip]
            ChildOrKey::Key(split_ins) => Self::do_insert_full_split_key(
                store, cb, tree_len,
                split_ins,
                slice, size,
                updates_callback,
            ),
        }
    }

    /// (*Internal*) Performs an insertion between two keys in a leaf node, as specified by the
    /// [`EdgeInsert`].
    ///
    /// Returns the result of completing the insertion.
    ///
    /// ## Safety
    ///
    /// This function requires that `insert.new_k_idx` is less than or equal to
    /// `insert.node.leaf().len()`.
    unsafe fn do_insert_full_edge<'t, 'c>(
        store: &mut P::SliceRefStore,
        mut cursor_builder: CursorBuilder<'c>,
        tree_len: &mut usize,
        insert: EdgeInsert<'t, I, S, P, M>,
        slice: S,
        size: I,
        updates_callback: UpdatesCallback<I, S, P, M>,
    ) -> PostInsertResult<'t, I, S, P, M> {
        unsafe { weak_assert!(insert.new_k_idx <= insert.node.leaf().len()) }

        match Self::do_insert_full_edge_try_join(
            store,
            cursor_builder.reborrow(),
            tree_len,
            slice,
            size,
            insert.adjacent_keys,
            updates_callback,
        ) {
            // If we successfully joined, we don't need to update `len`, because the number of
            // values stayed the same.
            Ok(update_state) => update_state,
            // SAFETY: `do_insert_full_edge_no_join` requires that `insert.new_k_idx` is less than
            // or equal to `insert.node.leaf().len()`, which is guaranteed by the
            #[rustfmt::skip]
            Err(slice) => unsafe {
                *tree_len += 1; // No joining means +1 value

                Self::do_insert_full_edge_no_join(
                    store,
                    cursor_builder,
                    insert.node,
                    insert.new_k_idx, None,
                    SliceSize { slice, size }, None,
                    updates_callback,
                    BubbledInsertState::identity(),
                )
            },
        }
    }

    /// (*Internal*) Performs an insertion in the middle of a key by splitting it, as specified by
    /// the [`SplitKeyInsert`].
    ///
    /// Returns the result of completing the insertion.
    fn do_insert_full_split_key<'t, 'c>(
        store: &mut P::SliceRefStore,
        cb: CursorBuilder<'c>,
        tree_len: &mut usize,
        mut info: SplitKeyInsert<'t, I, S, P, M>,
        slice: S,
        slice_size: I,
        updates_callback: UpdatesCallback<I, S, P, M>,
    ) -> PostInsertResult<'t, I, S, P, M> {
        let key_size = info.handle.slice_size();

        let rhs = info.handle.with_slice(|slice| slice.split_at(info.pos_in_key));
        let rhs_size = key_size.sub_left(info.pos_in_key);

        let (mut lhs_handle, lhs_size) = (info.handle, info.pos_in_key);

        // Fast path: no need to try joining.
        if !S::MAY_JOIN {
            *tree_len += 2;
            let (mid, mid_size) = (slice, slice_size);
            let fst = SliceSize::new(mid, mid_size);
            let snd = SliceSize::new(rhs, rhs_size);
            return Self::insert_rhs(
                store,
                cb,
                lhs_handle,
                lhs_size,
                CursorMode::Fst,
                fst,
                Some(snd),
                updates_callback,
            );
        }

        // Slow path: try to join with both sides.
        //
        // First, try the left-hand side:

        let lhs_slice = lhs_handle.take_value_and_leave_hole();
        let mut joined_with_lhs = false;

        let (mid, mid_size) = match lhs_slice.try_join(slice) {
            Ok(new) => {
                joined_with_lhs = true;
                (new, lhs_size.add_right(slice_size))
            }
            Err((l, s)) => {
                lhs_handle.fill_hole(l);
                (s, slice_size)
            }
        };

        match mid.try_join(rhs) {
            // "nice" case -- everything got joined back to where it should be:
            Ok(new) if joined_with_lhs => {
                lhs_handle.fill_hole(new);

                let inserted_slice = &lhs_handle;
                updates_callback(TraverseUpdate::insert(inserted_slice));

                let post_op_state = OpUpdateState {
                    child_or_key: ChildOrKey::Key((lhs_handle.idx, lhs_handle.node)),
                    override_pos: None,
                    old_size: lhs_size.add_right(rhs_size),
                    new_size: mid_size.add_right(rhs_size),
                    cursor_builder: cb,
                };

                let mapper = OpUpdateState::identity();
                post_op_state.finish(mapper, None);
                PostInsertResult::Done
            }
            // Couldn't join with lhs, but slice + rhs joined. Still need to insert those:
            Ok(new) => {
                *tree_len += 1;
                let size = mid_size.add_right(rhs_size);
                Self::insert_rhs(
                    store,
                    cb,
                    lhs_handle,
                    lhs_size,
                    CursorMode::Fst,
                    SliceSize::new(new, size),
                    None,
                    updates_callback,
                )
            }
            // Couldn't join with rhs, but did join with lhs. Still need to insert rhs:
            Err((l, r)) if joined_with_lhs => {
                *tree_len += 1;
                lhs_handle.fill_hole(l);
                let fst = SliceSize::new(r, rhs_size);
                updates_callback(TraverseUpdate::insert(&lhs_handle));
                let mut new_callback = TraverseUpdate::skip_insertions(updates_callback);
                Self::insert_rhs(
                    store,
                    cb,
                    lhs_handle,
                    mid_size,
                    CursorMode::Lhs,
                    fst,
                    None,
                    &mut new_callback,
                )
            }
            // Couldn't join with either lhs or rhs. Need to insert both original slice AND new rhs
            Err((m, r)) => {
                *tree_len += 2;
                let fst = SliceSize::new(m, mid_size);
                let snd = SliceSize::new(r, rhs_size);

                Self::insert_rhs(
                    store,
                    cb,
                    lhs_handle,
                    lhs_size,
                    CursorMode::Fst,
                    fst,
                    Some(snd),
                    updates_callback,
                )
            }
        }
    }

    ////////////////////////////////////////////////////////////
    // Mid-level functions                                    //
    // ---                                                    //
    // Joining logic to bridge the gap between high-level and //
    // low-level functions.                                   //
    ////////////////////////////////////////////////////////////

    /// (*Internal*) Tries to insert the slice by joining with either adjacent key, returning
    /// `Err(slice)` joining fails
    ///
    /// If joining succeeds, the insertion is processed upwards until all structural changes to the
    /// tree are resolved, and the [`PostInsertResult`] is returned. Otherwise, `Err(slice)` is
    /// returned.
    ///
    /// We return `Err` instead of handling that case as well because there's many code paths that
    /// can lead to being unable to join to anything, so it's easier to
    fn do_insert_full_edge_try_join<'t, 'c>(
        store: &mut P::SliceRefStore,
        cursor_builder: CursorBuilder<'c>,
        tree_len: &mut usize,
        slice: S,
        slice_size: I,
        adjacent_keys: AdjacentKeys<'t, I, S, P, M>,
        updates_callback: UpdatesCallback<I, S, P, M>,
    ) -> Result<PostInsertResult<'t, I, S, P, M>, S> {
        // Broadly speaking, there's four cases here:
        //  1. `slice` doesn't join with either of `adjacent_keys` -- returns Err(slice)
        //  2. `slice` only joins with `adjacent_keys.lhs`
        //  3. `slice` only joins with `adjacent_keys.rhs`
        //  4. `slice` joins with both keys
        // The logic for cases 2 & 3 are essentially the same, so our labeled block below produces
        // a value to be handled for those cases and returns for 1 & 4.

        // Fast path.
        if !S::MAY_JOIN {
            return Err(slice);
        }

        let lhs_with_size = adjacent_keys.lhs.map(|h| (h.slice_size(), h));
        let rhs_with_size = adjacent_keys.rhs.map(|h| (h.slice_size(), h));

        // TODO: This code could be more concise, probably by merging `None` with "couldn't join".
        // That *may* be slightly more difficult to reason about, though.
        let (handle, old_size, new_size) = 'single_join: {
            match (lhs_with_size, rhs_with_size) {
                // Nothing to do.
                (None, None) => unreachable!(
                    "edges have at least one key on either side because the tree is not empty"
                ),

                // Try to join with `lhs`. If that succeeds, break 'single_join. Otherwise, return
                // Err(slice).
                (Some((lhs_size, mut lhs)), None) => {
                    let lhs_slice = lhs.take_value_and_leave_hole();
                    match lhs_slice.try_join(slice) {
                        // Success -- fill the hole & break with the appropriate values
                        Ok(new) => {
                            lhs.fill_hole(new);
                            break 'single_join (lhs, lhs_size, lhs_size.add_right(slice_size));
                        }
                        // Failure - nothing left to try to join to
                        Err((l, s)) => {
                            lhs.fill_hole(l);
                            return Err(s);
                        }
                    }
                }
                // Try to join with `rhs` -- roughly the same as `lhs` above
                (None, Some((rhs_size, mut rhs))) => {
                    let rhs_slice = rhs.take_value_and_leave_hole();
                    match slice.try_join(rhs_slice) {
                        Ok(new) => {
                            rhs.fill_hole(new);
                            break 'single_join (rhs, rhs_size, rhs_size.add_left(slice_size));
                        }
                        Err((s, r)) => {
                            rhs.fill_hole(r);
                            return Err(s);
                        }
                    }
                }

                // Try to join with `lhs` first, then with `rhs` - regardless of whether `lhs`
                // failed.
                (Some((lhs_size, mut lhs)), Some((rhs_size, mut rhs))) => {
                    let lhs_slice = lhs.take_value_and_leave_hole();

                    let mut joined_with_lhs = false;
                    let (mid_slice, mid_size) = match lhs_slice.try_join(slice) {
                        Ok(new) => {
                            // Can't fill the hole in `lhs` yet; still need to try to join with
                            // `rhs`.
                            joined_with_lhs = true;
                            (new, lhs_size.add_right(slice_size))
                        }
                        Err((l, s)) => {
                            lhs.fill_hole(l);
                            (s, slice_size)
                        }
                    };

                    let rhs_slice = rhs.take_value_and_leave_hole();

                    // Now try to join with `rhs`:
                    match mid_slice.try_join(rhs_slice) {
                        // If joining with `rhs` succeeds and we never joined with `lhs`, then we
                        // can break 'single_join:
                        Ok(new) if !joined_with_lhs => {
                            rhs.fill_hole(new);
                            break 'single_join (rhs, rhs_size, slice_size.add_right(rhs_size));
                        }
                        // Otherwise, if we succeed in our join and we *already* joined with `lhs`,
                        // we have something a little bit more complex to handle because we're now
                        // *missing* a value in the tree, so we have to propagate that deletion.
                        Ok(new) => {
                            *tree_len -= 1;
                            // Fill `lhs` so that we can perform the deletion at `rhs`
                            lhs.fill_hole(new);
                            // Also, update `store` so that we appropriately redirect `rhs` to
                            // `lhs`.
                            rhs.redirect_to(&lhs, store);

                            // Before we perform the rest of the deletion, send `updates_callback`
                            // a signal that the inserted value was merged with `lhs`:
                            updates_callback(TraverseUpdate::insert(&lhs));

                            let new_lhs_size = mid_size.add_right(rhs_size);
                            return Ok(Self::handle_join_deletion(
                                store,
                                cursor_builder,
                                lhs,
                                lhs_size,
                                new_lhs_size,
                                rhs,
                                rhs_size,
                            ));
                        }
                        // If we couldn't join with anything, return `Err(slice)`
                        Err((s, r)) if !joined_with_lhs => {
                            rhs.fill_hole(r);
                            return Err(s);
                        }
                        // But if we joined with `lhs` and not `rhs`, we can again break to
                        // 'single_join:
                        Err((mid, r)) => {
                            // fill `rhs` first so that we don't violate stacked borrows (I think?)
                            rhs.fill_hole(r);
                            lhs.fill_hole(mid);

                            break 'single_join (lhs, lhs_size, mid_size);
                        }
                    }
                }
            }
        };

        // Set the initial insertion
        updates_callback(TraverseUpdate::insert(&handle));

        let update_state = OpUpdateState {
            child_or_key: ChildOrKey::Key((handle.idx, handle.node)),
            override_pos: None,
            old_size,
            new_size,
            cursor_builder,
        };

        let mapper = OpUpdateState::identity();
        update_state.finish(mapper, None);
        Ok(PostInsertResult::Done)
    }

    /// (*Internal*) Tries to insert the slice(s) _without_ joining with adjacent keys
    ///
    /// A second key to insert may also be provided, which will be inserted after `fst`. If a
    /// second key is provided, only the first key will be used for the slice reference. (I.e.,
    /// only the first key will generate an initial [`TraverseUpdate::PutBack`] event)
    ///
    /// `override_lhs_size` is used exclusively by [`insert_rhs`] in order to ensure that the key
    /// positions (and resulting increases in size) are correct from the moment of insertion.
    ///
    /// `updates_callback` is called with a `TraverseUpdate::Insertion` exactly once by this
    /// function, for `fst` -- regardless of whether `snd` is provided.
    ///
    /// ## Safety
    ///
    /// Callers must ensure that `new_key_idx <= node.leaf().len()`, and that if
    /// `override_lhs_size` is provided, `new_key_idx != 0`.
    ///
    /// [`insert_rhs`]: Self::insert_rhs
    unsafe fn do_insert_full_edge_no_join<'m, 't: 'm, 'c: 'm>(
        store: &mut P::SliceRefStore,
        cursor_builder: CursorBuilder<'c>,
        mut node: NodeHandle<ty::Leaf, borrow::Mut<'t>, I, S, P, M>,
        new_key_idx: u8,
        override_lhs_size: Option<I>,
        fst: SliceSize<I, S>,
        snd: Option<SliceSize<I, S>>,
        updates_callback: UpdatesCallback<I, S, P, M>,
        map_state: <BubbledInsertState<'t, 'c, I, S, P, M> as MapState>::Map<'m>,
    ) -> PostInsertResult<'t, I, S, P, M> {
        // SAFETY: Guaranteed by caller
        unsafe {
            weak_assert!(new_key_idx <= node.leaf().len());
            // override_lhs_size.is_some() => new_key_idx != 0
            weak_assert!(!(override_lhs_size.is_some() && new_key_idx == 0));
        }

        let (one_if_snd, added_size) = match &snd {
            Some(pair) => (1_u8, fst.size.add_right(pair.size)),
            None => (0, fst.size),
        };

        if node.leaf().len() >= node.leaf().max_len() - one_if_snd {
            // "hard" case: split the node on insertion:
            //
            // SAFETY: `do_insert_no_join_split` requires the same things that this function does,
            // and *also* the exact condition above:
            let bubble_state = unsafe {
                Self::do_insert_full_edge_no_join_split(
                    store,
                    cursor_builder,
                    node,
                    new_key_idx,
                    override_lhs_size,
                    fst,
                    snd,
                    updates_callback,
                )
            };
            let early_exit = None;
            return bubble_state.finish(store, map_state, updates_callback, early_exit);
        }

        // Otherwise, "easy" case: just add the slice(s) to the node

        let mut override_pos = None;
        let old_size;
        let new_size;

        if let Some(new_lhs_size) = override_lhs_size {
            let (lhs_pos, old_lhs_size) = {
                // SAFETY: `into_slice_handle` requires that `new_key_idx - 1` is a valid key
                // index, which is guaranteed by the caller to (a) not overflow if
                // `override_lhs_size.is_some()` and (b) be a valid key index.
                let handle = unsafe { node.borrow().into_slice_handle(new_key_idx - 1) };
                (handle.key_pos(), handle.slice_size())
            };

            let slice_pos = lhs_pos.add_right(new_lhs_size);
            unsafe {
                let _ = node.insert_key(store, new_key_idx, fst.slice);
                node.set_single_key_pos(new_key_idx, slice_pos);
                if let Some(pair) = snd {
                    node.insert_key(store, new_key_idx + 1, pair.slice);
                    node.set_single_key_pos(new_key_idx + 1, slice_pos.add_right(fst.size));
                }
            }

            override_pos = Some(lhs_pos);
            old_size = old_lhs_size;
            new_size = new_lhs_size.add_right(added_size);
        } else {
            // SAFETY: `insert_key` requires that the node is a leaf, which is guaranteed by the
            // caller of `do_insert_no_join`. It also requires that the node isn't full (checked
            // above), and that the key is not *greater* than the current number of keys
            // (guaranteed by caller, debug-checked above).
            unsafe {
                let slice_pos = node.insert_key(store, new_key_idx, fst.slice);
                //  ^^^^^^^^^ don't need to use `slice_pos` to shift the keys because that'll be
                //  handled by `PostInsertTraversalState::do_upward_step`.
                if let Some(pair) = snd {
                    node.insert_key(store, new_key_idx + 1, pair.slice);
                    node.set_single_key_pos(new_key_idx + 1, slice_pos.add_right(fst.size));
                    override_pos = Some(slice_pos);
                }
            }

            old_size = I::ZERO;
            new_size = added_size;
        }

        // SAFETY: we just inserted `new_slice_idx`, so it's a valid key index.
        let slice_handle = unsafe { node.into_slice_handle(new_key_idx) };

        // Set the initial insertion:
        updates_callback(TraverseUpdate::insert(&slice_handle));

        // We can allow `PostInsertTraversalState::do_upward_step` to shift the key positions,
        // because the "current" size of the newly inserted slice is zero, so the increase from
        // zero to `slice_size` in `do_upward_step` will have the desired effect anyways.
        let post_op_state = OpUpdateState {
            child_or_key: ChildOrKey::Key((
                new_key_idx + one_if_snd,
                slice_handle.node.erase_type(),
            )),
            override_pos,
            old_size,
            new_size,
            cursor_builder,
        };

        post_op_state.finish(map_state.post_op, None);
        PostInsertResult::Done
    }

    /// (*Internal*) Completes an insertion that resulted in the deletion of a slice
    ///
    /// This can happen when the two slices on either side of the insertion are able to join,
    /// turning two values (plus one) into one.
    ///
    /// This function expects, without checking, that `rhs.is_hole()` is true.
    fn handle_join_deletion<'t, 'c>(
        store: &mut P::SliceRefStore,
        cursor_builder: CursorBuilder<'c>,
        lhs: SliceHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
        old_lhs_size: I,
        new_lhs_size: I,
        rhs: SliceHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
        rhs_size: I,
    ) -> PostInsertResult<'t, I, S, P, M> {
        // todo: Pending deletion implementation.
        todo!()
    }

    /// Inserts the slice to the immediate right of another slice, recording the changed size from
    /// both the left (present) and right (inserted) sides
    ///
    /// This method calls `updates_callback` with `TraverseUpdate::Insertion` only for `fst`.
    fn insert_rhs<'t, 'c>(
        store: &mut P::SliceRefStore,
        cursor_builder: CursorBuilder<'c>,
        right_of: SliceHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
        new_lhs_size: I,
        cursor_mode: CursorMode,
        fst: SliceSize<I, S>,
        snd: Option<SliceSize<I, S>>,
        updates_callback: UpdatesCallback<I, S, P, M>,
    ) -> PostInsertResult<'t, I, S, P, M> {
        // If we're inserting to the right of the particular slice, we have to traverse down to the
        // leftmost leaf in the child right of `right_of` (if it's not yet in a leaf).
        //
        // After inserting, we have to carefully handle the `PostInsertTraversalState` or
        // `BubbledInsertState` as we traverse back up the tree so that we appropriately update the
        // size of the node.

        let lhs_pos = right_of.key_pos();
        let old_lhs_size = right_of.slice_size();
        let lhs_height = right_of.node.height();
        let lhs_k_idx = right_of.idx;

        // First step: find the leaf immediately to the right of `right_of`
        let (next_leaf, insert_idx) = match right_of.node.into_typed() {
            Type::Leaf(node) => (node, lhs_k_idx + 1),
            Type::Internal(node) => {
                let mut parent = node;
                let mut c_idx = lhs_k_idx + 1;
                loop {
                    // SAFETY: `c_idx` is either `lhs_k_idx + 1` (which is valid because
                    // `lhs_k_idx` is a valid key) or zero (which is always valid for internal
                    // nodes)
                    match unsafe { parent.into_child(c_idx).into_typed() } {
                        Type::Leaf(node) => break (node, 0),
                        Type::Internal(p) => {
                            parent = p;
                            c_idx = 0;
                        }
                    }
                }
            }
        };
        let leaf_height = next_leaf.height(); // stored for later

        /////////////////////////////////////////////////////////////
        // Mapping closures                                        //
        // ---                                                     //
        // The rest of this function is basically just a set of    //
        // closures to use as the fields of the state mapper. Some //
        // of them are only used in certain circumstances.         //
        /////////////////////////////////////////////////////////////

        macro_rules! ty {
            (NodeHandle) => { NodeHandle<ty::Unknown, _, I, S, P, M> };
        }

        let handled_shift = Cell::new(false);

        // If the bubble state *will* change the node at the height of `lhs`, then we need to
        // change the way that it internally calculates positions. The case for equal heights is
        // already handled above with `override_lhs_size`.
        //
        // This callback is only used when `lhs_height != leaf_height`.
        let mut map_shift_lhs = |lhs: &ty![NodeHandle], shift: Option<_>| {
            debug_assert!(shift.is_none());

            if lhs.height() + 1 == lhs_height {
                handled_shift.set(true);
                Some(BubbleLhs {
                    pos: lhs_pos,
                    old_size: old_lhs_size,
                    new_size: new_lhs_size,
                })
            } else {
                None
            }
        };

        let mut map_sizes =
            |node: &mut ty![NodeHandle], child_or_key, mut sizes: fix::OpUpdateStateSizes<_>| {
                if node.height() != lhs_height || handled_shift.get() {
                    return sizes;
                }

                let old_lhs_end = lhs_pos.add_right(old_lhs_size);
                let new_lhs_end = lhs_pos.add_right(new_lhs_size);

                let pos = sizes.override_pos.unwrap_or_else(|| match child_or_key {
                    // SAFETY: docs for `OpUpdateState` guarantee that `k_idx` is within bounds
                    // for `node`.
                    ChildOrKey::Key(k_idx) => unsafe { node.leaf().key_pos(k_idx) },
                    ChildOrKey::Child(c_idx) => {
                        let next_key_pos: I = node
                            .leaf()
                            .try_key_pos(c_idx)
                            .unwrap_or_else(|| node.leaf().subtree_size());
                        next_key_pos.sub_left(sizes.old_size)
                    }
                });

                let diff = pos.sub_left(old_lhs_end);
                sizes.old_size = sizes.old_size.add_left(diff).add_left(old_lhs_size);
                sizes.new_size = sizes.new_size.add_left(diff).add_left(new_lhs_size);
                sizes.override_pos = Some(lhs_pos);

                // We're currently at key `lhs_k_idx`, so we want to manually shift all of the keys
                // from `lhs_k_idx + 1` to the last key before what's automatically handled.
                //
                // However, because we know we just inserted to the right-hand side of `lhs`, we
                // know that we only actually need to shift a single key -- if at all. We shift the
                // key at `lhs_k_idx + 1` so long as `state.child_or_key` is
                // `ChildOrKey::Key((_, _))`.
                if let ChildOrKey::Key(_) = child_or_key {
                    // SAFETY: the calls to `key_pos` and `set_single_key_pos` both require only
                    // that `lhs_k_idx + 1` is a valid key index. Because (a) `node` is at the same
                    // height as `lhs` and (b) any splits are not continued at this height, we know
                    // that the index in `ChildOrKey::Key` must be equal to `lhs_k_idx + 1`. The
                    // documentation for `PostInsertTraversalState` guarantees that such a key
                    // index will be valid.
                    unsafe {
                        let pos = node.leaf().key_pos(lhs_k_idx + 1);
                        let new_pos = pos.sub_left(old_lhs_end).add_left(new_lhs_end);
                        node.set_single_key_pos(lhs_k_idx + 1, new_pos);
                    }
                }

                sizes
            };

        // Track the position of the inserted `fst`, so that we can update
        let fst_handle: Cell<Option<(node::NodePtr<I, S, P, M>, u8, u8)>> = Cell::new(None);

        let remap_cursor = cursor_mode == CursorMode::Lhs && !cursor_builder.is_nop();

        let mut new_updates_callback;
        let updates_callback: UpdatesCallback<I, S, P, M> = if remap_cursor {
            new_updates_callback = |upd: TraverseUpdate<I, S, P, M>| {
                match (&upd, fst_handle.get()) {
                    (&TraverseUpdate::Insertion { ptr, height, idx }, None) => {
                        fst_handle.set(Some((ptr, height, idx)));
                    }
                    (
                        TraverseUpdate::Move { old_ptr, old_range, new_ptr, new_height, new_range },
                        Some((fst_ptr, _, fst_idx)),
                    ) if *old_ptr == fst_ptr && old_range.contains(&fst_idx) => {
                        let new_idx = new_range.start + (old_range.start - fst_idx);

                        fst_handle.set(Some((*new_ptr, *new_height, new_idx)));
                    }
                    _ => (),
                }

                updates_callback(upd);
            };
            &mut new_updates_callback
        } else {
            updates_callback
        };

        let mut map_cursor = |node: &ty![NodeHandle], mut builder: CursorBuilder| {
            if node.height() == lhs_height {
                builder.reset();
            }
        };

        let mut map_cursor_side = |lhs: &ty![NodeHandle], mut side| {
            if lhs.height() == lhs_height {
                match fst_handle.get() {
                    None => side = Some(Side::Left),
                    Some((ptr, _, 0)) if ptr != lhs.ptr() => side = None,
                    _ => (),
                }
            }
            side
        };

        let mut mapper = MapBubbledInsertState {
            post_op: MapOpUpdateState {
                cursor: Some(&mut map_cursor),
                sizes: Some(&mut map_sizes),
            },
            cursor_side: Some(&mut map_cursor_side),
            shift_lhs: Some(&mut map_shift_lhs),
        };

        if !remap_cursor {
            mapper.post_op.cursor = None;
            mapper.cursor_side = None;
        }

        let override_lhs_size = match lhs_height == leaf_height {
            true => {
                mapper.post_op.sizes = None;
                mapper.shift_lhs = None;
                Some(new_lhs_size)
            }
            false => None,
        };

        // SAFETY: `do_insert_no_join` requires that `insert_idx <= next_leaf.leaf().len()`, which
        // we can see is true for both cases of the match on `right_of.node.into_typed()` above:
        // For `Type::Leaf`, we're guaranteed that `lhs_k_idx < right_of.leaf().len()` and
        // `next_leaf = right_of`. For `Type::Internal`, we only break with `insert_idx = 0`, which
        // is always <= len: u8.
        //
        // It also requires that `insert_idx != 0` if `override_lhs_size.is_some()`, which is
        // guaranteed by our search procedure above; `insert_idx = lhs_k_idx + 1` if `next_leaf` is
        // at the same height, and is otherwise equal to zero.
        unsafe {
            Self::do_insert_full_edge_no_join(
                store,
                cursor_builder,
                next_leaf,
                insert_idx,
                override_lhs_size,
                fst,
                snd,
                updates_callback,
                mapper,
            )
        }
    }

    //////////////////////////////////////////////////////////
    // Low-level functions                                  //
    // ---                                                  //
    // Lowest-level functionality, handling all the subtle  //
    // details of the implementation. Broadly does not call //
    // other functions in this impl block.                  //
    //                                                      //
    // There's also similar low-level functions defined for //
    // `BubbledInsertState` and `OpUpdateState`, as         //
    // `do_upward_step` (alongisde the mid-level `finish`)  //
    //////////////////////////////////////////////////////////

    /// (*Internal*) Traverses down the `RleTree` from the root, to find where to insert
    ///
    /// If this method returns a [`ChildOrKey::Key`], `idx` is contained within the returned key
    /// *and not at either of its borders*. Otherwise, if it returns a [`ChildOrKey::Child`], which
    /// points to the border between two keys at which the value should be inserted.
    ///
    /// **Note:** `cursor_hints` is received as an `impl Iterator` instead of the `Cursor` directly
    /// so that cursor types with the same [`PathIter`] associated type will use the same
    /// monomorphized function.
    ///
    /// [`PathIter`]: crate::Cursor::PathIter
    fn find_insertion_point<'t>(
        root: NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
        cursor_hints: impl Iterator<Item = PathComponent>,
        idx: I,
    ) -> InsertionPoint<'t, I, S, P, M> {
        let mut cursor_iter = Some(cursor_hints);
        let mut node = root;
        let mut target = idx;

        let mut adjacent_keys = AdjacentKeys { lhs: None, rhs: None };

        loop {
            let cursor_hint = cursor_iter.as_mut().and_then(|iter| iter.next());

            let mut search_result = search_step(node.borrow(), cursor_hint, target);
            // If the result was the start of a key, we actually want to insert in the child
            // immediately before it.
            if let ChildOrKey::Key((k_idx, k_pos)) = search_result {
                if k_pos == target {
                    // SAFETY: `k_idx` is guaranteed to be a valid key index, so it's therefore a valid
                    // child index as well.
                    let child_size = unsafe { node.try_child_size(k_idx).unwrap_or(I::ZERO) };
                    let child_pos = k_pos.sub_right(child_size);
                    search_result = ChildOrKey::Child((k_idx, child_pos));
                }
            }

            let hint_was_good = matches!(
                (search_result, cursor_hint),
                (ChildOrKey::Child((c_idx, _)), Some(h)) if c_idx == h.child_idx,
            );
            if !hint_was_good {
                cursor_iter = None;
            }

            match search_result {
                // A successful result. We don't need the adjacent keys anymore because the insertion
                // will split `key_idx`
                ChildOrKey::Key((k_idx, k_pos)) => {
                    // SAFETY: The call to `find_insertion_point` guarantees that return values of
                    // `ChildOrKey::Key` contains an index within the bounds of the node's keys.
                    let handle = unsafe { node.into_slice_handle(k_idx) };

                    return ChildOrKey::Key(SplitKeyInsert {
                        handle,
                        pos_in_key: target.sub_left(k_pos),
                    });
                }
                // An index between keys -- we don't yet know if it's an internal or leaf node.
                ChildOrKey::Child((child_idx, child_pos)) => {
                    // Later on, we'll distinguish the type of node, but either way we need to update
                    // the adjacent keys:
                    if S::MAY_JOIN {
                        // The key to the left of child index `i` is at key index `i - 1`. We should
                        // only override the left-hand adjacent key if this child isn't the first one
                        // in its node:
                        if let Some(ki) = child_idx.checked_sub(1) {
                            // SAFETY: the return from `find_insertion_point` guarantees that the child
                            // index is within bounds, relative to the number of keys (i.e. less than
                            // key len + 1). So subtracting 1 guarantees a valid key index.
                            adjacent_keys.lhs = Some(unsafe { node.split_slice_handle(ki) });
                        }

                        // `child_idx` is the rightmost child if it's *equal* to `node.len_keys()`.
                        // Anything less means we should get the key to the right of it.
                        if child_idx < node.leaf().len() {
                            // SAFETY: same as above. The right-hand side key is *at* key index
                            // `child_idx` because child indexing starts left of key indexing
                            adjacent_keys.rhs = Some(unsafe { node.split_slice_handle(child_idx) });
                        }
                    }

                    match node.into_typed() {
                        // If it's a leaf node, we've found an insertion point between two keys
                        Type::Leaf(leaf) => {
                            return ChildOrKey::Child(EdgeInsert {
                                node: leaf,
                                adjacent_keys,
                                // Two things here: (a) child indexes are offset to occur in the gaps
                                // *before* the key of the same index (i.e. child0, then key0, then child1)
                                // and (b) the guarantees of `find_insertion_point` mean that th
                                new_k_idx: child_idx,
                            });
                        }
                        // Otherwise, we should keep going:
                        Type::Internal(internal) => {
                            let pos_in_child = target.sub_left(child_pos);
                            target = pos_in_child;
                            // SAFETY: This function only requires that `child_idx` be in bounds. This
                            // is condition is guaranteed by `find_insertion_point`.
                            node = unsafe { internal.into_child(child_idx) };
                            continue;
                        }
                    }
                }
            }
        }
    }

    /// Helper function for [`do_insert_full_edge_no_join`]
    ///
    /// ## Safety
    ///
    /// This function has all of the same safety requirements as [`do_insert_full_edge_no_join`],
    /// and *also* requires that `node.leaf().len()` is sufficiently close to
    /// `node.leaf().max_len()` such that it *doesn't* support adding `fst` and `snd`, if `snd` is
    /// `Some`.
    ///
    /// [`do_insert_full_edge_no_join`]: Self::do_insert_full_edge_no_join
    unsafe fn do_insert_full_edge_no_join_split<'t, 'c>(
        store: &mut P::SliceRefStore,
        cursor_builder: CursorBuilder<'c>,
        mut node: NodeHandle<ty::Leaf, borrow::Mut<'t>, I, S, P, M>,
        mut new_key_idx: u8,
        mut override_lhs_size: Option<I>,
        fst: SliceSize<I, S>,
        snd: Option<SliceSize<I, S>>,
        updates_callback: UpdatesCallback<I, S, P, M>,
    ) -> BubbledInsertState<'t, 'c, I, S, P, M> {
        // There's a few cases here to determine where we split the node, depending both on
        // `new_key_idx` and `node.leaf().len()`. We'll illustrate them all, pretending for now
        // that `M = 3`. Our node is initially:
        //   ???????????????????????????????????????????????????????????????????????????
        //   ??? A ??? B ??? C ??? D ??? E ??? F ???    (guaranteed if `snd` is `None`)
        //   ???????????????????????????????????????????????????????????????????????????
        // ... or ...
        //   ???????????????????????????????????????????????????????????????
        //   ??? A ??? B ??? C ??? D ??? E ???        (possible if `snd` is `Some(_)`)
        //   ???????????????????????????????????????????????????????????????
        // There's also some particularly strange edge cases to handle if `M = 1`, in which case
        // our node would be:
        //   ???????????????????????????      ???????????????
        //   ??? A ??? B ???  or  ??? A ???
        //   ???????????????????????????      ???????????????
        // We'll handle the `M = 1` stuff in a moment.
        //
        // As a general rule, we represent `fst` with a slash ("/") and, if present, we represent
        // `snd` with a plus ("+").
        //
        // ---
        //
        // STEP 1: determine the midpoint for splitting
        //
        // Let's start with the end goals, and work backwards from there. Even though it's tedious,
        // we'll go through all 9(ish) cases because some of them are unexpected, and we can
        // categorize them more simply afterwards:
        //
        //   Case 1 ??? `new_key_idx < M` and `snd = None`:
        //                      ???????????????
        //        ??????????????????????????????????????? ??? C ??? ???????????????????????????????????????
        //        ??? A ??? B ??? / ??? ??????????????? ??? D ??? E ??? F ???  (note: '/' could also take the place of A or B)
        //        ???????????????????????????????????????       ???????????????????????????????????????
        //
        //   Case 2 ??? `new_key_idx < M` and `snd = Some(_)` and `len = max_len`:
        //                          ???????????????
        //        ??????????????????????????????????????????????????? ??? C ??? ???????????????????????????????????????
        //    (a) ??? A ??? B ??? / ??? + ??? ??????????????? ??? D ??? E ??? F ???
        //        ???????????????????????????????????????????????????       ???????????????????????????????????????
        //                      ???????????????                    (equivalent options)
        //        ??????????????????????????????????????? ??? + ??? ???????????????????????????????????????????????????
        //    (b) ??? A ??? B ??? / ??? ??????????????? ??? C ??? D ??? E ??? F ???
        //        ???????????????????????????????????????       ???????????????????????????????????????????????????
        //
        //   Case 3 ??? `new_key_idx < M` and `snd = Some(_)` and `len = max_len - 1`:
        //                      ???????????????
        //        ??????????????????????????????????????? ??? B ??? ???????????????????????????????????????
        //    (a) ??? A ??? / ??? + ??? ??????????????? ??? C ??? D ??? E ???   (if new_key_idx < M - 1)
        //        ???????????????????????????????????????       ???????????????????????????????????????
        //                      ???????????????
        //        ??????????????????????????????????????? ??? + ??? ???????????????????????????????????????
        //    (b) ??? A ??? B ??? / ??? ??????????????? ??? C ??? D ??? E ???   (if new_key_idx == M - 1)
        //        ???????????????????????????????????????       ???????????????????????????????????????
        //
        //   Case 4 ??? `new_key_idx = M` and `snd = None`:
        //                      ???????????????
        //        ??????????????????????????????????????? ??? / ??? ???????????????????????????????????????
        //        ??? A ??? B ??? C ??? ??????????????? ??? D ??? E ??? F ???
        //        ???????????????????????????????????????       ???????????????????????????????????????
        //
        //   Case 5 ??? `new_key_idx = M` and `snd = Some(_)` and `len = max_len`:
        //                          ???????????????
        //        ??????????????????????????????????????????????????? ??? + ??? ???????????????????????????????????????
        //    (a) ??? A ??? B ??? C ??? / ??? ??????????????? ??? D ??? E ??? F ???
        //        ???????????????????????????????????????????????????       ???????????????????????????????????????
        //                      ???????????????                     (equivalent options)
        //        ??????????????????????????????????????? ??? / ??? ???????????????????????????????????????????????????
        //    (b) ??? A ??? B ??? C ??? ??????????????? ??? + ??? D ??? E ??? F ???
        //        ???????????????????????????????????????       ???????????????????????????????????????????????????
        //
        //   Case 6 ??? `new_key_idx = M` and `snd = Some(_)` and `len = max_len - 1`:
        //                      ???????????????
        //        ??????????????????????????????????????? ??? / ??? ???????????????????????????????????????
        //        ??? A ??? B ??? C ??? ??????????????? ??? + ??? D ??? E ???
        //        ???????????????????????????????????????       ???????????????????????????????????????
        //
        //   Case 7 ??? `new_key_idx > M` and `snd = None`:
        //                      ???????????????
        //        ??????????????????????????????????????? ??? D ??? ???????????????????????????????????????
        //        ??? A ??? B ??? C ??? ??????????????? ??? / ??? E ??? F ???
        //        ???????????????????????????????????????       ???????????????????????????????????????
        //
        //   Case 8 ??? `new_key_idx > M` and `snd = Some(_)` and `len = max_len`:
        //                      ???????????????
        //        ??????????????????????????????????????? ??? D ??? ???????????????????????????????????????????????????
        //        ??? A ??? B ??? C ??? ??????????????? ??? / ??? + ??? E ??? F ???
        //        ???????????????????????????????????????       ???????????????????????????????????????????????????
        //
        //   Case 9 ??? `new_key_idx > M` and `snd = Some(_)` and `len = max_len - 1`:
        //                      ???????????????
        //        ??????????????????????????????????????? ??? D ??? ???????????????????????????????????????
        //        ??? A ??? B ??? C ??? ??????????????? ??? / ??? + ??? E ???
        //        ???????????????????????????????????????       ???????????????????????????????????????
        //
        // There's also two surprise cases we can get with `M = 1` and `snd = Some(_)`:
        //
        //   Case S1 ??? `M = 1` and `new_key_idx = 0` and `snd = Some(_)` and `len = 1`:
        //              ???????????????
        //        ??????????????? ??? + ??? ???????????????
        //        ??? / ??? ??????????????? ??? A ???
        //        ???????????????       ???????????????
        //
        //   Case S2 ??? `M = 1` and `new_key_idx = 1` and `snd = Some(_)` and `len = 1`:
        //              ???????????????
        //        ??????????????? ??? / ??? ???????????????
        //        ??? A ??? ??????????????? ??? + ???
        //        ???????????????       ???????????????
        //
        // So -- on to our choice of midpoint.
        //
        // We have to choose *a* midpoint, even if we'll replace it later. It's more efficient to
        // move the midpoint to the end of the left-hand node (rather than shift all the keys of
        // `rhs` after adding it there). So: where when there's two equivalent options, we pick the
        // one that avoids unnecessarily pushing something into the beginning of `rhs`. This isn't
        // possible for cases S1 or S2, but we can handle those separately.
        let midpoint_idx: u8;
        let insert_at: Result<Side, Side>; // Err(_) gives the direction of unbalance if `snd` is
                                           // Some(_). Must be `Side::Right` if `snd` is None

        let m_u8 = M as u8;

        (midpoint_idx, insert_at) = match new_key_idx.cmp(&m_u8) {
            // pre-handle cases S1 and S2
            _ if M == 1 && node.leaf().len() == 1 => {
                let snd = match snd {
                    Some(s) => s,
                    // SAFETY: we're guaranteed that `len = max_len` (which is 2, for M = 1),
                    // unless `snd` is `Some`, so it must be.
                    None => unsafe { weak_unreachable!() },
                };

                // SAFETY: `do_insert_no_join_split_small_m` has all the same requirements as this
                // function (so those are guaranteed by the caller), and also requires `M == 1`,
                // which we can see is true.
                unsafe {
                    return Self::do_insert_full_edge_no_join_split_small_m(
                        store,
                        cursor_builder,
                        node,
                        new_key_idx,
                        override_lhs_size,
                        fst,
                        snd,
                        updates_callback,
                    );
                };
            }
            Ordering::Less => match &snd {
                // Case 3:
                Some(_) if node.leaf().len() < node.leaf().max_len() => {
                    if new_key_idx < m_u8 - 1 {
                        // Case 3(a):
                        (m_u8 - 2, Ok(Side::Left))
                    } else {
                        // Case 3(b):
                        (m_u8 - 2, Err(Side::Left))
                    }
                } // Cases 1 and 2(a):    [we chose 2(a) instead of 2(b)]
                Some(_) | None => (m_u8 - 1, Ok(Side::Left)),
            },
            Ordering::Equal => match &snd {
                // Case 5(a):    [we chose 5(a) instead of 5(b) to avoid shifting rhs]
                Some(_) if node.leaf().len() == node.leaf().max_len() => {
                    (m_u8 - 1, Err(Side::Left))
                }
                // Cases 4 and 6:
                None | Some(_) => (m_u8 - 1, Err(Side::Right)),
            },
            // Cases 7, 8, and 9
            Ordering::Greater => (m_u8, Ok(Side::Right)),
        };

        // SAFETY: `split` requires that `midpoint_idx` is properly within the bounds of the node,
        // which we guarantee by the logic above, because `max_len = 2 * M`
        let (midpoint_key, rhs) = unsafe { node.split(midpoint_idx, store) };
        let mut rhs = NodeBox::new(rhs);

        // Handle initial updates, to correspond to the current state. We'll do more later, as
        // we sort out individual cases.
        for upd in TraverseUpdate::split(&node, midpoint_idx, &rhs.node) {
            updates_callback(upd);
        }

        let old_node_size = node.leaf().subtree_size();
        let rhs_start = rhs.as_ref().leaf().try_key_pos(0).unwrap_or(old_node_size);
        let midpoint_size;

        // Handle `override_lhs_size` for cases around the midpoint -- it's easier to handle here,
        // rather than down below. After the checks here are complete, we've guaranteed that
        // `override_lhs_size` is `Some` only when the left-hand key is in the same node as what
        // `insert_at` indicates.
        match override_lhs_size {
            Some(s)
                // Cases 3(b)/4/5/6 -- inserting into the midpoint
                if matches!(insert_at, Err(_))
                    // Cases 7/8/9 -- inserting as the first key in the right-hand node
                    || matches!(insert_at, Ok(Side::Right) if new_key_idx == midpoint_idx + 1) =>
            {
                // Any time we insert into the midpoint (except for cases S1/S2, which have already
                // been handled), we move the initial midpoint to the end of the left-hand node,
                // and use it as the left-hand key (so updating its size is sufficient).
                //
                // When we insert as the first key in the right-hand node, the midpoint is
                // similarly the left-hand key, but we keep it there. We can update it in the same
                // way.
                midpoint_size = s;
                // Remove `override_lhs_size` because we've fully handled it
                override_lhs_size = None;
            }
            _ => midpoint_size = rhs_start.sub_left(midpoint_key.pos),
        }

        node.set_subtree_size(midpoint_key.pos);

        // Before we do anything else, we'll update the positions in `rhs`. It's technically
        // possible to get away with only updating these once, but that's *really* complicated.
        //
        // SAFETY: `shift_keys_decrease` requires that `from <= rhs.leaf().len()`, which is always
        // true.
        unsafe {
            let opts = ShiftKeys {
                from: 0,
                pos: I::ZERO,
                old_size: rhs_start,
                new_size: I::ZERO,
            };
            shift_keys_decrease(rhs.as_mut(), opts);
        }

        match insert_at {
            Err(Side::Right) => {
                // put `midpoint_key` back into `node`, use the desired insertion as the midpoint
                //
                // We've already handled `override_lhs_size`; see above.
                let new_lhs_subtree_size = node.leaf().subtree_size().add_right(midpoint_size);
                // SAFETY: `push_key` requires only that there's space. This is guaranteed by
                // having a value of `midpoint_idx` that removes *some* values from `node`.
                unsafe { node.push_key(store, midpoint_key, new_lhs_subtree_size) };
                updates_callback(TraverseUpdate::PutBack {
                    ptr: node.ptr(),
                    height: node.height(),
                    idx: node.leaf().len() - 1,
                });

                // If we have a second slice that we're adding, it should go at the start of `rhs`,
                // which is directly after the midpoint.
                if let Some(SliceSize { slice, size }) = snd {
                    // SAFETY: `insert_key` requires that the key index less than or equal to the
                    // current length, which is always true for zero. `shift_keys_increase` requires
                    // the same for `from`, which is true after the insertion completes. The calls to
                    // `as_mut` require uniqueness, which is guaranteed because we just created it
                    unsafe {
                        let pos = rhs.as_mut().insert_key(store, 0, slice);

                        let opts = ShiftKeys { from: 1, pos, old_size: I::ZERO, new_size: size };
                        shift_keys_increase(rhs.as_mut(), opts);
                    }
                }

                BubbledInsertState {
                    lhs: node.erase_type(),
                    key: node::Key {
                        pos: rhs_start,
                        slice: fst.slice,
                        ref_id: Default::default(),
                    },
                    key_size: fst.size,
                    rhs: rhs.erase_type(),
                    old_size: old_node_size,
                    shift_lhs: None,
                    cursor_side: None,
                    inserted: false,
                    cursor_builder,
                }
            }
            Err(Side::Left) => {
                let snd = match snd {
                    // SAFETY: `Err(Side::Left)` is only given when `snd` is `None`.
                    None => unsafe { weak_unreachable!() },
                    Some(s) => s,
                };

                // put `midpoint_key` into the left-hand node, and then `fst` after it. We can
                // guarantee the positions are right by providing the correct values of the new
                // subtree size to `push_key`
                let lhs_key_end = node.leaf().subtree_size().add_right(midpoint_size);

                let fst_key_end = lhs_key_end.add_right(fst.size);
                let fst_key = node::Key {
                    pos: lhs_key_end,
                    slice: fst.slice,
                    ref_id: Default::default(),
                };

                // SAFETY: our careful case-by-case logic above guarantees that there's always room
                // to put the midpoint and first key into the left-hand node, which is all that
                // `push_key` requires.
                let mp_idx = node.leaf().len();
                unsafe {
                    node.push_key(store, midpoint_key, lhs_key_end);
                    node.push_key(store, fst_key, fst_key_end);
                }

                // SAFETY: the `push_key`s above guarantee that `node.leaf().len()` is at least two
                // greater than `mp_idx` (i.e. `mp_idx + 1 < node.leaf().len()`), which is required
                // by the calls to `into_slice_handle`.
                unsafe {
                    let midpoint_slice = node.borrow().into_slice_handle(mp_idx);
                    let fst_key_slice = node.borrow().into_slice_handle(mp_idx + 1);
                    updates_callback(TraverseUpdate::put_back(&midpoint_slice));
                    updates_callback(TraverseUpdate::insert(&fst_key_slice));
                }

                BubbledInsertState {
                    lhs: node.erase_type(),
                    key: node::Key {
                        pos: fst_key_end,
                        slice: snd.slice,
                        ref_id: Default::default(),
                    },
                    key_size: snd.size,
                    rhs: rhs.erase_type(),
                    old_size: old_node_size,
                    shift_lhs: None,
                    cursor_side: Some(Side::Left),
                    inserted: true,
                    cursor_builder,
                }
            }
            Ok(side) => {
                let insert_into = match side {
                    Side::Left => node.as_mut(),
                    Side::Right => {
                        // Adjust the insert position to be relative to the right-hand node
                        new_key_idx -= midpoint_idx + 1;
                        rhs.as_mut()
                    }
                };

                let (pos, old_size, mut new_size) = match override_lhs_size {
                    None => {
                        let p = insert_into
                            .leaf()
                            .try_key_pos(new_key_idx)
                            .unwrap_or_else(|| insert_into.leaf().subtree_size());
                        (p, I::ZERO, I::ZERO)
                    }
                    Some(s) => {
                        // SAFETY: the caller originally guarantees that
                        // `override_lhs_size.is_some()` implies that `new_key_idx != 0`. Our
                        // little bit of handling above extends this to apply to the two split
                        // halves of the original node -- the case where the insertion is at the
                        // start of `rhs` gets explicitly handled with `override_lhs_size` set to
                        // `None` before control flow gets here.
                        let handle =
                            unsafe { insert_into.borrow().into_slice_handle(new_key_idx - 1) };

                        (handle.key_pos(), handle.slice_size(), s)
                    }
                };

                // SAFETY: the calls to `insert_key` and `shift_keys_increase` together require
                // that `new_key_idx <= insert_into.leaf().len()`, which is guaranteed by the
                // values we chose for `midpoint_idx` and our comparison with `M` above. In the
                // case where `snd` is not `None`, the same guarantees are made, but shifted over
                // by one.
                //
                // The call to `set_single_key_pos`, if we make it, requires that `new_key_idx + 1`
                // is a valid key, which we know is true because we just inserted it.
                unsafe {
                    let _ = insert_into.insert_key(store, new_key_idx, fst.slice);
                    let fst_pos = pos.add_right(new_size);
                    insert_into.set_single_key_pos(new_key_idx, fst_pos);
                    new_size = new_size.add_right(fst.size);

                    let mut from = new_key_idx + 1;

                    if let Some(SliceSize { slice, size }) = snd {
                        insert_into.insert_key(store, new_key_idx + 1, slice);
                        new_size = new_size.add_right(size);
                        from += 1;

                        // Update the position of `snd` because currently it's the same as `fst`.
                        // We could make an extra call to `set_key_poss_with` but it's easier to
                        // just set the value directly.
                        let snd_pos = fst_pos.add_right(fst.size);
                        insert_into.set_single_key_pos(new_key_idx + 1, snd_pos);
                    }

                    let opts = ShiftKeys { from, pos, old_size, new_size };
                    shift_keys_increase(insert_into, opts);
                }

                // SAFETY: `into_slice_handle` requires `new_key_idx < insert_into.leaf().len()`,
                // which is guaranteed by `insert_key`.
                let inserted_handle =
                    unsafe { insert_into.borrow().into_slice_handle(new_key_idx) };
                updates_callback(TraverseUpdate::insert(&inserted_handle));

                BubbledInsertState {
                    lhs: node.erase_type(),
                    key: midpoint_key,
                    key_size: midpoint_size,
                    rhs: rhs.erase_type(),
                    old_size: old_node_size,
                    shift_lhs: None,
                    cursor_side: Some(side),
                    inserted: true,
                    cursor_builder,
                }
            }
        }
    }

    /// Helper function for [`do_insert_full_edge_no_join_split`]; refer to that function for
    /// context
    ///
    /// Even though both `fst` and `snd` are provided, `updates_callback` will only be called with
    /// a `TraverseUpdate::Insertion` for `fst`.
    ///
    /// ## Safety
    ///
    /// This function has all of the same safety requirements as
    /// [`do_insert_full_edge_no_join_split`] and *also* requires that `M = 1`.
    ///
    /// [`do_insert_full_edge_no_join_split`]: Self::do_insert_full_edge_no_join_split
    unsafe fn do_insert_full_edge_no_join_split_small_m<'t, 'c>(
        store: &mut P::SliceRefStore,
        cursor_builder: CursorBuilder<'c>,
        mut node: NodeHandle<ty::Leaf, borrow::Mut<'t>, I, S, P, M>,
        new_key_idx: u8,
        override_lhs_size: Option<I>,
        fst: SliceSize<I, S>,
        snd: SliceSize<I, S>,
        updates_callback: UpdatesCallback<I, S, P, M>,
    ) -> BubbledInsertState<'t, 'c, I, S, P, M> {
        // SAFETY: these are all guaranteed by the caller in some form. refer to the individual
        // comments for more information.
        unsafe {
            // directly required
            weak_assert!(M == 1);
            // copied from `do_insert_no_join`
            weak_assert!(!(override_lhs_size.is_some() && new_key_idx == 0));
            // derived from `max_len` and `M = 1` for `do_insert_no_join_split` requirements
            weak_assert!(node.leaf().len() == 1);
            // derived from the above assertion and `do_insert_no_join` requirements
            weak_assert!(new_key_idx <= 1);
        }

        let old_size = node.leaf().subtree_size();

        // Because there's only two outcomes here (refer to `do_insert_no_join_split, S1 and S2),
        // our first goal is just to get all of the values together outside of the nodes; we'll
        // then decide based on `new_key_idx` how to put them back.

        // SAFETY: `split` requires `0 < node.leaf().len()`, which is guaranteed by the assertion
        // above that `node.leaf().len() == 1`.
        let (midpoint_key, rhs) = unsafe { node.split(0, store) };
        let mut rhs = NodeBox::new(rhs);

        // Currently, both `node` and `rhs` have the wrong subtree size (should be zero). We need
        // to fix this because that's what'll be used as the positions of the keys we insert:
        node.set_subtree_size(I::ZERO);
        rhs.as_mut().set_subtree_size(I::ZERO);

        let fst_key = node::Key {
            pos: I::ZERO, // doesn't matter; won't be used
            slice: fst.slice,
            ref_id: Default::default(),
        };
        let snd_key = node::Key {
            pos: I::ZERO, // doesn't matter; won't be used
            slice: snd.slice,
            ref_id: Default::default(),
        };

        // Note: `override_lhs_size = Some(_)` means that `new_key_idx = 1` and the midpoint will
        // be used as its left-hand side.
        let midpoint_size = match override_lhs_size {
            Some(s) => s,
            None => old_size,
        };

        let updates;
        let insertion_side;

        // We either have [fst_key, snd_key, previous] or [previous, fst_key, snd_key]
        //
        // A note on updates: The only items that we care about updating are from `previous` and
        // `fst_key` (`snd_key` doesn't count for insertions).
        #[rustfmt::skip]
        let ((lhs_key, lhs_size), (mid_key, mid_size), (rhs_key, rhs_size)) = if new_key_idx == 0 {
            updates = Some([
                TraverseUpdate::Move {
                    old_ptr: node.ptr(), old_range: 0..1,
                    new_ptr: rhs.node.ptr(), new_height: rhs.node.height(), new_range: 0..1,
                },
                TraverseUpdate::Insertion { ptr: node.ptr(), height: node.height(), idx: 0 },
            ]);
            insertion_side = Some(Side::Left);
            ((fst_key, fst.size), (snd_key, snd.size), (midpoint_key, midpoint_size))
        } else /* new_key_idx == 1 */ {
            updates = None;
            insertion_side = None;
            ((midpoint_key, midpoint_size), (fst_key, fst.size), (snd_key, snd.size))
        };

        // SAFETY: because `node` originally had a length of 1, and we removed the only value into
        // `midpoint_key`, both `node` and `rhs` currently have a length of 0, which means that any
        // insertion satisfies the requirement of `push_key` that there's room for the insertion.
        unsafe {
            node.push_key(store, lhs_key, lhs_size);
            rhs.as_mut().push_key(store, rhs_key, rhs_size);
        }

        // Note: we wait to perform the updates until *after* the values have been put into their
        // places. Otherwise, we'd be handing out pointers to things that don't yet exist.
        if let Some([fst_upd, snd_upd]) = updates {
            updates_callback(fst_upd);
            updates_callback(snd_upd);
        }

        BubbledInsertState {
            lhs: node.erase_type(),
            key: mid_key,
            key_size: mid_size,
            rhs: rhs.erase_type(),
            old_size,
            shift_lhs: None,
            cursor_side: insertion_side,
            inserted: insertion_side.is_some(),
            cursor_builder,
        }
    }
}

/// A one-off struct used only in the `shift_lhs` field of [`BubbledInsertState`]
///
/// For further context of why this type is used, refer to the documentation on
/// [`BubbledInsertState`] itself.
#[derive(Debug, Copy, Clone)]
pub(super) struct BubbleLhs<I> {
    /// `pos` gives the "true" position of the left-hand node, which may disagree
    pub(super) pos: I,
    pub(super) old_size: I,
    pub(super) new_size: I,
}

impl<'t, 'c, I, S, P, const M: usize> BubbledInsertState<'t, 'c, I, S, P, M>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S, M>,
{
    fn finish<'m>(
        mut self,
        store: &mut P::SliceRefStore,
        mut map: <BubbledInsertState<'t, 'c, I, S, P, M> as MapState>::Map<'m>,
        updates_callback: UpdatesCallback<I, S, P, M>,
        early_exit: Option<&mut dyn FnMut(&OpUpdateState<'t, 'c, I, S, P, M>) -> ControlFlow<()>>,
    ) -> PostInsertResult<'t, I, S, P, M>
    where
        Self: 'm,
    {
        loop {
            self = self.map(&mut map);
            match self.do_upward_step(store, updates_callback) {
                BubbleStepResult::KeepGoing(new_state) => self = new_state,
                BubbleStepResult::Root(result) => return result,
                BubbleStepResult::Done(post_op_state) => {
                    post_op_state.finish(map.post_op, early_exit);
                    return PostInsertResult::Done;
                }
            }
        }
    }

    fn do_upward_step(
        mut self,
        store: &mut P::SliceRefStore,
        updates_callback: UpdatesCallback<I, S, P, M>,
    ) -> BubbleStepResult<'t, 'c, I, S, P, M> {
        let lhs_size = self.lhs.leaf().subtree_size();
        let new_total_size = lhs_size
            .add_right(self.key_size)
            .add_right(self.rhs.as_ref().leaf().subtree_size());

        let (mut parent, lhs_child_idx) = match self.lhs.into_parent() {
            Ok((p, idx)) => (p, idx),

            // Base case: no parent, need to create a new `Internal` root. That has to be done from
            // the main function, `insert_internal`, so we need to return a result indicating that.
            Err(lhs) => {
                assert!(self.shift_lhs.is_none(), "internal error: this is a bug");

                if let Some(side) = self.cursor_side {
                    self.cursor_builder.prepend_to_path(PathComponent {
                        // `Side` is explicitly tagged so that Side::Left = 0 and Side::Right = 1
                        child_idx: side as u8,
                    });
                }

                return BubbleStepResult::Root(PostInsertResult::NewRoot {
                    lhs,
                    key: self.key,
                    key_size: self.key_size,
                    rhs: self.rhs,
                });
            }
        };

        // No split, only insert:
        //
        // Do the minimal amount of work here, and let `PostInsertTraversalState::do_upward_step`
        // handle the rest.
        if parent.leaf().len() < parent.leaf().max_len() {
            let (override_pos, old_size, new_size, lhs_child_pos) = match self.shift_lhs {
                Some(k) => (
                    k.pos,
                    self.old_size.add_left(k.old_size),
                    new_total_size.add_left(k.new_size),
                    k.pos.add_right(k.new_size),
                ),
                None => {
                    // Our position for shifts in `PostInsertTraversalState` should be based on the
                    // left-hand child. We can't calculate that with current sizes, because those
                    // won't match the (old) information in `parent`, so we have to use
                    // `self.old_size` and the position of the next key.
                    let next_pos = parent
                        .leaf()
                        .try_key_pos(lhs_child_idx)
                        .unwrap_or_else(|| parent.leaf().subtree_size());

                    let lhs_child_pos = next_pos.sub_right(self.old_size);

                    (lhs_child_pos, self.old_size, new_total_size, lhs_child_pos)
                }
            };

            // SAFETY: The call to `insert_key_and_child` requires that `lhs_child_idx <=
            // parent.leaf().len()`. This is guaranteed because `lhs`'s parent information must
            // have a valid child index. It also requires that `self.rhs` is at the appropriate
            // height, which is guaranteed because `self.lhs` and `self.rhs` *were* at the same
            // height.
            //
            // The call to `set_single_key_pos` requires that `lhs_child_idx <
            // parent.leaf().len()`, which is guaranteed after `insert_key_and_child`.
            unsafe {
                let self_rhs = self.rhs.take().erase_unique();
                parent.insert_key_and_child(store, lhs_child_idx, self.key, self_rhs);
                let midpoint_pos = lhs_child_pos.add_right(lhs_size);
                parent.set_single_key_pos(lhs_child_idx, midpoint_pos);
            };

            // SAFETY: `into_slice_handle` requires that `key_idx < parent.leaf().len()`, which
            // we've already guaranteed by insertion.
            let handle = unsafe { parent.borrow().into_slice_handle(lhs_child_idx).erase_type() };
            let make_update = match self.inserted {
                // Already inserted our value; we're putting something else back
                true => TraverseUpdate::put_back,
                // This is an initial insertion of the target value
                false => TraverseUpdate::insert,
            };
            updates_callback(make_update(&handle));

            // Update the cursor
            //
            // We have to do this because we're lying to `OpUpdateState` about where the insertion
            // is in order to get it to use the correct key position for shifting and it won't
            // update the cursor if we don't say it's in a `Child`.
            if let Some(side) = self.cursor_side {
                self.cursor_builder
                    .prepend_to_path(PathComponent { child_idx: lhs_child_idx + side as u8 });
            }

            return BubbleStepResult::Done(OpUpdateState {
                // We need to use the midpoint key here so that
                // `OpUpdateState::do_upward_step` updates only the keys after the
                // lhs-key-rhs grouping. The midpoint key has the same index as the left-hand
                // child, so the shifting will start immediately after rhs.
                child_or_key: ChildOrKey::Key((lhs_child_idx, parent.erase_type())),
                // ^ because of the funky stuff we're doing above, we need to make sure that the
                // base positions for shifting are correct
                override_pos: Some(override_pos),
                old_size,
                new_size,
                cursor_builder: self.cursor_builder,
            });
        }
        // Couldn't just insert: need to split

        let mut new_key_idx = lhs_child_idx;
        let old_parent_size = parent.leaf().subtree_size();

        // There's a couple things going on here. Basically, there's three cases for where we
        // want `self.key` to end up: the left-hand node, right-hand node, or as the new
        // midpoint key.
        //
        // At the end of the process, there should be exactly `M` keys in both nodes, so if
        // `self.key` isn't the midpoint, we need to only put `M - 1` keys into the node it'll
        // end up in. If it *is* the midpoint (`new_key_idx == M`), then we have to do some
        // tricky manipulations that'll put a key into `lhs`.
        let midpoint_idx = if new_key_idx > M as u8 {
            M as u8
        } else {
            (M - 1) as u8
        };

        // SAFETY: `split` requires that `midpoint_idx` is within the bounds of the node, which
        // we guarantee above, knowing that `parent.leaf().len() == parent.leaf().max_len()`,
        // which is equal to `2 * M` (and `M >= 1`).
        let (midpoint_key, rhs) = unsafe { parent.split(midpoint_idx, store) };
        let mut rhs = NodeBox::new(rhs);
        parent.set_subtree_size(midpoint_key.pos);

        // We want to find `rhs_start` (i.e. the position of the first child in `rhs`), but there's
        // a couple edge cases we have to handle first:
        //
        // 1. If `M = 1` and `new_key_idx = 2`, then `rhs` will have no keys.
        // 2. If `new_key_idx = M` or `M + 1`, then the first child of `rhs` is actually
        //    `self.lhs`, so its size has been changed. We have to use `self.old_size` instead for
        //    the size of that first child.
        let rhs_start = {
            let first_key_pos = match rhs.as_ref().leaf().try_key_pos(0) {
                Some(p) => p,
                // SAFETY: see note above. If `M > 1`, then all configurations require at least 1
                // key in `rhs` to get `len >= M` by the end of this function; we're adding at most
                // one key to it.
                None if M > 1 => unsafe { weak_unreachable!() },
                None => rhs.as_ref().leaf().subtree_size(),
            };

            let first_child_size = if new_key_idx == M as u8 || new_key_idx == M as u8 + 1 {
                self.old_size
            } else {
                // SAFETY: `child` requires a valid child index. Internal nodes must always have at
                // least one child, so `0` is valid.
                unsafe { rhs.as_ref().child(0).leaf().subtree_size() }
            };
            first_key_pos.sub_right(first_child_size)
        };

        let mut midpoint_size = rhs_start.sub_left(midpoint_key.pos);

        // The logic can get tricky to follow throughout the rest of this function. To help make
        // things a bit easier, we'll provide illustrations to roughly show what the state of the
        // tree is at each point. These illustrations will pretend that M = 3, so `parent`
        // originally looked something like:
        //         ???????????????     ???????????????     ???????????????     ???????????????     ???????????????     ???????????????
        //    ?????????????????? A ????????????????????? B ????????????????????? C ????????????????????? D ????????????????????? E ????????????????????? F ??????????????????
        //    ???..a?????????????????????a-b?????????????????????b-c?????????????????????c-d?????????????????????d-e?????????????????????e-f?????????????????????f..???
        //    ???????????????     ???????????????     ???????????????     ???????????????     ???????????????     ???????????????     ???????????????
        // ... with the upper row representing keys of `parent` and the lower row representing its
        // children. One of the children is `self.lhs`.
        if new_key_idx != M as u8 {
            // Before we do anything else, update the positions in `rhs` so that they start at
            // zero, instead of one.
            //
            // SAFETY: `shift_keys_decrease` requires that `opts.from <= rhs.leaf().len()`,
            // which must always be true because `opts.from = 0`. `rhs.as_mut()` requires
            // unique access, which it has because `rhs` was just created.
            unsafe {
                let opts = ShiftKeys {
                    from: 0,
                    pos: I::ZERO,
                    old_size: rhs_start,
                    new_size: I::ZERO,
                };
                shift_keys_decrease(rhs.as_mut(), opts);
            }

            // Handle a special case with `shift_lhs`. All other cases (for `new_key_idx != M`)
            // will have the left-hand key on the same side of the split as the insertion.
            if let (Some(shift), true) = (self.shift_lhs, new_key_idx == M as u8 + 1) {
                // If we have change the side of the left-hand key, AND that key is the current
                // midpoint (because `new_key_idx` is at the start of `rhs`), then the change for
                // `shift_lhs` can be made *just* by changing the size of the midpoint key.
                //
                // For reference, this is what we're currently looking at:
                //                                       ???????????????
                //         ???????????????     ???????????????     ???????????????     ???mid???          ???????????????     ???????????????
                //    ?????????????????? A ????????????????????? B ????????????????????? C ????????????????????????????????????????????????     ??? E ????????????????????? F ??????????????????
                //    ???..a?????????????????????a-b?????????????????????b-c?????????????????????c-d???     ???lhs???     ??????????????????e-f?????????????????????f..???
                //    ???????????????     ???????????????     ???????????????     ???????????????     ???????????????          ???????????????     ???????????????
                //                                            ^^^^^ self.lhs
                midpoint_size = shift.new_size;

                // remove `shift_lhs` to mark it as being resolved
                self.shift_lhs = None;
            }

            let (side, insert_into) = if new_key_idx < M as u8 {
                (Side::Left, parent.as_mut())
            } else {
                // Adjust the position of `slice` to be relative to the right-hand node
                new_key_idx -= M as u8 + 1;

                // Appropriately adjsut `shift_lhs` to be relative to the right-hand node now
                if let Some(s) = self.shift_lhs.as_mut() {
                    s.pos = s.pos.sub_left(rhs_start);
                }

                (Side::Right, rhs.as_mut())
            };

            let (lhs_child_pos, shift_pos, mut old_size, mut new_size) = match self.shift_lhs {
                None => {
                    // Like our careful maneuvering above to get `rhs_start`, we have to similarly
                    // ignore the *current* influence of `self.lhs` when we get its position,
                    // instead using its previous size from `self.old_size`.
                    let next_key_pos = insert_into
                        .leaf()
                        .try_key_pos(new_key_idx)
                        .unwrap_or_else(|| insert_into.leaf().subtree_size());
                    let lhs_child_pos = next_key_pos.sub_right(self.old_size);
                    (lhs_child_pos, lhs_child_pos, I::ZERO, I::ZERO)
                }
                Some(lhs) => {
                    let lhs_child_pos = lhs.pos.add_right(lhs.new_size);
                    (lhs_child_pos, lhs.pos, lhs.old_size, lhs.new_size)
                }
            };

            let key_pos = lhs_child_pos.add_right(lhs_size);
            old_size = old_size.add_right(self.old_size);
            new_size = new_size.add_right(new_total_size);

            // SAFETY: the calls to `insert_key_and_child` and `shift_keys_increase` together
            // require that `new_key_idx <= insert_into.leaf().len()`, which is guaranteed by the
            // values we chose for `midpoint_idx` and our comparison with `M` above.
            unsafe {
                let self_rhs = self.rhs.take().erase_unique();
                let _ = insert_into.insert_key_and_child(store, new_key_idx, self.key, self_rhs);
                insert_into.set_single_key_pos(new_key_idx, key_pos);

                let opts = ShiftKeys {
                    from: new_key_idx + 1,
                    pos: shift_pos,
                    old_size,
                    new_size,
                };
                shift_keys_increase(insert_into, opts);
            }

            // At this point, we know that our value has definitely been inserted -- it was either
            // a key or child from `self.lhs/`self.rhs`, or `self.key`. We didn't take anything out
            // of `self.{lhs,rhs}`, so we don't need to worry about the pointer becoming invalid.
            if let Some(s) = self.cursor_side {
                self.cursor_builder
                    .prepend_to_path(PathComponent { child_idx: new_key_idx + s as u8 });
            }

            // Signal the insertion:
            let make_update = match self.inserted {
                true => TraverseUpdate::put_back,
                false => TraverseUpdate::insert,
            };

            // SAFETY: `into_slice_handle` requires that `new_key_idx < insert_into.leaf().len()`,
            // which is guaranteed by `insert_key_and_child`.
            let inserted_handle = unsafe { insert_into.borrow().into_slice_handle(new_key_idx) };
            updates_callback(make_update(&inserted_handle));

            BubbleStepResult::KeepGoing(BubbledInsertState {
                lhs: parent.erase_type(),
                key: midpoint_key,
                key_size: midpoint_size,
                rhs: rhs.erase_type(),
                old_size: old_parent_size,
                shift_lhs: None,
                inserted: true,
                cursor_side: Some(side),
                cursor_builder: self.cursor_builder,
            })
        } else {
            // ^ new_key_idx == M. We'll use `self.key` as the new midpoint. Our current state
            // looks something like:
            //                             ??????????????? midpoint
            //                             ???????????????
            //         ???????????????     ???????????????     ??? C ???     ???????????????     ???????????????     ???????????????
            //    ?????????????????? A ????????????????????? B ??????????????????????????????????????????????????? D ????????????????????? E ????????????????????? F ??????????????????
            //    ???..a?????????????????????a-b?????????????????????b-c???     ???c-d?????????????????????d-e?????????????????????e-f?????????????????????f..???
            //    ???????????????     ???????????????     ???????????????     ???????????????     ???????????????     ???????????????     ???????????????
            //                                  ^^^^^ self.lhs
            //
            // The eventual outcome will be that the current midpoint and leftmost child of `rhs`
            // are appended to `parent`, with `self.rhs` replacing `rhs`'s leftmost child, and
            // `self.key` replacing the midpoint, so:
            //                                       ???????????????
            //         ???????????????     ???????????????     ???????????????     ???key???     ???????????????     ???????????????     ???????????????
            //    ?????????????????? A ????????????????????? B ????????????????????? C ??????????????????????????????????????????????????? D ????????????????????? E ????????????????????? F ??????????????????
            //    ???..a?????????????????????a-b?????????????????????b-c?????????????????????c-d???     ???rhs?????????????????????d-e?????????????????????e-f?????????????????????f..???
            //    ???????????????     ???????????????     ???????????????     ???????????????     ???????????????     ???????????????     ???????????????     ???????????????
            //                                  ^^^^^ self.lhs
            // From this, it's also easy enough to see that any correction to the left-hand key
            // will happen to `C` -- the current midpoint and eventual last key in the left-hand
            // node. So we can fully account for `shift_lhs` by correcting the size of the midpoint
            // key.
            if let Some(lhs) = self.shift_lhs {
                midpoint_size = lhs.new_size;
            }

            // Replace the leftmost child of `rhs`.
            let new_first_child_size = self.rhs.as_ref().leaf().subtree_size();
            // SAFETY: `replace_first_child` requires that `self.rhs` is at the correct height to
            // be a child of `parent`. This is guaranteed by `self.rhs` being at the same height as
            // `self.lhs`, because `parent` *is* the parent of `self.lhs`. `as_mut` requires unique
            // access, which is guaranteed because we just created it. `replace_first_child`
            // requires unique access to `self.rhs`, which is guaranteed for the same reason.
            let old_first_child = unsafe { rhs.as_mut().replace_first_child(self.rhs.take()) };
            let old_first_child_size = old_first_child.leaf().subtree_size();

            // Make sure it'll drop if the arithmetic operations below fail
            let old_first_child = SharedNodeBox::new(old_first_child);

            // Add the key and child to the left-hand node (`parent`)
            let lhs_size = parent.leaf().subtree_size();
            let new_lhs_size = lhs_size.add_right(midpoint_size).add_right(old_first_child_size);

            // SAFETY: `push_key_and_child` requires that `parent.leaf().len()` is not equal to the
            // capacity, which we know is true because `parent.leaf().len() == M - 1` is less than
            // the capacity of `2 * M`.
            unsafe {
                let child = old_first_child.take();
                parent.push_key_and_child(store, midpoint_key, child, new_lhs_size);
            }

            // We haven't yet updated the positions in `rhs`, so they're still relative the base of
            // `parent`. After swapping out the leftmost child, we're now good to reclaculate
            // `rhs_start` and shift everything, as we did above.

            // SAFETY: `rhs.leaf().len()` is still >= 1
            let first_rhs_key_pos = unsafe { rhs.as_ref().leaf().key_pos(0) };

            // We need to shift the values, but it's entirely possible that the increase in size
            // from `self.rhs` (which may contain the insertion) is larger than the current start
            // position. So we have to determine which parameterization of `shift_keys` to call
            // based on whether it's a net increase or decrease in size.
            //
            // SAFETY: `shift_keys_auto` requires that `from <= rhs.leaf().len()`, which is always
            // true because `from = 0`. `as_mut` requires unique access, which is guaranteed
            // because `rhs` was just created.
            unsafe {
                let opts = ShiftKeys {
                    from: 0,
                    pos: I::ZERO,
                    old_size: first_rhs_key_pos,
                    new_size: new_first_child_size,
                };
                shift_keys_auto(rhs.as_mut(), opts)
            }

            // If `self.insertion_side` was `None` (i.e. the insertion was in `self.key`), then
            // it's still there. We only need to do anything if `self.insertion_side` is not `None`
            if let Some(side) = self.cursor_side {
                // In this case, the `side` associated with the insertion remained the same;
                // `self.lhs` ended up at the end of its node, and `self.rhs` ended up at the
                // beginning of the next.
                //
                // All we need to do here is record the child index for the cursor.

                let child_idx = match side {
                    Side::Right => 0,
                    // There are `M` keys in `parent`, so the index of the last child is also `M`
                    Side::Left => M as u8,
                };

                self.cursor_builder.prepend_to_path(PathComponent { child_idx });
            }

            BubbleStepResult::KeepGoing(BubbledInsertState {
                lhs: parent.erase_type(),
                key: self.key,
                key_size: self.key_size,
                rhs: rhs.erase_type(),
                old_size: old_parent_size,
                shift_lhs: None,
                cursor_side: self.cursor_side,
                inserted: self.inserted,
                cursor_builder: self.cursor_builder,
            })
        }
    }
}

/// The result of calling [`BubbledInsertState::do_upward_step`]
enum BubbleStepResult<'t, 'c, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    /// The [`BubbledInsertState`] has reached the root
    Root(PostInsertResult<'t, I, S, P, M>),
    /// Changes in tree structure from the [`BubbledInsertState`] have been resolved, and there are
    /// still more nodes to traverse
    Done(OpUpdateState<'t, 'c, I, S, P, M>),
    KeepGoing(BubbledInsertState<'t, 'c, I, S, P, M>),
}

/// Struct containing `dyn` functions that map [`BubbledInsertState`]s (See: [`MapState`])
///
/// This type also includes the mapping functions for the [`OpUpdateState`] that will continue the
/// traversal up the tree once we've stopped "bubbling" the insertion.
pub(super) struct MapBubbledInsertState<'f, 't, 'c: 'f, I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S, M>,
{
    pub(super) post_op: <OpUpdateState<'t, 'c, I, S, P, M> as MapState>::Map<'f>,
    pub(super) cursor_side: Option<
        &'f mut dyn FnMut(
            &NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
            Option<Side>,
        ) -> Option<Side>,
    >,
    pub(super) shift_lhs: Option<
        &'f mut dyn FnMut(
            &NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
            Option<BubbleLhs<I>>,
        ) -> Option<BubbleLhs<I>>,
    >,
}

impl<'t, 'c, I, S, P, const M: usize> MapState for BubbledInsertState<'t, 'c, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    type Map<'f> = MapBubbledInsertState<'f, 't, 'c, I, S, P, M> where Self: 'f;

    fn identity<'f>() -> Self::Map<'f>
    where
        Self: 'f,
    {
        MapBubbledInsertState {
            post_op: <OpUpdateState<'t, 'c, I, S, P, M> as MapState>::identity(),
            cursor_side: None,
            shift_lhs: None,
        }
    }

    fn map<'f>(mut self, f: &mut Self::Map<'f>) -> Self
    where
        Self: 'f,
    {
        if let Some(cursor_side) = f.cursor_side.as_mut() {
            self.cursor_side = cursor_side(&self.lhs, self.cursor_side)
        }

        if let Some(map_cursor) = f.post_op.cursor.as_mut() {
            map_cursor(&self.lhs, self.cursor_builder.reborrow());
        }

        if let Some(map_shift) = f.shift_lhs.as_mut() {
            self.shift_lhs = map_shift(&self.lhs, self.shift_lhs);
        }

        self
    }
}
