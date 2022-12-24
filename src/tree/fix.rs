//! Common tools for handling tree restructuring and post-operation updates
//!
//! Broadly speaking, this module is only intended to cover the _most_ generic pieces; modules like
//! `insert` are intended to handle their specific algorithms.
//!
//! A brief summary of items here:
//!
//! * [`MapState`] - a helper trait for creating and using state mapping callbacks
//! * [`TraverseUpdate`] - used with the [`apply_to_ref`] method to track [`SliceHandle`] positions
//!   without other algorithms needing to be aware of them. See also: [`UpdatesCallback`].
//! * [`OpUpdateState`] - provides general "shift positions around after we've done an operation"
//!   handling, for use in a variety of contexts.
//! * [`ShiftKeys`] - provides a some tools to shift a range of positions in a node, with the
//!   related [`shift_keys_auto`], [`shift_keys_increase`], and [`shift_keys_decrease`] functions.
//!
//! [`apply_to_ref`]: TraverseUpdate::apply_to_ref

use crate::cursor::CursorBuilder;
use crate::param::RleTreeConfig;
use crate::{Index, PathComponent, Slice};
use std::ops::{ControlFlow, Range};

use super::node::{borrow, ty, ChildOrKey, NodeHandle, NodePtr, SliceHandle};

#[cfg(test)]
use crate::MaybeDebug;
#[cfg(test)]
use std::fmt::{self, Debug, Formatter};

//////////////
// MapState //
//////////////

/// Helper trait to allow state mapping callback sets to be easily constructed and used
///
/// This trait _may_ have some issues around under-inlining due to the implementations' use of
/// `dyn` functions. On the other hand, under-inlining may actually provide a net benefit because
/// of reduced code output. In future, we can try benchmarking this version vs. a macro-based
/// replacement.
///
/// Implementors (and `Map` types):
///
///  * [`OpUpdateState`] ([`MapOpUpdateState`])
pub(super) trait MapState {
    /// Generic type for functions, or sets of functions, that can be used to map values of `Self`
    ///
    /// Typically, `Map`s are either `&'f mut dyn FnMut(Self) -> Self` or a struct containing
    /// multiple such functions. We use this to prevent certain properties of `Self` from being
    /// changed -- whether that's enum variants or a subset of struct fields.
    type Map<'f>
    where
        Self: 'f;

    /// Produces the identity mapping -- i.e., the one that when used with `map` will leave the
    /// input unchanged
    fn identity<'f>() -> Self::Map<'f>
    where
        Self: 'f;
    /// Calls the provided function(s) `self` (or pieces of it), returning the new value
    fn map<'f>(self, f: &mut Self::Map<'f>) -> Self
    where
        Self: 'f;
}

/// Struct containing `dyn` functions that map [`OpUpdateState`]s (See: [`MapState`])
///
/// Both functions are provided the [`NodeHandle`] taken from `child_or_key` to allow functions
/// using extra context.
pub(super) struct MapOpUpdateState<'f, 't, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    pub(super) cursor: Option<
        &'f mut (dyn for<'c> FnMut(
            &NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
            CursorBuilder<'c>,
        )),
    >,
    pub(super) sizes: Option<
        &'f mut dyn FnMut(
            &mut NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
            ChildOrKey<u8, u8>,
            OpUpdateStateSizes<I>,
        ) -> OpUpdateStateSizes<I>,
    >,
}

/// Struct representing the subset of fields in an [`OpUpdateState`] that refer to sizes and
/// positions
pub(super) struct OpUpdateStateSizes<I> {
    pub(super) override_pos: Option<I>,
    pub(super) old_size: I,
    pub(super) new_size: I,
}

#[rustfmt::skip]
impl<'t, 'c, I, S, P, const M: usize> MapState for OpUpdateState<'t, 'c, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    type Map<'f> = MapOpUpdateState<'f, 't, I, S, P, M>
        where Self: 'f;

    fn identity<'f>() -> Self::Map<'f>
        where Self: 'f,
    {
        MapOpUpdateState {
            cursor: None,
            sizes: None,
        }
    }

    fn map<'f>(mut self, f: &mut Self::Map<'f>) -> Self
        where Self: 'f,
    {
        let (node, child_or_key) = match &mut self.child_or_key {
            ChildOrKey::Key((ki, n)) => (n, ChildOrKey::Key(*ki)),
            ChildOrKey::Child((ci, n)) => (n.untyped_mut(), ChildOrKey::Child(*ci)),
        };

        if let Some(map_cursor) = f.cursor.as_mut() {
            map_cursor(node, self.cursor_builder.reborrow());
        }

        if let Some(map_sizes) = f.sizes.as_mut() {
            let new_sizes = map_sizes(node, child_or_key, OpUpdateStateSizes {
                override_pos: self.override_pos,
                old_size: self.old_size,
                new_size: self.new_size,
            });
            self.override_pos = new_sizes.override_pos;
            self.old_size = new_sizes.old_size;
            self.new_size = new_sizes.new_size;
        }

        self
    }
}

////////////////////
// TraverseUpdate //
////////////////////

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
    /// Records that an insertion occured.
    ///
    /// There are some operations that generate multiple insertion operations. It's typically the
    /// responsibility of the provided callback to filter them out appropriately.
    Insertion {
        ptr: NodePtr<I, S, P, M>,
        height: u8,
        idx: u8,
    },
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
        new_height: u8,
        new_range: Range<u8>,
    },
    /// Marks the slice as being temporarily removed from the tree
    Remove {
        ptr: NodePtr<I, S, P, M>,
        height: u8,
        idx: u8,
    },
    /// Returns a previously removed (via `Remove`) slice to a new position in the tree
    ///
    /// It is *guaranteed* by the caller of a callback using `TraverseUpdate` that `PutBack` will
    /// be provided exactly once after each `Remove`, without more than one `Remove` before the
    /// appropriate `PutBack`.
    ///
    /// **Note**: this variant is _also_ used to signal the initial creation of the `SliceRef`
    PutBack {
        ptr: NodePtr<I, S, P, M>,
        height: u8,
        idx: u8,
    },
}

/// Standard type definition for callback functions handling [`TraverseUpdate`]s
pub(super) type UpdatesCallback<'f, I, S, P, const M: usize> =
    &'f mut dyn FnMut(TraverseUpdate<I, S, P, M>);

#[cfg(test)]
impl<I, S, P, const M: usize> Debug for TraverseUpdate<I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Insertion { ptr, height, idx } => f
                .debug_struct("Insertion")
                .field("ptr", ptr)
                .field("height", height)
                .field("idx", idx)
                .finish(),
            #[rustfmt::skip]
            Self::Move { old_ptr, old_range, new_ptr, new_height, new_range } => f
                .debug_struct("Move")
                .field("old_ptr", old_ptr)
                .field("old_range", old_range)
                .field("new_ptr", new_ptr)
                .field("new_height", new_height)
                .field("new_range", new_range)
                .finish(),
            Self::Remove { ptr, height, idx } => f
                .debug_struct("Remove")
                .field("ptr", ptr)
                .field("height", height)
                .field("idx", idx)
                .finish(),
            Self::PutBack { ptr, height, idx } => f
                .debug_struct("PutBack")
                .field("ptr", ptr)
                .field("height", height)
                .field("idx", idx)
                .finish(),
        }
    }
}

impl<I, S, P, const M: usize> TraverseUpdate<I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Creates a new `TraverseUpdate::Insertion` referring to the given slice
    pub(super) fn insert<Ty: ty::TypeHint, B>(handle: &SliceHandle<Ty, B, I, S, P, M>) -> Self {
        TraverseUpdate::Insertion {
            ptr: handle.node.ptr(),
            height: handle.node.height(),
            idx: handle.idx,
        }
    }

    /// Creates a new `TraverseUpdate::PutBack` referring to the given slice
    pub(super) fn put_back<Ty: ty::TypeHint, B>(handle: &SliceHandle<Ty, B, I, S, P, M>) -> Self {
        TraverseUpdate::PutBack {
            ptr: handle.node.ptr(),
            height: handle.node.height(),
            idx: handle.idx,
        }
    }

    /// Creates an iterator of `TraverseUpdate`s corresponding to splitting the given node at the
    /// index
    #[rustfmt::skip]
    pub(super) fn split<Ty: ty::TypeHint, B>(
        lhs: &NodeHandle<Ty, B, I, S, P, M>,
        idx: u8,
        rhs: &NodeHandle<Ty, borrow::UniqueOwned, I, S, P, M>,
    ) -> impl IntoIterator<Item = TraverseUpdate<I, S, P, M>> {
        [
            Self::Move {
                old_ptr: lhs.ptr(), old_range: idx + 1..idx + 1 + rhs.leaf().len(),
                new_ptr: rhs.ptr(), new_range: 0..rhs.leaf().len(), new_height: rhs.height(),
            },
            Self::Remove { ptr: lhs.ptr(), height: lhs.height(), idx: lhs.leaf().len() },
        ]
    }

    /// Consumes a `TraverseUpdate` and does nothing
    ///
    /// It's generally preferred to use this function instead of providing your own, for two
    /// reasons:
    ///
    /// 1. This method is more explicit;
    /// 2. The validity of the arguments are checked in debug mode; and
    /// 3. While debugging, we can easily log the arguments passed to this function, whereas
    ///    tracking down all instances of `|_| {}` may be more tedious
    pub(super) fn do_nothing(self) {
        #[cfg(any(test, debug_assertions))]
        self.assert_valid();
    }

    #[cfg(any(test, debug_assertions))]
    fn assert_valid(&self) {
        match self {
            // Nothing we can validate, because it would make assumptions about the current state
            // of the node, which we generally aren't allowed to do.
            TraverseUpdate::Insertion { ptr: _, height: _, idx: _ }
            | TraverseUpdate::Remove { ptr: _, height: _, idx: _ }
            | TraverseUpdate::PutBack { ptr: _, height: _, idx: _ } => (),

            #[rustfmt::skip]
            TraverseUpdate::Move {
                old_ptr: _, old_range,
                new_ptr: _, new_range, new_height: _,
            } => {
                assert!(old_range.start <= old_range.end);
                assert!(new_range.start <= new_range.end);
                assert_eq!(old_range.len(), new_range.len());
            }
        }
    }

    /// Returns a closure that applies `TraverseUpdate`s to the given reference
    ///
    /// Roughly speaking, it starts from the first [`TraverseUpdate::Insertion`] and tracks the
    /// position afterwards.
    ///
    /// ## Panics
    ///
    /// Calling `apply_to_ref` will never _directly_ panic. The closure it returns can, however,
    /// and it's worth documenting those conditions. They are:
    ///
    /// * If the closure is called with `TraverseUpdate::Insertion` more than once (or: at all, if
    ///   `slice_ref` was initially `Some(_)`)
    /// * If the closure is called with `TraverseUpdate::Remove` in a row, without a
    ///   `TraverseUpdate::PutBack` between (disregarding `Insertion`s or `Move`s).
    ///
    /// ## Safety
    ///
    /// The caller guarantees that all values provided to the callback accurately reflect changes
    /// occuring in the tree. This can be assumed for most functions.
    pub(super) unsafe fn apply_to_ref(
        slice_ref: &mut Option<SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>>,
    ) -> impl '_ + FnMut(TraverseUpdate<I, S, P, M>) {
        // The implementation of the update step, written as a pure function with mutable inputs
        //
        // This is extracted so that it's easy to inject logging into the closure we return.
        fn do_update<I, S, P: RleTreeConfig<I, S, M>, const M: usize>(
            update: TraverseUpdate<I, S, P, M>,
            already_present: &mut bool,
            slice_ref: &mut Option<SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>>,
        ) {
            match (update, &slice_ref) {
                (TraverseUpdate::Insertion { .. }, Some(_)) => unreachable!(),
                (TraverseUpdate::Insertion { .. }, None) if *already_present => unreachable!(),
                (TraverseUpdate::Insertion { ptr, height, idx }, None) => {
                    *slice_ref = Some(unsafe { SliceHandle::from_raw_parts(ptr, height, idx) });
                    *already_present = true;
                }

                // `Move` with nothing stored does nothing
                (TraverseUpdate::Move { .. }, None) => (),
                // `Move` that doesn't overlap the contents does nothing
                (TraverseUpdate::Move { old_ptr, old_range, .. }, Some(r))
                    if old_ptr != r.node.ptr() || !old_range.contains(&r.idx) => {}
                // `Move` does the expected thing when it affects the stored handle
                (
                    TraverseUpdate::Move { old_range, new_ptr, new_height, new_range, .. },
                    Some(r),
                ) => {
                    debug_assert_eq!(old_range.len(), new_range.len());

                    let new_idx = (r.idx - old_range.start) + new_range.start;
                    *slice_ref =
                        Some(unsafe { SliceHandle::from_raw_parts(new_ptr, new_height, new_idx) });
                }

                // We shouldn't get `Remove`s when there's nothing there right now
                (TraverseUpdate::Remove { .. }, None) => {
                    if *already_present {
                        unreachable!()
                    }
                }
                // `Remove` doesn't do anything if it doesn't affect the stored handle
                (TraverseUpdate::Remove { ptr, idx, .. }, Some(r))
                    if ptr != r.node.ptr() || idx != r.idx => {}
                // `Remove` affecting the stored handle... removes it.
                (TraverseUpdate::Remove { height, .. }, Some(h)) => {
                    debug_assert_eq!(h.node.height(), height);
                    *slice_ref = None;
                }

                (TraverseUpdate::PutBack { .. }, Some(_)) => (),
                (TraverseUpdate::PutBack { ptr, height, idx }, None) => {
                    if *already_present {
                        *slice_ref = Some(unsafe { SliceHandle::from_raw_parts(ptr, height, idx) });
                    }
                }
            }
        }

        let mut already_present = slice_ref.is_some();
        move |update: TraverseUpdate<I, S, P, M>| {
            // space reserved for adding debugging :)
            do_update(update, &mut already_present, slice_ref);
        }
    }

    /// Produces a closure that calls `f` for all `TraverseUpdate`s except `Insertion`s
    pub(super) fn skip_insertions<'f>(
        mut f: impl 'f + FnMut(TraverseUpdate<I, S, P, M>),
    ) -> impl 'f + FnMut(TraverseUpdate<I, S, P, M>) {
        move |upd: TraverseUpdate<I, S, P, M>| match upd {
            TraverseUpdate::Insertion { .. } => {}
            u => f(u),
        }
    }
}

///////////////////
// OpUpdateState //
///////////////////

/// State recording how to propagate an update to the size of the tree as we traverse upwards
pub(super) struct OpUpdateState<'t, 'c, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
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
    pub(super) child_or_key: ChildOrKey<
        // Child: must be internal
        (u8, NodeHandle<ty::Internal, borrow::Mut<'t>, I, S, P, M>),
        // Key: unknown type
        (u8, NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>),
    >,

    /// Optional override in case the start position of the size change is not equal to the start
    /// position given by `child_or_key`
    pub(super) override_pos: Option<I>,

    /// The old size of the child or key containing the change
    pub(super) old_size: I,
    /// The new size of the child or key containing the change. This is provided separately so that
    /// it can be updated *after* the original size has been recorded
    pub(super) new_size: I,

    /// Cursor representing the path to the deepest node containing the change
    ///
    /// In theory, it's entirely possible to use a `SliceHandle` referencing the inserted slice to
    /// build a cursor after the fact and get the exact path to the insertion. In practice, this
    /// involves doing a second upwards tree traversal, so we opted for the harder solution.
    pub(super) cursor_builder: CursorBuilder<'c>,
}

#[cfg(test)]
impl<'t, 'c, I, S, P, const M: usize> Debug for OpUpdateState<'t, 'c, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let (idx_name, idx, node) = match &self.child_or_key {
            ChildOrKey::Key((k_idx, node)) => ("key_idx", k_idx, node),
            ChildOrKey::Child((c_idx, node)) => ("child_idx", c_idx, node.untyped_ref()),
        };

        f.debug_struct("OpUpdateState")
            .field("node", node)
            .field(idx_name, idx)
            .field("override_pos", self.override_pos.fallible_debug())
            .field("old_size", self.old_size.fallible_debug())
            .field("new_size", self.new_size.fallible_debug())
            .finish()
    }
}

impl<'t, 'c, I, S, P, const M: usize> OpUpdateState<'t, 'c, I, S, P, M>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S, M>,
{
    /// Runs the post-operation tree changes to completion, using the callbacks to modify the
    /// behavior as necessary
    ///
    /// Each call to `early_exit` is made _after_ `map_state` for a given node, and before calling
    /// [`do_upward_step`](Self::do_upward_step).
    pub(super) fn finish<'m>(
        mut self,
        mut map_state: <Self as MapState>::Map<'m>,
        mut early_exit: Option<&mut dyn FnMut(&Self) -> ControlFlow<()>>,
    ) where
        Self: 'm,
    {
        loop {
            self = self.map(&mut map_state);

            if let Some(ControlFlow::Break(())) = early_exit.as_mut().map(|f| f(&self)) {
                return;
            }

            match self.do_upward_step() {
                Ok(()) => return,
                Err(new_state) => self = new_state,
            }
        }
    }

    /// Performs a single upward step of the `OpUpdateState`, moving its focused node towards the
    /// root
    ///
    /// Typically this method will not be used directly, and instead
    pub(super) fn do_upward_step(mut self) -> Result<(), Self> {
        // The first part of the traversal step starts with repositioning all of the keys after
        // `self.child_or_key`.

        let (mut node, first_key_after, maybe_child_idx, pos) = match self.child_or_key {
            ChildOrKey::Key((k_idx, node)) => {
                let pos = self
                    .override_pos
                    // SAFETY: The documentation for `PostInsertTraversalState` guarantees that
                    // `k_idx` is within bounds.
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
            shift_keys_auto(&mut node, ShiftKeys {
                from: first_key_after,
                pos,
                old_size: self.old_size,
                new_size: self.new_size,
            });
        }

        let new_size = node.leaf().subtree_size();

        // Update the cursor
        //
        // Technically speaking, this relies on this node having `child_or_key =
        // ChildOrKey::Child(_)` as long as the cursor is non-empty, but that should always be
        // true.
        if let Some(child_idx) = maybe_child_idx {
            self.cursor_builder.prepend_to_path(PathComponent { child_idx });
        }

        // Update `self` to the parent, if there is one. Otherwise, this is the root node so we
        // should return `Err` with the cursor and reference to the insertion.
        match node.into_parent().ok() {
            None => Ok(()),
            Some((parent_handle, c_idx)) => Err(OpUpdateState {
                child_or_key: ChildOrKey::Child((c_idx, parent_handle)),
                override_pos: None,
                old_size,
                new_size,
                cursor_builder: self.cursor_builder,
            }),
        }
    }
}

///////////////////////
// `ShiftKeys` et al //
///////////////////////

/// (*Internal*) Helper struct for [`shift_keys`], so that the parameters are named and we can
/// provide a bit more documentation
pub(super) struct ShiftKeys<I> {
    /// First key index to update. May be equal to node length
    pub from: u8,
    /// The position of the earlier change in the node
    ///
    /// We're being intentionally vague here because this can correspond to multiple different
    /// things. `PostInsertTraversalState`, for example, uses this as the index of an updated
    /// child or key. But for cases where we've inserted multiple things (e.g., where
    /// [`insert_rhs`] is provided a second value), this might only correspond to the index of the
    /// first item, with `old_size` and `new_size` referring to the total size of the pieces.
    ///
    /// [`insert_rhs`]: crate::RleTree::insert_rhs
    pub pos: I,
    /// The old size of the object at `pos`
    pub old_size: I,
    /// The new size of the object at `pos`
    ///
    /// If [`shift_keys`] is given `IS_INCREASE = true`, then `new_size` will be expected to be
    /// greater than `old_size`. Otherwise, `new_size` *should* be less than `old_size`. We can't
    /// *guarantee* that this will be true, because of the limitations exposed by user-defined
    /// implementations of `Ord` for `I`, but in general these conditions should hold.
    pub new_size: I,
}

/// (*Internal*) Updates the positions of all keys in a node, calling either
/// [`shift_keys_increase`] or [`shift_keys_decrease`] depending on whether the size has increased
/// or decreased
///
/// ## Safety
///
/// The value of `opts.from` must be less than or equal to `node.leaf().len()`.
pub(super) unsafe fn shift_keys_auto<'t, Ty, I, S, P, const M: usize>(
    node: &mut NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
    opts: ShiftKeys<I>,
) where
    Ty: ty::TypeHint,
    I: Index,
    P: RleTreeConfig<I, S, M>,
{
    // SAFETY: guaranteed by caller
    unsafe {
        if opts.new_size > opts.old_size {
            shift_keys_increase(node, opts)
        } else {
            shift_keys_decrease(node, opts)
        }
    }
}

/// (*Internal*) Updates the positions of all keys in a node, expecting them to increase
///
/// ## Safety
///
/// The value of `opts.from` must be less than or equal to `node.leaf().len()`.
pub(super) unsafe fn shift_keys_increase<'t, Ty, I, S, P, const M: usize>(
    node: &mut NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
    opts: ShiftKeys<I>,
) where
    Ty: ty::TypeHint,
    I: Index,
    P: RleTreeConfig<I, S, M>,
{
    // SAFETY: guaranteed by caller
    unsafe { shift_keys::<Ty, I, S, P, M, true>(node, opts) }
}

/// (*Internal*) Updates the positions of all keys in a node, expecting them to decrease
///
/// ## Safety
///
/// The value of `opts.from` must be less than or equal to `node.leaf().len()`.
pub(super) unsafe fn shift_keys_decrease<'t, Ty, I, S, P, const M: usize>(
    node: &mut NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
    opts: ShiftKeys<I>,
) where
    Ty: ty::TypeHint,
    I: Index,
    P: RleTreeConfig<I, S, M>,
{
    // SAFETY: guaranteed by caller
    unsafe { shift_keys::<Ty, I, S, P, M, false>(node, opts) }
}

/// (*Internal*) Updates the positions of all keys in a node
///
/// Typically this function is called via [`shift_keys_increase`] or [`shift_keys_decrease`] to
/// avoid manually specifying the generic args.
///
/// ## Safety
///
/// The value of `opts.from` must be less than or equal to `node.leaf().len()`.
pub(super) unsafe fn shift_keys<'t, Ty, I, S, P, const M: usize, const IS_INCREASE: bool>(
    node: &mut NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
    opts: ShiftKeys<I>,
) where
    Ty: ty::TypeHint,
    I: Index,
    P: RleTreeConfig<I, S, M>,
{
    // SAFETY: guaranteed by caller
    unsafe { weak_assert!(opts.from <= node.leaf().len()) };

    let ShiftKeys { from, pos, old_size, new_size } = opts;

    // Because some of the jumping through hoops can get a little confusing in this function, we've
    // illustrated how all of the positions are laid out below.
    //
    // First, define `old_end = pos.add_right(old_size)` and `new_end = pos.add_right(new_size)`.
    // Visually, this is:
    //
    //   |------------- old_end ------------|
    //   |- ... ---- pos ----|-- old_size --|
    //
    //   |--------------- new_end --------------|
    //   |- ... ---- pos ----|---- new_size ----|
    //
    // These abbreviations help make the illustration below a little more compact. In either case
    // of `IS_INCREASE`, we have something like the following:
    //
    //   |- ... ----------------- k_pos -----------------|  (old key pos, of a later key in the node)
    //   |- ... - old_end -|
    //                     |-- k_pos.sub_left(old_end) --|  (offset)
    //   |- ... --- new_end ---|
    //                         |-- k_pos.sub_left(old_end) --|  (offset, shifted)
    //   |- ... - k_pos.sub_left(old_end).add_left(new_end) -|  (new key pos)
    //
    // So we have a workable formula here for shifitng keys. The thing is, we'd to make this a
    // single operation in the loop, in case the index type is complicated or the compiler has
    // failed to optimize away the extra work.
    //
    // We're guaranteed by the directional arithmetic rules that the following is equivalent to
    // the formula we got above:
    //
    //   k_pos.sub_left(old_end.sub_right(new_end))
    //
    // Of course, this is only valid if `new_end < old_end` -- so if `IS_INCREASE` is false.
    // Naturally, there's a rephrasing for when `new_end > old_end`:
    //
    //   k_pos.add_left(new_end.sub_left(old_end))           FIXME: prove this?
    //
    // This one is to be used when `IS_INCREASE` is true.
    //
    // ---
    //
    // With both of these, we can use the const generics from IS_INCREASE to minimize the final
    // amount of code that gets generated:
    let old_end = pos.add_right(old_size);
    let new_end = pos.add_right(new_size);

    let abs_difference = match IS_INCREASE {
        true => new_end.sub_left(old_end),
        false => old_end.sub_right(new_end),
    };

    let recalculate = |k_pos: I| -> I {
        match IS_INCREASE {
            true => k_pos.add_left(abs_difference),
            false => k_pos.sub_left(abs_difference),
        }
    };

    // SAFETY: `set_key_poss_with` requires that `from` is not greater than the number of keys
    // (just like this function). This is already guaranteed by the caller.
    unsafe { node.set_key_poss_with(recalculate, from..) };
}
