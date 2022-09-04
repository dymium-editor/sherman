//! Wrapper module containing the big-boy tree itself

use crate::param::{self, AllowSliceRefs, BorrowState, RleTreeConfig, StrongCount, SupportsInsert};
use crate::public_traits::{Index, Slice};
use crate::range::RangeBounds;
use crate::{Cursor, NoCursor, PathComponent};
use std::fmt::Debug;
use std::mem::{self, ManuallyDrop};
use std::ops::Range;
use std::panic::UnwindSafe;

#[cfg(test)]
use crate::{MaybeDebug, NoDebugImpl};
#[cfg(test)]
use std::fmt::{self, Formatter};

pub(crate) mod cow;
mod iter;
mod node;
pub(crate) mod slice_ref;
#[cfg(test)]
mod tests;

pub use iter::{Drain, Iter, SliceEntry};
pub use slice_ref::SliceRef;

use node::{borrow, ty, ChildOrKey, NodeHandle, SliceHandle, Type};
use slice_ref::ShouldDrop;

/// The default minimum number of keys in a node; default parameterization for [`RleTree`]
///
/// The value here wasn't chosen for any *particular* reason, aside from it mostly matching the
/// standard library's choice of size. Speed and memory usage will depend on the particular types
/// used to parameterize the tree, but there are a few things that are generally true:
///
/// * Larger nodes (in general) have more unused space
/// * Larger nodes require fewer memory accesses to traverse the tree
/// * Smaller nodes are better for mitigating the impacts of complex [`Index`] types, and the
///   corollary:
/// * Larger nodes are more costly to propagate changes through
pub const DEFAULT_MIN_KEYS: usize = 5;

/// *Raison d'être of the crate*: run-length encoded highly-parameterizable B-tree
///
/// Although this type is quite complex, it arises out of the simplest version of the problems it
/// solves. That is:
///
/// > Given a large `Vec<T>` with `r` runs of identical values, how do we represent the data with
/// size `O(r)`, supporting `O(log(r))` insert and delete
///
/// The general idea turns out to be pretty ok: a binary search tree with (a) keys as the offset of
/// the run's start index from the parent node's and (b) values as the length and content of each
/// run. Insertion and deletion -- which require updating the position of *every* value at a
/// greater index -- still only require `O(log(r))` updates because all the positions are relative,
/// except for the root.
///
/// Unfortunately, in order to squeeze more functionality out of this type, we gradually added more
/// and more features to it.
///
/// ## Ok, so why is `RleTree` so complicated?
///
/// Well, here's the thing. This crate was specifically made for use in a text editor, where every
/// time you find *one* use case for a run-length encoded tree, it turns out there's another one
/// just around the corner. We started with "*tag each byte in a file with the edit that last
/// touched it*" and moved to "*we can represent the file content itself*", to eventually "*hey, with
/// special index types, this can apply to line/column number pairs!*".
///
/// So the simplest version of this type would just represent a mapping `usize -> T`, where `T`
/// implements `PartialEq + Clone` and comparisons are handled by the tree. But to provide more
/// flexibility, we instead have a mapping `I -> S`, where `I` implements [`Index`] (but is nearly
/// always `usize`) and `S` implements [`Slice`], which provides utilites for joining and splitting
/// runs (instead of with `PartialEq` and `Clone`).
///
/// Before we go further, it may be helpful to see some examples.
///
/// ## Examples
///
/// FIXME
///
/// ## Hey what's that other stuff?
///
/// So our explanation above didn't *quite* go through everything that `RleTree` has to offer.
///
/// The two remaining parameters are `P`, the parameterized feature set (one of [`NoFeatures`],
/// [`AllowCow`], or [`AllowSliceRefs`]), and `M`, the minimum number of keys in a node.
///
/// There's some more information in the [`param`] module, but briefly: the type `P` allows the
/// clone-on-write and slice references features to be enabled or disabled at compile-time. It is
/// typically only specified if it's something other than [`NoFeatures`].
///
/// The constant `M` matches some traditional B-Tree literature (although not Knuth), and
/// singularly determines the various sizes of each node: the maximum number of keys is `2 * M`,
/// and therefore there's a maximum of `2 * M + 1` children for an internal node. We also require
/// that `M` is between 1 and 127, inclusive.
///
/// [`NoFeatures`]: param::NoFeatures
/// [`AllowCow`]: param::AllowCow
/// [`AllowSliceRefs`]: param::AllowSliceRefs
pub struct RleTree<I, S, P = param::NoFeatures, const M: usize = DEFAULT_MIN_KEYS>
where
    P: RleTreeConfig<I, S>,
{
    root: Option<Root<I, S, P, M>>,
}

// FIXME: This should be more precise, specifically around the interactions with `RefUnwindSafe`
// and `AllowSliceRefs` / `AllowCow`
impl<I, S, P, const M: usize> UnwindSafe for RleTree<I, S, P, M>
where
    I: UnwindSafe,
    S: UnwindSafe,
    P: RleTreeConfig<I, S>,
{
}

// Separate struct to handle the data associated with the root node - but only when it actually
// exists.
struct Root<I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S>,
{
    handle: ManuallyDrop<NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>>,
    refs_store: <P as RleTreeConfig<I, S>>::SliceRefStore,
    shared_total_strong_count: <P as RleTreeConfig<I, S>>::SharedStrongCount,
}

#[cfg(not(feature = "nightly"))]
impl<I, S, P, const M: usize> Drop for RleTree<I, S, P, M>
where
    P: RleTreeConfig<I, S>,
{
    fn drop(&mut self) {
        destruct_root(self.root.take())
    }
}

#[cfg(feature = "nightly")]
unsafe impl<#[may_dangle] I, #[may_dangle] S, P, const M: usize> Drop for RleTree<I, S, P, M>
where
    P: RleTreeConfig<I, S>,
{
    fn drop(&mut self) {
        destruct_root(self.root.take())
    }
}

#[cfg(test)]
impl<I: Index, S, P: RleTreeConfig<I, S>, const M: usize> Debug for Root<I, S, P, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        struct Nodes<'t, I, S, P: RleTreeConfig<I, S>, const M: usize> {
            root: NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
            indent: &'static str,
        }

        impl<'t, I: Index, S, P: RleTreeConfig<I, S>, const M: usize> Debug for Nodes<'t, I, S, P, M> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("{")?;
                let max_child_len = 2 * M + 1;
                let elem_pad = format!("{max_child_len}").len();
                let total_pad = match self.root.height() as usize {
                    0 => 0,
                    n => (elem_pad + 2) * n - 2, // +2 for the ", " between each
                };
                let mut path = Vec::new();
                write_nodes(self.root, &mut path, self.indent, elem_pad, total_pad, f)?;
                f.write_str("\n}")
            }
        }

        struct SliceContents<'a, T>(&'a [T]);

        impl<T: Debug> Debug for SliceContents<'_, T> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                for (i, v) in self.0.iter().enumerate() {
                    if i != 0 {
                        f.write_str(", ")?;
                    }
                    f.write_fmt(format_args!("{v:?}"))?;
                }
                Ok(())
            }
        }

        fn write_nodes<I: Index, S, P: RleTreeConfig<I, S>, const M: usize>(
            node: NodeHandle<ty::Unknown, borrow::Immut, I, S, P, M>,
            path: &mut Vec<u8>,
            indent: &'static str,
            elem_pad: usize,
            total_pad: usize,
            f: &mut Formatter,
        ) -> fmt::Result {
            let path_fmt = format!("{:<elem_pad$?}", SliceContents(path));
            f.write_fmt(format_args!(
                "\n{indent}[{path_fmt:<total_pad$}] @ {:p}: {:?}",
                node.ptr(),
                node.typed_debug(),
            ))?;

            if let Type::Internal(n) = node.typed_ref() {
                for c_idx in 0..=n.leaf().len() {
                    // SAFETY: `into_child` requires that `c_idx <= n.leaf().len()`, which is
                    // guaranteed by the iterator.
                    let child = unsafe { n.into_child(c_idx) };
                    path.push(c_idx);
                    write_nodes(child, path, indent, elem_pad, total_pad, f)?;
                    path.pop();
                }
            }

            Ok(())
        }

        let indent = match f.alternate() {
            false => "    ",
            true => "        ",
        };
        let nodes = Nodes {
            root: self.handle.borrow(),
            indent,
        };
        // FIXME: Add SliceRefStore to this if size_of::<P::SliceRefStore>() != 0
        f.debug_struct("Root").field("nodes", &nodes).finish()
    }
}

/// (*Internal*) Helper function that *actually* implements the destructor for [`RleTree`]
///
/// This is extracted out so that it can be used in other places (e.g., [`insert_internal`], where
/// we also need to call the destructor on a [`Root`])
///
/// If you follow the calls, the recursive destructor for the nodes themselves is implemented in
/// [`NodeHandle::do_drop`].
///
/// [`insert_internal`]: RleTree::insert_internal
fn destruct_root<I, S, P, const M: usize>(root: Option<Root<I, S, P, M>>)
where
    P: RleTreeConfig<I, S>,
{
    let mut r = match root {
        None => return,
        Some(r) => match r.refs_store.try_acquire_drop() {
            ShouldDrop::No => return,
            ShouldDrop::Yes => r,
        },
    };

    // Drop the slice reference store so that we don't need to update references in the node
    // destructors.
    drop(mem::take(&mut r.refs_store));

    // SAFETY: We're given the `Root` by value; it's plain to see that `r.ptr` is not accessed
    // again in this function, before it goes out of scope. That's all that's required of
    // `ManuallyDrop::take`.
    let p = unsafe { ManuallyDrop::take(&mut r.handle) };
    if let Some(handle) = p.try_drop() {
        handle.do_drop();
    }

    // Only now can we decrement the tree-wide strong count; otherwise we could run into race
    // conditions.
    r.shared_total_strong_count.decrement();
}

/// (*Internal*) Checks that the value of `M` provided for a `RleTree` is within the allowed bounds
const fn assert_reasonable_m<const M: usize>() {
    // We have a maximum key length of u8::MAX. Really there shouldn't be anything longer than
    // that -- a single page of memory is big enough as-is. Maximum number of keys is `2*M`, so
    // `u8::MAX / 2` is 127 -- the maximum value for M.
    if M > 127 {
        panic!("cannot construct RleTree: const M must be <= 127");
    }

    // We also have a minimum value of M -- the derivation for this one comes from the insertion
    // algorithm: when we're inserting something, the worst-case scenario is that we want to insert
    // two items into the same leaf node. In order to ensure that it's always possible to do this,
    // we need `maximum_keys + 2 <= 2 * maximum_keys`, so `maximum_keys >= 2`. This doesn't hold
    // for `M = 0` (which would be a binary tree), but it does for `M >= 1`.
    if M < 1 {
        panic!("cannot construct RleTree: const M must be >= 1");
    }
}

impl<I, S, P, const M: usize> RleTree<I, S, P, M>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    /// Creates a new, empty `RleTree`
    ///
    /// This method will panic if `M` is greater than 127, the current limit for `M`.
    pub const fn new_empty() -> Self {
        assert_reasonable_m::<M>();

        RleTree { root: None }
    }

    /// Creates an `RleTree` initialized to contain just the initial slice of the given size
    ///
    /// ## Panics
    ///
    /// Like [`new_empty`], this method will panic if `M` is greater than 127, the current limit
    /// for `M`.  It will also panic if `size` is not greater than zero -- i.e. if
    /// `size <= I::ZERO`.
    ///
    /// [`new_empty`]: Self::new_empty
    pub fn new(slice: S, size: I) -> Self {
        assert_reasonable_m::<M>();

        if size <= I::ZERO {
            panic!("cannot add slice with non-positive size {:?}", size);
        }

        RleTree {
            root: Some(Root {
                handle: ManuallyDrop::new(NodeHandle::new_root(slice, size).erase_type()),
                refs_store: Default::default(),
                shared_total_strong_count: StrongCount::one(),
            }),
        }
    }

    /// Returns the total size of the tree -- i.e. the sum of the sizes of all the sizes
    pub fn size(&self) -> I {
        match self.root.as_ref() {
            Some(root) => root.handle.leaf().subtree_size(),
            None => I::ZERO,
        }
    }

    /// Returns an iterator yielding all slices that touch the range
    ///
    /// The iterator is double-ended and produces [`SliceEntry`]s, where the entry's range *may*
    /// have one or both bounds outside `range`, but *will* contain some overlap with the original
    /// `range`.
    ///
    /// This guarantee is only guaranteed by the special [`RangeBounds`] trait we use, which is
    /// ever so slightly different from the standard library's.
    ///
    /// This method is identical to calling [`iter_with_cursor`] with [`NoCursor`].
    ///
    /// ## Panics
    ///
    /// This method panics either if (a) the range's start is after its end (b) the range's bounds
    /// include an index greater than or equal to [`self.size()`](Self::size).
    ///
    /// The description above is a bit vague; here's some more concrete examples:
    ///
    /// ```should_panic
    /// use sherman::{RleTree, Constant};
    ///
    /// let tree: RleTree<usize, Constant<char>> = RleTree::new(Constant('a'), 5);
    /// // panics, out of bounds:
    /// let _ = tree.iter(5..);
    /// ```
    ///
    /// ```should_panic
    /// # use sherman::{RleTree, Constant}; let tree: RleTree<usize, Constant<char>> = RleTree::new(Constant('a'), 5);
    /// // panics, out of bounds:
    /// let _ = tree.iter(..=5);
    /// ```
    ///
    /// ```should_panic
    /// # use sherman::{RleTree, Constant}; let tree: RleTree<usize, Constant<char>> = RleTree::new(Constant('a'), 5);
    /// // panics, invalid range:
    /// let _ = tree.iter(5..5);
    /// ```
    ///
    /// ```
    /// # use sherman::{RleTree, Constant}; let tree: RleTree<usize, Constant<char>> = RleTree::new(Constant('a'), 5);
    /// // doesn't panic, produces an empty iterator:
    /// let mut iter = tree.iter(..0);
    /// assert!(iter.next().is_none());
    /// ```
    ///
    /// [`iter_with_cursor`]: Self::iter_with_cursor
    pub fn iter<R>(&self, range: R) -> Iter<'_, NoCursor, I, S, P, M>
    where
        R: Debug + RangeBounds<I>,
    {
        // While this method is equivalent to `self.iter_with_cursor(NoCursor, range)`, it's better
        // for us to replicate the body here so that any call stacks look correct.

        let root = self
            .root
            .as_ref()
            .map(|r| (r.handle.borrow(), &r.refs_store));

        let size = self.size();

        Iter::new(range, size, NoCursor, root)
    }

    /// Like [`iter`], but uses a [`Cursor`] to provide a hint on the first call to `next` or
    /// `next_back`.
    ///
    /// The hint is used exactly once, on the initial call to either `next` or `next_back`,
    /// whichever is used first.
    ///
    /// ## Panics
    ///
    /// This method can panic with certain values of `range`; refer to [`iter`] for further
    /// information.
    ///
    /// [`iter`]: Self::iter
    pub fn iter_with_cursor<C, R>(&self, cursor: C, range: R) -> Iter<'_, C, I, S, P, M>
    where
        C: Cursor,
        R: Debug + RangeBounds<I>,
    {
        let root = self
            .root
            .as_ref()
            .map(|r| (r.handle.borrow(), &r.refs_store));

        let size = self.size();

        Iter::new(range, size, cursor, root)
    }

    /// Removes the range of values from the tree, returning an iterator over those slices
    ///
    /// The iterator is double-ended and produces `(Range<I>, S)` pairs, where the yielded range
    /// will be *entirely* contained within `range`, and slices that contain an endpoint will be
    /// split.
    ///
    /// If the iterator is dropped before all items have been yielded, then the remaining slices in
    /// the range will be removed during `Drain`'s destructor. If the destructor is never called,
    /// the tree will be left in an unspecified (but safe) state. It may leak memory.
    ///
    /// This method is equivalent to calling [`drain_with_cursor`] with [`NoCursor`].
    ///
    /// [`drain_with_cursor`]: Self::drain_with_cursor
    pub fn drain(&mut self, range: Range<I>) -> Drain<'_, NoCursor, I, S, P, M> {
        self.drain_with_cursor(NoCursor, range)
    }

    /// Like [`drain`], but uses a [`Cursor`] to provide a hint on the first call to `next` or
    /// `next_back`.
    ///
    /// The hint is used exactly once, on the initial call to either `next` or `next_back`,
    /// whichever is used first.
    ///
    /// For more information, refer to the documentation for [`drain`].
    ///
    /// [`drain`]: Self::drain
    pub fn drain_with_cursor<C>(&mut self, cursor: C, range: Range<I>) -> Drain<'_, C, I, S, P, M>
    where
        C: Cursor,
    {
        Drain::new(&mut self.root, range, cursor)
    }
}

/// Insertion can be done in all trees, except COW-enabled trees with elements that can't be cloned
impl<I, S, P, const M: usize> RleTree<I, S, P, M>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsInsert<I, S>,
{
    /// Inserts the slice at position `idx`, shifting all later entries by `size`
    ///
    /// If there is any entry that contains `idx`, it will be split and encompass `slice` on either
    /// side after the insertion (unless `slice` joins with either/both sides).
    ///
    /// If COW-capabilities are enabled (i.e. [`param::AllowCow`]), then the path in the tree down to
    /// the insertion will be made unique.
    ///
    /// **See also:** [`insert_ref`], [`insert_with_cursor`], [`insert_ref_with_cursor`].
    ///
    /// ## Panics
    ///
    /// This method will panic if `idx` is *greater* than [`self.size()`]. An index equal to the
    /// current size of the tree is explicitly allowed. It will also panic if the size of the new
    /// slice is not greater than zero -- i.e. if `size <= I::ZERO`.
    ///
    /// [`insert_ref`]: Self::insert_ref
    /// [`insert_with_cursor`]: Self::insert_with_cursor
    /// [`insert_ref_with_cursor`]: Self::insert_ref_with_cursor
    /// [`self.size()`]: Self::size
    pub fn insert(&mut self, idx: I, slice: S, size: I) {
        let _cursor = self.insert_with_cursor(NoCursor, idx, slice, size);
    }

    /// Inserts the slice at position `idx`, using the provided [`Cursor`] as a path hint,
    /// returning a cursor representing the path to the insertion
    ///
    /// **See also:** [`insert`], [`insert_ref`], [`insert_ref_with_cursor`].
    ///
    /// ## Panics
    ///
    /// This method panics on any of the same conditions as [`insert`], in addition to if any of
    /// the cursor's methods panic. Bad output from the cursor will be the same as if there was no
    /// output.
    ///
    /// [`insert`]: Self::insert
    /// [`insert_ref`]: Self::insert_ref
    /// [`insert_ref_with_cursor`]: Self::insert_ref_with_cursor
    pub fn insert_with_cursor<C: Cursor>(&mut self, cursor: C, idx: I, slice: S, size: I) -> C {
        let (cursor, _returned_ptr) = self.insert_internal(cursor, idx, slice, size);
        cursor
    }

    /// (*Internal*) Abstraction over the core insertion algorithm
    fn insert_internal<'t, C: Cursor>(
        &'t mut self,
        cursor: C,
        idx: I,
        slice: S,
        size: I,
    ) -> (C, SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>) {
        // All non-trivial operations on this tree are a pain in the ass, and insertion is no
        // exception. We'll try to have comments explaining some of the more hairy stuff.
        //
        // This function is mostly edge-case handling; all of the tree traversal & shfiting stuff
        // is done in other functions.

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
                let root_ptr = &mut self.root.as_mut().unwrap().handle;
                // SAFETY: We just created the root, so there can't be any conflicts to worry
                // about. The returned handle is already known to have the same lifetime as `self`.
                // `key_idx = 1` is also safe to pass, because the root is guaranteed to have a
                // single entry.
                let value_handle =
                    unsafe { root_ptr.borrow_mut().into_slice_handle(0).clone_slice_ref() };
                return (Cursor::new_empty(), value_handle);
            }
        };

        // Because we're modifying an existing tree, we need to acquire borrowing rights:
        if let Err(conflict) = root.refs_store.acquire_mutable() {
            panic!("{conflict}");
        }

        // And if COW is enabled, we'll need to make sure we have a fresh reference to each part of
        // the tree
        //
        // We don't need to do this if there's only one pointer to the entire tree, but if there's
        // at least one other shallow copy of the tree, we need to make our own. This type exists
        // as a separate copy that we can make changes to.
        let mut shallow_copy_store: Option<Root<I, S, P, M>> = None;

        let mut root_node = if root.shared_total_strong_count.is_unique() {
            // SAFETY: we know there aren't any other borrows on the tree (from acquire_mutable)
            // above, and there can't be more than one strong count -- it's us, so there's can't
            // have been a new one added in between `c.load()` and now. The `Acquire` ordering
            // matches with the `Release` of any dropped references
            unsafe { root.handle.borrow_mut() }
        } else {
            // There's another reference to *somewhere* in the tree -- we'll pessimistically
            // assume that we need to clone (this will usually be true)
            shallow_copy_store = Some(Root {
                // SAFETY: `shallow_clone` requires that `P = AllowCow`, which must betrue if the
                // strong count is not unqiue; all of ther `P: RleTreeConfig` have always-unique
                // strong counts.
                handle: ManuallyDrop::new(unsafe { root.handle.shallow_clone() }),
                // If there were slice references, we'd run into weird issues here. But this branch
                // can't be reached unless `P` is `AllowCow`, which has `SliceRefStore = ()`, so
                // there aren't any.
                refs_store: Default::default(),
                shared_total_strong_count: root.shared_total_strong_count.increment(),
            });
            let root_ptr = &mut shallow_copy_store.as_mut().unwrap().handle;
            // SAFETY: We just created this; it can't have any conflicting borrows/references.
            unsafe { root_ptr.borrow_mut() }
        };

        // With clone on write, we often have unreliable parent pointers. One piece of this is that
        // it *may* be possible for our root node to have an existing parent pointer. This
        // shouldn't *really* happen, but we should have this here until it's proven that it can't
        // happen.
        if P::COW {
            // SAFETY: `borrow_mut` requires unique access. We guaranteed that above.
            unsafe { root_node.borrow_mut().remove_parent() };
        }

        // Now on to the body of the insertion algorithm: continually dropping down the tree until
        // we find a match, then working the changes back up.
        //
        // The comments describing this are in `PreInsertTraversalState::do_search_step`.
        let mut downward_search_state: PreInsertTraversalState<C, I, S, P, M> =
            PreInsertTraversalState {
                cursor_iter: Some(cursor.into_path()),
                node: root_node,
                target: idx,
                adjacent_keys: AdjacentKeys {
                    lhs: None,
                    rhs: None,
                },
            };

        let insertion_point = loop {
            match downward_search_state.do_search_step() {
                Err(further_down) => downward_search_state = further_down,
                Ok(result) => break result,
            }
        };

        let mut post_insert_result: PostInsertTraversalResult<C, I, S, P, M>;

        post_insert_result = match insertion_point {
            ChildOrKey::Child(leaf) => {
                let join_result =
                    Self::do_insert_try_join(&mut root.refs_store, slice, size, leaf.adjacent_keys);

                match join_result {
                    Ok(post_insert_state) => PostInsertTraversalResult::Continue(post_insert_state),
                    // SAFETY: The function requires that the `NodeHandle` provided to it is a leaf
                    // node. This is guaranteed by the `InsertSearchResult`, which says that it's
                    // guaranteed a leaf node if `result` is `ChildOrKey::Child`. We're also
                    // guaranteed that `c_idx` is within the appropriate bounds by
                    // `InsertSearchResult`. The function has extra requirements if
                    // `override_lhs_size` is not `None`, but those don't apply here.
                    Err(slice) => unsafe {
                        let res = Self::do_insert_no_join(
                            &mut root.refs_store,
                            leaf.node,
                            leaf.new_k_idx,
                            None,
                            SliceSize { slice, size },
                            None,
                        );

                        match res {
                            Ok(r) => r,
                            Err(mut bubble_state) => loop {
                                match bubble_state.do_upward_step(&mut root.refs_store, None) {
                                    Err(b) => bubble_state = b,
                                    Ok(post_insert) => break post_insert,
                                }
                            },
                        }
                    },
                }
            }
            ChildOrKey::Key(key) => Self::split_insert(
                &mut root.refs_store,
                key.handle,
                key.pos_in_key,
                slice,
                size,
            ),
        };

        // Propagate the size change up the tree
        let (cursor, insertion) = loop {
            match post_insert_result {
                PostInsertTraversalResult::Continue(state) => {
                    post_insert_result = state.do_upward_step()
                }
                PostInsertTraversalResult::Root(cursor, insertion) => break (cursor, insertion),
                PostInsertTraversalResult::NewRoot {
                    cursor,
                    lhs,
                    key,
                    key_size,
                    rhs,
                    insertion,
                } => {
                    // we need to get rid of `lhs` first, so that we've released the borrow when we
                    // access the root.
                    let lhs_ptr = lhs.ptr();
                    let root = match shallow_copy_store.as_mut() {
                        Some(r) => r,
                        // Remember: far above in the function, we set `Some(root) = &mut self.root`
                        // handling the `None` case by returning.
                        None => root,
                    };

                    debug_assert!(root.handle.ptr() == lhs_ptr);

                    // SAFETY: `make_new_parent` requires that `root.handle` and `rhs` not have any
                    // parent already, which is guaranteed by `NewRoot`. It also requires that we
                    // have unique access to `root.handle` and `rhs`, which is guaranteed by
                    // `NewRoot` for `rhs`, and our operations at the beginning of this function
                    // for `root.handle`.
                    unsafe {
                        root.handle
                            .make_new_parent(&mut root.refs_store, key, key_size, rhs);
                    }
                    let insertion = insertion.unwrap_or_else(|| unsafe {
                        root.handle.borrow().into_slice_handle(0).clone_slice_ref()
                    });
                    break (cursor, insertion);
                }
            }
        };

        // And now, we do need to move `shallow_copy_store` into `self`, if it was used. It's
        // possible that other copies have sinced dropped the remaining references to `self`, so we
        // have to pessimistically perform a full drop.
        if let Some(copy) = shallow_copy_store {
            let old_root = mem::replace(&mut self.root, Some(copy));
            destruct_root(old_root);
        }

        // Release the mutable borrow we originally acquired. It doesn't matter whether we do this
        // before or after replacing from a shallow copy because shallow copies only exist with COW
        // functionality, and explicit borrow state only exists with slice references (which is
        // incompatible with COW stuff)
        match self.root.as_ref() {
            Some(r) => r.refs_store.release_mutable(),
            // SAFETY: We originally checekd that `self.root` is `Some(_)` up above, and while
            // insertion *can* remove up to one value, that value can never be the only value in
            // the tree -- so `self.root` must still be `Some(_)`
            None => unsafe { weak_unreachable!() },
        }

        (cursor, insertion)
    }
}

impl<I, S, const M: usize> RleTree<I, S, AllowSliceRefs, M>
where
    I: Index,
    S: Slice<I>,
{
    /// Inserts the slice, returning a [`SliceRef`] pointing to it
    ///
    /// If there is any entry that contains `idx`, it will be split, encompassing `slice` on either
    /// side after the insertion.
    ///
    /// If `slice` joins with another entry (it may join with at most two), the returned reference
    /// will point to the newly-joined entry containing all joined entries, and will be identical
    /// to any existing references to the entries that were joined with.
    ///
    /// **See also:** [`insert`], [`insert_with_cursor`], [`insert_ref_with_cursor`]
    ///
    /// [`insert`]: Self::insert
    /// [`insert_with_cursor`]: Self::insert_with_cursor
    /// [`insert_ref_with_cursor`]: Self::insert_ref_with_cursor
    pub fn insert_ref(&mut self, idx: I, slice: S, size: I) -> SliceRef<I, S, M> {
        let (_cursor, slice_ref) = self.insert_ref_with_cursor(NoCursor, idx, slice, size);
        slice_ref
    }

    /// Inserts the slice, using the provided [`Cursor`] as a path hint, returning a [`SliceRef`]
    /// pointing to the slice and a cursor representing the path to the insertion
    ///
    /// **See also:** [`insert`], [`insert_ref`], [`insert_with_cursor`]
    ///
    /// [`insert`]: Self::insert
    /// [`insert_ref`]: Self::insert_ref
    /// [`insert_with_cursor`]: Self::insert_with_cursor
    pub fn insert_ref_with_cursor<C: Cursor>(
        &mut self,
        cursor: C,
        idx: I,
        slice: S,
        size: I,
    ) -> (C, SliceRef<I, S, M>) {
        let (cursor, inserted_slice) = self.insert_internal(cursor, idx, slice, size);
        (cursor, self.make_ref(inserted_slice))
    }

    /// (*Internal*) Returns a reference to the slice, either duplicating an existing one or
    /// creating a new one
    fn make_ref(
        &mut self,
        slice_handle: SliceHandle<ty::Unknown, borrow::SliceRef, I, S, AllowSliceRefs, M>,
    ) -> SliceRef<I, S, M> {
        let root = match self.root.as_ref() {
            Some(r) => r,
            // SAFETY: The `SliceHandle` cannot exist without a root node
            None => unsafe { weak_unreachable!() },
        };

        // let ref_id = self.slice_ TODO

        // SliceRef::new(&root.refs_store, ref_id)
        todo!()
    }
}

/// (*Internal*) Helper struct to carry information for [`insert_internal`] and related functions
/// as we traverse down the tree to find an insertion point
///
/// [`insert_internal`]: RleTree::insert_internal
struct PreInsertTraversalState<'t, C, I, S, P, const M: usize>
where
    C: Cursor,
    P: RleTreeConfig<I, S>,
{
    /// If the provided cursor has not provided a bad hint yet, the remaining items in its iterator
    cursor_iter: Option<<C as Cursor>::PathIter>,
    /// The node at the top of the subtree we're looking in
    node: NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
    /// The position relative to the start of `node`'s subtree that we're looking to insert at
    target: I,
    /// The keys to the immediate left and right of the traversal path so far
    adjacent_keys: AdjacentKeys<'t, I, S, P, M>,
}

/// (*Internal*) The result of a successful downward search for an insertion point, returned by
/// [`PreInsertTraversalState::do_search_step`]
type InsertSearchResult<'t, I, S, P, const M: usize> =
    ChildOrKey<LeafInsert<'t, I, S, P, M>, SplitKeyInsert<'t, I, S, P, M>>;

/// (*Internal*) Component of an [`InsertSearchResult`] specific to inserting *between* keys in a
/// leaf
struct LeafInsert<'t, I, S, P: RleTreeConfig<I, S>, const M: usize> {
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

/// (*Internal*) Component of an [`InsertSearchResult`] for when an insertion point is found in the
/// middle of an existing key
///
/// In this case, the key must be split in order to perform the insertion.
struct SplitKeyInsert<'t, I, S, P: RleTreeConfig<I, S>, const M: usize> {
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
struct AdjacentKeys<'b, I, S, P: RleTreeConfig<I, S>, const M: usize> {
    lhs: Option<SliceHandle<ty::Unknown, borrow::Mut<'b>, I, S, P, M>>,
    rhs: Option<SliceHandle<ty::Unknown, borrow::Mut<'b>, I, S, P, M>>,
}

/// (*Internal*) Helper struct to carry the information we use when an insertion overflowed a node,
/// and we need to "bubble" the new midpoint and right-hand side up to the parent
struct BubbledInsertState<'t, C, I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S>,
{
    /// The existing node that `rhs` was split off from
    lhs: NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
    /// The key between `lhs` and `rhs`
    key: node::Key<I, S, P>,
    /// Size of slice in `key`
    key_size: I,
    /// The newly-created node, to the right of `key`
    ///
    /// This will be inserted in `lhs`'s parent as its sibling to the right, with `key` in between
    /// them.
    ///
    /// The height of `rhs` is *always* equal to the height of `lhs`.
    rhs: NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>,

    /// Total size of the node that was split into `lhs` and `rhs`, before we inserted anything or
    /// split it
    old_size: I,

    /// Handle to the slice that was inserted, plus which one of `lhs` or `rhs` it's in, only if it
    /// the inserted slice is actually in `key`
    ///
    /// ## Safety
    ///
    /// The slice behind the handle *cannot* be accessed until the borrow at `node` is released.
    insertion: Option<BubbledInsertion<I, S, P, M>>,

    /// Cursor representing the path to the insertion
    ///
    /// The cursor does not not yet include the path segment through `lhs` or `rhs`; we'd need to
    /// have already taken into account the shifts from inserting `key` at this level.
    partial_cursor: C,
}

/// (*Internal*) An inserted [`SliceHandle`] and where to find it ; for [`BubbledInsertState`]
struct BubbledInsertion<I, S, P: RleTreeConfig<I, S>, const M: usize> {
    /// Flag for which child the bubbled insertion is in
    ///
    /// This is tracked so that we can accurately update the cursor as we go up the tree.
    side: Side,
    /// Handle to the insertion
    handle: SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>,
}

/// (*Internal*) Helper type for [`BubbledInsertState`] and [`BubbledInsertion`]
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
enum Side {
    Left = 0,
    Right = 1,
}

/// (*Internal*) Helper struct to carry information for [`insert_internal`] and related functions
/// as we traverse up the tree to propagate insertion results
///
/// [`insert_internal`]: RleTree::insert_internal
struct PostInsertTraversalState<'t, C, I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S>,
{
    /// Handle to the slice that was inserted
    ///
    /// ## Safety
    ///
    /// The slice behind this handle *cannot* be accessed until the borrow at `node` is released.
    inserted_slice: SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>,

    /// The key or child containing the insertion as we're traversing upwards, alongside the typed
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
    /// tracked by a running total of the distance from each key back to the insertion child/key,
    /// adding the distance between keys each time. So the value is then
    /// `key.sub_left(pos).add_left(sum)`.
    ///
    /// This careful dance is only there because we have to make sure that we uphold the guarantees
    /// we provide for directional arithmetic -- we can't just `add_left(size)` because that's not
    /// where the slice is.
    ///
    /// If the usage around this is written in *just* the right way, the compiler should realize
    /// what's going on and optimize it to (approximately) `key += size`.
    child_or_key: ChildOrKey<
        // Child: must be internal
        (u8, NodeHandle<ty::Internal, borrow::Mut<'t>, I, S, P, M>),
        // Key: unknown type
        (u8, NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>),
    >,

    /// Optional override in case the start position of the size change is not equal to the start
    /// position given by `child_or_key`
    override_pos: Option<I>,

    /// The old size of the child or key containing the insertion
    old_size: I,
    /// The new size of the child or key containing the insertion. This is provided separately so
    /// that it can be updated *after* the original size has been recorded
    new_size: I,

    /// Cursor representing the path to the deepest node containing the insertion
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

/// (*Internal*) The result of an upward step from [`PostInsertTraversalState::do_upward_step`]
///
/// This type is extracted out in order to allow a common interface for functions that may need to
/// handle their own post-insert traversals
enum PostInsertTraversalResult<'t, C, I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S>,
{
    /// Traversal has not yet gone through the root (although it *may* be this field's `node`), and
    /// so should continue
    Continue(PostInsertTraversalState<'t, C, I, S, P, M>),
    /// Traversal has finished, with all nodes updated. The cursor and reference to the slice have
    /// been returned
    Root(C, SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>),
    /// The insertion resulted in creating a new root node, but that needs to be handled by
    /// [`insert_internal`], because no other method has access to the existing root with an
    /// `Owned` borrow
    ///
    /// [`insert_internal`]: RleTree::insert_internal
    NewRoot {
        /// The *completed* cursor to the insertion, taking into account the locations that `lhs`
        /// and `rhs` will have.
        cursor: C,
        /// The current root node, which will become the left-hand child in the new root
        lhs: NodeHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
        /// Key to be inserted between `lhs` and `rhs`
        key: node::Key<I, S, P>,
        key_size: I,
        /// New, right-hand node
        rhs: NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>,
        /// If the insertion is not `key`, then a handle on the inserted slice
        insertion: Option<SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>>,
    },
}

#[cfg(test)]
impl<'t, C, I, S, P, const M: usize> Debug for BubbledInsertState<'t, C, I, S, P, M>
where
    P: RleTreeConfig<I, S>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("BubbledInsertState")
            .field("lhs@", &self.lhs.ptr())
            .field("lhs", self.lhs.typed_debug())
            .field("key", &self.key)
            .field("key_size", self.key_size.fallible_debug())
            .field("rhs@", &self.rhs.ptr())
            .field("rhs", self.rhs.typed_debug())
            .field("old_size", &self.old_size.fallible_debug())
            .field("insertion", &self.insertion)
            .finish()
    }
}

#[cfg(test)]
impl<I, S, P: RleTreeConfig<I, S>, const M: usize> Debug for BubbledInsertion<I, S, P, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("BubbledInsertion")
            .field("side", &self.side)
            .field("handle", &NoDebugImpl)
            .finish()
    }
}

#[cfg(test)]
impl<'t, C, I, S, P, const M: usize> Debug for PostInsertTraversalState<'t, C, I, S, P, M>
where
    P: RleTreeConfig<I, S>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let (idx_name, idx, node) = match &self.child_or_key {
            ChildOrKey::Key((k_idx, node)) => ("key_idx", k_idx, node.typed_debug()),
            ChildOrKey::Child((c_idx, node)) => ("child_idx", c_idx, node.typed_debug()),
        };

        f.debug_struct("PostInsertTraversalState")
            .field("node", node)
            .field(idx_name, idx)
            .field("override_pos", self.override_pos.fallible_debug())
            .field("old_size", self.old_size.fallible_debug())
            .field("new_size", self.new_size.fallible_debug())
            .finish()
    }
}

#[cfg(test)]
impl<'t, C, I, S, P, const M: usize> Debug for PostInsertTraversalResult<'t, C, I, S, P, M>
where
    P: RleTreeConfig<I, S>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Continue(state) => f.debug_tuple("Continue").field(&state).finish(),
            Self::Root(_, _) => f.debug_tuple("Root").field(&NoDebugImpl).finish(),
            Self::NewRoot {
                lhs,
                key,
                key_size,
                rhs,
                ..
            } => f
                .debug_struct("NewRoot")
                .field("lhs", lhs.typed_debug())
                .field("key", &key)
                .field("key_size", key_size.fallible_debug())
                .field("rhs", rhs.typed_debug())
                .finish(),
        }
    }
}

/// (*Internal*) A slice / size pairing
struct SliceSize<I, S> {
    slice: S,
    size: I,
}

impl<I, S> SliceSize<I, S> {
    fn new(slice: S, size: I) -> Self {
        SliceSize { slice, size }
    }
}

#[cfg(test)]
impl<I, S> Debug for SliceSize<I, S> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("SliceSize")
            .field("slice", self.slice.fallible_debug())
            .field("size", self.size.fallible_debug())
            .finish()
    }
}

/// (*Internal*) Performs a single step of searching for the location of the target index within
/// the tree
///
/// The hint is typically produced by a [`Cursor`], and if present, serves as a first guess to
/// avoid a possibly-expensive binary search.
///
/// Either item in the [`ChildOrKey`] returned contains the index of the child or key containing
/// `target`, and the start position of that child or key, relative to the base of `node`.
fn search_step<'t, I: Index, S, P: RleTreeConfig<I, S>, const M: usize>(
    node: NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
    hint: Option<PathComponent>,
    target: I,
) -> ChildOrKey<(u8, I), (u8, I)> {
    let subtree_size = node.leaf().subtree_size();
    debug_assert!(target <= subtree_size);

    if let Some(PathComponent { child_idx }) = hint {
        'bad_hint: loop {
            let node = match node.typed_ref() {
                Type::Leaf(_) => break 'bad_hint,
                Type::Internal(n) => n,
            };

            // If the hint's good, we're looking at something like:
            //
            //    (k = child_idx -1)                     (k = child_idx)
            //   |- left-hand key -|                  |- right-hand key -|
            //                     |--- this child ---|
            //                        (c = child_idx)

            // num children is num keys + 1, max child index is equal to `node.len_keys()`.
            if child_idx > node.leaf().len() {
                break 'bad_hint;
            }

            // SAFETY: we just guaranteed that `child_idx <= node.leaf().len()`, which is all
            // that's required by `child_size`.
            let child_size = unsafe { node.child_size(child_idx) };
            let next_key_pos = node.leaf().try_key_pos(child_idx).unwrap_or(subtree_size);
            let child_pos = next_key_pos.sub_right(child_size);

            if child_pos <= target && target < next_key_pos {
                return ChildOrKey::Child((child_idx, child_pos));
            }

            break 'bad_hint;
        }
    }

    // Either there was no hint, or the hint was bad. We have to fully search the node
    //
    // TODO: At what point does it become faster to use a linear search? This should be determined
    // with heuristics based on `M` and `size_of::<I>()`, after benchmarking. Possibly we have a
    // separate parameter in `Index` that provides an alternate cost factor.
    let k_idx = match node.leaf().keys_pos_slice().binary_search(&target) {
        // Err(0) means that there's no key with position >= target, so the insertion point is
        // somewhere in that child. Because it's the first child in the node, its relative
        // position is I::ZERO
        Err(0) => return ChildOrKey::Child((0, I::ZERO)),
        // target > key_pos(i - 1); either key_pos(i) doesn't exist, or key_pos(i) > target.
        Err(i) => (i - 1) as u8,
        // target == key_pos(i) -- return the key
        Ok(i) => {
            let k_idx = i as u8;
            // SAFETY: `Ok(i)` means that it's an index within `keys_pos_slice`
            let k_pos = unsafe { node.leaf().key_pos(k_idx) };
            return ChildOrKey::Key((k_idx, k_pos));
        }
    };

    // At this point, we're either looking at a key `k_idx` or child `k_idx + 1` -- we have to look
    // at the startpoint of child `k_idx + 1` to determine where the target is.
    let next_key_pos = node.leaf().try_key_pos(k_idx + 1).unwrap_or(subtree_size);
    // SAFETY: `k_idx + 1` is a valid child index because `k_idx` is a valid key index, as
    // guaranteed by the binary search.
    let next_child_size = unsafe { node.try_child_size(k_idx + 1).unwrap_or(I::ZERO) };
    debug_assert!(target <= next_key_pos);

    let next_child_start = next_key_pos.sub_right(next_child_size);
    match target < next_child_start {
        // SAFETY: `k_idx` is known to be a valid key index from the binary search.
        true => ChildOrKey::Key((k_idx, unsafe { node.leaf().key_pos(k_idx) })),
        false => ChildOrKey::Child((k_idx + 1, next_child_start)),
    }
}

impl<'t, C, I, S, P, const M: usize> PreInsertTraversalState<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: Slice<I>,
    P: SupportsInsert<I, S> + RleTreeConfig<I, S>,
{
    /// Performs a single step of a downward search for an insertion point
    fn do_search_step(mut self) -> Result<InsertSearchResult<'t, I, S, P, M>, Self> {
        let cursor_hint = self.cursor_iter.as_mut().and_then(|iter| iter.next());

        let mut search_result = search_step(self.node.borrow(), cursor_hint, self.target);
        // If the result was the start of a key, we actually want to insert in the child
        // immediately before it.
        if let ChildOrKey::Key((k_idx, k_pos)) = search_result {
            if k_pos == self.target {
                // SAFETY: `k_idx` is guaranteed to be a valid key index, so it's therefore a valid
                // child index as well.
                let child_size = unsafe { self.node.try_child_size(k_idx).unwrap_or(I::ZERO) };
                let child_pos = k_pos.sub_right(child_size);
                search_result = ChildOrKey::Child((k_idx, child_pos));
            }
        }

        let hint_was_good = matches!(
            (search_result, cursor_hint),
            (ChildOrKey::Child((c_idx, _)), Some(h)) if c_idx == h.child_idx
        );

        if !hint_was_good {
            self.cursor_iter = None;
        }

        match search_result {
            // A successful result. We don't need the adjacent keys anymore because the insertion
            // will split `key_idx`
            ChildOrKey::Key((k_idx, k_pos)) => {
                // SAFETY: The call to `find_insertion_point` guarantees that return values of
                // `ChildOrKey::Key` contains an index within the bounds of the node's keys.
                let handle = unsafe { self.node.into_slice_handle(k_idx) };

                Ok(ChildOrKey::Key(SplitKeyInsert {
                    handle,
                    pos_in_key: self.target.sub_left(k_pos),
                }))
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
                        self.adjacent_keys.lhs = Some(unsafe { self.node.split_slice_handle(ki) });
                    }

                    // `child_idx` is the rightmost child if it's *equal* to `node.len_keys()`.
                    // Anything less means we should get the key to the right of it.
                    if child_idx < self.node.leaf().len() {
                        // SAFETY: same as above. The right-hand side key is *at* key index
                        // `child_idx` because child indexing starts left of key indexing
                        self.adjacent_keys.rhs =
                            Some(unsafe { self.node.split_slice_handle(child_idx) });
                    }
                }

                match self.node.into_typed() {
                    // If it's a leaf node, we've found an insertion point between two keys
                    Type::Leaf(leaf) => {
                        return Ok(ChildOrKey::Child(LeafInsert {
                            node: leaf,
                            adjacent_keys: self.adjacent_keys,
                            // Two things here: (a) child indexes are offset to occur in the gaps
                            // *before* the key of the same index (i.e. child0, then key0, then child1)
                            // and (b) the guarantees of `find_insertion_point` mean that th
                            new_k_idx: child_idx,
                        }));
                    }
                    // Otherwise, we should keep going:
                    Type::Internal(internal) => {
                        let pos_in_child = self.target.sub_left(child_pos);
                        self.target = pos_in_child;
                        // SAFETY: This function only requires that `child_idx` be in bounds. This
                        // is condition is guaranteed by `find_insertion_point`.
                        self.node = unsafe { internal.into_child(child_idx) };
                        Err(self)
                    }
                }
            }
        }
    }
}

// This block is only private stuff; doc links can reference other private stuff
#[allow(rustdoc::private_intra_doc_links)]
/// Internal-only helper functions for implementing [`insert_internal`](RleTree::insert_internal)
impl<I, S, P, const M: usize> RleTree<I, S, P, M>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsInsert<I, S>,
{
    /// (*Internal*) Tries to insert the slice by joining with either adjacent key, returning
    /// `Err(slice)` joining fails
    ///
    /// If joining succeeds, the insertion is processed upwards until all structural changes to the
    /// tree are resolved, and then the [`PostInsertTraversalState`] is returned.
    fn do_insert_try_join<'t, C: Cursor>(
        store: &mut <P as RleTreeConfig<I, S>>::SliceRefStore,
        slice: S,
        slice_size: I,
        adjacent_keys: AdjacentKeys<'t, I, S, P, M>,
    ) -> Result<PostInsertTraversalState<'t, C, I, S, P, M>, S> {
        // Broadly speaking, there's four cases here:
        //  1. `slice` doesn't join with either of `adjacent_keys` -- returns Err(slice)
        //  2. `slice` only joins with `adjacent_keys.lhs`
        //  3. `slice` only joins with `adjacent_keys.rhs`
        //  4. `slice` joins with both keys
        // The logic for cases 2 & 3 are essentially the same, so our labeled loop below produces a
        // value to be handled for those cases and returns for 1 & 4.

        // Fast path.
        if !S::MAY_JOIN {
            return Err(slice);
        }

        let lhs_with_size = adjacent_keys.lhs.map(|h| (h.slice_size(), h));
        let rhs_with_size = adjacent_keys.rhs.map(|h| (h.slice_size(), h));

        // TODO: This code could be more concise, probably by merging `None` with "couldn't join".
        // That *may* be slightly more difficult to reason about, though.
        let (handle, old_size, new_size) = 'single_join: loop {
            match (lhs_with_size, rhs_with_size) {
                // Nothing to do.
                (None, None) => return Err(slice),

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
                            // Can't fill the hole in `lhs` yet; stil need to try to join with
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
                            // Fill `lhs` so that we can perform the deletion at `rhs`
                            lhs.fill_hole(new);
                            // Also, update `store` so that we appropriately redirect `rhs` to
                            // `lhs`.
                            rhs.redirect_to(&lhs, store);

                            let new_lhs_size = mid_size.add_right(rhs_size);
                            return Ok(Self::handle_join_deletion(
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

        Ok(PostInsertTraversalState {
            // SAFETY: `clone_slice_ref` requires that we not access `inserted_slice` until the
            // borrow used by `node` is released. This is mirrored in the safety conditions in
            // `PostInsertTraversalState` itself.
            inserted_slice: unsafe { handle.clone_slice_ref() },
            child_or_key: ChildOrKey::Key((handle.idx, handle.node)),
            override_pos: None,
            old_size,
            new_size,
            partial_cursor: C::new_empty(),
        })
    }

    /// (*Internal*) Performs an insertion into the leaf node, with the key(s) at `new_key_idx`
    ///
    /// A second key to insert may also be provided, which will be inserted after `fst`. If a
    /// second key is provided, only the first key will be used for the slice reference.
    ///
    /// `override_lhs_size` is used exclusively by [`insert_rhs`] in order to ensure that the key
    /// positions (and resulting increases in size) are correct from the moment of insertion.
    ///
    /// ## Safety
    ///
    /// Callers must ensure that `new_key_idx <= node.leaf().len()`, and that if
    /// `override_lhs_size` is provided, `new_key_idx != 0`.
    unsafe fn do_insert_no_join<'t, C: Cursor>(
        store: &mut P::SliceRefStore,
        mut node: NodeHandle<ty::Leaf, borrow::Mut<'t>, I, S, P, M>,
        mut new_key_idx: u8,
        mut override_lhs_size: Option<I>,
        fst: SliceSize<I, S>,
        snd: Option<SliceSize<I, S>>,
    ) -> Result<PostInsertTraversalResult<'t, C, I, S, P, M>, BubbledInsertState<'t, C, I, S, P, M>>
    {
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

        // "easy" case: just add the slice(s) to the node
        if node.leaf().len() < node.leaf().max_len() - one_if_snd {
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

            // We can allow `PostInsertTraversalState::do_upward_step` to shift the key positions,
            // because the "current" size of the newly inserted slice is zero, so the increase from
            // zero to `slice_size` in `do_upward_step` will have the desired effect anyways.
            return Ok(PostInsertTraversalResult::Continue(
                PostInsertTraversalState {
                    // SAFETY: `clone_slice_ref` requires that the handle isn't used until after
                    // `node` is dropped, which is guaranteed by the safety bounds of
                    // `inserted_slice`.
                    inserted_slice: unsafe { slice_handle.clone_slice_ref().erase_type() },
                    child_or_key: ChildOrKey::Key((
                        new_key_idx + one_if_snd,
                        slice_handle.node.erase_type(),
                    )),
                    override_pos,
                    old_size,
                    new_size,
                    partial_cursor: C::new_empty(),
                },
            ));
        }

        // Otherwise, "hard" case: split the node on insertion

        // There's a couple cases here to determine where we split the node, depending on
        // `new_key_idx`.
        //
        // If we were to insert the new slice *before* splitting, we'd always extract the `M`th key
        // as the midpoint (max keys is 2*M, so splitting is on 2*M + 1 keys, splitting into two
        // nodes with size M, plus a midpoint)
        //
        // We can't insert the slice before splitting, but we still need to make sure the right
        // number of values end up in each node *after* the insertion. So depending on the value of
        // `new_key_idx`, we'll either split at `M` (`new_key_idx > M`) or `M - 1`
        // (`new_key_idx <= M`). If `new_key_idx == M`, then we'll end up adding the midpoint back
        // to the end of the left-hand node.
        let midpoint_idx = if new_key_idx > M as u8 {
            M as u8
        } else {
            (M - 1) as u8
        };

        // SAFETY: `split` requires that `midpoint_idx` is properly within the bounds of the node,
        // which we guarantee by the logic above, because `max_len = 2 * M`
        let (midpoint_key, mut rhs) = unsafe { node.split(midpoint_idx, store) };

        let old_node_size = node.leaf().subtree_size();
        let rhs_start = rhs.leaf().try_key_pos(0).unwrap_or(old_node_size);
        let midpoint_size;

        // We're handling `override_lhs_size` for some cases here. It's a little tricky, so we'll
        // illustrate each case. For the examples, let's say M = 3. This is our start node:
        //   ╔═══╤═══╤═══╤═══╤═══╤═══╗
        //   ║ A │ B │ C │ D │ E │ F ║
        //   ╚═══╧═══╧═══╧═══╧═══╧═══╝
        // We'll suppose there's a single insertion (having a second doesn't change the logic
        // here). It'll be labelled with a slash - "/"
        //
        // At the end of this block, we guarantee that `handled_override.is_some()` implies that
        // the left-hand key is in the same *new* node as the inserted value.
        match override_lhs_size {
            Some(s) if new_key_idx == M as u8 => {
                // Continuing the example, our final result if `new_key_idx == M` will be:
                //                 ╔═══╗
                //                 ║ / ║
                //   ╔═══╤═══╤═══╗ ╚═══╝ ╔═══╤═══╤═══╗
                //   ║ A │ B │ C ║       ║ D │ E │ F ║
                //   ╚═══╧═══╧═══╝       ╚═══╧═══╧═══╝
                // ... but currently, it's:
                //                 ╔═══╗
                //                 ║ C ║
                //   ╔═══╤═══╗     ╚═══╝ ╔═══╤═══╤═══╗
                //   ║ A │ B ║           ║ D │ E │ F ║
                //   ╚═══╧═══╝           ╚═══╧═══╧═══╝
                // ... so our left-hand key is in the midpoint. We can perform the necessary
                // correction just by changing the midpoint's size.
                midpoint_size = s;
                override_lhs_size = None;
            }
            Some(s) if new_key_idx == M as u8 + 1 => {
                // If `new_key_idx == M + 1`, then our final result should be:
                //                 ╔═══╗
                //                 ║ D ║
                //   ╔═══╤═══╤═══╗ ╚═══╝ ╔═══╤═══╤═══╗
                //   ║ A │ B │ C ║       ║ / │ E │ F ║
                //   ╚═══╧═══╧═══╝       ╚═══╧═══╧═══╝
                // ... but currently, it's:
                //                 ╔═══╗
                //                 ║ D ║
                //   ╔═══╤═══╤═══╗ ╚═══╝     ╔═══╤═══╗
                //   ║ A │ B │ C ║           ║ E │ F ║
                //   ╚═══╧═══╧═══╝           ╚═══╧═══╝
                // ... so our left-hand key is in the midpoint, and will stay there. We can do the
                // necessary correction *just* by changing the size of the midpoint (like above).
                midpoint_size = s;
                override_lhs_size = None;
            }
            // Nothing out of the ordinary yet; calculate the midpoint size as normal.
            _ => midpoint_size = rhs_start.sub_left(midpoint_key.pos),
        }

        node.set_subtree_size(midpoint_key.pos);

        // Before we do anything else, we'll update the positions in `rhs`. It's technically
        // possible to get away with only updating these once, but that's *really* complicated.
        //
        // SAFETY: `shift_keys_decrease` requires that `from <= rhs.leaf().len()`, which is always
        // true, and `rhs.as_mut()` requires unique access, which it has because `rhs` was just
        // created.
        unsafe {
            let opts = ShiftKeys {
                from: 0,
                pos: I::ZERO,
                old_size: rhs_start,
                new_size: I::ZERO,
            };
            shift_keys_decrease(rhs.as_mut(), opts);
        }

        if new_key_idx == M as u8 {
            // put `midpoint_key` back into `node`, use the desired insertion as the midpoint
            //
            // We've already handled `override_lhs_size`; see above.
            let new_lhs_subtree_size = node.leaf().subtree_size().add_right(midpoint_size);
            // SAFETY: `push_key` requires only that there's space. This is guaranteed by
            // having a value of `midpoint_idx` that removes *some* values from `node`.
            unsafe { node.push_key(store, midpoint_key, new_lhs_subtree_size) };

            // If we have a second slice that we're adding, it should go at the start of `rhs`,
            // which is directly after the midpoint.
            if let Some(SliceSize { slice, size }) = snd {
                // SAFETY: `insert_key` requires that the key index less than or equal to the
                // current length, which is always true for zero. `shift_keys_increase` requires
                // the same for `from`, which is true after the insertion completes. The calls to
                // `as_mut` require uniqueness, which is guaranteed because we just created it
                unsafe {
                    let pos = rhs.as_mut().insert_key(store, 0, slice);

                    let opts = ShiftKeys {
                        from: 1,
                        pos,
                        old_size: I::ZERO,
                        new_size: size,
                    };
                    shift_keys_increase(rhs.as_mut(), opts);
                }
            }

            Err(BubbledInsertState {
                lhs: node.erase_type(),
                key: node::Key {
                    pos: rhs_start,
                    slice: fst.slice,
                    ref_id: Default::default(),
                },
                key_size: fst.size,
                rhs: rhs.erase_type(),
                old_size: old_node_size,
                insertion: None,
                partial_cursor: C::new_empty(),
            })
        } else {
            let (side, insert_into) = if new_key_idx < M as u8 {
                // SAFETY: `as_mut` requires unqiue access; we're reborrowing and already have it.
                (Side::Left, unsafe { node.as_mut() })
            } else {
                // Adjust the position of `slice` to be relative to the right-hand node
                new_key_idx -= M as u8 + 1;

                // SAFETY: `as_mut` requires unique access; `rhs` was just created.
                (Side::Right, unsafe { rhs.as_mut() })
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
                    // SAFETY: the caller originally guarantees that `override_lhs_size.is_some()`
                    // implies that `new_key_idx != 0`. Our little bit of handling above extends
                    // this to apply to the two split halves of the original node -- the case where
                    // the insertion is at the start of `rhs` gets explicitly handled with
                    // `override_lhs_size` set to `None` before control flow gets here.
                    let handle = unsafe { insert_into.borrow().into_slice_handle(new_key_idx - 1) };

                    (handle.key_pos(), handle.slice_size(), s)
                }
            };

            // SAFETY: the calls to `insert_key` and `shift_keys_increase` together require that
            // `new_key_idx <= insert_into.leaf().len()`, which is guaranteed by the values we
            // chose for `midpoint_idx` and our comparison with `M` above. In the case where `snd`
            // is not `None`, the same guarantees are made, but shifted over by one.
            //
            // The call to `set_single_key_pos`, if we make it, requires that `new_key_idx + 1` is
            // a valid key, which we know is true because we just inserted it.
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

                    // Update the position of `snd` because currently it's the same as `fst`. We
                    // could make an extra call to `set_key_poss_with` but it's easier to just set
                    // the value directly.
                    let snd_pos = fst_pos.add_right(fst.size);
                    insert_into.set_single_key_pos(new_key_idx + 1, snd_pos);
                }

                let opts = ShiftKeys {
                    from,
                    pos,
                    old_size,
                    new_size,
                };
                shift_keys_increase(insert_into, opts);
            }

            // SAFETY: `into_slice_handle` requires that `new_key_idx < insert_into.leaf().len()`,
            // which is guaranteed by `insert_key`. `clone_slice_ref` requires that we not use it
            // until the other borrows on the tree are gone, which the safety docs for
            // `BubbledInsertState` guarantees.
            let inserted_handle = unsafe {
                insert_into
                    .borrow()
                    .into_slice_handle(new_key_idx)
                    .clone_slice_ref()
            };

            Err(BubbledInsertState {
                lhs: node.erase_type(),
                key: midpoint_key,
                key_size: midpoint_size,
                rhs: rhs.erase_type(),
                old_size: old_node_size,
                insertion: Some(BubbledInsertion {
                    side,
                    handle: inserted_handle.erase_type(),
                }),
                partial_cursor: C::new_empty(),
            })
        }
    }

    /// (*Internal*) Handles the deletion of `rhs` due to insertion that joins with `lhs`
    ///
    /// It's expected that there will be a hole at `rhs`, from the value that had joined with
    /// `lhs`.
    fn handle_join_deletion<'b, C: Cursor>(
        lhs: SliceHandle<ty::Unknown, borrow::Mut<'b>, I, S, P, M>,
        old_lhs_size: I,
        new_lhs_size: I,
        rhs: SliceHandle<ty::Unknown, borrow::Mut<'b>, I, S, P, M>,
        old_rhs_size: I,
    ) -> PostInsertTraversalState<'b, C, I, S, P, M> {
        todo!()
    }

    /// Inserts the slice into the key, splitting it and attempting to re-join
    fn split_insert<'t, C: Cursor>(
        store: &mut P::SliceRefStore,
        mut key_handle: SliceHandle<ty::Unknown, borrow::Mut<'t>, I, S, P, M>,
        split_pos: I,
        slice: S,
        slice_size: I,
    ) -> PostInsertTraversalResult<'t, C, I, S, P, M> {
        let key_size = key_handle.slice_size();

        let rhs = key_handle.with_slice(|slice| slice.split_at(split_pos));
        let rhs_size = key_size.sub_left(split_pos);

        let (mut lhs_handle, lhs_size) = (key_handle, split_pos);

        // Fast path: no need to try joining.
        if !S::MAY_JOIN {
            let (mid, mid_size) = (slice, slice_size);
            let fst = SliceSize::new(mid, mid_size);
            let snd = SliceSize::new(rhs, rhs_size);
            return Self::insert_rhs(store, lhs_handle, lhs_size, fst, Some(snd));
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

                PostInsertTraversalResult::Continue(PostInsertTraversalState {
                    // SAFETY: `clone_slice_ref` requires that we not access `inserted_slice` until
                    // the borrow used by `node` is released. This is mirrored in the safety
                    // conditions in `PostInsertTraversalState` itself.
                    inserted_slice: unsafe { lhs_handle.clone_slice_ref() },
                    child_or_key: ChildOrKey::Key((lhs_handle.idx, lhs_handle.node)),
                    override_pos: None,
                    old_size: lhs_size.add_right(rhs_size),
                    new_size: mid_size.add_right(rhs_size),
                    partial_cursor: C::new_empty(),
                })
            }
            // Couldn't join with lhs, but slice + rhs joined. Still need to insert those:
            Ok(new) => {
                let size = mid_size.add_right(rhs_size);
                Self::insert_rhs(store, lhs_handle, lhs_size, SliceSize::new(new, size), None)
            }
            // Couldn't join with rhs, but did join with lhs. Still need to insert rhs:
            Err((l, r)) if joined_with_lhs => {
                lhs_handle.fill_hole(l);
                let fst = SliceSize::new(r, rhs_size);
                Self::insert_rhs(store, lhs_handle, mid_size, fst, None)
            }
            // Couldn't join with either lhs or rhs. Need to insert both original slice AND new rhs
            Err((m, r)) => {
                let fst = SliceSize::new(m, mid_size);
                let snd = SliceSize::new(r, rhs_size);

                Self::insert_rhs(store, lhs_handle, lhs_size, fst, Some(snd))
            }
        }
    }

    /// Inserts the slice to the immediate right of another slice, recording the changed size from
    /// both the left (present) and right (inserted) sides
    fn insert_rhs<'b, C: Cursor>(
        store: &mut P::SliceRefStore,
        right_of: SliceHandle<ty::Unknown, borrow::Mut<'b>, I, S, P, M>,
        new_lhs_size: I,
        fst: SliceSize<I, S>,
        snd: Option<SliceSize<I, S>>,
    ) -> PostInsertTraversalResult<'b, C, I, S, P, M> {
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
        let override_lhs_size = match lhs_height == leaf_height {
            false => None,
            true => Some(new_lhs_size),
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
        let res = unsafe {
            Self::do_insert_no_join::<C>(store, next_leaf, insert_idx, override_lhs_size, fst, snd)
        };

        // If `override_lhs_size` was provided, the change in `lhs`'s size was entirely handled by
        // `do_insert_no_join`, so we don't need to do anything else here. Finish up the bubbled
        // insert state if required
        if override_lhs_size.is_some() {
            let mut res = res;
            loop {
                match res {
                    Ok(post_insert) => return post_insert,
                    Err(bubble_state) => res = bubble_state.do_upward_step(store, None),
                }
            }
        }

        let mut handled_shift = false;

        let post_insert_result = match res {
            Ok(s) => s,
            Err(mut bubble_state) => loop {
                let mut maybe_shift_lhs = None;

                // If the bubble state *will* change the node at the height of `lhs`, then we need
                // to change the way that it internally calculates positions. The case for equal
                // heights is already handled above with `override_lhs_size`.
                if bubble_state.lhs.height() + 1 == lhs_height {
                    handled_shift = true;

                    maybe_shift_lhs = Some(BubbleLhs {
                        pos: lhs_pos,
                        old_size: old_lhs_size,
                        new_size: new_lhs_size,
                    });
                }

                // SAFETY: `do_upward_step` requires: if `maybe_shift_lhs` is `Some(_)`, then there
                // must be a key to the left of the child `bubble_state.lhs`. This is guaranteed by
                // our logic above, where `maybe_shift_lhs` is only set to `Some(_)` if `right_of`
                // is at the height of `bubble_state.lhs`'s parent.
                match bubble_state.do_upward_step(store, maybe_shift_lhs) {
                    Ok(post_insert) if handled_shift => return post_insert,
                    Ok(post_insert) => break post_insert,
                    Err(b) => bubble_state = b,
                }
            },
        };

        // If we get here, we're handling a `PostInsertTraversalResult` until we get up to the
        // height of the original left-hand side.

        let mut post_insert_result = post_insert_result;

        loop {
            let mut state = match post_insert_result {
                PostInsertTraversalResult::Continue(s) => s,
                // FIXME: these shouldn't be possible, right?
                r => return r,
            };

            let node = match &mut state.child_or_key {
                ChildOrKey::Key((_, node)) => node,
                ChildOrKey::Child((_, internal_node)) => internal_node.untyped_mut(),
            };

            if node.height() != lhs_height {
                post_insert_result = state.do_upward_step();
                continue;
            }

            // node.height() == lhs_height: account for the change in `lhs` size
            //
            // The easiest way for us to do this is to just change the values of `child_or_key` (to
            // the left-hand key) and `old_size`/`new_size` (to incorporate the changes to `lhs`).
            //
            // Because we just inserted to the immediate right-hand side of `right_of`, we're
            // expecting `state.child_or_key` to either represent the key or child immediately to
            // the right (i.e. `ChildOrKey::{Key,Child}(lhs_k_idx + 1)`). Updating the values then
            // sets it to the `lhs` key.

            debug_assert!(
                matches!(&state.child_or_key, ChildOrKey::Child((i, _)) | ChildOrKey::Key((i, _)) if *i == lhs_k_idx + 1)
                    || (leaf_height == lhs_height
                        && matches!(&state.child_or_key, ChildOrKey::Key((i, _)) if *i == lhs_k_idx + 2))
            );

            let old_lhs_end = lhs_pos.add_right(old_lhs_size);
            let new_lhs_end = lhs_pos.add_right(new_lhs_size);

            let state_pos = state
                .override_pos
                .unwrap_or_else(|| match &state.child_or_key {
                    // SAFETY: docs for `PostInsertTraversalState` guarantee that `k_idx` is within
                    // bounds for `node`.
                    ChildOrKey::Key((k_idx, node)) => unsafe { node.leaf().key_pos(*k_idx) },
                    ChildOrKey::Child((c_idx, node)) => {
                        let next_key_pos = node
                            .leaf()
                            .try_key_pos(*c_idx)
                            .unwrap_or_else(|| node.leaf().subtree_size());
                        next_key_pos.sub_left(state.old_size)
                    }
                });

            let diff = state_pos.sub_left(old_lhs_end);
            state.old_size = state.old_size.add_left(diff).add_left(old_lhs_size);
            state.new_size = state.new_size.add_left(diff).add_left(new_lhs_size);
            state.override_pos = Some(lhs_pos);

            // We're currently at key `lhs_k_idx`, so we want to manually shift all of the keys
            // from `lhs_k_idx + 1` to the last key before what's automatically handled.
            //
            // However, because we know we just inserted to the right-hand side of `lhs`, we know
            // that we only actually need to shift a single key -- if at all. We shift the key at
            // `lhs_k_idx + 1` so long as `state.child_or_key` is `ChildOrKey::Key((_, _))`.
            if let ChildOrKey::Key((_, node)) = &mut state.child_or_key {
                // SAFETY: the calls to `key_pos` and `set_single_key_pos` both require only that
                // `lhs_k_idx + 1` is a valid key index. Because (a) `node` is at the same height
                // as `lhs` and (b) any splits are not continued at this height, we know that the
                // index in `ChildOrKey::Key` must be equal to `lhs_k_idx + 1`. The documentation
                // for `PostInsertTraversalState` guarantees that such a key index will be valid.
                unsafe {
                    let pos = node.leaf().key_pos(lhs_k_idx + 1);
                    let new_pos = pos.sub_left(old_lhs_end).add_left(new_lhs_end);
                    node.set_single_key_pos(lhs_k_idx + 1, new_pos);
                }
            }
            return PostInsertTraversalResult::Continue(state);
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct BubbleLhs<I> {
    pos: I,
    old_size: I,
    new_size: I,
}

impl<'t, C, I, S, P, const M: usize> BubbledInsertState<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    fn do_upward_step(
        mut self,
        store: &mut <P as RleTreeConfig<I, S>>::SliceRefStore,
        mut shift_lhs: Option<BubbleLhs<I>>,
    ) -> Result<PostInsertTraversalResult<'t, C, I, S, P, M>, Self> {
        let lhs_size = self.lhs.leaf().subtree_size();
        let new_total_size = lhs_size
            .add_right(self.key_size)
            .add_right(self.rhs.leaf().subtree_size());

        let (mut parent, lhs_child_idx) = match self.lhs.into_parent() {
            Ok((p, idx)) => (p, idx),

            // Base case: no parent, need to create a new `Internal` root. That has to be done from
            // the main function, `insert_internal`, so we need to return a result indicating that.
            Err(lhs) => {
                assert!(shift_lhs.is_none(), "internal error: this is a bug");

                let insertion = self.insertion.map(|insertion| {
                    self.partial_cursor.prepend_to_path(PathComponent {
                        // `Side` is explicitly tagged so that Side::Left = 0 and Side::Right = 1
                        child_idx: insertion.side as u8,
                    });
                    insertion.handle
                });

                return Ok(PostInsertTraversalResult::NewRoot {
                    cursor: self.partial_cursor,
                    lhs,
                    key: self.key,
                    key_size: self.key_size,
                    rhs: self.rhs,
                    insertion,
                });
            }
        };

        // Insert `self.key` and `self.rhs` into `parent`, just after `lhs_child_idx`. If there
        // isn't room, we should split `parent` and select a new midpoint key.
        //
        // Either way, we'll need to add where `insertion` is in the final node(s) to the cursor
        let added_size = self.key_size.add_right(self.rhs.leaf().subtree_size());

        // No split, only insert:
        //
        // Do the minimal amount of work here, and let `PostInsertTraversalState::do_upward_step`
        // handle the rest.
        if parent.leaf().len() < parent.leaf().max_len() {
            let (override_pos, old_size, new_size, lhs_child_pos) = match shift_lhs {
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
                parent.insert_key_and_child(store, lhs_child_idx, self.key, self.rhs);
                let midpoint_pos = lhs_child_pos.add_right(lhs_size);
                parent.set_single_key_pos(lhs_child_idx, midpoint_pos);
            };

            // Update the cursor -- and the insertion handle if necessary.
            //
            // We have to do this because we're lying to `PostInsertTraversalState` about where the
            // insertion is in order to get it to use the correct key position for shifting and it
            // won't update the cursor if we don't say it's in a `Child`.
            let insertion = match self.insertion {
                Some(insertion) => {
                    self.partial_cursor.prepend_to_path(PathComponent {
                        child_idx: lhs_child_idx + insertion.side as u8,
                    });

                    insertion.handle
                }
                None => {
                    // keys and their left-hand child always have the same index.
                    let key_idx = lhs_child_idx;

                    // SAFETY: `into_slice_handle` requires that `key_idx < parent.leaf().len()`,
                    // which we've already guaranteed by insertion. `clone_slice_ref` requires that
                    // we not use the handle until the original borrow on the tree is gone, which
                    // is mirrored in the docs for `PostInsertTraversalState`.
                    let handle =
                        unsafe { parent.borrow().into_slice_handle(key_idx).clone_slice_ref() };
                    handle.erase_type()
                }
            };

            return Ok(PostInsertTraversalResult::Continue(
                PostInsertTraversalState {
                    inserted_slice: insertion,
                    // We need to use the midpoint key here so that
                    // `PostInsertTraversalState::do_upward_step` updates only the keys after the
                    // lhs-key-rhs grouping. The midpoint key has the same index as the left-hand
                    // child, so the shifting will start immediately after rhs.
                    child_or_key: ChildOrKey::Key((lhs_child_idx, parent.erase_type())),
                    // ^ because of the funky stuff we're doing above, we need to make sure that
                    // the base positions for shifting are correct
                    override_pos: Some(override_pos),
                    old_size,
                    new_size,
                    partial_cursor: self.partial_cursor,
                },
            ));
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
        let (midpoint_key, mut rhs) = unsafe { parent.split(midpoint_idx, store) };
        parent.set_subtree_size(midpoint_key.pos);

        // SAFETY: `rhs.leaf().len() >= 1`
        let rhs_start = unsafe {
            let first_key_pos = rhs.leaf().key_pos(0);
            let first_child_size = rhs.borrow().child(0).leaf().subtree_size();
            first_key_pos.sub_right(first_child_size)
        };

        let mut midpoint_size = rhs_start.sub_left(midpoint_key.pos);

        // The logic can get tricky to follow throughout the rest of this function. To help make
        // things a bit easier, we'll provide illustrations to roughly show what the state of the
        // tree is at each point. These illustrations will pretend that M = 3, so `parent`
        // originally looked something like:
        //         ╔═══╗     ╔═══╗     ╔═══╗     ╔═══╗     ╔═══╗     ╔═══╗
        //    ╔═══╗║ A ║╔═══╗║ B ║╔═══╗║ C ║╔═══╗║ D ║╔═══╗║ E ║╔═══╗║ F ║╔═══╗
        //    ║..a║╚═══╝║a-b║╚═══╝║b-c║╚═══╝║c-d║╚═══╝║d-e║╚═══╝║e-f║╚═══╝║f..║
        //    ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝
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
            if let (Some(shift), true) = (shift_lhs, new_key_idx == M as u8 + 1) {
                // If we have change the side of the left-hand key, AND that key is the current
                // midpoint (because `new_key_idx` is at the start of `rhs`), then the change for
                // `shift_lhs` can be made *just* by changing the size of the midpoint key.
                //
                // For reference, this is what we're currently looking at:
                //                                       ╔═══╗
                //         ╔═══╗     ╔═══╗     ╔═══╗     ║mid║          ╔═══╗     ╔═══╗
                //    ╔═══╗║ A ║╔═══╗║ B ║╔═══╗║ C ║╔═══╗╚═══╝╔═══╗     ║ E ║╔═══╗║ F ║╔═══╗
                //    ║..a║╚═══╝║a-b║╚═══╝║b-c║╚═══╝║c-d║     ║lhs║     ╚═══╝║e-f║╚═══╝║f..║
                //    ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝          ╚═══╝     ╚═══╝
                //                                            ^^^^^ self.lhs
                midpoint_size = shift.new_size;

                // remove `shift_lhs` to mark it as being resolved
                shift_lhs = None;
            }

            let (side, insert_into) = if new_key_idx < M as u8 {
                // SAFETY: `as_mut` requires unique access; we're reborrowing and already have it.
                (Side::Left, unsafe { parent.as_mut() })
            } else {
                // Adjust the position of `slice` to be relative to the right-hand node
                new_key_idx -= M as u8 + 1;

                // SAFETY: `as_mut` requires unique access; `rhs` was just created.
                (Side::Right, unsafe { rhs.as_mut() })
            };

            let (key_pos, shift_pos, old_size, new_size) = match shift_lhs {
                None => {
                    let p = insert_into
                        .leaf()
                        .try_key_pos(new_key_idx)
                        .unwrap_or_else(|| insert_into.leaf().subtree_size());
                    (p, p, I::ZERO, added_size)
                }
                Some(lhs) => {
                    let key_pos = lhs.pos.add_right(lhs.new_size);
                    (key_pos, lhs.pos, lhs.old_size, lhs.new_size)
                }
            };

            // SAFETY: the calls to `insert_key_and_child` and `shift_keys_increase` together
            // require that `new_key_idx <= insert_into.leaf().len()`, which is guaranteed by the
            // values we chose for `midpoint_idx` and our comparison with `M` above.
            unsafe {
                let _ = insert_into.insert_key_and_child(store, new_key_idx, self.key, self.rhs);
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
            let insertion = match self.insertion {
                Some(insertion) => {
                    self.partial_cursor.prepend_to_path(PathComponent {
                        child_idx: new_key_idx + insertion.side as u8,
                    });

                    BubbledInsertion {
                        side,
                        handle: insertion.handle,
                    }
                }

                // the insertion was in `self.key`. There isn't much we need to do because the
                // cursor can't be written to yet.
                None => {
                    // SAFETY: `into_slice_handle` requires that
                    // `new_key_idx < insert_into.leaf().len()`, which is guaranteed by
                    // `insert_key_and_child`. `clone_slice_ref` requires that we not use it until
                    // the other borrows on the tree are gone, which the safety docs for
                    // `BubbledInsertState` guarantees.
                    let inserted_handle = unsafe {
                        insert_into
                            .borrow()
                            .into_slice_handle(new_key_idx)
                            .clone_slice_ref()
                    };

                    // Don't have to update `self.partial_cursor` because the value got inserted as
                    // a key, not a child.

                    BubbledInsertion {
                        side,
                        handle: inserted_handle.erase_type(),
                    }
                }
            };

            Err(BubbledInsertState {
                lhs: parent.erase_type(),
                key: midpoint_key,
                key_size: midpoint_size,
                rhs: rhs.erase_type(),
                old_size: old_parent_size,
                insertion: Some(insertion),
                partial_cursor: self.partial_cursor,
            })
        } else {
            // ^ new_key_idx == M. We'll use `self.key` as the new midpoint. Our current state
            // looks something like:
            //                             ∨∨∨∨∨ midpoint
            //                             ╔═══╗
            //         ╔═══╗     ╔═══╗     ║ C ║     ╔═══╗     ╔═══╗     ╔═══╗
            //    ╔═══╗║ A ║╔═══╗║ B ║╔═══╗╚═══╝╔═══╗║ D ║╔═══╗║ E ║╔═══╗║ F ║╔═══╗
            //    ║..a║╚═══╝║a-b║╚═══╝║b-c║     ║c-d║╚═══╝║d-e║╚═══╝║e-f║╚═══╝║f..║
            //    ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝
            //                                  ^^^^^ self.lhs
            //
            // The eventual outcome will be that the current midpoint and leftmost child of `rhs`
            // are appended to `parent`, with `self.rhs` replacing `rhs`'s leftmost child, and
            // `self.key` replacing the midpoint, so:
            //                                       ╔═══╗
            //         ╔═══╗     ╔═══╗     ╔═══╗     ║key║     ╔═══╗     ╔═══╗     ╔═══╗
            //    ╔═══╗║ A ║╔═══╗║ B ║╔═══╗║ C ║╔═══╗╚═══╝╔═══╗║ D ║╔═══╗║ E ║╔═══╗║ F ║╔═══╗
            //    ║..a║╚═══╝║a-b║╚═══╝║b-c║╚═══╝║c-d║     ║rhs║╚═══╝║d-e║╚═══╝║e-f║╚═══╝║f..║
            //    ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝
            //                                  ^^^^^ self.lhs
            // From this, it's also easy enough to see that any correction to the left-hand key
            // will happen to `C` -- the current midpoint and eventual last key in the left-hand
            // node. So we can fully account for `shift_lhs` by correcting the size of the midpoint
            // key.
            if let Some(lhs) = shift_lhs {
                midpoint_size = lhs.new_size;
            }

            // Replace the leftmost child of `rhs`.
            let new_first_child_size = self.rhs.leaf().subtree_size();
            // SAFETY: `replace_first_child` requires that `self.rhs` is at the correct height to
            // be a child of `parent`. This is guaranteed by `self.rhs` being at the same height as
            // `self.lhs`, because `parent` *is* the parent of `self.lhs`.
            let old_first_child = unsafe { parent.replace_first_child(self.rhs) };
            let old_first_child_size = old_first_child.leaf().subtree_size();

            // Add the key and child to the left-hand node (`parent`)
            let lhs_size = parent.leaf().subtree_size();
            let new_lhs_size = lhs_size
                .add_right(midpoint_size)
                .add_right(old_first_child_size);

            // SAFETY: `push_key_and_child` requires that `parent.leaf().len()` is not equal to the
            // capacity, which we know is true because `parent.leaf().len() == M - 1` is less than
            // the capacity of `2 * M`.
            unsafe {
                parent.push_key_and_child(store, midpoint_key, old_first_child, new_lhs_size);
            }

            // We haven't yet updated the positions in `rhs`, so they're still relative the base of
            // `parent`. After swapping out the leftmost child, we're now good to reclaculate
            // `rhs_start` and shift everything, as we did above.

            // SAFETY: `rhs.leaf().len()` is still >= 1
            let first_rhs_key_pos = unsafe { rhs.leaf().key_pos(0) };

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

            // If `self.insertion` was `None` (i.e. the insertion was in `self.key`), then it's
            // still there. We only need to do anything if `self.insertion` is not `None`
            if let Some(insertion) = self.insertion.as_ref() {
                // In this case, the `side` associated with the insertion remained the same;
                // `self.lhs` ended up at the end of its node, and `self.rhs` ended up at the
                // beginning of the next.
                //
                // All we need to do here is record the child index for the cursor.

                let child_idx = match insertion.side {
                    Side::Left => 0,
                    // There are `M` keys in `parent`, so the index of the last child is also `M`
                    Side::Right => M as u8,
                };

                self.partial_cursor
                    .prepend_to_path(PathComponent { child_idx });
            }

            Err(BubbledInsertState {
                lhs: parent.erase_type(),
                key: self.key,
                key_size: self.key_size,
                rhs: rhs.erase_type(),
                old_size: old_parent_size,
                insertion: self.insertion,
                partial_cursor: self.partial_cursor,
            })
        }
    }
}

impl<'t, C, I, S, P, const M: usize> PostInsertTraversalState<'t, C, I, S, P, M>
where
    C: Cursor,
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    fn do_upward_step(mut self) -> PostInsertTraversalResult<'t, C, I, S, P, M> {
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
            shift_keys_increase(
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
        // should return `Err` with the cursor and reference to the insertion.
        match node.into_parent().ok() {
            None => PostInsertTraversalResult::Root(self.partial_cursor, self.inserted_slice),
            Some((parent_handle, c_idx)) => {
                PostInsertTraversalResult::Continue(PostInsertTraversalState {
                    inserted_slice: self.inserted_slice,
                    child_or_key: ChildOrKey::Child((c_idx, parent_handle)),
                    override_pos: None,
                    old_size,
                    new_size,
                    partial_cursor: self.partial_cursor,
                })
            }
        }
    }
}

/// (*Internal*) Helper struct for [`shift_keys`], so that the parameters are named and we can
/// provide a bit more documentation
struct ShiftKeys<I> {
    /// First key index to update. May be equal to node length
    from: u8,
    /// The position of the earlier change in the node
    ///
    /// We're being intentionally vague here because this can correspond to multiple different
    /// things. [`PostInsertTraversalState`], for example, uses this as the index of an updated
    /// child or key. But for cases where we've inserted multiple things (e.g., where
    /// [`insert_rhs`] is provided a second value), this might only correspond to the index of the
    /// first item, with `old_size` and `new_size` referring to the total size of the pieces.
    ///
    /// [`insert_rhs`]: RleTree::insert_rhs
    pos: I,
    /// The old size of the object at `pos`
    old_size: I,
    /// The new size of the object at `pos`
    ///
    /// If [`shift_keys`] is given `IS_INCREASE = true`, then `new_size` will be expected to be
    /// greater than `old_size`. Otherwise, `new_size` *should* be less than `old_size`. We can't
    /// *guarantee* that this will be true, because of the limitations exposed by user-defined
    /// implementations of `Ord` for `I`, but in general these conditions should hold.
    new_size: I,
}

/// (*Internal*) Updates the positions of all keys in a node, calling either
/// [`shift_keys_increase`] or [`shift_keys_decrease`] depending on whether the size has increased
/// or decreased
///
/// ## Safety
///
/// The value of `opts.from` must be less than or equal to `node.leaf().len()`.
unsafe fn shift_keys_auto<'t, Ty, I, S, P, const M: usize>(
    node: &mut NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
    opts: ShiftKeys<I>,
) where
    Ty: ty::TypeHint,
    I: Index,
    P: RleTreeConfig<I, S>,
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
unsafe fn shift_keys_increase<'t, Ty, I, S, P, const M: usize>(
    node: &mut NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
    opts: ShiftKeys<I>,
) where
    Ty: ty::TypeHint,
    I: Index,
    P: RleTreeConfig<I, S>,
{
    // SAFETY: guaranteed by caller
    unsafe { shift_keys::<Ty, I, S, P, M, true>(node, opts) }
}

/// (*Internal*) Updates the positions of all keys in a node, expecting them to decrease
///
/// ## Safety
///
/// The value of `opts.from` must be less than or equal to `node.leaf().len()`.
unsafe fn shift_keys_decrease<'t, Ty, I, S, P, const M: usize>(
    node: &mut NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
    opts: ShiftKeys<I>,
) where
    Ty: ty::TypeHint,
    I: Index,
    P: RleTreeConfig<I, S>,
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
unsafe fn shift_keys<'t, Ty, I, S, P, const M: usize, const IS_INCREASE: bool>(
    node: &mut NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
    opts: ShiftKeys<I>,
) where
    Ty: ty::TypeHint,
    I: Index,
    P: RleTreeConfig<I, S>,
{
    // SAFETY: guaranteed by caller
    unsafe { weak_assert!(opts.from <= node.leaf().len()) };

    let ShiftKeys {
        from,
        pos,
        old_size,
        new_size,
    } = opts;

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

#[cfg(test)]
macro_rules! valid_assert {
    ($path:ident: $cond:expr) => {
        if !$cond {
            panic!(
                concat!("assertion failed: `", stringify!($cond), "` for path {:?}",),
                $path
            );
        }
    };
}

#[cfg(test)]
impl<I, S, P, const M: usize> RleTree<I, S, P, M>
where
    I: Index,
    P: RleTreeConfig<I, S>,
{
    /// Validates the tree, panicking if the indexes don't add up.
    ///
    /// This method basically exists for tests so that we can quickly narrow down exactly when a
    /// failure is introduced in a particular test case.
    fn validate(&self) {
        let root = match self.root.as_ref() {
            Some(r) => r,
            None => return,
        };

        Self::validate_node(root.handle.borrow(), &mut Vec::new())
    }

    /// Called by `validate` to check a node
    fn validate_node(
        node: NodeHandle<ty::Unknown, borrow::Immut, I, S, P, M>,
        path: &mut Vec<ChildOrKey<u8, u8>>,
    ) {
        match node.into_typed() {
            Type::Leaf(node) => Self::validate_leaf(node, path),
            Type::Internal(node) => Self::validate_internal(node, &mut Vec::new()),
        }
    }

    /// Called by `validate_node` to check a leaf node
    fn validate_leaf(
        node: NodeHandle<ty::Leaf, borrow::Immut, I, S, P, M>,
        path: &mut Vec<ChildOrKey<u8, u8>>,
    ) {
        valid_assert!(path: path.is_empty() != node.leaf().parent().is_some());
        valid_assert!(path: node.leaf().len() >= 1);
        valid_assert!(path: path.is_empty() || node.leaf().len() >= M as u8);

        let poss = node.leaf().keys_pos_slice();
        valid_assert!(path: poss[0] == I::ZERO);

        for i in 1..node.leaf().len() {
            path.push(ChildOrKey::Key(i));
            valid_assert!(path: poss[i as usize] > poss[(i - 1) as usize]);
            path.pop();
        }

        valid_assert!(path: poss[poss.len() - 1] < node.leaf().subtree_size());
    }

    /// Called by `validate_internal` to check an internal node
    fn validate_internal(
        node: NodeHandle<ty::Internal, borrow::Immut, I, S, P, M>,
        path: &mut Vec<ChildOrKey<u8, u8>>,
    ) {
        valid_assert!(path: path.is_empty() != node.leaf().parent().is_some());
        valid_assert!(path: node.leaf().len() >= 1);
        valid_assert!(path: path.is_empty() || node.leaf().len() >= M as u8);

        let poss = node.leaf().keys_pos_slice();

        path.push(ChildOrKey::Child(0));
        // SAFETY: internal nodes are guaranteed to have at least one child
        Self::validate_node(unsafe { node.borrow().into_child(0) }, path);
        path.pop();

        for i in 0..node.leaf().len() {
            path.push(ChildOrKey::Key(i));
            // SAFETY: `i` is a valid key index, so it's also a valid child index.
            let previous_child_size = unsafe { node.child_size(i) };
            if i == 0 {
                valid_assert!(path: poss[i as usize] == previous_child_size);
            } else {
                valid_assert!(path: previous_child_size < poss[i as usize]);
                let previous_child_start = poss[i as usize].sub_right(previous_child_size);
                valid_assert!(path: poss[i as usize - 1] < previous_child_start);
            }
            path.pop();

            let ci = i + 1;
            path.push(ChildOrKey::Child(ci));
            // SAFETY: `i` is a valid key index and there's always one more child than key
            Self::validate_node(unsafe { node.borrow().into_child(ci) }, path);
            path.pop();
        }

        // SAFETY: `leaf.len` is the last valid child index.
        let last_child_size = unsafe { node.child_size(node.leaf().len()) };
        valid_assert!(path: last_child_size < node.leaf().subtree_size());
        let last_child_start = node.leaf().subtree_size().sub_right(last_child_size);
        valid_assert!(path: poss[poss.len() - 1] < last_child_start);
    }
}
