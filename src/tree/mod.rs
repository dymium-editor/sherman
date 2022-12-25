//! Wrapper module containing the big-boy tree itself

use crate::cursor::{Cursor, CursorBuilder, NoCursor, PathComponent};
use crate::param::{
    self, AllowSliceRefs, BorrowState, RleTreeConfig, RleTreeIsSend, SliceRefStore as _,
    SupportsInsert,
};
use crate::public_traits::{Index, Slice};
use crate::range::RangeBounds;
use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::panic::{RefUnwindSafe, UnwindSafe};

#[cfg(test)]
use crate::MaybeDebug;
#[cfg(test)]
use std::fmt::{self, Formatter};
#[cfg(any(test, feature = "fuzz"))]
use std::ptr::NonNull;

pub(crate) mod cow;
mod drain;
mod entry;
mod fix;
mod insert;
mod iter;
mod node;
pub(crate) mod slice_ref;
#[cfg(test)]
mod tests;

pub use drain::Drain;
pub use entry::SliceEntry;
pub use iter::Iter;
pub use slice_ref::{Borrow, SliceRef};

use node::{borrow, ty, ChildOrKey, NodeHandle, Type};
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
/// always `usize`) and `S` implements [`Slice`], which provides utilities for joining and splitting
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
    P: RleTreeConfig<I, S, M>,
{
    root: Option<Root<I, S, P, M>>,
}

////////////////////////
// Marker trait impls //
////////////////////////
//
// We want to implement these explicitly (or not) for `RleTree` so that error messages refer
// directly to the parameterizations of `RleTree` instead of e.g., if we had implemented these only
// once, for `{Node,Slice}Handle`
//
// To briefly explain the reasoning here:
//
// In general, the factors that would prevent us implementing any of the marker traits below would
// come from various kinds of interior mutability. For `NoFeatures`, we don't have any, so the
// tl;dr is that we're good to implement all of `[Ref]UnwindSafe` and `Send`/`Sync`. For
// `AllowCow`, the interior mutability that we *do* have is solely around the `Arc`s or
// `AtomicUsize`s for tracking reference counts, which are both `Send + Sync`, and should be fine
// from `[Ref]UnwindSafe`. However, because `AllowCow` *also* allows the same value to be viewed
// from multiple threads, implementing `Send` requires `Send + Sync` on both `I` and `S`. For more,
// refer to the docs on `RleTreeIsSend`.
//
// The tricky one is `AllowSliceRefs`. With `SliceRef`s, we cannot be `Send` or `Sync`, because the
// shared ownership is managed by `Cell` and `RefCell`. They *can* be `UnwindSafe` or
// `RefUnwindSafe` because the interior mutability won't cause panics to mess up *our* invariants,
// and there isn't interior mutability exposed to user-defined code.
impl<I: UnwindSafe, S: UnwindSafe, P, const M: usize> UnwindSafe for RleTree<I, S, P, M> where
    P: RleTreeConfig<I, S, M>
{
}

impl<I: RefUnwindSafe, S: RefUnwindSafe, P, const M: usize> RefUnwindSafe for RleTree<I, S, P, M> where
    P: RleTreeConfig<I, S, M>
{
}

unsafe impl<I, S, P, const M: usize> Send for RleTree<I, S, P, M> where
    P: RleTreeConfig<I, S, M> + RleTreeIsSend<I, S>
{
}

unsafe impl<I: Sync, S: Sync, P: Sync, const M: usize> Sync for RleTree<I, S, P, M> where
    P: RleTreeConfig<I, S, M>
{
}

// Separate struct to handle the data associated with the root node - but only when it actually
// exists.
struct Root<I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S, M>,
{
    handle: ManuallyDrop<NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>>,
    len: usize,
    refs_store: <P as RleTreeConfig<I, S, M>>::SliceRefStore,
}

#[cfg(not(feature = "nightly"))]
impl<I, S, P, const M: usize> Drop for RleTree<I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn drop(&mut self) {
        destruct_root(self.root.take())
    }
}

#[cfg(feature = "nightly")]
unsafe impl<#[may_dangle] I, #[may_dangle] S, P, const M: usize> Drop for RleTree<I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn drop(&mut self) {
        destruct_root(self.root.take())
    }
}

#[cfg(test)]
impl<I: Index, S, P: RleTreeConfig<I, S, M>, const M: usize> Debug for Root<I, S, P, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        struct Nodes<'t, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
            root: NodeHandle<ty::Unknown, borrow::Immut<'t>, I, S, P, M>,
            indent: &'static str,
        }

        impl<'t, I: Index, S, P: RleTreeConfig<I, S, M>, const M: usize> Debug for Nodes<'t, I, S, P, M> {
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

        fn write_nodes<I: Index, S, P: RleTreeConfig<I, S, M>, const M: usize>(
            node: NodeHandle<ty::Unknown, borrow::Immut, I, S, P, M>,
            path: &mut Vec<u8>,
            indent: &'static str,
            elem_pad: usize,
            total_pad: usize,
            f: &mut Formatter,
        ) -> fmt::Result {
            let path_fmt = format!("{:<elem_pad$?}", SliceContents(path));
            f.write_fmt(format_args!("\n{indent}[{path_fmt:<total_pad$}]: {node:?}"))?;

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
        let nodes = Nodes { root: self.handle.borrow(), indent };
        let mut s = f.debug_struct("Root");
        if P::SLICE_REFS {
            s.field("refs_store", &self.refs_store.fallible_debug());
        }
        s.field("nodes", &nodes).finish()
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
    P: RleTreeConfig<I, S, M>,
{
    let mut r = match root {
        None => return,
        Some(r) => match r.refs_store.try_acquire_drop() {
            ShouldDrop::No => return,
            ShouldDrop::Yes => r,
        },
    };

    // SAFETY: We're given the `Root` by value; it's plain to see that `r.ptr` is not accessed
    // again in this function, before it goes out of scope. That's all that's required of
    // `ManuallyDrop::take`.
    let p = unsafe { ManuallyDrop::take(&mut r.handle) };
    if let Some(handle) = p.try_drop() {
        handle.do_drop();
    }
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
    P: RleTreeConfig<I, S, M>,
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

        let handle = NodeHandle::new_root(slice, size).erase_type();
        // SAFETY: we're required to only use this for the `SliceRefStore`, which *is* what we're
        // doing here.
        let store_handle = unsafe { handle.clone_root_for_refs_store() };

        RleTree {
            root: Some(Root {
                handle: ManuallyDrop::new(handle),
                len: 1,
                refs_store: P::SliceRefStore::new(store_handle),
            }),
        }
    }

    /// Returns the total size of the tree -- i.e. the sum of the sizes of all the slices
    pub fn size(&self) -> I {
        match self.root.as_ref() {
            Some(root) => root.handle.leaf().subtree_size(),
            None => I::ZERO,
        }
    }

    /// Returns the number of slices contained in the tree -- equivalent to the number of elements
    /// [`.iter(..)`](Self::iter) will produce
    ///
    /// This method is O(1).
    pub fn number_of_slices(&self) -> usize {
        match self.root.as_ref() {
            Some(root) => root.len,
            None => 0,
        }
    }

    /// Returns an object with information about the slice containing the index
    ///
    /// Through the returned [`SliceEntry`], both the slice `S` and the range of values covered
    /// `Range<I>` can be retrieved.
    ///
    /// **See also:** [`get_with_cursor`](Self::get_with_cursor)
    ///
    /// ## Panics
    ///
    /// This method will panic if `idx` is out of bounds -- i.e., if it is less than zero or
    /// greater than `self.size()`.
    pub fn get(&self, idx: I) -> SliceEntry<I, S, P, M> {
        self.get_with_cursor(NoCursor, idx)
    }

    /// Returns an object with information about the slice containing the index, using a provided
    /// [`Cursor`] as a path hint
    ///
    /// For more information, refer to [`get`](Self::get).
    pub fn get_with_cursor<C: Cursor>(&self, cursor: C, idx: I) -> SliceEntry<I, S, P, M> {
        if idx < I::ZERO {
            panic!("index {idx:?} out of bounds, less than zero");
        } else if idx >= self.size() {
            panic!("index {idx:?} out of bounds for size {:?}", self.size());
        }

        let root = self.root.as_ref().expect("`self.root` should be `Some` if `0 <= idx < size`");

        let mut node = root.handle.borrow();
        let mut cursor_iter = Some(cursor.into_path());
        let mut target = idx;

        loop {
            let hint = cursor_iter.as_mut().and_then(|it| it.next());
            let search_result = search_step(node, hint, target);

            let hint_was_good = matches!(
                (search_result, hint),
                (ChildOrKey::Child((c_idx, _)), Some(h)) if c_idx == h.child_idx
            );
            if !hint_was_good {
                cursor_iter = None;
            }

            match search_result {
                ChildOrKey::Key((k_idx, k_start)) => {
                    let slice = unsafe { node.into_slice_handle(k_idx) };
                    let diff_from_base_to_target = target.sub_left(k_start);
                    let range_start = idx.sub_right(diff_from_base_to_target);
                    let range_end = range_start.add_right(slice.slice_size());

                    return SliceEntry {
                        range_start,
                        range_end,
                        slice,
                        store: &root.refs_store,
                    };
                }
                ChildOrKey::Child((c_idx, c_start)) => {
                    target = target.sub_left(c_start);
                    node = match node.into_typed() {
                        Type::Leaf(_) => unreachable!(),
                        // SAFETY: `search_step` guarantees that - if a `Child` is returned - the
                        // index associated with it will be a valid child index.
                        Type::Internal(node) => unsafe { node.into_child(c_idx) },
                    };
                }
            }
        }
    }

    /// Returns a [`Cursor`] to the value at the provided index
    ///
    /// This method is functionally the same as [`SliceEntry::cursor`], except it is also available
    /// on COW-enabled trees. If you already have a [`SliceEntry`] pointing to a value, its
    /// `cursor` method should be preferred over this one.
    ///
    /// ## Panics
    ///
    /// This method will panic if `idx` is out of bounds -- i.e., if it is less than zero or
    /// greater than `self.size()`
    pub fn cursor_to<C: Cursor>(&self, idx: I) -> C {
        if idx < I::ZERO {
            panic!("index {idx:?} out of bounds, less than zero");
        } else if idx >= self.size() {
            panic!("index {idx:?} out of bounds for size {:?}", self.size());
        }

        let root = self.root.as_ref().expect("`self.root` should be `Some` if `0 <= idx < size`");

        // Short-circuit for no-op cursors
        if C::IS_NOP {
            return C::new_empty();
        }

        // recursive helper function, required in order to support COW-enabled trees. For all
        // others, this *should* compile down to a simple loop
        fn recurse<C: Cursor, I: Index, S, P: RleTreeConfig<I, S, M>, const M: usize>(
            mut node: NodeHandle<ty::Unknown, borrow::Immut, I, S, P, M>,
            mut target: I,
        ) -> C {
            let top_height = node.height();

            // Traverse down the tree:
            let mut cursor = loop {
                match node.into_typed() {
                    Type::Leaf(_) => break C::new_empty(),
                    Type::Internal(n) => match search_step(n.erase_type(), None, target) {
                        // If it's a key, return an empty cursor to this node
                        ChildOrKey::Key(_) => break C::new_empty(),
                        ChildOrKey::Child((c_idx, c_pos)) => {
                            target = target.sub_left(c_pos);

                            // SAFETY: search_step guarantees that c_idx is valid
                            let child = unsafe { n.into_child(c_idx) };
                            // note: adding P::COW means the compiler knows it will always be false
                            let mismatched_parent = P::COW
                                && matches!(child.leaf().parent(), Some(p) if p.ptr != n.ptr());
                            if mismatched_parent {
                                let mut c: C = recurse(node, target);
                                c.prepend_to_path(PathComponent { child_idx: c_idx });
                                break c;
                            }

                            node = child;
                            continue;
                        }
                    },
                }
            };

            // Traverse back up until we get to `top_height` (or the top of the tree, if !COW)
            loop {
                if P::COW && node.height() == top_height {
                    break;
                }

                match node.into_parent() {
                    Err(_) => break,
                    Ok((parent, child_idx)) => {
                        cursor.prepend_to_path(PathComponent { child_idx });
                        node = parent.erase_type();
                    }
                }
            }

            cursor
        }

        recurse(root.handle.borrow(), idx)
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
    /// use sherman::{Constant, RleTree};
    ///
    /// let tree: RleTree<usize, Constant<char>> =
    ///     RleTree::new(Constant('a'), 5);
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

        let root = self.root.as_ref().map(|r| (r.handle.borrow(), &r.refs_store));

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
        let root = self.root.as_ref().map(|r| (r.handle.borrow(), &r.refs_store));

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
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
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
        let mut result_cursor = C::new_empty();
        let cursor_builder = CursorBuilder::new(&mut result_cursor);
        let mut path_iter = &mut cursor.into_path();
        let _returned_ptr = self.insert_internal(cursor_builder, &mut path_iter, idx, slice, size);
        result_cursor
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
        let mut result_cursor = C::new_empty();
        let cursor_builder = CursorBuilder::new(&mut result_cursor);
        let mut path_iter = cursor.into_path();
        let inserted_slice = self.insert_internal(cursor_builder, &mut path_iter, idx, slice, size);
        let root = match self.root.as_ref() {
            Some(r) => r,
            // SAFETY: insertion always results in at least a root node remaining in the tree
            None => unsafe { weak_unreachable!() },
        };

        let insertion = match inserted_slice {
            Some(h) => h,
            None => unreachable!(),
        };

        let slice_ref = root.refs_store.make_ref(insertion);
        (result_cursor, slice_ref)
    }
}

/// `Clone` is implemented for trees with no features and those with COW enabled, with COW-enabled
/// trees performing a cheap, shallow clone
impl<I, S, const M: usize> Clone for RleTree<I, S, param::NoFeatures, M>
where
    I: Copy,
    S: Clone,
{
    fn clone(&self) -> Self {
        match self.root.as_ref() {
            None => RleTree { root: None },
            Some(root) => RleTree {
                root: Some(Root {
                    handle: ManuallyDrop::new(root.handle.as_immut().deep_clone().erase_unique()),
                    len: root.len,
                    refs_store: Default::default(),
                }),
            },
        }
    }
}

/// `Clone` is implemented for trees with no features and those with COW enabled, with COW-enabled
/// trees performing a cheap, shallow clone
impl<I, S: Clone, const M: usize> Clone for RleTree<I, S, param::AllowCow, M> {
    fn clone(&self) -> Self {
        match self.root.as_ref() {
            None => RleTree { root: None },
            Some(root) => {
                // SAFETY: `increase_strong_count_and_clone` requires that `P = AllowCow`, which is
                // guaranteed by the impl block.
                let handle = unsafe { root.handle.increase_strong_count_and_clone() };

                RleTree {
                    root: Some(Root {
                        handle: ManuallyDrop::new(handle),
                        len: root.len,
                        // refs_store will be empty because this is a COW-enabled tree
                        refs_store: Default::default(),
                    }),
                }
            }
        }
    }
}

/// (*Internal*) Helper type for various tree operations
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
enum Side {
    Left = 0,
    Right = 1,
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
fn search_step<'t, I: Index, S, P: RleTreeConfig<I, S, M>, const M: usize>(
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

macro_rules! define_node_box {
    (
        $(#[$attrs:meta])*
        $vis:vis struct $name:ident<$borrow:path, ...> { ... }

        $(impl {
            $($methods:item)*
        })?

        impl Drop {
            fn drop(&mut $drop_this:ident) {
                $($drop_body:tt)*
            }
        }
    ) => {
        $(#[$attrs])*
        $vis struct $name<Ty: node::ty::TypeHint, I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
            node: ManuallyDrop<NodeHandle<Ty, $borrow, I, S, P, M>>,
        }

        #[cfg(test)]
        impl<Ty, I, S, P, const M: usize> Debug for $name<Ty, I, S, P, M>
        where
            Ty: node::ty::TypeHint,
            P: RleTreeConfig<I, S, M>,
        {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                Debug::fmt(&*self.node, f)
                //         ^^ deref through the ManuallyDrop
            }
        }

        #[allow(dead_code)]
        impl<Ty, I, S, P, const M: usize> $name<Ty, I, S, P, M>
        where
            Ty: node::ty::TypeHint,
            P: RleTreeConfig<I, S, M>,
        {
            #[doc = concat!("Creates a new `", stringify!($name), "` to store the handle")]
            fn new(node: NodeHandle<Ty, $borrow, I, S, P, M>) -> Self {
                Self {
                    node: ManuallyDrop::new(node)
                }
            }

            #[doc = concat!(
                "Extracts the `Nodehandle` out of the `",
                stringify!($name),
                "`, without running its destructor"
            )]
            fn take(self) -> NodeHandle<Ty, $borrow, I, S, P, M> {
                let mut this = ManuallyDrop::new(self);
                // SAFETY: `take` requires that the `ManuallyDrop` is never used again, which is
                // guaranteed by putting `self` into a new `ManuallyDrop` so that its destructor is
                // not run
                unsafe { ManuallyDrop::take(&mut this.node) }
            }

            #[doc = concat!("Returns a type-erased version of the `", stringify!($name), "`")]
            fn erase_type(self) -> $name<ty::Unknown, I, S, P, M> {
                $name::new(self.take().erase_type())
            }

            /// Returns the result of calling [`as_immut`] on the inner `NodeHandle`
            ///
            /// [`as_immut`]: NodeHandle::as_immut
            fn as_ref(&self) -> &NodeHandle<Ty, borrow::Immut, I, S, P, M> {
                self.node.as_immut()
            }

            $($($methods)*)?
        }

        impl<Ty, I, S, P, const M: usize> Drop for $name<Ty, I, S, P, M>
        where
            Ty: node::ty::TypeHint,
            P: RleTreeConfig<I, S, M>,
        {
            fn drop(&mut self) {
                #[allow(unused_mut)]
                let mut $drop_this = self;
                $($drop_body)*
            }
        }
    };
}

define_node_box! {
    struct NodeBox<borrow::UniqueOwned, ...> { ... }

    impl {
        fn as_mut(&mut self) -> &mut NodeHandle<Ty, borrow::Mut, I, S, P, M> {
            self.node.as_mut()
        }
    }

    impl Drop {
        fn drop(&mut this) {
            // SAFETY: `take` requires that the `ManuallyDrop` is never used again, which is
            // guaranteed because this is the destructor.
            let node = unsafe { ManuallyDrop::take(&mut this.node) };
            // Note: we do not need to handle updating slice references because the tree will stay
            // marked as mutably borrowed until it's dropped, preventing access through the stored
            // `SliceHandle`s
            node.erase_type().into_drop().do_drop();
        }
    }
}

define_node_box! {
    struct SharedNodeBox<borrow::Owned, ...> { ... }

    impl Drop {
        fn drop(&mut this) {
            // SAFETY: `take` requires that the `ManuallyDrop` is never used again, which is
            // guaranteed because this is the destructor.
            let node = unsafe { ManuallyDrop::take(&mut this.node) };

            if let Some(handle) = node.erase_type().try_drop() {
                handle.do_drop();
            }
        }
    }
}

#[cfg(any(test, feature = "fuzz"))]
macro_rules! valid_assert {
    ($path:ident: $cond:expr) => {
        if !$cond {
            panic!(concat!("assertion failed: `", stringify!($cond), "` for path {:?}"), $path);
        }
    };
}

#[cfg(any(test, feature = "fuzz"))]
macro_rules! valid_assert_eq {
    ($path:ident: $lhs:expr, $rhs:expr) => {
        let left = $lhs;
        let right = $rhs;
        if left != right {
            panic!(
                concat!(
                    "assertion failed: `",
                    stringify!($lhs == $rhs),
                    "` for path {:?}:\n",
                    " left: {:?}\n",
                    "right: {:?}",
                ),
                $path, left, right,
            );
        }
    };
}

#[cfg(any(test, feature = "fuzz"))]
impl<I, S, P, const M: usize> RleTree<I, S, P, M>
where
    I: Index,
    P: RleTreeConfig<I, S, M>,
{
    /// (*Test-only*) Validates the tree, panicking if the indexes don't add up.
    ///
    /// This method basically exists for tests so that we can quickly narrow down exactly when a
    /// failure is introduced in a particular test case.
    pub fn validate(&self) {
        let root = match self.root.as_ref() {
            Some(r) => r,
            None => return,
        };

        let total_count = Self::validate_node(root.handle.borrow(), &mut Vec::new(), None);
        assert_eq!(total_count, root.len);
    }

    /// Called by `validate` to check a node. Returns the total number of slices contained in the
    /// subtree rooted at the node
    fn validate_node(
        node: NodeHandle<ty::Unknown, borrow::Immut, I, S, P, M>,
        path: &mut Vec<ChildOrKey<u8, u8>>,
        parent: Option<(NonNull<()>, u8)>,
    ) -> usize {
        valid_assert!(path: path.is_empty() != node.leaf().parent().is_some());
        valid_assert!(path: node.leaf().len() >= 1);
        valid_assert!(path: path.is_empty() || node.leaf().len() >= M as u8);
        if !P::COW {
            let node_parent = match node.leaf().parent() {
                None => None,
                Some(p) => Some((p.ptr.cast::<()>(), p.idx_in_parent)),
            };
            valid_assert_eq!(path: parent, node_parent);
        }

        match node.into_typed() {
            Type::Leaf(node) => Self::validate_leaf(node, path),
            Type::Internal(node) => Self::validate_internal(node, path),
        }
    }

    /// Called by `validate_node` to check a leaf node. Returns the number of slices
    fn validate_leaf(
        node: NodeHandle<ty::Leaf, borrow::Immut, I, S, P, M>,
        path: &mut Vec<ChildOrKey<u8, u8>>,
    ) -> usize {
        let poss = node.leaf().keys_pos_slice();
        valid_assert_eq!(path: poss[0], I::ZERO);

        for i in 1..node.leaf().len() {
            path.push(ChildOrKey::Key(i));
            valid_assert!(path: poss[i as usize] > poss[(i - 1) as usize]);
            path.pop();
        }

        valid_assert!(path: poss[poss.len() - 1] < node.leaf().subtree_size());
        node.leaf().len() as usize
    }

    /// Called by `validate_internal` to check an internal node. Returns the total number of slices
    /// contained in the subtree rooted at the node
    fn validate_internal(
        node: NodeHandle<ty::Internal, borrow::Immut, I, S, P, M>,
        path: &mut Vec<ChildOrKey<u8, u8>>,
    ) -> usize {
        let poss = node.leaf().keys_pos_slice();

        let this_ptr = node.ptr().cast();

        let mut number_of_slices = 0;

        path.push(ChildOrKey::Child(0));
        // SAFETY: internal nodes are guaranteed to have at least one child
        let first_child = unsafe { node.borrow().into_child(0) };
        number_of_slices += Self::validate_node(first_child, path, Some((this_ptr, 0)));
        path.pop();

        for i in 0..node.leaf().len() {
            number_of_slices += 1; // add each key
            path.push(ChildOrKey::Key(i));
            // SAFETY: `i` is a valid key index, so it's also a valid child index.
            let previous_child_size = unsafe { node.child_size(i) };
            if i == 0 {
                valid_assert_eq!(path: poss[i as usize], previous_child_size);
            } else {
                valid_assert!(path: previous_child_size < poss[i as usize]);
                let previous_child_start = poss[i as usize].sub_right(previous_child_size);
                valid_assert!(path: poss[i as usize - 1] < previous_child_start);
            }
            path.pop();

            let ci = i + 1;
            path.push(ChildOrKey::Child(ci));
            // SAFETY: `i` is a valid key index and there's always one more child than key
            number_of_slices += Self::validate_node(
                unsafe { node.borrow().into_child(ci) },
                path,
                Some((this_ptr, ci)),
            );
            path.pop();
        }

        // SAFETY: `leaf.len` is the last valid child index.
        let last_child_size = unsafe { node.child_size(node.leaf().len()) };
        valid_assert!(path: last_child_size < node.leaf().subtree_size());
        let last_child_start = node.leaf().subtree_size().sub_right(last_child_size);
        valid_assert!(path: poss[poss.len() - 1] < last_child_start);

        number_of_slices
    }
}
