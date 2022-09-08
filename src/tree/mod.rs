//! Wrapper module containing the big-boy tree itself

use crate::param::{
    self, AllowSliceRefs, BorrowState, RleTreeConfig, SliceRefStore as _, StrongCount,
    SupportsInsert,
};
use crate::public_traits::{Index, Slice};
use crate::range::RangeBounds;
use crate::{Cursor, NoCursor, PathComponent};
use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::panic::UnwindSafe;

#[cfg(test)]
use crate::MaybeDebug;
#[cfg(test)]
use std::fmt::{self, Formatter};
#[cfg(any(test, feature = "fuzz"))]
use std::ptr::NonNull;

pub(crate) mod cow;
mod insert;
mod iter;
mod node;
pub(crate) mod slice_ref;
#[cfg(test)]
mod tests;

pub use iter::{Drain, Iter, SliceEntry};
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
    P: RleTreeConfig<I, S, M>,
{
    root: Option<Root<I, S, P, M>>,
}

// FIXME: This should be more precise, specifically around the interactions with `RefUnwindSafe`
// and `AllowSliceRefs` / `AllowCow`
impl<I, S, P, const M: usize> UnwindSafe for RleTree<I, S, P, M>
where
    I: UnwindSafe,
    S: UnwindSafe,
    P: RleTreeConfig<I, S, M>,
{
}

// Separate struct to handle the data associated with the root node - but only when it actually
// exists.
struct Root<I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S, M>,
{
    handle: ManuallyDrop<NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>>,
    refs_store: <P as RleTreeConfig<I, S, M>>::SliceRefStore,
    shared_total_strong_count: <P as RleTreeConfig<I, S, M>>::SharedStrongCount,
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
        let nodes = Nodes {
            root: self.handle.borrow(),
            indent,
        };
        // FIXME: Add SliceRefStore to this if size_of::<P::SliceRefStore>() != 0
        let mut s = f.debug_struct("Root");
        if P::COW {
            s.field(
                "shared_total_strong_count",
                &self.shared_total_strong_count.count(),
            );
        } else if std::mem::size_of::<P::SliceRefStore>() != 0 {
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

    // Only now do we drop the tree-wide strong count - doing it earlier could run into race
    // conditions.
    //
    // This is either an `Arc<()>` or nothing, and `Arc` doesn't support `decrement` because it
    // doesn't make any sense.
    drop(r.shared_total_strong_count);
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
                refs_store: P::SliceRefStore::new(store_handle),
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
        let (cursor, _returned_ptr) = self.insert_internal(cursor, idx, slice, size);
        cursor
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
        let root = match self.root.as_ref() {
            Some(r) => r,
            // SAFETY: insertion always results in at least a root node remaining in the tree
            None => unsafe { weak_unreachable!() },
        };

        let slice_ref = root.refs_store.make_ref(inserted_slice);
        (cursor, slice_ref)
    }
}

/// `Clone` is only implemented directly for trees without additional features
///
/// For cloning a COW-enabled tree, refer to [`shallow_clone`] and [`deep_clone`]. For a tree with
/// slice references, refer to [`clone_with_refs`] and [`clone_without_refs`].
///
/// [`shallow_clone`]: Self::shallow_clone
/// [`deep_clone`]: Self::deep_clone
/// [`clone_with_refs`]: Self::clone_with_refs
/// [`clone_without_refs`]: Self::clone_without_refs
impl<I, S, const M: usize> Clone for RleTree<I, S, param::NoFeatures, M>
where
    I: Clone,
    S: Clone,
{
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<I: Index, S: Clone, const M: usize> RleTree<I, S, param::AllowCow, M> {
    /// Creates a shallow clone of the tree, sharing nodes until they or their children are
    /// modified
    ///
    /// ## Examples
    ///
    /// FIXME
    pub fn shallow_clone(&self) -> Self {
        match self.root.as_ref() {
            None => RleTree { root: None },
            Some(root) => {
                // SAFETY: `increase_strong_count_and_clone` requires that `P = AllowCow`, which is
                // guaranteed by the impl block.
                let handle = unsafe { root.handle.increase_strong_count_and_clone() };

                RleTree {
                    root: Some(Root {
                        handle: ManuallyDrop::new(handle),
                        // refs_store will be empty because this is a COW-enabled tree
                        refs_store: Default::default(),
                        shared_total_strong_count: root.shared_total_strong_count.increment(),
                    }),
                }
            }
        }
    }

    /// Creates a deep clone of the tree, creating fresh copies of every node
    ///
    /// For sets of shallow clone'd `RleTree`s that have significantly diverged, fully separating
    /// them with a deep clone can signal to the implementation that they're no longer related, and
    /// so the reference count on the tree itself is reset to one. Insertion, for example, can be
    /// unnecessarily costly otherwise (creating copies of all nodes down to the insertion, and
    /// then removing the old ones).
    ///
    /// This method essentially exists to allow more optimal behavior, and will not be necessary in
    /// the future (we *will* improve the [`insert`] implementation). However, this method will still
    /// be kept around because there are subtle semantic differences between a shallow and deep
    /// clone that are still sometimes useful.
    ///
    /// ## Examples
    ///
    /// FIXME
    ///
    /// [`insert`]: Self::insert
    pub fn deep_clone(&self) -> Self
    where
        I: Clone,
        S: Clone,
    {
        todo!()
    }
}

impl<I: Clone, S: Clone, const M: usize> RleTree<I, S, param::AllowSliceRefs, M> {
    pub fn clone_with_refs(&self) -> Self {
        todo!()
    }

    pub fn clone_without_refs(&self) -> Self {
        todo!()
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
unsafe fn shift_keys_increase<'t, Ty, I, S, P, const M: usize>(
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
unsafe fn shift_keys_decrease<'t, Ty, I, S, P, const M: usize>(
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
unsafe fn shift_keys<'t, Ty, I, S, P, const M: usize, const IS_INCREASE: bool>(
    node: &mut NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>,
    opts: ShiftKeys<I>,
) where
    Ty: ty::TypeHint,
    I: Index,
    P: RleTreeConfig<I, S, M>,
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

#[cfg(any(test, feature = "fuzz"))]
macro_rules! valid_assert {
    ($path:ident: $cond:expr) => {
        if !$cond {
            panic!(
                concat!("assertion failed: `", stringify!($cond), "` for path {:?}"),
                $path
            );
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

        Self::validate_node(root.handle.borrow(), &mut Vec::new(), None)
    }

    /// Called by `validate` to check a node
    fn validate_node(
        node: NodeHandle<ty::Unknown, borrow::Immut, I, S, P, M>,
        path: &mut Vec<ChildOrKey<u8, u8>>,
        parent: Option<(NonNull<()>, u8)>,
    ) {
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

    /// Called by `validate_node` to check a leaf node
    fn validate_leaf(
        node: NodeHandle<ty::Leaf, borrow::Immut, I, S, P, M>,
        path: &mut Vec<ChildOrKey<u8, u8>>,
    ) {
        let poss = node.leaf().keys_pos_slice();
        valid_assert_eq!(path: poss[0], I::ZERO);

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
        let poss = node.leaf().keys_pos_slice();

        let this_ptr = node.ptr().cast();

        path.push(ChildOrKey::Child(0));
        // SAFETY: internal nodes are guaranteed to have at least one child
        Self::validate_node(
            unsafe { node.borrow().into_child(0) },
            path,
            Some((this_ptr, 0)),
        );
        path.pop();

        for i in 0..node.leaf().len() {
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
            Self::validate_node(
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
    }
}
