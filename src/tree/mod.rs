#![allow(clippy::type_complexity)]

#[cfg(test)]
use std::fmt::{self, Debug};
use std::ops::{ControlFlow, Range};

use crate::param::{self, EnableCow, RleTreeConfig, SupportsUpdate};
use crate::{Index, Slice};

#[macro_use]
pub(crate) mod borrow;

mod builder;
mod drain;
mod entry;
mod fix;
mod iter;
pub(crate) mod node;
pub(crate) mod rc;
mod remove;

pub(crate) mod tests;

pub use builder::Builder;
pub use drain::Drain;
pub use entry::{SliceEntry, StableRef};
use fix::FixMode;
pub use iter::{IntoIter, Iter};
use node::Side;
use rc::Redirect as _;
pub use remove::Removed;

/// Generalized run-length encoded balanced binary search tree
///
/// This data structure represents a continuous range of an [`Index`] type (e.g., `usize`), broken
/// up into smaller ranges of a [`Slice`] value (e.g., a chunk of text).
///
/// It supports `O(log(n))` point lookup, range insertion, and range deletion. On each insertion or
/// deletion, all ranges of values after the modification point are shifted accordingly.
///
/// One way of thinking about this type is that it's like a [rope] that's generic over the value
/// type (so long as it can be split at a point and optionally merged back together) and also
/// generic over the index (so long as we can compute the unsigned offset between any two points).
///
/// [rope]: https://en.wikipedia.org/wiki/Rope_(data_structure)
///
/// The implementation of all operations on this data structure do not use unbounded recursion
/// (i.e. in the places where recursion is used, it is at most a constant number of recursive
/// calls).
///
/// For more details on concepts or motivation, refer to the [crate-level documentation](crate).  
/// For details on the internal implementation, see [ARCHITECTURE.md]
///
/// [ARCHITECTURE.md]: https://github.com/dymium-editor/sherman/blob/main/ARCHITECTURE.md
pub struct RleTree<I, S, P: RleTreeConfig<I, S> = param::NoFeatures> {
    root: Option<Root<I, S, P>>,
}

/// The (owned) root node of an `RleTree`
pub(super) struct Root<I, S, P: RleTreeConfig<I, S>> {
    handle: node::HandleOwned<I, S, P>,
}

#[cfg(not(feature = "nightly"))]
impl<I, S, P: RleTreeConfig<I, S>> Drop for RleTree<I, S, P> {
    fn drop(&mut self) {}
}

#[cfg(feature = "nightly")]
unsafe impl<#[may_dangle] I, #[may_dangle] S, P: RleTreeConfig<I, S>> Drop for RleTree<I, S, P> {
    fn drop(&mut self) {}
}

#[cfg(test)]
impl<I: Debug, S: Debug, P: RleTreeConfig<I, S>> Debug for Root<I, S, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = f.debug_struct("Root");
        s.field("node", &self.handle);
        s.finish()
    }
}

impl<I, S, P> RleTree<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    /// Creates a new, empty `RleTree`.
    ///
    /// # Example usage
    ///
    /// ```
    /// use sherman::{Constant, RleTree};
    ///
    /// let tree: RleTree<usize, Constant<&str>> = RleTree::new_empty();
    /// assert_eq!(tree.size(), 0);
    /// assert_eq!(tree.iter(..).count(), 0);
    /// ```
    pub const fn new_empty() -> Self {
        RleTree { root: None }
    }

    /// Creates an `RleTree` initialized to contain only the initial slice of the given size
    ///
    /// # Panics
    ///
    /// This method will panic if `size` is not greater than zero — i.e., if `size <= I::ZERO`.
    ///
    /// # Example usage
    ///
    /// ```
    /// use sherman::{Constant, RleTree};
    ///
    /// let tree: RleTree<usize, Constant<&str>> =
    ///     RleTree::new(10, Constant("foo"));
    /// assert_eq!(tree.size(), 10);
    /// assert_eq!(tree.iter(..).count(), 1);
    ///
    /// let entry = tree.get(5);
    /// assert_eq!(entry.range(), 0..10);
    /// assert_eq!(entry.slice(), &Constant("foo"));
    /// ```
    pub fn new(size: I, slice: S) -> Self {
        let mut slice = Some(slice);

        if size <= I::ZERO {
            panic!("cannot create new slice with non-positive size {size:?}");
        }

        Self::new_from_opt(size, &mut slice)
    }

    fn new_from_opt(size: I, slice: &mut Option<S>) -> Self {
        RleTree {
            root: Some(Root {
                handle: node::NodeHandle::alloc_new(slice, size).erase(),
            }),
        }
    }

    /// Returns a new [`Builder`] for efficient construction of an [`RleTree`].
    ///
    /// This is an alias for [`Builder::new`]. Refer to the documentation on [`Builder`] for
    /// example usage and more information.
    pub fn builder() -> Builder<I, S, P> {
        Builder::new()
    }

    /// Internal helper method, mostly for tests
    fn root(&self) -> Option<&Root<I, S, P>> {
        self.root.as_ref()
    }

    /// Asserts that the tree is balanced
    #[cfg(any(test, feature = "fuzz"))]
    pub(crate) fn validate_balance(&self) {
        if let Some(root) = self.root.as_ref() {
            fix::validate_balance(root.handle.borrow());
        }
    }

    /// Returns the total size of the tree — i.e., the sum of the sizes of all the slices
    ///
    /// # Algorithmic complexity
    ///
    /// This operation is `O(1)`.
    ///
    /// # Example usage
    ///
    /// ```
    /// use sherman::{Constant, RleTree};
    ///
    /// let mut tree: RleTree<usize, Constant<&str>> =
    ///     RleTree::new(10, Constant("foo"));
    /// assert_eq!(tree.size(), 10);
    ///
    /// tree.insert(3, Constant("baz"), 4);
    /// assert_eq!(tree.size(), 14);
    ///
    /// let removed = tree.remove(2..7);
    /// assert_eq!(tree.size(), 9);
    /// assert_eq!(removed.size(), 5);
    /// ```
    pub fn size(&self) -> I {
        match &self.root {
            Some(r) => r.handle.subtree_size(),
            None => I::ZERO,
        }
    }

    /// Returns an object with information about the slice containing the index
    ///
    /// Through the returned [`SliceEntry`], both the slice `S` and the range of values covered
    /// `Range<I>` can be retrieved.
    ///
    /// # Algorithmic complexity
    ///
    /// This operation is `O(log(r))`, where `r` is the number of ranges of values in the tree.
    ///
    /// # Panics
    ///
    /// This method will panic if `idx` is out of bounds — i.e., if it is less than `I::ZERO` or
    /// greater than `self.size()`.
    ///
    /// # Drop glue
    ///
    /// **Note:** Currently, [`SliceEntry`] can often require explicitly dropping it to release the
    /// borrow on [`RleTree`]. The "nightly" feature adds `#[may_dangle]` to skip that requirement,
    /// but we may also improve this crate's internals in the future to remove that need in the
    /// future.
    ///
    /// # Example usage
    ///
    /// ```
    /// use sherman::{Constant, RleTree};
    ///
    /// let mut tree: RleTree<usize, Constant<&str>> = RleTree::new_empty();
    ///
    /// tree.insert(0, Constant("foo"), 4);
    /// tree.insert(4, Constant("bar"), 4);
    ///
    /// // Values now look like:
    /// //
    /// //   | "foo" | "bar" |
    /// //   0       4       8
    /// //
    ///
    /// // `get()` returns a `SliceEntry` that can be used to give the range
    /// // of the value containing the index, and the value itself:
    /// let fst = tree.get(2);
    /// assert_eq!(fst.range(), 0..4);
    /// assert_eq!(fst.slice(), &Constant("foo"));
    ///
    /// // Anything in the range returns the same slice:
    /// assert_eq!(tree.get(0).slice(), fst.slice());
    /// assert_eq!(tree.get(1).slice(), fst.slice());
    /// assert_eq!(tree.get(2).slice(), fst.slice());
    /// assert_eq!(tree.get(3).slice(), fst.slice());
    ///
    /// // The edge of two values is contained by the right-hand side:
    /// let snd = tree.get(4);
    /// assert_eq!(snd.range(), 4..8);
    /// assert_eq!(snd.slice(), &Constant("bar"));
    ///
    /// drop((fst, snd)); // (required without the "nightly" feature)
    ///
    /// // `get()` at the end of the tree panics:
    /// assert_eq!(tree.size(), 8);
    /// assert!(std::panic::catch_unwind(move || _ = tree.get(8)).is_err());
    /// ```
    pub fn get(&self, idx: I) -> SliceEntry<'_, I, S, P> {
        if idx < I::ZERO {
            panic!("index {idx:?} out of bounds, less than zero");
        } else if idx >= self.size() {
            panic!("index {idx:?} out of bounds for size {:?}", self.size());
        }

        let Some(root) = self.root.as_ref() else {
            crate::panic_internal_error_or_bad_index::<I>(
                "`self.root` should be `Some` if `0 <= idx < size`",
            );
        };

        let (node, range, offset_in_range) =
            search_node(root.handle.borrow(), SearchBound::Included(idx));

        // To reconstruct the absolute positions of the slice, we can subtract offset from idx to
        // get the absolute position of range.start (and therefore range.end as well.
        let abs_start = idx.sub_right(offset_in_range);
        let abs_end = abs_start.add_right(range.end.sub_left(range.start));
        SliceEntry { range: abs_start..abs_end, slice: node }
    }

    /// Returns an iterator yielding all slices that intersect with the range
    ///
    /// The iterator is double-ended and produces [`SliceEntry`]s. Each entry's range *may* have
    /// one or both bounds outside `range`, but *will* contain some overlap with the requested
    /// `range`.
    ///
    /// # Algorithmic complexity
    ///
    /// Creating the iterator is `O(log(r))`, where `r` is the number of ranges of values in the
    /// tree.  
    /// Each step of iteration is `O(log(r))` but amortized `O(1)`.
    ///
    /// With [`EnableCow`], the iterator maintains a stack of up to `O(log(r))` parent pointers as
    /// it traverses the tree (depending on the number of shallow clones created). Otherwise, the
    /// iterator uses `O(1)` memory.
    ///
    /// # Panics
    ///
    /// This method panics if any of the following are true:
    ///
    /// 1. The start of the range is greater than its end (or, greater than or equal, if the end is
    ///    exclusive);
    /// 2. The start of the range is less than `I::ZERO`;
    /// 3. The end of the range is greater than `self.size()` if `Excluded`, or greater than or
    ///    equal to `self.size()` if `Included`.
    ///
    /// ALSO: This method will panic if the start bound is `Excluded`.
    ///
    /// # Drop glue
    ///
    /// **Note:** Currently, [`Iter`] can often require explicitly dropping it to release the
    /// borrow on [`RleTree`]. The "nightly" feature adds `#[may_dangle]` to skip that requirement,
    /// but we may also improve this crate's internals in the future to remove that need in the
    /// future.
    ///
    /// # Example usage
    ///
    /// ```
    /// # use std::ops::Range;
    /// use sherman::{Constant, RleTree, SliceEntry};
    ///
    /// let mut tree: RleTree<usize, Constant<&'static str>> = RleTree::new_empty();
    ///
    /// // Add values so that the tree looks like:
    /// //
    /// //   | "foo" | "bar" | "baz" |
    /// //   0       5       10      15
    /// //
    /// tree.insert(0, Constant("foo"), 5);
    /// tree.insert(5, Constant("bar"), 5);
    /// tree.insert(10, Constant("baz"), 5);
    ///
    /// // Helper for assertions below
    /// fn pair<'e, 's>(
    ///     entry: SliceEntry<'e, usize, Constant<&'s str>>,
    /// ) -> (Range<usize>, &'e Constant<&'s str>) {
    ///     (entry.range(), entry.slice())
    /// }
    ///
    /// // Iterating over the full range:
    /// let mut iter = tree.iter(..).map(pair);
    /// assert_eq!(iter.next(), Some((0..5, &Constant("foo"))));
    /// assert_eq!(iter.next_back(), Some((10..15, &Constant("baz"))));
    /// assert_eq!(iter.next(), Some((5..10, &Constant("bar"))));
    /// assert_eq!(iter.next(), None);
    ///
    /// // Iterating over a partial range includes all values it touches:
    /// assert_eq!(
    ///     tree.iter(6..12).map(pair).collect::<Vec<_>>(),
    ///     [(5..10, &Constant("bar")), (10..15, &Constant("baz")),],
    /// );
    ///
    /// // Zero-length iteration only returns something if it's within a value:
    /// assert_eq!(tree.iter(3..3).count(), 1);
    /// assert_eq!(tree.iter(5..5).count(), 0);
    /// ```
    pub fn iter<R>(&self, range: R) -> Iter<'_, I, S, P>
    where
        R: std::ops::RangeBounds<I>,
    {
        Iter::new(self, range.start_bound(), range.end_bound())
    }

    /// Inserts the slice at position `idx`, shifting all later entries by `size`
    ///
    /// If there is any entry that contains `idx`, it will be split and encompass `slice` on either
    /// side after the insertion (unless `slice` joins with either/both sides).
    ///
    /// # Algorithmic complexity
    ///
    /// This operation is `O(log(r))`, where `r` is the number of ranges of values in the tree.
    ///
    /// # Panics
    ///
    /// This method will panic if `idx` is *greater* than [`self.size()`]. An index equal to the
    /// current size of the tree is explicitly allowed. It will also panic if the size of the new
    /// slice is not greater than zero — i.e. if `size <= I::ZERO`.
    ///
    /// [`self.size()`]: Self::size
    ///
    /// # Example usage
    ///
    /// ```
    /// use sherman::{Constant, RleTree, SliceEntry};
    ///
    /// let mut tree: RleTree<usize, Constant<&'static str>> = RleTree::new_empty();
    ///
    /// // Append values to the end, so the tree looks like:
    /// //
    /// //   | "foo" | "bar" |
    /// //   0       5       10
    /// //
    /// tree.insert(0, Constant("foo"), 5);
    /// tree.insert(5, Constant("bar"), 4);
    ///
    /// let fst = tree.get(3);
    /// assert_eq!(fst.range(), 0..5);
    /// assert_eq!(fst.slice(), &Constant("foo"));
    /// let snd = tree.get(5);
    /// assert_eq!(snd.range(), 5..9);
    /// assert_eq!(snd.slice(), &Constant("bar"));
    /// drop((fst, snd)); // (required without the "nightly" feature)
    ///
    /// // Adding a new value in the middle shifts the indexes of later values.
    /// tree.insert(5, Constant("baz"), 3);
    /// // See how "bar" has been shifted:
    /// let last = tree.get(10);
    /// assert_eq!(last.range(), 8..12);
    /// assert_eq!(last.slice(), &Constant("bar"));
    /// drop(last);
    ///
    /// // Inserting in the middle of a value will split it.
    /// // For `Constant`, splitting just means cloning the inner value:
    /// tree.insert(2, Constant("<split>"), 4);
    ///
    /// let mut iter = tree.iter(..9).map(|entry| (entry.range(), entry.slice()));
    /// assert_eq!(iter.next(), Some((0..2, &Constant("foo"))));
    /// assert_eq!(iter.next(), Some((2..6, &Constant("<split>"))));
    /// assert_eq!(iter.next(), Some((6..9, &Constant("foo"))));
    /// drop(iter);
    ///
    /// // If the inserted value can "join" with either adjacent value, they
    /// // will be merged into a single slice.
    /// // For `Constant`, joining is allowed if the values are equal.
    /// //
    /// // Before:
    /// let mid = tree.get(9);
    /// assert_eq!(mid.range(), 9..12);
    /// assert_eq!(mid.slice(), &Constant("baz"));
    /// drop(mid);
    /// // Join at the edge:
    /// tree.insert(12, Constant("baz"), 2);
    /// assert_eq!(tree.get(10).range(), 9..14);
    /// // Re-join after insertion in the middle:
    /// tree.insert(10, Constant("baz"), 4);
    /// assert_eq!(tree.get(10).range(), 9..18);
    /// ```
    #[inline(always)]
    pub fn insert(&mut self, idx: I, slice: S, size: I)
    where
        P: SupportsUpdate<I, S>,
    {
        let mut slice = Some(slice); // Wrap in `Some(_)` so that we can pass &mut Option<S>

        if idx < I::ZERO {
            panic!("index {idx:?} out of bounds, less than zero");
        } else if idx > self.size() {
            panic!("index {idx:?} out of bounds for size {:?}", self.size());
        } else if size <= I::ZERO {
            panic!("cannot insert new slice with non-positive size {size:?}");
        }

        let mut root = match self.root.take() {
            Some(r) => r.handle.into_unique(),
            None => {
                // This tree is completely empty, so we can actually just initialize it to just
                // contain the value we want, and return. Given that `self.size()` must be zero, we
                // know that `idx` is also zero.
                *self = Self::new_from_opt(size, &mut slice);
                return;
            }
        };

        run_insert(root.borrow_mut(), None, false, DownwardInsertState::new(idx, &mut slice, size));
        root = fix::fix_unique_owned(root, FixMode::Normal);
        self.root = Some(Root { handle: root.erase() });
    }

    /// Replaces a range of values in the tree with a single new value, returning a [`Removed`]
    /// object representing what was replaced
    ///
    /// # Algorithmic complexity
    ///
    /// This operation is `O(log(r))`, where `r` is the number of ranges of values in the tree.
    ///
    /// Without COW, dropping the [`Removed`] values is `O(k)` (where `k` is the number of ranges
    /// of values removed).  
    /// With [`EnableCow`], it is `O(q + log(s))` (where `q` is the number of *unique* ranges of
    /// values removed and `s` is the number of shared ranges of values).
    ///
    /// # Panics
    ///
    /// This method panics under the same conditions as [`remove`](Self::remove):
    ///
    /// * The range's start bound is exclusive
    /// * The range's **end bound is inclusive** (e.g. `1..=3`)
    /// * The range's start bound is less than `I::ZERO`
    /// * The range's end bound is greater than the [`size`](Self::size) of the tree
    /// * The range's start bound is greater than its end bound
    ///
    /// # Example usage
    ///
    /// ```
    /// use sherman::{Constant, RleTree};
    ///
    /// let mut tree: RleTree<usize, Constant<&'static str>> = RleTree::new_empty();
    ///
    /// // Insert values so the tree looks like:
    /// //
    /// //   | "foo" | "bar" |
    /// //   0       5       10
    /// //
    /// tree.insert(0, Constant("foo"), 5);
    /// tree.insert(5, Constant("bar"), 5);
    ///
    /// assert_eq!(tree.size(), 10);
    ///
    /// // Replacing leaves the size of the tree unchanged.
    /// let removed = tree.replace(3..7, Constant("baz"));
    /// assert_eq!(tree.size(), 10);
    ///
    /// // ... and the values are as described:
    /// assert_eq!(
    ///     tree.iter(..).map(|e| (e.range(), e.slice())).collect::<Vec<_>>(),
    ///     [
    ///         (0..3, &Constant("foo")),
    ///         (3..7, &Constant("baz")), // <- replacement
    ///         (7..10, &Constant("bar")),
    ///     ],
    /// );
    /// assert_eq!(
    ///     removed.into_iter().collect::<Vec<_>>(),
    ///     // Note that the original ranges are preserved:
    ///     [(3..5, Constant("foo")), (5..7, Constant("bar"))],
    /// );
    /// ```
    #[inline(always)]
    pub fn replace<R>(&mut self, range: R, value: S) -> Removed<I, S, P>
    where
        P: SupportsUpdate<I, S>,
        R: std::ops::RangeBounds<I>,
    {
        let mut value = Some(value);
        let range = remove::check_bounds(self, "replace", range.start_bound(), range.end_bound());
        remove::remove(self, range, Some(&mut value))
    }

    /// Removes a range of values from the tree, returning a [`Removed`] object representing them.
    ///
    /// To process the removed values, use [`drain`](Self::drain) on the tree or
    /// [`Removed::into_tier`](IntoIterator); or [`Removed::into_tree`] to turn it into a full
    /// [`RleTree`].
    ///
    /// # Algorithmic complexity
    ///
    /// This operation is `O(log(r))`, where `r` is the number of ranges of values in the tree.
    ///
    /// Without COW, dropping the [`Removed`] values is `O(k)` (where `k` is the number of ranges
    /// of values removed).  
    /// With [`EnableCow`], it is `O(q + log(s))` (where `q` is the number of *unique* ranges of
    /// values removed and `s` is the number of shared ranges of values).
    ///
    /// # Panics
    ///
    /// This method panics if:
    ///
    /// * The range's start bound is exclusive
    /// * The range's **end bound is inclusive** (e.g. `1..=3`)
    /// * The range's start bound is less than `I::ZERO`
    /// * The range's end bound is greater than the [`size`](Self::size) of the tree
    /// * The range's start bound is greater than its end bound
    ///
    /// We cannot allow inclusive end bounds because that would require an increment operator to
    /// transform into an equivalent exclusive end bound, and index types might not necessarily
    /// have a natural definition of "increment".
    ///
    /// # Example usage
    ///
    /// ```
    /// use sherman::{Constant, RleTree};
    ///
    /// let mut tree: RleTree<usize, Constant<&'static str>> = RleTree::new_empty();
    ///
    /// // Insert values so the tree looks like:
    /// //
    /// //   | "foo" | "bar" |
    /// //   0       5       10
    /// //
    /// tree.insert(0, Constant("foo"), 5);
    /// tree.insert(5, Constant("bar"), 5);
    ///
    /// assert_eq!(tree.size(), 10);
    ///
    /// let removed = tree.remove(2..6);
    ///
    /// // The tree is now smaller by the removed size:
    /// assert_eq!(removed.size(), 4);
    /// assert_eq!(tree.size(), 6);
    ///
    /// // ... and the parts of the original values are no longer present:
    /// assert_eq!(
    ///     tree.iter(..).map(|e| (e.range(), e.slice())).collect::<Vec<_>>(),
    ///     [(0..2, &Constant("foo")), (2..6, &Constant("bar"))],
    /// );
    ///
    /// // The removed range can also be turned into its own tree:
    /// let removed_tree = removed.into_tree();
    ///
    /// assert_eq!(
    ///     removed_tree
    ///         .iter(..)
    ///         .map(|e| (e.range(), e.slice()))
    ///         .collect::<Vec<_>>(),
    ///     [(0..3, &Constant("foo")), (3..4, &Constant("bar"))],
    /// );
    /// ```
    #[inline(always)]
    pub fn remove<R>(&mut self, range: R) -> Removed<I, S, P>
    where
        P: SupportsUpdate<I, S>,
        R: std::ops::RangeBounds<I>,
    {
        let range = remove::check_bounds(self, "remove", range.start_bound(), range.end_bound());
        remove::remove(self, range, None)
    }

    /// Removes a range of values from the tree, returning an iterator over the values
    ///
    /// This is basically just a convenience method for `self.remove().into_iter()`.
    ///
    /// # Algorithmic complexity
    ///
    /// Creating the iterator is `O(log(r))`, where `r` is the number of ranges of values in the
    /// tree. Each step of iterating the [`Drain`] is `O(log(k))` but amortized `O(1)`, where `k`
    /// is the number of ranges of values removed.
    ///
    /// Without COW, dropping the [`Drain`] is `O(k)`.  
    /// With [`EnableCow`], it is `O(q + log(s))` (where `q` is the number of *unique* ranges of
    /// values removed and `s` is the number of shared ranges of values).
    ///
    /// # Panics
    ///
    /// This method panics under the same conditions as [`remove`](Self::remove):
    ///
    /// * The range's start bound is exclusive
    /// * The range's **end bound is inclusive** (e.g. `1..=3`)
    /// * The range's start bound is less than `I::ZERO`
    /// * The range's end bound is greater than the [`size`](Self::size) of the tree
    /// * The range's start bound is greater than its end bound
    pub fn drain<R>(&mut self, range: R) -> Drain<I, S, P>
    where
        P: SupportsUpdate<I, S>,
        R: std::ops::RangeBounds<I>,
    {
        let range = remove::check_bounds(self, "drain", range.start_bound(), range.end_bound());
        let removed = remove::remove(self, range, None);
        Drain::new(removed)
    }
}

impl<I, S> RleTree<I, S, EnableCow>
where
    I: Index,
    S: Slice<I>,
{
    /// Creates a new copy-on-write reference to the same tree
    ///
    /// # Algorithmic complexity
    ///
    /// This operation is `O(1)`.
    pub fn shallow_clone(&self) -> Self {
        let root = self.root.as_ref().map(|r| Root { handle: r.handle.shallow_clone() });

        RleTree { root }
    }
}

#[derive(Debug)]
enum SearchResult<I> {
    Lhs { offset: I },
    LhsEdge,
    Value { range: Range<I>, offset_in_range: I },
    RhsEdge,
    Rhs { offset: I },
}

fn search_step<I, S, P>(node: node::HandleImmut<I, S, P>, target: I) -> SearchResult<I>
where
    I: Index,
    P: RleTreeConfig<I, S>,
{
    let value_range = node.value_range();

    if target < value_range.start {
        SearchResult::Lhs { offset: target }
    } else if target > value_range.end {
        SearchResult::Rhs { offset: target.sub_left(value_range.end) }
    } else if target == value_range.start {
        SearchResult::LhsEdge
    } else if target == value_range.end {
        SearchResult::RhsEdge
    } else {
        SearchResult::Value {
            offset_in_range: target.sub_left(value_range.start),
            range: value_range,
        }
    }
}

#[derive(Copy, Clone)]
#[cfg_attr(test, derive(Debug))]
enum SearchBound<I> {
    Included(I),
    Excluded(I),
}

fn search_node<'t, I, S, P>(
    root: node::HandleImmut<'t, I, S, P>,
    target: SearchBound<I>,
) -> (node::HandleImmut<'t, I, S, P>, Range<I>, I)
where
    I: Index,
    P: RleTreeConfig<I, S>,
{
    search(root, target, |n| n.reborrow(), |_, child, _| child)
}

fn search<'t, I, S, P, St, Bo, Ch>(
    state: St,
    target: SearchBound<I>,
    borrow: Bo,
    into_child: Ch,
) -> (St, Range<I>, I)
where
    I: 't + Index,
    S: 't,
    P: RleTreeConfig<I, S>,
    Bo: Fn(&St) -> node::HandleImmut<'t, I, S, P>,
    Ch: Fn(St, node::HandleImmut<'t, I, S, P>, Side) -> St,
{
    let mut node = state;
    let (mut target, exclusive) = match target {
        SearchBound::Included(i) => (i, false),
        SearchBound::Excluded(i) => (i, true),
    };

    let (range, offset_in_range) = loop {
        match search_step(borrow(&node), target) {
            SearchResult::Lhs { offset } => {
                target = offset;
                node = match borrow(&node).into_lhs() {
                    Some(n) => into_child(node, n, Side::Lhs),
                    None => crate::panic_internal_error_or_bad_index::<I>(
                        "`SearchResult::Lhs` implies the left-hand child should exist",
                    ),
                };
            }
            SearchResult::RhsEdge => {
                if exclusive {
                    let r = borrow(&node).value_range();
                    break (r.clone(), r.end.sub_left(r.start));
                } else {
                    target = I::ZERO;
                    node = match borrow(&node).into_rhs() {
                        Some(n) => into_child(node, n, Side::Rhs),
                        None => crate::panic_internal_error_or_bad_index::<I>(
                            "`SearchResult::RhsEdge` implies the right-hand child should exist",
                        ),
                    };
                }
            }
            SearchResult::Rhs { offset } => {
                target = offset;
                node = match borrow(&node).into_rhs() {
                    Some(n) => into_child(node, n, Side::Rhs),
                    None => crate::panic_internal_error_or_bad_index::<I>(
                        "`SearchResult::Rhs` implies the right-hand child should exist",
                    ),
                };
            }
            SearchResult::LhsEdge => {
                if !exclusive {
                    break (borrow(&node).value_range(), I::ZERO);
                } else {
                    node = match borrow(&node).into_lhs() {
                        Some(n) => into_child(node, n, Side::Lhs),
                        None => crate::panic_internal_error_or_bad_index::<I>(
                            "`SearchResult::LhsEdge` implies the left-hand child should exist",
                        ),
                    };
                    target = borrow(&node).subtree_size();
                }
            }
            SearchResult::Value { range, offset_in_range } => break (range, offset_in_range),
        }
    };

    (node, range, offset_in_range)
}

struct DownwardInsertState<'s, I, S> {
    target: I,
    fst_value: InsertionValue<'s, I, S>,
    snd_value: Option<InsertionValue<'s, I, S>>,
    allow_joining: bool,
    already_split_once: bool,
}

#[cfg(test)]
impl<'s, I: Debug, S: Debug> Debug for DownwardInsertState<'s, I, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = f.debug_struct("DownwardInsertState");
        s.field("target", &self.target);
        s.field("fst_value", &format_args!("{:?}", self.fst_value));
        s.field("snd_value", &format_args!("{:?}", self.snd_value));
        s.field("allow_joining", &self.allow_joining);
        s.field("already_split_once", &self.already_split_once);
        s.finish()
    }
}

#[cfg_attr(test, derive(Debug))]
struct InsertionValue<'s, I, S> {
    slice: &'s mut Option<S>,
    size: I,
}

#[cfg_attr(test, derive(Debug))]
struct UpwardUpdateState<I> {
    old_size: I,
}

/// Traverses down the tree, inserts the value, and traverses back up the tree to adjust the
/// subtree sizes of each node.
///
/// Upward traversal will terminate at `root`, and `root`'s subtree size will be updated to match.
fn run_insert<'t, I, S, P>(
    root: node::HandleMut<'t, I, S, P>,
    root_subtree_size: Option<I>,
    mut force_edge_rhs: bool,
    state: DownwardInsertState<I, S>,
) -> node::HandleMut<'t, I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    let root_addr = root.addr();

    let mut down_state = state;
    let mut node = root;

    let mut up_state = loop {
        let (n, cf) = if force_edge_rhs {
            force_edge_rhs = false;
            down_state.step_edge_rhs(node)
        } else {
            down_state.step(node)
        };
        node = n;
        match cf {
            ControlFlow::Continue(s) => down_state = s,
            ControlFlow::Break(up_state) => break up_state,
        }
    };

    while node.addr() != root_addr {
        let parent_addr = node.parent_addr().expect("parent addr should be Some(_)");

        node = fix::fix_mut(node, FixMode::Normal);

        let override_size = match parent_addr == root_addr {
            true => root_subtree_size,
            false => None,
        };
        (node, up_state) = up_state.step(node, override_size);
    }

    if node.has_parent() {
        node = fix::fix_mut(node, FixMode::Normal);
    }

    node
}

impl<'s, I: Index, S: Slice<I>> DownwardInsertState<'s, I, S> {
    fn new(target: I, slice: &'s mut Option<S>, size: I) -> Self {
        DownwardInsertState {
            target,
            fst_value: InsertionValue { slice, size },
            snd_value: None,
            allow_joining: true,
            already_split_once: false,
        }
    }

    fn step<P>(
        self,
        node: node::HandleMut<I, S, P>,
    ) -> (node::HandleMut<I, S, P>, ControlFlow<UpwardUpdateState<I>, Self>)
    where
        P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
    {
        match search_step(node.borrow(), self.target) {
            SearchResult::Lhs { offset } => match node.into_lhs() {
                // Recurse into LHS child:
                Ok(child) => (child, ControlFlow::Continue(Self { target: offset, ..self })),
                // No LHS, insert a new one.
                Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                    "search step returned LHS without LHS child",
                ),
            },
            SearchResult::Rhs { offset } => match node.into_rhs() {
                // Recurse into RHS child:
                Ok(child) => (child, ControlFlow::Continue(Self { target: offset, ..self })),
                // No RHS, even though our search told us we were outside the range of this value.
                Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                    "search step returned RHS without RHS child",
                ),
            },
            SearchResult::LhsEdge => self.step_edge_lhs(node),
            SearchResult::RhsEdge => self.step_edge_rhs(node),
            SearchResult::Value { range, offset_in_range } => {
                self.step_split_value(node, range, offset_in_range)
            }
        }
    }

    /// Sub-case of `step` that handles the case where the target index is at the edge of the node
    /// and its LHS child (if there is one).
    fn step_edge_lhs<P>(
        self,
        mut node: node::HandleMut<I, S, P>,
    ) -> (node::HandleMut<I, S, P>, ControlFlow<UpwardUpdateState<I>, Self>)
    where
        P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
    {
        // Target is on the edge of this node and LHS subtree; try joining the slice with this
        // node.
        if self.allow_joining {
            debug_assert!(self.snd_value.is_none());

            let node_value = node.value_mut();
            S::try_join_into_rhs(self.fst_value.slice, node_value);
            match self.fst_value.slice {
                Some(_) => {
                    // Couldn't join. Values are left as they are, so nothing left to do.
                }
                None => {
                    // Successfully joined with this node. We potentially could still join with a
                    // node to the left, if there is one. Any left-hand node higher up the tree
                    // would have already been attempted and failed, so we just need to traverse
                    // down the tree.

                    let old_subtree_size = node.subtree_size();
                    let new_subtree_size;
                    {
                        let range = node.value_range();
                        let lhs_size = range.start;
                        let value_size = range.end.sub_left(range.start);
                        let rhs_size = old_subtree_size.sub_left(range.end);

                        new_subtree_size = lhs_size
                            .add_right(self.fst_value.size)
                            .add_right(value_size)
                            .add_right(rhs_size);
                    }

                    let redirect = P::Redirect::to(node.borrow());
                    Self::try_join_traverse_lhs(node.borrow_mut(), redirect);

                    node.set_subtree_size(new_subtree_size);
                    return (
                        node,
                        ControlFlow::Break(UpwardUpdateState { old_size: old_subtree_size }),
                    );
                }
            }
        }

        // Couldn't join; recurse into the child or insert a new one.
        match node.into_lhs() {
            // Recurse into LHS child:
            Ok(child) => (child, ControlFlow::Continue(Self { target: self.target, ..self })),
            // No LHS, insert a new one:
            Err(n) => {
                let mut new_lhs =
                    node::NodeHandle::alloc_new(self.fst_value.slice, self.fst_value.size);

                if let Some(snd_value) = self.snd_value {
                    new_lhs.borrow_mut().insert_rhs(
                        node::NodeHandle::alloc_new(snd_value.slice, snd_value.size).erase(),
                    );
                    new_lhs.set_subtree_size(self.fst_value.size.add_right(snd_value.size));
                    new_lhs = fix::fix_unique_owned(new_lhs, FixMode::Normal);
                }

                let child = n.insert_into_lhs(new_lhs);

                (child, ControlFlow::Break(UpwardUpdateState { old_size: I::ZERO }))
            }
        }
    }

    /// Sub-case of `step` that handles the case where the target index is at the edge of the node
    /// and its RHS child (if there is one).
    fn step_edge_rhs<P>(
        self,
        mut node: node::HandleMut<I, S, P>,
    ) -> (node::HandleMut<I, S, P>, ControlFlow<UpwardUpdateState<I>, Self>)
    where
        P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
    {
        // Target is on the edge of this node and RHS subtree; try joining the slice
        // with this node.
        if self.allow_joining {
            debug_assert!(self.snd_value.is_none());

            let node_value = node.value_mut();
            S::try_join_into_lhs(node_value, self.fst_value.slice);
            match self.fst_value.slice {
                Some(_) => {
                    // Couldn't join. Values are left as they are, so nothing left to do.
                }
                None => {
                    // Successfully joined with this node. If we haven't already tried joining with
                    // the node immediately to the right, we need to traverse the tree to try_join
                    // again.

                    let old_subtree_size = node.subtree_size();
                    let new_subtree_size;
                    {
                        let range = node.value_range();
                        let lhs_size = range.start;
                        let value_size = range.end.sub_left(range.start);
                        let rhs_size = old_subtree_size.sub_left(range.end);

                        new_subtree_size = lhs_size
                            .add_right(value_size)
                            .add_right(self.fst_value.size)
                            .add_right(rhs_size);
                    }

                    let redirect = P::Redirect::to(node.borrow());
                    Self::try_join_traverse_rhs(node.borrow_mut(), redirect);

                    node.set_subtree_size(new_subtree_size);
                    return (
                        node,
                        ControlFlow::Break(UpwardUpdateState { old_size: old_subtree_size }),
                    );
                }
            }
        }

        // Couldn't join; recurse into the child or insert a new one.
        match node.into_rhs() {
            // Recurse into RHS child:
            Ok(child) => (child, ControlFlow::Continue(Self { target: I::ZERO, ..self })),
            // No RHS, insert a new one:
            Err(n) => {
                let mut new_rhs =
                    node::NodeHandle::alloc_new(self.fst_value.slice, self.fst_value.size);

                if let Some(snd_value) = self.snd_value {
                    new_rhs.borrow_mut().insert_rhs(
                        node::NodeHandle::alloc_new(snd_value.slice, snd_value.size).erase(),
                    );
                    new_rhs.set_subtree_size(self.fst_value.size.add_right(snd_value.size));
                }

                let child = n.insert_into_rhs(new_rhs);

                (child, ControlFlow::Break(UpwardUpdateState { old_size: I::ZERO }))
            }
        }
    }

    /// Tries to join `root_value` with the node immediately left of `subtree_root`, via its
    /// left-hand child (i.e., the RIGHT-MOST node in the subtree rooted at its left-hand child).
    ///
    /// Returns the result of attempting to join with `root_value`.
    fn try_join_traverse_lhs<P>(mut subtree_root: node::HandleMut<I, S, P>, redirect: P::Redirect)
    where
        P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
    {
        let root_addr = subtree_root.addr();

        // SAFETY: We're generating a longer-lived reference `subtree_root`'s value. This is sound
        // only because further usage of `subtree_root` only touches the `lhs` field.
        let root_value = unsafe {
            let v: &mut Option<S> = subtree_root.value_mut();
            &mut *(v as *mut Option<S>) // produce a reference, not directly borrowing subtree_root
        };

        let mut immediate_lhs = match subtree_root.into_lhs() {
            Err(_) => return,
            Ok(child) => child,
        };

        loop {
            // To get the *immediate* left-hand node, we have to first get the left-hand child, and
            // then keep following right-hand children until we get to the end.
            match immediate_lhs.into_rhs() {
                // Has a right-hand child! Keep going down the tree.
                Ok(c) => immediate_lhs = c,
                // No more right-hand children; must be this node!
                // Try joining.
                Err(n) => {
                    immediate_lhs = n;
                    break;
                }
            }
        }

        let lhs_value = immediate_lhs.value_mut();
        S::try_join_into_rhs(lhs_value, root_value);
        match lhs_value {
            // Couldn't join, nothing left to do.
            Some(_) => return,
            // Did join -- we have complex operations ahead. More below.
            None => immediate_lhs.write_redirect(redirect),
        }

        // Joined! Let's remove this lower node, replacing it with its own left-hand child, if
        // there is one - because we already know it doesn't have a right-hand child.
        let mut lower_lhs = immediate_lhs.take_lhs();

        let lower_lhs_size = lower_lhs.as_ref().map(|lhs| lhs.subtree_size()).unwrap_or(I::ZERO);
        let removed_size = immediate_lhs.subtree_size().sub_left(lower_lhs_size);

        let mut replaced_empty_node = false;

        // Traverse back up the tree, until we get back to `subtree_root`.
        // On the first parent, we'll need to reinsert `lower_lhs`.
        let mut upward_child = immediate_lhs;
        loop {
            let (mut parent, side) = upward_child
                .into_parent()
                .expect("internal error: bad traversal: node must have a parent");

            match side {
                // On the left-hand side of the parent? That means we must *already* be at
                // `subtree_root`. Replace the LHS if we haven't already, and we're done.
                Side::Lhs => {
                    assert!(parent.addr() == root_addr);

                    if !replaced_empty_node {
                        // remove the node we took the value from.
                        drop(parent.take_lhs());
                        // reinsert the node's left-hand child, if it had one:
                        if let Some(lhs) = lower_lhs.take() {
                            parent.insert_lhs(lhs);
                        }
                    }

                    return;
                }
                // Right-hand side of the parent means there's more recursion we'll have to do.
                // Let's fix the immediate parent if we need to, and then continue upwards.
                Side::Rhs => {
                    if !replaced_empty_node {
                        // remove the node we took the value from.
                        drop(parent.take_rhs());
                        // reinsert the node's left-hand child, if it had one:
                        if let Some(lhs) = lower_lhs.take() {
                            parent.insert_rhs(lhs);
                        }
                        replaced_empty_node = true;
                    }

                    let new_subtree_size = parent.subtree_size().sub_right(removed_size);
                    parent.set_subtree_size(new_subtree_size);

                    parent = fix::fix_mut(parent, FixMode::Normal);

                    // recurse upwards
                    upward_child = parent;
                }
            }
        }
    }

    /// Tries to join `root_value` with the node immediately right of `subtree_root`, via its
    /// right-hand child (i.e., the LEFT-MOST node in the subtree rooted at its right-hand child).
    ///
    /// Returns the result of attempting to join with `root_value`.
    fn try_join_traverse_rhs<P>(mut subtree_root: node::HandleMut<I, S, P>, redirect: P::Redirect)
    where
        P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
    {
        let root_addr = subtree_root.addr();

        // SAFETY: We're generating a longer-lived reference `subtree_root`'s value. This is sound
        // only because further usage of `subtree_root` only touches the `rhs` field.
        let root_value = unsafe {
            let v: &mut Option<S> = subtree_root.value_mut();
            &mut *(v as *mut Option<S>) // produce a reference, not directly borrowing subtree_root
        };

        let mut immediate_rhs = match subtree_root.into_rhs() {
            Err(_) => return,
            Ok(child) => child,
        };

        loop {
            // To get the *immediate* right-hand node, we have to first get the right-hand child, and
            // then keep following left-hand children until we get to the end.
            match immediate_rhs.into_lhs() {
                // Has a left-hand child! Keep going down the tree.
                Ok(c) => immediate_rhs = c,
                // No more left-hand children; must be this node!
                // Try joining.
                Err(n) => {
                    immediate_rhs = n;
                    break;
                }
            }
        }

        let rhs_value = immediate_rhs.value_mut();
        S::try_join_into_lhs(root_value, rhs_value);
        match rhs_value {
            // Couldn't join, nothing left to do.
            Some(_) => return,
            // Did join -- we have complex operations ahead. More below.
            None => immediate_rhs.write_redirect(redirect),
        }

        // Joined! Let's remove this lower node, replacing it with its own left-hand child, if
        // there is one - because we already know it doesn't have a right-hand child.
        let mut lower_rhs = immediate_rhs.take_rhs();

        let lower_rhs_size = lower_rhs.as_ref().map(|rhs| rhs.subtree_size()).unwrap_or(I::ZERO);
        let removed_size = immediate_rhs.subtree_size().sub_right(lower_rhs_size);

        let mut replaced_empty_node = false;

        // Traverse back up the tree, until we get back to `subtree_root`.
        // On the first parent, we'll need to reinsert `lower_lhs`.
        let mut upward_child = immediate_rhs;
        loop {
            let (mut parent, side) = upward_child
                .into_parent()
                .expect("internal error: bad traversal: node must have a parent");

            match side {
                // On the right-hand side of the parent? That means we must *already* be at
                // `subtree_root`. Replace the RHS if we haven't already, and we're done.
                Side::Rhs => {
                    assert!(parent.addr() == root_addr);

                    if !replaced_empty_node {
                        // remove the node we took the value from.
                        drop(parent.take_rhs());
                        // reinsert the node's left-hand child, if it had one:
                        if let Some(rhs) = lower_rhs.take() {
                            parent.insert_rhs(rhs);
                        }
                    }

                    return;
                }
                // Left-hand side of the parent means there's more recursion we'll have to do.
                // Let's fix the immediate parent if we need to, then continue upwards.
                Side::Lhs => {
                    if !replaced_empty_node {
                        // Remove the node we took the value from.
                        drop(parent.take_lhs());
                        // reinsert the node's right-hand child, if it had one:
                        if let Some(rhs) = lower_rhs.take() {
                            parent.insert_lhs(rhs);
                        }
                        replaced_empty_node = true;
                    }

                    let new_subtree_size = parent.subtree_size().sub_left(removed_size);
                    parent.set_subtree_size(new_subtree_size);

                    parent = fix::fix_mut(parent, FixMode::Normal);

                    // recurse upwards
                    upward_child = parent;
                }
            }
        }
    }

    fn step_split_value<P>(
        mut self,
        mut node: node::HandleMut<I, S, P>,
        range: Range<I>,
        offset_in_range: I,
    ) -> (node::HandleMut<I, S, P>, ControlFlow<UpwardUpdateState<I>, Self>)
    where
        P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
    {
        // At this point: the value is in the middle of the range, so we need to split this
        // node.
        // We'll end up with something like:
        //
        //            ┏━━━━━━━━━━━┓
        //            ┃ this node ┃
        //            ┗━━━━━━━━━━━┛
        //                  ⇓
        //   ┏━━━━━━━━━━┱───────┲━━━━━━━━━━┓
        //   ┃ this lhs ┃ slice ┃ this rhs ┃
        //   ┗━━━━━━━━━━┹───────┺━━━━━━━━━━┛
        //
        // and `slice` can join with either (or both!) of the parts of the original node.
        // If `slice` joins with only one, we'll have to insert one value.
        // If `slice` joins with neither, we'll have to insert two values (!!)

        if self.already_split_once {
            // When we've already perfored a single split, the 1-2 new values will be inserted
            // EXACTLY at the right-hand edge of where the preexisting value was, so we should
            // never find that we need to split another value.
            //
            // So if we get to here (needing to perform a split) finding that we ALREADY split
            // once, then something has gone wrong.
            crate::panic_internal_error_or_bad_index::<I>("would double-split");
        }

        let original_size = node.subtree_size();
        let node_lhs_size = range.start;
        let node_rhs_size = original_size.sub_left(range.end);

        let split_lhs = node.value_mut();
        let mut split_rhs = None;
        S::split_at_mut(split_lhs, offset_in_range, &mut split_rhs);
        let split_lhs_size = offset_in_range;
        let split_rhs_size = range.end.sub_left(range.start).sub_left(offset_in_range);

        let replacement_size: I;
        let to_insert: Option<(InsertionValue<I, S>, Option<InsertionValue<I, S>>)>;

        // Try joining `slice` to lhs:
        S::try_join_into_lhs(split_lhs, self.fst_value.slice);
        match self.fst_value.slice {
            None => {
                // Joined with LHS. Try to re-join with RHS.
                S::try_join_into_lhs(split_lhs, &mut split_rhs);
                match &split_rhs {
                    None => {
                        // Successfully joined all three pieces. Nothing left to do.
                        // replacement = final_value;
                        replacement_size =
                            split_lhs_size.add_right(self.fst_value.size).add_right(split_rhs_size);
                        to_insert = None;
                    }
                    Some(_) => {
                        // Joined LHS+slice but not RHS. We'll have to re-insert it.
                        replacement_size = split_lhs_size.add_right(self.fst_value.size);
                        to_insert = Some((
                            InsertionValue { slice: &mut split_rhs, size: split_rhs_size },
                            None,
                        ));
                    }
                }
            }
            Some(_) => {
                // Couldn't join with LHS. Try joining with RHS.
                replacement_size = split_lhs_size;

                S::try_join_into_lhs(self.fst_value.slice, &mut split_rhs);
                match &split_rhs {
                    None => {
                        // Joined slice+RHS, but didn't join with LHS earlier.
                        // We'll have to re-insert slice+RHS.
                        self.fst_value.size = self.fst_value.size.add_right(split_rhs_size);
                        to_insert = Some((self.fst_value, None));
                    }
                    Some(_) => {
                        // Didn't join with either LHS or RHS.
                        to_insert = Some((
                            self.fst_value,
                            Some(InsertionValue { slice: &mut split_rhs, size: split_rhs_size }),
                        ));
                    }
                }
            }
        }

        let old_subtree_size = original_size;
        let insertion_size = match &to_insert {
            None => I::ZERO,
            Some((fst_value, None)) => fst_value.size,
            Some((fst_value, Some(snd_value))) => fst_value.size.add_right(snd_value.size),
        };

        let new_subtree_size = node_lhs_size
            .add_right(replacement_size)
            .add_right(insertion_size)
            .add_right(node_rhs_size);
        node.set_subtree_size(new_subtree_size);

        if let Some((fst_value, snd_value)) = to_insert {
            // Insert the new value(s) - recursion will be bounded by the check at the top of THIS
            // function, which checks that we don't perform a second split.
            // `run_insert` will stop at this node.
            node = run_insert(
                node,
                Some(new_subtree_size),
                true,
                DownwardInsertState {
                    target: new_subtree_size.sub_right(node_rhs_size),
                    fst_value,
                    snd_value,
                    allow_joining: false, // already checked joining above.
                    already_split_once: true,
                },
            );
        }

        (node, ControlFlow::Break(UpwardUpdateState { old_size: old_subtree_size }))
    }
}

impl<I: Index> UpwardUpdateState<I> {
    fn step<S, P>(
        self,
        node: node::HandleMut<I, S, P>,
        override_parent_subtree_size: Option<I>,
    ) -> (node::HandleMut<I, S, P>, Self)
    where
        S: Slice<I>,
        P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
    {
        let lower_old_subtree_size = self.old_size;
        let lower_new_subtree_size = node.subtree_size();
        let lower_addr = node.addr();

        let Some((mut parent, side)) = node.into_parent() else {
            panic!("internal error: tried to `UpwardUpdateState::step` a node with no parent");
        };

        match side {
            Side::Lhs => {
                assert_eq!(parent.borrow().into_lhs().map(|n| n.addr()), Some(lower_addr));

                let old_parent_size = parent.subtree_size();
                let new_parent_size = override_parent_subtree_size.unwrap_or_else(|| {
                    old_parent_size
                        .sub_left(lower_old_subtree_size)
                        .add_left(lower_new_subtree_size)
                });
                parent.set_subtree_size(new_parent_size);

                (parent, UpwardUpdateState { old_size: old_parent_size })
            }
            Side::Rhs => {
                assert_eq!(parent.borrow().into_rhs().map(|n| n.addr()), Some(lower_addr));

                let old_parent_size = parent.subtree_size();
                let new_parent_size = override_parent_subtree_size.unwrap_or_else(|| {
                    old_parent_size
                        .sub_right(lower_old_subtree_size)
                        .add_right(lower_new_subtree_size)
                });
                parent.set_subtree_size(new_parent_size);

                (parent, UpwardUpdateState { old_size: old_parent_size })
            }
        }
    }
}
