use std::cell::Cell;
use std::ops::Range;

use super::{borrow, node};
use crate::param::{self, RleTreeConfig};
use crate::{Index, RleTree, Slice};

/// Information about a single slice in an [`RleTree`], returned by [`RleTree::get`] or yielded by
/// [`RleTree::iter`].
///
/// Conceptually, this type is basically just `(Range<I>, &'t S)`, with the notable addition that
/// it can also be used to produce a "stable reference" to the value acros modification (see
/// [`stable_ref`] or [`StableRef`] for more).
///
/// [`stable_ref`]: Self::stable_ref
pub struct SliceEntry<'t, I, S, P: RleTreeConfig<I, S> = param::NoFeatures> {
    pub(super) range: Range<I>,
    pub(super) slice: node::NodeHandle<borrow::Immut<'t, node::Node<I, S, P>>>,
}

// SAFETY: A SliceEntry has the same capabilities as `&RleTree<I, S>`, and it is safe to send an
// immutable reference across threads if and only if the type implements Sync.
// So we can implement Send for SliceEntry only when the RleTree is Sync.
unsafe impl<'t, I, S, P> Send for SliceEntry<'t, I, S, P> where
    P: RleTreeConfig<I, S> + param::RleTreeIsSync<I, S>
{
}

impl<'t, I, S, P> SliceEntry<'t, I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    /// Returns the range of values covered by this entry
    pub fn range(&self) -> Range<I> {
        self.range.start..self.range.end
    }

    /// Returns the length of the range covered by this entry
    ///
    /// This is roughly equivalent to `self.range().len()`, but respects the potentially
    /// directional aspect of [`Index`] arithmetic.
    pub fn size(&self) -> I {
        self.range.end.sub_right(self.range.start)
    }

    /// Returns a reference to the slice for this entry
    pub fn slice(&self) -> &'t S {
        self.slice.value()
    }
}

impl<'t, I, S> SliceEntry<'t, I, S, param::EnableRefs> {
    /// Produces a stable reference to a value, so that it can be tracked across modifications to
    /// the tree
    ///
    /// See [`StableRef`] for more.
    pub fn stable_ref(&self) -> StableRef<I, S> {
        StableRef { handle: Cell::new(Some(self.slice.weak())) }
    }
}

/// Stable reference to a value in an [`RleTree`] to allow fetching after updates to the tree
///
/// New `StableRef`s are produced by [`SliceEntry::stable_ref`].
///
/// # Semantics
///
/// A `StableRef` will track a value as it evolves in the tree, referencing the same [`Slice`] after
/// merging with an adjacent value and tracking the left-hand side if it is ever split.
///
/// Some examples:
///
/// ```
/// # fn main() {
/// use sherman::{Constant, RleTree, param};
///
/// let mut tree: RleTree<usize, Constant<char>, param::EnableRefs> =
///     RleTree::new_empty();
/// tree.insert(0, Constant('A'), 5);
///
/// let ref_a = tree.get(3).stable_ref();
/// assert_eq!(ref_a.range(&tree), Some(0..5));
/// assert_eq!(ref_a.slice(&tree), Some(&Constant('A')));
///
/// // If we insert values to the left, we can fetch the new position:
/// tree.insert(0, Constant('B'), 4);
/// tree.insert(0, Constant('A'), 3);
/// // The tree is now:
/// //   | A | B | A |
/// //           ^^^^^ `ref_a`
/// assert_eq!(ref_a.range(&tree), Some(7..12));
///
/// // If we split the value, it stays with the left-hand side:
/// tree.insert(10, Constant('C'), 2);
/// // The tree is now:
/// // | A | B | A | C | A |
/// assert_eq!(ref_a.range(&tree), Some(7..10));
/// // See, for example, that the right-hand side is later on:
/// assert_eq!(tree.get(13).range(), 12..14);
/// assert_eq!(tree.get(13).slice(), &Constant('A'));
///
/// // If we remove a range from the tree in a way that causes this value
/// // to join with either its left or right, it will follow those.
/// //
/// // Before:
/// assert_eq!(
///     tree.iter(..10).map(|e| (e.range(), e.slice())).collect::<Vec<_>>(),
///     [
///         (0..3, &Constant('A')),
///         (3..7, &Constant('B')),
///         (7..10, &Constant('A')),
///     ],
/// );
/// // Join with left by removing 'B':
/// _ = tree.remove(3..7);
/// assert_eq!(ref_a.range(&tree), Some(0..6));
/// // The state now:
/// assert_eq!(
///     tree.iter(..).map(|e| (e.range(), e.slice())).collect::<Vec<_>>(),
///     [
///         (0..6, &Constant('A')),
///         (6..8, &Constant('C')),
///         (8..10, &Constant('A')),
///     ],
/// );
/// // Join with right by removing 'C':
/// _ = tree.remove(6..8);
/// assert_eq!(ref_a.range(&tree), Some(0..8));
/// assert_eq!(tree.size(), 8);
///
/// // NOTE: if we remove the start of the value, this reference is no
/// // longer valid for this tree (because the value was split, and
/// // `ref_a` followed the left-hand side)
/// let rm = tree.remove(..3);
/// assert!(ref_a.range(&tree).is_none());
/// // ... but note that the value *is* present in what was removed:
/// let rm_tree = rm.into_tree();
/// assert_eq!(ref_a.range(&rm_tree), Some(0..3));
/// # }
/// ```
pub struct StableRef<I, S> {
    pub(super) handle: Cell<Option<node::HandleWeak<I, S, param::EnableRefs>>>,
}

impl<I, S> StableRef<I, S>
where
    I: Index,
    S: Slice<I>,
{
    /// Returns a [`SliceEntry`] for this stable reference, if it's still active as part of the
    /// [`RleTree`].
    ///
    /// There is no requirement that the `RleTree` be the original tree that produced this
    /// `StableRef`; you can, for example, `get` the entry for a stable ref in the new `RleTree`
    /// generated by removing a range.
    pub fn get<'t>(
        &self,
        tree: &'t RleTree<I, S, param::EnableRefs>,
    ) -> Option<SliceEntry<'t, I, S, param::EnableRefs>> {
        // note: If the tree has no root node, then this `StableRef` is definitely not part of the
        // tree, so we can exit early.
        let tree_root_ptr = tree.root.as_ref()?.handle.ptr();

        let this = self.handle.take()?.walk();
        let (this_root_ptr, range) = this.root();

        let entry_handle = match this_root_ptr == tree_root_ptr {
            false => None,
            true => {
                // SAFETY: `upgrade_ref` requires that the tree is not mutably borrowed for the
                // entire lifetime. This is guaranteed by (a) checking that the root at the top of
                // this node is the root of the tree passed by the caller, and (b) immutably
                // borrowing the tree passed by that caller for the lifetime of this handle.
                let h = unsafe { this.upgrade_ref() };
                Some(h)
            }
        };

        self.handle.set(Some(this.clone()));
        entry_handle.map(|slice| SliceEntry { range, slice })
    }

    /// Alias for `self.get().map(SliceEntry::range)`
    pub fn range(&self, tree: &RleTree<I, S, param::EnableRefs>) -> Option<Range<I>> {
        self.get(tree).map(|e| e.range())
    }

    /// Alias for `self.get().map(SliceEntry::size)`
    pub fn size(&self, tree: &RleTree<I, S, param::EnableRefs>) -> Option<I> {
        self.get(tree).map(|e| e.size())
    }

    /// Alias for `self.get().map(SliceEntry::slice)`
    pub fn slice<'t>(&self, tree: &'t RleTree<I, S, param::EnableRefs>) -> Option<&'t S> {
        self.get(tree).map(|e| e.slice())
    }
}
