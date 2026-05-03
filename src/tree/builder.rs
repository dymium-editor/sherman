#[cfg(test)]
use std::fmt::{self, Debug};
use std::num::NonZeroU16;

use crate::param::{self, RleTreeConfig, SupportsUpdate};
use crate::{Index, RleTree, Slice};

use super::Root;
use super::fix::{self, FixMode};
use super::node;

/// Helper for `O(n)` construction of an [`RleTree`]; internally used for `FromIterator`
///
/// A new builder can be created either by [`Builder::new`] or [`RleTree::builder`].
///
/// Elements are added to the end with [`push`], given their size & value.
///
/// The builder is converted into an [`RleTree`] with [`finish`].
///
/// # Algorithmic complexity
///
/// A series of `n` calls to [`push`], followed by [`finish`], is `O(n)`. Refer to the individual
/// methods for more detail.
///
/// [`push`]: Builder::push
/// [`finish`]: Builder::finish
///
/// # Example usage
///
/// **First:** Note that often it's not necessary to use this type directly! `Builder` is what's
/// used internally for [`RleTree`]'s implementation of [`FromIterator`], allowing e.g.
/// `collect()`ing into an [`RleTree`].
///
/// ```
/// use sherman::{Constant, RleTree};
///
/// let size_value_pairs = [
///     (1, Constant('A')),
///     (2, Constant('B')),
///     (3, Constant('C')),
///     (4, Constant('D')),
/// ];
///
/// let tree: RleTree<usize, Constant<char>> =
///     size_value_pairs.into_iter().collect();
///
/// assert_eq!(
///     tree.iter(..)
///         .map(|entry| (entry.range(), entry.slice()))
///         .collect::<Vec<_>>(),
///     [
///         (0..1, &Constant('A')),
///         (1..3, &Constant('B')),
///         (3..6, &Constant('C')),
///         (6..10, &Constant('D')),
///     ]
/// );
/// ```
///
/// But `Builder` is public so that it can be used manually for other patterns:
///
/// ```
/// use sherman::{Builder, Constant, RleTree};
///
/// let mut builder: Builder<usize, Constant<char>> = Builder::new();
///
/// builder.push(1, Constant('A'));
/// builder.push(2, Constant('B'));
/// builder.push(3, Constant('C'));
/// builder.push(4, Constant('D'));
///
/// let tree: RleTree<_, _> = builder.finish();
///
/// assert_eq!(
///     tree.iter(..)
///         .map(|entry| (entry.range(), entry.slice()))
///         .collect::<Vec<_>>(),
///     [
///         (0..1, &Constant('A')),
///         (1..3, &Constant('B')),
///         (3..6, &Constant('C')),
///         (6..10, &Constant('D')),
///     ]
/// );
/// ```
pub struct Builder<I, S, P: RleTreeConfig<I, S> = param::NoFeatures> {
    inner: Option<BuilderInner<I, S, P>>,
}

struct BuilderInner<I, S, P: RleTreeConfig<I, S>> {
    /// Root node of the tree we're building.
    ///
    /// NOTE: The tree rooted at this node violates the typical invariants in a very precise way.
    ///
    /// Given that `edge_ptr` points to the rightmost node in this tree, we have the following
    /// guarantees:
    ///
    /// 1. All nodes that do not have `edge_ptr` as a descendent will uphold the typical tree
    ///    invariants, and *additionally* will be perfectly balanced (recursively equal left and
    ///    right child heights)
    ///
    /// 2. All nodes that *do* have `edge_ptr` as a descendent:
    ///    a. Will exclude the size of their right-hand child in their `subtree_size`
    ///    b. May have their right-hand child arbitrarily less tall than their left-hand child
    ///    (however the right-hand child must not be taller).
    ///
    /// This allows amortized `O(1)` push and `O(log n)` fixing at the end by traversing back up
    /// the tree to fix 2(a) and 2(b).
    root: node::HandleUniqueOwned<I, S, P>,

    /// Pointer to the rightmost node in the tree rooted at `root`
    ///
    /// # Safety
    ///
    /// This pointer is effectively mutably borrowing `root`, and so is intended to be accessed as
    /// such with `HandleMut::from_ptr`. Accessing `root` (a) must only be done while any handle
    /// derived from `edge_ptr` has been dropped, and (b) invalidates `edge_ptr` such that it must
    /// be replaced.
    edge_ptr: node::Pointer<I, S, P>,
}

#[cfg(test)]
impl<I, S, P> Debug for BuilderInner<I, S, P>
where
    I: Debug,
    S: Debug,
    P: RleTreeConfig<I, S>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Note: This slightly violates the access invariants that we hold on BuilderInner's
        // fields. The debug impl is only for testing, so this is mostly ok.
        f.debug_struct("BuilderInner")
            .field("root", &self.root)
            .field("edge_ptr", &self.edge_ptr)
            .finish()
    }
}

impl<I, S, P: RleTreeConfig<I, S>> Default for Builder<I, S, P> {
    fn default() -> Self {
        Builder { inner: None }
    }
}

impl<I, S, P> Builder<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    /// Creates a new, empty [`RleTree`] builder
    pub fn new() -> Self {
        Builder { inner: None }
    }

    // Test-only helper for debugging
    #[cfg(test)]
    #[allow(unused)]
    fn root(&self) -> Option<node::HandleImmut<'_, I, S, P>> {
        self.inner.as_ref().map(|inner| inner.root.borrow())
    }

    /// Pushes a new slice `S` with its size `I` onto the end of the builder
    ///
    /// If the slice can be joined with its immediate neighbor (from the last call to `push`), then
    /// they will be joined together, as would normally be the case during insertion.
    ///
    /// # Algorithmic complexity
    ///
    /// This operation is `O(log n)` but amortized `O(1)`, where `n` is the number of unjoined
    /// values previously added with `push`.
    ///
    /// # Panics
    ///
    /// This method panics if `size` is not greater than zero — i.e. if `size <= I::ZERO`.
    #[inline]
    pub fn push(&mut self, size: I, slice: S) {
        let mut slice = Some(slice);

        if size <= I::ZERO {
            panic!("cannot push slice with non-positive size {size:?}");
        }

        match self.inner.as_mut() {
            None => self.inner = Some(BuilderInner::new(size, &mut slice)),
            Some(inner) => inner.push(&mut slice, size),
        }
    }

    /// Completes the builder and produces an `RleTree` from all the values that were previously
    /// [`push`]ed.
    ///
    /// # Algorithmic complexity
    ///
    /// This operation is `O(log n)`, where `n` is the number of unjoined values added with `push`.
    ///
    /// [`push`]: Self::push
    pub fn finish(self) -> RleTree<I, S, P>
    where
        P: SupportsUpdate<I, S>,
    {
        match self.inner {
            None => RleTree::new_empty(),
            Some(inner) => inner.finish(),
        }
    }
}

impl<I, S, P> FromIterator<(I, S)> for RleTree<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    fn from_iter<Iter>(iter: Iter) -> Self
    where
        Iter: IntoIterator<Item = (I, S)>,
    {
        let mut builder = Builder::new();
        iter.into_iter().for_each(|(i, s)| builder.push(i, s));
        builder.finish()
    }
}

impl<I, S, P> BuilderInner<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    fn new(size: I, slice: &mut Option<S>) -> Self {
        let root = node::HandleUniqueOwned::alloc_new(slice, size);

        BuilderInner { edge_ptr: root.ptr(), root }
    }

    fn push(&mut self, slice: &mut Option<S>, size: I) {
        // SAFETY: the BuilderInner invariants guarantee that `self.edge_ptr` points to a valid
        // node, and that we can create a mutable handle to it with this pointer. In return, we are
        // required to drop `edge` before accessing `self.root` (and reset `self.edge_ptr` after),
        // which we do at the very end of this method.
        let mut edge = unsafe { node::HandleMut::from_ptr(self.edge_ptr) };

        // Try to join with `edge`.
        let edge_value = edge.value_mut();
        S::try_join_into_lhs(edge_value, slice);
        match slice {
            // Couldn't join - continue below.
            Some(_) => (),
            // Successfully joined - we're basically done.
            None => {
                // Because `edge_ptr` will always be the rightmost node, and its ancestors don't
                // include its subtree size, it's sound to just add to the subtree size of the edge
                let new_subtree_size = edge.subtree_size().add_right(size);
                edge.set_subtree_size(new_subtree_size);
                return;
            }
        }

        // At this point, we know we'll have to add a new node. Let's allocate it up front.
        let mut new_node = node::HandleUniqueOwned::alloc_new(slice, size);

        // There are a few possible states for the current edge node:
        //
        // 1. It only has a left-hand child
        //    => We should insert `new_node` as the right-hand child
        // 2. It does not have any children
        //    => Its first ancestor `n` where `n`'s left-hand child is taller than its right-hand
        //       child should have the right-hand child replaced by `new_node`, and `n`'s former
        //       right-hand child set as `new_node`'s left-hand child.
        //    => If no such ancestor exists, then the root is fully saturated at its current
        //       height, and so we must replace the root with `new_node` and add the old root as
        //       its left-hand child.
        //
        // The edge node should not have a right-hand child, because otherwise it would not be the
        // rightmost node in the tree.
        debug_assert!(!edge.has_rhs());
        // Let's start with case (1).
        if edge.has_lhs() {
            // This is pretty straightforward: We just insert the node, and point `edge_ptr` at it,
            // now that it's the new rightmost node in the tree. We don't have to update the
            // parent's subtree size, because the expectation is that this is done on the upward
            // pass. We also don't have to correct their height, because by virtue of already
            // having a left-hand child, we know that adding a right-hand child won't change the
            // height of the parent!
            edge = edge.insert_into_rhs(new_node);
            self.edge_ptr = edge.ptr();
            return;
        }

        // Now for case (2). We should traverse up the tree until we find a parent that is
        // unbalanced where we can insert `new_node` as the new right-hand child.
        //
        // This traversal is why this operation is O(log n), but the average number of steps here
        // is also O(1), which is why the operation is amortized O(1).
        //
        // While we're traversing up the tree, we also need to correct each parent's subtree size.
        // The height does not need correcting, because we already guarantee that each right-hand
        // child is at most as tall as the left-hand child.

        let mut subtree_size = edge.subtree_size();

        while let Some((mut parent, _)) = edge.into_parent() {
            let rhs_height = parent.rhs_height();

            // If this parent's left-hand child is the same height as its right-hand child, keep
            // traversing up the tree.
            if parent.lhs_height() == rhs_height {
                // `edge` is guaranteed to have been the rightmost node, so as we're correcting each
                // ancestor's subtree size to include it, we can just `add_right`.
                subtree_size = parent.subtree_size().add_right(subtree_size);
                parent.set_subtree_size(subtree_size);
                edge = parent;
                continue;
            }

            // Otherwise: We guarantee the left-hand child is at least as tall as the right-hand
            // child, so "not equal" means it's taller. We should replace `parent`'s right-hand
            // child with `new_node`, and make the parent's old right-hand child (the subtree
            // containing `edge`) the left-hand child of `new_node`.
            let old_rhs = parent
                .take_rhs()
                .expect("recursing upwards from `edge` should always have a parent with RHS");

            let new_node_height =
                NonZeroU16::new(1 + rhs_height).expect("node height should not overflow u16");
            let new_node_subtree_size = subtree_size.add_right(new_node.subtree_size());

            new_node.borrow_mut().insert_lhs(old_rhs);
            new_node.set_height(new_node_height);
            new_node.set_subtree_size(new_node_subtree_size);

            edge = parent.insert_into_rhs(new_node);
            self.edge_ptr = edge.ptr();
            return;
        }

        // Couldn't find any ancestor node with mismatched left/right child heights, so the root
        // node is fully saturated at its current height (i.e. each node is perfectly symmetric).
        // To extend the tree, we must place the current root node as the child of `new_node`, and
        // make `new_node` the new root.

        let new_node_height = NonZeroU16::new(1 + self.root.height().get())
            .expect("node height should not overflow u16");
        let new_node_subtree_size = self.root.subtree_size().add_right(new_node.subtree_size());

        let old_root = std::mem::replace(&mut self.root, new_node);
        self.root.borrow_mut().insert_lhs(old_root.erase());
        self.root.set_height(new_node_height);
        self.root.set_subtree_size(new_node_subtree_size);
        self.edge_ptr = self.root.ptr();
    }

    fn finish(self) -> RleTree<I, S, P>
    where
        P: SupportsUpdate<I, S>,
    {
        if self.edge_ptr != self.root.ptr() {
            // SAFETY: the BuilderInner invariants guarantee that `self.edge_ptr` points to a valid
            // node, and that we can create a mutable handle to it with this pointer. In return, we
            // are required to drop `edge` before accessing `self.root` (and reset `self.edge_ptr`
            // before using it again), which we guarantee by (a) dropping `edge` when the while
            // loop ends, and (b) destroying `self` so that `edge_ptr` is not accessed again.
            let mut edge = unsafe { node::HandleMut::from_ptr(self.edge_ptr) };

            let mut subtree_size = edge.subtree_size();

            if edge.has_parent() {
                edge = fix::fix_mut(edge, FixMode::Unbounded);
            }

            while let Some((mut parent, _)) = edge.into_parent() {
                // `edge` is guaranteed to have been the rightmost node, so as we're correcting
                // each ancestor's subtree size to include it, we can just `add_right`.
                subtree_size = parent.subtree_size().add_right(subtree_size);
                parent.set_subtree_size(subtree_size);

                edge = parent;

                if edge.has_parent() {
                    edge = fix::fix_mut(edge, FixMode::Unbounded);
                }
            }
        }

        let handle = fix::fix_unique_owned(self.root, FixMode::Unbounded).erase();
        RleTree { root: Some(Root { handle }) }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::ops::Range;

    use crate::param::{self, RleTreeConfig};
    use crate::{Constant, Index, RleTree, Slice};

    #[test]
    fn test_basic() {
        let mut builder = <RleTree<usize, Constant<char>>>::builder();
        builder.push(1, Constant('A'));
        builder.push(2, Constant('B'));
        builder.push(3, Constant('C'));
        builder.push(2, Constant('D'));
        builder.push(2, Constant('D')); // <- join
        builder.push(5, Constant('E'));

        let tree = builder.finish();
        tree.validate_balance();

        assert_eq!(
            tree.iter(..).map(|e| (e.range(), e.slice())).collect::<Vec<_>>(),
            [
                (0..1, &Constant('A')),
                (1..3, &Constant('B')),
                (3..6, &Constant('C')),
                (6..10, &Constant('D')),
                (10..15, &Constant('E')),
            ],
        );
    }

    fn run_exhaustive<I, S, P>(values: &[(I, S)])
    where
        I: Index,
        S: Slice<I> + Copy + PartialEq + Debug,
        P: RleTreeConfig<I, S> + param::SupportsUpdate<I, S>,
    {
        let mut expected_ranges: Vec<(Range<I>, &S)> = Vec::new();
        for &(i, ref s) in values {
            let start = expected_ranges.last().map(|(r, _)| r.end).unwrap_or(I::ZERO);
            let end = start.add_right(i);
            expected_ranges.push((start..end, s));
        }

        for len in 0..values.len() {
            let mut builder = <RleTree<I, S, P>>::builder();
            for &(i, s) in &values[..len] {
                builder.push(i, s);
            }

            let tree = builder.finish();

            tree.validate_balance();

            // Confirm that we have an optimally balanced tree:
            if let Some(r) = tree.root() {
                let expected_height = len.ilog2() + 1;
                assert_eq!(expected_height, r.handle.height().get() as u32);
            }

            assert_eq!(
                tree.iter(..).map(|e| (e.range(), e.slice())).collect::<Vec<_>>(),
                &expected_ranges[..len]
            );
        }
    }

    fn generate_values() -> Vec<(usize, Constant<char>)> {
        // Use a smaller threshold with miri, to avoid taking unbearably long.
        #[cfg(miri)]
        let max = 20_usize;
        #[cfg(not(miri))]
        let max = 500_usize;
        (0..max)
            .map(|i| {
                let ch = (b'A' + (i % 26) as u8) as char;
                ((i % 20) + 1, Constant(ch))
            })
            .collect()
    }

    #[test]
    fn test_exhaustive_no_features() {
        run_exhaustive::<_, _, param::NoFeatures>(&generate_values());
    }

    #[test]
    fn test_exhaustive_enable_cow() {
        run_exhaustive::<_, _, param::EnableCow>(&generate_values());
    }

    #[test]
    fn test_exhaustive_enable_refs() {
        run_exhaustive::<_, _, param::EnableRefs>(&generate_values());
    }
}
