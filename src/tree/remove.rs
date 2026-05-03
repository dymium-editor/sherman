//! Implementation of [`RleTree::remove`]

use std::ops::{Bound, Range};

use crate::param::{self, RleTreeConfig, SupportsUpdate};
use crate::{Index, RleTree, Slice};

use super::drain::Drain;
use super::fix::{self, FixMode};
use super::rc::Redirect as _;
use super::{
    DownwardInsertState, InsertionValue, Root, SearchResult, Side, UpwardUpdateState, node,
};

/// Values removed from an [`RleTree`]
///
/// Generally this type exists so that we don't need to create a full `RleTree` for the smallest
/// removals.
///
/// There are a few ways to use this:
///
/// * Get the [`size`](Removed::size) or original [`range`](Removed::range) that was removed
/// * Extract a single value with [`try_into_value()`](Removed::try_into_value)
/// * Turn the removed values into a full `RleTree` with [`into_tree`](Removed::into_tree)
/// * Iterate over the values with `into_iter` (into a [`Drain`])
pub struct Removed<I, S, P: RleTreeConfig<I, S> = param::NoFeatures> {
    pub(super) start: I,
    pub(super) kind: RemovedKind<I, S, P>,
}

// SAFETY: This is the same condition as `RleTree`; see crate::param for more.
unsafe impl<I, S, P> Send for Removed<I, S, P> where
    P: RleTreeConfig<I, S> + param::RleTreeIsSend<I, S>
{
}
// SAFETY: This is the same condition as `RleTree`; see crate::param for more.
unsafe impl<I, S, P> Sync for Removed<I, S, P> where
    P: RleTreeConfig<I, S> + param::RleTreeIsSync<I, S>
{
}

#[cfg_attr(test, derive(Debug))]
pub(super) enum RemovedKind<I, S, P: RleTreeConfig<I, S>> {
    Tree(node::HandleOwned<I, S, P>),
    Slice { size: I, slice: S },
    Empty,
}

impl<I, S, P> Removed<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    pub(super) fn empty(idx: I) -> Self {
        Removed { start: idx, kind: RemovedKind::Empty }
    }

    pub(super) fn from_tree(mut tree: RleTree<I, S, P>) -> Self {
        Removed {
            start: I::ZERO,
            kind: match tree.root.take() {
                None => RemovedKind::Empty,
                Some(root) => RemovedKind::Tree(root.handle),
            },
        }
    }

    /// Returns the total size of the range of values that were removed
    pub fn size(&self) -> I {
        match &self.kind {
            RemovedKind::Empty => I::ZERO,
            RemovedKind::Slice { size, .. } => *size,
            RemovedKind::Tree(handle) => handle.subtree_size(),
        }
    }

    /// Returns the range of the original [`RleTree`] that was removed
    pub fn range(&self) -> Range<I> {
        let start = self.start;
        let end = self.start.add_right(self.size());
        start..end
    }

    /// If this `Removed` represents a single value, returns that value and its size. Otherwise,
    /// returns `Err(self)`.
    pub fn try_into_value(self) -> Result<(I, S), Self> {
        match self.kind {
            RemovedKind::Slice { size, slice } => Ok((size, slice)),
            _ => Err(self),
        }
    }

    /// Turns the removed values into a proper [`RleTree`]
    ///
    /// This is an `O(1)` operation. If there were several values removed, they are already
    /// internally stored as a tree.
    pub fn into_tree(self) -> RleTree<I, S, P> {
        match self.kind {
            RemovedKind::Empty => RleTree::new_empty(),
            RemovedKind::Slice { size, slice } => RleTree::new(size, slice),
            RemovedKind::Tree(handle) => RleTree { root: Some(Root { handle }) },
        }
    }
}

impl<I, S, P> IntoIterator for Removed<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    type Item = (Range<I>, S);
    type IntoIter = Drain<I, S, P>;

    fn into_iter(self) -> Drain<I, S, P> {
        Drain::new(self)
    }
}

pub(super) fn check_bounds<I, S, P>(
    tree: &RleTree<I, S, P>,
    verb: &str,
    start_bound: Bound<&I>,
    end_bound: Bound<&I>,
) -> Range<I>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    let start = match start_bound {
        Bound::Excluded(_) => panic!("cannot {verb} with exclusive start bound"),
        Bound::Included(idx) if *idx < I::ZERO => {
            panic!("start bound {idx:?} out of bounds, less than zero");
        }

        Bound::Unbounded => I::ZERO,
        Bound::Included(&idx) => idx,
    };

    let size = tree.size();
    let end = match end_bound {
        Bound::Included(_) => panic!("cannot {verb} with inclusive end bound"),
        Bound::Excluded(idx) if *idx > size => {
            panic!("end bound {idx:?} out of bounds, greater than size {size:?}")
        }

        Bound::Unbounded => size,
        Bound::Excluded(&idx) => idx,
    };

    // Check for invalid ranges:
    if end < start {
        panic!("bad range for {verb}: end bound {end:?} less than start {start:?}");
    }

    start..end
}

pub(super) fn remove<I, S, P>(
    tree: &mut RleTree<I, S, P>,
    range: Range<I>,
    replacement: Option<&mut Option<S>>,
) -> Removed<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    // Special case: allow empty ranges, but don't do anything with them
    if range.start == range.end {
        return Removed::empty(range.start);
    }

    // Special case: If the range is the entire tree, handle that separately. This way, we can
    // assume later that we'll have at least one node left over after removal.
    if range.start == I::ZERO && range.end == tree.size() {
        let new = match replacement {
            None => RleTree::new_empty(),
            Some(r) => RleTree::new_from_opt(range.end, r),
        };
        let old = std::mem::replace(tree, new);
        return Removed::from_tree(old);
    }

    let root = match tree.root.take() {
        Some(r) => r,
        None => {
            crate::panic_internal_error_or_bad_index::<I>("got non-empty range but empty tree root")
        }
    };

    let mut removed_kind = None;
    let new_root = run_removal(root, range.start, range.end, replacement, &mut removed_kind);
    tree.root = Some(new_root);

    Removed {
        start: range.start,
        kind: removed_kind.expect("`removed_kind` should be `Some(_)`"),
    }
}

fn run_removal<I, S, P>(
    tree: Root<I, S, P>,
    start: I,
    end: I,
    replacement: Option<&mut Option<S>>,
    removed: &mut Option<RemovedKind<I, S, P>>,
) -> Root<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    let replacement = replacement.map(|slice| (end.sub_left(start), slice));

    // First, find the node where the removal range is fully contained *within* its subtree.
    // In other words: we want to find the node whose subtree start/end bounds most narrowly
    // contain fully the `start` end `end` of the removal range (acknowledging that this may not be
    // satisfied if they're at the start/end of the tree itself).
    let mut tree_root = tree.handle.into_unique();
    let mut removal_root = tree_root.borrow_mut();
    let mut removal_root_offset = I::ZERO;

    let (search_start, search_end) = loop {
        let search_start =
            super::search_step(removal_root.borrow(), start.sub_left(removal_root_offset));
        let search_end =
            super::search_step(removal_root.borrow(), end.sub_left(removal_root_offset));

        match (search_start, search_end) {
            // Both ends of the removal are contained by the left-hand child. Recurse into that.
            (SearchResult::Lhs { .. }, SearchResult::Lhs { .. }) => {
                removal_root = match removal_root.into_lhs() {
                    Ok(n) => n,
                    Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                        "`SearchResult::Lhs` implies the left-hand child should exist",
                    ),
                };
            }
            // Both ends of the removal are contained by right-hand child. Recurse into that.
            (SearchResult::Rhs { .. }, SearchResult::Rhs { .. }) => {
                let upper_subtree_size = removal_root.subtree_size();
                removal_root = match removal_root.into_rhs() {
                    Ok(n) => n,
                    Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                        "`SearchResult::Rhs` implies the left-hand child should exist",
                    ),
                };
                let lower_subtree_size = removal_root.subtree_size();
                removal_root_offset =
                    removal_root_offset.add_right(upper_subtree_size.sub_right(lower_subtree_size));
            }
            // Otherwise, no child of this node could fully contain both the start and end
            // boundary, so we've found our immediate target.
            (search_start, search_end) => break (search_start, search_end),
        }
    };

    // Temporarily remove the node from its parent, and then run the removal on that owned subtree.
    if let Some((mut parent, side)) = removal_root.into_parent() {
        let mut child = match side {
            Side::Lhs => parent
                .take_lhs()
                .expect("child is marked as LHS so parent should have LHS child")
                .into_unique(),
            Side::Rhs => parent
                .take_rhs()
                .expect("child is marked as LHS so parent should have LHS child")
                .into_unique(),
        };

        let mut up_state;
        (child, up_state) =
            run_removal_within_subtree(child, search_start, search_end, replacement, removed);

        // Put the child back into its parent
        let mut node = match side {
            Side::Lhs => parent.insert_into_lhs(child),
            Side::Rhs => parent.insert_into_rhs(child),
        };
        // ... and then traverse back up the tree to fix the changes
        while node.has_parent() {
            node = fix::fix_mut(node, FixMode::Unbounded);
            (node, up_state) = up_state.step(node, None);
        }
    } else {
        (tree_root, _) =
            run_removal_within_subtree(tree_root, search_start, search_end, replacement, removed);
    }

    // perform a final fix on the root, if needed:
    tree_root = fix::fix_unique_owned(tree_root, FixMode::Unbounded);

    Root { handle: tree_root.erase() }
}

/// Performs a removal whose bounds are just barely contained by the subtree rooted at the node
fn run_removal_within_subtree<I, S, P>(
    mut root: node::HandleUniqueOwned<I, S, P>,
    start: SearchResult<I>,
    mut end: SearchResult<I>,
    replacement: Option<(I, &mut Option<S>)>,
    removed: &mut Option<RemovedKind<I, S, P>>,
) -> (node::HandleUniqueOwned<I, S, P>, UpwardUpdateState<I>)
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    let root_side = match (&start, &end) {
        // Special case: Both ends of the removal are contained by this root node!
        // We should handle that separately.
        (
            SearchResult::Value { range, offset_in_range: start_offset },
            SearchResult::Value { offset_in_range: end_offset, .. },
        ) => {
            if let Some((_, repl_value)) = replacement {
                return run_replacement_within_node(
                    root,
                    range.clone(),
                    *start_offset,
                    *end_offset,
                    repl_value,
                    removed,
                );
            } else {
                return run_removal_within_node(
                    root,
                    range.clone(),
                    *start_offset,
                    *end_offset,
                    removed,
                );
            }
        }

        // In some cases, we must treat the root node as belonging to a particular side.
        //
        // 1. LHS edge is inside the root node
        (SearchResult::Value { .. }, SearchResult::RhsEdge | SearchResult::Rhs { .. }) => Side::Lhs,
        // 2. LHS edge is exactly at the end of the root node's value
        (SearchResult::RhsEdge, SearchResult::Rhs { .. }) => Side::Lhs,
        // 3. RHS edge is inside the root node
        (SearchResult::Lhs { .. } | SearchResult::LhsEdge, SearchResult::Value { .. }) => Side::Rhs,
        // 4. RHS edge is exactly at the start of the root node's value
        (SearchResult::Lhs { .. }, SearchResult::LhsEdge) => Side::Rhs,
        // 5. If neither side touches the root node, we can just arbitrarily pick a side to get the
        //    root node for further processing.
        (
            SearchResult::Lhs { .. } | SearchResult::LhsEdge,
            SearchResult::RhsEdge | SearchResult::Rhs { .. },
        ) => Side::Lhs,
        // And if none of the above cases match, then we have a bad set of `SearchResult`s
        combo => crate::panic_internal_error_or_bad_index::<I>(&format!(
            "got invalid `SearchResult`s while generating removal: {combo:?}"
        )),
    };

    let root_size = root.subtree_size();

    let (lhs_tree, rhs_tree) = match root_side {
        Side::Lhs => {
            let rhs = root.take_rhs();
            if let Some(n) = rhs.as_ref() {
                root.set_subtree_size(root_size.sub_right(n.subtree_size()));
            }
            (Some(root.erase()), rhs)
        }
        Side::Rhs => {
            let lhs = root.take_lhs();
            let lhs_size = lhs.as_ref().map(|n| n.subtree_size()).unwrap_or(I::ZERO);
            root.set_subtree_size(root_size.sub_left(lhs_size));

            // Update `end` to account for removing `lhs`
            end = match end {
                SearchResult::Value { range, offset_in_range } => SearchResult::Value {
                    range: range.start.sub_left(lhs_size)..range.end.sub_left(lhs_size),
                    offset_in_range,
                },
                SearchResult::Lhs { .. } => unreachable!(),
                s @ (SearchResult::LhsEdge | SearchResult::RhsEdge | SearchResult::Rhs { .. }) => s,
            };

            (lhs, Some(root.erase()))
        }
    };

    let lhs_split = split_removal_lhs(lhs_tree, root_side, start);
    let rhs_split = split_removal_rhs(rhs_tree, root_side, end);

    let final_tree = match replacement {
        Some(pair) => join_trees_with_middle(lhs_split.lhs, pair, rhs_split.rhs),
        None => join_trees(lhs_split.lhs, rhs_split.rhs, true).into_unique(),
    };
    let final_removed = join_trees(lhs_split.rhs, rhs_split.lhs, false);
    *removed = Some(RemovedKind::Tree(final_removed));

    let up_state = UpwardUpdateState { old_size: root_size };

    (final_tree, up_state)
}

/// Performs a replacement whose bounds are contained by the values of a single node
/// (special case of `run_removal_within_subtree`)
fn run_replacement_within_node<I, S, P>(
    mut node: node::HandleUniqueOwned<I, S, P>,
    value_range: Range<I>,
    start_offset: I,
    end_offset: I,
    replacement: &mut Option<S>,
    removed: &mut Option<RemovedKind<I, S, P>>,
) -> (node::HandleUniqueOwned<I, S, P>, UpwardUpdateState<I>)
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    let mut node_mut = node.borrow_mut();

    let value_size = end_offset.sub_left(start_offset);
    let range_size = value_range.end.sub_left(value_range.start);
    let rhs_size = range_size.sub_left(end_offset);
    let subtree_size = node_mut.subtree_size();

    let lhs = node_mut.value_mut();
    let mut removed_value = None;
    let mut rhs = None;
    S::split_at_mut(lhs, start_offset, &mut removed_value);
    S::split_at_mut(&mut removed_value, value_size, &mut rhs);

    // Try merging the replacement with LHS and/or RHS
    S::try_join_into_lhs(lhs, replacement);
    match replacement {
        None => {
            // Joined with LHS. Try with RHS now.
            S::try_join_into_lhs(lhs, &mut rhs);
            match rhs {
                // Joined with both RHS & LHS - we're actually all done, because the node is still
                // the same size and we didn't generate anything new.
                None => drop(node_mut),
                // Joined with LHS but not RHS. We'll need to re-insert RHS.
                Some(_) => {
                    super::run_insert(
                        node_mut,
                        Some(subtree_size),
                        true,
                        DownwardInsertState {
                            target: value_range.end,
                            fst_value: InsertionValue { slice: &mut rhs, size: rhs_size },
                            snd_value: None,
                            allow_joining: false,
                            already_split_once: true,
                        },
                    );
                }
            }
        }
        Some(_) => {
            // Couldn't join with LHS. Try with RHS.
            S::try_join_into_lhs(replacement, &mut rhs);
            match rhs {
                // Joined with RHS but not LHS. We need to (re-)insert the joined RHS.
                None => {
                    super::run_insert(
                        node_mut,
                        Some(subtree_size),
                        true,
                        DownwardInsertState {
                            target: value_range.end,
                            fst_value: InsertionValue {
                                slice: replacement,
                                size: value_size.add_right(rhs_size),
                            },
                            snd_value: None,
                            allow_joining: false,
                            already_split_once: true,
                        },
                    );
                }
                // Couldn't join with either LHS or RHS. We need to insert replacement and RHS.
                Some(_) => {
                    super::run_insert(
                        node_mut,
                        Some(subtree_size),
                        true,
                        DownwardInsertState {
                            target: value_range.end,
                            fst_value: InsertionValue { slice: replacement, size: value_size },
                            snd_value: Some(InsertionValue { slice: &mut rhs, size: rhs_size }),
                            allow_joining: false,
                            already_split_once: true,
                        },
                    );
                }
            }
        }
    }

    let up_state = UpwardUpdateState { old_size: subtree_size };
    *removed = Some(RemovedKind::Slice {
        size: value_size,
        slice: removed_value.expect("`removed_value` should be `Some(_)`"),
    });
    (node, up_state)
}

/// Performs a removal whose bounds are contained by the values of a single node
/// (special case of `run_removal_within_subtree`)
fn run_removal_within_node<I, S, P>(
    mut node: node::HandleUniqueOwned<I, S, P>,
    value_range: Range<I>,
    start_offset: I,
    end_offset: I,
    removed: &mut Option<RemovedKind<I, S, P>>,
) -> (node::HandleUniqueOwned<I, S, P>, UpwardUpdateState<I>)
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    let mut node_mut = node.borrow_mut();

    let lhs = node_mut.value_mut();
    let mut removed_value = None;
    let mut rhs = None;
    S::split_at_mut(lhs, start_offset, &mut removed_value);
    let value_size = end_offset.sub_left(start_offset);
    S::split_at_mut(&mut removed_value, value_size, &mut rhs);

    // Try merging `lhs` and `rhs`. If successful, we'll have nothing left to do but recurse back
    // up the tree.
    S::try_join_into_lhs(lhs, &mut rhs);

    drop(node_mut); // Done accessing the value directly; need to release the borrow.

    // Before removal:
    //
    //              |----|-----|----|
    //   |----------|               |----------|
    //     node lhs                   node lhs
    //
    //   ---------->|
    //   value_range.start
    //              ---->|
    //              start_offset
    //              ---------->|
    //              end_offset
    //                         ---->|
    //              value_range.end.sub_left(value_range.start).sub_left(end_offset)
    //   -------------------------->|
    //   value_range.end
    //                              ---------->|
    //                      subtree_size.sub_left(value_range.end)
    //
    //
    // After removal:
    //
    //              |----/----|
    //   |----------|         |----------|
    //     node lhs             node lhs
    //
    //   ---------->|
    //   value_range.start
    //              ---->|
    //              start_offset
    //                   ---->|
    //              value_range.end.sub_left(end_offset)
    //
    let old_subtree_size = node.subtree_size();
    let lhs_subtree_size = value_range.start;
    let rhs_subtree_size = old_subtree_size.sub_left(value_range.end);
    let value_lhs_size = start_offset;
    let value_rhs_size = value_range.end.sub_left(value_range.start).sub_left(end_offset);
    let new_subtree_size = lhs_subtree_size
        .add_right(value_lhs_size)
        .add_right(value_rhs_size)
        .add_right(rhs_subtree_size);
    node.set_subtree_size(new_subtree_size);

    if rhs.is_some() {
        super::run_insert(
            node.borrow_mut(),
            Some(new_subtree_size),
            true,
            DownwardInsertState {
                target: new_subtree_size.sub_right(rhs_subtree_size),
                fst_value: InsertionValue { slice: &mut rhs, size: value_rhs_size },
                snd_value: None,
                allow_joining: false,
                already_split_once: true,
            },
        );
    }

    let up_state = UpwardUpdateState { old_size: old_subtree_size };
    *removed = Some(RemovedKind::Slice {
        size: value_size,
        slice: removed_value.expect("`removed_value` should be `Some(_)`"),
    });

    (node, up_state)
}

fn split_removal_lhs<I, S, P>(
    node: Option<node::HandleOwned<I, S, P>>,
    root_side: Side,
    search: SearchResult<I>,
) -> Split<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    let Some(mut node) = node else {
        return Split { lhs: None, rhs: None };
    };

    let target = match search {
        SearchResult::Lhs { offset } => offset,
        SearchResult::LhsEdge => match root_side {
            Side::Lhs => node.value_range().start,
            Side::Rhs => node.subtree_size(),
        },
        SearchResult::Value { offset_in_range, range } => match root_side {
            Side::Lhs => range.start.add_right(offset_in_range),
            Side::Rhs => panic!(
                "internal error: `split_removal_lhs` got `SearchResult::Value` but `root_side = Rhs`"
            ),
        },
        SearchResult::RhsEdge => {
            assert!(root_side == Side::Lhs);
            assert!(node.borrow().into_rhs().is_none());

            node = fix::fix_owned(node, FixMode::Unbounded);
            return Split { lhs: Some(node), rhs: None };
        }
        s @ SearchResult::Rhs { .. } => {
            panic!("internal error: bad `SearchResult` for `split_removal_lhs`: {s:?}")
        }
    };

    split_tree(node.into_unique(), target)
}

fn split_removal_rhs<I, S, P>(
    node: Option<node::HandleOwned<I, S, P>>,
    root_side: Side,
    search: SearchResult<I>,
) -> Split<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    let Some(mut node) = node else {
        return Split { lhs: None, rhs: None };
    };

    let base = match root_side {
        Side::Lhs => I::ZERO,
        Side::Rhs => node.value_range().end,
    };

    let target = match search {
        SearchResult::Rhs { offset } => base.add_right(offset),
        SearchResult::RhsEdge => base,
        SearchResult::Value { offset_in_range, range } => match root_side {
            Side::Rhs => range.start.add_right(offset_in_range),
            Side::Lhs => panic!(
                "internal error: `split_removal_rhs` got `SearchResult::Value` but `root_side = Lhs`"
            ),
        },
        SearchResult::LhsEdge => {
            assert!(root_side == Side::Rhs);
            assert!(node.borrow().into_lhs().is_none());

            node = fix::fix_owned(node, FixMode::Unbounded);
            return Split { lhs: None, rhs: Some(node) };
        }
        s @ SearchResult::Lhs { .. } => {
            panic!("internal error: bad `SearchResult` for `split_removal_rhs`: {s:?}")
        }
    };

    split_tree(node.into_unique(), target)
}

#[cfg_attr(test, derive(Debug))]
struct Split<I, S, P: RleTreeConfig<I, S>> {
    lhs: Option<node::HandleOwned<I, S, P>>,
    rhs: Option<node::HandleOwned<I, S, P>>,
}

fn split_tree<I, S, P>(mut root: node::HandleUniqueOwned<I, S, P>, mut target: I) -> Split<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    // Do this in two steps:
    // 1. Traverse down the tree until we find the split point
    // 2. Traverse back up the tree, constructing the left- and right-hand trees on either side of
    //    the cut.
    let mut node = root.borrow_mut();

    let (mut node, side) = loop {
        // Boundary conditions: if we're at either edge of the subtree rooted at this node, then
        // the entire subtree will end up on one side of the split.
        if target == I::ZERO {
            break (node, Ok(Side::Rhs));
        } else if target == node.subtree_size() {
            break (node, Ok(Side::Lhs));
        }

        match super::search_step(node.borrow(), target) {
            SearchResult::Lhs { offset } => {
                match node.into_lhs() {
                    Ok(n) => node = n,
                    Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                        "`SearchResult::Lhs` implies the left-hand child should exist",
                    ),
                }
                target = offset;
            }
            SearchResult::LhsEdge => {
                match node.into_lhs() {
                    Ok(n) => node = n,
                    Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                        "`SearchResult::LhsEdge` implies the left-hand child should exist when `target != I::ZERO`",
                    ),
                }
                target = node.subtree_size();
            }
            SearchResult::Value { range, offset_in_range } => {
                // This node is not cleanly on one side or another because the split runs through
                // the node's value. We'll have to handle it separately when constructing the
                // initial upward state.
                break (node, Err((range, offset_in_range)));
            }
            SearchResult::RhsEdge => {
                match node.into_rhs() {
                    Ok(n) => node = n,
                    Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                        "`SearchResult::RhsEdge` implies the right-hand child should exist when `target != subtree_size`",
                    ),
                }
                target = I::ZERO;
            }
            SearchResult::Rhs { offset } => {
                match node.into_rhs() {
                    Ok(n) => node = n,
                    Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                        "`SearchResult::Rhs` implies the right-hand child should exist",
                    ),
                }
                target = offset;
            }
        }
    };

    // Create the initial state for the upward traversal, where we'll rebuild each side.
    let (mut parent, mut last_side, mut old_size, mut lhs_tree, mut rhs_tree) = match side {
        // The subtree rooted at this node cleanly belongs to one of the sides
        Ok(s) => {
            let old_size = node.subtree_size();
            let p = node.into_parent();
            let lhs_tree: Option<node::HandleUniqueOwned<I, S, P>> = None;
            let rhs_tree: Option<node::HandleUniqueOwned<I, S, P>> = None;
            (p, s, old_size, lhs_tree, rhs_tree)
        }
        Err((range, offset_in_range)) => {
            // Split this node's value
            let rhs_value_size = range.end.sub_left(range.start).sub_left(offset_in_range);
            let lhs_value = node.value_mut();
            let mut rhs_value = None;
            S::split_at_mut(lhs_value, offset_in_range, &mut rhs_value);

            let rhs_subtree = node.take_rhs();
            let rhs_subtree_size =
                rhs_subtree.as_ref().map(|n| n.subtree_size()).unwrap_or(I::ZERO);

            let old_subtree_size = node.subtree_size();
            node.set_subtree_size(
                old_subtree_size.sub_right(rhs_subtree_size).sub_right(rhs_value_size),
            );

            let mut rhs_node = node::HandleUniqueOwned::alloc_new(&mut rhs_value, rhs_value_size);
            if let Some(n) = rhs_subtree {
                rhs_node.borrow_mut().insert_rhs(n);
                rhs_node.set_subtree_size(rhs_value_size.add_right(rhs_subtree_size));
                rhs_node = fix::fix_unique_owned(rhs_node, FixMode::Unbounded);
            }

            if node.has_parent() {
                node = fix::fix_mut(node, FixMode::Unbounded);
            }

            let p = node.into_parent();
            (p, Side::Lhs, old_subtree_size, None, Some(rhs_node))
        }
    };

    while let Some((mut parent_node, parent_side)) = parent {
        let parent_old_size = parent_node.subtree_size();

        // As we traverse back up the tree:
        // One of the invariants that a balanced binary search tree gives us is that, as we
        // traverse the tree upwards from a node, any time we find we were part of a node's
        // left-hand subtree, that node's position will be further to the right than any node we've
        // seen so far (and vice versa, w.r.t. right-hand subtree & further to the left).
        //
        // This means that when we find the node we're traversing upwards *from* was in its
        // parent's left-hand subtree, the parent MUST be on the right-hand side of the split.
        //
        // It also means that - because we're continually removing each node from its parent - if
        // the node we just came *from* was on the left-hand side of the split, it should have an
        // empty right-hand child where we can place the existing left-hand split.
        let mut child = match parent_side {
            Side::Lhs => {
                parent_node.set_subtree_size(parent_old_size.sub_left(old_size));
                parent_node
                    .take_lhs()
                    .expect("child is marked as LHS so parent should have LHS child")
                    .into_unique()
            }
            Side::Rhs => {
                parent_node.set_subtree_size(parent_old_size.sub_right(old_size));
                parent_node
                    .take_rhs()
                    .expect("child is marked as RHS so parent should have RHS child")
                    .into_unique()
            }
        };

        match last_side {
            Side::Lhs => {
                if let Some(subtree) = lhs_tree {
                    // The child should be missing a right-hand child where we can insert the
                    // existing left-hand split.
                    let old_subtree_size = subtree.subtree_size();
                    let new_subtree_size = child.subtree_size().add_right(old_subtree_size);
                    child.borrow_mut().insert_rhs(subtree.erase());
                    child.set_subtree_size(new_subtree_size);
                }

                child = fix::fix_unique_owned(child, FixMode::Unbounded);
                lhs_tree = Some(child);
            }
            Side::Rhs => {
                if let Some(subtree) = rhs_tree {
                    // The child should be missing a left-hand child where we can insert the
                    // existing right-hand split.
                    let old_subtree_size = subtree.subtree_size();
                    let new_subtree_size = child.subtree_size().add_left(old_subtree_size);
                    child.borrow_mut().insert_lhs(subtree.erase());
                    child.set_subtree_size(new_subtree_size);
                }

                child = fix::fix_unique_owned(child, FixMode::Unbounded);
                rhs_tree = Some(child);
            }
        }

        // Set last_side based on the position of the parent.
        // Remember: this is flipped, because the split point being in the LHS child of the parent
        // means that the parent is on the right-hand side of the split (and vice versa).
        last_side = match parent_side {
            Side::Lhs => Side::Rhs,
            Side::Rhs => Side::Lhs,
        };

        old_size = parent_old_size;
        parent = parent_node.into_parent();
    }

    // We traversed all the way back to the root of the tree. At this point, we need to
    // appropriately merge the root into one of the sides of the split.
    drop(parent);

    match last_side {
        Side::Lhs => {
            // Root node is on the left-hand side of the split. It should be missing a right-hand
            // child that we can insert the existing LHS split tree into.
            if let Some(subtree) = lhs_tree {
                let old_subtree_size = subtree.subtree_size();
                let new_subtree_size = root.subtree_size().add_right(old_subtree_size);
                root.borrow_mut().insert_rhs(subtree.erase());
                root.set_subtree_size(new_subtree_size);
            }

            root = fix::fix_unique_owned(root, FixMode::Unbounded);
            lhs_tree = Some(root);
        }
        Side::Rhs => {
            // Root node is on the right-hand side of the split. It should be missing a left-hand
            // child that we can insert the existing RHS split tree into.
            if let Some(subtree) = rhs_tree {
                let old_subtree_size = subtree.subtree_size();
                let new_subtree_size = root.subtree_size().add_left(old_subtree_size);
                root.borrow_mut().insert_lhs(subtree.erase());
                root.set_subtree_size(new_subtree_size);
            }

            root = fix::fix_unique_owned(root, FixMode::Unbounded);
            rhs_tree = Some(root);
        }
    }

    Split {
        lhs: lhs_tree.map(|n| n.erase()),
        rhs: rhs_tree.map(|n| n.erase()),
    }
}

fn join_trees_with_middle<I, S, P>(
    mut lhs: Option<node::HandleOwned<I, S, P>>,
    (mid_size, mid_slice): (I, &mut Option<S>),
    mut rhs: Option<node::HandleOwned<I, S, P>>,
) -> node::HandleUniqueOwned<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    // Try joining with LHS
    if let Some(lhs_node) = lhs {
        let mut lhs_node = lhs_node.into_unique();
        let mut lhs_rightmost_child = find_transitive_rightmost_child(lhs_node.borrow_mut());

        let lhs_value = lhs_rightmost_child.value_mut();
        S::try_join_into_lhs(lhs_value, mid_slice);

        match &mid_slice {
            // Couldn't join with LHS
            Some(_) => {
                drop(lhs_rightmost_child);
            }
            None => {
                // Update subtree sizes all the way up the tree
                let mut node = Some(lhs_rightmost_child);
                while let Some(mut n) = node {
                    let old_size = n.subtree_size();
                    n.set_subtree_size(old_size.add_right(mid_size));
                    node = n.into_parent().map(|(n, _)| n);
                }
            }
        }

        lhs = Some(lhs_node.erase());
        if mid_slice.is_none() {
            // If we successfully joined with LHS, we can fall through to `join_trees` to finish
            // merging the two sides.
            return join_trees(lhs, rhs, true).into_unique();
        }
    }

    // Try joining with RHS
    if let Some(rhs_node) = rhs {
        let mut rhs_node = rhs_node.into_unique();
        let mut rhs_leftmost_child = find_transitive_leftmost_child(rhs_node.borrow_mut());

        let rhs_value = rhs_leftmost_child.value_mut();
        S::try_join_into_rhs(mid_slice, rhs_value);
        match &mid_slice {
            // Couldn't join with RHS
            Some(_) => {
                drop(rhs_leftmost_child);
            }
            None => {
                // Update subtree sizes all the way up the tree
                let mut node = Some(rhs_leftmost_child);
                while let Some(mut n) = node {
                    let old_size = n.subtree_size();
                    n.set_subtree_size(mid_size.add_right(old_size));
                    node = n.into_parent().map(|(n, _)| n);
                }
            }
        }

        rhs = Some(rhs_node.erase());
        if mid_slice.is_none() {
            // If we successfully joined with RHS, we can fall through to `join_trees` to finish
            // merging the two sides.
            //
            // note: unlike with LHS, we don't need try_join_slices = true, because we already
            // tried joining with RHS.
            return join_trees(lhs, rhs, false).into_unique();
        };
    }

    // If we couldn't join with either side, then - in the general case - we can use the middle
    // value as the new root (and then do an unbounded rebalance).
    // But before that, if we only have one of LHS or RHS, we can just do an insertion!
    match (lhs, rhs) {
        (None, None) => crate::panic_internal_error_or_bad_index::<I>(
            "got no nodes to join together for final tree",
        ),
        (Some(lhs), None) => {
            let mut lhs = lhs.into_unique();
            // Only LHS - insert at the end.
            let lhs_size = lhs.subtree_size();
            super::run_insert(
                lhs.borrow_mut(),
                None,
                false,
                DownwardInsertState {
                    target: lhs_size,
                    fst_value: InsertionValue { slice: mid_slice, size: mid_size },
                    snd_value: None,
                    allow_joining: true,
                    already_split_once: false,
                },
            );

            fix::fix_unique_owned(lhs, FixMode::Normal)
        }
        (None, Some(rhs)) => {
            let mut rhs = rhs.into_unique();
            // Only RHS - insert at the start.
            super::run_insert(
                rhs.borrow_mut(),
                None,
                false,
                DownwardInsertState {
                    target: I::ZERO,
                    fst_value: InsertionValue { slice: mid_slice, size: mid_size },
                    snd_value: None,
                    allow_joining: true,
                    already_split_once: false,
                },
            );

            fix::fix_unique_owned(rhs, FixMode::Normal)
        }
        (Some(lhs), Some(rhs)) => {
            // Both LHS and RHS, make a new tree with the middle value at the root.
            let root_subtree_size =
                lhs.subtree_size().add_right(mid_size).add_right(rhs.subtree_size());
            let mut root = node::NodeHandle::alloc_new(mid_slice, root_subtree_size);
            root.borrow_mut().insert_lhs(lhs);
            root.borrow_mut().insert_rhs(rhs);
            fix::fix_unique_owned(root, FixMode::Unbounded)
        }
    }
}

fn join_trees<I, S, P>(
    lhs: Option<node::HandleOwned<I, S, P>>,
    rhs: Option<node::HandleOwned<I, S, P>>,
    try_join_slices: bool,
) -> node::HandleOwned<I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    let (mut lhs, rhs) = match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => (lhs, rhs),
        (Some(n), None) | (None, Some(n)) => return n,
        (None, None) => crate::panic_internal_error_or_bad_index::<I>(
            "got no nodes to join together for final tree or removal nodes",
        ),
    };

    let (mut middle_node, maybe_rhs) = remove_transitive_leftmost_child(rhs.into_unique());
    let middle_size = middle_node.subtree_size();

    if try_join_slices {
        let mut lhs_unique = lhs.into_unique();

        let mut lhs_rightmost_child = find_transitive_rightmost_child(lhs_unique.borrow_mut());
        let mut middle_node_mut = middle_node.borrow_mut();

        let lhs_value = lhs_rightmost_child.value_mut();
        let rhs_value = middle_node_mut.value_mut();
        S::try_join_into_lhs(lhs_value, rhs_value);

        match &rhs_value {
            // Couldn't join. We'll fall through to the normal case below.
            // (i.e. as if !try_join_slices)
            Some(_) => {
                // We already gave up `lhs` to make it unique; erase the type again.
                drop(lhs_rightmost_child);
                lhs = lhs_unique.erase();
            }
            // Otherwise, we just joined the middle value into LHS, so we need to:
            // 1. Record the updated size back up the LHS tree
            // 2. Try joining the LHS & RHS trees again, now that we just consumed middle_node
            None => {
                middle_node_mut.write_redirect(P::Redirect::to(lhs_rightmost_child.borrow()));

                let old_subtree_size = lhs_rightmost_child.subtree_size();
                lhs_rightmost_child.set_subtree_size(old_subtree_size.add_right(middle_size));

                // Walk back up the tree & update the sizes of all nodes
                let mut up_state = UpwardUpdateState { old_size: old_subtree_size };
                let mut node = lhs_rightmost_child;
                while node.has_parent() {
                    (node, up_state) = up_state.step(node, None);
                }
                drop(node);

                // Now that we've consumed middle_node, we need to try joining LHS & RHS again.
                // RHS might be None, and we'll need to pick a new middle node, so it's easiest to
                // just call this function again. It's guaranteed not to recurse this time, because
                // we won't allow joining.
                return join_trees(Some(lhs_unique.erase()), maybe_rhs, false);
            }
        }
    }

    let new_subtree_size = lhs
        .subtree_size()
        .add_right(middle_size)
        .add_right(maybe_rhs.as_ref().map(|n| n.subtree_size()).unwrap_or(I::ZERO));
    middle_node.set_subtree_size(new_subtree_size);

    middle_node.borrow_mut().insert_lhs(lhs);
    if let Some(rhs) = maybe_rhs {
        middle_node.borrow_mut().insert_rhs(rhs);
    }

    fix::fix_unique_owned(middle_node, FixMode::Unbounded).erase()
}

fn find_transitive_leftmost_child<I, S, P>(
    mut node: node::HandleMut<'_, I, S, P>,
) -> node::HandleMut<'_, I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    loop {
        match node.into_lhs() {
            Ok(child) => node = child,
            Err(n) => return n,
        }
    }
}

fn find_transitive_rightmost_child<I, S, P>(
    mut node: node::HandleMut<'_, I, S, P>,
) -> node::HandleMut<'_, I, S, P>
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    loop {
        match node.into_rhs() {
            Ok(child) => node = child,
            Err(n) => return n,
        }
    }
}

fn remove_transitive_leftmost_child<I, S, P>(
    mut tree: node::HandleUniqueOwned<I, S, P>,
) -> (node::HandleUniqueOwned<I, S, P>, Option<node::HandleOwned<I, S, P>>)
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    let Ok(mut node) = tree.borrow_mut().into_lhs() else {
        // No left-hand child, but we still might have a right-hand child.
        let rhs = tree.take_rhs();
        if let Some(n) = &rhs {
            let new_subtree_size = tree.subtree_size().sub_right(n.subtree_size());
            tree.set_subtree_size(new_subtree_size);
        }
        return (tree, rhs);
    };

    loop {
        match node.into_lhs() {
            Ok(child) => node = child,
            Err(n) => {
                node = n;
                break;
            }
        }
    }

    // At this point, we know that:
    // 1. `node` was its parent's LHS child; and
    // 2. `node` has no LHS child
    let (mut parent, _) =
        node.into_parent().expect("node was some LHS child, so should have parent");
    let original_parent_size = parent.subtree_size();

    let mut leftmost_child = parent
        .take_lhs()
        .expect("parent previously had LHS child, so should have one still")
        .into_unique();

    let original_leftmost_child_size = leftmost_child.subtree_size();

    let leftmost_child_rhs = leftmost_child.take_rhs();
    let leftmost_child_rhs_size =
        leftmost_child_rhs.as_ref().map(|n| n.subtree_size()).unwrap_or(I::ZERO);
    leftmost_child
        .set_subtree_size(original_leftmost_child_size.sub_right(leftmost_child_rhs_size));

    if let Some(rhs) = leftmost_child_rhs {
        parent.borrow_mut().insert_lhs(rhs);
    }

    parent.set_subtree_size(
        original_parent_size
            .sub_left(original_leftmost_child_size)
            .add_left(leftmost_child_rhs_size),
    );

    let mut up_state = UpwardUpdateState { old_size: original_parent_size };
    while parent.has_parent() {
        parent = fix::fix_mut(parent, FixMode::Normal);
        (parent, up_state) = up_state.step(parent, None);
    }

    drop(parent);

    tree = fix::fix_unique_owned(tree, FixMode::Normal);
    (leftmost_child, Some(tree.erase()))
}
