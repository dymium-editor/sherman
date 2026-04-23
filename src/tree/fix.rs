//! Tree rebalancing

use std::num::NonZeroU16;

use super::node::{HandleImmut, HandleMut, HandleOwned, HandleUniqueOwned, Side};
use crate::Index;
#[cfg(any(test, feature = "fuzz"))]
use crate::Slice;
use crate::param::{RleTreeConfig, SupportsUpdate};

/// Recursively validates that the subtree rooted at this node is balanced
#[cfg(any(test, feature = "fuzz"))]
pub(super) fn validate_balance<I, S, P>(node: HandleImmut<'_, I, S, P>)
where
    I: Index,
    S: Slice<I>,
    P: RleTreeConfig<I, S>,
{
    let addr = node.addr();
    let height = node.height();
    let lhs_height = node.lhs_height();
    let rhs_height = node.rhs_height();

    if height.get() != 1 + u16::max(lhs_height, rhs_height) {
        panic!(
            "invalid height for node {addr:x}: height={height}, lhs_height={lhs_height}, rhs_height={rhs_height}"
        );
    }

    if lhs_height + 1 < rhs_height || lhs_height > rhs_height + 1 {
        panic!("unbalanced node {addr:x}: lhs_height={lhs_height}, rhs_height={rhs_height}");
    }

    if let Some(lhs) = node.borrow().into_lhs() {
        validate_balance(lhs);
    }
    if let Some(rhs) = node.borrow().into_rhs() {
        validate_balance(rhs);
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(Debug))]
pub(super) enum FixMode {
    /// A normal deviation must be corrected, e.g. a single insertion or deletion
    ///
    /// At most two rotations will be performed.
    Normal,
    /// The (sub)tree may be arbitrarily imbalanced due to removing a large number of nodes
    ///
    /// Note that this only accommodates imbalance *between* the two sides of this subtree, rather
    /// than imbalance internal to either child. It is expected that callers would have handled
    /// imbalanced children on their own, only coming here to handle rebalancing between them.
    ///
    /// As many rotations will be performed as is required to correct the imbalance.
    Unbounded,
}

/// Rebalances the subtree rooted at the node, given a mutable reference
///
/// Rotations will be performed (limited by the [`FixMode`]), the height of this node will be
/// corrected, and a mutable handle on the new subtree root will be returned.
///
/// # Panics
///
/// This function panics if the node does not have a parent. To fix a node with no parent, you
/// should use [`fix_owned`] instead.
pub(super) fn fix_mut<I, S, P>(mut node: HandleMut<I, S, P>, mode: FixMode) -> HandleMut<I, S, P>
where
    I: Index,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    if !node.has_parent() {
        panic!("internal error: cannot fix() node that has no parent");
    }

    reset_height(node.borrow_mut());

    let fix_side = match needs_rebalance(node.borrow()) {
        Some(s) => s,
        None => return node,
    };

    // We need to rebalance this node.
    // Temporarily remove it from its parent in order to get an owned handle.
    let (mut parent, side) = node.into_parent().expect("parent should still be Some(_)");

    let mut this = match side {
        Side::Lhs => parent
            .take_lhs()
            .expect("parent's LHS should be Some(_) if this node's parent side is LHS")
            .into_unique(),
        Side::Rhs => parent
            .take_rhs()
            .expect("parent's LHS should be Some(_) if this node's parent side is LHS")
            .into_unique(),
    };

    this = fix_invasive(this, fix_side, mode);

    match side {
        Side::Lhs => parent.insert_into_lhs(this),
        Side::Rhs => parent.insert_into_rhs(this),
    }
}

/// Rebalances the subtree rooted at this node, only if it is unique.
///
/// If the node is *not* unique, we can assume it was already fixed when it was unique.
pub(super) fn fix_owned<I, S, P>(node: HandleOwned<I, S, P>, mode: FixMode) -> HandleOwned<I, S, P>
where
    I: Index,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    match node.try_into_unique() {
        Ok(unique) => fix_unique_owned(unique, mode).erase(),
        Err(n) => n,
    }
}

/// Rebalances the subtree rooted at this node
pub(super) fn fix_unique_owned<I, S, P>(
    mut node: HandleUniqueOwned<I, S, P>,
    mode: FixMode,
) -> HandleUniqueOwned<I, S, P>
where
    I: Index,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    reset_height(node.borrow_mut());

    if let Some(side) = needs_rebalance(node.borrow()) {
        node = fix_invasive(node, side, mode);
    }

    node
}

/// Sets the height of the node to match the left and right child, if any
fn reset_height<I, S, P>(mut node: HandleMut<'_, I, S, P>)
where
    I: Index,
    P: RleTreeConfig<I, S>,
{
    let lhs_height = node.lhs_height();
    let rhs_height = node.rhs_height();
    let height = NonZeroU16::new(1 + u16::max(lhs_height, rhs_height))
        .expect("node height should not overflow u16");
    node.set_height(height);
}

/// If the subtree rooted at this node should be rebalanced, returns the child that is too tall
fn needs_rebalance<I, S, P>(node: HandleImmut<'_, I, S, P>) -> Option<Side>
where
    P: RleTreeConfig<I, S>,
{
    let lhs_height = node.lhs_height();
    let rhs_height = node.rhs_height();

    if lhs_height > rhs_height + 1 {
        // LHS is too high; we'll need to rebalance away from it.
        Some(Side::Lhs)
    } else if rhs_height > lhs_height + 1 {
        // RHS is too high; we'll need to rebalance away from it.
        Some(Side::Rhs)
    } else {
        None
    }
}

/// Fixes an owned node, where `side` is higher.
///
/// The left- and right-hand children of the node must already be balanced.
fn fix_invasive<I, S, P>(
    mut node: HandleUniqueOwned<I, S, P>,
    side: Side,
    #[cfg_attr(not(debug_assertions), expect(unused))] mode: FixMode,
) -> HandleUniqueOwned<I, S, P>
where
    I: Index,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    match side {
        Side::Lhs => {
            // LHS is too tall compared to RHS; we need to rotate_right() on this node, up to as
            // many times as the difference in height.
            //
            // Each time we rotate_right(), we need to ensure that the left-right child has the
            // same height as the right child (or at most greater by one) - otherwise, when we
            // rotate_right(), the new right child will itself be unbalanced, *and* potentially
            // taller than the left child (e.g., if left-left child is shorter than left-right).
            'rebalance_root: loop {
                // Assume we need to fix; check again at the end.
                let rhs_height = node.rhs_height();
                let mut lhs =
                    node.take_lhs().expect("lhs height cannot be > rhs height if lhs is None");

                'prepare_lhs: loop {
                    // We must rotate_left(lhs) under two conditions:
                    //
                    // 1. If the root would be unbalanced after rotate_right(), i.e. new RHS would
                    //    have 2+ more height than new LHS. This happens if left-left child height
                    //    is 2+ less than 1 + left-right child height (i.e. left-left < left-right)
                    //
                    //    (This is the typical condition for a left-right rotation in AVL trees.)
                    //
                    let root_would_be_unbalanced = lhs.lhs_height() < lhs.rhs_height();
                    //
                    // 2. If new RHS would be unbalanced after rotate_right(), i.e. new RHS's LHS
                    //    would have 2+ more height than new RHS's RHS. This happens if left-right
                    //    child height is 2+ more than current right child.
                    //
                    //    (This cannot happen in normally balanced AVL trees; we handle it here
                    //    because we may have an unbounded height difference between LHS/RHS.)
                    //
                    let rhs_would_be_unbalanced = lhs.rhs_height() > rhs_height + 1;
                    debug_assert!(mode == FixMode::Unbounded || !rhs_would_be_unbalanced);
                    //
                    // If we don't need to rotate_left(lhs), we can proceed to rotate_right() on
                    // the node.
                    if !root_would_be_unbalanced && !rhs_would_be_unbalanced {
                        break 'prepare_lhs;
                    }

                    lhs = rotate_left(lhs.into_unique()).erase();
                }
                // Put LHS back; we're done preparing it for rotate_right()
                node.borrow_mut().insert_lhs(lhs);

                // NOW we can rotate_right() to shrink the difference between LHS and RHS by 1-2.
                node = rotate_right(node);

                // At this point, it may be the case that we did multiple rotate_left() operations
                // on LHS, and as we're unpacking those via rotate_right(), there's differences in
                // the distribution of heights that cause RHS to be unbalanced.
                //
                // In particular, we might find that the left-right child's height is less than the
                // right child's height. We can't fix that by rotate_right(lhs) because that allows
                // us to progress too quickly through the "stack" that we've built through left
                // rotations. Because this also might involve e.g. right-left rotations on RHS in
                // order to resolve the imbalance, we'll do a recursive call. BUT we only need to
                // make a single change, not an unbounded number.
                if mode == FixMode::Unbounded {
                    let mut rhs = node
                        .take_rhs()
                        .expect("rhs must be present after rotate_right with non-zero lhs height");
                    rhs = fix_unique_owned(rhs.into_unique(), FixMode::Normal).erase();
                    node.borrow_mut().insert_rhs(rhs);
                    reset_height(node.borrow_mut());
                }

                // Potentially keep fixing the root node
                if mode == FixMode::Unbounded {
                    // There are two conditions for continuing to fix the node:
                    //
                    //  1. If it is unbalanced on its face (LHS height 2+ more than RHS height); or
                    if node.lhs_height() > node.rhs_height() + 1 {
                        continue 'rebalance_root;
                    }
                    //  2. If LHS is still unbalanced because of earlier rotate_left()
                    if let Some(lhs) = node.borrow().into_lhs()
                        && lhs.lhs_height() > lhs.rhs_height() + 1
                    {
                        continue 'rebalance_root;
                    }
                }

                // All done!
                break node;
            }
        }
        Side::Rhs => {
            // RHS is too tall compared to LHS; we need to rotate_left() on this node, up to as
            // many times as the difference in height.
            //
            // Each time we rotate_left(), we need to ensure that the right-left child has the same
            // height as the left child (or at most greater than by one) - otherwise, when we
            // rotate_left(), the new left child will itself be unbalanced, *and* potentially
            // taller than the right child (e.g., if right-right child is shorter than right-left).
            'rebalance_root: loop {
                // Assume we need to fix; check again at the end.
                let lhs_height = node.lhs_height();
                let mut rhs =
                    node.take_rhs().expect("rhs height cannot be > rhs height if rhs is None");

                'prepare_rhs: loop {
                    // We must rotate_right(rhs) under two conditions:
                    //
                    // 1. If the root would be unbalanced after rotate_left(), i.e. new LHS would
                    //    have 2+ more height than new RHS. This happens if right-right child height
                    //    is 2+ less than 1 + right-left child height (i.e. right-right < right-left)
                    //
                    //    (This is the typical condition for a right-left rotation in AVL trees.)
                    let root_would_be_unbalanced = rhs.rhs_height() < rhs.lhs_height();
                    //
                    // 2. If new LHS would be unbalanced after rotate_left(), i.e. new LHS's RHS
                    //    would have 2+ more height than new LHS's LHS. This happens if right-left
                    //    child height is 2+ more than current left child.
                    //
                    //    (This cannot happen in normally in balanced AVL trees; we handle it here
                    //    because we may have an unbounded height difference between LHS/RHS.)
                    //
                    let lhs_would_be_unbalanced = rhs.lhs_height() > lhs_height + 1;
                    debug_assert!(mode == FixMode::Unbounded || !lhs_would_be_unbalanced);
                    //
                    // If we don't need to rotate_right(rhs), we can proceed to rotate_left() on
                    // the node.
                    if !root_would_be_unbalanced && !lhs_would_be_unbalanced {
                        break 'prepare_rhs;
                    }

                    rhs = rotate_right(rhs.into_unique()).erase();
                }
                // Put RHS back; we're done with it for now.
                node.borrow_mut().insert_rhs(rhs);

                // NOW we can rotate_left() to shrink the difference between RHS and LHS by 1-2.
                node = rotate_left(node);

                // At this point, it may be the case that we did multiple rotate_right() operations
                // on RHS, and as we're unpacking those via rotate_left(), there's differences in
                // the distribution of heights that cause LHS to be unbalanced.
                //
                // In particular, we might find that the right-left child's height is less than the
                // left child's height. We can't fix that by rotate_left(rhs) because that allows
                // us to progress too quickly through the "stack" that we've built through right
                // rotations. Because this also might involve e.g. left-right rotations on LHS in
                // order to resolve the imbalance, we'll do a recursive call. BUT we only need to
                // make a single change, not an unbounded number.
                if mode == FixMode::Unbounded {
                    debug_assert!(mode == FixMode::Unbounded);
                    let mut lhs = node
                        .take_lhs()
                        .expect("lhs must be present after rotate_left with non-zero rhs height");
                    lhs = fix_unique_owned(lhs.into_unique(), FixMode::Normal).erase();
                    node.borrow_mut().insert_lhs(lhs);
                    reset_height(node.borrow_mut());
                }

                // Potentially keep fixing the root node
                if mode == FixMode::Unbounded {
                    // There are two conditions for continuing to fix the node:
                    //
                    //  1. If it is unbalanced on its face (RHS height 2+ more than LHS height); or
                    if node.rhs_height() > node.lhs_height() + 1 {
                        continue 'rebalance_root;
                    }
                    //  2. If RHS is still unbalanced because of earlier rotate_right()
                    if let Some(rhs) = node.borrow().into_rhs()
                        && rhs.rhs_height() > rhs.lhs_height() + 1
                    {
                        continue 'rebalance_root;
                    }
                }

                // All done!
                break node;
            }
        }
    }
}

/// Performs a "rotate left" operation on the subtree rooted at the node
fn rotate_left<I, S, P>(node: HandleUniqueOwned<I, S, P>) -> HandleUniqueOwned<I, S, P>
where
    I: Index,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    // A "left" rotation makes the following transformation:
    // Given a node A with children B and C, with node C having children D and E, replace the
    // subtree rooted at A with: node C with children A and E, with node A having children B and D.
    //
    // In visual form, convert:
    //           ┏━━━┓
    //           ┃ A ┃
    //   ┏━━━┓   ┗━━━┛   ┏━━━┓
    //   ┃ B ┃           ┃ C ┃
    //   ┗━━━┛      ┏━━━┓┗━━━┛┏━━━┓
    //              ┃ D ┃     ┃ E ┃
    //              ┗━━━┛     ┗━━━┛
    // into:
    //                ┏━━━┓
    //                ┃ C ┃
    //        ┏━━━┓   ┗━━━┛   ┏━━━┓
    //        ┃ A ┃           ┃ E ┃
    //   ┏━━━┓┗━━━┛┏━━━┓      ┗━━━┛
    //   ┃ B ┃     ┃ D ┃
    //   ┗━━━┛     ┗━━━┛
    // We'll refer to these nodes with their letters.
    let mut node_a = node;
    let mut node_c = node_a
        .take_rhs()
        .expect("rotate_left() should only be called when RHS is Some(_)")
        .into_unique();
    let node_d = node_c.take_lhs();

    // Collect starting information:
    let orig_subtree_size_a = node_a.subtree_size();
    let orig_subtree_size_b =
        node_a.borrow().into_lhs().map(|n| n.subtree_size()).unwrap_or(I::ZERO);
    let orig_subtree_size_c = node_c.subtree_size();
    let orig_subtree_size_d = node_d.as_ref().map(|n| n.subtree_size()).unwrap_or(I::ZERO);
    let orig_subtree_size_e =
        node_c.borrow().into_rhs().map(|n| n.subtree_size()).unwrap_or(I::ZERO);

    let value_size_a =
        orig_subtree_size_a.sub_left(orig_subtree_size_b).sub_right(orig_subtree_size_c);
    let value_size_c =
        orig_subtree_size_c.sub_left(orig_subtree_size_d).sub_right(orig_subtree_size_e);

    // Set new sizing for 'A' and 'C', then set the node relationships
    let new_subtree_size_a =
        orig_subtree_size_b.add_right(value_size_a).add_right(orig_subtree_size_d);
    let new_subtree_size_c =
        new_subtree_size_a.add_right(value_size_c).add_right(orig_subtree_size_e);
    node_a.set_subtree_size(new_subtree_size_a);
    node_c.set_subtree_size(new_subtree_size_c);

    if let Some(d) = node_d {
        node_a.borrow_mut().insert_rhs(d);
    }
    reset_height(node_a.borrow_mut());
    node_c.borrow_mut().insert_lhs(node_a.erase());
    reset_height(node_c.borrow_mut());
    node_c
}

/// Performs a "rotate right" operation on the subtree rooted at the node
fn rotate_right<I, S, P>(node: HandleUniqueOwned<I, S, P>) -> HandleUniqueOwned<I, S, P>
where
    I: Index,
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    // A "right" rotation makes the following transformation:
    // Given a node A with children B and C, with node B having children D and E, replace the
    // subtree rooted at A with: node B with children D and A, with node A having children E and C.
    //
    // In visual form, convert:
    //                ┏━━━┓
    //                ┃ A ┃
    //        ┏━━━┓   ┗━━━┛   ┏━━━┓
    //        ┃ B ┃           ┃ C ┃
    //   ┏━━━┓┗━━━┛┏━━━┓      ┗━━━┛
    //   ┃ D ┃     ┃ E ┃
    //   ┗━━━┛     ┗━━━┛
    // into:
    //           ┏━━━┓
    //           ┃ B ┃
    //   ┏━━━┓   ┗━━━┛   ┏━━━┓
    //   ┃ D ┃           ┃ A ┃
    //   ┗━━━┛      ┏━━━┓┗━━━┛┏━━━┓
    //              ┃ E ┃     ┃ C ┃
    //              ┗━━━┛     ┗━━━┛
    // We'll refer to these nodes with their letters.
    let mut node_a = node;
    let mut node_b = node_a
        .take_lhs()
        .expect("rotate_right() should only be called when LHS is Some(_)")
        .into_unique();
    let node_e = node_b.take_rhs();

    // Collect starting information:
    let orig_subtree_size_a = node_a.subtree_size();
    let orig_subtree_size_b = node_b.subtree_size();
    let orig_subtree_size_c =
        node_a.borrow().into_rhs().map(|n| n.subtree_size()).unwrap_or(I::ZERO);
    let orig_subtree_size_d =
        node_b.borrow().into_lhs().map(|n| n.subtree_size()).unwrap_or(I::ZERO);
    let orig_subtree_size_e = node_e.as_ref().map(|n| n.subtree_size()).unwrap_or(I::ZERO);

    let value_size_a =
        orig_subtree_size_a.sub_left(orig_subtree_size_b).sub_right(orig_subtree_size_c);
    let value_size_b =
        orig_subtree_size_b.sub_left(orig_subtree_size_d).sub_right(orig_subtree_size_e);

    // Set new sizing for 'A' and 'B', then set the node relationships
    let new_subtree_size_a =
        orig_subtree_size_e.add_right(value_size_a).add_right(orig_subtree_size_c);
    let new_subtree_size_b =
        orig_subtree_size_d.add_right(value_size_b).add_right(new_subtree_size_a);
    node_a.set_subtree_size(new_subtree_size_a);
    node_b.set_subtree_size(new_subtree_size_b);

    if let Some(e) = node_e {
        node_a.borrow_mut().insert_lhs(e);
    }
    reset_height(node_a.borrow_mut());
    node_b.borrow_mut().insert_rhs(node_a.erase());
    reset_height(node_b.borrow_mut());
    node_b
}
