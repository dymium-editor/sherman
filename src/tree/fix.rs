//! Tree rebalancing

use std::num::NonZeroU16;

use super::node::{HandleImmut, HandleMut, HandleUniqueOwned, Side};
use crate::{Index, Slice};

/// Recursively validates that the subtree rooted at this node is balanced
#[cfg(any(test, feature = "fuzz"))]
pub(super) fn validate_balance<I, S>(node: HandleImmut<'_, I, S>)
where
    I: Index,
    S: Slice<I>,
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

/// Rebalances the subtree rooted at the node, given a mutable reference
///
/// Up to two rotations will be performed, the height of this node will be corrected, and a mutable
/// handle on the new subtree root will be returned.
///
/// # Panics
///
/// This function panics if the node does not have a parent. To fix a node with no parent, you
/// should use [`fix_owned`] instead.
pub(super) fn fix_mut<I, S>(mut node: HandleMut<I, S>) -> HandleMut<I, S>
where
    I: Index,
{
    if node.parent_addr().is_none() {
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
            .expect("parent's LHS should be Some(_) if this node's parent side is LHS"),
        Side::Rhs => parent
            .take_rhs()
            .expect("parent's LHS should be Some(_) if this node's parent side is LHS"),
    };

    this = fix_invasive(this, fix_side);

    match side {
        Side::Lhs => parent.insert_lhs(this),
        Side::Rhs => parent.insert_rhs(this),
    }
}

/// Rebalances the subtree rooted at this node
pub(super) fn fix_owned<I, S>(mut node: HandleUniqueOwned<I, S>) -> HandleUniqueOwned<I, S>
where
    I: Index,
{
    reset_height(node.borrow_mut());

    if let Some(side) = needs_rebalance(node.borrow()) {
        node = fix_invasive(node, side);
    }

    node
}

/// Sets the height of the node to match the left and right child, if any
fn reset_height<I, S>(mut node: HandleMut<'_, I, S>)
where
    I: Index,
{
    let lhs_height = node.lhs_height();
    let rhs_height = node.rhs_height();
    let height = NonZeroU16::new(1 + u16::max(lhs_height, rhs_height))
        .expect("node height should not overflow u16");
    node.set_height(height);
}

/// If the subtree rooted at this node should be rebalanced, returns the child that is too tall
fn needs_rebalance<I, S>(node: HandleImmut<'_, I, S>) -> Option<Side> {
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

fn fix_invasive<I, S>(mut node: HandleUniqueOwned<I, S>, side: Side) -> HandleUniqueOwned<I, S>
where
    I: Index,
{
    match side {
        Side::Lhs => {
            // LHS is too tall compared to RHS; we need to rotate_right() on this node.
            let requires_lhs_rotate_left = {
                // ... but if LHS's right-hand child is larger than its left-hand child, we'll need
                // to rotate_left() on LHS first, to end up splitting the higher left-right subtree
                // between lhs and this node.
                let lhs = node
                    .borrow()
                    .into_lhs()
                    .expect("lhs height cannot be > rhs height if lhs is None");
                lhs.rhs_height() > lhs.lhs_height()
            };
            if requires_lhs_rotate_left {
                let mut lhs = node.take_lhs().unwrap();
                lhs = rotate_left(lhs);
                node.borrow_mut().insert_lhs(lhs);
            }
            rotate_right(node)
        }
        Side::Rhs => {
            // RHS is too tall compared to LHS; we need to rotate_left() on this node.
            let requires_rhs_rotate_right = {
                // ... but if RHS's left-hand child is larger than its right-hand child, we'll need
                // to rotate_right() on RHS first, to end up splitting the larger right-left
                // subtree between rhs and this node.
                let rhs = node
                    .borrow()
                    .into_rhs()
                    .expect("rhs height cannot be > lhs height if rhs is None");
                rhs.lhs_height() > rhs.rhs_height()
            };
            if requires_rhs_rotate_right {
                let mut rhs = node.take_rhs().unwrap();
                rhs = rotate_right(rhs);
                node.borrow_mut().insert_rhs(rhs);
            }
            rotate_left(node)
        }
    }
}

/// Performs a "rotate left" operation on the subtree rooted at the node
fn rotate_left<I, S>(node: HandleUniqueOwned<I, S>) -> HandleUniqueOwned<I, S>
where
    I: Index,
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
        .expect("rotate_left() should only be called when RHS is Some(_)");
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
    node_c.borrow_mut().insert_lhs(node_a);
    reset_height(node_c.borrow_mut());
    node_c
}

/// Performs a "rotate right" operation on the subtree rooted at the node
fn rotate_right<I, S>(node: HandleUniqueOwned<I, S>) -> HandleUniqueOwned<I, S>
where
    I: Index,
{
    // A "right" rotation makes the following transformation:
    // Given a node A with children B and C, with node B having children D and E, replace the
    // subtree rooted at A with: node B with children D and A, with node A having chidren E and C.
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
        .expect("rotate_right() should only be called when LHS is Some(_)");
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
    node_b.borrow_mut().insert_rhs(node_a);
    reset_height(node_b.borrow_mut());
    node_b
}
