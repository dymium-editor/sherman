//! Tests generated from fuzzing for `BasicOperation<u8, CharRange>`

use super::CharRange;
use crate::RleTree;

#[test]
fn test_01_rhs_edge_join_removal_rebalance() {
    let mut tree: RleTree<u8, CharRange> = RleTree::new_empty();
    tree.insert(0, CharRange('B'..'G'), 4);
    tree.insert(1, CharRange('B'..'G'), 4);
    tree.insert(4, CharRange('E'..'J'), 4);
    tree.insert(4, CharRange('J'..'O'), 4);
    tree.insert(4, CharRange('J'..'O'), 4);
    // On this insertion, we end up joining on either side with 'B'..'E' and 'J'..'O', which means
    // we end up removing a node.
    tree.insert(4, CharRange('E'..'J'), 4);
    tree.validate_balance();
}

#[test]
fn test_02_removal_recursive_rebalance_shrink_child_subtree() {
    let mut tree: RleTree<u8, CharRange> = RleTree::new_empty();
    tree.insert(0, CharRange('C'..'F'), 2);
    tree.insert(2, CharRange('C'..'F'), 18);
    tree.insert(2, CharRange('C'..'F'), 2);
    tree.insert(2, CharRange('C'..'V'), 2);
    tree.insert(8, CharRange('I'..'R'), 8);
    tree.insert(8, CharRange('I'..'R'), 8);
    tree.insert(8, CharRange('I'..'R'), 8);
    tree.insert(2, CharRange('C'..'F'), 47);
    tree.insert(2, CharRange('C'..'F'), 38);
    tree.insert(2, CharRange('I'..'R'), 8);
    tree.insert(8, CharRange('I'..'R'), 8);
    tree.insert(8, CharRange('I'..'R'), 8);
    tree.insert(8, CharRange('A'..'J'), 8);
    tree.insert(8, CharRange('I'..'R'), 8);
    tree.insert(8, CharRange('I'..'R'), 8);
    tree.insert(8, CharRange('I'..'U'), 8);
    tree.insert(8, CharRange('I'..'R'), 8);
    tree.insert(8, CharRange('I'..'R'), 45);
    // During this removal, we perform a recursive rebalancing that ends up shrinking the height of
    // one of the subtrees after we've already recursed into it.
    _ = tree.remove(..91);
    tree.validate_balance();
}

#[test]
fn test_03_deeply_recursive_removal_imbalance() {
    // This is a heavily trimmed version of the original fuzz case.
    // The first version was 900+ lines long, and `cargo fuzz tmin` got it down to about 600 lines.
    // Most of that was tons of calls to drain/iter .next(), which have now been removed (and in
    // the case of drain, changed to remove() instead).
    // And similarly, zero-sized removals and non-critical validate_balance()s were also cleaned up.

    let mut tree: RleTree<u8, CharRange> = RleTree::new_empty();
    tree.insert(0, CharRange('A'..'E'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(8, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('W'..'X'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 2);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(8, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('W'..'X'), 3);
    tree.insert(3, CharRange('T'..'X'), 3);
    tree.insert(19, CharRange('D'..'H'), 2);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 1);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('T'..'X'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('W'..'X'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('U'..'Y'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('U'..'Y'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('W'..'X'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('U'..'Y'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(..51);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'V'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 1);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('T'..'X'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('W'..'X'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('U'..'Y'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('U'..'Y'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('U'..'Y'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(19, CharRange('D'..'H'), 3);
    tree.insert(2, CharRange('E'..'I'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    _ = tree.remove(3..19);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.insert(3, CharRange('D'..'H'), 3);
    tree.validate_balance();
    // During this removal, the initial version of the resulting tree ends up with deeply nested
    // tolerable imbalances:
    //
    // * The root is height 7 with no right-hand child (expected)
    // * Root's LHS (N₁) has balanced children
    // * N₁'s RHS (N₂) has RHS taller than LHS
    // * N₂'s RHS (N₃) has LHS taller than RHS
    // * And both of N₃'s children are outwardly balanced (e.g. LHS LHS taller than LHS RHS)
    //
    // This makes rebalancing tricky: we must accommodate many potential edge cases.
    _ = tree.remove(82..);
    tree.validate_balance();
}
