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
