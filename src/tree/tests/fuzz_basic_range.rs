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
    enable_debug!();
    tree.insert(4, CharRange('E'..'J'), 4);
    tree.validate_balance();
}
