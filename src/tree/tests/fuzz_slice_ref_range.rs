//! Tests generated from fuzzing for `SliceRefOperation<u8, CharRange>`

use super::CharRange;
use crate::RleTree;
use crate::param::EnableRefs;

#[test]
fn test_01_insert_edge_lhs_join_both() {
    let mut tree: RleTree<u8, CharRange, EnableRefs> = RleTree::new_empty();
    tree.insert(0, CharRange('A'..'X'), 93);
    let _ref_0 = tree.get(0).stable_ref();
    let _ref_1 = tree.get(19).stable_ref();
    tree.validate_balance();
    tree.insert(19, CharRange('V'..'Z'), 19);
    tree.insert(19, CharRange('T'..'V'), 19);
}
