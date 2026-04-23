//! Tests generated from fuzzing for `SliceRefOperation<u8, Constant<UpperLetter>>`

use crate::param::EnableRefs;
use crate::{Constant, RleTree};

#[test]
fn test_01_basic_insert_merge_edge_lhs() {
    let mut tree: RleTree<u8, Constant<char>, EnableRefs> = RleTree::new_empty();
    tree.insert(0, Constant('A'), 14);
    tree.insert(0, Constant('A'), 14);
}

#[test]
fn test_02_basic_removal() {
    let mut tree: RleTree<u8, Constant<char>, EnableRefs> = RleTree::new_empty();
    tree.insert(0, Constant('L'), 96);
    let ref_0 = tree.get(18).stable_ref();
    _ = tree.remove(53..54);
    // This actually surfaced an issue with the fake API's handling of slice references, but it's
    // worth including here as well for coverage.
    {
        let entry = ref_0.get(&tree).unwrap();
        assert_eq!(entry.range(), 0..95);
        assert_eq!(entry.slice(), &Constant('L'));
    }
}
