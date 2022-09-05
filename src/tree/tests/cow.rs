use crate::{param::AllowCow, Constant, RleTree};

type CowFuzzTree = RleTree<u8, Constant<char>, AllowCow, 3>;

#[test]
fn auto_fuzz_1_simple_diverge_insert() {
    let mut tree_0: CowFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('A'), 78);
    tree_0.insert(1, Constant('T'), 62);
    tree_0.insert(58, Constant('A'), 1);
    tree_0.insert(45, Constant('K'), 1);
    let mut tree_1 = tree_0.shallow_clone();
    tree_0.insert(1, Constant('A'), 1);
    tree_1.insert(101, Constant('B'), 27);
    tree_0.validate();
    tree_1.validate();
    drop(tree_1);
    drop(tree_0);
}
