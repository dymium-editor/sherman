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

#[test]
fn auto_fuzz_2_override_parent_pointer_when_already_unique() {
    let mut tree_0: CowFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('I'), 78);
    tree_0.insert(2, Constant('N'), 1);
    tree_0.insert(0, Constant('B'), 58);
    let mut tree_1 = tree_0.shallow_clone();
    assert!(std::panic::catch_unwind(move || { tree_1.insert(189, Constant('B'), 1) }).is_err());
    tree_0.insert(1, Constant('B'), 1);
    tree_0.insert(1, Constant('I'), 1);
    tree_0.insert(1, Constant('B'), 3);
    tree_0.insert(58, Constant('A'), 73);
    tree_0.insert(78, Constant('B'), 36);
    let mut tree_2 = tree_0.shallow_clone();
    tree_0.insert(1, Constant('B'), 1);
    tree_2.insert(1, Constant('B'), 4);
    assert!(std::panic::catch_unwind(move || { tree_2.insert(189, Constant('H'), 72) }).is_err());
    let mut tree_3 = tree_0.shallow_clone();
    assert!(std::panic::catch_unwind(move || { tree_3.insert(189, Constant('H'), 189) }).is_err());
    assert!(std::panic::catch_unwind(move || { tree_0.insert(189, Constant('H'), 128) }).is_err());
}
