use crate::{Constant, RleTree};

#[test]
fn basic_insert() {
    let mut tree: RleTree<usize, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 4);
    tree.insert(0, Constant('b'), 2);
    tree.insert(6, Constant('d'), 3);
    tree.insert(6, Constant('e'), 3);
    tree.insert(6, Constant('f'), 3);
    tree.insert(6, Constant('g'), 3);
}
