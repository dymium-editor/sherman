//! Small, simple unit tests for sanity checks
//!
//! This module really isn't the main piece of testing. That's all done separately, with fuzzing.

use crate::{param::NoFeatures, Constant, RleTree};

#[test]
fn basic_insert() {
    let mut tree: RleTree<usize, Constant<char>, NoFeatures, 2> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 4);
    tree.insert(0, Constant('b'), 2);
    tree.insert(6, Constant('c'), 3);
    tree.insert(6, Constant('d'), 3);
    tree.insert(6, Constant('e'), 3);
    tree.insert(6, Constant('f'), 3);
    enable_debug!();
    tree.insert(6, Constant('g'), 3); // panics
}
