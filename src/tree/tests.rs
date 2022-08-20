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
    tree.insert(6, Constant('g'), 3);
}

#[test]
fn basic_iter() {
    let mut tree: RleTree<usize, Constant<char>, NoFeatures, 2> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 4);
    tree.insert(0, Constant('b'), 2);
    tree.insert(6, Constant('c'), 3);
    tree.insert(6, Constant('d'), 3);
    tree.insert(6, Constant('e'), 3);
    tree.insert(6, Constant('f'), 3);
    tree.insert(6, Constant('g'), 3);

    let expected = vec![
        (0_usize..2, 'b'),
        (2..6, 'a'),
        (6..9, 'g'),
        (9..12, 'f'),
        (12..15, 'e'),
        (15..18, 'd'),
        (18..21, 'c'),
    ];

    enable_debug!();
    let result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();

    assert_eq!(result, expected);
}
