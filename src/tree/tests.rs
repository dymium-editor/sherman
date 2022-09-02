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
fn medium_complex_insert() {
    let mut tree: RleTree<usize, Constant<char>, NoFeatures, 1> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 4); // [ 'a' ]
    tree.insert(4, Constant('b'), 3); // [ 'a', 'b' ]
    tree.insert(2, Constant('c'), 5); // [ [ 'a' ], 'c', [ 'a', 'b' ] ]

    let fst_expected = vec![(0_usize..2, 'a'), (2..7, 'c'), (7..9, 'a'), (9..12, 'b')];

    let fst_result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();
    assert_eq!(fst_result, fst_expected);

    tree.insert(5, Constant('d'), 1); // [ [ 'a' ], 'c', [ 'd', 'c' ], 'a', [ 'b' ] ]
    tree.validate();

    let snd_expected = vec![
        (0_usize..2, 'a'),
        (2..5, 'c'),
        (5..6, 'd'),
        (6..8, 'c'),
        (8..10, 'a'),
        (10..13, 'b'),
    ];
    let snd_result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();
    assert_eq!(snd_result, snd_expected);
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

    let result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();

    assert_eq!(result, expected);
}

#[test]
fn auto_fuzz_2_iter_rangefrom_out_of_bounds_panic() {
    let tree_0: RleTree<usize, Constant<char>> = RleTree::new_empty();
    assert!(std::panic::catch_unwind(move || {
        let _ = tree_0.iter(738590338888761098..);
    })
    .is_err());
}

// superset of the original auto_fuzz_1
#[test]
fn auto_fuzz_3_iter_rangefull_bkwd_fwd_none() {
    let mut tree_0: RleTree<usize, Constant<char>> = RleTree::new_empty();
    tree_0.insert(0, Constant('C'), 145138940641343);
    {
        let mut iter = tree_0.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 0..145138940641343);
            assert_eq!(item.size(), 145138940641343);
            assert_eq!(item.slice(), &Constant('C'));
        }
        assert!(iter.next().is_none());
    }
}

#[test]
fn auto_fuzz_4_middle_iter() {
    let mut tree_0: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 147);
    tree_0.insert(28, Constant('C'), 28);
    {
        let mut iter = tree_0.iter(28..);
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 28..56);
            assert_eq!(item.size(), 28);
            assert_eq!(item.slice(), &Constant('C'));
        }
    }
}

#[test]
fn auto_fuzz_5_insert_nearly_full() {
    let mut tree_0: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 147);
    tree_0.insert(147, Constant('R'), 45);
    tree_0.insert(0, Constant('A'), 1);
    tree_0.insert(147, Constant('B'), 33);
}

#[test]
fn auto_fuzz_6_iter_excluded_slice_start_boundary() {
    let mut tree_0: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 147);
    tree_0.insert(147, Constant('R'), 1);
    {
        let mut iter = tree_0.iter(..147);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 0..147);
            assert_eq!(item.size(), 147);
            assert_eq!(item.slice(), &Constant('V'));
        }
    }
}

#[test]
fn auto_fuzz_7_insert_and_join_lhs() {
    let mut tree_0: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree_0.insert(0, Constant('F'), 5);
    tree_0.insert(5, Constant('H'), 1);
    tree_0.insert(5, Constant('F'), 5);
}
