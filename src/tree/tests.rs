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
    tree.insert(2, Constant('c'), 5); // [ [ 'a', 'c' ], 'a', [ 'b' ] ]

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
fn insert_and_split_with_len_1() {
    // There's a particular edge case here that we need to be able to handle -- when we try to
    // insert two values into a node with a length of 1, and that causes us to split. There's two
    // ways this can happen -- either before or after the existing value.
    let mut tree: RleTree<u8, Constant<char>, NoFeatures, 1> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 5); // [ 'a' ]

    // First split: end of the node
    tree.insert(3, Constant('b'), 4_u8); // [ [ 'a' ] 'b' [ 'a' ] ]
    tree.validate();
    let fst_result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();
    assert_eq!(fst_result, [(0..3, 'a'), (3..7, 'b'), (7..9, 'a')]);

    // Second split: start of the right-hand child node.
    tree.insert(6, Constant('c'), 2); // [ [ 'a' ] 'b' [ 'c' ] 'b' [ 'a' ] ]
    tree.validate();
    let snd_result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();
    assert_eq!(
        snd_result,
        [
            (0..3, 'a'),
            (3..6, 'b'),
            (6..8, 'c'),
            (8..9, 'b'),
            (9..11, 'a')
        ]
    );
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

type BasicFuzzTree = RleTree<u8, Constant<char>, NoFeatures, 3>;

#[test]
fn auto_fuzz_2_iter_rangefrom_out_of_bounds_panic() {
    let tree_0: BasicFuzzTree = RleTree::new_empty();
    assert!(std::panic::catch_unwind(move || {
        let _ = tree_0.iter(39..);
    })
    .is_err());
}

// superset of the original auto_fuzz_1
#[test]
fn auto_fuzz_3_iter_rangefull_bkwd_fwd_none() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('C'), 217);
    {
        let mut iter = tree_0.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 0..217);
            assert_eq!(item.size(), 217);
            assert_eq!(item.slice(), &Constant('C'));
        }
        assert!(iter.next().is_none());
    }
}

#[test]
fn auto_fuzz_4_middle_iter() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
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
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 147);
    tree_0.insert(147, Constant('R'), 45);
    tree_0.insert(0, Constant('A'), 1);
    tree_0.insert(147, Constant('B'), 33);
}

#[test]
fn auto_fuzz_6_iter_excluded_slice_start_boundary() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
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
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('F'), 5);
    tree_0.insert(5, Constant('H'), 1);
    tree_0.insert(5, Constant('F'), 5);
}

#[test]
fn auto_fuzz_8_split_key_causes_split_node() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 147);
    tree_0.insert(24, Constant('C'), 28);
    tree_0.insert(45, Constant('R'), 2);
    tree_0.insert(1, Constant('R'), 1);
    {
        let mut iter = tree_0.iter(0..=147);
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 0..1);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('V'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 1..2);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('R'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 55..178);
            assert_eq!(item.size(), 123);
            assert_eq!(item.slice(), &Constant('V'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 2..25);
            assert_eq!(item.size(), 23);
            assert_eq!(item.slice(), &Constant('V'));
        }
    }
}

#[test]
fn auto_fuzz_9_misc_insert() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 147);
    tree_0.insert(26, Constant('F'), 49);
    tree_0.insert(147, Constant('R'), 2);
    tree_0.insert(41, Constant('A'), 5);
    tree_0.insert(5, Constant('L'), 5);
    tree_0.insert(5, Constant('A'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('R'), 5);
    tree_0.insert(5, Constant('F'), 5);
    {
        let mut iter = tree_0.iter(0..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 184..233);
            assert_eq!(item.size(), 49);
            assert_eq!(item.slice(), &Constant('V'));
        }
    }
    {
        let _iter = tree_0.iter(0..=0);
    }
}

#[test]
fn auto_fuzz_10_iter_back_into_child() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 147);
    tree_0.insert(24, Constant('C'), 28);
    tree_0.insert(45, Constant('R'), 2);
    tree_0.insert(1, Constant('R'), 1);
    {
        let mut iter = tree_0.iter(..45);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 25..46);
            assert_eq!(item.size(), 21);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 2..25);
            assert_eq!(item.size(), 23);
            assert_eq!(item.slice(), &Constant('V'));
        }
    }
}

#[test]
fn auto_fuzz_11_bubble_bubble_middle() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 147);
    tree_0.insert(147, Constant('R'), 2);
    tree_0.insert(41, Constant('A'), 5);
    tree_0.insert(67, Constant('L'), 5);
    tree_0.insert(0, Constant('N'), 5);
    tree_0.insert(10, Constant('L'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(64, Constant('N'), 5);
    tree_0.insert(66, Constant('Y'), 5);
    tree_0.insert(76, Constant('S'), 5);
    tree_0.insert(147, Constant('R'), 2);
    tree_0.insert(41, Constant('A'), 5);
    tree_0.insert(4, Constant('L'), 5);
    tree_0.insert(21, Constant('F'), 5);
    tree_0.insert(5, Constant('R'), 2);
    tree_0.insert(41, Constant('A'), 5);
    tree_0.insert(67, Constant('L'), 5);
    tree_0.insert(0, Constant('N'), 5);
    tree_0.insert(10, Constant('L'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('N'), 5);
    tree_0.insert(66, Constant('Y'), 5);
    tree_0.insert(76, Constant('S'), 5);
}

#[test]
fn auto_fuzz_12_bubble_first_child() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 147);
    tree_0.insert(147, Constant('R'), 2);
    tree_0.insert(41, Constant('A'), 5);
    tree_0.insert(67, Constant('L'), 5);
    tree_0.insert(0, Constant('N'), 5);
    tree_0.insert(10, Constant('L'), 5);
    tree_0.insert(0, Constant('N'), 5);
    tree_0.insert(10, Constant('L'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('N'), 5);
    tree_0.insert(66, Constant('Y'), 5);
    tree_0.insert(76, Constant('S'), 5);
    tree_0.insert(147, Constant('R'), 2);
    tree_0.insert(41, Constant('A'), 5);
    tree_0.insert(4, Constant('L'), 5);
    tree_0.insert(147, Constant('R'), 5);
    tree_0.insert(5, Constant('R'), 2);
    tree_0.insert(41, Constant('A'), 5);
    tree_0.insert(67, Constant('L'), 5);
    tree_0.insert(0, Constant('M'), 5);
    tree_0.insert(10, Constant('L'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('N'), 5);
    tree_0.insert(66, Constant('Y'), 5);
    assert!(std::panic::catch_unwind(move || { tree_0.insert(76, Constant('S'), 5) }).is_err());
}

#[test]
fn auto_fuzz_13_bubble_bubble_middle() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('J'), 63);
    tree_0.insert(0, Constant('G'), 58);
    tree_0.insert(33, Constant('L'), 33);
    tree_0.insert(52, Constant('B'), 1);
    tree_0.insert(63, Constant('H'), 5);
    tree_0.insert(21, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(0, Constant('G'), 5);
    tree_0.insert(133, Constant('A'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('N'), 5);
    tree_0.insert(2, Constant('H'), 5);
    tree_0.insert(21, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(0, Constant('G'), 5);
    tree_0.insert(133, Constant('A'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('N'), 5);
    tree_0.insert(2, Constant('Z'), 5);
    tree_0.insert(63, Constant('G'), 5);
    tree_0.insert(69, Constant('A'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(0, Constant('G'), 5);
    tree_0.insert(69, Constant('A'), 5);
    assert!(std::panic::catch_unwind(move || { tree_0.insert(5, Constant('F'), 5) }).is_err());
}

#[test]
fn auto_fuzz_14_bubble_bubble_leftmost_key() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('A'), 63);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(13, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(2, Constant('F'), 33);
    tree_0.insert(63, Constant('L'), 33);
    tree_0.insert(52, Constant('B'), 1);
    tree_0.insert(63, Constant('F'), 5);
    tree_0.insert(21, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(0, Constant('G'), 5);
    tree_0.insert(69, Constant('A'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(2, Constant('Z'), 5);
    tree_0.insert(63, Constant('G'), 5);
    tree_0.insert(69, Constant('A'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(1, Constant('F'), 5);
    tree_0.insert(2, Constant('Y'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('P'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('C'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 2);
    assert!(std::panic::catch_unwind(move || { tree_0.insert(5, Constant('F'), 63) }).is_err());
}

#[test]
fn auto_fuzz_15_misc_large_insert() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('A'), 63);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(13, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(2, Constant('F'), 33);
    tree_0.insert(63, Constant('L'), 33);
    tree_0.insert(52, Constant('B'), 1);
    tree_0.insert(63, Constant('F'), 5);
    tree_0.insert(21, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(0, Constant('G'), 5);
    tree_0.insert(69, Constant('A'), 5);
    tree_0.insert(5, Constant('R'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(2, Constant('Z'), 5);
    tree_0.insert(63, Constant('G'), 5);
    tree_0.insert(69, Constant('A'), 5);
    tree_0.insert(5, Constant('R'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(2, Constant('Z'), 5);
    tree_0.insert(63, Constant('G'), 5);
    tree_0.insert(69, Constant('A'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(178, Constant('F'), 5);
    assert!(std::panic::catch_unwind(move || { tree_0.insert(150, Constant('F'), 0) }).is_err());
}

#[test]
fn auto_fuzz_16_bubble_bubble_leftmost_rhs() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('Y'), 77);
    tree_0.insert(3, Constant('S'), 3);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(3, Constant('J'), 3);
    tree_0.insert(1, Constant('D'), 3);
    tree_0.insert(35, Constant('D'), 3);
    tree_0.insert(49, Constant('D'), 3);
    tree_0.insert(3, Constant('D'), 6);
    tree_0.insert(17, Constant('R'), 21);
    tree_0.insert(21, Constant('V'), 21);
    tree_0.insert(21, Constant('V'), 21);
    tree_0.insert(21, Constant('O'), 3);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(1, Constant('D'), 3);
    tree_0.insert(35, Constant('D'), 3);
    tree_0.insert(35, Constant('X'), 3);
    tree_0.insert(49, Constant('G'), 15);
    tree_0.insert(17, Constant('F'), 21);
    tree_0.insert(21, Constant('V'), 3);
    tree_0.insert(131, Constant('D'), 3);
    tree_0.insert(1, Constant('D'), 3);
    tree_0.insert(35, Constant('D'), 3);
    tree_0.insert(35, Constant('D'), 3);
    tree_0.insert(49, Constant('D'), 3);
    tree_0.insert(3, Constant('H'), 3);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(131, Constant('D'), 3);
    assert!(std::panic::catch_unwind(move || { tree_0.insert(35, Constant('Z'), 182) }).is_err());
}

#[test]
fn auto_fuzz_17_misc_large_insert() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('Y'), 77);
    tree_0.insert(0, Constant('U'), 3);
    tree_0.insert(49, Constant('D'), 3);
    tree_0.insert(3, Constant('H'), 3);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(33, Constant('D'), 3);
    tree_0.insert(3, Constant('B'), 3);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(3, Constant('J'), 3);
    tree_0.insert(49, Constant('D'), 3);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(33, Constant('D'), 3);
    tree_0.insert(3, Constant('B'), 3);
    tree_0.insert(3, Constant('J'), 3);
    tree_0.insert(3, Constant('X'), 3);
    tree_0.insert(49, Constant('D'), 3);
    tree_0.insert(17, Constant('R'), 21);
    tree_0.insert(21, Constant('V'), 21);
    tree_0.insert(21, Constant('V'), 21);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(33, Constant('D'), 1);
    tree_0.insert(3, Constant('U'), 3);
    tree_0.insert(3, Constant('J'), 3);
    tree_0.insert(3, Constant('X'), 3);
    tree_0.insert(81, Constant('A'), 3);
    tree_0.insert(3, Constant('D'), 3);
    tree_0.insert(3, Constant('J'), 3);
    tree_0.insert(65, Constant('D'), 3);
    tree_0.insert(40, Constant('O'), 3);
    tree_0.insert(49, Constant('D'), 3);
    tree_0.insert(3, Constant('D'), 6);
    assert!(std::panic::catch_unwind(move || { tree_0.insert(17, Constant('R'), 77) }).is_err());
}
