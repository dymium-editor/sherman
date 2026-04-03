//! Tests generated from fuzzing for `BasicOperation<u8, Constant<UpperLetter>>`

use crate::{Constant, RleTree};

#[test]
fn test_01_insert_middle() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('A'), 65);
    tree.insert(1, Constant('D'), 37);
}

#[test]
fn test_02_get_boundary() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('A'), 40);
    tree.insert(38, Constant('M'), 38);
    {
        let entry = tree.get(38);
        assert_eq!(entry.range(), 38..76);
        assert_eq!(entry.slice(), &Constant('M'));
    }
}

#[test]
fn test_03_validate_split_insert() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('Z'), 45);
    tree.insert(44, Constant('E'), 164);
    tree.validate_balance();
}

#[test]
fn test_04_boundary_insert_repeat_rotate_left_right() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('Q'), 37);
    tree.insert(37, Constant('L'), 37);
    tree.insert(37, Constant('A'), 38);
}

#[test]
fn test_05_split_root() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('C'), 8);
    tree.insert(2, Constant('P'), 10);
    tree.insert(10, Constant('K'), 8);
}

#[test]
fn test_06_insert_risk_overflow_root() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('G'), 85);
    tree.insert(85, Constant('H'), 85);
    tree.insert(21, Constant('A'), 1);
}

#[test]
fn test_07_insert_risk_overflow_rhs_edge() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('W'), 89);
    tree.insert(0, Constant('S'), 73);
    tree.insert(89, Constant('B'), 59);
}

#[test]
fn test_08_insert_nonempty_lhs_edge() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('B'), 6);
    tree.insert(1, Constant('T'), 167);
    tree.insert(1, Constant('A'), 44);
    {
        let entry = tree.get(0);
        assert_eq!(entry.range(), 0..1);
        assert_eq!(entry.slice(), &Constant('B'));
    }
}

#[test]
fn test_09_get_rhs_edge() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('Q'), 92);
    tree.insert(92, Constant('B'), 14);
    {
        let entry = tree.get(92);
        assert_eq!(entry.range(), 92..106);
        assert_eq!(entry.slice(), &Constant('B'));
    }
}

#[test]
fn test_10_empty_iter() {
    let tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    let _iter = tree.iter(..);
}

#[test]
fn test_11_iter_full_single_back() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('V'), 78);
    {
        let mut iter = tree.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 0..78);
            assert_eq!(item.slice(), &Constant('V'));
        }
    }
}

#[test]
fn test_12_iter_end_edge_empty() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('V'), 25);
    {
        let mut iter = tree.iter(25..);
        assert!(iter.next().is_none());
    }
}

#[test]
fn test_13_iter_start_edge_empty() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('Q'), 90);
    {
        let mut iter = tree.iter(0..0);
        assert!(iter.next_back().is_none());
    }
}

#[test]
fn test_14_basic_iter_several() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('A'), 65);
    tree.insert(1, Constant('N'), 10);
    {
        let mut iter = tree.iter(..);
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 0..1);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 1..11);
            assert_eq!(item.slice(), &Constant('N'));
        }
    }
}

#[test]
fn test_15_iter_aligned_end_boundary() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('O'), 66);
    tree.insert(0, Constant('A'), 10);
    {
        let mut iter = tree.iter(..10);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 0..10);
            assert_eq!(item.slice(), &Constant('A'));
        }
    }
}

#[test]
fn test_16_iter_back_nonempty_lhs() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('V'), 72);
    tree.insert(0, Constant('A'), 150);
    {
        let mut iter = tree.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 150..222);
            assert_eq!(item.slice(), &Constant('V'));
        }
    }
}

#[test]
fn test_17_drain_offset() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('A'), 169);
    {
        let mut drain = tree.drain(117..);
        assert_eq!(drain.next(), Some((117..169, Constant('A'))));
    }
}

#[test]
fn test_18_remove_aligned() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('B'), 14);
    tree.insert(0, Constant('M'), 142);
    _ = tree.remove(..142);
}

#[test]
fn test_19_remove_split_root() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('D'), 122);
    tree.insert(89, Constant('L'), 89);
    _ = tree.remove(89..124);
}

#[test]
fn test_20_remove_from_split_root() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('A'), 75);
    tree.insert(0, Constant('L'), 63);
    _ = tree.remove(66..);
    assert_eq!(tree.size(), 66);
}

#[test]
fn test_21_drain_mixed_directions() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('G'), 126);
    tree.insert(93, Constant('Z'), 16);
    {
        let mut drain = tree.drain(11..);
        assert_eq!(drain.next_back(), Some((109..142, Constant('G'))));
        assert_eq!(drain.next(), Some((11..93, Constant('G'))));
    }
}

#[test]
fn test_22_removal_splits_rhs_no_lhs() {
    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();

    // // Original fuzz test:
    // tree.insert(0, Constant('L'), 63);
    // tree.insert(7, Constant('H'), 119);
    // _ = tree.remove(9..9);
    // _ = tree.remove(..9);
    // tree.insert(14, Constant('A'), 9);
    // _ = tree.remove(9..47);
    // tree.insert(2, Constant('J'), 14);
    // {
    //     let mut drain = tree.drain(..);
    //     assert_eq!(drain.next_back(), Some((102..158, Constant('L'))));
    // }

    // Manually minimized version:
    tree.insert(0, Constant('L'), 63);
    tree.insert(7, Constant('H'), 119);
    _ = tree.remove(..9);
    tree.insert(14, Constant('A'), 9);
    _ = tree.remove(9..47);
    let contents = tree.iter(..).map(|e| (e.range(), e.slice())).collect::<Vec<_>>();
    assert_eq!(contents, [(0..88, &Constant('H')), (88..144, &Constant('L'))]);
}

#[test]
fn test_23_removal_recursive_imbalance() {
    // Minimized lightly from the original fuzz test, removing gratuitous calls to
    // `tree.validate_balance()`, `tree.get(...)`, zero-length `tree.drain(...)`, and eariler calls
    // to `drain.next()`.

    let mut tree: RleTree<u8, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('C'), 125);
    tree.insert(2, Constant('A'), 8);
    tree.insert(133, Constant('V'), 2);
    tree.insert(2, Constant('C'), 2);
    tree.insert(8, Constant('F'), 8);
    tree.insert(0, Constant('A'), 8);
    tree.insert(2, Constant('A'), 8);
    tree.insert(133, Constant('V'), 2);
    tree.insert(2, Constant('C'), 2);
    tree.insert(2, Constant('G'), 2);
    tree.insert(2, Constant('A'), 8);
    tree.insert(125, Constant('I'), 2);
    tree.insert(133, Constant('V'), 2);
    tree.insert(2, Constant('A'), 8);
    tree.insert(133, Constant('V'), 2);
    tree.insert(1, Constant('C'), 2);
    tree.insert(2, Constant('G'), 2);
    tree.insert(133, Constant('V'), 2);
    tree.insert(2, Constant('C'), 2);
    tree.insert(2, Constant('G'), 2);
    tree.insert(2, Constant('A'), 8);
    tree.insert(133, Constant('V'), 2);
    tree.insert(2, Constant('I'), 2);
    tree.insert(133, Constant('V'), 2);
    tree.insert(2, Constant('C'), 2);
    tree.insert(2, Constant('G'), 2);
    tree.insert(2, Constant('G'), 2);
    tree.insert(133, Constant('V'), 2);
    tree.insert(2, Constant('C'), 2);
    tree.insert(2, Constant('G'), 2);
    tree.insert(2, Constant('A'), 8);
    let _ = tree.drain(49..64);
    let _ = tree.drain(39..64);
    // At this point, the removal results in a complicated imbalance, where there are rotations
    // required on both sides of a removed subtree in order to resolve the imbalance.
    let _ = tree.drain(48..64);
    tree.validate_balance(); // <- original fuzz test failed here
}
