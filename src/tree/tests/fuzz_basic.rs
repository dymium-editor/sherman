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
