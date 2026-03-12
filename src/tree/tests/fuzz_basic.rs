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
    assert!(std::panic::catch_unwind(move || tree.insert(0, Constant('A'), 0)).is_err());
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
