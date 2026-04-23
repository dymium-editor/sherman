//! Tests generated from fuzzing for `MultiCowOperation<u8, Constant<UpperLetter>>`

use crate::param::EnableCow;
use crate::{Constant, RleTree};

#[test]
fn test_01_basic_bidi_retry_iter() {
    let mut tree_0: RleTree<u8, Constant<char>, EnableCow> = RleTree::new_empty();
    tree_0.insert(0, Constant('V'), 255);
    {
        let mut iter = tree_0.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 0..255);
            assert_eq!(item.slice(), &Constant('V'));
        }
        assert!(iter.next_back().is_none());
        assert!(iter.next().is_none());
    }
}

#[test]
fn test_02_into_iter_next_back() {
    let mut tree_0: RleTree<u8, Constant<char>, EnableCow> = RleTree::new_empty();
    tree_0.insert(0, Constant('A'), 87);
    tree_0.insert(82, Constant('E'), 2);
    {
        let mut iter = tree_0.into_iter();
        assert_eq!(iter.next_back(), Some((84..89, Constant('A'))));
    }
}

#[test]
fn test_03_iter_next_after_end() {
    let mut tree_0: RleTree<u8, Constant<char>, EnableCow> = RleTree::new_empty();
    tree_0.insert(0, Constant('A'), 249);
    {
        let mut iter = tree_0.iter(..);
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 0..249);
            assert_eq!(item.slice(), &Constant('A'));
        }
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }
}

#[test]
fn test_04_basic_update_nonunique() {
    let mut tree_0: RleTree<u8, Constant<char>, EnableCow> = RleTree::new_empty();
    tree_0.insert(0, Constant('A'), 87);
    tree_0.insert(17, Constant('U'), 146);
    let mut tree_1 = tree_0.shallow_clone();
    _ = tree_1.replace(..162, Constant('E'));
}

#[test]
fn test_05_multi_drop_rewrite_parent() {
    let mut tree_0: RleTree<u8, Constant<char>, EnableCow> = RleTree::new_empty();
    tree_0.insert(0, Constant('N'), 128);
    tree_0.insert(63, Constant('A'), 11);
    let tree_1 = tree_0.shallow_clone();
    let tree_2 = tree_1.shallow_clone();
    let tree_3 = tree_2.shallow_clone();
    let _ = tree_3.iter(128..128);
    {
        let entry = tree_0.get(128);
        assert_eq!(entry.range(), 74..139);
        assert_eq!(entry.slice(), &Constant('N'));
    }
    tree_0.insert(36, Constant('B'), 23);
    let tree_4 = tree_0.shallow_clone();
    {
        let mut iter = tree_1.into_iter();
        assert_eq!(iter.next(), Some((0..63, Constant('N'))));
    }
    let tree_5 = tree_0.remove(..90).into_tree();
    {
        let entry = tree_0.get(0);
        assert_eq!(entry.range(), 0..7);
        assert_eq!(entry.slice(), &Constant('A'));
    }

    // When this test case was found, it relied upon the drop ordering of the trees that were
    // created earlier -- specifically, that the fuzzing implementation in increasing order.
    // So we've manually recreated that here (+ also changed the fuzzing implementation to use a
    // reversed drop order, matching the normal behavior for variables at the end of a scope).
    drop(tree_0);
    drop(tree_2);
    drop(tree_3);
    drop(tree_4);
    // Originally failed on drop here with use-after-free, because of a parent pointer not getting
    // overwritten on downward traversal during drop.
    drop(tree_5);
}
