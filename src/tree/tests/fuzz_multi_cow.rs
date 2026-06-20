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
    _ = tree_1.replace(..162, Constant('E'), 162);
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

#[test]
fn test_06_recursive_replace_requires_rebalance() {
    // Trimmed down from the original test, for clarity.
    let mut tree_0: RleTree<u8, Constant<char>, EnableCow> = RleTree::new_empty();
    tree_0.insert(0, Constant('J'), 255);
    _ = tree_0.replace(51.., Constant('U'), 3).into_tree();
    let mut tree_1 = tree_0.shallow_clone();
    tree_1.insert(51, Constant('C'), 104);
    _ = tree_0.replace_many(2..43, tree_1).into_tree();
    // This will panic if we do not adequately rebalance `tree_0` as part the replacement.
    tree_0.validate_balance();
}

#[test]
fn test_07_iter_across_duplicate_addrs() {
    // Trimmed down from the original test, for clarity.
    let mut tree_0: RleTree<u8, Constant<char>, EnableCow> = RleTree::new_empty();
    tree_0.insert(0, Constant('N'), 4);
    tree_0.insert(4, Constant('T'), 95);
    tree_0.insert(40, Constant('A'), 1);

    // At this point, tree_0 looks like:
    //
    //               |- A -|
    //   |- N -|           |- T -|
    //         |- T -|
    //   0     4    40    41    100
    //
    // and after replacement, we should expect a series of values like:
    //
    //   | N | T | A | N | T | A | T |
    //   0   4  40  42  46  82  83  142
    //
    // But there's some complications because we can find the same node in multiple places
    // representing different ranges.
    let tree_1 = tree_0.shallow_clone();
    _ = tree_0.replace_many(42..47, tree_1);
    {
        let mut iter = tree_0.iter(0..80);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 46..82);
            assert_eq!(item.slice(), &Constant('T'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 0..4);
            assert_eq!(item.slice(), &Constant('N'));
        }
        // The address of the node representing this 'T' is actually the same as the one
        // representing the 47..82 range, so if the iterator implementation is checking completion
        // by comparing addresses, it can prematurely terminate here.
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 4..40);
            assert_eq!(item.slice(), &Constant('T'));
        }
    }
}
