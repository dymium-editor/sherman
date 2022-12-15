use crate::{param::NoFeatures, BoundedCursor, Constant, Cursor, RleTree};

type BasicFuzzTree<const M: usize = 3> = RleTree<u8, Constant<char>, NoFeatures, M>;

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

#[test]
fn auto_fuzz_18_misc_large_insert() {
    let mut tree_0: BasicFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('B'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('R'), 5);
    tree_0.insert(0, Constant('B'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('R'), 5);
    tree_0.insert(1, Constant('E'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('N'), 5);
    tree_0.insert(7, Constant('F'), 5);
    tree_0.insert(5, Constant('E'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(4, Constant('Q'), 5);
    tree_0.insert(63, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('H'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('H'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('P'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('N'), 5);
    tree_0.insert(7, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(5, Constant('F'), 5);
    tree_0.insert(120, Constant('Q'), 80);
    tree_0.validate();
}

#[test]
#[allow(unused_variables, unused_mut)]
fn auto_fuzz_19_bubble_deferred_insert() {
    let mut tree_0: BasicFuzzTree<3> = RleTree::new_empty();
    assert_eq!(
        tree_0.insert_with_cursor::<BoundedCursor>(Cursor::new_empty(), 0, Constant('L'), 26),
        tree_0.cursor_to(0),
    );
    assert_eq!(
        tree_0.insert_with_cursor::<BoundedCursor>(Cursor::new_empty(), 1, Constant('A'), 76),
        tree_0.cursor_to(1),
    );
    assert_eq!(
        tree_0.insert_with_cursor::<BoundedCursor>(Cursor::new_empty(), 71, Constant('B'), 26),
        tree_0.cursor_to(71),
    );
    assert_eq!(
        tree_0.insert_with_cursor::<BoundedCursor>(Cursor::new_empty(), 42, Constant('V'), 1),
        tree_0.cursor_to(42),
    );
    {
        let entry = tree_0.get(26);
        assert_eq!(entry.range(), 1..42);
        assert_eq!(entry.slice(), &Constant('A'));
        assert_eq!(entry.cursor::<BoundedCursor>(), tree_0.cursor_to(26));
    }
    assert_eq!(
        tree_0.insert_with_cursor::<BoundedCursor>(Cursor::new_empty(), 73, Constant('A'), 26),
        tree_0.cursor_to(73),
    );
    assert_eq!(
        tree_0.insert_with_cursor::<BoundedCursor>(Cursor::new_empty(), 71, Constant('I'), 61),
        tree_0.cursor_to(71),
    );
    assert!(std::panic::catch_unwind(move || { tree_0.insert(0, Constant('A'), 0) }).is_err());
}
