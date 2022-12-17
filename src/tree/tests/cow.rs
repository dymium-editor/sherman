use crate::{param::AllowCow, Constant, RleTree};

type CowFuzzTree<const M: usize = 3> = RleTree<u8, Constant<char>, AllowCow, M>;

#[test]
fn auto_fuzz_1_simple_diverge_insert() {
    let mut tree_0: CowFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('A'), 78);
    tree_0.insert(1, Constant('T'), 62);
    tree_0.insert(58, Constant('A'), 1);
    tree_0.insert(45, Constant('K'), 1);
    let mut tree_1 = tree_0.clone();
    tree_0.insert(1, Constant('A'), 1);
    tree_1.insert(101, Constant('B'), 27);
    tree_0.validate();
    tree_1.validate();
    drop(tree_1);
    drop(tree_0);
}

#[test]
fn auto_fuzz_2_override_parent_pointer_when_already_unique() {
    let mut tree_0: CowFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('I'), 78);
    tree_0.insert(2, Constant('N'), 1);
    tree_0.insert(0, Constant('B'), 58);
    let mut tree_1 = tree_0.clone();
    assert!(std::panic::catch_unwind(move || { tree_1.insert(189, Constant('B'), 1) }).is_err());
    tree_0.insert(1, Constant('B'), 1);
    tree_0.insert(1, Constant('I'), 1);
    tree_0.insert(1, Constant('B'), 3);
    tree_0.insert(58, Constant('A'), 73);
    tree_0.insert(78, Constant('B'), 36);
    let mut tree_2 = tree_0.clone();
    tree_0.insert(1, Constant('B'), 1);
    tree_2.insert(1, Constant('B'), 4);
    assert!(std::panic::catch_unwind(move || { tree_2.insert(189, Constant('H'), 72) }).is_err());
    let mut tree_3 = tree_0.clone();
    assert!(std::panic::catch_unwind(move || { tree_3.insert(189, Constant('H'), 189) }).is_err());
    assert!(std::panic::catch_unwind(move || { tree_0.insert(189, Constant('H'), 128) }).is_err());
}

// This test was auto-generated by the fuzzer and is a *little* bit too long, tbh
//
// We cleaned it up a bit before adding it because the additional test coverage we get from it is
// good. We verified that the cleaned up test still triggered the previously incorrect behavior.
#[test]
#[allow(unused_mut)]
fn auto_fuzz_3_iter_over_child_with_bad_parent_index() {
    let mut tree_0: CowFuzzTree = RleTree::new_empty();
    tree_0.insert(0, Constant('A'), 78);
    tree_0.insert(1, Constant('T'), 65);
    tree_0.insert(58, Constant('A'), 1);
    tree_0.insert(48, Constant('F'), 53);
    tree_0.insert(49, Constant('C'), 49);
    tree_0.insert(180, Constant('O'), 1);
    {
        let mut iter = tree_0.iter(49..);
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 49..98);
            assert_eq!(item.size(), 49);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 98..150);
            assert_eq!(item.size(), 52);
            assert_eq!(item.slice(), &Constant('F'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 150..160);
            assert_eq!(item.size(), 10);
            assert_eq!(item.slice(), &Constant('T'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 160..161);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 181..247);
            assert_eq!(item.size(), 66);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 180..181);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('O'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 169..180);
            assert_eq!(item.size(), 11);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 161..169);
            assert_eq!(item.size(), 8);
            assert_eq!(item.slice(), &Constant('T'));
        }
        assert!(iter.next_back().is_none());
        assert!(iter.next().is_none());
    }
    {
        let mut iter = tree_0.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 181..247);
            assert_eq!(item.size(), 66);
            assert_eq!(item.slice(), &Constant('A'));
        }
    }
    let mut tree_1 = tree_0.clone();
    assert!(std::panic::catch_unwind(move || { tree_0.insert(170, Constant('O'), 170) }).is_err());
    let mut tree_2 = tree_1.clone();
    let mut tree_3 = tree_1.clone();
    let mut tree_4 = tree_3.clone();
    let mut tree_5 = tree_3.clone();
    drop(tree_1);
    assert!(std::panic::catch_unwind(move || { tree_2.insert(249, Constant('O'), 170) }).is_err());
    drop(tree_4);
    let mut tree_6 = tree_5.clone();
    let mut tree_7 = tree_6.clone();
    tree_5.insert(1, Constant('B'), 1);
    tree_5.insert(1, Constant('B'), 1);
    tree_5.insert(1, Constant('B'), 1);
    tree_5.insert(1, Constant('B'), 1);
    assert!(std::panic::catch_unwind(move || { tree_5.insert(1, Constant('A'), 0) }).is_err());
    assert!(std::panic::catch_unwind(move || { tree_3.insert(1, Constant('T'), 65) }).is_err());
    let mut tree_8 = tree_7.clone();
    assert!(std::panic::catch_unwind(move || { tree_6.insert(1, Constant('W'), 57) }).is_err());
    assert!(std::panic::catch_unwind(move || { tree_8.insert(49, Constant('C'), 49) }).is_err());
    tree_7.insert(78, Constant('Y'), 1);
    {
        let mut iter = tree_7.iter(49..);
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 49..78);
            assert_eq!(item.size(), 29);
            assert_eq!(item.slice(), &Constant('C'));
        }
    }
    let mut tree_9 = tree_7.clone();
    assert!(std::panic::catch_unwind(move || { tree_9.insert(1, Constant('O'), 170) }).is_err());
    let mut tree_10 = tree_7.clone();
    let mut tree_11 = tree_7.clone();
    let mut tree_12 = tree_11.clone();
    let mut tree_13 = tree_7.clone();
    let mut tree_14 = tree_11.clone();
    let mut tree_15 = tree_11.clone();
    let mut tree_16 = tree_15.clone();
    drop(tree_13);
    tree_11.insert(187, Constant('A'), 1);
    assert!(std::panic::catch_unwind(move || { tree_10.insert(170, Constant('O'), 170) }).is_err());
    let mut tree_17 = tree_14.clone();
    assert!(std::panic::catch_unwind(move || {
        let _ = tree_7.iter(11..=0);
    })
    .is_err());
    {
        let mut iter = tree_11.iter(1..=1);
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 1..48);
            assert_eq!(item.size(), 47);
            assert_eq!(item.slice(), &Constant('T'));
        }
        assert!(iter.next().is_none());
    }
    {
        let mut iter = tree_16.iter(60..);
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 49..78);
            assert_eq!(item.size(), 29);
            assert_eq!(item.slice(), &Constant('C'));
        }
    }
    drop(tree_17);
    drop(tree_11);
    {
        let mut iter = tree_12.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 182..248);
            assert_eq!(item.size(), 66);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 181..182);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('O'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 170..181);
            assert_eq!(item.size(), 11);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 162..170);
            assert_eq!(item.size(), 8);
            assert_eq!(item.slice(), &Constant('T'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 161..162);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 151..161);
            assert_eq!(item.size(), 10);
            assert_eq!(item.slice(), &Constant('T'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 99..151);
            assert_eq!(item.size(), 52);
            assert_eq!(item.slice(), &Constant('F'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 79..99);
            assert_eq!(item.size(), 20);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 78..79);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('Y'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 49..78);
            assert_eq!(item.size(), 29);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 48..49);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('F'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 1..48);
            assert_eq!(item.size(), 47);
            assert_eq!(item.slice(), &Constant('T'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 0..1);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('A'));
        }
        assert!(iter.next_back().is_none());
        assert!(iter.next().is_none());
    }
    {
        let mut iter = tree_12.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 182..248);
            assert_eq!(item.size(), 66);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 181..182);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('O'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 170..181);
            assert_eq!(item.size(), 11);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 162..170);
            assert_eq!(item.size(), 8);
            assert_eq!(item.slice(), &Constant('T'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 161..162);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('A'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 151..161);
            assert_eq!(item.size(), 10);
            assert_eq!(item.slice(), &Constant('T'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 99..151);
            assert_eq!(item.size(), 52);
            assert_eq!(item.slice(), &Constant('F'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 79..99);
            assert_eq!(item.size(), 20);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 78..79);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('Y'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 49..78);
            assert_eq!(item.size(), 29);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 48..49);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('F'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 0..1);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('A'));
        }
    }
}

#[test]
#[allow(unused_variables, unused_mut)]
fn auto_fuzz_4_iter_inner_child_good_parent_bad_idx() {
    let mut tree_0: CowFuzzTree<3> = RleTree::new_empty();
    tree_0.insert(0, Constant('D'), 16);
    tree_0.insert(2, Constant('C'), 134);
    tree_0.insert(39, Constant('H'), 2);
    tree_0.insert(3, Constant('M'), 6);
    tree_0.insert(7, Constant('H'), 7);
    tree_0.insert(38, Constant('W'), 7);
    tree_0.insert(168, Constant('H'), 7);
    {
        let entry = tree_0.get(0);
        assert_eq!(entry.range(), 0..2);
        assert_eq!(entry.slice(), &Constant('D'));
    }
    tree_0.insert(37, Constant('H'), 2);
    let mut tree_1 = tree_0.clone();
    {
        let entry = tree_0.get(3);
        assert_eq!(entry.range(), 3..7);
        assert_eq!(entry.slice(), &Constant('M'));
    }
    tree_0.insert(2, Constant('C'), 50);
    {
        let entry = tree_0.get(7);
        assert_eq!(entry.range(), 2..53);
        assert_eq!(entry.slice(), &Constant('C'));
    }
    tree_0.insert(3, Constant('M'), 6);
    tree_1.insert(7, Constant('H'), 7);
    let mut tree_2 = tree_0.clone();
    {
        let mut iter = tree_0.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 233..237);
            assert_eq!(item.size(), 4);
            assert_eq!(item.slice(), &Constant('D'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 226..233);
            assert_eq!(item.size(), 7);
            assert_eq!(item.slice(), &Constant('H'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 0..2);
            assert_eq!(item.size(), 2);
            assert_eq!(item.slice(), &Constant('D'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 2..3);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('C'));
        }
    }
    assert!(std::panic::catch_unwind(move || { tree_0.insert(7, Constant('H'), 38) }).is_err());
    tree_1.insert(7, Constant('M'), 7);
    {
        let entry = tree_2.get(0);
        assert_eq!(entry.range(), 0..2);
        assert_eq!(entry.slice(), &Constant('D'));
    }
    tree_2.insert(7, Constant('C'), 3);
    {
        let entry = tree_1.get(3);
        assert_eq!(entry.range(), 3..14);
        assert_eq!(entry.slice(), &Constant('M'));
    }
    tree_1.insert(2, Constant('C'), 50);
    {
        let entry = tree_1.get(7);
        assert_eq!(entry.range(), 2..53);
        assert_eq!(entry.slice(), &Constant('C'));
    }
    tree_1.insert(3, Constant('M'), 6);
    assert!(std::panic::catch_unwind(move || { tree_2.insert(255, Constant('V'), 255) }).is_err());
    {
        let mut iter = tree_1.iter(..);
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 247..251);
            assert_eq!(item.size(), 4);
            assert_eq!(item.slice(), &Constant('D'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 240..247);
            assert_eq!(item.size(), 7);
            assert_eq!(item.slice(), &Constant('H'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 230..240);
            assert_eq!(item.size(), 10);
            assert_eq!(item.slice(), &Constant('D'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 133..230);
            assert_eq!(item.size(), 97);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 131..133);
            assert_eq!(item.size(), 2);
            assert_eq!(item.slice(), &Constant('H'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 117..131);
            assert_eq!(item.size(), 14);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 110..117);
            assert_eq!(item.size(), 7);
            assert_eq!(item.slice(), &Constant('W'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 0..2);
            assert_eq!(item.size(), 2);
            assert_eq!(item.slice(), &Constant('D'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 109..110);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 107..109);
            assert_eq!(item.size(), 2);
            assert_eq!(item.slice(), &Constant('H'));
        }
        {
            let item = iter.next().unwrap();
            assert_eq!(item.range(), 2..3);
            assert_eq!(item.size(), 1);
            assert_eq!(item.slice(), &Constant('C'));
        }
        {
            let item = iter.next_back().unwrap();
            assert_eq!(item.range(), 86..107); // <- problem!
            assert_eq!(item.size(), 21);
            assert_eq!(item.slice(), &Constant('C'));
        }
    }
}
