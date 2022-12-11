use crate::BoundedCursor;
use crate::{param::AllowSliceRefs, Constant, RleTree};

type SliceRefFuzzTree = RleTree<u8, Constant<char>, AllowSliceRefs, 3>;

#[test]
#[allow(unused_variables)]
fn auto_fuzz_1_basic_clone() {
    let mut tree_0: SliceRefFuzzTree = RleTree::new_empty();
    let ref_0 = tree_0.insert_ref(0, Constant('W'), 40);
    let ref_1 = ref_0.clone();
    assert_eq!(&*ref_0.borrow_slice(), &Constant('W'));
}

#[test]
#[allow(unused_variables)]
fn auto_fuzz_2_insert_panics() {
    let mut tree_0: SliceRefFuzzTree = RleTree::new_empty();
    let ref_0 = tree_0.insert_ref(0, Constant('X'), 75);
    assert!(
        std::panic::catch_unwind(move || { tree_0.insert_ref(75, Constant('F'), 245) }).is_err()
    );
}

#[test]
#[allow(unused_variables)]
fn auto_fuzz_3_insert_deferred_after_split() {
    let mut tree_0: SliceRefFuzzTree = RleTree::new_empty();
    let ref_0 = tree_0.insert_ref(0, Constant('A'), 69);
    assert_eq!(ref_0.range(), 0..69);
    assert_eq!(ref_0.range(), 0..69);
    let ref_1 = tree_0.insert_ref(12, Constant('M'), 12);
    let ref_2 = tree_0.insert_ref(49, Constant('M'), 12);
    let ref_3 = tree_0.insert_ref(12, Constant('L'), 51);
    assert_eq!(ref_3.range(), 12..63);
    assert_eq!(ref_0.range(), 0..12);
    let ref_4 = tree_0.insert_ref(12, Constant('V'), 12);
    let ref_5 = tree_0.insert_ref(51, Constant('M'), 12);
    let ref_6 = tree_0.insert_ref(47, Constant('M'), 12);
    assert!(std::panic::catch_unwind(move || { tree_0.insert_ref(0, Constant('A'), 0) }).is_err());
}

#[test]
#[allow(unused_variables)]
fn auto_fuzz_4_generic_entry_cursor() {
    let mut tree_0: SliceRefFuzzTree = RleTree::new_empty();
    let ref_0 = tree_0.insert_ref(0, Constant('L'), 98);
    assert!(std::panic::catch_unwind(|| { tree_0.get(248).make_ref() }).is_err());
    assert!(std::panic::catch_unwind(|| { tree_0.get(226).make_ref() }).is_err());
    let ref_1 = tree_0.insert_ref(17, Constant('V'), 21);
    let ref_2 = tree_0.insert_ref(84, Constant('R'), 21);
    let ref_3 = tree_0.insert_ref(21, Constant('L'), 98);
    {
        let entry = tree_0.get(63);
        assert_eq!(entry.range(), 21..119);
        assert_eq!(entry.slice(), &Constant('L'));
        assert_eq!(entry.cursor::<BoundedCursor>(), tree_0.cursor_to(63));
    }
}
