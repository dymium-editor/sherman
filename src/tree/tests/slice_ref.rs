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
