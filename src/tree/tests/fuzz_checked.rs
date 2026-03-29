use crate::fuzz::{IndexInfo, TrackedIndex, TrackedSlice};
use crate::{Constant, RleTree};

#[test]
fn test_01_basic_insert() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 135);
    tree.insert(idx, TrackedSlice(Constant('A')), size);
    let (idx, size) = t.prepare_insert(94, 94);
    tree.insert(idx, TrackedSlice(Constant('Y')), size);
}

#[test]
fn test_02_panic_caught() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    // If `TrackedIndex` stores the end point, we can end up overflowing outside the `catch_unwind`
    // rather than inside it.
    let (idx, size) = t.prepare_insert(137, 152);
    assert!(
        std::panic::catch_unwind(move || tree.insert(idx, TrackedSlice(Constant('P')), size))
            .is_err()
    );
}
