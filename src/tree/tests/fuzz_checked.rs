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
