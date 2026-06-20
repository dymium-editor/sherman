use crate::RleTree;
use crate::fuzz::{CharRange, IndexInfo, TrackedIndex, TrackedSlice};

#[test]
fn test_01_insert_once_remove_full() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<CharRange>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 229);
    tree.insert(idx, TrackedSlice(CharRange('A'..'C')), size);
    let (start, _) = t.prepare_remove(0, 229);
    _ = tree.remove(start..);
}
