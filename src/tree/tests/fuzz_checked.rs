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

#[test]
fn test_03_repeat_insertion_point() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 10);
    tree.insert(idx, TrackedSlice(Constant('I')), size);
    // On this repeat insertion at the same index, it's easy for the tracking logic to get tripped
    // up trying to interpret things in one epoch or another.
    let (idx, size) = t.prepare_insert(0, 14);
    tree.insert(idx, TrackedSlice(Constant('F')), size);
}

#[test]
fn test_04_drain_remove_only_node_end() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 250);
    tree.insert(idx, TrackedSlice(Constant('Q')), size);
    let (start, _) = t.prepare_remove(187, 250);
    let _ = tree.drain(start..);
}

#[test]
fn test_05_insert_end_join() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 10);
    tree.insert(idx, TrackedSlice(Constant('A')), size);
    let (idx, size) = t.prepare_insert(10, 92);
    tree.insert(idx, TrackedSlice(Constant('A')), size);
}

#[test]
fn test_06_remove_within_only_node() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 94);
    tree.insert(idx, TrackedSlice(Constant('A')), size);
    let (start, end) = t.prepare_remove(81, 85);
    _ = tree.remove(start..end);
}

#[test]
fn test_07_remove_within_only_node_zero_sized() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 255);
    tree.insert(idx, TrackedSlice(Constant('V')), size);
    let (start, end) = t.prepare_remove(249, 249);
    _ = tree.remove(start..end);
    {
        let entry = tree.get(t.i(0));
        assert_eq!(entry.range(), t.i(0)..t.i(255));
        assert_eq!(entry.slice(), &TrackedSlice(Constant('V')));
    }
}

#[test]
fn test_08_insert_root_rejoin_lhs_rejoin() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 204);
    tree.insert(idx, TrackedSlice(Constant('A')), size);
    let (idx, size) = t.prepare_insert(22, 22);
    tree.insert(idx, TrackedSlice(Constant('W')), size);
    let (idx, size) = t.prepare_insert(22, 22);
    tree.insert(idx, TrackedSlice(Constant('W')), size);
    let (idx, size) = t.prepare_insert(0, 4);
    tree.insert(idx, TrackedSlice(Constant('A')), size);
}
