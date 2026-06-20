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

#[test]
fn test_09_drain_full() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 152);
    tree.insert(idx, TrackedSlice(Constant('E')), size);
    let (_, _) = t.prepare_remove(0, 152);
    {
        let mut drain = tree.drain(..);
        assert_eq!(drain.next(), Some((t.p(0)..t.p(152), TrackedSlice(Constant('E')))));
    }
}

#[test]
fn test_10_remove_within_root() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 58);
    tree.insert(idx, TrackedSlice(Constant('T')), size);
    let (idx, size) = t.prepare_insert(0, 39);
    tree.insert(idx, TrackedSlice(Constant('A')), size);
    let (range, size) = t.prepare_replace(47..70, 23);
    _ = tree.replace(range, TrackedSlice(Constant('A')), size);
}

#[test]
fn test_11_replace_underflow() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    #[allow(clippy::reversed_empty_ranges)]
    let (range, size) = t.prepare_replace(107..0, 0); // <- must be careful to avoid underflow
    assert!(
        std::panic::catch_unwind(move || tree.replace(
            range.start..,
            TrackedSlice(Constant('A')),
            size
        ))
        .is_err()
    );
}

#[test]
fn test_12_replace_overflow() {
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(160, 205); // <- must be careful to avoid overflow
    assert!(
        std::panic::catch_unwind(move || tree.insert(idx, TrackedSlice(Constant('X')), size))
            .is_err()
    );
}

#[test]
fn test_13_removal_epoch() {
    // Minimized lightly from the original fuzz test.
    let t: IndexInfo<u8> = IndexInfo::new();
    let mut tree: RleTree<TrackedIndex<u8>, TrackedSlice<Constant<char>>> = RleTree::new_empty();
    let (idx, size) = t.prepare_insert(0, 45);
    tree.insert(idx, TrackedSlice(Constant('J')), size);
    let (idx, size) = t.prepare_insert(43, 43);
    tree.insert(idx, TrackedSlice(Constant('R')), size);
    let (idx, size) = t.prepare_insert(43, 43);
    tree.insert(idx, TrackedSlice(Constant('R')), size);
    let (range, size) = t.prepare_replace(35..116, 43);
    _ = tree.replace(range.start..range.end, TrackedSlice(Constant('T')), size);
    let (range, size) = t.prepare_replace(35..43, 43);
    _ = tree.replace(range.start..range.end, TrackedSlice(Constant('R')), size);
    let (idx, size) = t.prepare_insert(35, 43);
    tree.insert(idx, TrackedSlice(Constant('H')), size);
    let (range, size) = t.prepare_replace(35..35, 16);
    _ = tree.replace(range.start..range.end, TrackedSlice(Constant('T')), size);
    let (start, end) = t.prepare_remove(35, 43);
    _ = tree.remove(start..end);
    let (start, end) = t.prepare_remove(35, 43);
    _ = tree.remove(start..end);
    let (range, size) = t.prepare_replace(43..43, 43);
    // The error surfaced by fuzzing originally had issues on this replacement, where translating
    // the tracked indexes requires rewinding an index 43 back past the 35..43 removal.
    _ = tree.replace(range.start..range.end, TrackedSlice(Constant('R')), size);
}
