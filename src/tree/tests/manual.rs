use crate::{Constant, RleTree, param};

#[test]
fn basic_insert() {
    let mut tree: RleTree<usize, Constant<char>> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 4);
    tree.insert(0, Constant('b'), 2);
    tree.insert(6, Constant('d'), 3);
    tree.insert(6, Constant('e'), 3);
    tree.insert(6, Constant('f'), 3);
    tree.insert(6, Constant('g'), 3);
}

// Roughly a copy of the `StableRef` docs.
#[test]
fn varied_stable_ref() {
    let mut tree: RleTree<usize, Constant<char>, param::EnableRefs> = RleTree::new_empty();
    tree.insert(0, Constant('A'), 5);

    let ref_a = tree.get(3).stable_ref();
    assert_eq!(ref_a.range(&tree), Some(0..5));
    assert_eq!(ref_a.slice(&tree), Some(&Constant('A')));

    tree.insert(0, Constant('B'), 4);
    tree.insert(0, Constant('A'), 3);
    assert_eq!(ref_a.range(&tree), Some(7..12));

    tree.insert(10, Constant('C'), 2);
    assert_eq!(ref_a.range(&tree), Some(7..10));
    assert_eq!(tree.get(13).range(), 12..14);
    assert_eq!(tree.get(13).slice(), &Constant('A'));

    assert_eq!(
        tree.iter(..10).map(|e| (e.range(), e.slice())).collect::<Vec<_>>(),
        [
            (0..3, &Constant('A')),
            (3..7, &Constant('B')),
            (7..10, &Constant('A')),
        ],
    );
    _ = tree.remove(3..7);
    assert_eq!(ref_a.range(&tree), Some(0..6));
    assert_eq!(
        tree.iter(..).map(|e| (e.range(), e.slice())).collect::<Vec<_>>(),
        [
            (0..6, &Constant('A')),
            (6..8, &Constant('C')),
            (8..10, &Constant('A')),
        ],
    );
    _ = tree.remove(6..8);
    assert_eq!(ref_a.range(&tree), Some(0..8));
    assert_eq!(tree.size(), 8);

    let rm = tree.remove(..3);
    assert!(ref_a.range(&tree).is_none());
    let rm_tree = rm.into_tree();
    assert_eq!(ref_a.range(&rm_tree), Some(0..3));
}

// nightly-only test to confirm that `may_dangle` works as advertised
#[test]
#[cfg(feature = "nightly")]
fn may_dangle() {
    // Loosely adapted from https://doc.rust-lang.org/nomicon/dropck.html
    struct Pair<'a> {
        tree: Option<RleTree<usize, Constant<&'a str>>>,
        string: String,
    }

    let mut pair = Pair { tree: None, string: "test string".to_owned() };

    // Without #[may_dangle], the drop checker cannot guarantee that `tree` doesn't access `string`
    // when `p` is dropped, and so `tree` would have to strictly outlive the data it borrows.
    //
    // With #[may_dangle], there is a guarantee from us that the destructor for `RleTree` doesn't
    // access `string`, so we are able to put it in the same struct, dropped at the same time.
    let mut t: RleTree<usize, Constant<&str>> = RleTree::new_empty();
    t.insert(0, Constant(&pair.string), 1);
    pair.tree = Some(t);
}
