use crate::param::{AllowSliceRefs, NoFeatures};
use crate::{Constant, RleTree};

#[test]
fn basic_insert() {
    let mut tree: RleTree<usize, Constant<char>, NoFeatures, 2> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 4);
    tree.insert(0, Constant('b'), 2);
    tree.insert(6, Constant('c'), 3);
    tree.insert(6, Constant('d'), 3);
    tree.insert(6, Constant('e'), 3);
    tree.insert(6, Constant('f'), 3);
    tree.insert(6, Constant('g'), 3);
}

#[test]
fn medium_complex_insert() {
    let mut tree: RleTree<usize, Constant<char>, NoFeatures, 1> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 4); // [ 'a' ]
    tree.insert(4, Constant('b'), 3); // [ 'a', 'b' ]
    tree.insert(2, Constant('c'), 5); // [ [ 'a', 'c' ], 'a', [ 'b' ] ]

    let fst_expected = vec![(0_usize..2, 'a'), (2..7, 'c'), (7..9, 'a'), (9..12, 'b')];

    let fst_result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();
    assert_eq!(fst_result, fst_expected);

    tree.insert(5, Constant('d'), 1); // [ [ 'a' ], 'c', [ 'd', 'c' ], 'a', [ 'b' ] ]
    tree.validate();

    let snd_expected = vec![
        (0_usize..2, 'a'),
        (2..5, 'c'),
        (5..6, 'd'),
        (6..8, 'c'),
        (8..10, 'a'),
        (10..13, 'b'),
    ];
    let snd_result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();
    assert_eq!(snd_result, snd_expected);
}

#[test]
fn insert_and_split_with_len_1() {
    // There's a particular edge case here that we need to be able to handle -- when we try to
    // insert two values into a node with a length of 1, and that causes us to split. There's two
    // ways this can happen -- either before or after the existing value.
    let mut tree: RleTree<u8, Constant<char>, NoFeatures, 1> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 5); // [ 'a' ]

    // First split: end of the node
    tree.insert(3, Constant('b'), 4_u8); // [ [ 'a' ] 'b' [ 'a' ] ]
    tree.validate();
    let fst_result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();
    assert_eq!(fst_result, [(0..3, 'a'), (3..7, 'b'), (7..9, 'a')]);

    // Second split: start of the right-hand child node.
    tree.insert(6, Constant('c'), 2); // [ [ 'a' ] 'b' [ 'c' ] 'b' [ 'a' ] ]
    tree.validate();
    let snd_result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();
    assert_eq!(
        snd_result,
        [
            (0..3, 'a'),
            (3..6, 'b'),
            (6..8, 'c'),
            (8..9, 'b'),
            (9..11, 'a')
        ]
    );
}

#[test]
fn basic_iter() {
    let mut tree: RleTree<usize, Constant<char>, NoFeatures, 2> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 4);
    tree.insert(0, Constant('b'), 2);
    tree.insert(6, Constant('c'), 3);
    tree.insert(6, Constant('d'), 3);
    tree.insert(6, Constant('e'), 3);
    tree.insert(6, Constant('f'), 3);
    tree.insert(6, Constant('g'), 3);

    let expected = vec![
        (0_usize..2, 'b'),
        (2..6, 'a'),
        (6..9, 'g'),
        (9..12, 'f'),
        (12..15, 'e'),
        (15..18, 'd'),
        (18..21, 'c'),
    ];

    let result: Vec<_> = tree.iter(..).map(|e| (e.range(), e.slice().0)).collect();

    assert_eq!(result, expected);
}

#[test]
fn basic_slice_ref() {
    let mut tree: RleTree<u8, Constant<char>, AllowSliceRefs> = RleTree::new_empty();
    let r0 = tree.insert_ref(0, Constant('a'), 5);
    tree.insert(2, Constant('b'), 3);
    assert_eq!(r0.range(), 0..2);
    assert_eq!(&*r0.borrow_slice(), &Constant('a'));
}

#[test]
fn basic_deep_clone() {
    let mut tree: RleTree<u8, Constant<char>, NoFeatures> = RleTree::new_empty();
    tree.insert(0, Constant('a'), 4);
    tree.insert(0, Constant('b'), 2);
    tree.insert(6, Constant('c'), 3);
    tree.insert(6, Constant('d'), 3);
    tree.insert(6, Constant('e'), 3);
    tree.insert(6, Constant('f'), 3);
    tree.insert(6, Constant('g'), 3);
    tree.validate();

    let copied_tree = tree.clone();
    copied_tree.validate();
}
