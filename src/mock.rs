//! Mock implementation with essentially the same API as [`RleTree`], but backed by a vector
//!
//! [`RleTree`]: crate::RleTree

use crate::range::{EndBound, RangeBounds, StartBound};
use crate::{Index, Slice};
use std::mem;
use std::ops::Range;

/// A mock, inefficient implementation of the [`RleTree`](crate::RleTree) interface
#[derive(Debug, Clone)]
pub struct Mock<I, S> {
    // list of end positions and the slices in the run
    runs: Vec<(I, S)>,
}

impl<I: Index, S: Slice<I>> Mock<I, S> {
    pub fn new_empty() -> Self {
        Mock { runs: Vec::new() }
    }

    pub fn new(slice: S, size: I) -> Self {
        if size <= I::ZERO {
            panic!("size less than zero");
        }

        Mock {
            runs: vec![(size, slice)],
        }
    }

    pub fn size(&self) -> I {
        self.runs.last().map(|(i, _)| *i).unwrap_or(I::ZERO)
    }

    pub fn insert(&mut self, index: I, mut slice: S, size: I) {
        if size == I::ZERO {
            panic!("invalid insertion size");
        } else if index > self.size() {
            panic!("index out of bounds");
        }

        // find the insertion point
        let mut idx = match self.runs.binary_search_by_key(&index, |(i, _)| *i) {
            Err(i) => i,
            Ok(i) => i + 1,
        };

        let key_start = if idx < self.runs.len() {
            Some(
                idx.checked_sub(1)
                    .map(|i| self.runs[i].0)
                    .unwrap_or(I::ZERO),
            )
        } else {
            None
        };

        // If the index is greater than the start of the key it's in, then we need to split
        // that key
        if let Some(s) = key_start && s < index {
            // let (key_end, mut lhs) = self.runs.remove(idx);
            let pos_in_key = index.sub_left(s);
            let rhs = self.runs[idx].1.split_at(pos_in_key);
            let rhs_end = mem::replace(&mut self.runs[idx].0, pos_in_key);
            self.runs[idx].0 = index;
            self.runs.insert(idx + 1, (rhs_end, rhs));
            idx += 1;
        }

        let mut base_pos = index;
        let mut old_size = I::ZERO;
        let mut new_size = size;
        let mut lhs_end_override = None;

        // insert at the the point between this key and the one before

        if let Some(p) = idx.checked_sub(1) {
            let (lhs_end, lhs) = self.runs.remove(p);
            assert_eq!(lhs_end, index);
            match lhs.try_join(slice) {
                Err((lhs, s)) => {
                    self.runs.insert(p, (lhs_end, lhs));
                    slice = s;
                }
                Ok(new) => {
                    let lhs_start = p.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);
                    let lhs_size = lhs_end.sub_left(lhs_start);
                    old_size = old_size.add_left(lhs_size);
                    new_size = new_size.add_left(lhs_size);
                    base_pos = lhs_start;
                    slice = new;
                    idx = p;
                    lhs_end_override = Some(lhs_end);
                }
            }
        }

        // `idx` is already the right-hand node, because `index` is equal to the end of `lhs`
        if idx < self.runs.len() {
            let (rhs_end, rhs) = self.runs.remove(idx);
            match slice.try_join(rhs) {
                Err((s, rhs)) => {
                    self.runs.insert(idx, (rhs_end, rhs));
                    slice = s;
                }
                Ok(new) => {
                    let rhs_start = lhs_end_override.unwrap_or_else(|| {
                        idx.checked_sub(1)
                            .map(|i| self.runs[i].0)
                            .unwrap_or(I::ZERO)
                    });
                    let rhs_size = rhs_end.sub_left(rhs_start);
                    old_size = old_size.add_right(rhs_size);
                    new_size = new_size.add_right(rhs_size);
                    slice = new;
                }
            }
        }

        self.runs.insert(idx, (base_pos.add_right(new_size), slice));

        let diff = base_pos
            .add_right(new_size)
            .sub_left(base_pos.add_right(old_size));

        for (i, _) in self.runs.get_mut(idx + 1..).unwrap_or(&mut []) {
            *i = i.add_left(diff);
        }
    }

    pub fn iter(&self, range: impl RangeBounds<I>) -> MockIter<'_, I, S> {
        let start_pos;
        let unbounded_start;

        let start = range.start_bound().cloned();
        let end = range.end_bound().cloned();

        let fwd_idx = match start {
            StartBound::Unbounded => {
                start_pos = I::ZERO;
                unbounded_start = true;
                0
            }
            StartBound::Included(i) => {
                start_pos = i;
                unbounded_start = false;
                if i >= self.size() {
                    panic!("start index out of bounds");
                }

                match self.runs.binary_search_by_key(&i, |(i, _)| *i) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                }
            }
        };

        let bkwd_idx = match end {
            EndBound::Included(i) if i < start_pos || i >= self.size() => {
                panic!("invalid range, or end index out of bounds")
            }
            EndBound::Excluded(i)
                if i < start_pos || (i == start_pos && !unbounded_start) || i > self.size() =>
            {
                panic!("invalid range, or end index out of bounds")
            }

            // bkwd_idx is always +1 from the next index, so most of these are +1 from the first
            // backward index
            EndBound::Unbounded => self.runs.len(),
            EndBound::Included(i) => match self.runs.binary_search_by_key(&i, |(i, _)| *i) {
                Ok(i) => i + 2,
                Err(i) => i + 1,
            },
            // Special case empty iterator
            EndBound::Excluded(i) if i == I::ZERO => 0,
            EndBound::Excluded(i) => match self.runs.binary_search_by_key(&i, |(i, _)| *i) {
                Ok(i) => i + 1,
                Err(i) => i + 1,
            },
        };

        MockIter {
            runs: &self.runs,
            fwd_idx,
            bkwd_idx,
        }
    }
}

#[derive(Debug)]
pub struct MockIter<'t, I, S> {
    runs: &'t [(I, S)],
    fwd_idx: usize,
    bkwd_idx: usize,
}

impl<'t, I: Index, S> Iterator for MockIter<'t, I, S> {
    type Item = (Range<I>, I, &'t S);

    fn next(&mut self) -> Option<Self::Item> {
        if self.fwd_idx >= self.bkwd_idx {
            return None;
        }

        let start = self
            .fwd_idx
            .checked_sub(1)
            .map(|i| self.runs[i].0)
            .unwrap_or(I::ZERO);

        let &(end, ref slice) = &self.runs[self.fwd_idx];
        self.fwd_idx += 1;

        Some((start..end, end.sub_left(start), slice))
    }
}

impl<'t, I: Index, S> DoubleEndedIterator for MockIter<'t, I, S> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.fwd_idx >= self.bkwd_idx {
            return None;
        }

        self.bkwd_idx -= 1;
        let start = self
            .bkwd_idx
            .checked_sub(1)
            .map(|i| self.runs[i].0)
            .unwrap_or(I::ZERO);

        let &(end, ref slice) = &self.runs[self.bkwd_idx];

        Some((start..end, end.sub_left(start), slice))
    }
}

#[cfg(test)]
mod tests {
    use super::Mock;
    use crate::Constant;

    // Tests copied from the `RleTree::iter` documentation
    #[test]
    fn basic_iter_tests() {
        assert!(std::panic::catch_unwind(|| {
            let tree: Mock<usize, Constant<char>> = Mock::new(Constant('a'), 5);
            let _ = tree.iter(5..);
        })
        .is_err());

        assert!(std::panic::catch_unwind(|| {
            let tree: Mock<usize, Constant<char>> = Mock::new(Constant('a'), 5);
            let _ = tree.iter(..=5);
        })
        .is_err());

        assert!(std::panic::catch_unwind(|| {
            let tree: Mock<usize, Constant<char>> = Mock::new(Constant('a'), 5);
            let _ = tree.iter(5..5);
        })
        .is_err());

        let tree: Mock<usize, Constant<char>> = Mock::new(Constant('a'), 5);
        let _ = tree.iter(..0);
    }

    #[test]
    fn auto_fuzz_1() {
        let mut tree_0: Mock<usize, Constant<char>> = Mock::new_empty();
        tree_0.insert(0, Constant('A'), 16146052610957303968);
        {
            let mut iter = tree_0.iter(..=10978558926184448);
            {
                let item = iter.next_back().unwrap();
                assert_eq!(item.0, 0..16146052610957303968);
                assert_eq!(item.1, 16146052610957303968);
                assert_eq!(item.2, &Constant('A'));
            }
        }
    }

    #[test]
    fn auto_fuzz_2() {
        // manual state checks have been added to this function because they were useful during
        // debugging, and there's not really a reason to remove them.

        let mut tree_0: Mock<u8, Constant<char>> = Mock::new_empty();
        tree_0.insert(0, Constant('V'), 147);
        assert_eq!(tree_0.runs, [(147, Constant('V'))]);

        tree_0.insert(0, Constant('A'), 9);
        assert_eq!(tree_0.runs, [(9, Constant('A')), (156, Constant('V'))]);

        tree_0.insert(98, Constant('A'), 8);
        assert_eq!(
            tree_0.runs,
            [
                (9, Constant('A')),
                (98, Constant('V')),
                (106, Constant('A')),
                (164, Constant('V'))
            ]
        );
    }

    #[test]
    fn auto_fuzz_3() {
        let mut tree_0: Mock<u8, Constant<char>> = Mock::new_empty();
        {
            let mut iter = tree_0.iter(..0);
            assert!(iter.next().is_none());
        }
    }
}
