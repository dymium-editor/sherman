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
            self.runs[idx].0 = pos_in_key;
            self.runs.insert(idx + 1, (rhs_end, rhs));
        }

        let mut base_pos = index;
        let mut old_size = I::ZERO;
        let mut new_size = size;

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
                }
            }
        }

        // `idx` is already the right-hand node, because `index` is equal to the end of `lhs`
        if idx < self.runs.len() {
            let (rhs_end, rhs) = self.runs.remove(idx);
            match rhs.try_join(slice) {
                Err((s, rhs)) => {
                    self.runs.insert(idx + 1, (rhs_end, rhs));
                    slice = s;
                }
                Ok(new) => {
                    let rhs_start = idx
                        .checked_sub(1)
                        .map(|i| self.runs[i].0)
                        .unwrap_or(I::ZERO);
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

            EndBound::Unbounded => self.runs.len(),
            EndBound::Included(i) => match self.runs.binary_search_by_key(&i, |(i, _)| *i) {
                Ok(i) => i + 1,
                Err(i) => i,
            },
            EndBound::Excluded(i) => match self.runs.binary_search_by_key(&i, |(i, _)| *i) {
                Ok(i) => i,
                Err(i) => i,
            },
        };

        MockIter {
            runs: &self.runs,
            fwd_idx,
            bkwd_idx,
        }
    }
}

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
