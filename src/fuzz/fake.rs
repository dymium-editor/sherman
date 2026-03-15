//! See [`Fake`]

use std::mem;
use std::ops::Range;

use crate::{Index, Slice};

/// A fake implementation of the [`RleTree`] interface, backed by a vector
///
/// This is used during fuzzing to compare `RleTree` behavior against a "ground truth"
/// implementation that's much less efficient.
///
/// [`RleTree`]: crate::RleTree
#[derive(Debug, Clone)]
pub struct Fake<I, S> {
    /// List of *end* positions and the slice in each run
    ///
    /// Slices are expected to be `Some(_)` except for transitional states.
    runs: Vec<(I, Option<S>)>,
}

impl<I: Index, S: Slice<I>> Fake<I, S> {
    pub fn new_empty() -> Self {
        Fake { runs: Vec::new() }
    }

    pub fn new(slice: S, size: I) -> Self {
        if size <= I::ZERO {
            panic!("size less than zero");
        }

        Fake { runs: vec![(size, Some(slice))] }
    }

    pub fn size(&self) -> I {
        self.runs.last().map(|(i, ..)| *i).unwrap_or(I::ZERO)
    }

    pub fn get(&self, index: I) -> (Range<I>, &S) {
        assert!(index >= I::ZERO && index < self.size());
        let idx = match self.runs.binary_search_by_key(&index, |(i, ..)| *i) {
            Ok(i) => i + 1,
            Err(i) => i,
        };

        let end = self.runs[idx].0;
        let start = idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);
        (start..end, self.runs[idx].1.as_ref().unwrap())
    }

    pub fn iter(&self, range: impl std::ops::RangeBounds<I>) -> FakeIter<'_, I, S> {
        use std::ops::Bound;

        let (start_pos, front_idx) = match range.start_bound() {
            Bound::Unbounded => (I::ZERO, 0),
            Bound::Excluded(_) => panic!("excluded start bound disallowed"),
            Bound::Included(&i) => {
                if i > self.size() {
                    panic!("start index out of bounds");
                }

                let idx = match self.runs.binary_search_by_key(&i, |(i, ..)| *i) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                };

                (i, idx)
            }
        };

        let back_idx = match range.end_bound() {
            Bound::Included(&idx) if idx >= self.size() || idx < start_pos => {
                panic!("invalid range or end index out of bounds")
            }
            Bound::Excluded(&idx) if idx > self.size() || idx < start_pos => {
                panic!("invalid range or end index out of bounds")
            }

            Bound::Unbounded => self.runs.len(),
            Bound::Included(i) => match self.runs.binary_search_by_key(i, |(i, ..)| *i) {
                Ok(i) => i + 2,
                Err(i) => i + 1,
            },
            Bound::Excluded(&i) if i == I::ZERO => 0,
            Bound::Excluded(i) => match self.runs.binary_search_by_key(i, |(i, ..)| *i) {
                Ok(i) => i + 1,
                Err(i) => i + 1,
            },
        };

        FakeIter { runs: &self.runs, front_idx, back_idx }
    }

    pub fn insert(&mut self, index: I, mut slice: S, size: I) {
        if size == I::ZERO {
            panic!("invalid insertion size");
        } else if index > self.size() {
            panic!("index out of bounds");
        }

        // find the insertion point
        let mut idx = match self.runs.binary_search_by_key(&index, |(i, ..)| *i) {
            Err(i) => i,
            Ok(i) => i + 1,
        };

        let key_start = if idx < self.runs.len() {
            Some(idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO))
        } else {
            None
        };

        // If the index is greater than the start of the key it's in, then we need to split
        // that key
        if let Some(s) = key_start
            && s < index
        {
            let pos_in_key = index.sub_left(s);
            let (lhs, rhs) = self.runs[idx].1.take().unwrap().split_at(pos_in_key);
            self.runs[idx].1 = Some(lhs);
            let rhs_end = mem::replace(&mut self.runs[idx].0, pos_in_key);
            self.runs[idx].0 = index;
            self.runs.insert(idx + 1, (rhs_end, Some(rhs)));
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
            match lhs.unwrap().try_join(slice) {
                Err((lhs, s)) => {
                    self.runs.insert(p, (lhs_end, Some(lhs)));
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
            match slice.try_join(rhs.unwrap()) {
                Err((s, rhs)) => {
                    self.runs.insert(idx, (rhs_end, Some(rhs)));
                    slice = s;
                }
                Ok(new) => {
                    let rhs_start = lhs_end_override.unwrap_or_else(|| {
                        idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO)
                    });
                    let rhs_size = rhs_end.sub_left(rhs_start);
                    old_size = old_size.add_right(rhs_size);
                    new_size = new_size.add_right(rhs_size);
                    slice = new;
                }
            }
        }

        self.runs.insert(idx, (base_pos.add_right(new_size), Some(slice)));

        let diff = base_pos.add_right(new_size).sub_left(base_pos.add_right(old_size));

        for (i, ..) in self.runs.get_mut(idx + 1..).unwrap_or(&mut []) {
            *i = i.add_left(diff);
        }
    }
}

#[derive(Debug)]
pub struct FakeIter<'a, I, S> {
    runs: &'a [(I, Option<S>)],
    front_idx: usize,
    back_idx: usize,
}

impl<'a, I, S> Iterator for FakeIter<'a, I, S>
where
    I: Index,
{
    type Item = (Range<I>, &'a S);

    fn next(&mut self) -> Option<Self::Item> {
        if self.front_idx >= self.back_idx {
            return None;
        }

        let start = self.front_idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);

        let (end, slice) = self.runs.get(self.front_idx)?;
        self.front_idx += 1;

        Some((start..*end, slice.as_ref().unwrap()))
    }
}

impl<'a, I, S> DoubleEndedIterator for FakeIter<'a, I, S>
where
    I: Index,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_idx >= self.back_idx {
            return None;
        }

        self.back_idx -= 1;
        let start = self.back_idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);

        let (end, slice) = self.runs.get(self.back_idx)?;

        Some((start..*end, slice.as_ref().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::Fake;
    use crate::Constant;

    #[test]
    fn test_empty_iter() {
        let mut fake: Fake<u8, Constant<char>> = Fake::new_empty();
        // Everything valid on an empty tree should return nothing
        assert_eq!(fake.iter(..).count(), 0);
        assert_eq!(fake.iter(..).rev().count(), 0);
        assert_eq!(fake.iter(0..).count(), 0);
        assert_eq!(fake.iter(0..).rev().count(), 0);
        assert_eq!(fake.iter(..0).count(), 0);
        assert_eq!(fake.iter(..0).rev().count(), 0);
        assert_eq!(fake.iter(0..0).count(), 0);
        assert_eq!(fake.iter(0..0).rev().count(), 0);

        fake.insert(0, Constant('A'), 5);
        fake.insert(5, Constant('B'), 5);

        // Iterators at an edge should be empty:
        assert_eq!(fake.iter(..0).count(), 0);
        assert_eq!(fake.iter(..0).rev().count(), 0);
        assert_eq!(fake.iter(5..5).count(), 0);
        assert_eq!(fake.iter(5..5).rev().count(), 0);
        assert_eq!(fake.iter(10..).count(), 0);
        assert_eq!(fake.iter(10..).rev().count(), 0);
        // ... but they should be non-empty if they're in the middle of a value
        assert_eq!(fake.iter(2..2).count(), 1);
        assert_eq!(fake.iter(2..2).rev().count(), 1);
        assert_eq!(fake.iter(7..7).count(), 1);
        assert_eq!(fake.iter(7..7).rev().count(), 1);
    }
}
