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

    pub fn remove(&mut self, range: impl std::ops::RangeBounds<I>) -> Fake<I, S> {
        let std::ops::Range { start, end } = self.resolve_removal_bounds(range);
        self.remove_internal(start, end)
    }

    pub fn drain(&mut self, range: impl std::ops::RangeBounds<I>) -> FakeDrain<I, S> {
        let std::ops::Range { start, end } = self.resolve_removal_bounds(range);
        let inner = self.remove_internal(start, end);
        FakeDrain::new(inner, start)
    }

    fn resolve_removal_bounds(&self, range: impl std::ops::RangeBounds<I>) -> Range<I> {
        use std::ops::Bound;

        let start = match range.start_bound() {
            Bound::Unbounded => I::ZERO,
            Bound::Excluded(_) => panic!("exclusive start bound disallowed"),
            Bound::Included(&i) if i < I::ZERO => panic!("bad start: {i:?} less than zero"),
            Bound::Included(&i) => i,
        };

        let end = match range.end_bound() {
            Bound::Unbounded => self.size(),
            Bound::Excluded(&i) if i > self.size() => {
                panic!("bad end: {i:?} greater than size {:?}", self.size())
            }
            Bound::Excluded(&i) => i,
            Bound::Included(_) => panic!("inclusive end bound disallowed"),
        };

        if start > end {
            panic!("bad range: start {start:?} > end {end:?}");
        }

        start..end
    }

    fn remove_internal(&mut self, start: I, end: I) -> Fake<I, S> {
        if start == end {
            return Fake::new_empty();
        }

        let (mut front_idx, split_front) =
            match self.runs.binary_search_by_key(&start, |(i, ..)| *i) {
                Ok(i) => (i + 1, false),
                Err(i) => (i, start != I::ZERO),
            };
        let (back_idx, split_back) = match self.runs.binary_search_by_key(&end, |(i, ..)| *i) {
            Ok(i) => (i + 1, false),
            Err(i) => (i, true),
        };

        let mut new_runs = Vec::new();

        if front_idx == back_idx && split_front && split_back {
            let idx = front_idx;
            let value = self.runs[idx].1.take().unwrap();
            let value_start = idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);

            let (part, rhs) = value.split_at(end.sub_left(value_start));
            let (lhs, mid) = part.split_at(start.sub_left(value_start));

            new_runs = vec![(end.sub_left(start), Some(mid))];

            // Try joining lhs & rhs. If they can't join, we'll need to re-insert.
            match lhs.try_join(rhs) {
                Ok(new_value) => {
                    // All good
                    self.runs[idx].1 = Some(new_value);
                }
                Err((lhs, rhs)) => {
                    // Put rhs back, and then insert a new lhs matching its NEW position.
                    // We'll increment front_idx to make sure we skip it when shifting back all the
                    // following indexes.
                    self.runs[idx].1 = Some(rhs);
                    self.runs.insert(idx, (start, Some(lhs)));
                    front_idx += 1;
                }
            }
        } else {
            let back = if !split_back {
                None
            } else {
                let rhs_start = back_idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);
                let (lhs, rhs) =
                    self.runs[back_idx].1.take().unwrap().split_at(end.sub_left(rhs_start));
                self.runs[back_idx].1 = Some(rhs);

                Some((end.sub_left(start), Some(lhs)))
            };

            let front = if !split_front {
                None
            } else {
                let lhs_start = front_idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);
                let (lhs, rhs) =
                    self.runs[front_idx].1.take().unwrap().split_at(start.sub_left(lhs_start));
                self.runs[front_idx].1 = Some(lhs);

                let front_size = self.runs[front_idx].0.sub_left(start);
                self.runs[front_idx].0 = start;

                front_idx += 1;

                Some((front_size, Some(rhs)))
            };

            if front_idx < back_idx {
                new_runs = self
                    .runs
                    .drain(front_idx..back_idx)
                    .map(|(value_end, value)| (value_end.sub_left(start), value))
                    .collect();
            }
            if let Some(pair) = front {
                new_runs.insert(0, pair);
            }
            if let Some(pair) = back {
                new_runs.push(pair);
            }
        }

        for (value_end_pos, _) in &mut self.runs[front_idx..] {
            *value_end_pos = value_end_pos.sub_left(end).add_left(start);
        }

        // Try to join the adjacent sides
        if front_idx > 0 && front_idx < self.runs.len() {
            let lhs = self.runs[front_idx - 1].1.take().unwrap();
            let rhs = self.runs[front_idx].1.take().unwrap();
            match lhs.try_join(rhs) {
                Ok(joined) => {
                    self.runs[front_idx].1 = Some(joined);
                    self.runs.remove(front_idx - 1);
                }
                Err((lhs, rhs)) => {
                    self.runs[front_idx - 1].1 = Some(lhs);
                    self.runs[front_idx].1 = Some(rhs);
                }
            }
        }

        Fake { runs: new_runs }
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

#[derive(Debug)]
pub struct FakeIntoIter<I, S> {
    inner: FakeDrain<I, S>,
}

impl<I, S> Iterator for FakeIntoIter<I, S>
where
    I: Index,
    S: Slice<I>,
{
    type Item = (Range<I>, S);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<I, S> DoubleEndedIterator for FakeIntoIter<I, S>
where
    I: Index,
    S: Slice<I>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<I, S> IntoIterator for Fake<I, S>
where
    I: Index,
    S: Slice<I>,
{
    type Item = (Range<I>, S);
    type IntoIter = FakeIntoIter<I, S>;

    fn into_iter(self) -> Self::IntoIter {
        FakeIntoIter { inner: FakeDrain::new(self, I::ZERO) }
    }
}

#[derive(Debug)]
pub struct FakeDrain<I, S> {
    inner: Fake<I, S>,

    // Original start bound of the drain
    start: I,

    // Iteration indexes
    front_idx: usize,
    back_idx: usize,
}

impl<I, S> FakeDrain<I, S>
where
    I: Index,
    S: Slice<I>,
{
    fn new(inner: Fake<I, S>, start: I) -> Self {
        let len = inner.runs.len();
        FakeDrain { inner, start, front_idx: 0, back_idx: len }
    }

    fn item(&mut self, idx: usize) -> (Range<I>, S) {
        let start = idx.checked_sub(1).map(|i| self.inner.runs[i].0).unwrap_or(I::ZERO);

        let end = self.inner.runs[idx].0;
        let slice = self.inner.runs[idx].1.take().unwrap();

        (self.start.add_right(start)..self.start.add_right(end), slice)
    }
}

impl<I, S> Iterator for FakeDrain<I, S>
where
    I: Index,
    S: Slice<I>,
{
    type Item = (Range<I>, S);

    fn next(&mut self) -> Option<Self::Item> {
        if self.front_idx >= self.back_idx {
            return None;
        }

        let item = self.item(self.front_idx);
        self.front_idx += 1;
        Some(item)
    }
}

impl<I, S> DoubleEndedIterator for FakeDrain<I, S>
where
    I: Index,
    S: Slice<I>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_idx >= self.back_idx {
            return None;
        }

        self.back_idx -= 1;
        Some(self.item(self.back_idx))
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

    #[test]
    fn test_drain() {
        let mut fake: Fake<u8, Constant<char>> = Fake::new_empty();
        // Empty drain:
        assert_eq!(fake.drain(..).count(), 0);

        fake.insert(0, Constant('A'), 3);
        fake.insert(3, Constant('B'), 3);
        fake.insert(6, Constant('C'), 3);
        fake.insert(9, Constant('D'), 3);

        // Aligned drain:
        let drained = fake.drain(3..9).collect::<Vec<_>>();
        assert_eq!(drained, [(3..6, Constant('B')), (6..9, Constant('C'))]);

        let new_contents = fake.iter(..).collect::<Vec<_>>();
        assert_eq!(new_contents, [(0..3, &Constant('A')), (3..6, &Constant('D'))]);

        // return to previous state
        fake.insert(3, Constant('B'), 3);
        fake.insert(6, Constant('C'), 3);
        let new_contents = fake.iter(..).collect::<Vec<_>>();
        assert_eq!(
            new_contents,
            [
                (0..3, &Constant('A')),
                (3..6, &Constant('B')),
                (6..9, &Constant('C')),
                (9..12, &Constant('D')),
            ]
        );

        // Split drain:
        let drained = fake.drain(4..10).collect::<Vec<_>>();
        assert_eq!(
            drained,
            [
                (4..6, Constant('B')),
                (6..9, Constant('C')),
                (9..10, Constant('D'))
            ],
        );

        let new_contents = fake.iter(..).collect::<Vec<_>>();
        assert_eq!(
            new_contents,
            [
                (0..3, &Constant('A')),
                (3..4, &Constant('B')),
                (4..6, &Constant('D'))
            ],
        );

        // Point drain, should be empty:
        let drained = fake.drain(2..2).collect::<Vec<_>>();
        assert_eq!(drained, []);

        // Drain everything:
        let drained = fake.drain(..).collect::<Vec<_>>();
        assert_eq!(
            drained,
            [
                (0..3, Constant('A')),
                (3..4, Constant('B')),
                (4..6, Constant('D'))
            ],
        );
        assert_eq!(fake.iter(..).count(), 0);
    }

    #[test]
    fn test_remove_join() {
        // This was actually found by fuzzing, where the `RleTree` impl was correct and the fake
        // implementation wasn't.
        let mut fake: Fake<u8, Constant<char>> = Fake::new_empty();
        fake.insert(0, Constant('Q'), 117);
        fake.remove(50..75);
        assert_eq!(fake.get(42), (0..92, &Constant('Q')));
    }
}
