//! See [`Fake`]

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::AtomicU64;

use crate::{Index, Slice};

/// A fake implementation of the [`RleTree`] interface, backed by a vector
///
/// This is used during fuzzing to compare `RleTree` behavior against a "ground truth"
/// implementation that's much less efficient.
///
/// [`RleTree`]: crate::RleTree
#[derive(Debug)]
pub struct Fake<I, S> {
    /// List of *end* positions and the slice in each run
    ///
    /// Slices are expected to be `Some(_)` except for transitional states.
    runs: Vec<(I, Option<S>, Option<RefId>)>,
    /// For each `RefId`, the `RefId` that they should be redirected to, if any.
    redirects: BTreeMap<RefId, RefId>,
}

#[derive(Debug, Clone)]
pub struct FakeStableRef {
    id: RefId,
}

static NEXT_REF_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RefId(u64);

impl RefId {
    fn next() -> Self {
        let id = NEXT_REF_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        RefId(id)
    }
}

impl<I: Clone, S: Clone> Clone for Fake<I, S> {
    fn clone(&self) -> Self {
        // On clone, remap the RefIds so that clone -> re-insert doesn't end up with slices with
        // the same id. That shouldn't be used in practice, because our fuzzing doesn't clone AND
        // use stable refs, but it's still worth making this sound.
        let mut ref_remapping = BTreeMap::new();
        let mut remap =
            |ref_id: RefId| -> RefId { *ref_remapping.entry(ref_id).or_insert_with(RefId::next) };

        let mut runs = self.runs.clone();
        for (_, _, r) in &mut runs {
            *r = r.map(&mut remap);
        }

        let redirects =
            self.redirects.iter().map(|(to, from)| (remap(*to), remap(*from))).collect();

        Fake { runs, redirects }
    }
}

impl<I: Index, S: Slice<I>> Fake<I, S> {
    pub fn new_empty() -> Self {
        Fake { runs: Vec::new(), redirects: BTreeMap::new() }
    }

    pub fn new(slice: S, size: I) -> Self {
        if size <= I::ZERO {
            panic!("size less than or equal to zero");
        }

        Fake {
            runs: vec![(size, Some(slice), None)],
            redirects: BTreeMap::new(),
        }
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

    pub fn make_ref(&mut self, index: I) -> FakeStableRef {
        assert!(index >= I::ZERO && index < self.size());
        let idx = match self.runs.binary_search_by_key(&index, |(i, ..)| *i) {
            Ok(i) => i + 1,
            Err(i) => i,
        };

        // Return an existing id if we have it
        if let &(.., Some(id)) = &self.runs[idx] {
            return FakeStableRef { id };
        }

        // ... otherwise, allocate a new RefId
        let id = RefId::next();
        self.runs[idx].2 = Some(id);
        FakeStableRef { id }
    }

    pub fn get_ref(&self, r: &FakeStableRef) -> Option<(Range<I>, &S)> {
        let mut id = r.id;
        while let Some(&redirect) = self.redirects.get(&id) {
            id = redirect;
        }

        let (idx, (end, value, _)) =
            self.runs.iter().enumerate().find(|&(_, &(.., r))| r == Some(id))?;
        let start = idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);
        Some((start..*end, value.as_ref().unwrap()))
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

    pub fn insert(&mut self, index: I, slice: S, size: I) {
        if size == I::ZERO {
            panic!("invalid insertion size");
        } else if index > self.size() {
            panic!("index out of bounds");
        }

        self.replace_internal(index, index, Some(Fake::new(slice, size)));
    }

    pub fn replace(
        &mut self,
        range: impl std::ops::RangeBounds<I>,
        with: S,
        size: I,
    ) -> Fake<I, S> {
        let std::ops::Range { start, end } = self.resolve_removal_bounds(range);
        let replacement = Fake::new(with, size);
        self.replace_internal(start, end, Some(replacement))
    }

    pub fn remove(&mut self, range: impl std::ops::RangeBounds<I>) -> Fake<I, S> {
        let std::ops::Range { start, end } = self.resolve_removal_bounds(range);
        self.replace_internal(start, end, None)
    }

    pub fn drain(&mut self, range: impl std::ops::RangeBounds<I>) -> FakeDrain<I, S> {
        let std::ops::Range { start, end } = self.resolve_removal_bounds(range);
        let inner = self.replace_internal(start, end, None);
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

    fn replace_internal(
        &mut self,
        start: I,
        end: I,
        replacement: Option<Fake<I, S>>,
    ) -> Fake<I, S> {
        if start == end && replacement.is_none() {
            return Fake::new_empty();
        }

        let (mut front_idx, split_front) =
            match self.runs.binary_search_by_key(&start, |(i, ..)| *i) {
                Ok(i) => (i + 1, false),
                Err(i) => (i, start != I::ZERO),
            };
        let (back_idx, split_back) = match self.runs.binary_search_by_key(&end, |(i, ..)| *i) {
            Ok(i) => (i + 1, false),
            Err(i) => (i, end != I::ZERO && end < self.size()),
        };

        let mut new_runs = Vec::new();

        if front_idx == back_idx && split_front && split_back {
            let idx = front_idx;
            let value = self.runs[idx].1.take().unwrap();
            let value_ref = self.runs[idx].2.take();
            let value_start = idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);

            let (part, rhs) = value.split_at(end.sub_left(value_start));
            let lhs = if start != end {
                let (lhs, mid) = part.split_at(start.sub_left(value_start));
                new_runs = vec![(end.sub_left(start), Some(mid), None)];
                lhs
            } else {
                part
            };

            // Put RHS back, and insert a new entry for LHS - we'll try joining later.
            self.runs[idx].1 = Some(rhs);
            self.runs.insert(idx, (start, Some(lhs), value_ref));
            front_idx += 1;
        } else {
            let back = if !split_back {
                None
            } else {
                let rhs_start = back_idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);
                let (lhs, rhs) =
                    self.runs[back_idx].1.take().unwrap().split_at(end.sub_left(rhs_start));
                let lhs_ref = self.runs[back_idx].2.take();
                self.runs[back_idx].1 = Some(rhs);

                Some((end.sub_left(start), Some(lhs), lhs_ref))
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
                    .map(|(value_end, value, r)| (value_end.sub_left(start), value, r))
                    .collect();
            }
            if let Some((idx, value)) = front {
                new_runs.insert(0, (idx, value, None));
            }
            if let Some((idx, value, r)) = back {
                new_runs.push((idx, value, r));
            }
        }

        let repl_size = replacement.as_ref().map(Fake::size).unwrap_or(I::ZERO);
        for (value_end_pos, ..) in &mut self.runs[front_idx..] {
            *value_end_pos = value_end_pos.sub_left(end).add_left(repl_size).add_left(start);
        }

        let mut repl_start = front_idx;
        let mut repl_len = 0;
        if let Some(mut f) = replacement {
            for (i, _, _) in &mut f.runs {
                *i = i.add_left(start);
            }

            repl_len = f.runs.len();
            // Insert `f.runs` at `replacement_start_idx`. To do so, we'll extend to the end of the
            // vector and then `rotate_right` to wrap them around into place.
            self.runs.extend(f.runs);
            self.runs[repl_start..].rotate_right(repl_len);
            // ... and then, also add any redirects that are there.
            self.redirects.append(&mut f.redirects);
        }

        // Try to join on both sides of the range that was replaced.
        //
        // Start by trying to join with the left-hand side.
        if repl_start > 0 && repl_start < self.runs.len() {
            let lhs = self.runs[repl_start - 1].1.take().unwrap();
            let rhs = self.runs[repl_start].1.take().unwrap();

            match lhs.try_join(rhs) {
                Err((lhs, rhs)) => {
                    // Put the values back
                    self.runs[repl_start - 1].1 = Some(lhs);
                    self.runs[repl_start].1 = Some(rhs);
                }
                Ok(slice) => {
                    let lhs_ref = self.runs[repl_start - 1].2;
                    let rhs_ref = self.runs[repl_start].2;
                    let new_ref = match (lhs_ref, rhs_ref) {
                        (r, None) | (None, r) => r,
                        (Some(rx), Some(ry)) => {
                            // Set up a redirect from ry -> rx
                            assert!(!self.redirects.contains_key(&ry));
                            self.redirects.insert(ry, rx);
                            Some(rx)
                        }
                    };

                    self.runs[repl_start].1 = Some(slice);
                    self.runs[repl_start].2 = new_ref;
                    self.runs.remove(repl_start - 1);
                    repl_start -= 1;
                }
            }
        }

        // And then also try with right-hand side:
        let repl_end = repl_start + repl_len;
        if repl_len != 0 && repl_end < self.runs.len() {
            let lhs = self.runs[repl_end - 1].1.take().unwrap();
            let rhs = self.runs[repl_end].1.take().unwrap();

            match lhs.try_join(rhs) {
                Err((lhs, rhs)) => {
                    // Put the values back
                    self.runs[repl_end - 1].1 = Some(lhs);
                    self.runs[repl_end].1 = Some(rhs);
                }
                Ok(slice) => {
                    let lhs_ref = self.runs[repl_end - 1].2;
                    let rhs_ref = self.runs[repl_end].2;
                    let new_ref = match (lhs_ref, rhs_ref) {
                        (r, None) | (None, r) => r,
                        (Some(rx), Some(ry)) => {
                            // Set up a redirect from ry -> rx
                            assert!(!self.redirects.contains_key(&ry));
                            self.redirects.insert(ry, rx);
                            Some(rx)
                        }
                    };

                    self.runs[repl_end].1 = Some(slice);
                    self.runs[repl_end].2 = new_ref;
                    self.runs.remove(repl_end - 1);
                }
            }
        }

        Fake { runs: new_runs, redirects: self.redirects.clone() }
    }
}

#[derive(Debug)]
pub struct FakeIter<'a, I, S> {
    runs: &'a [(I, Option<S>, Option<RefId>)],
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

        let (end, slice, _ref_id) = self.runs.get(self.front_idx)?;
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

        let (end, slice, _ref_id) = self.runs.get(self.back_idx)?;

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
    use crate::fuzz::CharRange;

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

    #[test]
    fn test_replace() {
        let mut fake: Fake<u8, CharRange> = Fake::new_empty();
        // Empty removal:
        let removed = fake.remove(..);
        assert_eq!(removed.into_iter().collect::<Vec<_>>(), []);

        fake.insert(0, CharRange('A'..'G'), 6);
        fake.insert(6, CharRange('X'..'Z'), 2);
        fake.insert(8, CharRange('P'..'T'), 4);

        // Replace within a single value
        let removed = fake.replace(2..4, CharRange('L'..'N'), 2);
        assert_eq!(removed.into_iter().collect::<Vec<_>>(), [(0..2, CharRange('C'..'E'))],);
        assert_eq!(
            fake.iter(..).collect::<Vec<_>>(),
            [
                (0..2, &CharRange('A'..'C')),
                (2..4, &CharRange('L'..'N')),
                (4..6, &CharRange('E'..'G')),
                (6..8, &CharRange('X'..'Z')),
                (8..12, &CharRange('P'..'T')),
            ],
        );

        // Replace, breaking up values
        let removed = fake.replace(7..9, CharRange('U'..'W'), 2);
        assert_eq!(
            removed.into_iter().collect::<Vec<_>>(),
            [(0..1, CharRange('Y'..'Z')), (1..2, CharRange('P'..'Q')),]
        );
        assert_eq!(
            fake.iter(..).collect::<Vec<_>>(),
            [
                (0..2, &CharRange('A'..'C')),
                (2..4, &CharRange('L'..'N')),
                (4..6, &CharRange('E'..'G')),
                (6..7, &CharRange('X'..'Y')),
                (7..9, &CharRange('U'..'W')),
                (9..12, &CharRange('Q'..'T')),
            ]
        );

        // Replace, split & join with LHS only
        let removed = fake.replace(4..5, CharRange('N'..'O'), 1);
        assert_eq!(removed.into_iter().collect::<Vec<_>>(), [(0..1, CharRange('E'..'F'))],);
        assert_eq!(
            fake.iter(..).collect::<Vec<_>>(),
            [
                (0..2, &CharRange('A'..'C')),
                (2..5, &CharRange('L'..'O')),
                (5..6, &CharRange('F'..'G')),
                (6..7, &CharRange('X'..'Y')),
                (7..9, &CharRange('U'..'W')),
                (9..12, &CharRange('Q'..'T')),
            ]
        );

        // Replace & join with RHS only
        let removed = fake.replace(5..6, CharRange('W'..'X'), 1);
        assert_eq!(removed.into_iter().collect::<Vec<_>>(), [(0..1, CharRange('F'..'G'))],);
        assert_eq!(
            fake.iter(..).collect::<Vec<_>>(),
            [
                (0..2, &CharRange('A'..'C')),
                (2..5, &CharRange('L'..'O')),
                (5..7, &CharRange('W'..'Y')),
                (7..9, &CharRange('U'..'W')),
                (9..12, &CharRange('Q'..'T')),
            ]
        );

        // Replace & join with both sides. Do a removal first so we get slices aligned.
        _ = fake.remove(7..9);
        assert_eq!(
            fake.iter(..).collect::<Vec<_>>(),
            [
                (0..2, &CharRange('A'..'C')),
                (2..5, &CharRange('L'..'O')),
                (5..7, &CharRange('W'..'Y')),
                (7..10, &CharRange('Q'..'T')),
            ]
        );
        let removed = fake.replace(5..7, CharRange('O'..'Q'), 2);
        assert_eq!(removed.into_iter().collect::<Vec<_>>(), [(0..2, CharRange('W'..'Y'))],);
        assert_eq!(
            fake.iter(..).collect::<Vec<_>>(),
            [(0..2, &CharRange('A'..'C')), (2..10, &CharRange('L'..'T')),]
        );

        // Replace leftmost value
        let removed = fake.replace(0..2, CharRange('B'..'D'), 2);
        assert_eq!(removed.into_iter().collect::<Vec<_>>(), [(0..2, CharRange('A'..'C'))]);
        assert_eq!(
            fake.iter(..).collect::<Vec<_>>(),
            [(0..2, &CharRange('B'..'D')), (2..10, &CharRange('L'..'T')),]
        );

        // Replace rightmost value
        let removed = fake.replace(2..10, CharRange('K'..'S'), 8);
        assert_eq!(removed.into_iter().collect::<Vec<_>>(), [(0..8, CharRange('L'..'T'))]);
        assert_eq!(
            fake.iter(..).collect::<Vec<_>>(),
            [(0..2, &CharRange('B'..'D')), (2..10, &CharRange('K'..'S')),]
        );
    }

    #[test]
    fn fuzz_01_stable_ref_basic_removal() {
        // Discovered by fuzzing StableRefOperation<u8, Constant<UpperLetter>>
        let mut fake: Fake<u8, Constant<char>> = Fake::new_empty();
        fake.insert(0, Constant('L'), 96);
        let ref_0 = fake.make_ref(18);
        _ = fake.remove(53..54);
        {
            let (range, slice) = fake.get_ref(&ref_0).unwrap();
            assert_eq!(range, 0..95);
            assert_eq!(slice, &Constant('L'));
        }
    }

    #[test]
    fn fuzz_02_replace_remove() {
        // Discovered by fuzzing BasicOperation<u8, Constant<UpperLetter>>
        let mut fake: Fake<u8, Constant<char>> = Fake::new_empty();
        _ = fake.replace(.., Constant('W'), 204);
        _ = fake.remove(78..);
    }

    #[test]
    fn fuzz_03_insert_split_range() {
        // Discovered by fuzzing BasicOperation<u8, CharRange>
        let mut fake: Fake<u8, CharRange> = Fake::new_empty();
        fake.insert(0, CharRange('D'..'T'), 15);
        fake.insert(3, CharRange('A'..'W'), 62);
    }

    #[test]
    fn fuzz_04_replace_start_ref() {
        let mut fake: Fake<u8, CharRange> = Fake::new_empty();
        _ = fake.replace(..0, CharRange('D'..'O'), 75);
        let ref_0 = fake.make_ref(8);
        fake.insert(4, CharRange('A'..'R'), 50);
        _ = fake.replace(..0, CharRange('A'..'G'), 28);
        assert!(fake.get_ref(&ref_0).is_some());
    }
}
