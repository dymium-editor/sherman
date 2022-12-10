//! Mock implementation with essentially the same API as [`RleTree`], but backed by a vector
//!
//! [`RleTree`]: crate::RleTree

use crate::range::{EndBound, RangeBounds, StartBound};
use crate::{Index, Slice};
use std::mem;
use std::ops::Range;

/// A mock, inefficient implementation of the [`RleTree`](crate::RleTree) interface
#[derive(Debug)]
pub struct Mock<I, S> {
    // list of end positions and the slices in the run, along with stable index
    runs: Vec<(I, S, StableId)>,
    // mapping of `RefId` to index in `runs`
    refs_map: RefMap,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct RefId(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct StableId(usize);

/// Mock `SliceRef`
#[derive(Copy, Clone, Debug)]
pub struct Ref {
    id: RefId,
}

#[derive(Debug)]
struct RefMap {
    // mapping StableId -> maybe value index. Guaranteed one vaule in `stable_indexes` per
    // element in `Mock.runs`
    stable_indexes: Vec<Option<usize>>,
    // mapping RefId -> StableId
    index_refs: Vec<StableId>,
}

// `Clone` is only legal if we don't also have slice references
impl<I: Clone, S: Clone> Clone for Mock<I, S> {
    fn clone(&self) -> Self {
        assert!(self.refs_map.index_refs.is_empty());
        Mock {
            runs: self.runs.clone(),
            refs_map: RefMap {
                stable_indexes: self.refs_map.stable_indexes.clone(),
                index_refs: Vec::new(),
            },
        }
    }
}

impl<I: Index, S: Slice<I>> Mock<I, S> {
    pub fn new_empty() -> Self {
        Mock {
            runs: Vec::new(),
            refs_map: RefMap { stable_indexes: Vec::new(), index_refs: Vec::new() },
        }
    }

    pub fn new(slice: S, size: I) -> Self {
        if size <= I::ZERO {
            panic!("size less than zero");
        }

        let mut refs_map = RefMap { stable_indexes: Vec::new(), index_refs: Vec::new() };
        let stable_id = refs_map.new_stableid(Some(0));

        Mock { runs: vec![(size, slice, stable_id)], refs_map }
    }

    #[cfg(test)]
    fn runs(&self) -> Vec<(I, S)>
    where
        S: Clone,
    {
        self.runs.iter().map(|(i, s, _)| (*i, s.clone())).collect()
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
        (start..end, &self.runs[idx].1)
    }

    pub fn make_ref(&mut self, index: I) -> Ref {
        assert!(index >= I::ZERO && index < self.size());
        let idx = match self.runs.binary_search_by_key(&index, |(i, ..)| *i) {
            Ok(i) => i + 1,
            Err(i) => i,
        };

        let stable_id = self.runs[idx].2;
        Ref { id: self.refs_map.new_refid(stable_id) }
    }

    pub fn insert(&mut self, index: I, mut slice: S, size: I, make_ref: bool) -> Option<Ref> {
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
        match key_start {
            // if-let expressions, my beloved. What have they done to you :'(
            Some(s) if s < index => {
                let pos_in_key = index.sub_left(s);
                let rhs = self.runs[idx].1.split_at(pos_in_key);
                let rhs_end = mem::replace(&mut self.runs[idx].0, pos_in_key);
                self.runs[idx].0 = index;
                let stable_id = self.refs_map.insert(idx + 1);
                self.runs.insert(idx + 1, (rhs_end, rhs, stable_id));
                idx += 1;
            }
            _ => (),
        }

        let mut base_pos = index;
        let mut old_size = I::ZERO;
        let mut new_size = size;
        let mut lhs_end_override = None;

        // insert at the the point between this key and the one before

        let mut lhs_stable_id = None;
        let mut rhs_stable_id = None;

        if let Some(p) = idx.checked_sub(1) {
            let (lhs_end, lhs, lhs_id) = self.runs.remove(p);
            self.refs_map.remove(lhs_id, p);
            assert_eq!(lhs_end, index);
            match lhs.try_join(slice) {
                Err((lhs, s)) => {
                    self.runs.insert(p, (lhs_end, lhs, lhs_id));
                    self.refs_map.reinsert(p, lhs_id);
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
                    lhs_stable_id = Some(lhs_id);
                }
            }
        }

        // `idx` is already the right-hand node, because `index` is equal to the end of `lhs`
        if idx < self.runs.len() {
            let (rhs_end, rhs, rhs_id) = self.runs.remove(idx);
            self.refs_map.remove(rhs_id, idx);
            match slice.try_join(rhs) {
                Err((s, rhs)) => {
                    self.runs.insert(idx, (rhs_end, rhs, rhs_id));
                    self.refs_map.reinsert(idx, rhs_id);
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
                    rhs_stable_id = Some(rhs_id)
                }
            }
        }

        let stable_id = self.refs_map.insert(idx);
        self.runs.insert(idx, (base_pos.add_right(new_size), slice, stable_id));

        if let Some(id) = lhs_stable_id {
            self.refs_map.remap(id, stable_id);
        }
        if let Some(id) = rhs_stable_id {
            self.refs_map.remap(id, stable_id);
        }

        let diff = base_pos.add_right(new_size).sub_left(base_pos.add_right(old_size));

        for (i, ..) in self.runs.get_mut(idx + 1..).unwrap_or(&mut []) {
            *i = i.add_left(diff);
        }

        if !make_ref {
            None
        } else {
            Some(Ref { id: self.refs_map.new_refid(stable_id) })
        }
    }

    pub fn ref_slice(&self, r: &Ref) -> Option<&S> {
        Some(&self.runs[self.refs_map.idx(r.id)?].1)
    }

    pub fn ref_range(&self, r: &Ref) -> Option<Range<I>> {
        let idx = self.refs_map.idx(r.id)?;
        let end = self.runs[idx].0;
        let start = idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);
        Some(start..end)
    }

    pub fn ref_size(&self, r: &Ref) -> Option<I> {
        let Range { start, end } = self.ref_range(r)?;
        Some(end.sub_left(start))
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

                match self.runs.binary_search_by_key(&i, |(i, ..)| *i) {
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
            EndBound::Included(i) => match self.runs.binary_search_by_key(&i, |(i, ..)| *i) {
                Ok(i) => i + 2,
                Err(i) => i + 1,
            },
            // Special case empty iterator
            EndBound::Excluded(i) if i == I::ZERO => 0,
            EndBound::Excluded(i) => match self.runs.binary_search_by_key(&i, |(i, ..)| *i) {
                Ok(i) => i + 1,
                Err(i) => i + 1,
            },
        };

        MockIter { runs: &self.runs, fwd_idx, bkwd_idx }
    }
}

impl RefMap {
    fn idx(&self, id: RefId) -> Option<usize> {
        self.stable_indexes[self.index_refs[id.0].0]
    }

    fn new_stableid(&mut self, idx: Option<usize>) -> StableId {
        let stable_id = StableId(self.stable_indexes.len());
        self.stable_indexes.push(idx);
        stable_id
    }

    fn new_refid(&mut self, points_to: StableId) -> RefId {
        let ref_id = RefId(self.index_refs.len());
        self.index_refs.push(points_to);
        ref_id
    }

    fn stable_apply(&mut self, filter: impl Fn(usize) -> bool, map: impl Fn(usize) -> usize) {
        self.stable_indexes
            .iter_mut()
            .filter_map(|opt| opt.as_mut())
            .filter(|i| filter(**i))
            .for_each(|i| *i = map(*i));
    }

    fn insert(&mut self, idx: usize) -> StableId {
        self.stable_apply(|i| i >= idx, |i| i + 1);
        self.new_stableid(Some(idx))
    }

    fn reinsert(&mut self, idx: usize, stable_id: StableId) {
        self.stable_apply(|i| i >= idx, |i| i + 1);
        self.stable_indexes[stable_id.0] = Some(idx);
    }

    fn remove(&mut self, stable_id: StableId, idx: usize) {
        self.stable_indexes[stable_id.0] = None;
        self.stable_apply(|i| i > idx, |i| i - 1);
    }

    fn remap(&mut self, from: StableId, to: StableId) {
        self.index_refs.iter_mut().filter(|id| **id == from).for_each(|id| *id = to);
    }
}

#[derive(Debug)]
pub struct MockIter<'t, I, S> {
    runs: &'t [(I, S, StableId)],
    fwd_idx: usize,
    bkwd_idx: usize,
}

impl<'t, I: Index, S> Iterator for MockIter<'t, I, S> {
    type Item = (Range<I>, I, &'t S);

    fn next(&mut self) -> Option<Self::Item> {
        if self.fwd_idx >= self.bkwd_idx {
            return None;
        }

        let start = self.fwd_idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);

        let &(end, ref slice, _) = &self.runs[self.fwd_idx];
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
        let start = self.bkwd_idx.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);

        let &(end, ref slice, _) = &self.runs[self.bkwd_idx];

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
        tree_0.insert(0, Constant('A'), 16146052610957303968, false);
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
        tree_0.insert(0, Constant('V'), 147, false);
        assert_eq!(tree_0.runs(), [(147, Constant('V'))]);

        tree_0.insert(0, Constant('A'), 9, false);
        assert_eq!(tree_0.runs(), [(9, Constant('A')), (156, Constant('V'))]);

        tree_0.insert(98, Constant('A'), 8, false);
        assert_eq!(tree_0.runs(), [
            (9, Constant('A')),
            (98, Constant('V')),
            (106, Constant('A')),
            (164, Constant('V'))
        ]);
    }

    #[test]
    fn auto_fuzz_3() {
        let tree_0: Mock<u8, Constant<char>> = Mock::new_empty();
        {
            let mut iter = tree_0.iter(..0);
            assert!(iter.next().is_none());
        }
    }
}
