use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt::{self, Debug};
use std::ops::Range;
use std::panic::{RefUnwindSafe, UnwindSafe};

use crate::{DirectionalAdd, DirectionalSub, Index, Slice};

/// Datastore for [`TrackedIndex`]
#[derive(Debug)]
pub struct IndexInfo<I> {
    epochs: RefCell<Vec<Operation<I>>>,
}

#[derive(Copy, Clone, Debug)]
struct Operation<I> {
    start: I,
    end: I,
    size: I,
}

/// [`Index`] implementor that checks its usage is correct; see [`IndexInfo`] for more.
#[derive(Copy, Clone)]
pub struct TrackedIndex<'t, I> {
    kind: IndexKind<'t, I>,
}

impl<'t, I: UnwindSafe> UnwindSafe for TrackedIndex<'t, I> {}
impl<'t, I: RefUnwindSafe> RefUnwindSafe for TrackedIndex<'t, I> {}

#[derive(Copy, Clone)]
enum IndexKind<'t, I> {
    Zero,
    Range(IndexRange<'t, I>),
}

#[derive(Copy, Clone)]
struct IndexRange<'t, I> {
    info: &'t IndexInfo<I>,
    epoch: usize,
    base: I,
    size: I,
}

/// [`Slice`] implementor that converts a [`TrackedIndex`] into its underlying type
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TrackedSlice<S>(pub S);

//
// "Public" API
//

impl<'t, I: Index> TrackedIndex<'t, I> {
    /// Returns the underlying value represented by the `TrackedIndex`
    pub fn value(&self) -> I {
        match self.kind {
            IndexKind::Zero => I::ZERO,
            IndexKind::Range(r) => r.size,
        }
    }

    // Helper function: Normalize zero-length IndexRange values into IndexKind::Zero, because
    // otherwise forward/rewind behavior is nondeterministic.
    fn normalize(self) -> Self {
        match self.kind {
            IndexKind::Range(r) if r.size == I::ZERO => TrackedIndex::ZERO,
            _ => self,
        }
    }
}

impl<I: Index> Default for IndexInfo<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Index> IndexInfo<I> {
    /// Creates a new store for index tracking
    pub fn new() -> Self {
        IndexInfo { epochs: RefCell::new(Vec::new()) }
    }

    /// Creates a [`TrackedIndex`] representing the index `idx`, offset from zero
    pub fn i(&self, idx: I) -> TrackedIndex<'_, I> {
        let epoch = self.epochs.borrow().len();
        let idx = TrackedIndex {
            kind: IndexKind::Range(IndexRange { info: self, epoch, base: I::ZERO, size: idx }),
        };
        idx.normalize()
    }

    /// Creates a [`TrackedIndex`] representing the index `idx` *before the most recent change*
    ///
    /// # Panics
    ///
    /// This method panics if there haven't yet been any operations.
    pub fn p(&self, idx: I) -> TrackedIndex<'_, I> {
        let epoch = self.epochs.borrow().len()
            .checked_sub(1)
            .unwrap_or_else(|| {
                panic!("cannot represent `TrackedIndex` before previous operation if there are no operations")
            });
        let idx = TrackedIndex {
            kind: IndexKind::Range(IndexRange { info: self, epoch, base: I::ZERO, size: idx }),
        };
        idx.normalize()
    }

    /// Creates a pair of [`TrackedIndex`]es representing the position and length of an insertion
    ///
    /// Note that future calls to [`self.i()`](Self::i) will be tracked as if their positions were
    /// first evaluated after the insertion.
    pub fn prepare_insert(&self, pos: I, size: I) -> (TrackedIndex<'_, I>, TrackedIndex<'_, I>) {
        let (range, size) = self.prepare_replace(pos..pos, size);
        (range.start, size)
    }

    /// Creates a pair of [`TrackedIndex`]es representing the start and end of a range removal
    ///
    /// Note that future calls to [`self.i()`](Self::i) will be tracked as if their positions were
    /// first evaluated after the removal.
    pub fn prepare_remove(&self, start: I, end: I) -> (TrackedIndex<'_, I>, TrackedIndex<'_, I>) {
        let (range, _) = self.prepare_replace(start..end, I::ZERO);
        (range.start, range.end)
    }

    /// Creates a range of [`TrackedIndex`]es representing the start and end of a range
    /// replacement, with another [`TrackedIndex`] representing the replacement size
    ///
    /// Note that future calls to [`self.i()`](Self::i) will be tracked as if their positions were
    /// first evaluated after the removal.
    pub fn prepare_replace(
        &self,
        range: Range<I>,
        size: I,
    ) -> (Range<TrackedIndex<'_, I>>, TrackedIndex<'_, I>) {
        let mut epochs = self.epochs.borrow_mut();

        let epoch = epochs.len();

        // Special case: Don't add an epoch if the replacement is the same size.
        if range.start > range.end || range.end.sub_left(range.start) != size {
            epochs.push(Operation { start: range.start, end: range.end, size });
        }

        let tracked_start = TrackedIndex {
            kind: IndexKind::Range(IndexRange {
                info: self,
                epoch,
                base: I::ZERO,
                size: range.start,
            }),
        };
        let tracked_end = TrackedIndex {
            kind: IndexKind::Range(IndexRange {
                info: self,
                epoch,
                base: I::ZERO,
                size: range.end,
            }),
        };
        let tracked_size = TrackedIndex {
            kind: IndexKind::Range(IndexRange {
                info: self,
                epoch: epochs.len(), // epoch + (did we add an op) ? 1 : 0
                base: range.start,
                size,
            }),
        };

        (tracked_start.normalize()..tracked_end.normalize(), tracked_size.normalize())
    }
}

//
// Internal implementation
//

impl<'t, I: Index, S: Slice<I>> Slice<TrackedIndex<'t, I>> for TrackedSlice<S> {
    fn split_at(self, idx: TrackedIndex<'t, I>) -> (Self, Self) {
        let (lhs, rhs) = self.0.split_at(idx.value());
        (TrackedSlice(lhs), TrackedSlice(rhs))
    }

    fn try_join(self, other: Self) -> Result<Self, (Self, Self)> {
        match self.0.try_join(other.0) {
            Ok(new) => Ok(TrackedSlice(new)),
            Err((lhs, rhs)) => Err((TrackedSlice(lhs), TrackedSlice(rhs))),
        }
    }
}

impl<'t, I: Debug> Debug for TrackedIndex<'t, I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.kind {
            IndexKind::Zero => f.write_str("0"),
            IndexKind::Range(r) => IndexRange::fmt(r, f),
        }
    }
}

impl<'t, I: Debug> Debug for IndexRange<'t, I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let IndexRange { epoch, base, size, .. } = self;
        f.write_fmt(format_args!("{epoch:#x}:({base:?}+{size:?})"))
    }
}

impl<'t, I: Index> IndexRange<'t, I> {
    /// Aligns the two `IndexRange` values to the same epoch, or panics if they cannot be compared
    fn align(op: &str, x: Self, y: Self) -> (Self, Self) {
        fn meet<'t, I: Index>(
            epochs: &[Operation<I>],
            lo: IndexRange<'t, I>,
            hi: IndexRange<'t, I>,
        ) -> Result<(IndexRange<'t, I>, IndexRange<'t, I>), String> {
            let lo_ff = lo.forward(hi.epoch);
            let hi_rr = hi.rewind(lo_ff.epoch);

            if lo_ff.epoch == hi_rr.epoch {
                Ok((lo_ff, hi_rr))
            } else {
                Err(format!(
                    "cannot forward {lo:?} past {loe:#x} (translated {lo_ff:?} overlaps {lo_op:?}) but cannot rewind {hi:?} past {hie:#x} (translated {hi_rr:?} overlaps {hi_op:?})",
                    loe = lo_ff.epoch,
                    hie = hi_rr.epoch,
                    lo_op = epochs[lo_ff.epoch],
                    hi_op = epochs[hi_rr.epoch - 1],
                ))
            }
        }

        let epochs = x.info.epochs.borrow();
        // Optimistically forward the values as far as we can - otherwise, we won't end up properly
        // processing things like removals, where all of the input values are the old epoch but
        // joining across the removal should result in the new epoch.
        let x = x.forward(epochs.len());
        let y = y.forward(epochs.len());

        let result = match x.epoch.cmp(&y.epoch) {
            Ordering::Equal => Ok((x, y)),
            Ordering::Less => meet(&epochs, x, y),
            Ordering::Greater => meet(&epochs, y, x).map(|(y_ff, x_rr)| (x_rr, y_ff)),
        };

        result.unwrap_or_else(|e| {
            panic!("invalid operation: attempted to {op}({x:?}, {y:?}), but {e}")
        })
    }

    fn forward(mut self, goal_epoch: usize) -> Self {
        let epochs = self.info.epochs.borrow();

        'done: for op in &epochs[self.epoch..goal_epoch] {
            if op.end <= self.base {
                // Operation is before this index, so we should shift to adjust
                self.base = self.base.sub_left(op.end).add_left(op.size).add_left(op.start);
            } else if op.start >= self.base.add_right(self.size) {
                // `op.start` is beyond the bounds of this index, so we can ignore it
            } else {
                // `op` overlaps with `self`, so it cannot be compared!
                break 'done;
            }

            self.epoch += 1;
        }

        self
    }

    fn rewind(mut self, goal_epoch: usize) -> Self {
        let epochs = self.info.epochs.borrow();

        'done: for op in epochs[goal_epoch..self.epoch].iter().rev() {
            let op_end = op.start.add_right(op.size);
            if op_end <= self.base {
                // Operation is before this index, so we should shift to adjust
                self.base = self.base.sub_left(op_end).add_left(op.end);
            } else if op.start >= self.base.add_right(self.size) {
                // `op.start` is beyond the bounds of this index, so we can ignore it
            } else {
                // `op` overlaps with `self`, so it cannot be compared!
                break 'done;
            }

            self.epoch -= 1;
        }

        self
    }
}

impl<'t, I: Index> Index for TrackedIndex<'t, I> {
    const ZERO: Self = TrackedIndex { kind: IndexKind::Zero };
}

impl<'t, I: Index> DirectionalAdd for TrackedIndex<'t, I> {
    fn add_right(self, right: Self) -> Self {
        let (lhs, rhs) = match (self.kind, right.kind) {
            (IndexKind::Zero, _) => return right,
            (_, IndexKind::Zero) => return self,
            (IndexKind::Range(lhs), _) if lhs.size == I::ZERO => return right,
            (_, IndexKind::Range(rhs)) if rhs.size == I::ZERO => return self,
            (IndexKind::Range(lhs), IndexKind::Range(rhs)) => (lhs, rhs),
        };

        let (lhs, rhs) = IndexRange::align("add_right", lhs, rhs);

        if lhs.base.add_right(lhs.size) != rhs.base {
            panic!(
                "invalid operation: attempted to add_right({self:?}, {right:?}), but translated add_right({lhs:?}, {rhs:?}) isn't adjacent"
            )
        }

        TrackedIndex {
            kind: IndexKind::Range(IndexRange {
                info: lhs.info,
                epoch: lhs.epoch,
                base: lhs.base,
                size: lhs.size.add_right(rhs.size),
            }),
        }
    }
}

impl<'t, I: Index> DirectionalSub for TrackedIndex<'t, I> {
    fn sub_left(self, left: Self) -> Self {
        let (this, lhs) = match (self.kind, left.kind) {
            (_, IndexKind::Zero) => return self,
            (_, IndexKind::Range(lhs)) if lhs.size == I::ZERO => return self,
            (IndexKind::Range(this), IndexKind::Range(lhs)) => (this, lhs),
            (IndexKind::Zero, IndexKind::Range(_)) => {
                panic!("invalid operation: attempted to sub_left({self:?}, {left:?})")
            }
        };

        let (this, lhs) = IndexRange::align("sub_left", this, lhs);

        if this.base != lhs.base {
            panic!(
                "invalid operation: attempted to sub_left({self:?}, {left:?}), but translated sub_left({this:?}, {lhs:?}) isn't aligned at base"
            )
        } else if lhs.base.add_right(lhs.size) > this.base.add_right(this.size) {
            panic!(
                "invalid operation: attempted to sub_left({self:?}, {left:?}), but translated sub_left({this:?}, {lhs:?}) has larger subtrahend"
            )
        }

        let idx = TrackedIndex {
            kind: IndexKind::Range(IndexRange {
                info: this.info,
                epoch: this.epoch,
                base: lhs.base.add_right(lhs.size),
                size: this.size.sub_left(lhs.size),
            }),
        };

        idx.normalize()
    }

    fn sub_right(self, right: Self) -> Self {
        let (this, rhs) = match (self.kind, right.kind) {
            (_, IndexKind::Zero) => return self,
            (_, IndexKind::Range(rhs)) if rhs.size == I::ZERO => return self,
            (IndexKind::Range(this), IndexKind::Range(rhs)) => (this, rhs),
            (IndexKind::Zero, IndexKind::Range(_)) => {
                panic!("invalid operation: attempted to sub_right({self:?}, {right:?})")
            }
        };

        let (this, rhs) = IndexRange::align("sub_right", this, rhs);

        if this.base.add_right(this.size) != rhs.base.add_right(rhs.size) {
            panic!(
                "invalid operation: attempted to sub_right({self:?}, {right:?}), but translated sub_right({this:?}, {rhs:?}) isn't aligned at end"
            )
        } else if rhs.base < this.base {
            panic!(
                "invalid operation: attempted to sub_right({self:?}, {right:?}), but translated sub_right({this:?}, {rhs:?}) has larger subtrahend"
            )
        }

        let idx = TrackedIndex {
            kind: IndexKind::Range(IndexRange {
                info: this.info,
                epoch: this.epoch,
                base: this.base,
                size: this.size.sub_right(rhs.size),
            }),
        };

        idx.normalize()
    }
}

impl<'t, I: Index> Ord for TrackedIndex<'t, I> {
    fn cmp(&self, other: &Self) -> Ordering {
        let (lhs, rhs) = match (self.kind, other.kind) {
            (IndexKind::Zero, IndexKind::Zero) => return Ordering::Equal,
            (IndexKind::Range(lhs), IndexKind::Zero) => {
                return lhs.size.cmp(&I::ZERO);
            }
            (IndexKind::Zero, IndexKind::Range(rhs)) => {
                return I::ZERO.cmp(&rhs.size);
            }
            (IndexKind::Range(lhs), IndexKind::Range(rhs)) => (lhs, rhs),
        };

        let (lhs, rhs) = IndexRange::align("cmp", lhs, rhs);

        if lhs.base != rhs.base {
            panic!(
                "invalid operation: attempted to cmp({self:?}, {other:?}), but translated cmp({lhs:?}, {rhs:?}) isn't aligned at base"
            )
        }

        lhs.size.cmp(&rhs.size)
    }
}

impl<'t, I: Index> PartialOrd for TrackedIndex<'t, I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'t, I: Index> PartialEq for TrackedIndex<'t, I> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl<'t, I: Index> Eq for TrackedIndex<'t, I> {}
