//! Wrapper module for [`RecycleVec`], used for slice references

use std::cell::Cell;
use std::mem;
use std::num::NonZeroUsize;

#[cfg(test)]
use std::fmt::{self, Debug, Formatter};

pub struct RecycleVec<T> {
    vals: Vec<Entry<T>>,
    head_empty: Option<LinkId>,
}

/// Unique identifier for an entry in a [`RecycleVec`]
///
/// `EntryId`s are special, and destruction must be handled carefully; if they are not destructed
/// with a call to [`RecycleVec::recycle`], then the destructor will panic.
//
// We store the index plus one so that Option<EntryId> is 8 bytes instead of 16 (on x86-64 or other
// 64-bit targets)
//
// The main invariant of an `EntryId` is that its existence guarantees the entry it references is a
// `Value`.
#[derive(PartialEq, Eq)]
pub struct EntryId {
    idx_plus_one: NonZeroUsize,
}

// Helper struct for using a value's hex formatting for `Debug`
#[cfg(test)]
struct DebugHex<T>(T);

#[cfg(test)]
impl<T: fmt::LowerHex> Debug for DebugHex<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
impl Debug for EntryId {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        DebugHex(self.idx()).fmt(f)
    }
}

/// Like an `EntryId`, but for entries in the "empty" list
#[derive(Copy, Clone)]
struct LinkId {
    idx_plus_one: NonZeroUsize,
}

enum Entry<T> {
    Link(Option<LinkId>),
    Value(Value<T>),
}

struct Value<T> {
    ref_count: Cell<NonZeroUsize>,
    inner: T,
}

impl<T> Default for RecycleVec<T> {
    fn default() -> Self {
        RecycleVec { vals: Vec::new(), head_empty: None }
    }
}

#[cfg(test)]
impl<T: Debug> Debug for RecycleVec<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        struct E<'a, T> {
            entry: &'a Entry<T>,
            refcount_pad: usize,
        }

        impl<'a, T: Debug> Debug for E<'a, T> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                match self.entry {
                    Entry::Link(link) => {
                        let link_idx = link.map(|l| DebugHex(l.idx_plus_one.get() - 1));
                        f.write_fmt(format_args!("Link -> {link_idx:?}"))
                    }
                    Entry::Value(val) => {
                        let refcount = val.ref_count.get();
                        f.write_fmt(format_args!(
                            "Value[refs={refcount:>pad$}]: {:?}",
                            val.inner,
                            pad = self.refcount_pad,
                        ))
                    }
                }
            }
        }

        let max_refcount = self
            .vals
            .iter()
            .filter_map(|e| match e {
                Entry::Value(v) => Some(v.ref_count.get().get()),
                Entry::Link(_) => None,
            })
            .max()
            .unwrap_or(0);

        let idx_pad = self.vals.len().saturating_sub(1).to_string().len();
        let refcount_pad = max_refcount.to_string().len();

        f.write_str("[")?;
        let indent = match f.alternate() {
            true => "        ",
            false => "    ",
        };

        for (idx, entry) in self.vals.iter().enumerate() {
            let e = E { entry, refcount_pad };
            f.write_fmt(format_args!("\n{indent}{idx:#<idx_pad$x}: {e:?}"))?;
        }

        if !self.vals.is_empty() {
            f.write_str("\n]")
        } else {
            f.write_str("]")
        }
    }
}

impl<T> RecycleVec<T> {
    /// Adds a new item to the `RecycleVec`, returning a unique identifier for it
    pub fn push(&mut self, val: T) -> EntryId {
        if let Some(id) = self.head_empty.take() {
            self.head_empty = self.next_link(id);
            let (v, e_id) = Value::new(val, id.idx_plus_one);
            self.vals[id.idx()] = Entry::Value(v);
            e_id
        } else {
            let idx_plus_one = match NonZeroUsize::new(self.vals.len() + 1) {
                Some(n) => n,
                // SAFETY: `Vec::len` is always <= isize::MAX, so `Vec::len() + 1` should always be
                // less than or equal to usize::MAX; it can't wrap to zero
                None => unsafe { weak_unreachable!() },
            };
            let (v, id) = Value::new(val, idx_plus_one);
            self.vals.push(Entry::Value(v));
            id
        }
    }

    /// Drops the `EntryId`, returning the contents of the entry if it's the last reference to it
    #[must_use = "recycling can return the old value, which must be explicitly handled"]
    pub fn recycle(&mut self, id: EntryId) -> Option<T> {
        // Our filler value *happens* to be the thing we'll want to set the entry to if we do end
        // up removing it.
        let next_link = self.head_empty;
        let dummy_empty = Entry::Link(next_link);

        // Temporarily extract the value, so that we can return the inner `T` if we need to.
        // Otherwise, we'll put the value back.
        let val = match mem::replace(&mut self.vals[id.idx()], dummy_empty) {
            Entry::Value(v) => v,
            // SAFETY: Because the `EntryId` exists, this entry is not a link.
            Entry::Link(_) => unsafe { weak_unreachable!() },
        };

        match NonZeroUsize::new(val.ref_count.get().get() - 1) {
            // We're done with this value; add it to the front of the empty values list
            None => {
                self.head_empty = Some(LinkId { idx_plus_one: id.idx_plus_one });
                mem::forget(id);
                Some(val.inner)
            }
            Some(c) => {
                val.ref_count.set(c);
                self.vals[id.idx()] = Entry::Value(val);
                mem::forget(id);
                None
            }
        }
    }

    /// Clones the `EntryId`, increasing its ref count
    pub fn clone(&self, id: &EntryId) -> EntryId {
        let rc = match &self.vals[id.idx()] {
            Entry::Value(v) => &v.ref_count,
            // SAFETY: Because the `EntryId` exists, this entry is not a link.
            Entry::Link(_) => unsafe { weak_unreachable!() },
        };

        rc.set(rc.get().checked_add(1).unwrap());
        EntryId { idx_plus_one: id.idx_plus_one }
    }

    /// Returns a reference to the value
    pub fn get(&self, id: &EntryId) -> &T {
        match &self.vals[id.idx()] {
            Entry::Value(v) => &v.inner,
            // SAFETY: Because the `EntryId` exists, this entry is not a link.
            Entry::Link(_) => unsafe { weak_unreachable!() },
        }
    }

    /// Replaces the entry, returning the previous value
    pub fn replace(&mut self, id: &EntryId, val: T) -> T {
        match &mut self.vals[id.idx()] {
            Entry::Value(v) => mem::replace(&mut v.inner, val),
            // SAFETY: Because the `EntryId` exists, this entry is not a link.
            Entry::Link(_) => unsafe { weak_unreachable!() },
        }
    }

    /// Returns the `LinkId` of the next entry in the linked list, starting from `id`
    fn next_link(&self, id: LinkId) -> Option<LinkId> {
        match &self.vals[id.idx()] {
            Entry::Link(next) => *next,
            // SAFETY: Existence of a `LinkId` guarantees that it references an `Entry::Link`
            Entry::Value(_) => unsafe { weak_unreachable!() },
        }
    }
}

impl EntryId {
    /// Helper method to get the index in `vals` this `EntryId` corresponds to
    ///
    /// Exposed semi-publically so that it can be used as a source of pseudo-randomness.
    pub(crate) fn idx(&self) -> usize {
        self.idx_plus_one.get() - 1
    }

    /// Helper method for cloning the `EntryId`, only to be used when the original [`RecycleVec`] has
    /// been dropped
    ///
    /// ## Safety
    ///
    /// The [`RecycleVec`] that this `EntryId` belongs to must have been dropped.
    pub unsafe fn clone_because_vec_is_dropped(&self) -> Self {
        EntryId { idx_plus_one: self.idx_plus_one }
    }
}

impl LinkId {
    /// Internal helper method to get the index in `vals` this `LinkId` corresponds to
    fn idx(&self) -> usize {
        self.idx_plus_one.get() - 1
    }
}

impl<T> Value<T> {
    /// Creates a new `Value` at the given index, also returning the [`EntryId`] referencing it
    fn new(val: T, idx_plus_one: NonZeroUsize) -> (Self, EntryId) {
        let v = Value {
            ref_count: Cell::new(NonZeroUsize::new(1).unwrap()),
            inner: val,
        };
        let e_id = EntryId { idx_plus_one };
        (v, e_id)
    }
}
