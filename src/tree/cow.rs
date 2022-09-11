//! Internal handling for concurrent clone-on-write functionality

use crate::param;
use std::sync::atomic::{AtomicUsize, Ordering};

#[rustfmt::skip]
impl param::StrongCount for () {
    fn count(&self) -> usize { 1 }
    fn one() -> Self { () }
    fn is_unique(&self) -> bool { true }
    fn increment(&self) -> Self { () }
    fn decrement(&self) -> bool { true }
}

impl param::sealed::YouCantImplementThis for AtomicUsize {}

impl param::StrongCount for AtomicUsize {
    fn count(&self) -> usize {
        self.load(Ordering::Acquire)
    }

    fn one() -> Self {
        AtomicUsize::new(1)
    }

    fn is_unique(&self) -> bool {
        // the `Acquire` here matches with the `Release` in `decrement` so that a decrement
        // operation from another thread cannot occur after we read is_unique() as `true`.
        self.load(Ordering::Acquire) == 1
    }

    fn increment(&self) -> Self {
        // A relaxed ordering is ok here because we really *only* care about whether the count is
        // at least 1, so the original value guarantees that just fine. Dropping the original value
        // can't be reordered *before* this increment
        let old = self.fetch_add(1, Ordering::Relaxed);
        if old > isize::MAX as usize {
            panic!("more than isize::MAX references to the same node");
        }
        AtomicUsize::new(old)
    }

    fn decrement(&self) -> bool {
        let is_now_zero = self.fetch_sub(1, Ordering::Release) == 1;
        if is_now_zero {
            // This additional load is required so that writes made in another thread are now
            // visible when the destructor for any/all attached data is run. Without this, it's
            // possible to use interior mutability (e.g., with a Mutex) to set a value in Thread A,
            // drop the tree in Thread A, then drop the final reference in Thread B, where the
            // change might not be observed (and thus the destructor is called with stale data).
            //
            // The `Acquire` ordering here matches the `Release` ordering of any prior decrement(),
            // which ensures that writes that occur before the decrement() are now visible after
            // this function call ends.
            //
            // Also this is what `Arc` uses (because it's correct).
            self.load(Ordering::Acquire);
        }

        is_now_zero
    }
}
