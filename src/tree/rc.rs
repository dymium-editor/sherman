//! Handling of strong counts for `EnableCow`, plus weak counts & borrowing for `EnableRefs`

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::node::{HandleImmut, HandleWeak};
use crate::param::{self, RleTreeConfig};

/// Abstraction for reference counts across `RleTree` parameterizations
///
/// # Safety
///
/// This trait is relied upon in unsafe code to be implemented correctly.
/// Individual methods list guarantees that are required by implementors; it is unsafe to implement
/// this trait in a way that doesn't satisfy each of those guarantees.
pub unsafe trait RefCount: Sized {
    /// Creates a new `RefCount` with a value of 1.
    fn one() -> Self;

    /// Returns whether the `RefCount`'s value is 1.
    ///
    /// Implementors guarantee that, after returning `is_unique() == true`, unsynchronized mutable
    /// access to the values in the node is safe.
    fn is_unique(&self) -> bool;

    /// Increments the value of the `RefCount`.
    ///
    /// Callers guarantee that this is not called when `is_zero() == true`.
    fn increment(&self);

    /// Resets the value back to 1.
    ///
    /// Callers guarantee that this is only called when the value is *semantically* unique,
    /// allowing that the actual count at this point may be zero.
    fn reset(&self);

    /// Decrements the value of the `RefCount`, returning if it is now equal to zero.
    ///
    /// Implementors guarantee that, after returning `decrement_and_is_zero() == true`,
    /// unsynchronized mutable access to the values in the node is safe.
    fn decrement_and_is_zero(&self) -> bool;
}

unsafe impl RefCount for () {
    fn one() -> Self {}

    fn is_unique(&self) -> bool {
        true
    }

    fn increment(&self) {
        unreachable!("cannot increment `()` as refcount");
    }

    fn reset(&self) {}

    fn decrement_and_is_zero(&self) -> bool {
        true
    }
}

/// Wrapper around `Cell<usize>` so that we can make it `impl RefUnwindSafe`
pub struct UsizeCell(Cell<usize>);

impl RefUnwindSafe for UsizeCell {}

unsafe impl RefCount for UsizeCell {
    fn one() -> Self {
        UsizeCell(Cell::new(1))
    }

    fn is_unique(&self) -> bool {
        self.0.get() == 1
    }

    fn increment(&self) {
        let old = self.0.get();
        let new = old.checked_add(1).unwrap_or_else(|| panic!("refcount overflow"));
        self.0.set(new);
    }

    fn reset(&self) {
        self.0.set(1);
    }

    fn decrement_and_is_zero(&self) -> bool {
        let old = self.0.get();
        let new = old.checked_sub(1).unwrap_or_else(|| panic!("refcount underflow"));
        self.0.set(new);
        new == 0
    }
}

unsafe impl RefCount for AtomicUsize {
    fn one() -> Self {
        AtomicUsize::new(1)
    }

    fn is_unique(&self) -> bool {
        // `Acquire` here matches with the `Release` in decrement so that any writes before
        // decrementing in another thread must be visible. This is needed because `is_unique`
        // allows mutable access, so we must protect against stale reads.
        //
        // If we didn't `Acquire` here, it would be possible for the following to occur:
        //
        // 1. Thread A writes to a value behind a mutex
        // 2. Thread A drops the node (calling `decrement_and_is_zero`)
        // 3. Thread B observes `is_unique() == true`
        // 4. Thread B uses `Mutex::get_mut()` which skips synchronization because `&mut Mutex<T>`
        //    statically guarantees unique access.
        // 5. Thread B makes unsynchronized reads/writes that race with step (1)
        //
        // So instead, `Acquire` at step (3) enforces a happens-after relationship with the
        // decrement at step (2) and the writes at step (1).
        //
        // See `decrement_and_is_zero` for more.
        self.load(Ordering::Acquire) == 1
    }

    fn increment(&self) {
        // `Relaxed` is ok here because we only care that the count is *at least* one, and the
        // caller should not increment if the value is zero. Dropping the handle on this ref count
        // (and therefore calling `decrement()`) cannot be reordered before this increment.
        let old = self.fetch_add(1, Ordering::Relaxed);
        if old > isize::MAX as usize {
            panic!("refcount greater than isize::MAX");
        }
    }

    fn reset(&self) {
        // `Relaxed` is ok here because the caller guarantees that current ownership is unique.
        // This is similar to how `increment` is relaxed.
        self.store(1, Ordering::Relaxed);
    }

    fn decrement_and_is_zero(&self) -> bool {
        // Two parts to note here.
        //
        // 1. We want to ensure any writes before `decrement()` are visible in any other thread
        //    that later uses the value, because things like `is_unique()` may result in mutable
        //    access and must avoid stale reads. So we use `Acquire` in `is_unique()` and `Release`
        //    here.
        // 2. Similarly, if this decrement drops the count to zero, we might also use it for
        //    mutable access *here*, and we need to protect against stale reads. So we do an
        //    additional `Acquire` if the count is now zero, matching the `Release` from any prior
        //    `decrement()` operation.
        //
        // It might not immediately be obvious why something like (2) is needed. The `Arc`
        // internals spell it out:
        //
        // > Since a Mutex is not acquired when it is deleted, we can't rely on its synchronization
        // > logic to make writes in thread A visible to a destructor running in thread B.
        //
        // In short: If we write to a mutex in thread A, then drop the node in thread A (which
        // calls `decrement()`), and then run the (mutex) destructor in thread B, the destructor
        // might not observe the writes from thread A, unless we explicitly `Acquire` here to match
        // the `Release` from thread A's `decrement()`.

        let is_now_zero = self.fetch_sub(1, Ordering::Release) == 1;
        if is_now_zero {
            self.load(Ordering::Acquire);
        }
        is_now_zero
    }
}

pub trait Redirect<I, S, P: RleTreeConfig<I, S>> {
    fn empty() -> Self;
    fn to(node: HandleImmut<'_, I, S, P>) -> Self;
    fn replace(&self, h: Option<HandleWeak<I, S, P>>) -> Option<HandleWeak<I, S, P>>;
}

impl<I, S, P: RleTreeConfig<I, S>> Redirect<I, S, P> for () {
    fn empty() -> Self {}

    fn to(_node: HandleImmut<'_, I, S, P>) -> Self {}

    fn replace(&self, _h: Option<HandleWeak<I, S, P>>) -> Option<HandleWeak<I, S, P>> {
        None
    }
}

pub struct RedirectCell<I, S>(Cell<Option<HandleWeak<I, S, param::EnableRefs>>>);

impl<I, S> UnwindSafe for RedirectCell<I, S>
where
    I: UnwindSafe,
    S: UnwindSafe,
{
}
impl<I, S> RefUnwindSafe for RedirectCell<I, S>
where
    I: RefUnwindSafe,
    S: RefUnwindSafe,
{
}

impl<I, S> Redirect<I, S, param::EnableRefs> for RedirectCell<I, S> {
    fn empty() -> Self {
        RedirectCell(Cell::new(None))
    }

    fn to(node: HandleImmut<'_, I, S, param::EnableRefs>) -> Self {
        RedirectCell(Cell::new(Some(node.weak())))
    }

    fn replace(
        &self,
        h: Option<HandleWeak<I, S, param::EnableRefs>>,
    ) -> Option<HandleWeak<I, S, param::EnableRefs>> {
        self.0.replace(h)
    }
}

/// Abstraction for `RefCell`-like borrowing *or* nothing at all
pub trait BorrowState {
    fn new() -> Self;

    fn borrow_mut(&self);
    fn unborrow_mut(&self);

    fn borrow_weak(&self);
    fn unborrow_weak(&self);
}

impl BorrowState for () {
    fn new() -> Self {}

    fn borrow_mut(&self) {}
    fn unborrow_mut(&self) {}
    fn borrow_weak(&self) {}
    fn unborrow_weak(&self) {}
}

#[derive(Copy, Clone)]
#[cfg_attr(test, derive(Debug))]
pub enum RealBorrowState {
    Open,
    Mutable(NonZeroUsize),
    WeakImmutable(NonZeroUsize),
}

impl RealBorrowState {
    fn error_msg(&self) -> &'static str {
        match self {
            RealBorrowState::Open => "node is not borrowed",
            RealBorrowState::Mutable(_) => "node is already mutably borrowed",
            RealBorrowState::WeakImmutable(_) => "node is already weak-immutably borrowed",
        }
    }

    fn panic_internal_err(&self) -> ! {
        panic!("internal error: {}", self.error_msg())
    }

    fn panic_maybe_user_err(&self) -> ! {
        panic!("internal error or reentrant access: {}", self.error_msg())
    }
}

/// Wrapper around `Cell<RealBorrowState>` so we an make it `impl RefUnwindSafe`
#[cfg_attr(test, derive(Debug))]
pub struct BorrowStateCell(Cell<RealBorrowState>);

impl RefUnwindSafe for BorrowStateCell {}

impl BorrowState for BorrowStateCell {
    fn new() -> Self {
        BorrowStateCell(Cell::new(RealBorrowState::Open))
    }

    fn borrow_mut(&self) {
        self.0.update(|state| match state {
            RealBorrowState::Open => RealBorrowState::Mutable(NonZeroUsize::MIN),
            RealBorrowState::Mutable(count) => {
                let new_count = count.checked_add(1).expect("borrow count should not overflow");
                RealBorrowState::Mutable(new_count)
            }
            RealBorrowState::WeakImmutable(_) => state.panic_maybe_user_err(),
        });
    }

    fn unborrow_mut(&self) {
        self.0.update(|state| match state {
            RealBorrowState::Mutable(count) => match NonZeroUsize::new(count.get() - 1) {
                None => RealBorrowState::Open,
                Some(new_count) => RealBorrowState::Mutable(new_count),
            },
            _ => state.panic_internal_err(),
        });
    }

    fn borrow_weak(&self) {
        self.0.update(|state| match state {
            RealBorrowState::Open => RealBorrowState::WeakImmutable(NonZeroUsize::MIN),
            RealBorrowState::WeakImmutable(count) => {
                let new_count = count.checked_add(1).expect("borrow count should not overflow");
                RealBorrowState::WeakImmutable(new_count)
            }
            RealBorrowState::Mutable(_) => state.panic_maybe_user_err(),
        });
    }

    fn unborrow_weak(&self) {
        self.0.update(|state| match state {
            RealBorrowState::WeakImmutable(count) => match NonZeroUsize::new(count.get() - 1) {
                None => RealBorrowState::Open,
                Some(new_count) => RealBorrowState::WeakImmutable(new_count),
            },
            _ => state.panic_internal_err(),
        });
    }
}
