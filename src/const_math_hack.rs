//! Array hack for doing additive math in const args
//!
//! If a length can be expressed in terms of another constant by a tree of addition and
//! multiplication operations, then we can build an array-like object with that length.
//!
//! The basic unit of interaction in this module is the [`ArrayHack`] trait, which has some
//! documentation explaining what's going on.

use std::mem::{self, size_of, MaybeUninit};
use std::ptr;
use std::slice;

/// Struct that represents an array with length `L::LEN + R::LEN`
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Add<L, R> {
    lhs: L,
    rhs: R,
}

/// Struct that represents an array with length `A::LEN * M`
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Mul<A, const M: usize> {
    items: [A; M],
}

/// Blanket trait for types that are represented in memory in the same way a fixed-length array is
///
/// This trait is implemented by [`Add`], [`Mul`] and all fixed-length arrays.
///
/// When we say "represented in memory", we mean that an array `[Self::Element; LEN]` would have
/// the same size, alignment, and position of elements as the type. This trait is `unsafe` to
/// because it *assumes* this fact.
///
/// The above is clearly true for array types, but not so obviously so for [`Add`] and [`Mul`].
///
/// The implementations for [`Add`] and [`Mul`] rely on the `#[repr(C)]` attached to them: for
/// [`Add`], it ensures that two array-like objects placed sequentially will behave as if they're
/// one array (alignment ensures the `rhs[0]` is at the position of where `lhs[L::LEN]` would be).
/// And for [`Mul`], the same effect occurs, but with sequential values in the same array. If they
/// weren't `#[repr(C)]`, we could run into issues around field reordering.
pub unsafe trait ArrayHack: Sized {
    /// Length of the array
    const LEN: usize;

    /// The type of element stored in the array
    type Element;

    /// Convenience function to return the length of the array, equal to `Self::LEN`
    fn len(&self) -> usize {
        Self::LEN
    }

    /// Gets a pointer to the element at index `idx`, without performing any checks
    ///
    /// ## Safety
    ///
    /// The pointer `this` must *actually* point to a valid instance of `Self`, and the value `idx`
    /// must be less than or equal to `Self::LEN`. The result is immediate UB if either of these
    /// are violated.
    ///
    /// We allow `idx == Self::LEN` so that it's possible to produce dangling pointers one-past the
    /// end of the array. Those pointers are not valid for reads or writes, but may be useful for
    /// zero-size copies.
    unsafe fn get_ptr_unchecked(this: *const Self, idx: usize) -> *const Self::Element {
        // This function is "unchecked" so it doesn't *really* matter, but it's still worthwhile
        // having this debug-only check. If idx >= LEN at release time, then it's UB either way;
        // `weak_assert` just makes that official.
        unsafe { weak_assert!(idx <= Self::LEN) };

        // So, pointer::add requires some interesting things about its input -- notably that the
        // offset (i.e. `idx * size_of::<T>()`) can't overflow an `isize`. We can't guarnatee this
        // for arbitrary types on arbitrary architectures (technically it can occur), but we *can*
        // decide at compile-time if this can occur.
        //
        // The other things pointer::add requires are guaranteed if `this` does point to a valid
        // instance of `Self`, so we don't need to worry about them.
        if Self::LEN * size_of::<Self::Element>() < isize::MAX as usize {
            // SAFETY: See above.
            unsafe { (this as *const Self::Element).add(idx) }
        } else {
            // On weird architectures, fallback to `wrapping_add`
            (this as *const Self::Element).wrapping_add(idx)
        }
    }

    /// Gets a mutable pointer to the element at index `idx`, without performing any checks
    ///
    /// ## Safety
    ///
    /// The pointer `this` must *actually* point to a valid instance of `Self`, and the value `idx`
    /// must be less than or equal to `Self::LEN`. The result is immediate UB if either of these
    /// are violated.
    ///
    /// We allow `idx == Self::LEN` so that it's possible to produce dangling pointers one-past the
    /// end of the array. Those pointers are not valid for reads or writes, but may be useful for
    /// zero-size copies.
    unsafe fn get_mut_ptr_unchecked(this: *mut Self, idx: usize) -> *mut Self::Element {
        unsafe { weak_assert!(idx <= Self::LEN) };
        if Self::LEN * size_of::<Self::Element>() < isize::MAX as usize {
            // SAFETY: See note above in `get_ptr_unchecked`; it's the same here.
            unsafe { (this as *mut Self::Element).add(idx) }
        } else {
            (this as *mut Self::Element).wrapping_add(idx)
        }
    }

    /// Returns a reference to the element at index `idx`, without performing any checks
    ///
    /// ## Safety
    ///
    /// `idx` must be less than `Self::LEN`. The result is immediate UB if this is violated.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn get_unchecked(&self, idx: usize) -> &Self::Element {
        weak_assert!(idx < Self::LEN);
        let ptr = Self::get_ptr_unchecked(self as *const Self, idx);
        &*ptr
    }

    /// Returns a mutable reference to the element at index `idx`, without performing any checks
    ///
    /// ## Safety
    ///
    /// `idx` must be less than `Self::LEN`. The result is immediate UB if this is violated.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn get_mut_unchecked(&mut self, idx: usize) -> &mut Self::Element {
        weak_assert!(idx < Self::LEN);
        let ptr = Self::get_mut_ptr_unchecked(self as *mut Self, idx);
        &mut *ptr
    }

    fn as_slice(&self) -> &[Self::Element] {
        let ptr = self as *const Self as *const Self::Element;
        let len = Self::LEN;
        // SAFETY: The conditions on implementing this trait guarantee for us that this is valid.
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Element] {
        let ptr = self as *mut Self as *mut Self::Element;
        let len = Self::LEN;
        // SAFETY: The conditions on implementing this trait guarantee for us that this is valid.
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }
}

unsafe impl<T, const N: usize> ArrayHack for [T; N] {
    const LEN: usize = N;
    type Element = T;
}

unsafe impl<T, L, R> ArrayHack for Add<L, R>
where
    L: ArrayHack<Element = T>,
    R: ArrayHack<Element = T>,
{
    const LEN: usize = L::LEN + R::LEN;
    type Element = T;
}

unsafe impl<A: ArrayHack, const M: usize> ArrayHack for Mul<A, M> {
    const LEN: usize = A::LEN * M;
    type Element = A::Element;
}

/// Trait for mapping `ArrayHack` arrays from one type to another
///
/// The method `map_array` is roughly similar to the standard library's `map` method on arrays, but
/// does not have the same possibility of relatively poor stack performance.
///
/// Internally, we use a [`GradualInitArray`]-[`GradualUninitArray`] pair to manage the conversion.
pub unsafe trait MappedArray<E>: ArrayHack {
    type Mapped: ArrayHack<Element = E>;

    /// Produces a new [`ArrayHack`] array of the same length
    fn map_array<F: FnMut(Self::Element) -> E>(self, mut f: F) -> Self::Mapped {
        self.map_array_with_index(|_i, e| f(e))
    }

    fn map_array_with_index<F: FnMut(usize, Self::Element) -> E>(self, mut f: F) -> Self::Mapped {
        let mut uninit = GradualUninitArray::new(self);
        let mut init = GradualInitArray::new();

        for i in 0..Self::LEN {
            unsafe { init.push_unchecked(f(i, uninit.take_unchecked())) };
        }

        mem::forget(uninit);
        unsafe { init.into_init_unchecked() }
    }
}

unsafe impl<T, S, const N: usize> MappedArray<S> for [T; N] {
    type Mapped = [S; N];
}

unsafe impl<T, S, L, R> MappedArray<S> for Add<L, R>
where
    L: MappedArray<S, Element = T>,
    R: MappedArray<S, Element = T>,
{
    type Mapped = Add<<L as MappedArray<S>>::Mapped, <R as MappedArray<S>>::Mapped>;
}

unsafe impl<A: MappedArray<E>, E, const M: usize> MappedArray<E> for Mul<A, M> {
    type Mapped = Mul<<A as MappedArray<E>>::Mapped, M>;
}

/// Helper type for gradually initializing an [`ArrayHack`] array
pub struct GradualInitArray<A: ArrayHack> {
    // need `ArrayHack` bound so we can have it in `Drop`
    array: MaybeUninit<A>,
    len: usize,
}

impl<A: ArrayHack> GradualInitArray<A> {
    /// Creates a new, completely uninitialized array
    pub fn new() -> Self {
        GradualInitArray { array: MaybeUninit::uninit(), len: 0 }
    }

    /// Initializes the next value in the array
    ///
    /// ## Panics
    ///
    /// This method panics if the array has already been fully initialized; i.e. if `A::LEN` calls
    /// to `push` or [`push_unchecked`] have already been made.
    ///
    /// [`push_unchecked`]: Self::push_unchecked
    #[allow(unused)]
    pub fn push(&mut self, val: A::Element) {
        assert!(self.len < A::LEN);
        // SAFETY: guaranteed by the assertion
        unsafe { self.push_unchecked(val) };
    }

    /// Initializes the next value in the array, without checking that there is room
    ///
    /// ## Safety
    ///
    /// There cannot have been `A::LEN` prior calls to [`push`] or `push_unchecked` before this
    /// function.
    ///
    /// [`push`]: Self::push
    pub unsafe fn push_unchecked(&mut self, val: A::Element) {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(self.len < A::LEN) };
        let idx = self.len;
        // SAFETY: `get_mut_ptr_unchecked` is satisfied by `idx = len < A::LEN`, and `write` is
        // valid because the pointer points to a well-aligned element within the array
        unsafe { A::get_mut_ptr_unchecked(self.array.as_mut_ptr(), idx).write(val) };
        self.len += 1;
    }

    /// Converts the partially initialized array into one of the same length, where
    /// each element is wrapped in a `MaybeUninit`
    ///
    /// The first `n` elements, where `n` is the number of calls that have been made to [`push`] or
    /// [`push_unchecked`], are guaranteed to be initialized.
    ///
    /// [`push`]: Self::push
    /// [`push_unchecked`]: Self::push_unchecked
    pub fn into_uninit(mut self) -> <A as MappedArray<MaybeUninit<A::Element>>>::Mapped
    where
        A: MappedArray<MaybeUninit<<A as ArrayHack>::Element>>,
    {
        // Overwrite the length so calling the destructor is fine
        self.len = 0;

        // Sickos: yes... ha ha ha... YES!
        unsafe { std::mem::transmute_copy(&self.array) }
    }

    /// Converts the `GradualInitArray` into the initialized inner array, if it has been completely
    /// initialized
    ///
    /// ## Panics
    ///
    /// This method panics if there have not been *exactly* `A::LEN` prior calls to [`push`] or
    /// [`push_unchecked`] made.
    ///
    /// [`push`]: Self::push
    /// [`push_unchecked`]: Self::push_unchecked
    #[allow(unused)]
    pub fn into_init(self) -> A {
        assert!(self.len == A::LEN);
        // SAFETY: guaranteed by the assertion
        unsafe { self.into_init_unchecked() }
    }

    /// Converts the `GradualInitArray` into the initialized inner array, without checking whether
    /// it has been completely initialized
    ///
    /// ## Safety
    ///
    /// There must have been exactly `A::LEN` prior calls to [`push`] or [`push_unchecked`] made.
    ///
    /// [`push`]: Self::push
    /// [`push_unchecked`]: Self::push_unchecked
    pub unsafe fn into_init_unchecked(mut self) -> A {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(self.len == A::LEN) };

        // Overwrite the length first so that we don't try to drop any values after we've moved
        // them out of the array
        self.len = 0;

        // SAFETY: because `self.len` was `A::LEN`, all the elements of the array were initialized.
        // Because `ArrayHack` guarantees that `A` has nothing more than its elements, we know that
        // the array is now initialized
        unsafe { self.array.assume_init_read() }
    }
}

impl<A: ArrayHack> Drop for GradualInitArray<A> {
    fn drop(&mut self) {
        unsafe {
            let base_ptr: *mut A::Element = self.array.as_mut_ptr() as _;
            let slice: &mut [A::Element] = slice::from_raw_parts_mut(base_ptr, self.len);
            ptr::drop_in_place(slice as *mut [A::Element]);
        }
    }
}

/// Helper type for gradually taking values out of an [`ArrayHack`] array
pub struct GradualUninitArray<A: ArrayHack> {
    // need the `ArrayHack` bound to have it in `Drop`
    array: MaybeUninit<A>,
    next: usize,
}

impl<A: ArrayHack> GradualUninitArray<A> {
    /// Creates a new, fully initialized array to read from
    pub fn new(array: A) -> Self {
        GradualUninitArray { array: MaybeUninit::new(array), next: 0 }
    }

    /// Takes the next value, going from the start to the end of the array
    ///
    /// ## Panics
    ///
    /// This method panics if all of the elements have already been taken -- i.e., if `A::LEN`
    /// calls to `take` or [`take_unchecked`] have been made.
    ///
    /// [`take_unchecked`]: Self::take_unchecked
    #[allow(unused)]
    pub fn take(&mut self) -> A::Element {
        assert!(self.next < A::LEN);
        // SAFETY: guaranteed by the assertion
        unsafe { self.take_unchecked() }
    }

    /// Takes the next value, going from the start to the end of the array, without checking if
    /// there are any values left
    ///
    /// ## Safety
    ///
    /// There cannot have been `A::LEN` prior calls to [`take`] or `take_unchecked`.
    ///
    /// [`take`]: Self::take
    pub unsafe fn take_unchecked(&mut self) -> A::Element {
        let idx = self.next;
        self.next += 1;
        unsafe { A::get_mut_ptr_unchecked(self.array.as_mut_ptr(), idx).read() }
    }
}

impl<A: ArrayHack> Drop for GradualUninitArray<A> {
    fn drop(&mut self) {
        unsafe {
            let start_ptr = A::get_mut_ptr_unchecked(self.array.as_mut_ptr(), self.next);
            let slice = slice::from_raw_parts_mut(start_ptr, A::LEN - self.next);
            ptr::drop_in_place(slice as *mut [A::Element]);
        }
    }
}
