//! Array hack for doing additive math in const args
//!
//! If a length can be expressed in terms of another constant by a tree of addition and
//! multiplication operations, then we can build an array-like object with that length.
//!
//! The basic unit of interaction in this module is the [`ArrayHack`] trait, which has some
//! documentation explaining what's going on.

use std::mem::size_of;
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
