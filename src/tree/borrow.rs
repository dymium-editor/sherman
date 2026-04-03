//! Abstraction over borrows for a node handle
//!
//! There are two primary goals here:
//!
//! 1. Abstracting the type of borrow, so that we can have more generic methods; and
//! 2. Prevent materializing any direct references to a `Node`, so that we don't violate stacked
//!    borrows.
//!
//! The main interface is through the `Borrowed<B>` type (abstracted over the `Borrow` trait).
//! Immutable and mutable "borrows" can be created with the [`borrow`] and [`borrow_mut`] methods,
//! although the inner `B::Value` cannot ever be directly accessed.
//!
//! Instead, all access to the values is moderated by the [`Field`] trait.

use std::marker::PhantomData;
use std::ptr::NonNull;

/// Abstraction over marker types that represent borrowing or owning a `T`.
///
/// # Safety
///
/// Implementors guarantee that `drop_value` works as documented, dropping the value only if the
/// borrow represents ownership of it.
pub unsafe trait Borrow: Sized {
    type Value;

    /// Drops the inner value in place, if it is owned.
    ///
    /// # Safety
    ///
    /// This method can only be called from the implementation of `Drop`, at most once, and the
    /// value may be left in an invalid state afterwards.
    unsafe fn drop_value(_this: &mut Borrowed<Self>) {}
}

/// An "owned" handle on a node and the subtree rooted at that node.
///
/// Owned handles never have a parent.
#[repr(transparent)]
pub struct UniqueOwned<T>(PhantomData<T>);
// SAFETY: we're required to ensure that `as_unique_owned_mut` returns the same reference that was
// passed in, which is clearly true.
unsafe impl<T: Allocatable> Borrow for UniqueOwned<T> {
    type Value = T;

    unsafe fn drop_value(this: &mut Borrowed<Self>) {
        // SAFETY: same requirements as `drop_value`; we're just passing it through.
        unsafe { this.drop_impl() }
    }
}

/// An immutable reference on a node and the subtree rooted at that node.
#[repr(transparent)]
pub struct Immut<'a, T>(PhantomData<&'a T>);
// SAFETY: Implementations that don't override `as_unique_owned_mut` are safe by default.
unsafe impl<'a, T> Borrow for Immut<'a, T> {
    type Value = T;
}

/// An mutable reference on a node and the subtree rooted at that node.
#[repr(transparent)]
pub struct Mut<'a, T>(PhantomData<&'a mut T>);
// SAFETY: Implementations that don't override `as_unique_owned_mut` are safe by default.
unsafe impl<'a, T> Borrow for Mut<'a, T> {
    type Value = T;
}

/// Marker for [`Borrow`]s that allow mutable access
pub trait BorrowAsMut: Borrow {}
impl<T: Allocatable> BorrowAsMut for UniqueOwned<T> {}
impl<'a, T> BorrowAsMut for Mut<'a, T> {}

/// Types that can be used for a [`UniqueOwned`] borrow
///
/// # Safety
///
/// Implementors must guarantee that calls to `alloc()` return a pointer that's properly aligned
/// and valid for reads / writes.
///
/// Additionally, implementors **must not implement a destructor**, because actually using the
/// reference in the `Drop` implementation violates the tree borrows model. Instead, this trait
/// provides `pre_drop` here; see that function for more.
pub unsafe trait Allocatable: Sized {
    /// Allocates space for the value
    fn alloc() -> NonNull<Self>;

    /// A pseudo-destructor for allocated values. This is called before the real destructor, so the
    /// value must remain valid, but it allows doing some operations when the values is dropped,
    /// without producing UB.
    fn pre_drop(_this: &mut Borrowed<UniqueOwned<Self>>) {}

    /// Frees a pointer previously returned by `alloc`
    ///
    /// ## Safety
    ///
    /// The caller guarantees that the pointer was previously returned by `alloc`, and has not yet
    /// been freed.
    unsafe fn free(ptr: NonNull<Self>);
}

#[repr(C)]
pub struct Borrowed<B: Borrow> {
    ptr: NonNull<B::Value>,
    marker: B,
}

impl<B: Borrow> Drop for Borrowed<B> {
    fn drop(&mut self) {
        // SAFETY: drop_impl can be called at most once, from Drop, which is true here.
        unsafe { B::drop_value(self) }
    }
}

impl<T: Allocatable> Borrowed<UniqueOwned<T>> {
    /// Allocates the value and returns a uniquely owned "borrow" for that value
    pub fn alloc_new(value: T) -> Self {
        let ptr = T::alloc();
        // SAFETY: `.write()` requires that `p.as_ptr()` is valid for writes, and is
        // properly aligned. This is guaranteed by implementors of `Allocatable`.
        unsafe { ptr.as_ptr().write(value) };
        Borrowed { ptr, marker: UniqueOwned(PhantomData) }
    }

    /// Implementation of `Drop`
    ///
    /// # Safety
    ///
    /// This method can only be called from the implementation of `Drop`, at most once, and the
    /// value may be left in an invalid state afterwards.
    unsafe fn drop_impl(&mut self) {
        T::pre_drop(self);
        // SAFETY: drop_in_place requires that the pointer is valid for reads and writes
        // (guaranteed by the original allocation), and that we're ok to drop it right now, which
        // is guarnateed by the caller.
        unsafe { std::ptr::drop_in_place(self.ptr.as_ptr()) };
        // SAFETY: T::free requires that self.ptr was returned by a call to T::alloc, which is an
        // invariant of Borrowed<UniqueOwned<...>>
        unsafe { T::free(self.ptr) };
    }
}

impl<'a, T> Borrowed<Immut<'a, T>> {
    // SAFETY: `v` must be properly aligned, and point to a value of `T`.
    pub unsafe fn from_non_null(ptr: NonNull<T>) -> Self {
        Borrowed { ptr, marker: Immut(PhantomData) }
    }
}

impl<'a, T> Borrowed<Mut<'a, T>> {
    // SAFETY: `v` must be properly aligned, and point to a value of `T`.
    pub unsafe fn from_non_null(ptr: NonNull<T>) -> Self {
        Borrowed { ptr, marker: Mut(PhantomData) }
    }
}

impl<B: Borrow> Borrowed<B> {
    pub fn as_ptr(&self) -> NonNull<B::Value> {
        self.ptr
    }

    pub fn addr(&self) -> usize {
        self.ptr.as_ptr().addr()
    }

    pub fn borrow(&self) -> Borrowed<Immut<'_, B::Value>> {
        // SAFETY: Pointer is already guaranteed to be valid by this borrow.
        // We're just going through `NonNull` directly in order to circumvent stacked borrows.
        unsafe { <Borrowed<Immut<_>>>::from_non_null(self.as_ptr()) }
    }

    pub fn borrow_mut(&mut self) -> Borrowed<Mut<'_, B::Value>>
    where
        B: BorrowAsMut,
    {
        // SAFETY: Pointer is already guaranteed to be valid by this borrow.
        // We're just going through `NonNull` directly in order to circumvent stacked borrows.
        unsafe { <Borrowed<Mut<_>>>::from_non_null(self.as_ptr()) }
    }

    pub fn get<'a, F>(&'a self) -> F::Ref
    where
        F: Field<'a, Container = B::Value>,
    {
        F::as_ref(self.borrow())
    }

    pub fn get_mut<'a, F>(&'a mut self) -> F::Mut
    where
        B: BorrowAsMut,
        F: Field<'a, Container = B::Value>,
    {
        F::as_mut(self.borrow_mut())
    }
}

impl<'a, T> Borrowed<Immut<'a, T>> {
    pub fn reborrow(&self) -> Self {
        // SAFETY: Pointer is already valid because of the existence of `self`
        unsafe { Self::from_non_null(self.ptr) }
    }

    pub fn into_ref<F>(self) -> F::Ref
    where
        F: Field<'a, Container = T>,
    {
        // SAFETY: This is copying a shared reference. It is sound for the same reason that a
        // function like this is sound:
        //
        //   struct Foo<'a>(&'a str);
        //
        //   fn inner<'a>(foo: &Foo<'a>) -> &'a str {
        //       foo.0
        //   }
        let this = unsafe { Self::from_non_null(self.ptr) };
        F::as_ref(this)
    }
}

impl<'a, T> Borrowed<Mut<'a, T>> {
    pub fn into_mut<F>(self) -> F::Mut
    where
        F: Field<'a, Container = T>,
    {
        F::as_mut(self)
    }

    /// Turns `self` into a mutable reference the inner value of an `Option` field, or returns
    /// `Err(self)` if the field was `None`.
    pub fn try_into_mut<F, V: 'a>(self) -> Result<&'a mut V, Self>
    where
        F: Field<'a, Container = T, Mut = &'a mut Option<V>>,
    {
        let this_ptr = self.ptr; // keep this for later, if needed.

        match F::as_mut(self) {
            Some(v) => Ok(v),
            // SAFETY: this is recreating `self` after it was consumed by `F::as_mut`, which is now
            // no longer in use because it returned `None`.
            None => Err(unsafe { Self::from_non_null(this_ptr) }),
        }
    }
}

macro_rules! borrowable {
    (
        $modvis:vis mod $borrowmod:ident;

        $(#[$attrs:meta])*
        $vis:vis struct $ty:ident $(< $( $generic:ident ),+ $(,)? >)? {
            $(
            $(#[$fieldattrs:meta])*
            $fieldvis:vis $fieldname:ident : $fieldty:ty,
            )*
        }
    ) => {
        $(#[$attrs])*
        $vis struct $ty $(< $( $generic ),+ >)? {
            $(
            $(#[$fieldattrs])*
            $fieldvis $fieldname : $fieldty,
            )*
        }

        /// Autogenerated by the [`borrowable!`](crate::tree::borrow::borrowable) macro
        $modvis mod $borrowmod {
            use super::*;

            borrowable!(
                @generateimpl
                ty = $ty;
                generics = { $( $( $generic ),+ )? };
                fields = [ $($fieldname = $fieldty),* ]
            );
        }
    };
    // base case
    (
        @generateimpl
        ty = $ty:ident;
        generics = { $($generics:ident),* };
        fields = []
    ) => {};
    // recursive case:
    (
        @generateimpl
        ty = $ty:ident;
        generics = { $($generics:ident),* };
        fields = [ $fstfieldname:ident = $fstfieldty:ty $(, $fieldnames:ident = $fieldtys:ty )* ]
    ) => {
        #[allow(non_camel_case_types, dead_code)]
        pub(super) struct $fstfieldname < $($generics),* > ( ::std::marker::PhantomData<($($generics),*)> );

        unsafe impl < $($generics),* > $crate::tree::borrow::BorrowableField for $fstfieldname < $($generics),* > {
            type ContainerType = $ty < $($generics),* >;
            type FieldType = $fstfieldty;

            fn offset() -> usize {
                ::std::mem::offset_of!($ty < $($generics),* >, $fstfieldname)
            }
        }

        borrowable!(
            @generateimpl
            ty = $ty;
            generics = { $($generics),* };
            fields = [ $($fieldnames = $fieldtys),* ]
        );
    };
}

/// Not intended for direct use, produced by the `borrowable!` macro
///
/// This trait marks helper types that provide access to fields on borrowed types.
///
/// # Safety
///
/// Implementors of this trait guarantee that `offset()` returns the result of the `offset_of!`
/// macro for the `ContainerType`'s field, which must have type `FieldType`.
pub unsafe trait BorrowableField {
    type ContainerType;
    type FieldType;

    fn offset() -> usize;
}

pub trait Field<'a> {
    type Container;

    type Ref: 'a;
    type Mut: 'a;

    fn as_ref(borrow: Borrowed<Immut<'a, Self::Container>>) -> Self::Ref;
    fn as_mut(borrow: Borrowed<Mut<'a, Self::Container>>) -> Self::Mut;
}

impl<'a, F: BorrowableField> Field<'a> for F
where
    F::FieldType: 'a,
{
    type Container = F::ContainerType;

    type Ref = &'a F::FieldType;
    type Mut = &'a mut F::FieldType;

    fn as_ref(borrow: Borrowed<Immut<'a, Self::Container>>) -> Self::Ref {
        let container_ptr = borrow.ptr.as_ptr();
        // SAFETY: Implementors of `BorrowableField` guarantee that `F::offset()` returns the
        // actual offset of a field with type `F::FieldType`, so validity of the original
        // `container_ptr` implies validity of `field_ptr`. And we know it's valid for the lifetime
        // of `Self::Ref` because `Immut<'a, ...>` represents a borrow on the container with
        // lifetime `'a`, which carries over to borrows on the fields.
        unsafe {
            let field_ptr = (container_ptr as *const u8).add(F::offset()) as *const F::FieldType;
            &*field_ptr
        }
    }

    fn as_mut(borrow: Borrowed<Mut<'a, Self::Container>>) -> Self::Mut {
        let container_ptr = borrow.ptr.as_ptr();
        // SAFETY: Same as for `as_ref()`.
        unsafe {
            let field_ptr = (container_ptr as *mut u8).add(F::offset()) as *mut F::FieldType;
            &mut *field_ptr
        }
    }
}
