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
//!
//! [`borrow`]: Borrowed::borrow
//! [`borrow_mut`]: Borrowed::borrow_mut

use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};
use std::ptr::NonNull;

use super::rc::{self, BorrowState as _, RefCount};

/// Abstraction over marker types that represent borrowing or owning a `T`.
///
/// # Safety
///
/// Implementors guarantee that `drop_value` works as documented, dropping the value only if the
/// borrow represents ownership of it.
///
/// It is safe by default to implement this trait if no methods are overridden.
pub unsafe trait Borrow: Sized {
    type Value: Borrowable;

    /// Drops the inner value in place, if it is owned.
    ///
    /// # Safety
    ///
    /// This method can only be called from the implementation of `Drop`, at most once, and the
    /// value may be left in an invalid state afterwards.
    unsafe fn drop_value(_this: &mut Borrowed<Self>) {}
}

/// A uniquely owned handle on a node and the subtree rooted at that node.
///
/// Specifically, for various parameterizations:
///
/// * with `NoFeatures`, all nodes are implicitly this type (even if erased to [`Owned`])
/// * with `EnableCow`, this represents a node with `StrongCount = 1`, i.e. only owned by the
///   thread holding it
/// * with `EnableRefs`, this represents a normal owned copy of a node (which may, technically,
///   have other references to it)
#[repr(transparent)]
pub struct UniqueOwned<T>(PhantomData<T>);
// SAFETY: we're required to implement `drop_value` correctly, which is correct here.
unsafe impl<T: Borrowable> Borrow for UniqueOwned<T> {
    type Value = T;

    unsafe fn drop_value(this: &mut Borrowed<Self>) {
        // SAFETY: Mostly the same requirements as `drop_value`, which we're just passing through.
        // The other requirement is satisfied because this is a `UniqueOwned` directly.
        unsafe { this.drop_impl() }
    }
}

/// An owned handle on a node and the subtree rooted at that node that may not be unique
#[repr(transparent)]
pub struct Owned<T>(PhantomData<T>);
// SAFETY: we're required to implement `drop_value` correctly, which is correct here.
unsafe impl<T: Borrowable> Borrow for Owned<T> {
    type Value = T;

    unsafe fn drop_value(this: &mut Borrowed<Self>) {
        // SAFETY: same requirements as `drop_value`; we're just passing it through.
        unsafe { this.drop_impl() }
    }
}

/// A weak handle on a node, giving access to the same tree without copy-on-write and also after
/// the owned references have been dropped
pub struct Weak<T>(PhantomData<T>);
// SAFETY: we're required to implement `drop_value` correctly, which is correct here.
unsafe impl<T: Borrowable> Borrow for Weak<T> {
    type Value = T;

    unsafe fn drop_value(this: &mut Borrowed<Self>) {
        // SAFETY: same requirements as `drop_value`; we're just passing it through.
        unsafe { this.drop_impl() }
    }
}

/// An immutable reference on a node and the subtree rooted at that node.
#[repr(transparent)]
pub struct Immut<'a, T>(PhantomData<&'a T>);
// SAFETY: Implementing is safe by default because we don't override methods.
unsafe impl<'a, T: Borrowable> Borrow for Immut<'a, T> {
    type Value = T;
}

#[repr(transparent)]
pub struct WeakImmut<'a, T>(PhantomData<&'a T>);
// SAFETY: Implementing is safe by default because we don't override methods.
unsafe impl<'a, T: Borrowable> Borrow for WeakImmut<'a, T> {
    type Value = T;

    unsafe fn drop_value(this: &mut Borrowed<Self>) {
        // SAFETY: same requirements as `drop_value`; we're just passing it through.
        unsafe { this.drop_impl() }
    }
}

/// An mutable reference on a node and the subtree rooted at that node.
#[repr(transparent)]
pub struct Mut<'a, T>(PhantomData<&'a mut T>);
// SAFETY: Implementing is safe by default because we don't override methods.
unsafe impl<'a, T: Borrowable> Borrow for Mut<'a, T> {
    type Value = T;

    unsafe fn drop_value(this: &mut Borrowed<Self>) {
        // SAFETY: same requirements as `drop_value`; we're just passing it through.
        unsafe { this.drop_impl() }
    }
}

/// Marker for [`Borrow`]s that allow immutable access
///
/// This notably *excludes* [`Weak`] borrows, which instead must be upgraded before any access.
pub trait BorrowAsImmut: Borrow {}
impl<T: Borrowable> BorrowAsImmut for UniqueOwned<T> {}
impl<T: Borrowable> BorrowAsImmut for Owned<T> {}
impl<'a, T: Borrowable> BorrowAsImmut for Immut<'a, T> {}
impl<'a, T: Borrowable> BorrowAsImmut for WeakImmut<'a, T> {}
impl<'a, T: Borrowable> BorrowAsImmut for Mut<'a, T> {}

/// Marker for [`Borrow`]s that allow mutable access
pub trait BorrowAsMut: BorrowAsImmut {}
impl<T: Borrowable> BorrowAsMut for UniqueOwned<T> {}
impl<'a, T: Borrowable> BorrowAsMut for Mut<'a, T> {}

/// Types that can be used for a [`UniqueOwned`] borrow
///
/// # Safety
///
/// Implementors **must not implement a destructor**, because actually using the reference in the
/// `Drop` implementation violates the tree borrows model. Instead, this trait provides `pre_drop`
/// here; see that function for more.
pub unsafe trait Borrowable: Sized {
    type StrongCountField: Field<Container = Self, Value: rc::RefCount> + ImmutField;
    type WeakCountField: Field<Container = Self, Value: rc::RefCount> + ImmutField;
    type BorrowStateField: Field<Container = Self, Value: rc::BorrowState> + ImmutField;

    /// First part of the pseudo-destructor for allocated values.
    ///
    /// This is called when the strong count drops to zero, and before the real destructor, so the
    /// value must remain valid.
    fn drop_strong(this: &mut Borrowed<UniqueOwned<Self>>);

    /// Second part of the pseudo-destructor for allocated values.
    ///
    /// This is called after `drop_strong`, when the weak count drops to zero.
    fn drop_weak(this: &mut Borrowed<Weak<Self>>);
}

#[repr(C)]
pub struct Borrowed<B: Borrow> {
    ptr: NonNull<B::Value>,
    marker: B,
}

#[cfg(not(feature = "nightly"))]
impl<B: Borrow> Drop for Borrowed<B> {
    fn drop(&mut self) {
        // SAFETY: drop_impl can be called at most once, from Drop, which is true here.
        unsafe { B::drop_value(self) }
    }
}

#[cfg(feature = "nightly")]
unsafe impl<#[may_dangle] B: Borrow> Drop for Borrowed<B> {
    fn drop(&mut self) {
        // SAFETY: drop_impl can be called at most once, from Drop, which is true here.
        unsafe { B::drop_value(self) }
    }
}

fn alloc<T>() -> NonNull<T> {
    assert!(mem::size_of::<T>() != 0);

    let layout = Layout::new::<T>();

    // SAFETY: `alloc` requires that `layout` does not have a size of zero. We checked above that
    // the size of T (and therefore the size of the layout) is non-zero.
    let maybe_null_ptr = unsafe { std::alloc::alloc(layout) } as *mut T;

    NonNull::new(maybe_null_ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
}

// SAFETY: The caller guarantees that the pointer was previously created by a call to
// `alloc::<T>()` and has not yet been freed.
unsafe fn free<T>(ptr: NonNull<T>) {
    let layout = Layout::new::<T>();

    // SAFETY: `dealloc` requires that `ptr` is currently allocated by the same allocated (our own
    // caller guarantees it comes from a previous call to this module's alloc<T> and has not yet
    // been freed), and that the layout is the same (guaranteed by same <T>).
    unsafe { std::alloc::dealloc(ptr.as_ptr() as *mut u8, layout) }
}

impl<T: Borrowable> Borrowed<UniqueOwned<T>> {
    /// Allocates the value and returns a uniquely owned "borrow" for that value
    pub fn alloc_new(value: T) -> Self {
        let ptr = alloc::<T>();
        // SAFETY: `.write()` requires that `p.as_ptr()` is valid for writes, and is
        // properly aligned. This is guaranteed by alloc()
        unsafe { ptr.as_ptr().write(value) };
        Borrowed { ptr, marker: UniqueOwned(PhantomData) }
    }

    /// Implementation of `Drop`
    ///
    /// # Safety
    ///
    /// This method can only be called from the implementation of `Drop`, at most once, and the
    /// value may be left in an invalid state afterwards.
    ///
    /// The caller also guarantees either:
    /// 1. `self` is derived from an [`Owned`] borrow, and the strong count is now zero; or
    /// 2. `self` is actually derived from `UniqueOwned`
    unsafe fn drop_impl(&mut self) {
        T::drop_strong(self);

        // At the point where we've dropped the final "strong" reference, we should also drop the
        // "weak" reference implicitly shared by all strong references (that weak reference is also
        // why both ref counts are always initialized = 1).
        let weak: Borrowed<Weak<T>> = Borrowed { ptr: self.ptr, marker: Weak(PhantomData) };
        drop(weak);
    }

    /// Erases the uniqueness of this handle, downgrading it to just `Owned`
    pub fn erase(self) -> Borrowed<Owned<T>> {
        // SAFETY:
        let this = ManuallyDrop::new(self);
        Borrowed { ptr: this.ptr, marker: Owned(PhantomData) }
    }
}

impl<T: Borrowable> Borrowed<Owned<T>> {
    /// Implementation of `Drop`
    ///
    /// # Safety
    ///
    /// This method can only be called from the implementation of `Drop`, at most once, and the
    /// value may be left in an invalid state afterwards.
    unsafe fn drop_impl(&mut self) {
        let strong = self.get::<T::StrongCountField>();

        let is_zero = strong.decrement_and_is_zero();
        if !is_zero {
            return;
        }

        // That was the last strong count; convert to unique owned and run the drop logic.
        let unique_owned: Borrowed<UniqueOwned<T>> =
            Borrowed { ptr: self.ptr, marker: UniqueOwned(PhantomData) };
        drop(unique_owned);
    }

    /// If this is the only `Owned` handle on the allocation, return it as `UniqueOwned`.
    /// Otherwise, create a new allocation with contents populated by calling `clone` on this
    /// allocation.
    ///
    /// # Safety
    ///
    /// If `clone` is called, the caller guarantees that the resulting strong & weak reference
    /// counts will both be 1.
    pub(super) unsafe fn into_unique<C>(self, clone: C) -> Borrowed<UniqueOwned<T>>
    where
        C: for<'a> FnOnce(Borrowed<Immut<'a, T>>) -> T,
    {
        if self.get::<T::StrongCountField>().is_unique() {
            // This handle is already unique - convert it in-place.
            let this = ManuallyDrop::new(self);
            return Borrowed { ptr: this.ptr, marker: UniqueOwned(PhantomData) };
        }

        // Not unique, so we must clone.
        Borrowed::alloc_new(clone(self.borrow()))
    }

    /// If this is the only `Owned` handle on the allocation, return it as `UniqueOwned`.
    /// Otherwise, drop this handle.
    pub(super) fn into_unique_or_discard(self) -> Option<Borrowed<UniqueOwned<T>>> {
        // We're going to temporarily violate the invariants of the strong count, so prevent
        // dropping the handle during that time.
        let this = ManuallyDrop::new(self);

        let is_unique = this.get::<T::StrongCountField>().decrement_and_is_zero();
        if !is_unique {
            // If not unique, we've already decremented the strong count, so we're actually done.
            return None;
        }

        // `this` is guaranteed to be unique, BUT the strong count is currently zero when it should
        // be one. Make the unique handle & then fix the strong count.
        let new = Borrowed { ptr: this.ptr, marker: UniqueOwned(PhantomData) };
        new.get::<T::StrongCountField>().reset();

        Some(new)
    }

    /// If this is the only `Owned` handle on the allocation, return it as `UniqueOwned`.
    /// Otherwise, return `Err(self)`.
    pub(super) fn try_into_unique(self) -> Result<Borrowed<UniqueOwned<T>>, Self> {
        if self.get::<T::StrongCountField>().is_unique() {
            let this = ManuallyDrop::new(self);
            Ok(Borrowed { ptr: this.ptr, marker: UniqueOwned(PhantomData) })
        } else {
            Err(self)
        }
    }

    /// Converts the mutable reference to an `Owned` handle to a `UniqueOwned` handle, if it's
    /// actually unique. Otherwise returns `Err`.
    pub(super) fn try_as_unique<'b>(&'b mut self) -> Result<&'b mut Borrowed<UniqueOwned<T>>, ()> {
        if self.get::<T::StrongCountField>().is_unique() {
            // SAFETY: We're directly changing the `&mut` reference type here.
            // The #[repr(transparent)] on all the borrow types to just be PhantomData, alongside
            // having #[repr(C)] on `Borrowed` means the types they reference have the same layout.
            // And finally, it's safe to provide access as if it's UniqueOwned because there is
            // exactly one strong reference to this node, coming from `self`.
            unsafe {
                Ok(std::mem::transmute::<
                    &'b mut Borrowed<Owned<T>>,
                    &'b mut Borrowed<UniqueOwned<T>>,
                >(self))
            }
        } else {
            Err(())
        }
    }

    /// Creates a copy of the same handle, incrementing the strong count
    pub(super) fn shallow_clone(&self) -> Self {
        self.get::<T::StrongCountField>().increment();
        Borrowed { ptr: self.ptr, marker: Owned(PhantomData) }
    }
}

impl<T: Borrowable> Borrowed<Weak<T>> {
    /// Implementation of `Drop`
    ///
    /// # Safety
    ///
    /// This method can only be called from the implementation of `Drop`, at most once, and the
    /// value may be left in an invalid state afterwards.
    unsafe fn drop_impl(&mut self) {
        // note: we use `get_immut()` here instead of `.access().get()` because
        let weak = self.get_immut::<T::WeakCountField>();
        let is_zero = weak.decrement_and_is_zero();
        if !is_zero {
            return;
        }

        T::drop_weak(self);

        // SAFETY: drop_in_place requires that the pointer is valid for reads and writes
        // (guaranteed by the original allocation), and that we're ok to drop it right now, which
        // is guaranteed by the caller.
        unsafe { std::ptr::drop_in_place(self.ptr.as_ptr()) };
        // SAFETY: free requires that self.ptr was returned by a call to alloc(), which is an
        // invariant of Borrowed<UniqueOwned<...>>
        unsafe { free::<T>(self.ptr) };
    }

    /// Borrows the value
    pub(super) fn access(&self) -> Borrowed<WeakImmut<'_, T>> {
        // SAFETY: The pointer is currently valid - guaranteed by the existence of `Weak` - and we
        // know that it will be for the
        unsafe { <Borrowed<WeakImmut<_>>>::from_non_null(self.ptr) }
    }

    /// Provides access to a read-only field, *without* borrowing the value.
    ///
    /// This is sound only because having implemented `ImmutField` guarantees that there will never
    /// be any mutable references to the field given out.
    pub(super) fn get_immut<F>(&self) -> &F::Value
    where
        F: ImmutField + Field<Container = T>,
    {
        // Produce a single-use Immut borrow. This is not valid for use beyond this function,
        // because we must only hand out usage to this particular field.
        let b = Borrowed { ptr: self.ptr, marker: Immut(PhantomData) };
        F::as_ref(b)
    }
}

impl<B: Borrow> Borrowed<B> {
    pub fn as_ptr(&self) -> NonNull<B::Value> {
        self.ptr
    }

    pub fn borrow(&self) -> Borrowed<Immut<'_, B::Value>>
    where
        B: BorrowAsImmut,
    {
        Borrowed { ptr: self.ptr, marker: Immut(PhantomData) }
    }

    pub fn weak(&self) -> Borrowed<Weak<B::Value>>
    where
        B: BorrowAsImmut,
    {
        self.get::<<B::Value as Borrowable>::WeakCountField>().increment();
        Borrowed { ptr: self.ptr, marker: Weak(PhantomData) }
    }

    pub fn borrow_mut(&mut self) -> Borrowed<Mut<'_, B::Value>>
    where
        B: BorrowAsMut,
    {
        // Set the borrow state.
        self.get::<<B::Value as Borrowable>::BorrowStateField>().borrow_mut();

        Borrowed { ptr: self.as_ptr(), marker: Mut(PhantomData) }
    }

    pub fn get<F>(&self) -> &F::Value
    where
        B: BorrowAsImmut,
        F: Field<Container = B::Value>,
    {
        F::as_ref(self.borrow())
    }
}

impl<'a, T: Borrowable> Borrowed<Immut<'a, T>> {
    pub fn reborrow(&self) -> Self {
        Borrowed { ptr: self.ptr, marker: Immut(PhantomData) }
    }

    /// Creates a new immutable borrow from a pointer to the value
    ///
    /// # Safety
    ///
    /// `ptr` must be properly aligned and point to a value of `T` that is not mutably borrowed for
    /// the entire lifetime of the borrow.
    pub unsafe fn from_non_null(ptr: NonNull<T>) -> Self {
        Borrowed { ptr, marker: Immut(PhantomData) }
    }

    pub fn into_ref<F>(self) -> &'a F::Value
    where
        F: Field<Container = T>,
    {
        F::as_ref(self)
    }
}

impl<'a, T: Borrowable> Borrowed<WeakImmut<'a, T>> {
    /// Implementation of `Drop`
    ///
    /// # Safety
    ///
    /// This method can only be called from the implementation of `Drop`, at most once, and the
    /// value may be left in an invalid state afterwards.
    unsafe fn drop_impl(&mut self) {
        let borrow = self.get::<<T as Borrowable>::BorrowStateField>();
        borrow.unborrow_weak();
    }

    /// Creates a new immutable borrow from a pointer to the value
    ///
    /// # Safety
    ///
    /// `ptr` must be properly aligned and point to a value of `T` that will be valid for access
    /// for the lifetime `'a`.
    pub unsafe fn from_non_null(ptr: NonNull<T>) -> Self {
        let this = ManuallyDrop::new(Borrowed { ptr, marker: WeakImmut(PhantomData) });
        this.get::<T::BorrowStateField>().borrow_weak();
        ManuallyDrop::into_inner(this)
    }
}

impl<'a, T: Borrowable> Borrowed<Mut<'a, T>> {
    pub fn get_mut<F>(&mut self) -> &mut F::Value
    where
        F: Field<Container = T> + MutField,
    {
        F::as_mut(self)
    }

    /// Turns `self` into a mutable reference to a value behind an `Option<Borrowed<Owned<T>>>`, or
    /// returns `Err(self)` if the field is `None`
    ///
    /// If the `Owned` reference is not unique, it will be updated in-place with `clone()` to make
    /// a unique reference.
    ///
    /// # Safety
    ///
    /// If `clone` is called, the caller guarantees that the resulting strong & weak reference
    /// counts will both be 1.
    pub unsafe fn try_into_some_mut<F, C>(mut self, clone: C) -> Result<Self, Self>
    where
        F: Field<Container = T, Value = Option<Borrowed<Owned<T>>>> + MutField,
        C: for<'c> FnOnce(Borrowed<Immut<'c, T>>) -> T,
    {
        let Some(owned) = self.get_mut::<F>() else {
            // Currently `None`, just return `Err`.
            return Err(self);
        };

        // If it's not unique: clone it and replace the value in the field.
        if !owned.get::<T::StrongCountField>().is_unique() {
            std::hint::cold_path();
            let new_value = Borrowed::alloc_new(clone(owned.borrow()));
            *owned = new_value.erase();
        }

        // At this point: We know that the value at `*owned` is uniquely owned by this thread, so
        // we can hand out a mutable reference to it.
        //
        // `self` will drop as it goes out of scope and release any explicit borrowing we have.
        //
        // SAFETY: See above. Also, `owned.ptr` is guaranteed to be valid by the invariants from
        // the existence of a `Borrowed<Owned<T>>`.
        Ok(unsafe { <Borrowed<Mut<T>>>::from_non_null(owned.ptr) })
    }

    /// Turns `self` into a mutable reference to a value behind an `Option<Borrowed<Owned<T>>>` if
    /// it's already unique, or returns `Err(self)` if the field is `None` or shared.
    pub fn try_into_some_mut_or_discard<F>(mut self) -> Result<Self, Self>
    where
        F: Field<Container = T, Value = Option<Borrowed<Owned<T>>>> + MutField,
    {
        let field = self.get_mut::<F>();

        match field.take().and_then(|b| b.into_unique_or_discard()) {
            None => Err(self),
            Some(unique_owned) => {
                let unique_owned = field.insert(unique_owned.erase());

                // SAFETY: At this point we know that `unique_owned` is actually uniquely owned by
                // this thread, just that it's currently erased to `Owned`, so it's safe to provide
                // exclusive access to it. And `unique_owned.ptr` is guaranteed to be valid for
                // access by the invariants from the existence of a `Borrowed<Owned<T>>`.
                Ok(unsafe { <Borrowed<Mut<T>>>::from_non_null(unique_owned.ptr) })
            }
        }
    }

    /// Inserts `unique` into the field, turning this mutable borrow into one on that field, with
    /// the same lifetime.
    pub fn insert_into_some_unique<F>(mut self, unique: Borrowed<UniqueOwned<T>>) -> Self
    where
        F: Field<Container = T, Value = Option<Borrowed<Owned<T>>>> + MutField,
    {
        let field = self.get_mut::<F>();
        let unique = field.insert(unique.erase());

        // SAFETY: At this point, we know that `unique` is an `&mut` reference to a value that's
        // actually uniquely owned by this thread, just that it's currently erased to `Owned`.
        // So it's safe to provide exclusive access to it. And `unique.ptr` is guaranteed to be
        // valid for access by the invariants from the existence of a `Borrowed<Owned<T>>`.
        unsafe { <Borrowed<Mut<T>>>::from_non_null(unique.ptr) }
    }

    /// Creates a new mutable borrow from a pointer to the value
    ///
    /// # Safety
    ///
    /// `ptr` must be properly aligned and point to a value of `T` that is not otherwise borrowed
    /// (or if it could be otherwise borrowed, `T` must have a `BorrowState` impl that checks.)
    pub unsafe fn from_non_null(ptr: NonNull<T>) -> Self {
        let this = ManuallyDrop::new(Borrowed { ptr, marker: Mut(PhantomData) });

        this.get::<T::BorrowStateField>().borrow_mut();

        ManuallyDrop::into_inner(this)
    }

    /// Implementation of `Drop`
    ///
    /// # Safety
    ///
    /// This method can only be called from the implementation of `Drop`, at most once, and the
    /// value may be left in an invalid state afterwards.
    unsafe fn drop_impl(&mut self) {
        let borrow = self.get::<<T as Borrowable>::BorrowStateField>();
        borrow.unborrow_mut();
    }
}

macro_rules! borrowable {
    (
        $modvis:vis mod $borrowmod:ident;

        $(#[$attrs:meta])*
        $vis:vis struct $ty:ident $(< $( $generic:ident ),+ $(,)? >)?
        $(where $( $boundparam:ident : $boundtrait:ident $(< $($boundp:ident),+ >)? ),+ $(,)? )?
        {
            $(
            $(#[$fieldattrs:meta])*
            $( ($($fieldconfig:tt)*) )? $fieldname:ident : $fieldty:ty,
            )*
        }
    ) => {
        $(#[$attrs])*
        $vis struct $ty $(< $( $generic ),+ >)?
        $( where $( $boundparam : $boundtrait $(< $($boundp),+ >)? ,)+ )?
        {
            $(
            $(#[$fieldattrs])*
            $fieldname : $fieldty,
            )*
        }

        /// Autogenerated by the [`borrowable!`](borrowable) macro
        $modvis mod $borrowmod {
            use super::*;

            borrowable!(
                @generateimpl
                ty = $ty;
                generics = {
                    $( $( $generic ),+ )?
                    $(where $($boundparam : $boundtrait < $($($boundp),+)? > ,)+ )?
                };
                fields = [
                    $($( ($($fieldconfig)*) )? $fieldname = $fieldty),*
                ]
            );
        }
    };
    // base case
    (
        @generateimpl
        ty = $ty:ident;
        generics = {
            $($generics:ident),*
            $(where $($boundparam:ident : $boundtrait:ident < $($boundp:ident),* >  ,)+ )?
        };
        fields = []
    ) => {};
    // recursive case:
    (
        @generateimpl
        ty = $ty:ident;
        generics = {
            $($generics:ident),*
            $(where $($boundparam:ident : $boundtrait:ident < $($boundp:ident),* >  ,)+ )?
        };
        fields = [
            $( ($($fstfieldconfig:tt)*) )? $fstfieldname:ident = $fstfieldty:ty
            $(, $(($($fieldconfigs:tt)*))? $fieldnames:ident = $fieldtys:ty )*
        ]
    ) => {
        #[allow(non_camel_case_types, dead_code)]
        pub struct $fstfieldname < $($generics),* > ( ::std::marker::PhantomData<($($generics),*)> )
        $(where $($boundparam : $boundtrait < $($boundp),+ > ,)+ )?
        ;

        unsafe impl < $($generics),* > $crate::tree::borrow::BorrowableField for $fstfieldname < $($generics),* >
        $(where $($boundparam : $boundtrait < $($boundp),+ > ,)+ )?
        {
            type ContainerType = $ty < $($generics),* >;
            type FieldType = $fstfieldty;

            fn offset() -> usize {
                ::std::mem::offset_of!($ty < $($generics),* >, $fstfieldname)
            }
        }

        borrowable!(
            @access_trait
            generics = {
                $($generics),*
                $(where $($boundparam:$boundtrait < $($boundp),* >  ,)+ )?
            };
            field = $( $($fstfieldconfig)* )? $fstfieldname
        );

        borrowable!(
            @generateimpl
            ty = $ty;
            generics = {
                $($generics),*
                $(where $($boundparam : $boundtrait < $($boundp),+ > ,)+ )?
            };
            fields = [
                $($( ($($fieldconfigs)*) )? $fieldnames = $fieldtys),*
            ]
        );
    };

    (
        @access_trait
        generics = {
            $($generics:ident),*
            $(where $($boundparam:ident : $boundtrait:ident < $($boundp:ident),* >  ,)+ )?
        };
        field = !mut $fieldname:ident
    ) => {
        unsafe impl < $($generics),* > $crate::tree::borrow::ImmutField for $fieldname < $($generics),* >
        $(where $($boundparam : $boundtrait < $($boundp),+ > ,)+ )?
        { }
    };
    (
        @access_trait
        generics = {
            $($generics:ident),*
            $(where $($boundparam:ident : $boundtrait:ident < $($boundp:ident),* >  ,)+ )?
        };
        field = $fieldname:ident
    ) => {
        unsafe impl < $($generics),* > $crate::tree::borrow::MutField for $fieldname < $($generics),* >
        $(where $($boundparam : $boundtrait < $($boundp),+ > ,)+ )?
        { }
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
    type ContainerType: Borrowable;
    type FieldType;

    fn offset() -> usize;
}

/// Not intended for direct use, produced by the `borrowable!` macro
///
/// This trait marks helper types for access to fields where we are allowed to produce mutable
/// references to that field.
///
/// # Safety
///
/// Implementors guarantee that they do not also implement [`ImmutField`].
pub unsafe trait MutField: Field {}

/// Not intended for direct use, produced by the `borrowable!` macro
///
/// This trait marks helper types for access to fields where we are *never* allowed to produce
/// mutable references to that field.
///
/// This distinction is useful for fields like reference counts, where we want to ensure that
/// `Weak` handles on a value are able to remove themselves from the weak count, even when the
/// value is otherwise mutably borrowed (and so the borrow state would disallow it).
///
/// # Safety
///
/// Implementors guarantee that they do not also implement [`MutField`].
pub unsafe trait ImmutField: Field {}

pub trait Field: BorrowableField {
    type Container: Borrowable;
    type Value;

    fn as_ref<'a>(borrow: Borrowed<Immut<'a, Self::Container>>) -> &'a Self::Value;
    fn as_mut<'a>(borrow: &'a mut Borrowed<Mut<'_, Self::Container>>) -> &'a mut Self::Value;
}

impl<F: BorrowableField> Field for F {
    type Container = F::ContainerType;
    type Value = F::FieldType;

    fn as_ref<'a>(borrow: Borrowed<Immut<'a, Self::Container>>) -> &'a F::FieldType {
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

    fn as_mut<'a>(borrow: &'a mut Borrowed<Mut<'_, Self::Container>>) -> &'a mut F::FieldType {
        let container_ptr = borrow.ptr.as_ptr();
        // SAFETY: Same as for `as_ref()`.
        unsafe {
            let field_ptr = (container_ptr as *mut u8).add(F::offset()) as *mut F::FieldType;
            &mut *field_ptr
        }
    }
}
