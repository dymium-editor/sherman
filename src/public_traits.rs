//! Public-facing traits for implementing [`RleTree`](crate::RleTree)

use std::fmt::Debug;
use std::ops::{Add, Sub};

/// Blanket trait for types that can be used as an index in an [`RleTree`]
///
/// This is abstracted, instead of just `usize` because there's some cases where it's valuable to
/// allow multiple dimensions in indexes -- e.g., line & column number pairs. Treating these
/// abstractly means that cases don't need to maintain a persistent 1D to 2D mapping.
///
/// An implementation of this trait is already provided for types that implement the component
/// supertraits.
///
/// [`RleTree`]: crate::RleTree
pub trait Index: Debug + Copy + Ord + Zero + DirectionalAdd + DirectionalSub {}

impl<I: Debug + Copy + Ord + Zero + DirectionalAdd + DirectionalSub> Index for I {}

/// Trait for values that can be stored in an [`RleTree`]
///
/// Because values are run-length encoded, there must be some way of splitting runs (for insertions
/// in the middle) or joining runs of the same value. Abstracting this allows for more interesting
/// behaviors, but this is still fundamentally what this trait represents.
///
/// `Slice`s implicitly have an attached length, but that length is separately tracked by the
/// containing [`RleTree`] from the moment of its creation.
///
/// [`RleTree`]: crate::RleTree
pub trait Slice<Idx>: Sized /* Sized required to return Self */ {
    /// Optimization: If false, `try_join` will never be attempted, which allows [`RleTree`]
    /// operations to skip somewhat-costly
    ///
    /// [`RleTree`]: crate::RleTree
    //
    // TODO: We could probably eliminate the `key_hole` field in `crate::tree::node::Leaf` by
    // making this trait member a type instead of a const. `key_hole` isn't necessary when
    // `MAY_JOIN` is false, but that's also not too many implementations.
    const MAY_JOIN: bool;

    /// Splits the slice at given index, setting `self ≈ self[..idx]` and returning `self[idx..]`.
    /// It can be assumed that the index is non-zero and less than the length of the slice, though
    /// you shouldn't rely upon this for unsafe code.
    ///
    /// For implementations of `Slice` that *are* just run-length encodings (like [`Constant`]),
    /// simply copying the inner value suffices -- the index isn't necessary.
    ///
    /// Note for clarity: The provided index here is *within* the range covered by the slice, i.e.
    /// not global.
    ///
    /// [`Constant`]: crate::Constant
    fn split_at(&mut self, idx: Idx) -> Self;

    /// Attempts to join two slices into one, returning `Ok(joined)` or `Err((self, other))`
    ///
    /// This will *never* be called unless `MAY_JOIN` is `true`. The default implementation always
    /// errors (i.e. never successfully joins).
    ///
    /// Additionally, if `MAY_JOIN` is true, this function will *always* be called whenever the
    /// slice on either side of this one changes -- but only at most once. In general, we assume
    /// that joining means there's *some* level of equality between slices, but that it may depend
    /// on position. So the equality should be transitive if the ordering of slices isn't changed:
    /// if `x.try_join(y)` succeeds but `y.try_join(z)` doesn't, then `x.try_join(y).try_join(z)`
    /// shouldn't, either. However it could still be the case that `z.try_join(x.try_join(y))`
    /// succeeds.
    ///
    /// Any calls to this function will have `self` at a position immediately less than `other`, in
    /// order to uphold this notion of positions.
    fn try_join(self, other: Self) -> Result<Self, (Self, Self)> {
        Err((self, other))
    }
}

/// Types that have a "zero" value, in the mathematical sense
///
/// Put simply, types should only implement `Zero` if there is some constant `ZERO` such that any
/// addition or subtraction by `ZERO` returns the original value -- i.e.
/// `x + ZERO == x == x - ZERO`. Addition and subtraction is a little bit more relaxed here,
/// instead referring to [`DirectionalAdd`] and [`DirectionalSub`].
///
/// Unlike [`num::Zero`], we use a constant for this trait's zero value, because of its use as a
/// supertrait of [`Index`] -- we need it to be guaranteed to be essentially "zero"-cost.
///
/// [`num::Zero`]: https://docs.rs/num/0.4.0/num/trait.Zero.html
pub trait Zero {
    /// Constant value of zero
    const ZERO: Self;
}

/// Directional-arithmetic counterpart to [`std::ops::Add`]
///
/// This trait goes alongside [`DirectionalSub`] as supertraits of [`Index`]. The explainer on what
/// directional arithmetic *actually* is, and why we have it.
///
/// A blanket implementation is provided for all `T: Add<Output = Self>`, so **you do not need to
/// interact with this trait if you are not writing a custom index type**. That said, this
/// documentation is equally provided for the merely curious.
///
/// ## Directional arithmetic -- Motivation
///
/// First off, it's worth addressing why we've gone to all these lengths to have "directional
/// arithmetic" in the first place -- why couldn't the plain `Add` and `Sub` traits have worked?
///
/// Well, one of the abilities we *needed* to have when designing this crate was to be able to use
/// two-dimensional indexes -- e.g., line & column number. It turns out these are really tricky to
/// fit into a one-dimensional system, particularly because any single index for 2D values won't
/// have commutative addition/subtraction. But it *is* possible with some restrictions, and much of
/// the weird pieces of design in this crate are essentially required in order to enable this
/// functionality.
///
/// Thankfully for you, dear reader, any [`Index`] type for line/column pairs works as an effective
/// approximation of the minimal set of guarantees provided by an [`Index`] type, so we'll tend to
/// use that in the explanations here -- either implicitly (illustrations of ranges) or explicitly
/// (referring to some hypothetical file, with line/column pairs as points within it).
///
/// ---
///
/// The first key thing to notice about [`Index`] types is that they *also* represent the offset
/// between two points -- this is required internally for [`RleTree`], but also implicit in the API
/// of [`Slice`]. If we were just looking at integers, we might've assumed that this was part of
/// indexes without second thought, but 2D index types require careful consideration.
///
/// The essence of it is that offsets between 2D index types only *really* make sense when used
/// between the original locations. This is basically what directional arithmetic provides --
/// operations where the original positions of the values are explicit in the function. In the
/// documentation for each of the arithmetic operations, there are diagrams showing the
/// relationship between the ranges of the inputs and outputs.
///
/// It's precisely the need for operations to use the original locations that requires directional
/// arithmetic. Addition can have a default implementation (as `add_left` already does), but
/// there's no way to represent either `sub_left` or `sub_right` in terms of any of the other
/// operations.
///
/// ## Directional Arithmetic -- Precise semantics
///
//
// TODO: There's some work that could be done here to clarify the relationship between indexes and
// ranges. The essential idea is that indexes are offsets, and offsets are attched to unique
// ranges, so operations on indexes require that the ranges line up right.
//
///
/// This section gets a little bit notation-heavy. This is only really intended to be used if
/// you're testing the implementation of some [`Index`] type to make sure that it upholds the
/// necessary properties. The general idea here is that indexes are only valid when used with
/// something from their original range, so any implementation that operates within those bounds
/// should be ok. The expected behavior of addition and subtraction *mostly* follows from the idea
/// of operations on ranges.
///
/// When we say "original range", really what we're talking about is that -- conceptually -- the
/// ranges at the borders of a range must remain the same. So if we know that `x.add_right(y)` is
/// valid, `x.add_right(y.add_right(z))` must also be, but `x.add_right(z.add_right(y))` might not.
/// Expressed visually:
///
/// ```text
/// |--- x.add_right(y) --|  (ok)
/// |---- x ----|--- y ---|
///
/// |- x.add_right(y.add_right(z)) -|  (ok)
///             |-- y.add_right(z) -|
/// |---- x ----|--- y ---|--- z ---|
///
/// |- x.add_right(z.add_right(y)) -|  (not ok, out of order)
///             |-- z.add_right(y) -|
/// |---- x ----|--- z ---|--- y ---|
/// ```
///
/// In the final case, the result is undefined -- implementors can return whatever value they like,
/// or panic (though you must not trigger UB). It is the caller's (i.e. *our*) responsibility to
/// ensure that invalid operations are never attempted.
///
/// The above property is a bit difficult to formalize, so I haven't attempted to here -- it is my hope
/// that the description as is should be clear *enough*. If there are any doubts, feel free to
/// reach out.
///
/// ---
///
/// The remaining properties are all equalities. They can get a bit dense, so we'll introduce some
/// notation to help out: we'll write `add_right` as `+>`, `add_left` as `<+`, `sub_right` as `->`,
/// and `sub_left` as `<-`. The properties are then:
///
///  * Commutativity: `x +> y == y <+ x`
///  * Associativity: `(x +> y) +> z == x +> (y +> z)`
///  * Subtraction as the inverse of addition: `(x +> y) -> y == x == (x <+ z) <- z` and
///    `(x -> y) +> y == x == (x <- z) <+ z`
///  * Idempotency of zero: if `Self: Zero`, then `x +> 0 == x == x <+ 0` and
///    `x -> 0 == x == x <- 0`
///
/// We also have distributivity over subtraction, which is each of the following six equalities:
///
///  * `x -> (y <+ z) == (x -> y) -> z`
///  * `x <- (y +> z) == (x <- y) <- z`
///
///  * `x -> (y -> z) == (x +> z) -> y`
///  * `x <- (y <- z) == (x <+ z) <- y`
///
///  * `x -> (y <- z) == (x -> y) +> z`
///  * `x <- (y -> z) == (x <- y) <+ z`
///
/// All of the above properties are expected to be *logically* valid over an infinite-ish range in
/// either direction (i.e., including negative numbers), but may *in practice* fail to produce
/// values outside of a certain range (e.g., above a maximum value, or below zero). This allows us
/// to still reason about [`Index`]es using negative numbers without requring that the types
/// themselves be signed.
///
/// The justification for most of the properties should hopefully make sense, but distributivity
/// can be particularly hard to visualize. If you are interested, the source for this trait
/// contains detailed ASCII-art illustrations to provide some justification for the rules above.
///
/// [`RleTree`]: crate::RleTree
//
// Illustrations for distributivity rules:
//
//   x -> (y <+ z)         (x -> y) -> z                 x <- (y +> z)          (x <- y) <- z
//  |      x      |   =>    |  x -> y |                 |      x      |   =>     | x <- y  |
//        | z | y |               | z |                 | y | z |                | z |
//
//
//   x -> (y -> z)          (x +> z) -> y                x <- (y <- z)           (x <+ z) <- y
//  |    x    |            |    x +> z   |                  |    x    |         |   x <+ z    |
//        |   y   |   =>         |   y   |              |   y   |         =>    |   y   |
//            | z |                                     | z |
//
//
//   x -> (y <- z)         (x -> y) +> z                 x <- (y -> z)         (x <- y) <+ z
//  |      x      |         | x->y|                     |      x      |             | x->y|
//        |   y   |   =>          | z |                 |   y   |         =>    | z |
//        | z |                                             | z |
pub trait DirectionalAdd: Sized {
    /// Adds a value to the left
    ///
    /// The default implementation returns `left.add_right(self)`
    ///
    /// Visually, this is:
    ///
    /// ```text
    /// |--- left ---|--- self ---|
    /// |-- self.add_left(left) --|
    /// ```
    fn add_left(self, left: Self) -> Self {
        left.add_right(self)
    }

    /// Adds a value to the right
    ///
    /// Visually, this is:
    ///
    /// ```text
    /// |---- self ---|--- right ---|
    /// |-- self.add_right(right) --|
    /// ```
    fn add_right(self, right: Self) -> Self;
}

/// Directional-arithmetic counterpart to [`std::ops::Sub`]
///
/// This trait goes alongside [`DirectionalAdd`] as supertraits of [`Index`]. For information about
/// directional arithmetic, please refer to the documentation on [`DirectionalAdd`].
///
/// Like [`DirectionalAdd`], a blanket implementation is provided for all `T: Sub<Output = Self>`.
pub trait DirectionalSub {
    /// Subtracts a value from the left
    ///
    /// Visually, this is:
    ///
    /// ```text
    /// |--------------- self ---------------|
    /// |-- left --|-- self.sub_left(left) --|
    /// ```
    fn sub_left(self, left: Self) -> Self;

    /// Subtracts a value from the right
    ///
    /// Visually, this is:
    ///
    /// ```text
    /// |----------------- self ----------------|
    /// |-- self.sub_right(right) --|-- right --|
    /// ```
    fn sub_right(self, right: Self) -> Self;
}

impl<T: Add<Output = Self>> DirectionalAdd for T {
    fn add_right(self, right: Self) -> Self {
        self + right
    }
}

impl<T: Sub<Output = Self>> DirectionalSub for T {
    fn sub_left(self, left: Self) -> Self {
        self - left
    }

    fn sub_right(self, right: Self) -> Self {
        self - right
    }
}

macro_rules! impl_for_unsigned_primitive {
    ($ty:ident) => {
        impl Zero for $ty {
            const ZERO: $ty = 0;
        }
    };
}

impl_for_unsigned_primitive!(u8);
impl_for_unsigned_primitive!(u16);
impl_for_unsigned_primitive!(u32);
impl_for_unsigned_primitive!(u64);
impl_for_unsigned_primitive!(u128);
impl_for_unsigned_primitive!(usize);
