//! Types to go with a custom [`std::ops::RangeBounds`] trait
//!
//! This crate uses a custom range interface in order to make a number of operations simpler. It
//! doesn't change too much -- just removes the ability for excluded start bounds, which would
//! require an increment operator to successfully use with [`RleTree`].
//!
//! The redefined [`RangeBounds`] trait is implemented for all the standard library's range types,
//! and contains a number of methods that wouldn't be possible with exclusive start bounds.
//!
//! This module also contains distinct [`StartBound`] and [`EndBound`] types adapted from the
//! standard library's [`Bound`](std::ops::Bound).
//!
//! [`RleTree`]: crate::RleTree

#[cfg(feature = "fuzz")]
use arbitrary::Arbitrary;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// The starting bound of a range
///
/// Refer to the [module documentation](self) for more information.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "fuzz", derive(Arbitrary))]
pub enum StartBound<T> {
    Included(T),
    Unbounded,
}

/// The ending bound of a range
///
/// Refer to the [module documentation](self) for more information.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "fuzz", derive(Arbitrary))]
pub enum EndBound<T> {
    Included(T),
    Excluded(T),
    Unbounded,
}

/// A `std::ops::RangeBounds`-like trait that disallows exclusive start bounds
///
/// Aside from disallowing exclusive starting bounds, this trait is essentially the same as the
/// standard library's version. We actually also provide implementations for all of the range types
/// in [`std::ops`] as well, because none of them have exclusive start bounds.
///
/// We also provide a number of additional methods that are not (and cannot) be provided by the
/// standard library, notably: [`overlaps_naive`] and [`starts_after_end`].
///
/// [`overlaps_naive`]: Self::overlaps_naive
/// [`starts_after_end`]: Self::starts_after_end
pub trait RangeBounds<T> {
    /// Returns the starting bound of the range
    fn start_bound(&self) -> StartBound<&T>;
    /// Returns the ending bound of the range
    fn end_bound(&self) -> EndBound<&T>;

    /// Returns `true` if `item` is contained within the range
    fn contains<U>(&self, item: &U) -> bool
    where
        T: PartialOrd<U>,
        U: ?Sized + PartialOrd<T>,
    {
        // This implementation is essentially taken directly from the standard library's version of
        // this function.
        (match self.start_bound() {
            StartBound::Included(start) => start <= item,
            StartBound::Unbounded => true,
        }) && (match self.end_bound() {
            EndBound::Included(end) => item <= end,
            EndBound::Excluded(end) => item < end,
            EndBound::Unbounded => true,
        })
    }

    /// Returns whether the start of the range is after the end
    ///
    /// This can be used to identify invalid ranges passed as input, like with [`RleTree::iter`].
    ///
    /// [`RleTree::iter`]: crate::RleTree::iter
    fn starts_after_end(&self) -> bool
    where
        T: PartialOrd<T>,
    {
        match (self.start_bound(), self.end_bound()) {
            (StartBound::Unbounded, _) | (_, EndBound::Unbounded) => false,

            (StartBound::Included(x), EndBound::Included(y)) => x > y,
            (StartBound::Included(x), EndBound::Excluded(y)) => x >= y,
        }
    }

    /// Returns `true` if there are no values for which `self.contains(value)` is true -- with some
    /// exceptions
    ///
    /// **Be warned:** This method may return incorrect results when one of the bounds is the
    /// maximum or minimum value and the other is unbounded. For example:
    ///
    /// ```
    /// # use sherman::range::RangeBounds;
    /// // Doesn't panic:
    /// assert!( !(..0_u8).is_empty_naive() );
    /// ```
    ///
    /// Because there aren't any values less than zero, we can see that this should be false. But
    /// the general implementation doesn't have that luxury, and so it will incorrectly classify
    /// ranges like `..0`.
    ///
    /// For our usage, we check with [`T::ZERO`](crate::Zero) to catch this case. The
    /// implementation is guaranteed to be correct for all standard range types except [`RangeTo`].
    fn is_empty_naive(&self) -> bool
    where
        T: Ord,
    {
        match (self.start_bound(), self.end_bound()) {
            (StartBound::Unbounded, _) | (_, EndBound::Unbounded) => false,

            (StartBound::Included(x), EndBound::Included(y)) => x > y,
            (StartBound::Included(x), EndBound::Excluded(y)) => x >= y,
        }
    }

    /// Returns `true` if the other range overlaps with `self` -- with some exceptions
    ///
    /// ## Semantics
    ///
    /// For this method, "overlapping" means that there is at least one value `v` such that
    /// `self.contains(v)` and `range.contains(v)`. This means that we return `false` if either
    /// range is empty.
    ///
    /// **Be warned:** This method may return incorrect results when both ranges share an unbounded
    /// side. The limitations causing this are discussed at further length in the documentation for
    /// [`is_empty_naive`](Self::is_empty_naive).
    ///
    /// This method is still guaranteed to be correct if neither range shares an unbounded edge
    /// side -- either on the start or end. It is in some sense *more correct* than
    /// `is_empty_naive` in this respect (which may be incorrect for any range with an unbounded
    /// edge).
    fn overlaps_naive<R>(&self, other: R) -> bool
    where
        T: Ord,
        R: RangeBounds<T>,
    {
        if self.is_empty_naive() || other.is_empty_naive() {
            return false;
        }

        // If `self` starts after the end of `range`, they cannot overlap
        match (self.start_bound(), other.end_bound()) {
            (StartBound::Included(s), EndBound::Excluded(e)) if s >= e => return false,
            (StartBound::Included(s), EndBound::Included(e)) if s > e => return false,
            _ => (),
        }

        // Similarly, if `self` ends before the start of `range`, they also cannot overlap
        match (self.end_bound(), other.start_bound()) {
            (EndBound::Excluded(e), StartBound::Included(s)) if e <= s => return false,
            (EndBound::Included(e), StartBound::Included(s)) if e < s => return false,
            _ => (),
        }

        true
    }
}

impl<T> StartBound<T> {
    fn as_ref(&self) -> StartBound<&T> {
        match self {
            StartBound::Included(v) => StartBound::Included(v),
            StartBound::Unbounded => StartBound::Unbounded,
        }
    }
}

impl<T: Clone> StartBound<&T> {
    pub fn cloned(&self) -> StartBound<T> {
        match self {
            StartBound::Included(v) => StartBound::Included(T::clone(v)),
            StartBound::Unbounded => StartBound::Unbounded,
        }
    }
}

impl<T> EndBound<T> {
    fn as_ref(&self) -> EndBound<&T> {
        match self {
            EndBound::Included(v) => EndBound::Included(v),
            EndBound::Excluded(v) => EndBound::Excluded(v),
            EndBound::Unbounded => EndBound::Unbounded,
        }
    }
}

impl<T: Clone> EndBound<&T> {
    pub fn cloned(&self) -> EndBound<T> {
        match self {
            EndBound::Included(v) => EndBound::Included(T::clone(v)),
            EndBound::Excluded(v) => EndBound::Excluded(T::clone(v)),
            EndBound::Unbounded => EndBound::Unbounded,
        }
    }
}

// Blanket implementation for references
impl<T, R> RangeBounds<T> for &R
where
    R: RangeBounds<T>,
{
    fn start_bound(&self) -> StartBound<&T> {
        (*self).start_bound()
    }
    fn end_bound(&self) -> EndBound<&T> {
        (*self).end_bound()
    }
}

impl<T> RangeBounds<T> for (StartBound<T>, EndBound<T>) {
    fn start_bound(&self) -> StartBound<&T> {
        self.0.as_ref()
    }
    fn end_bound(&self) -> EndBound<&T> {
        self.1.as_ref()
    }
}

impl<T> RangeBounds<T> for (StartBound<&T>, EndBound<&T>) {
    fn start_bound(&self) -> StartBound<&T> {
        self.0
    }
    fn end_bound(&self) -> EndBound<&T> {
        self.1
    }
}

impl<T> RangeBounds<T> for RangeInclusive<T> {
    fn start_bound(&self) -> StartBound<&T> {
        StartBound::Included(self.start())
    }
    fn end_bound(&self) -> EndBound<&T> {
        EndBound::Included(self.end())
    }
}
impl<T> RangeBounds<T> for RangeInclusive<&T> {
    fn start_bound(&self) -> StartBound<&T> {
        StartBound::Included(*self.start())
    }
    fn end_bound(&self) -> EndBound<&T> {
        EndBound::Included(*self.end())
    }
}

impl<T> RangeBounds<T> for RangeFull {
    fn start_bound(&self) -> StartBound<&T> {
        StartBound::Unbounded
    }
    fn end_bound(&self) -> EndBound<&T> {
        EndBound::Unbounded
    }
}

// Helper macro to implement `RangeBounds` for the ranges with `start` and `end` fields that we can
// use.
macro_rules! impl_rangebounds {
    ( $base_ty:ident: $start:ident .. $end:ident ) => {
        impl<T> RangeBounds<T> for $base_ty<T> {
            impl_rangebounds!(@bound ref start_bound StartBound start $start);
            impl_rangebounds!(@bound ref end_bound EndBound end $end);
        }

        impl<T> RangeBounds<T> for $base_ty<&T> {
            impl_rangebounds!(@bound noref start_bound StartBound start $start);
            impl_rangebounds!(@bound noref end_bound EndBound end $end);
        }
    };

    (@bound $ref:tt $method:ident $ty:ident $field:ident Included) => {
        fn $method(&self) -> $ty<&T> {
            $ty::Included( impl_rangebounds!(@maybe_ref $ref self.$field) )
        }
    };
    (@bound $ref:tt $method:ident $ty:ident $field:ident Excluded) => {
        fn $method(&self) -> $ty<&T> {
            $ty::Excluded( impl_rangebounds!(@maybe_ref $ref self.$field) )
        }
    };
    (@bound $ref:tt $method:ident $ty:ident $get:ident Unbounded) => {
        fn $method(&self) -> $ty<&T> {
            $ty::Unbounded
        }
    };

    (@maybe_ref ref $($args:tt)*) => { &( $($args)* ) };
    (@maybe_ref noref $($args:tt)*) => { $($args)* };
}

impl_rangebounds!(Range: Included..Excluded);
impl_rangebounds!(RangeFrom: Included..Unbounded);
impl_rangebounds!(RangeTo: Unbounded..Excluded);
impl_rangebounds!(RangeToInclusive: Unbounded..Included);
