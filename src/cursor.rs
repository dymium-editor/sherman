//! Wrapper module for [`Cursor`] and related types

use std::fmt::{self, Debug, Formatter};

/// Interface to store prior paths through the [`RleTree`], so that they can be used to speed up
/// subsequent accesses
///
/// An implementation for fixed-size cursors is supplied with the [`BoundedCursor`] type, but the
/// trait is public so that it may be implemented by anyone. There's also the [`NoCursor`] type,
/// which has an implementation that does nothing.
///
/// [`RleTree`]: crate::RleTree
pub trait Cursor: Sized {
    /// Marker for whether the cursor does nothing
    ///
    /// Practically speaking, this only really exists so that we can optimize uses of [`NoCursor`],
    /// but you're welcome to set this if you'd like :)
    const IS_NOP: bool = false;

    /// Creates a new `Cursor` with an empty path
    fn new_empty() -> Self;

    /// After creating a new `Cursor`, we repeatedly call this method while traversing up the tree
    /// to build the path -- from the end to the start
    fn prepend_to_path(&mut self, component: PathComponent);

    /// An iterator over the components of the path, produced by `into_path`
    type PathIter: Iterator<Item = PathComponent>;

    /// Turns the cursor into an iterator over the components of the path, from start to end
    /// (i.e. the opposite order they are provided with `prepend_to_path`)
    fn into_path(self) -> Self::PathIter;
}

/// Single component in a [`Cursor`]'s path through the tree
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathComponent {
    pub(crate) child_idx: u8,
}

/// [`Cursor`] with a bounded length
///
/// The type uses an internal buffer of size `LEN`, stored in-line (i.e. on the stack, for normal
/// use). We default to a length of 8 -- which should provide a majority of the benefit for *most*
/// [`RleTree`]s, though some applications may desire more.
///
/// A length-bounded cursor is useful because the exponential growth in B-tree size with height
/// means that it's essentially impossible to get trees that are above a certain height. In our
/// case, the default configuration of [`RleTree`]s means that a height of 8 has *at minimum* 600k
/// values. Any additional searching beyond that point is likely a reasonable cost to pay (note:
/// still O(log n) -- cursors are just O((log n) / M)).
///
/// Note: this type expects that the length won't be greater than `u8::MAX` -- i.e. `LEN` cannot be
/// more than 255. Calls to [`Cursor::new_empty`] will fail if this isn't the case, but this
/// *really* should not be an issue.
///
/// [`RleTree`]: crate::RleTree
#[derive(Copy, Clone)]
#[cfg_attr(any(test, feature = "fuzz"), derive(PartialEq, Eq))]
pub struct BoundedCursor<const LEN: usize = 8> {
    path: [PathComponent; LEN],
    start_at: u8,
}

impl<const LEN: usize> Debug for BoundedCursor<LEN> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        struct P<'a>(&'a [PathComponent]);

        impl Debug for P<'_> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                let absent = format_args!("-");

                let mut list = f.debug_list();
                for c in self.0 {
                    if c.child_idx == 0xFF {
                        list.entry(&absent);
                    } else {
                        list.entry(&c.child_idx);
                    }
                }

                list.finish()
            }
        }

        f.debug_struct("BoundedCursor")
            .field("start_at", &self.start_at)
            .field("path", &P(&self.path))
            .finish()
    }
}

impl<const LEN: usize> Cursor for BoundedCursor<LEN> {
    const IS_NOP: bool = LEN == 0;

    fn new_empty() -> Self {
        let len_u8 = match u8::try_from(LEN) {
            Err(_) => panic!("tried to construct BoundedCursor with LEN > u8::MAX"),
            Ok(0) => 0,
            Ok(len) => len - 1,
        };

        BoundedCursor {
            path: [PathComponent { child_idx: 0xFF }; LEN],
            start_at: len_u8,
        }
    }

    fn prepend_to_path(&mut self, component: PathComponent) {
        if self.start_at == 0 {
            if LEN == 0 {
                return;
            }

            // if the cursor is already full,
            self.path.copy_within(0..LEN - 1, 1);
            self.path[0] = component;
        } else {
            self.path[self.start_at as usize] = component;
            self.start_at -= 1;
        }
    }

    type PathIter = Self;

    fn into_path(self) -> Self {
        self
    }
}

impl<const LEN: usize> Iterator for BoundedCursor<LEN> {
    type Item = PathComponent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_at as usize == LEN {
            return None;
        }

        let c = self.path[self.start_at as usize];
        self.start_at += 1;
        Some(c)
    }
}

/// Dummy implementor of [`Cursor`] that does nothing
///
/// This type is provided in case it's something you need -- both for convenience and because the
/// implementation of [`RleTree::insert`] (the method that *doesn't* use a cursor) uses it
/// internally.
///
/// [`RleTree::insert`]: crate::RleTree::insert
#[derive(Copy, Clone)]
pub struct NoCursor;

#[rustfmt::skip]
impl Cursor for NoCursor {
    const IS_NOP: bool = true;
    fn new_empty() -> Self { NoCursor }
    fn prepend_to_path(&mut self, _: PathComponent) {}
    type PathIter = std::iter::Empty<PathComponent>;
    fn into_path(self) -> Self::PathIter { std::iter::empty() }
}

/// Handler for creating a [`Cursor`] without needing to be generic
pub(crate) struct CursorBuilder<'c> {
    builder: Option<&'c mut dyn CursorBuilderDispatch>,
}

// note: we require MaybeDebug so that the Debug impl for CursorBuilder is possible
trait CursorBuilderDispatch: crate::MaybeDebug {
    fn reset(&mut self);
    fn prepend_to_path(&mut self, component: PathComponent);
}

impl<C: Cursor> CursorBuilderDispatch for C {
    fn reset(&mut self) {
        *self = C::new_empty();
    }

    fn prepend_to_path(&mut self, component: PathComponent) {
        self.prepend_to_path(component);
    }
}

impl<'c> CursorBuilder<'c> {
    /// Creates a new `CursorBuilder` borrowing the provided `Cursor`
    pub(crate) fn new<C: Cursor>(cursor: &'c mut C) -> Self {
        CursorBuilder {
            builder: match C::IS_NOP {
                true => None,
                false => Some(cursor),
            },
        }
    }

    /// Returns the value of the inner cursor's [`IS_NOP`](Cursor::IS_NOP)
    pub(crate) fn is_nop(&self) -> bool {
        self.builder.is_none()
    }

    /// Dynamic dispatch for [`Cursor::prepend_to_path`]
    pub(crate) fn prepend_to_path(&mut self, component: PathComponent) {
        if let Some(c) = self.builder.as_mut() {
            c.prepend_to_path(component);
        }
    }

    /// Dynamic dispatch for `*cursor = Cursor::new_empty()`
    pub(crate) fn reset(&mut self) {
        if let Some(c) = self.builder.as_mut() {
            c.reset();
        }
    }

    /// Convenience method to reborrow the `CursorBuilder`, much like you would a `&mut` reference
    ///
    /// This allows us to always pass around `CursorBuilder` objects directly, which means we never
    /// have to double-dereference.
    pub(crate) fn reborrow<'b>(&'b mut self) -> CursorBuilder<'b> {
        CursorBuilder {
            builder: match self.builder.as_mut() {
                None => None,
                Some(c) => Some(*c),
            },
        }
    }
}
