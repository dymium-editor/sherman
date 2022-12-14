//! Management of nodes
//!
//! This file takes heavy inspiration from the standard library's implementation of B-trees, but
//! separate explanations are given here as well.
//!
//! Like the standard library, we're determining the type of a node (either [`Leaf`] or
//! [`Internal`]) by the height at which we're accessing it. For this, we use [`NodeHandle`]s,
//! which all originate with a single "owned" handle at the root of the tree. As we traverse the
//! tree, we appropriately decrement the height of the handle(s), interpreting the node as a
//! [`Leaf`] only once the height is zero.
//!
//! `NodeHandle`s are parameterized over a few things, *in addition* to the parameters of the tree
//! itself (`I`, `S`, `P`, and `const M`). These parameters indicate the type of node, if known
//! (`Ty`) and dictate the type of access that the handle has to the tree (the "borrow" -- `B`).
//! These are defined in the [`ty`] and [`borrow`] modules, respectively.
//!
//! Because the type of a node is uniquely determined by the handle's height, we've made the
//! `height` field of `NodeHandle` parameterized by the type itself -- internal nodes have a
//! non-zero height, so `NonZeroU8`; "unknown" nodes can be anything, so `u8`; and leaf nodes are
//! always zero, so `ZeroU8` (a `#[repr(u8)]` type that's always zero). These are handled by the
//! [`TypeHint`] and [`Height`] traits.

use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::num::NonZeroU8;
use std::ops::{Range, RangeFrom};
use std::ptr::{self, addr_of_mut, NonNull};
use std::slice;

#[cfg(test)]
use crate::MaybeDebug;
#[cfg(test)]
use std::fmt::{self, Debug, Formatter};

use super::NodeBox;
use crate::const_math_hack::{self as hack, ArrayHack, GradualInitArray, MappedArray};
use crate::param::{self, RleTreeConfig, SliceRefStore, StrongCount, SupportsInsert};
use crate::public_traits::Index;

use self::ty::{Height, TypeHint};

// Helper macro for referring to parameter-related associated types. Without this, it gets *very*
// verbose, and much harder to read.
macro_rules! resolve {
    ( $base:ident :: StrongCount ) => {
        <$base as RleTreeConfig<I, S, M>>::StrongCount
    };
    ( $base:ident :: SliceRefStore ) => {
        <$base as RleTreeConfig<I, S, M>>::SliceRefStore
    };
    ( $base:ident :: SliceRefStore :: OptionRefId ) => {
        <
            <$base as RleTreeConfig<I, S, M>>::SliceRefStore as param::SliceRefStore<I, S, P, M>
        >::OptionRefId
    };
}

/// The typical size, in bytes, of a CPU cache line
///
/// This is relevant as a micro-optimization because the CPU will generally load entire cache lines
/// at a time into the L1/L2 cache,
const CACHE_LINE_SIZE: usize = 64;

/// Helper alias for a pointer to a node
pub(super) type NodePtr<I, S, P, const M: usize> = NonNull<AbstractNode<I, S, P, M>>;

/// A node of indeterminate type -- either a [`Leaf`] or [`Internal`]
///
/// An "abstract" node is really just a newtype'd `Leaf`, so that the pointer carries the type
/// information we need without running the risk of being unaligned. The actual type of the
/// allocation is determined by the height of the [`NodeHandle`] used to access this node.
pub struct AbstractNode<I, S, P, const M: usize>(Leaf<I, S, P, M>)
where
    P: RleTreeConfig<I, S, M>;

/// Reference to a node in the tree, abstracted over borrowing and node-type hints
///
/// The parameters `I`, `S`, `P`, and `M` are passed directly to the inner node, but `T` and `B`
/// are used for type hints and borrowing, respectively.
///
/// Borrowing assumes that if a handle has a certain type of access to *one* node, it has the same
/// type of access to all other nodes it can reach -- including through parent pointers.
///
/// ## Safety
///
/// This type is `#[repr(C)]` so that the layout is guaranteed to be the same if `T::Height` is the
/// same. This is true for all `TypeHint`s (they're all equivalent to `u8`s), which means we *can*
/// transmute between `NodeHandle` parameterizations, or reinterpret references.
//
// Needs to be `pub` so that we can expose it via `RawSliceRef`
#[repr(C)]
pub struct NodeHandle<Ty, B, I, S, P, const M: usize>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    ptr: NodePtr<I, S, P, M>,
    /// The height of the tree rooted at this node
    ///
    /// All nodes at a `height` of zero are leaf nodes, and all nodes at `height > 0` are internal
    /// nodes. This is enforced by the implementations of [`TypeHint`] for the various types, where
    /// `u8` is only used for `ty::Unknown`.
    height: Ty::Height,
    /// Borrow marker indicating what this `NodeHandle` is allowed to access
    borrow: PhantomData<B>,
}

/// Handle on a *particular* slice in a node
#[repr(C)]
pub struct SliceHandle<Ty, B, I, S, P, const M: usize>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    pub(super) node: NodeHandle<Ty, B, I, S, P, M>,
    /// The index of the particular slice in the node
    ///
    /// ## Safety
    ///
    /// Existence of this `SliceHandle` guarantees that `idx < node.len`; you can assume it without
    /// checking.
    pub(super) idx: u8,
}

#[cfg(test)]
impl<Ty, B, I, S, P, const M: usize> Debug for NodeHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let ty = match self.typed_ref() {
            Type::Leaf(_) => "Leaf",
            Type::Internal(_) => "Internal",
        };

        let mut s = f.debug_struct(ty);
        s.field("addr", &self.ptr);
        s.field("height", &self.height.as_u8());
        match self.typed_ref() {
            Type::Leaf(node) => node.leaf().append_fields(&mut s),
            Type::Internal(node) => node.internal().append_fields(&mut s),
        }
        s.finish()
    }
}

#[cfg(test)]
impl<Ty, B, I, S, P, const M: usize> Debug for SliceHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("SliceHandle")
            .field("idx", &self.idx)
            .field("node", &self.node)
            .finish()
    }
}

/////////////////////////////////////////////////////
// impl Copy for suitable NodeHandles/SliceHandles //
/////////////////////////////////////////////////////

#[rustfmt::skip]
impl<'t, T: TypeHint, I, S, P, const M: usize> Copy for NodeHandle<T, borrow::Immut<'t>, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{}

#[rustfmt::skip]
impl<'t, T: TypeHint, I, S, P, const M: usize> Clone for NodeHandle<T, borrow::Immut<'t>, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn clone(&self) -> Self { *self }
}

#[rustfmt::skip]
impl<'t, T: TypeHint, I, S, P, const M: usize> Copy for SliceHandle<T, borrow::Immut<'t>, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{}

#[rustfmt::skip]
impl<'t, T: TypeHint, I, S, P, const M: usize> Clone for SliceHandle<T, borrow::Immut<'t>, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    fn clone(&self) -> Self { *self }
}

//////////////////////////////////////////////////////////
// impl PartialEq for suitable NodeHandles/SliceHandles //
//////////////////////////////////////////////////////////

impl<Ty, B, I, S, P, const M: usize> PartialEq for NodeHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

#[rustfmt::skip]
impl<Ty, B, I, S, P, const M: usize> Eq for NodeHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{}

impl<Ty, B, I, S, P, const M: usize> PartialEq for SliceHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.idx == other.idx
    }
}

#[rustfmt::skip]
impl<Ty, B, I, S, P, const M: usize> Eq for SliceHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{}

/// Abstraction over known/unknown types of a [`NodeHandle`] or [`SliceHandle`], so that we can
/// provide certain guarantees via the type system
///
/// Everything is `Unknown` by default.
#[rustfmt::skip]
pub(super) mod ty {
    use std::num::NonZeroU8;

    pub struct Unknown;
    pub struct Leaf;
    pub struct Internal;

    /// Trait implemented for all types, giving a specialized height type
    pub trait TypeHint {
        type Height: Copy + Height;
    }

    impl TypeHint for Unknown { type Height = u8; }
    impl TypeHint for Leaf { type Height = ZeroU8; }
    impl TypeHint for Internal { type Height = NonZeroU8; }

    pub trait Height {
        fn as_u8(&self) -> u8;
    }

    /// A `#[repr(u8)]` type that's always zero, so that the layout of a `NodeHandle` doesn't
    /// change when we switch height types
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub struct ZeroU8(u8);

    impl ZeroU8 {
        pub fn new() -> Self {
            ZeroU8(0)
        }

        pub fn get(&self) -> u8 {
            // SAFETY: the value's always zero.
            unsafe { weak_assert!(self.0 == 0) };
            self.0
        }
    }

    impl Height for u8 { fn as_u8(&self) -> u8 { *self } }
    impl Height for ZeroU8 { fn as_u8(&self) -> u8 { self.get() } }
    impl Height for NonZeroU8 { fn as_u8(&self) -> u8 { self.get() } }
}

/// Abstraction over borrows in a `NodeHandle` or `SliceHandle`
pub(super) mod borrow {
    use super::NodePtr;
    use crate::param::{RleTreeConfig, SupportsInsert};
    use std::marker::PhantomData;

    /// An "owned" handle on a node and the subtree rooted at that node.
    ///
    /// Owned handles never have a parent, except during very temporary extractions, like in
    /// [`replace_first_child`]. They also may have shared ownership if COW is enabled;
    /// [`UniqueOwned`] guarantees unique ownership.
    ///
    /// [`replace_first_child`]: super::NodeHandle::replace_first_child
    pub struct Owned;
    /// Like `Owned`, but guarantees that the node does not have any shallow clones pointing to it
    ///
    /// Allows one-way conversion into [`Owned`] through the [`erase_unqiue`] method.
    ///
    /// [`erase_unqiue`]: super::NodeHandle::erase_unique
    pub struct UniqueOwned;
    /// Borrow for use solely by a [`SliceRef`](crate::SliceRef). Essentially like a `'static`
    /// immutable borrow
    pub struct SliceRef;
    /// Immutable borrow
    pub struct Immut<'a>(PhantomData<&'a ()>);
    /// Mutable borrow on a node and all of its ancestors up the tree.
    ///
    /// Can only be produced by a [`UniqueOwned`], but multiple coexisting `Mut` borrows are
    /// explicitly permitted -- even to the same leaf, and it is up to the implementor to ensure
    /// that they don't produce aliasing issues. `miri` (i.e. Stacked Borrows) only accepts it
    /// because the entire thing uses raw pointers.
    pub struct Mut<'a>(PhantomData<&'a mut ()>);
    /// Borrow that's used only for dropping the tree, produced by [`NodeHandle::try_drop`] or
    /// [`NodeHandle::into_drop`]
    ///
    /// [`NodeHandle::try_drop`]: super::NodeHandle::try_drop
    /// [`NodeHandle::into_drop`]: super::NodeHandle::into_drop
    pub struct Dropping;

    /// Marker trait for borrow types that are not [`Owned`], [`UniqueOwned`], or [`Dropping`]
    pub trait NotOwned {}
    impl NotOwned for SliceRef {}
    impl<'a> NotOwned for Immut<'a> {}
    impl<'a> NotOwned for Mut<'a> {}

    /// Marker for borrow types that can be immutably borrowed
    pub trait AsImmut {}
    impl AsImmut for Owned {}
    impl AsImmut for UniqueOwned {}
    impl AsImmut for SliceRef {}
    impl<'a> AsImmut for Mut<'a> {}
    impl<'a> AsImmut for Immut<'a> {}

    /// Marker for borrow types that can be mutably borrowed
    pub trait AsMut {}
    impl AsMut for UniqueOwned {}
    impl AsMut for Dropping {}
    impl<'a> AsMut for Mut<'a> {}

    /// Helper trait that allows us to take actions requiring `SupportsInsert` only for mutable
    /// borrows
    ///
    /// The implementation for `Mut` requires `P: SupportsInsert`, and sets `Param = P`. The
    /// implementation for `Immut` sets an arbitrary `Param` and does nothing.
    ///
    /// The name "cast" for `cast_ref_if_mut` isn't quite accurate; it either returns its input
    /// (with added guarantees about the types), or does nothing.
    pub trait SupportsInsertIfMut<I, S, P, const M: usize>
    where
        P: RleTreeConfig<I, S, M>,
    {
        type Param: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>;

        fn cast_ref_if_mut(
            r: &mut NodePtr<I, S, P, M>,
        ) -> Option<&mut NodePtr<I, S, Self::Param, M>>;
    }
    impl<'a, I, S, P, const M: usize> SupportsInsertIfMut<I, S, P, M> for Immut<'a>
    where
        P: RleTreeConfig<I, S, M>,
    {
        // This may not match `P`, but is still sound because `cast_ref_if_mut` will always return
        // `None`, meaning that it's never used.
        type Param = crate::param::NoFeatures;

        fn cast_ref_if_mut(
            _: &mut NodePtr<I, S, P, M>,
        ) -> Option<&mut NodePtr<I, S, Self::Param, M>> {
            None
        }
    }
    impl<'a, I, S, P, const M: usize> SupportsInsertIfMut<I, S, P, M> for Mut<'a>
    where
        P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
    {
        type Param = P;

        fn cast_ref_if_mut(r: &mut NodePtr<I, S, P, M>) -> Option<&mut NodePtr<I, S, P, M>> {
            Some(r)
        }
    }
}

/// A leaf node -- one without children
///
/// Because [`Internal`] nodes also require all of the same information as leaves, they just embed
/// a `Leaf`.
///
/// Leaf nodes are all at `height = 0`.
///
/// ## Safety
///
/// The safety requirements for individual fields in the `Leaf` are worth discussing -- there are a
/// few things to consider in order to ensure that we abide by the Stacked Borrows model. For ease
/// of use, we assume that any [`NodeHandle`] can construct a reference to the inner `Leaf`, which
/// means that any mutation must be handled with care.
///
/// Furthermore, we require that handles with `borrow::Mut` can *always* assume that there are
/// *currently* no other references to the node. Specifically, methods on an `&mut NodeHandle` with
/// a mutable borrow can always *construct* a `&mut Leaf`, but the reference cannot live beyond the
/// method. Also, only `borrow::Owned` can be mutably borrowed to produce a `borrow::Mut`;
/// `borrow::SliceRef` *does* have a kind of ownership, but it can produce at most a `Dropping`
/// borrow, not `Mut`.
///
/// Both mutable *and* immutable references are constructed temporarily by borrowing the
/// [`NodeHandle`] they originate from in the appropriate manner, and -- depending on the enabled
/// features -- must ensure that access to the tree with the given borrow (i.e., `Immut` or `Mut`)
/// is sound. These guarantees tend to be trivially true for trees with `NoFeatures`, must be
/// guaranteed by tre--wide borrows for [`AllowSliceRefs`], and must be guaranteed by unique,
/// cloned access to a particular node for [`AllowCow`] trees.
///
/// It's assumed that `borrow::Mut` guarantees exclusive mutable access to the node and its
/// parent(s), but not necessarily to its children (which may have multiple shared references with
/// COW enabled).
///
/// ## Safety continued: upholding Stacked Borrows
///
/// The rules above basically ensure that we uphold the Stacked Borrows model, because of how raw
/// pointers work: access from any raw pointers may be interleaved, essentially meaning that we can
/// interleave usages of `borrow::Mut` and `borrow::Immut` handles, so long as the usage of
/// *references constructed from them* isn't interleaved. This is partially why we need to be
/// careful about producing mutable references -- they *cannot* mutate out from underneath another
/// *reference* (`NodeHandle`s are ok).
///
/// ## Safety continued: individual fields
///
/// There's a number of fields here with individually-marked safety considerations. When those are
/// absent, you may assume that they are otherwise safe to read or modify through references with
/// the appropriate access.
///
/// [`AllowSliceRefs`]: crate::param::AllowSliceRefs
/// [`AllowCow`]: crate::param::AllowCow
pub(super) struct Leaf<I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Parent pointer, always present so that insertion and other algorithms can be done in a
    /// loop, rather than recursively or with an explicit stack
    ///
    /// When clone-on-write is enabled, the parent pointer *may* be invalid. As such, all routines
    /// that modify the tree can only access parent pointers when they're sure that they have
    /// unique access to the child (e.g., by `shallow_clone`-ing the child and setting the parent
    /// pointer). This is what [`into_child`] does to handle this problem.
    ///
    /// ## Safety
    ///
    /// For trees with [`AllowCow`], the parent is not guaranteed to point to a valid allocation
    /// unless through a [`NodeHandle`] with `borrow::Mut`. Otherwise, you may *assume* that the
    /// inner value, if present, points to an internal node at height one greater than this one.
    ///
    /// Before writing to this field, you must *first* set `idx_in_parent`.
    ///
    /// [`into_child`]: NodeHandle::into_child
    /// [`AllowCow`]: crate::param::AllowCow
    parent: Option<NodePtr<I, S, P, M>>,
    /// The index of this child in its parent, only initialized if `parent` is `Some(_)`
    ///
    /// While we *could* group this value into the `Option` for additional type safety, that would
    /// unfortunately cause the size of `parent` to double. Leaving `idx_in_parent` out in the main
    /// body of the struct means that we can pad the extra room with `len` and `holes` to fill up
    /// another 4 bytes.
    ///
    /// ## Safety
    ///
    /// This value is considered initialized exactly when `parent` is not `None`. It must be
    /// written to before `parent`.
    idx_in_parent: MaybeUninit<u8>,

    /// Sometimes, we need to temporarily extract values out of `self.vals` -- this field stores up
    /// to two indexes in `vals` that are temporarily uninitialized.
    ///
    /// The indexes are stored as their value plus one, in `Option<NonZeroU8>`s to save space. This
    /// is safe because `key_index < len` and `len: u8` means `len <= u8::MAX`, so
    /// `key_index + 1 <= u8::MAX`.
    ///
    /// ## Safety
    ///
    /// Operations on the `Leaf` that read or modify `vals` *must* check `holes` before the
    /// operation, and it *cannot* be assumed that `holes` is empty unless previously guaranteed.
    /// For example, it is both possible and safe for a user to panic during a call to
    /// `Slice::try_join` (which uses `holes`) and later access the `RleTree` in a messy state.
    ///
    /// The only valid configurations of `holes` are exhaustively enumerated as:
    ///
    /// * `[None, None]`
    /// * `[None, Some(x)]` where `x.get() - 1 < len`
    /// * `[Some(x), Some(y)]` where `x < y` and `y.get() - 1 < len`
    ///
    /// This means that it is always true that either `holes[0] < holes[1]`, or both are equal to
    /// `None`.
    ///
    /// In general, writes to this field should prefer writing to the entire field, instead of
    /// individual elements in the array.
    holes: [Option<NonZeroU8>; 2],

    /// The number of keys in this node
    ///
    /// Look, if you're using more than 255 values in each node, let's talk. There's probably no
    /// valid reason to keep a tree *that* shallow, and so big that each node (most likely)
    /// stretches across multiple memory pages.
    ///
    /// In other B-trees this *might* make sense, but it doesn't here -- specifically because of
    /// the added relative indexing information.
    ///
    /// ## Safety
    ///
    /// When increasing `len`, initialized values *must* be added before increasing `len`.
    /// Decreasing `len` requires the opposite order: changing `len` before removing the values.
    len: u8,

    /// `AtomicUsize` counting references to this node, only present if clone-on-write is enabled
    strong_count: resolve![P::StrongCount],

    /// The full size of a subtree rooted at this node. Storing this additional value ensures that
    /// we're always able to calculate the size of the last key from its position and (if present)
    /// the child following it. Other keys' sizes can already be calculated from the size of the
    /// child & position of the next key.
    ///
    /// ## Safety
    ///
    /// No assumptions can be made about this field in `unsafe` code; implementations of [`Index`]
    /// for `I` may be malicious.
    total_size: I,

    /// The offset of each slice from the start of the node
    ///
    /// This field, `vals`, and `refs` form the pieces of information paired with every *logical*
    /// "key", but they're extracted into separate arrays to improve locality of reference while
    /// we're searching in a particular node.
    ///
    /// ## Safety
    ///
    /// The values up to `len` are initialized. Because `I: Index` implies `I: Copy`, and the only
    /// way to build a non-empty tree is through methods that require `I: Index`, keys do not need
    /// to be dropped.
    ///
    /// No assumptions can be made about the *values* of the keys in `unsafe` code; implementations
    /// of [`Index`] for `I` may be malicious.
    keys: KeyArray<MaybeUninit<I>, M>,

    /// The slices in this node
    ///
    /// ## Safety
    ///
    /// The values up to `len` are initialized, except for indexes referenced in `holes`. Both
    /// `len` and `holes` *must* be checked before accessing or modifying any elements.
    vals: KeyArray<MaybeUninit<S>, M>,

    /// The [`RefId`] for each slice, if it has one
    ///
    /// ## Safety
    ///
    /// The values up to `len` are initialized.
    ///
    /// Accessing or modifying any elements of the array can only be done through methods that are
    /// live *only* inside private `NodeHandle`/`SliceHandle` methods. Inside these methods, the
    /// reference cannot be held across any operation that runs user-defined code, like dropping a
    /// user-supplied value or calling any [`Slice`] method. Any user-defined code can drop a
    /// [`SliceRef`], which may then remove its own value in `refs`, which would write to the
    /// value.
    ///
    /// The [`RefId`]s don't need to be dropped when the node is. We always guarantee that the
    /// [`SliceRefStore`] is dropped before the tree itself, so the [`RefId`]s have already become
    /// meaningless by the time we *would* drop them.
    ///
    /// [`Slice`]: crate::public_traits::Slice
    /// [`SliceRef`]: super::SliceRef
    /// [`RefId`]: super::slice_ref::RefId
    /// [`SliceRefStore`]: super::slice_ref::SliceRefStore
    refs: KeyArray<MaybeUninit<UnsafeCell<resolve![P::SliceRefStore::OptionRefId]>>, M>,
}

/// Internal node -- all the same contents as a [`Leaf`] node, but with children
///
/// This type is `#[repr(C)]` so that a pointer to `Internal` can be soundly reinterpreted as a
/// pointer to a `Leaf`.
#[repr(C)]
pub(super) struct Internal<I, S, P, const M: usize>
where
    P: RleTreeConfig<I, S, M>,
{
    /// The data from the leaf
    leaf: Leaf<I, S, P, M>,
    /// Pointers to each child.
    ///
    /// ## Safety
    ///
    /// The elements up to `leaf.len + 1` are initialized, and nothing else.
    child_ptrs: ChildArray<MaybeUninit<NodePtr<I, S, P, M>>, M>,
}

/// A child or key index
#[derive(Debug, Copy, Clone)]
pub(super) enum ChildOrKey<C, K> {
    Child(C),
    Key(K),
}

////////////////////////
// HELPER DEBUG IMPLS //
////////////////////////

#[cfg(test)]
impl<I, S, P: RleTreeConfig<I, S, M>, const M: usize> Leaf<I, S, P, M> {
    fn append_fields(&self, s: &mut fmt::DebugStruct) {
        let parent = self.parent();

        struct Ks<'a, I, S, P: RleTreeConfig<I, S, M>, const M: usize>(&'a Leaf<I, S, P, M>);

        impl<'a, I, S, P: RleTreeConfig<I, S, M>, const M: usize> Debug for Ks<'a, I, S, P, M> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                let mut ls = f.debug_list();
                let poss = self.0.keys_pos_slice();
                for i in 0..self.0.len {
                    ls.entry(&K {
                        pos: &poss[i as usize],
                        // SAFETY: `is_hole` requires `i < u8::MAX`, which is guaranteed because
                        // `i < self.0.len`.
                        slice: match unsafe { self.0.is_hole(i) } {
                            true => None,
                            // SAFETY: the key at `i` isn't a hole and it's within the initialized
                            // length, so we're good to read it.
                            false => unsafe {
                                Some(self.0.vals.get_unchecked(i as usize).assume_init_ref())
                            },
                        },
                        // SAFETY: we're constructing a temporary reference to an initialized
                        // `Option<RefId>`
                        ref_id: unsafe {
                            &*self.0.refs.get_unchecked(i as usize).assume_init_ref().get()
                        },
                    });
                }
                ls.finish()
            }
        }

        struct K<'a, I, S, R> {
            pos: &'a I,
            slice: Option<&'a S>,
            ref_id: &'a R,
        }

        impl<'a, I, S, R> Debug for K<'a, I, S, R> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                struct IsHole;

                impl Debug for IsHole {
                    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                        f.write_str("<Hole>")
                    }
                }

                let mut s = f.debug_struct("Key");
                s.field("pos", self.pos.fallible_debug())
                    .field("slice", self.slice.map(|s| s.fallible_debug()).unwrap_or(&IsHole));
                if mem::size_of::<R>() != 0 {
                    s.field("ref_id", self.ref_id.fallible_debug());
                }
                s.finish()
            }
        }

        if P::COW {
            s.field("strong_count", self.strong_count.fallible_debug());
        }

        s.field("parent", &parent.as_ref().map(|p| p.ptr))
            .field("idx_in_parent", &parent.as_ref().map(|p| p.idx_in_parent))
            .field("holes", &self.holes)
            .field("len", &self.len)
            .field("keys", &Ks(&self))
            .field("total_size", self.total_size.fallible_debug());
    }
}

#[cfg(test)]
impl<I, S, P: RleTreeConfig<I, S, M>, const M: usize> Internal<I, S, P, M> {
    fn append_fields(&self, s: &mut fmt::DebugStruct) {
        self.leaf.append_fields(s);
        s.field("child_ptrs", &self.child_ptrs());
    }
}

//////////////////////////////
// CRAZY ARRAY LENGTH HACKS //
//////////////////////////////

/// Wild hack to represent `[T; 2 * M]` with a const parameter `M`
///
/// ... because types cannot be parameterized by const expressions that non-trivially depend on
/// input generics (at time of writing, Rust 1.62)
///
/// For more, see [`const_math_hack`](crate::const_math_hack)
type KeyArray<T, const M: usize> = hack::Mul<[T; 2], M>; // 2*M

/// Wild hack to represent `[T; 2 * M + 1]` with a const parameter `M`
///
/// ... because types cannot be parameterized by const expressions that non-trivially depend on
/// input generics (at time of writing, Rust 1.62)
///
/// For more, see [`const_math_hack`](crate::const_math_hack)
type ChildArray<T, const M: usize> = hack::Add<hack::Mul<[T; 2], M>, [T; 1]>; // 2*M + 1

// A handful of tests just to make sure that our evaluation works properly. Because this is here,
// and guaranteed to fail if the definitions of `KeyArray` or `ChildArray` change, here's the
// things that you need to remember to update if you change the parameters:
//
//   * RleTree docs
//   * fn assert_reasonable_m (in 'src/tree/mod.rs')
//   * fn RleTree::do_insert_full_edge_no_join{,split} (in 'src/tree/insert.rs')
//   * fn BubbledInsertState::do_upward_step (in 'src/tree/insert.rs')
#[cfg(test)]
const _: () = {
    assert!(KeyArray::<u8, 4>::LEN == 4 * 2);
    assert!(KeyArray::<u8, 7>::LEN == 7 * 2);
    assert!(KeyArray::<u8, 10>::LEN == 10 * 2);

    assert!(ChildArray::<u8, 4>::LEN == 4 * 2 + 1);
    assert!(ChildArray::<u8, 7>::LEN == 7 * 2 + 1);
    assert!(ChildArray::<u8, 10>::LEN == 10 * 2 + 1);
};

impl<T, const M: usize> KeyArray<MaybeUninit<T>, M> {
    const fn new() -> Self {
        // We have to initialize like this because nested array initializations fail if T: !Copy
        //
        // SAFETY: The inner contents are uninitialized anyways, so we're never *actually* reading
        // uninitialized memory.
        unsafe { MaybeUninit::uninit().assume_init() }
    }
}

impl<T, const M: usize> ChildArray<MaybeUninit<T>, M> {
    const fn new() -> Self {
        // We have to initialize like this because nested array initializations fail if T: !Copy
        //
        // SAFETY: The inner contents are uninitialized anyways, so we're never *actually* reading
        // uninitialized memory.
        unsafe { MaybeUninit::uninit().assume_init() }
    }
}

/////////////////////////////////////
// START OF ACTUAL IMPLEMENTATIONS //
/////////////////////////////////////

/// Panics with the message "tree is in an invalid state"
#[cold]
#[inline(never)]
fn panic_invalid_state() -> ! {
    panic!("tree is in an invalid state");
}

#[derive(Copy, Clone)]
pub enum Type<L, I> {
    Leaf(L),
    Internal(I),
}

/// Helper trait for abbreviating type signatures
pub trait Typed {
    type Leaf;
    type Internal;
    type Unknown;
}

impl<Ty, B, I, S, P, const M: usize> Typed for NodeHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    type Leaf = NodeHandle<ty::Leaf, B, I, S, P, M>;
    type Internal = NodeHandle<ty::Internal, B, I, S, P, M>;
    type Unknown = NodeHandle<ty::Unknown, B, I, S, P, M>;
}

impl<Ty, B, I, S, P, const M: usize> Typed for SliceHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    type Leaf = SliceHandle<ty::Leaf, B, I, S, P, M>;
    type Internal = SliceHandle<ty::Internal, B, I, S, P, M>;
    type Unknown = SliceHandle<ty::Unknown, B, I, S, P, M>;
}

// Any params
//  * ptr
//  * height
//  * leaf
//  * borrow (where B: borrow::AsImmut)
//  * as_immut (where B: borrow::AsImmut)
//  * borrow_mut (where B: borrow::AsMut)
//  * as_mut (where B: borrow::AsMut)
//  * into_typed
//  * typed_ref
//  * typed_mut
//  * erase_type
//  * into_parent
//  * into_slice_handle (where B: borrow::NotOwned)
//  * try_child_size (where I: Copy)
//  * shallow_clone (where I: Index, P: SupportsInsert)
//  * increase_strong_count_and_clone
impl<Ty, B, I, S, P, const M: usize> NodeHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    /// Returns the inner pointer this `NodeHandle` uses
    ///
    /// This method exists (and should be used) only for `NodeHandle` comparison.
    pub fn ptr(&self) -> NonNull<AbstractNode<I, S, P, M>> {
        self.ptr
    }

    /// Returns the height of the node
    pub fn height(&self) -> u8 {
        self.height.as_u8()
    }

    /// Returns a reference to the inner `Leaf`
    pub(super) fn leaf(&self) -> &Leaf<I, S, P, M> {
        // SAFETY: references to `Internal` nodes can be cast to `Leaf`s, as described in the
        // "safety" comment for `Internal` -- so regardless of the type of the node, we can read it
        // as a `Leaf`.
        unsafe { self.ptr.cast().as_ref() }
    }

    /// Produces an immutable handle for the node, borrowing the handle for its duration
    pub fn borrow<'h>(&'h self) -> NodeHandle<Ty, borrow::Immut<'h>, I, S, P, M>
    where
        B: borrow::AsImmut,
    {
        NodeHandle {
            ptr: self.ptr,
            height: self.height,
            borrow: PhantomData,
        }
    }

    /// Casts a reference to the `NodeHandle` into one with an immutable borrow
    pub fn as_immut<'h>(&'h self) -> &'h NodeHandle<Ty, borrow::Immut<'h>, I, S, P, M>
    where
        B: borrow::AsImmut,
    {
        // SAFETY: the layout of `NodeHandle` doesn't actually depend on its type parameters, so
        // casting the reference is sound. The lifetime 'h guarantees that the reference to the
        // handle itself won't be aliased.
        unsafe { &*(self as *const _ as *const _) }
    }

    /// Produces a mutable handle for the node, borrowing the handle for its duration
    pub fn borrow_mut<'h>(&'h mut self) -> NodeHandle<Ty, borrow::Mut<'h>, I, S, P, M>
    where
        B: borrow::AsMut,
    {
        NodeHandle {
            ptr: self.ptr,
            height: self.height,
            borrow: PhantomData,
        }
    }

    /// Casts a mutable reference to the `NodeHandle` into one with a mutable borrow
    pub fn as_mut<'h>(&'h mut self) -> &'h mut NodeHandle<Ty, borrow::Mut<'h>, I, S, P, M>
    where
        B: borrow::AsMut,
    {
        // SAFETY: the layout of `NodeHandle` doesn't actually depend on its type parameters, so
        // casting the reference is sound. The lifetime 'h guarantees that the reference to the
        // handle itself won't be aliased.
        unsafe { &mut *(self as *mut _ as *mut _) }
    }

    /// Converts the `NodeHandle` into one where the type has been resolved
    pub fn into_typed(self) -> Type<<Self as Typed>::Leaf, <Self as Typed>::Internal> {
        match NonZeroU8::new(self.height.as_u8()) {
            None => Type::Leaf(NodeHandle {
                ptr: self.ptr,
                height: ty::ZeroU8::new(),
                borrow: PhantomData,
            }),
            Some(h) => Type::Internal(NodeHandle { ptr: self.ptr, height: h, borrow: PhantomData }),
        }
    }

    /// Converts a reference to the `NodeHandle` into one where the type has been resolved
    pub fn typed_ref(&self) -> Type<&<Self as Typed>::Leaf, &<Self as Typed>::Internal> {
        // SAFETY: The basic idea here is that we're reinterpreting the reference to change the
        // parameters. This happens to be sound because `NodeHandle`s are #[repr(C)], meaning that
        // because the individual `height`s are all `u8`s, we can "just" cast the reference.
        // This is made explicit in the "safety" note for `NodeHandle`.
        unsafe {
            match self.height.as_u8() {
                0 => Type::Leaf(&*(self as *const _ as *const _)),
                _ => Type::Internal(&*(self as *const _ as *const _)),
            }
        }
    }

    /// Converts a reference to the `NodeHandle` into one where the type is now unknown
    ///
    /// This method is essentially the opposite of [`typed_ref`](Self::typed_ref).
    pub fn untyped_ref(&self) -> &<Self as Typed>::Unknown {
        // SAFETY: `TypeHint::Height` always has a size of 1 (so it's valid to reinterpret as u8),
        // and we ensure elsewhere that the bit value of `self.height` is always equal to
        // `self.height.as_u8()`.
        unsafe { weak_assert!(self.height.as_u8() == *(&self.height as *const _ as *const u8)) };

        // SAFETY: This is *mostly* the same as in `typed_ref`. The extra piece here is that we're
        // only guaranteed that the height stays valid because of conditions guaranteeing the
        // assertion above.
        unsafe { &*(self as *const _ as *const _) }
    }

    /// Converts a mutable reference to the `NodeHandle` into one where the type has been resolved
    pub fn typed_mut(
        &mut self,
    ) -> Type<&mut <Self as Typed>::Leaf, &mut <Self as Typed>::Internal> {
        // SAFETY: This is the same as in `typed_ref`; refer there for more info.
        unsafe {
            match self.height.as_u8() {
                0 => Type::Leaf(&mut *(self as *mut _ as *mut _)),
                _ => Type::Internal(&mut *(self as *mut _ as *mut _)),
            }
        }
    }

    /// Converts a mutable reference to the `NodeHandle` into one where the type is now unknown
    ///
    /// This method is essentially the opposite of [`typed_mut`](Self::typed_mut).
    pub fn untyped_mut(&mut self) -> &mut <Self as Typed>::Unknown {
        // SAFETY: See `untyped_ref`.
        unsafe { weak_assert!(self.height.as_u8() == *(&self.height as *const _ as *const u8)) };

        // SAFETY: This is the same as in `untyped_ref`; refer there for more info.
        unsafe { &mut *(self as *mut _ as *mut _) }
    }

    /// Converts this `NodeHandle` into one with `ty::Unknown` instead of the current type tag
    pub fn erase_type(self) -> <Self as Typed>::Unknown {
        NodeHandle {
            ptr: self.ptr,
            height: self.height.as_u8(),
            borrow: self.borrow,
        }
    }

    /// Converts this node into a handle on the given slice
    ///
    /// ## Safety
    ///
    /// The provided `key_idx` *must* be less than [`self.leaf().len()`](Leaf::len). If not, the
    /// immediate result will be UB.
    pub unsafe fn into_slice_handle(self, key_idx: u8) -> SliceHandle<Ty, B, I, S, P, M>
    where
        B: borrow::NotOwned,
    {
        // SAFETY: Guaranteed by the safety requirements for this method
        unsafe { weak_assert!(key_idx < self.leaf().len) };
        SliceHandle { node: self, idx: key_idx }
    }

    /// Returns this node's parent, alongside the child index that `self` was. If this node has no
    /// parent, this method returns `Err(self)`.
    pub fn into_parent(self) -> Result<(NodeHandle<ty::Internal, B, I, S, P, M>, u8), Self>
    where
        B: borrow::NotOwned,
    {
        let Parent { ptr, idx_in_parent } = match self.leaf().parent() {
            Some(p) => p,
            None => return Err(self),
        };
        // SAFETY: This node has a parent, so h + 1 is valid (i.e. h + 1 <= u8::MAX). It's then
        // trivially true that `h + 1 != 0` (otherwise we'd worry about wrapping overflow).
        let height = unsafe { NonZeroU8::new_unchecked(self.height.as_u8() + 1) };

        let parent_handle = NodeHandle { ptr, height, borrow: PhantomData };
        Ok((parent_handle, idx_in_parent))
    }

    /// Returns the size of the child at `child_idx`, if this node has children. Otherwise returns
    /// `None`.
    ///
    /// ## Safety
    ///
    /// Whether or not this node has children, `child_idx` must be less than or equal to
    /// `self.leaf().len()`. Failing this condition results in immediate UB.
    pub unsafe fn try_child_size(&self, child_idx: u8) -> Option<I>
    where
        I: Copy,
    {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(child_idx <= self.leaf().len) };

        match self.typed_ref() {
            Type::Leaf(_) => None,
            // SAFETY: guaranteed by above
            Type::Internal(handle) => Some(unsafe { handle.child_size(child_idx) }),
        }
    }

    /// Creates a shallow clone of the node, cloning only the keys inside it and incrementing the
    /// strong counts of all child nodes
    ///
    /// ## Safety
    ///
    /// This function requires that `P = AllowCow`, and *will* trigger UB if that is not true.
    pub unsafe fn shallow_clone(&self) -> NodeHandle<Ty, borrow::Owned, I, S, P, M>
    where
        B: borrow::AsImmut,
        I: Index,
        P: SupportsInsert<I, S, M>,
    {
        // SAFETY: Guaranteed by caller
        unsafe { weak_assert!(P::COW) };

        let this_leaf = self.leaf();

        if this_leaf.has_holes() {
            panic_invalid_state();
        }

        let (parent, idx_in_parent) = match this_leaf.parent() {
            None => (None, MaybeUninit::uninit()),
            Some(p) => (Some(p.ptr), MaybeUninit::new(p.idx_in_parent)),
        };

        // Copy all of the slices
        let mut vals: GradualInitArray<KeyArray<S, M>> = GradualInitArray::new();
        for i in 0..this_leaf.len() as usize {
            // SAFETY: `i < this_leaf.len` guarantees it's within bounds and initialized. Within
            // bounds means we won't have written off the end of `vals` as well. The call to
            // `P::clone_slice` requires that `P = AllowCow`, which is guaranteed by the caller
            unsafe {
                let this_val = this_leaf.vals.get_unchecked(i).assume_init_ref();
                let new_val = P::clone_slice(this_val);
                vals.push_unchecked(new_val);
            }
        }

        let new_leaf: Leaf<I, S, P, M> = Leaf {
            parent,
            idx_in_parent,
            holes: [None; 2],
            len: this_leaf.len,
            strong_count: P::StrongCount::one(),
            keys: this_leaf.keys,
            vals: vals.into_uninit(),
            // Note: `P::SliceRefStore::OptionRefId` is zero-sized for `param::AllowCow`, so we can
            // leave the entire array uninitialized.
            refs: KeyArray::new(),
            total_size: this_leaf.subtree_size(),
        };

        match self.typed_ref() {
            Type::Leaf(_) => NodeHandle {
                ptr: alloc_aligned(new_leaf).cast(),
                height: self.height,
                borrow: PhantomData,
            },
            Type::Internal(this) => {
                let b = this.borrow();

                // Increment all the strong counts of child pointers
                for ci in 0..=this_leaf.len {
                    // SAFETY: `into_child` for immutable borrows only requires that `ci <= len`,
                    // which is guaranteed by the range above
                    unsafe { b.into_child(ci) }.leaf().strong_count.increment();
                }

                let new_internal = Internal {
                    leaf: new_leaf,
                    child_ptrs: this.internal().child_ptrs,
                };

                NodeHandle {
                    ptr: alloc_aligned(new_internal).cast(),
                    height: self.height,
                    borrow: PhantomData,
                }
            }
        }
    }

    /// Increases the strong count of this node and produces a copy
    ///
    /// ## Safety
    ///
    /// This function requires that `P = AllowCow` and *will* trigger UB if that is not true.
    pub unsafe fn increase_strong_count_and_clone(
        &self,
    ) -> NodeHandle<Ty, borrow::Owned, I, S, P, M> {
        // SAFETY: Guaranteed by caller
        unsafe { weak_assert!(P::COW) };

        self.leaf().strong_count.increment();
        NodeHandle {
            ptr: self.ptr,
            height: self.height,
            borrow: PhantomData,
        }
    }
}

// any type, borrow::Immut, param::NoFeatures
//  * deep_clone (where I: Copy, S: Clone)
impl<'t, Ty, I, S, const M: usize> NodeHandle<Ty, borrow::Immut<'t>, I, S, param::NoFeatures, M>
where
    Ty: TypeHint,
{
    /// Creates a deep clone of the node, cloning it and performing a deep clone of all of its
    /// children
    pub fn deep_clone(&self) -> NodeHandle<Ty, borrow::UniqueOwned, I, S, param::NoFeatures, M>
    where
        I: Copy,
        S: Clone,
    {
        let this_leaf = self.leaf();

        if this_leaf.has_holes() {
            panic_invalid_state();
        }

        // Copy all of the slices
        let mut vals: GradualInitArray<KeyArray<S, M>> = GradualInitArray::new();
        for i in 0..this_leaf.len() as usize {
            // SAFETY: `i < this_leaf.len` guarantees it's within bounds and initialized. Within
            // bounds means we won't have written off the end of `vals` as well.
            unsafe {
                let val = this_leaf.vals.get_unchecked(i).assume_init_ref().clone();
                vals.push_unchecked(val);
            }
        }

        let new_leaf: Leaf<I, S, param::NoFeatures, M> = Leaf {
            parent: None,
            idx_in_parent: MaybeUninit::uninit(),
            holes: [None; 2],
            len: this_leaf.len,
            strong_count: (),
            keys: this_leaf.keys,
            vals: vals.into_uninit(),
            // Note: We can leave `refs` uninitialized because `param::NoFeatures` has
            // `OptionRefId = ()`, so the total size of `refs` is zero.
            refs: KeyArray::new(),
            total_size: this_leaf.subtree_size(),
        };

        match self.typed_ref() {
            Type::Leaf(_) => NodeHandle {
                ptr: alloc_aligned(new_leaf).cast::<AbstractNode<I, S, param::NoFeatures, M>>(),
                height: self.height,
                borrow: PhantomData,
            },
            Type::Internal(this) => {
                let mut child_ptrs: GradualInitArray<
                    ChildArray<NodeBox<ty::Unknown, I, S, param::NoFeatures, M>, M>,
                > = GradualInitArray::new();

                for i in 0..=this_leaf.len {
                    // SAFETY: `i <= this_leaf.len` guarantees it's within bounds and initiaized.
                    // Within bounds means we won't write off the end of `child_ptrs` as well.
                    unsafe {
                        let mut child = NodeBox::new(this.child(i).deep_clone());
                        child.as_mut().with_mut(|leaf| {
                            let _ = leaf.idx_in_parent.write(i);
                        });
                        child_ptrs.push_unchecked(child);
                    }
                }

                let new_internal = Internal {
                    leaf: new_leaf,
                    child_ptrs: child_ptrs.into_uninit().map_array_with_index(|i, p| {
                        if i as u8 <= this_leaf.len {
                            // SAFETY: the bound `i <= this_leaf.len` means `p` is initialized
                            unsafe { MaybeUninit::new(p.assume_init().take().ptr) }
                        } else {
                            MaybeUninit::uninit()
                        }
                    }),
                };

                // Before we return, we have to overwrite the parent pointers in all of this node's
                // children.
                let mut handle: NodeHandle<ty::Internal, borrow::UniqueOwned, _, _, _, M> =
                    NodeHandle {
                        ptr: alloc_aligned(new_internal)
                            .cast::<AbstractNode<I, S, param::NoFeatures, M>>(),
                        // SAFETY: `self.height` is known to be non-zero because we already found
                        // that this is an internal node
                        height: unsafe { NonZeroU8::new_unchecked(self.height.as_u8()) },
                        borrow: PhantomData,
                    };

                let new_ptr = handle.ptr;

                unsafe {
                    handle.as_mut().with_internal(|internal| {
                        for i in 0..=this_leaf.len {
                            let p =
                                internal.child_ptrs.get_mut_unchecked(i as usize).assume_init_mut();
                            let leaf_mut = p.cast::<Leaf<I, S, param::NoFeatures, M>>().as_mut();
                            leaf_mut.parent = Some(new_ptr);
                        }
                    });
                }

                NodeHandle {
                    ptr: handle.ptr,
                    height: self.height,
                    borrow: PhantomData,
                }
            }
        }
    }
}

/// An owned, extracted key-value pair from a node
pub struct Key<I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    pub pos: I,
    pub slice: S,
    pub ref_id: resolve![P::SliceRefStore::OptionRefId],
}

#[cfg(test)]
impl<I, S, P: RleTreeConfig<I, S, M>, const M: usize> Debug for Key<I, S, P, M> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut s = f.debug_struct("Key");
        s.field("pos", self.pos.fallible_debug())
            .field("slice", self.slice.fallible_debug());
        if P::SLICE_REFS {
            s.field("ref_id", self.ref_id.fallible_debug());
        }
        s.finish()
    }
}

// any type, borrow::Mut
//  * with_mut
//  * remove_parent
//  * set_subtree_size (where I: Copy)
//  * set_key_poss_with (where I: Copy)
//  * split (where I: Copy)
impl<'t, Ty, I, S, P, const M: usize> NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    /// Calls the given function with a mutable reference to the leaf
    ///
    /// ## Safety
    ///
    /// The inner function *must not* under any circumstances produce other references to the
    /// `Leaf`. In particular, this means it *cannot* call any user-defined code (which, for
    /// example, can drop a [`RefId`] - an operation that must always succeed *and* constructs a
    /// `&Leaf`).
    ///
    /// It must, of course, also be safe to produce the `&mut Leaf`, although this safety is
    /// already guaranteed by the methods on handles with `borrow::Mut`.
    ///
    /// It is explicitly safe for `func` to panic.
    ///
    /// [`RefId`]: super::slice_ref::RefId
    pub(super) unsafe fn with_mut<R>(
        &mut self,
        func: impl FnOnce(&mut Leaf<I, S, P, M>) -> R,
    ) -> R {
        // SAFETY: The existence of a `borrow::Mut` means that we can construct a `&mut Leaf`. The
        // caller guarantees it won't be aliased so calling `func` with it is ok.
        unsafe { func(self.ptr.cast().as_mut()) }
    }

    /// Orphans the node, without touching the parent
    pub fn remove_parent(&mut self) {
        // SAFETY: `with_mut` requires we don't call user-defined code. We're not.
        unsafe { self.with_mut(|leaf| leaf.parent = None) }
    }

    /// Sets the size of the subtree rooted at the node
    pub fn set_subtree_size(&mut self, size: I) {
        // SAFETY: `with_mut` requires that we don't call any user-defined code. Because `I`
        // doesn't implement `Drop` (it conflicts with `Copy`), dropping `leaf.total_size` won't do
        // anything.
        unsafe { self.with_mut(|leaf| leaf.total_size = size) };
    }

    /// Sets the position of a single key in the node
    ///
    /// Note: If you're calling this multiple times, you probably want [`set_key_poss_with`]. This
    /// method is only provided so that we can be efficient if a change doesn't *exactly* fit
    /// [`set_key_poss_with`].
    ///
    /// ## Safety
    ///
    /// If `k_idx` is not less than `self.leaf().len()`, this method will invoke immediate UB.
    ///
    /// [`set_key_poss_with`]: Self::set_key_poss_with
    pub unsafe fn set_single_key_pos(&mut self, k_idx: u8, pos: I) {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(k_idx < self.leaf().len()) };

        // SAFETY: `get_mut_unchecked` is guaranteed by caller; `with_mut` requires that we don't
        // call user-defined code, which is clearly the case here.
        unsafe {
            self.with_mut(|leaf| {
                let _ = leaf.keys.get_mut_unchecked(k_idx as usize).write(pos);
            })
        };
    }

    /// Sets the positions of all of the keys, plus the total subtree size of the node, using the
    /// provided function
    ///
    /// Only the keys starting from `range.start` are set.
    ///
    /// ## Safety
    ///
    /// If `range.start` is *greater* than `self.leaf().len()`, this method will invoke immediate
    /// UB.
    pub unsafe fn set_key_poss_with(&mut self, f: impl Fn(I) -> I, range: RangeFrom<u8>)
    where
        I: Copy,
    {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(range.start <= self.leaf().len) };

        // We can't construct a `&mut Leaf` here because the implementation of dirctional
        // arithmetic *could* possibly drop a `SliceRef`, which would mutate out from underneath
        // the mutable reference (which, even though it's through an `UnsafeCell`, would still be
        // unsound because the `&mut Leaf` cannot be aliased *at all*).
        for idx in (range.start as usize)..(self.leaf().len as usize) {
            // SAFETY: There's basically two safety concerns here: (1) that `keys[idx]` is valid
            // and initialized, and (2) that we won't alias the `&mut Leaf` in the call to
            // `with_mut`. For (2), we take care of this by moving the call to `f` outside the call
            // to `with_mut`, because it's possible it could drop a `SliceRef`. We're guaranteed
            // that (1) holds because `idx < self.leaf().len`
            unsafe {
                let old_i = self.leaf().keys.get_unchecked(idx).assume_init();
                let new_i = f(old_i);
                // `drop(...)` so we don't return the `&mut I`, which `with_mut` won't allow.
                self.with_mut(|leaf| drop(leaf.keys.get_mut_unchecked(idx).write(new_i)));
            }
        }

        let new_end = f(self.leaf().subtree_size());
        self.set_subtree_size(new_end);
    }

    /// Splits the node at the given key index, returning the key
    ///
    /// The `SliceRefStore` must be provided so that (a) the values can all be updated to the new
    /// node, and (b) the returned [`Key`] can be [suspended], with its location to be updated once
    /// it has been placed in this node's parent.
    ///
    /// The returned `Key`'s position remains relative to the position of `self`. **The positions
    /// of keys in the new node and the subtree size of `self` are not updated, and must be done by
    /// the caller.**
    ///
    /// The new value of `self.leaf().len()` will be equal to `midpoint_idx`.
    ///
    /// ## Safety
    ///
    /// The key index `midpoint_idx` must be within the bounds of the node (i.e.
    /// `midpoint_idx < self.leaf().len()`).
    ///
    /// [suspended]: SliceRefStore::suspend
    pub unsafe fn split(
        &mut self,
        midpoint_idx: u8,
        store: &mut resolve![P::SliceRefStore],
    ) -> (Key<I, S, P, M>, NodeHandle<Ty, borrow::UniqueOwned, I, S, P, M>)
    where
        I: Copy,
    {
        let old_len = self.leaf().len();
        let src_start = (midpoint_idx + 1) as usize;
        let copy_len = old_len as usize - src_start;

        if self.leaf().has_holes() {
            panic_invalid_state();
        }

        let new_leaf: Leaf<I, S, P, M> = Leaf {
            parent: None,
            idx_in_parent: MaybeUninit::uninit(),
            holes: [None; 2],
            len: 0,
            strong_count: P::StrongCount::one(),
            keys: KeyArray::new(),
            vals: KeyArray::new(),
            refs: KeyArray::new(),
            total_size: self.leaf().subtree_size(),
        };

        let height = self.height.as_u8();

        // Note: this function expects `dst_ptr` to be a heap-allocated `Leaf` or `Internal` node
        // where the `Leaf` part has already been initialized to `new_leaf`.
        #[rustfmt::skip]
        let mut move_leaf_parts = |src: &mut Leaf<I, S, P, M>, dst_ptr: NonNull<Leaf<I, S, P, M>>| {
            src.len = midpoint_idx;

            // SAFETY: `move_leaf_parts` is only called with `dst_ptr` containing `new_leaf`.
            let dst = unsafe { &mut *dst_ptr.as_ptr() };

            // copy `keys`, `vals`, `refs` into the new leaf.
            //
            // SAFETY: `copy_nonoverlapping` requires that the source be valid for reads of
            // `copy_len` copies of the type, and that the destination be valid for `copy_len`
            // writes of the same. We've defined `copy_len` so that `src_start + copy_len` is still
            // equal to `old_len`, and the caller guaranteed that `old_len >= src_start`. So the
            // sizes are all ok -- and we don't have to worry about initialization (yet) because
            // we're copying `MaybeUninit`s. Also `src` and `dst` can't overlap, but that's
            // trivially true.
            unsafe {
                let keys_ptr = &src.keys as *const _;
                let vals_ptr = &src.vals as *const _;
                let refs_ptr = &src.refs as *const _;
                let keys_src = ArrayHack::get_ptr_unchecked(keys_ptr, src_start);
                let vals_src = ArrayHack::get_ptr_unchecked(vals_ptr, src_start);
                let refs_src = ArrayHack::get_ptr_unchecked(refs_ptr, src_start);

                ptr::copy_nonoverlapping(keys_src, &mut dst.keys as *mut _ as *mut _, copy_len);
                ptr::copy_nonoverlapping(vals_src, &mut dst.vals as *mut _ as *mut _, copy_len);
                ptr::copy_nonoverlapping(refs_src, &mut dst.refs as *mut _ as *mut _, copy_len);
            }

            // update the positions in the store
            let ptr = dst_ptr.cast::<AbstractNode<I, S, P, M>>();
            for i in 0..copy_len as u8 {
                // SAFETY: `i` is in bounds and initialized because we just wrote `copy_len`
                // `RefId`s to `dst.refs`, and we're safe to access through the `UnsafeCell`
                // because there's no other access to `dst` yet.
                unsafe {
                    let r = &*dst.refs.get_unchecked(i as usize).assume_init_ref().get();
                    let handle = SliceHandle {
                        idx: i,
                        node: NodeHandle { ptr, height, borrow: PhantomData },
                    };
                    store.update(r, handle);
                }
            }

            // Extract the key from the middle
            //
            // SAFETY: the calls to `get_unchecked` and `assume_init[_read]` rely on the original
            // length of `self` being greater than `midpoint_idx` (guaranteed by caller), and that
            // the length has now been reduced so that it's no longer required to be initialized
            // (i.e., we can *move* the values out of the index with `assume_init_read`).
            let k = unsafe {
                Key {
                    pos: src.keys.get_unchecked(midpoint_idx as usize).assume_init(),
                    slice: src.vals.get_unchecked(midpoint_idx as usize).assume_init_read(),
                    ref_id: src.refs
                        .get_unchecked(midpoint_idx as usize) // &MaybeUninit<UnsafeCell<Option<RefId>>>
                        .assume_init_read() // UnsafeCell<Option<RefId>>
                        .into_inner(), // Option<RefId>
                }
            };

            // We moved the key out; can't keep any references to it.
            store.suspend(&k.ref_id);

            k
        };

        let height = self.height;
        match self.typed_mut() {
            Type::Leaf(this) => {
                let new_ptr: NonNull<MaybeUninit<Leaf<I, S, P, M>>> =
                    alloc_aligned(MaybeUninit::uninit());

                // SAFETY: there's two things of note here. The call to `with_mut` requires that we
                // don't call any user-defined code, which `move_leaf_parts` is specifically
                // designed not to do. The call to `move_leaf_parts` is also valid because we just
                // allocated `new_ptr` and wrote `new_leaf` to it.
                let key = unsafe {
                    (*new_ptr.as_ptr()).write(new_leaf);
                    this.with_mut(|leaf| move_leaf_parts(leaf, new_ptr.cast()))
                };

                let mut handle = NodeHandle {
                    ptr: new_ptr.cast(),
                    height,
                    borrow: PhantomData as PhantomData<borrow::UniqueOwned>,
                };

                // Update the length of the new node now that everything's been moved
                //
                // SAFETY: `with_mut` requires we don't call user-defined code, which we're clearly
                // not doing here.
                unsafe { handle.as_mut().with_mut(|leaf| leaf.len = copy_len as u8) };

                (key, handle)
            }
            Type::Internal(this) => {
                let mut new_ptr: NonNull<MaybeUninit<Internal<I, S, P, M>>> =
                    alloc_aligned(MaybeUninit::uninit());

                // SAFETY: This is largely the same as the similar block above, for `Type::Leaf`.
                // The other piece of note is that we can leave `(*new_ptr).child_ptrs`
                // uninitialized because `MaybeUninit<[MaybeUninit<T>; N]>` can be safely cast to
                // `[MaybeUninit<T>; N]`.
                let key = unsafe {
                    (*new_ptr.cast::<MaybeUninit<Leaf<I, S, P, M>>>().as_ptr()).write(new_leaf);
                    this.with_mut(|leaf| move_leaf_parts(leaf, new_ptr.cast()))
                };

                // Move all of the child nodes from `this` to `new_ptr`.
                let internal = this.internal();

                // SAFETY: Producing `dst` is safe for the same reason that it is in
                // `move_leaf_parts`: we just allocated and initialized the value, and still have
                // unique access. The calls to `get_ptr_unchecked` and `copy_nonoverlapping` are
                // guaranteed to be sound by the prior bounds of `src_start <= self.leaf().len()`
                // and `src_start + copy_len = self.leaf().len()`.
                unsafe {
                    let child_ptrs = &internal.child_ptrs as *const _;
                    // Note: using `src_start` here -- i.e. the SAME as keys, etc. because the
                    // child to the left of the first key in `dst` has the same index, so we need
                    // to copy from the same index in children as we do the keys.
                    let childs_src = ArrayHack::get_ptr_unchecked(child_ptrs, src_start);

                    let dst_ptr: NonNull<AbstractNode<I, S, P, M>> = new_ptr.cast();
                    let dst: &mut Internal<I, S, P, M> = new_ptr.as_mut().assume_init_mut();
                    let childs_dst = &mut dst.child_ptrs as *mut _ as *mut _;

                    // Note: `copy_len + 1` here because we have to account for the children on
                    // either side of the leftmost *and* rightmost keys being transferred to the
                    // new node.
                    ptr::copy_nonoverlapping(childs_src, childs_dst, copy_len + 1);

                    // At this point, we've initialized everything in the new node, but the children
                    // still have the old parent pointer (and position in the parent). We need to
                    // overwrite those:
                    for ci in 0..(copy_len + 1) {
                        let mut p: NonNull<Leaf<I, S, P, M>> =
                            dst.child_ptrs.get_unchecked(ci).assume_init().cast();

                        // Only overwrite the pointer if we have unique access
                        if p.as_ref().strong_count.is_unique() {
                            let leaf = p.as_mut();
                            leaf.parent = Some(dst_ptr);
                            leaf.idx_in_parent.write(ci as u8);
                        }
                    }
                }

                let mut handle = NodeHandle {
                    ptr: new_ptr.cast(),
                    height,
                    borrow: PhantomData as PhantomData<borrow::UniqueOwned>,
                };

                // Update the length of the new node now that everything's been moved
                //
                // SAFETY: `with_mut` requires we don't call user-defined code, which we're clearly
                // not doing here.
                unsafe { handle.as_mut().with_mut(|leaf| leaf.len = copy_len as u8) };

                (key, handle)
            }
        }
    }
}

// ty::Unknown, borrow::Owned
//  * try_drop
//  * make_unique (where I: Index, P: SupportsInsert)
//  * as_mut_if_unique
//  * clone_root_for_refs_store
impl<I, S, P, const M: usize> NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Decrements the strong count on the references to this node, returning a dropping handle if
    /// there's none left
    ///
    /// This is not just called in destructors; certain inherently racy behavior can mean that we
    /// have to destruct unnecessary COW copies as we traverse the tree with the intent of
    /// modifying it.
    pub fn try_drop(self) -> Option<NodeHandle<ty::Unknown, borrow::Dropping, I, S, P, M>> {
        // `decrement()` returns `true` only if we had the last reference. If (and only if) that's
        // the case, we return a dropping handle.
        if self.leaf().strong_count.decrement() {
            Some(NodeHandle {
                ptr: self.ptr,
                height: self.height,
                borrow: PhantomData,
            })
        } else {
            None
        }
    }

    /// Overwrites the existing pointer with one to a unique copy of this node (if it's not already
    /// unique), and returns a `UniqueOwned` reference to it
    pub fn make_unique(&mut self) -> &mut NodeHandle<ty::Unknown, borrow::UniqueOwned, I, S, P, M>
    where
        I: Index,
        P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
    {
        // SAFETY: `ensure_unique` requires that `**self.ptr` is a valid node at the height we say
        // it is
        unsafe { ensure_unique(&mut self.ptr, self.height) };
        // SAFETY: we're casting the borrow from `Owned` to `UniqueOwned`, which is valid because
        // (a) we just made sure it's unique and (b) the layout of `NodeHandle` is not affected by
        // the borrow used (because it's `#[repr(C)]` and the borrow is in a `PhantomData`)
        unsafe {
            let this = self as *mut NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>;
            let uniq = this as *mut NodeHandle<ty::Unknown, borrow::UniqueOwned, I, S, P, M>;
            &mut *uniq
        }
    }

    /// Converts the reference into one with a mutable borrow, only if our handle on this node is
    /// unique
    pub fn as_mut_if_unique(
        &mut self,
    ) -> Option<&mut NodeHandle<ty::Unknown, borrow::Mut, I, S, P, M>> {
        if self.leaf().strong_count.is_unique() {
            // SAFETY: we're casting the reference here, essentially the same manner as
            // `make_unique`. The layout is invariant under different borrow types, and we're good
            // to provide a `Mut` borrow for the same reason that it's sound for `UniqueOwned` to
            // provide `as_mut`.
            Some(unsafe { &mut *(self as *mut _ as *mut _) })
        } else {
            None
        }
    }

    /// Produces a clone of the root pointer, for use only within the `SliceRefStore`
    ///
    /// ## Safety
    ///
    /// The returned `Owned` handle can only be used inside the `SliceRefStore` attached to the
    /// tree, with all appropriate care taken to avoid double-drops and such.
    pub unsafe fn clone_root_for_refs_store(&self) -> Self {
        NodeHandle { ..*self }
    }
}

// ty::Unknown, borrow::UniqueOwned
//  * into_drop
//  * erase_unique
//  * make_new_parent (where I: Index)
impl<I, S, P, const M: usize> NodeHandle<ty::Unknown, borrow::UniqueOwned, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Converts the `UniqueOwned` borrow into a dropping one, using the existing knowledge
    /// that the reference is unique
    pub fn into_drop(self) -> NodeHandle<ty::Unknown, borrow::Dropping, I, S, P, M> {
        // SAFETY: guaranteed by the methods that produce `UniqueOwned`
        unsafe { weak_assert!(self.leaf().strong_count.is_unique()) };
        NodeHandle {
            ptr: self.ptr,
            height: self.height,
            borrow: PhantomData,
        }
    }

    /// "Erases" the uniqueness from this type, converting its borrow from a [`UniqueOwned`] to an
    /// [`Owned`]
    ///
    /// [`UniqueOwned`]: borrow::UniqueOwned
    /// [`Owned`]: borrow::Owned
    pub fn erase_unique(self) -> NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M> {
        NodeHandle {
            ptr: self.ptr,
            height: self.height,
            borrow: PhantomData,
        }
    }

    /// Overwrites `self` with the result of creating a new parent node containing `self`, `key`,
    /// and `rhs`
    ///
    /// The new node's size is appropriately set from the sizes of `self`, `key_size`, and `rhs`.
    ///
    /// ## Safety
    ///
    /// `self` and `rhs` must not have an existing parent node, and must have the same height
    pub unsafe fn make_new_parent(
        &mut self,
        store: &mut resolve![P::SliceRefStore],
        key: Key<I, S, P, M>,
        key_size: I,
        mut rhs: Self,
    ) where
        I: Index,
    {
        // SAFETY: guaranteed by caller
        unsafe {
            weak_assert!(self.leaf().parent.is_none());
            weak_assert!(rhs.leaf().parent.is_none());
            weak_assert!(self.height == rhs.height);
        }

        let new_height = match self.height.checked_add(1) {
            Some(h) => h,
            None => panic!(
                "tree height overflowed. this should be impossible, and likely indicates a bug"
            ),
        };

        let key_pos = self.leaf().subtree_size();
        let total_size = key_pos.add_right(key_size).add_right(rhs.leaf().subtree_size());

        let new_internal: NonNull<Internal<I, S, P, M>> = alloc_aligned(Internal {
            leaf: Leaf {
                parent: None,
                idx_in_parent: MaybeUninit::uninit(),
                holes: [None; 2],
                len: 1,
                strong_count: P::StrongCount::one(),
                keys: {
                    let mut ks = KeyArray::new();
                    ks.as_mut_slice()[0].write(key_pos);
                    ks
                },
                vals: {
                    let mut vs = KeyArray::new();
                    vs.as_mut_slice()[0].write(key.slice);
                    vs
                },
                refs: {
                    let mut rs = KeyArray::new();
                    rs.as_mut_slice()[0].write(UnsafeCell::new(key.ref_id));
                    rs
                },
                total_size,
            },
            child_ptrs: {
                let mut cs = ChildArray::new();
                cs.as_mut_slice()[0].write(self.ptr);
                cs.as_mut_slice()[1].write(rhs.ptr);
                cs
            },
        });

        // Now that we've allocated the node, we can update the parent pointers in `self` and
        // `rhs`, plus update the position of `key`

        let parent_ptr = new_internal.cast::<AbstractNode<I, S, P, M>>();
        // SAFETY: the unique access required by `borrow_mut` is guaranteed by the caller, and the
        // calls to `with_mut` only require that we don't run user-defined code, which we're
        // clearly not doing here.
        unsafe {
            self.borrow_mut().with_mut(|leaf| {
                leaf.parent = Some(parent_ptr);
                leaf.idx_in_parent = MaybeUninit::new(0);
            });
            rhs.borrow_mut().with_mut(|leaf| {
                leaf.parent = Some(parent_ptr);
                leaf.idx_in_parent = MaybeUninit::new(1);
            });
        }

        // SAFETY: we just successfully allocated the value, so we have valid access to it. The
        // call to `store.update` requires that its arguments are valid, which is trivially true
        // here.
        unsafe {
            let internal_ref = &*new_internal.as_ptr();
            #[rustfmt::skip]
            let r = &*internal_ref.leaf.refs
                .get_unchecked(0)
                .assume_init_ref()
                .get();
            let handle = SliceHandle {
                idx: 0,
                node: NodeHandle {
                    ptr: new_internal.cast::<AbstractNode<I, S, P, M>>(),
                    height: new_height,
                    borrow: PhantomData,
                },
            };
            store.update(r, handle);
        };

        // Finally, update `self` to be the new parent.
        *self = NodeHandle {
            ptr: parent_ptr,
            height: new_height,
            borrow: PhantomData,
        };
    }
}

// ty::Leaf, borrow::UniqueOwned
//  * new_root
impl<I, S, P, const M: usize> NodeHandle<ty::Leaf, borrow::Owned, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Creates a new root node with the given slice and size
    pub fn new_root(slice: S, size: I) -> Self
    where
        I: Index,
    {
        let mut leaf: Leaf<I, S, P, M> = Leaf {
            parent: None,
            idx_in_parent: MaybeUninit::uninit(),
            holes: [None; 2],
            len: 0,
            strong_count: P::StrongCount::one(),
            keys: KeyArray::new(),
            vals: KeyArray::new(),
            refs: KeyArray::new(),
            total_size: size,
        };

        // Set the initial key:
        leaf.keys.as_mut_slice()[0].write(I::ZERO);
        leaf.refs.as_mut_slice()[0].write(Default::default());
        leaf.vals.as_mut_slice()[0].write(slice);
        leaf.len = 1;

        NodeHandle {
            ptr: alloc_aligned(leaf).cast(),
            height: ty::ZeroU8::new(),
            borrow: PhantomData,
        }
    }
}

// ty::Unknown, borrow::Dropping
//  * do_drop
//  * drop_vals
impl<I, S, P, const M: usize> NodeHandle<ty::Unknown, borrow::Dropping, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Actually performs the destructor for a node
    ///
    /// With the way that this method is called, slice reference storage is guaranteed to have been
    /// removed, so they can be `mem::forgot`ten.
    ///
    /// Construction of a "dropping" handle already implies uniqueness; they are only returned by
    /// [`try_drop`](Self::try_drop).
    pub fn do_drop(mut self) {
        // One of the particularly wonderful things about the destructor here is that there just
        // isn't THAT much to do -- the only field in the embedded `Leaf` that needs dropping is
        // `vals`. The other suspects (i.e., things containing `I` or `RefId`) don't need to be
        // dropped because either they implement Copy (as in `I`) or are now nonsense because the
        // slice ref store has been dropped (as in `RefId`)

        let with_leaf_mut = |leaf: &mut Leaf<I, S, P, M>| {
            // Drop `(*leaf_ptr).vals`
            let vals = leaf.vals.as_mut_slice();
            // SAFETY: This is guaranteed already; we have this assertion so that the compiler
            // optimizes away the checks on indexing.
            unsafe { weak_assert!(leaf.len as usize <= vals.len()) };
            let vals = &mut vals[..leaf.len as usize];
            // SAFETY: We're guaranteed that everything that's *not* marked as a hole, within
            // `leaf.len`, is initialized.
            unsafe {
                match leaf.holes {
                    [None, None] => Self::drop_vals(vals),
                    [None, Some(h_nz)] => {
                        let h = h_nz.get() - 1;
                        weak_assert!(h < leaf.len);
                        Self::drop_vals(&mut vals[..h as usize]);
                        // skip vals[h]
                        Self::drop_vals(&mut vals[(h + 1) as usize..]);
                    }
                    [Some(_), None] => weak_unreachable!("invalid state for Leaf holes"),
                    [Some(hx_nz), Some(hy_nz)] => {
                        let (hx, hy) = (hx_nz.get() - 1, hy_nz.get() - 1);
                        weak_assert!(hx < hy);
                        weak_assert!(hy < leaf.len);
                        Self::drop_vals(&mut vals[..hx as usize]);
                        // skip vals[hx]
                        Self::drop_vals(&mut vals[(hx + 1) as usize..hy as usize]);
                        // skip vals[hy]
                        Self::drop_vals(&mut vals[(hy + 1) as usize..]);
                    }
                }
            }

            // Deallocate the node, then handle the children.
            let child_len = leaf.len as usize + 1; // cast before add because len can == u8::MAX
            child_len
        };

        // SAFETY: `borrow_mut` is sound because we have unique access during the destructor. The
        // calls to `with_internal` and `with_mut` rely only on the fact that `with_leaf_mut` does
        // not call any user-defined code, which is simple enough to verify by looking at it.
        let (child_len, child_ptrs) = unsafe {
            match self.borrow_mut().into_typed() {
                Type::Internal(mut node) => node.with_internal(|internal| {
                    let cps = mem::replace(&mut internal.child_ptrs, ChildArray::new());
                    (with_leaf_mut(&mut internal.leaf), Some(cps))
                }),
                Type::Leaf(mut node) => node.with_mut(|leaf| (with_leaf_mut(leaf), None)),
            }
        };

        let this_height = self.height;

        // SAFETY: `dealloc_aligned` requires that the node was allocated with `alloc_aligned`,
        // which must be the case because `node` originated from `self`, and allocated
        // `NodeHandle`s are only produced by `alloc_aligned`. The pointer casts are guaranteed to
        // be valid by the correctness of `into_typed`.
        unsafe {
            match self.into_typed() {
                Type::Leaf(n) => dealloc_aligned(n.ptr.cast::<Leaf<I, S, P, M>>()),
                Type::Internal(n) => dealloc_aligned(n.ptr.cast::<Internal<I, S, P, M>>()),
            }
        };

        if let Some(ps) = child_ptrs {
            let child_height = this_height.checked_sub(1).unwrap_or_else(|| unsafe {
                weak_unreachable!("child pointers only exist for internal nodes, with height >= 1")
            });

            for ptr_uninit in &ps.as_slice()[..child_len] {
                // SAFETY: we know that the pointer's initialized because it's within `child_len`k
                let ptr = unsafe { MaybeUninit::assume_init(*ptr_uninit) };

                let child: NodeHandle<ty::Unknown, _, _, _, _, M> = NodeHandle {
                    ptr,
                    height: child_height,
                    borrow: PhantomData as PhantomData<borrow::Owned>,
                };

                if let Some(drop_handle) = child.try_drop() {
                    drop_handle.do_drop();
                }
            }
        }
    }

    /// Drops the entire slice at once -- i.e. instead of looping through and dropping each value
    /// individually
    ///
    /// ## Safety
    ///
    /// The entries in the slice are assumed to be initialized. They will be dropped (and therefore
    /// uninitialized) after calling this function.
    unsafe fn drop_vals(vals: &mut [MaybeUninit<S>]) {
        // SAFETY: Guaranteed by function safety bounds
        let slice: *mut [S] = unsafe { &mut *(vals as *mut [MaybeUninit<S>] as *mut _) };
        // SAFETY: safety bound for this function guarantees data is initialized (+therefore valid)
        unsafe { std::ptr::drop_in_place(slice) }
    }
}

// ty::Internal
//  * internal
//  * child_size (where I: Copy)
//  * child_pos (where I: Index)
//  * child
//  * into_child (where B: borrow::SupportsInsertIfMut)
//  * typed_ptr
impl<B, I, S, P, const M: usize> NodeHandle<ty::Internal, B, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Produces a reference to the `Internal`
    pub(super) fn internal(&self) -> &Internal<I, S, P, M> {
        // SAFETY: The type hint guarantees that this is an `Internal` node
        unsafe { self.ptr.cast().as_ref() }
    }

    /// Returns an immutable handle for a child of the node
    ///
    /// ## Safety
    ///
    /// `idx` must be less than or equal to `self.leaf().len()`.
    pub unsafe fn child(&self, idx: u8) -> NodeHandle<ty::Unknown, borrow::Immut, I, S, P, M> {
        // SAFETY: Guaranteed by caller
        unsafe { weak_assert!(idx <= self.leaf().len()) };

        // SAFETY: the condition above guarantees that `idx <= self.leaf.len`, which is the section
        // of the node that's initialized.
        let ptr = unsafe { self.internal().child_ptrs.get_unchecked(idx as usize).assume_init() };

        NodeHandle {
            ptr,
            height: self.height.as_u8() - 1,
            borrow: PhantomData,
        }
    }

    /// Returns the size of the child at `child_idx`
    ///
    /// ## Safety
    ///
    /// `child_idx` must be less than or equal to `self.leaf().len()`. Failing this condition
    /// results in immediate UB.
    pub unsafe fn child_size(&self, child_idx: u8) -> I
    where
        I: Copy,
    {
        // SAFETY: `child` just requires that `child_idx <= self.leaf().len()`, which is guaranteed
        // by the caller.
        unsafe { self.child(child_idx).leaf().subtree_size() }
    }

    /// Returns the position of the child at `child_idx`
    ///
    /// ## Safety
    ///
    /// `child_idx` must be less than or equal to `self.leaf().len()`.
    pub unsafe fn child_pos(&self, idx: u8) -> I
    where
        I: Index,
    {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(idx <= self.leaf().len()) };

        // SAFETY: `child_size` requires `idx <= self.leaf().len()`. Guaranteed by caller
        let size = unsafe { self.child_size(idx) };

        let next_key_pos =
            self.leaf().try_key_pos(idx).unwrap_or_else(|| self.leaf().subtree_size());

        next_key_pos.sub_right(size)
    }

    /// Converts this reference into one pointing to the particular child
    ///
    /// ## Behavior for mutable borrows
    ///
    /// Mutable borrows have some extra behavior that's worth going over here -- required for
    /// COW-enabled trees to ensure traversal ends up ok.
    ///
    /// If COW-functionality is enabled AND the child has multiple strong references, then a
    /// shallow clone will be made, the parent's link set, and the strong count to the original
    /// child decremented.
    ///
    /// ## Safety
    ///
    /// This function assumes (without checking) that `child_idx` is within the currently
    /// initialized length of the array *and* that if the borrow is mutable, access to the node is
    /// unique (only relevant, and not trivially true, for COW-enabled trees).
    pub unsafe fn into_child<'t>(self, child_idx: u8) -> NodeHandle<ty::Unknown, B, I, S, P, M>
    where
        B: borrow::SupportsInsertIfMut<I, S, P, M>,
        I: Index,
    {
        unsafe { weak_assert!(child_idx <= self.leaf().len) };

        let ptr = self.typed_ptr();
        // SAFETY: Existence of this handle means both that `self.ptr` is valid and we're allowed
        // to access it. `ty::Internal` guarantees that it's the correct type. Even if we're not
        // using
        let child_ptr_array = unsafe { addr_of_mut!((*ptr.as_ptr()).child_ptrs) };

        // SAFETY: This is sound so long as `child_idx` is within the bounds of initialized
        // children -- which in turn guarantees `get_mut_ptr_unchecked` and the cast away from
        // `MaybeUninit`
        let child_ptr_ptr: *mut NonNull<AbstractNode<I, S, P, M>> = unsafe {
            let p: *mut MaybeUninit<NonNull<_>> =
                ArrayHack::get_mut_ptr_unchecked(child_ptr_array, child_idx as usize);
            p as *mut _
        };

        let ptr_ref = unsafe { &mut *child_ptr_ptr };

        if let Some(r) = B::cast_ref_if_mut(ptr_ref) {
            unsafe { ensure_unique(r, self.height.get() - 1) };

            // Because we have unique mutable access now, make sure that the child's parent pointer
            // is properly set. We can skip this step if COW is not enabled, because it's already
            // guaranteed:
            if P::COW {
                // SAFETY: `ptr` is safe to dereference because `SupportsInsertIfMut` guarantees that
                // it was originally a `borrow::Mut`, and the call to `ensure_unique` ensures that we
                // now have unique mutable access.
                let mut leaf = unsafe { &mut *r.cast::<Leaf<I, S, P, M>>().as_ptr() };
                leaf.parent = Some(self.ptr.cast());
                leaf.idx_in_parent.write(child_idx);
            }
        }

        NodeHandle {
            ptr: *ptr_ref,
            height: self.height.get() - 1,
            borrow: self.borrow,
        }
    }

    /// Returns `self.ptr`, but cast to the appropriate type for this node's `ty::Internal`
    fn typed_ptr(&self) -> NonNull<Internal<I, S, P, M>> {
        self.ptr.cast()
    }
}

// ty::Unknown, borrow::Mut
//  * split_slice_handle
impl<'t, Ty, I, S, P, const M: usize> NodeHandle<Ty, borrow::Mut<'t>, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    pub unsafe fn split_slice_handle(
        &self,
        key_idx: u8,
    ) -> SliceHandle<Ty, borrow::Mut<'t>, I, S, P, M>
    where
        P: SupportsInsert<I, S, M>,
    {
        unsafe { weak_assert!(key_idx < self.leaf().len) };

        SliceHandle {
            node: NodeHandle {
                ptr: self.ptr,
                height: self.height,
                borrow: self.borrow,
            },
            idx: key_idx,
        }
    }
}

// ty::Leaf, borrow::Mut
//  * push_key
//  * insert_key (where I: Copy)
impl<'t, I, S, P, const M: usize> NodeHandle<ty::Leaf, borrow::Mut<'t>, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Inserts the key into this leaf node, shifting the later keys *without* adjusting their
    /// positions
    ///
    /// The position of the inserted slice is set to the previous position of the slice at
    /// `key_idx`, and returned. If `key_idx == self.leaf().len()`, then the returned value is
    /// equal to `self.leaf().subtree_size()`.
    ///
    /// Typically [`set_key_poss_with`] is used to update the position of keys calling this method.
    ///
    /// For the counterpart of this method for `Internal` nodes, see: [`insert_key_and_child`].
    ///
    /// ## Safety
    ///
    /// This method assumes two things:
    ///
    ///  1. That `key_idx` is less than or equal to `self.leaf().len()`, and
    ///  2. That the node is not full -- i.e., `self.leaf().len() < self.leaf().max_len()`
    ///
    /// Failing to satisfy either of these conditions will trigger immediate UB.
    ///
    /// [`set_key_poss_with`]: Self::set_key_poss_with
    /// [`insert_key_and_child`]: Self::insert_key_and_child
    pub unsafe fn insert_key(
        &mut self,
        store: &mut resolve![P::SliceRefStore],
        key_idx: u8,
        slice: S,
    ) -> I
    where
        I: Copy,
    {
        // SAFETY: guaranteed by caller.
        unsafe {
            weak_assert!(key_idx <= self.leaf().len);
            weak_assert!(self.leaf().len < self.leaf().max_len());
        }

        let refid = UnsafeCell::new(<resolve![P::SliceRefStore::OptionRefId]>::default());

        let this_ptr = self.ptr;
        let height = self.height.as_u8();

        // SAFETY: most of the stuff here is guaranteed by the fact that `key_idx` is guaranteed to
        // be within bounds by the caller, plus the node not being full, plus existence of the
        // `NodeHandle` meaning the pointer is valid.
        //
        // The safety around `with_mut` specifically requires that we don't run any user-defined
        // code during the function call, which we can see is the case by looking at it and the
        // knowledge that *no* `SliceRefStore` methods call user-defined code.
        unsafe {
            self.with_mut(|leaf| {
                let pos = leaf.try_key_pos(key_idx).unwrap_or(leaf.total_size);

                shift(&mut leaf.keys, key_idx as usize, leaf.len as usize);
                shift(&mut leaf.vals, key_idx as usize, leaf.len as usize);
                shift(&mut leaf.refs, key_idx as usize, leaf.len as usize);

                leaf.keys.get_mut_unchecked(key_idx as usize).write(pos);
                leaf.vals.get_mut_unchecked(key_idx as usize).write(slice);
                leaf.refs.get_mut_unchecked(key_idx as usize).write(refid);
                leaf.len += 1;

                // Update the positions of all the references to the new, shifted indexes
                for i in (key_idx + 1)..leaf.len {
                    // SAFETY: the relevant piece here is conversion from `&UnsafeCell<_>` to `&_`,
                    // which is sound because we know that no aliases to the node exist, and the
                    // reference is temporary.
                    let r = &*leaf.refs.get_unchecked(i as usize).assume_init_ref().get();
                    let handle = SliceHandle {
                        idx: i,
                        node: NodeHandle { ptr: this_ptr, height, borrow: PhantomData },
                    };
                    store.update(r, handle);
                }

                pos
            })
        }
    }

    /// Appends a key to the end of the node, setting its position to the current subtree size
    ///
    /// `store` is provided so that the key's location can be updated, if there are any existing
    /// [`SliceRef`]s pointing to it.
    ///
    /// See also: [`push_key_and_child`](Self::push_key_and_child)
    ///
    /// ## Safety
    ///
    /// This method requires that `self.leaf().len() < self.leaf().max_len()` -- i.e. that there is
    /// room to add the key. If this is not the case, it will immediately cause UB.
    ///
    /// [`SliceRef`]: crate::SliceRef
    pub unsafe fn push_key(
        &mut self,
        store: &mut resolve![P::SliceRefStore],
        key: Key<I, S, P, M>,
        new_subtree_size: I,
    ) {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(self.leaf().len() < self.leaf().max_len()) };

        let k_idx = self.leaf().len();
        // SAFETY: `with_mut` requires that we don't call any user-defined code. By looking at the
        // function below, we can see that's the case. Inside the closure, we're performing a
        // number of `get_mut_unchecked` calls, each of which are only valid because `k_idx` is
        // within `leaf.max_len()`, and we're guaranteed unique access through the `&mut Leaf`.
        unsafe {
            self.with_mut(|leaf| {
                let pos = mem::replace(&mut leaf.total_size, new_subtree_size);

                leaf.keys.get_mut_unchecked(k_idx as usize).write(pos);
                leaf.vals.get_mut_unchecked(k_idx as usize).write(key.slice);
                let ref_id = UnsafeCell::new(key.ref_id);
                leaf.refs.get_mut_unchecked(k_idx as usize).write(ref_id);

                leaf.len += 1;
            });
        }

        // Update the recorded pointer to the key, if necessary.
        //
        // SAFETY: getting an immutable reference to the OptionRefId is always sound, and we know
        // it's initialized because we did that above.
        unsafe {
            let ref_ptr: *mut resolve![P::SliceRefStore::OptionRefId] =
                self.leaf().refs.get_unchecked(k_idx as usize).assume_init_ref().get();

            let handle = SliceHandle {
                idx: k_idx,
                node: NodeHandle {
                    ptr: self.ptr,
                    height: self.height.as_u8(),
                    borrow: PhantomData,
                },
            };

            store.update(&*ref_ptr, handle);
        }
    }
}

// ty::Internal, borrow::Mut
//  * with_internal
//  * replace_first_child
//  * push_key_and_child
//  * insert_key_and_child (where I: Copy)
impl<'t, I, S, P, const M: usize> NodeHandle<ty::Internal, borrow::Mut<'t>, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Calls the given function with a mutable reference to the internal node
    ///
    /// ## Safety
    ///
    /// This method has the same safety requirements as [`with_mut`]. Refer to that method for more
    /// information.
    ///
    /// [`with_mut`]: Self::with_mut
    pub(super) unsafe fn with_internal<R>(
        &mut self,
        func: impl FnOnce(&mut Internal<I, S, P, M>) -> R,
    ) -> R {
        // SAFETY: The existence of a `borrow::Mut` with `ty::Internal` means that we can construct
        // a `&mut Internal`. The caller guarantees it won't be aliased so calling `func` with it
        // is ok.
        unsafe { func(self.ptr.cast().as_mut()) }
    }

    /// Replaces and returns the internal node's leftmost child
    ///
    /// This function is safe because there's always one more child than key, so there is
    /// guaranteed to be at least one child in any internal node.
    ///
    /// ## Safety
    ///
    /// The child's height must be one less than this node's and we must have unique access to the
    /// child.
    pub unsafe fn replace_first_child(
        &mut self,
        mut child: NodeHandle<ty::Unknown, borrow::UniqueOwned, I, S, P, M>,
    ) -> NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M> {
        let child_height = self.height.as_u8() - 1;

        // SAFETY: guaranteed by caller.
        unsafe { weak_assert!(child.height == child_height) };

        let this_ptr = self.ptr;

        let mut func = |internal: &mut Internal<I, S, P, M>| {
            // set the child's parent information to `self`:
            //
            // SAFETY: `with_mut` requires that we don't call any user-defined code, which we
            // aren't.
            unsafe {
                child.as_mut().with_mut(|leaf| {
                    leaf.parent = Some(this_ptr);
                    leaf.idx_in_parent.write(0);
                })
            }

            // SAFETY: all internal nodes are guaranteed to have at least one initialized child.
            let fst = unsafe { internal.child_ptrs.get_mut_unchecked(0).assume_init_mut() };
            // The replacement is valid because we know that `child` has the correct height.
            mem::replace(fst, child.ptr)
        };

        // SAFETY: `with_internal` requires that we don't run any user-defined code. We aren't.
        let old_ptr = unsafe { self.with_internal(|internal| func(internal)) };

        NodeHandle {
            ptr: old_ptr,
            height: child_height,
            borrow: PhantomData,
        }
    }

    /// Appends a key and child to the end of the node, setting the key's position to the current
    /// subtree size
    ///
    /// `store` is provided so that the key's location can be updated, if there are any existing
    /// [`SliceRef`]s pointing to it.
    ///
    /// See also: [`push_key`](Self::push_key)
    ///
    /// ## Safety
    ///
    /// This method requires that `self.leaf().len() < self.leaf().max_len()` -- i.e. that there is
    /// room to add the key. If this is not the case, it will immediately cause UB.
    ///
    /// [`SliceRef`]: crate::SliceRef
    pub unsafe fn push_key_and_child(
        &mut self,
        store: &mut resolve![P::SliceRefStore],
        key: Key<I, S, P, M>,
        mut child: NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>,
        new_subtree_size: I,
    ) {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(self.leaf().len() < self.leaf().max_len()) };

        let self_ptr = self.ptr;
        let height = self.height.as_u8();

        let func = |internal: &mut Internal<I, S, P, M>| {
            let key_idx = internal.leaf.len as usize;
            let child_idx = key_idx + 1; // the index of a child *after* a key is at `+1`
            let key_pos = mem::replace(&mut internal.leaf.total_size, new_subtree_size);

            // Change `child` to use `self` as the parent, if we have unique access
            if let Some(uniq) = child.as_mut_if_unique() {
                // SAFETY: `as_mut` requires unique access, which is guaranteed because `child` is
                // `Owned` and we've verified that the handle is unique. `with_mut` requires that
                // we don't call any user-defined code, which we aren't.
                unsafe {
                    uniq.with_mut(|leaf| {
                        leaf.parent = Some(self_ptr);
                        leaf.idx_in_parent.write(child_idx as u8);
                    });
                }
            }

            // SAFETY: the various calls to `get_mut_unchecked` are in general sound because
            // they're only required to be within the bounds of `{KeyArray,ChildArray}::LEN`, which
            // is guaranteed by the caller up above. The call to `store.update` is the only other
            // unsafe function, and it basically just requires that the arguments are valid.
            #[rustfmt::skip]
            unsafe {
                internal.leaf.keys.get_mut_unchecked(key_idx).write(key_pos);
                internal.leaf.vals.get_mut_unchecked(key_idx).write(key.slice);
                let r = internal.leaf.refs
                    .get_mut_unchecked(key_idx)
                    .write(UnsafeCell::new(key.ref_id));

                internal.child_ptrs.get_mut_unchecked(child_idx).write(child.ptr);

                // Bump the length to indicate that we've actually added the things.
                internal.leaf.len += 1;

                // And then update the store
                let handle = SliceHandle {
                    idx: key_idx as u8,
                    node: NodeHandle {
                        ptr: self_ptr.cast(),
                        height,
                        borrow: PhantomData,
                    },
                };
                store.update(r.get_mut(), handle);
            };
        };

        // SAFETY: `with_internal` requires that we don't call any user-defined code. it's easy
        // enough to see that we don't, by looking at the contents of `func`.
        unsafe { self.with_internal(func) };
    }

    /// Inserts the key and child into this internal node, shifting the later keys *without*
    /// adjusting their positions
    ///
    /// The child's parent pointer (and index within the parent) is set to point to `self` *only if
    /// we have unique access*.
    ///
    /// The position of the inserted slice is set to the previous position of the slice at
    /// `key_idx`, and returned. If `key_idx == self.leaf().len()`, then the returned value is
    /// equal to `self.leaf().subtree_size()`.
    ///
    /// Typically [`set_key_poss_with`] is used to update the position of keys calling this method.
    ///
    /// For the counterpart of this method for `Leaf` nodes, see: [`insert_key`].
    ///
    /// ## Safety
    ///
    /// This method assumes three things:
    ///
    ///  1. That `key_idx` is less than or equal to `self.leaf().len()`,
    ///  2. That the node is not full -- i.e., `self.leaf().len() < self.leaf().max_len()`, and
    ///  3. That the child has the correct height for a child of this node
    ///
    /// Failing to satisfy any of these conditions will trigger immediate UB.
    ///
    /// [`set_key_poss_with`]: Self::set_key_poss_with
    /// [`insert_key`]: Self::insert_key
    pub unsafe fn insert_key_and_child(
        &mut self,
        store: &mut resolve![P::SliceRefStore],
        key_idx: u8,
        key: Key<I, S, P, M>,
        child: NodeHandle<ty::Unknown, borrow::Owned, I, S, P, M>,
    ) -> I
    where
        I: Copy,
    {
        // SAFETY: guaranteed by caller
        unsafe {
            weak_assert!(key_idx <= self.leaf().len);
            weak_assert!(self.leaf().len < self.leaf().max_len());
            weak_assert!(self.height.as_u8() - 1 == child.height);
        }

        let this_ptr = self.ptr;
        let height = self.height.as_u8();
        let pos = self.leaf().try_key_pos(key_idx).unwrap_or(self.leaf().total_size);
        let func = |internal: &mut Internal<I, S, P, M>| {
            let ki = key_idx as usize;
            let ci = ki + 1;
            let k_len = internal.leaf.len as usize;
            let c_len = k_len + 1;

            // Step 1: Shift all the bits over to make room
            //
            // SAFETY: `shift` requires that `ki <= k_len` and `k_len < leaf.max_len()`. This is
            // guaranteed by the caller, via the bounds above.
            unsafe {
                shift(&mut internal.leaf.keys, ki, k_len);
                shift(&mut internal.leaf.vals, ki, k_len);
                shift(&mut internal.leaf.refs, ki, k_len);
            }
            // SAFETY: the bounds here are essentially the same as above, but the start and end
            // indexes are allowed to be one greater. Adding one means that we're shifting the
            // children starting from *after* the key at `key_idx`, which *is* what we want.
            unsafe {
                shift(&mut internal.child_ptrs, ci, c_len);
            }

            // Step 2: insert the new values
            //
            // SAFETY: the only requirement here is that `ki` and `ci` are within bounds, which
            // we've already established above.
            unsafe {
                internal.leaf.keys.get_mut_unchecked(ki).write(pos);
                internal.leaf.vals.get_mut_unchecked(ki).write(key.slice);
                internal.leaf.refs.get_mut_unchecked(ki).write(UnsafeCell::new(key.ref_id));
                internal.child_ptrs.get_mut_unchecked(ci).write(child.ptr);
            }

            // Now that we've initialized everything (shifted, leaving an uninitialized hole, then
            // initialized the hole), we can update the length.
            internal.leaf.len += 1;

            // Step 3: update the positions of the values
            for i in ki as u8..internal.leaf.len {
                // SAFETY: we know that getting index `i` is valid from everything above; the
                // conversion from `&UnsafeCell<_>` to `&_` is sound because we know that no
                // aliases to the node exist and the reference here is temporary.
                //
                // The call to `store.update` just requires that the arguments are valid.
                unsafe {
                    let r = &*internal.leaf.refs.get_unchecked(i as usize).assume_init_ref().get();
                    let handle = SliceHandle {
                        idx: i,
                        node: NodeHandle { ptr: this_ptr.cast(), height, borrow: PhantomData },
                    };
                    store.update(r, handle);
                }
            }

            // Step 4: update parent positions of all?? children
            //
            // ?? if the tree has COW enabled, we can't guarantee unique access to any child except
            // the one we just added. While we could just skip all children if COW is enabled, that
            // tends to produce inefficient trees (specifically: `Iter` requires a larger stack).
            //
            // While we're at it, we'll also make sure that the parent pointers are correct (which,
            // again: may not be, if COW is enabled.
            for i in ci as u8..=internal.leaf.len {
                // SAFETY: !P::COW and Owned borrows guarantees unique access. The only reason
                // we're not going through `into_child` is because we don't want to deal with
                // `SupportsInsertIfMut`. The child pointer is valid for writes because it's
                // stored here, and the child indexes are valid because everything <= len is
                // valid and our steps above maintained that. Also accessing it as a `*mut
                // Leaf` instead of `*mut <whatever type it is>` is ok because `struct
                // Internal` is `#[repr(C)]` and starts with the `Leaf` it contains.
                unsafe {
                    let mut p: NonNull<Leaf<I, S, P, M>> =
                        internal.child_ptrs.get_unchecked(i as usize).assume_init().cast();

                    // Note: it's always valid to construct an immutable reference to a node
                    // (provided it's not also mutably borrowed, which the borrows on `self`
                    // and `child` guarantee here), and we're only mutably borrowing the node
                    // if there aren't other pointers to it. The existing borrow means we
                    // won't have existing mutable references to it, either.
                    if p.as_ref().strong_count.is_unique() {
                        let leaf = p.as_mut();
                        // For `child` (i = ci), we need to set the parent pointer. But if we
                        // have unique access, it's also worth doing that for the other values
                        // as well; in general, the CPU will be smart enough to allow us to
                        // continue before these writes have made it to the actual RAM.
                        if i == ci as u8 || P::COW {
                            leaf.parent = Some(this_ptr);
                        }
                        // Account for shifting
                        leaf.idx_in_parent.write(i);
                    }
                }
            }
        };

        // SAFETY: `with_internal` requires that `func` doesn't call any user-defined code. This is
        // simple enough to verify by looking at the contents of the function.
        unsafe { self.with_internal(func) };
        pos
    }
}

/// Ensures that the node at `**ptr` is unique, performing a shallow clone and replacing `*ptr` if
/// necessary
///
/// ## Safety
///
/// The node pointer `*ptr` must point to a valid node, with a `height` matching the type of node.
/// The caller must have mutable access to `*ptr` and immutable access to `**ptr` through the
/// lifetime of the call to `ensure_unique`.
unsafe fn ensure_unique<I, S, P, const M: usize>(ptr: &mut NodePtr<I, S, P, M>, height: u8)
where
    I: Index,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    // Even though we don't *actually* own the node, we can pretend like we do for now, because
    // we'll need to call `try_drop` later, which requires `borrow::Owned`. This is safe because
    // the `Owned` borrow doesn't escape this function.
    let node: NodeHandle<ty::Unknown, borrow::Owned, _, _, _, M> =
        NodeHandle { ptr: *ptr, height, borrow: PhantomData };

    if node.leaf().strong_count.is_unique() {
        return;
    }

    // SAFETY: `shallow_clone` requires that `P = AllowCow`, which is guaranteed by a non-unique
    // strong count; all other `P: RleTreeConfig` have always-unique strong count implementations.
    *ptr = unsafe { node.borrow().shallow_clone() }.ptr;

    // After creating a shallow clone, it's possible that the existing node has since become the
    // last reference (aside from the new clone). If that happens, we still need to call its
    // destructor.
    if let Some(dropping_handle) = node.try_drop() {
        dropping_handle.do_drop();
    }
}

/// Helper function to allocate a value, aligned to the typical cache line size
fn alloc_aligned<T>(val: T) -> NonNull<T> {
    // Required for the allocation later, should be removed at compile-time.
    assert!(mem::size_of::<T>() != 0);

    let layout = Layout::new::<T>()
        .align_to(CACHE_LINE_SIZE)
        .unwrap_or_else(|_| panic!("allocation would overflow `usize::MAX`"));

    // SAFETY: `alloc` may produce UB if `layout` has a size of zero. We checked above that the
    // size of `T` (and therefore the size of the layout) is non-zero.
    let maybe_null_ptr = unsafe { alloc::alloc(layout) } as *mut T;

    match NonNull::new(maybe_null_ptr) {
        Some(p) => {
            // SAFETY: `.write()` requires that `p.as_ptr()` is valid for writes, and is
            // properly aligned. This is guaranteed by a non-null return from `alloc::alloc()`
            unsafe { p.as_ptr().write(val) };
            p
        }
        None => alloc::handle_alloc_error(layout),
    }
}

/// Deallocation counterpart to `alloc_aligned`
///
/// ## Safety
///
/// `ptr` must correspond to the result of a prior successful allocation of a value with the same
/// type, from `alloc_aligned`. `ptr` cannot have been deallocated yet.
unsafe fn dealloc_aligned<T>(ptr: NonNull<T>) {
    let layout = Layout::new::<T>()
        .align_to(CACHE_LINE_SIZE)
        // SAFETY: the layout was successfully created above in a corresponding call to
        // `alloc_aligned`, so we *should* be able to recreate it here.
        .unwrap_or_else(|_| unsafe { weak_unreachable!() });

    // SAFETY: `dealloc` requires that the pointer refer to a current allocation (guaranteed by
    // caller) and that the layout is the same as what was originally used (which we can see is the
    // case by comparing with `alloc_aligned`.
    unsafe { alloc::dealloc(ptr.as_ptr() as *mut u8, layout) };
}

/// Shifts the elements `start..end` forward by one in the slice, leaving the value at `start`
/// uninitialized (so long as the range is not empty)
///
/// ## Safety
///
/// The range `start..end` must be valid (i.e. `start <= end`) and within bounds so that
/// `end < A::LEN`. Normally, range operations would require `end <= A::LEN`, but because we're
/// going to write to `end + 1`, we need `end` to be *strictly* less than `A::LEN`.
unsafe fn shift<A: ArrayHack<Element = T>, T>(slice: &mut A, start: usize, end: usize) {
    // SAFETY: guaranteed by caller
    unsafe {
        weak_assert!(start <= end);
        weak_assert!(end < A::LEN);
    }

    // We have to return early if `start == end` so that we don't create a `dst` pointer off the
    // end of the array when `end == A::LEN - 1`.
    if start == end {
        return;
    }

    // SAFETY: all of the values are `MaybeUninit`, so we don't need to worry about initialization
    // status. The pointers at `start` through and including `end` are valid for reads and writes
    // because of (a) the `&mut A` the caller provided and (b) the prior guarantees around the
    // bounds of the range.
    unsafe {
        let src = A::get_ptr_unchecked(slice as *const _, start);
        let dst = A::get_mut_ptr_unchecked(slice as *mut _, start + 1);
        let len = end - start;
        // We could use `slice::rotate_right` here, but that function is actually quite
        // complicated, and `ptr::copy` is an LLVM intrinsic.
        ptr::copy(src, dst, len);
    }
}

// Leaf (node)
//  * len
//  * max_len
//  * is_hole
//  * key_pos (where I: Copy)
//  * try_key_pos (where I: Copy)
//  * keys_pos_slice
//  * subtree_size (where I: Copy)
//  * parent
impl<I, S, P, const M: usize> Leaf<I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Returns the number of keys/values in the node
    pub fn len(&self) -> u8 {
        self.len
    }

    /// Returns the maximum number of keys that any node with this value of `M` supports
    pub fn max_len(&self) -> u8 {
        self.keys.len() as u8
    }

    /// Returns whether the leaf has any holes
    pub fn has_holes(&self) -> bool {
        // from the docs on `Leaf.holes` - the only configurations of `self.holes` with at least
        // one hole *must* have `self.holes[1].is_some()`. Otherwise, there are no holes. So we can
        // just check for that:
        self.holes[1].is_some()
    }

    /// Returns whether `self` has a hole at `k_idx`
    ///
    /// ## Safety
    ///
    /// `k_idx` must be less than `u8::MAX` (guaranteed for all valid key indexes).
    pub unsafe fn is_hole(&self, k_idx: u8) -> bool {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(k_idx < u8::MAX) };

        // This function has been optimized with help from godbolt. With rustc 1.62, it essentially
        // compiles to the following instructions:
        //   a := (*self).idx
        //   a += 1
        //   c := (a == (*holes)[0])
        //   a = (a == (*holes)[1])
        //   return (a || c)
        // I don't think it's feasible to go faster, and LLVM doesn't seem smart enough to produce
        // this from match-based solutions (which tended to use multiple conditions).
        //
        // It's worth optimizing this function in this way because `is_hole` may be quite hot.
        let hole_x = self.holes[0].map(|x| x.get()).unwrap_or(0); // no-op
        let hole_y = self.holes[0].map(|y| y.get()).unwrap_or(0); // no-op

        let plus_one = k_idx + 1;
        plus_one == hole_x || plus_one == hole_y
    }

    /// Returns the position of the key, relative to the start of the node
    ///
    /// A leaf node's first key will always have `key_pos(0) == I::ZERO` because it *is* the start
    /// of the node.
    ///
    /// ## Safety
    ///
    /// `k_idx` must be less than `self.len()`. Failing this condition will result in immediate UB.
    pub unsafe fn key_pos(&self, k_idx: u8) -> I
    where
        I: Copy,
    {
        // SAFETY: guaranteed by caller
        unsafe { weak_assert!(k_idx < self.len) };

        // SAFETY: `self.keys` is initialized up to `self.len`, and `self.len <= self.keys.len()`,
        // so `get_unchecked` and `assume_init` are both sound here.
        unsafe { self.keys.get_unchecked(k_idx as usize).assume_init() }
    }

    /// Fallible, safe version of [`key_pos`](Self::key_pos)
    pub fn try_key_pos(&self, k_idx: u8) -> Option<I>
    where
        I: Copy,
    {
        if k_idx >= self.len {
            return None;
        } else {
            // SAFETY: `key_pos` requires `k_idx < self.len`, which is guaranteed by the condition
            // above.
            Some(unsafe { self.key_pos(k_idx) })
        }
    }

    /// Returns a slice of key positions, relative to the start of the node
    ///
    /// For convenience, [`key_pos`] is provided to get the position of an individual key. This
    /// method is more useful in cases where having access to the entire slice is useful.
    ///
    /// [`key_pos`]: Self::key_pos
    pub fn keys_pos_slice(&self) -> &[I] {
        let ptr = &self.keys as *const _ as *const _;
        let len = self.len as usize;
        // SAFETY: We're guaranteed that `self.keys` is initialized up to `self.len`
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    /// Returns the size of the subtree rooted at this node
    pub fn subtree_size(&self) -> I
    where
        I: Copy,
    {
        self.total_size
    }

    /// If this node has a parent, return a [`Parent`] with information about it, in relation to
    /// this node
    pub fn parent(&self) -> Option<Parent<I, S, P, M>> {
        Some(Parent {
            ptr: self.parent?,
            // SAFETY: As described in the "safety" section of the docs for `idx_in_parent`, the
            // field is initialized when `parent` is not `None`. We just got `ptr` out of it, so we
            // now know `idx_in_parent` is initialized.
            idx_in_parent: unsafe { self.idx_in_parent.assume_init() },
        })
    }
}

/// The pointer to a parent node, alongside the child's index in it
pub struct Parent<I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    pub ptr: NodePtr<I, S, P, M>,
    pub idx_in_parent: u8,
}

// Internal (node)
//  * child_ptrs
impl<I, S, P, const M: usize> Internal<I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Returns the slice of pointers to children, used to determine what the index of a particular
    /// child is
    #[cfg(test)]
    pub fn child_ptrs(&self) -> &[NodePtr<I, S, P, M>] {
        // We need to cast `leaf.len` before incrementing because it can - in rare cases - be equal
        // to u8::MAX
        let child_len = self.leaf.len as usize + 1;
        let children = self.child_ptrs.as_slice().as_ptr();
        // SAFETY: the number of initialized children is always the number of leafs, plus one. So
        // we know that (a) the original array is within that length, and (b) the child pointers
        // are initialized up to `child_len`.
        unsafe { std::slice::from_raw_parts(children as *const _, child_len) }
    }
}

// Any params
//  * borrow (where B: borrow::AsImmut)
//  * erase_type
//  * key_pos
//  * slice_size
//  * is_hole
//  * clone_slice_ref
//  * take_refid
//  * replace_refid
//  * redirect_to
impl<Ty, B, I, S, P, const M: usize> SliceHandle<Ty, B, I, S, P, M>
where
    Ty: TypeHint,
    B: borrow::AsImmut,
    P: RleTreeConfig<I, S, M>,
{
    /// Produces an immutable handle for the slice, borrowing the handle for its duration
    pub fn borrow<'h>(&'h self) -> SliceHandle<Ty, borrow::Immut<'h>, I, S, P, M>
    where
        B: borrow::AsImmut,
    {
        SliceHandle { node: self.node.borrow(), idx: self.idx }
    }

    /// Converts this `SliceHandle` into one with `ty::Unknown` instead of the current type tag
    pub fn erase_type(self) -> <Self as Typed>::Unknown {
        SliceHandle { node: self.node.erase_type(), idx: self.idx }
    }

    /// Returns the position of the slice within its node
    pub fn key_pos(&self) -> I
    where
        I: Copy,
    {
        let idx = self.idx as usize;
        let keys = self.node.leaf().keys.as_slice();
        // SAFETY: the existence of this `SliceHandle` guarantees `self.idx` is less than
        // `self.node.len`, and all keys with index < len are initialized.
        unsafe {
            weak_assert!(idx < keys.len());
            keys[idx].assume_init()
        }
    }

    /// Calculates the size of the slice this handle references, using the information around it in
    /// the node
    pub fn slice_size(&self) -> I
    where
        I: Index,
    {
        let leaf = self.node.leaf();

        // SAFETY: the existence of `self` guarantees that `self.idx` is a valid key
        let k_pos = unsafe { leaf.key_pos(self.idx) };

        let mut next_pos = leaf.try_key_pos(self.idx + 1).unwrap_or_else(|| leaf.subtree_size());

        // SAFETY: `try_child_size` requires the same condition as `key_pos` returning `None`,
        // which we've already guaranteed.
        if let Some(s) = unsafe { self.node.try_child_size(self.idx + 1) } {
            next_pos = next_pos.sub_right(s);
        }

        next_pos.sub_left(k_pos)
    }

    /// Returns whether this handle's slice is currently a "hole"
    fn is_hole(&self) -> bool {
        // SAFETY: `is_hole` requires `self.idx < u8::MAX`, which is guaranteed for all key
        // indexes
        unsafe { self.node.leaf().is_hole(self.idx) }
    }

    /// Creates a `SliceHandle` with a `SliceRef` borrow from this one
    ///
    /// ## Safety
    ///
    /// The new `SliceHandle` cannot be accessed until the borrow used by *this* `SliceHandle` is
    /// released.
    pub unsafe fn clone_slice_ref(&self) -> SliceHandle<Ty, borrow::SliceRef, I, S, P, M> {
        SliceHandle {
            node: NodeHandle {
                ptr: self.node.ptr,
                height: self.node.height,
                borrow: PhantomData as PhantomData<borrow::SliceRef>,
            },
            idx: self.idx,
        }
    }

    /// Removes the `RefId` associated with the slice and returns it, if there is one
    pub fn take_refid(&self) -> resolve![P::SliceRefStore::OptionRefId] {
        let default = <resolve![P::SliceRefStore::OptionRefId]>::default();
        self.replace_refid(default)
    }

    /// Replaces the `RefId`, returning the old value that was there before
    pub fn replace_refid(
        &self,
        new_ref: resolve![P::SliceRefStore::OptionRefId],
    ) -> resolve![P::SliceRefStore::OptionRefId] {
        unsafe {
            let unsafecell =
                self.node.leaf().refs.get_unchecked(self.idx as usize).assume_init_ref();
            mem::replace(&mut *unsafecell.get(), new_ref)
        }
    }

    /// Informs the `SliceRefStore` to redirect any references to `self` to `other`
    pub fn redirect_to(&self, other: &Self, store: &mut resolve![P::SliceRefStore]) {
        // SAFETY: in general, producing a reference to something in `refs` is only valid as long
        // as we don't reentrantly call anything that would also modify the value, which we happen
        // to know `SliceRefStore` won't do for the call to `redirect`.
        #[rustfmt::skip]
        let (self_id, other_id) = unsafe {
            let sid = &mut *self.node.leaf()
                .refs.get_unchecked(self.idx as usize)
                .assume_init_ref()
                .get();
            let oid = &mut *other.node.leaf()
                .refs.get_unchecked(other.idx as usize)
                .assume_init_ref()
                .get();
            (sid, oid)
        };

        store.redirect(self_id, other_id)
    }
}

// any type, borrow::Immut
//  * into_ref
impl<'t, Ty, I, S, P, const M: usize> SliceHandle<Ty, borrow::Immut<'t>, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    /// Produces a reference to the inner slice, lasting for the duration of the original borrow
    ///
    /// ## Panics
    ///
    /// This method will panic if there is currently a hole at the slice
    pub fn into_ref(self) -> &'t S {
        if self.is_hole() {
            panic_invalid_state();
        }

        // SAFETY: `self.idx` is always in bounds in `self.node` (so the calls to `get_unchecked`
        // and `assume_init_ref`) are fine, and it's ok to hold onto the reference for 't because
        // the original borrow to create the `Immut` requires a reference with lifetime 't.
        unsafe {
            let r: &S = self.node.leaf().vals.get_unchecked(self.idx as usize).assume_init_ref();
            // Extend the lifetime beyond borrowing `self`.
            &*(r as *const S)
        }
    }
}

// any type, borrow::Mut
//  * take_value_and_leave_hole
//  * fill_hole
//  * with_slice
impl<'t, Ty, I, S, P, const M: usize> SliceHandle<Ty, borrow::Mut<'t>, I, S, P, M>
where
    Ty: TypeHint,
    P: RleTreeConfig<I, S, M>,
{
    /// Temporarily removes the slice, taking the value
    ///
    /// ## Panics
    ///
    /// This method will panic if there is a hole at the value, or if there is not capacity to make
    /// another hole in the node.
    pub fn take_value_and_leave_hole(&mut self) -> S {
        // SAFETY: `self.idx < self.node.leaf().len && self.node.leaf().len <= u8::MAX`, so the +1
        // can't overflow, and stays non-zero.
        let idx_plus_one = unsafe { NonZeroU8::new_unchecked(self.idx + 1) };

        let new_holes = match self.node.leaf().holes {
            [None, opt] => match opt.cmp(&Some(idx_plus_one)) {
                Ordering::Less => [opt, Some(idx_plus_one)],
                // note: maintains ordering because `opt > Some(idx)` means `opt = Some(i)`, with
                // `i > idx`
                Ordering::Greater => [Some(idx_plus_one), opt],
                Ordering::Equal => panic!("cannot take slice: it is currently a hole"),
            },
            // SAFETY: this configuration is invalid, as described in the "safety" comment above
            // `Leaf.holes`
            [Some(_), None] => unsafe { weak_unreachable!() },
            [Some(hx), Some(hy)] => match hx == idx_plus_one || hy == idx_plus_one {
                true => panic!("cannot take slice: it is currently a hole"),
                false => panic!("cannot take slice: already at capacity for holes"),
            },
        };

        // SAFETY: `with_mut` requires that we don't construct any other references to the leaf. We
        // aren't. For the inner `get_unchecked`/`assume_init_read`: We previously guaranteed that
        // `leaf.vals[self.idx]` wasn't a hole, has been appropriately marked as such (so we won't,
        // e.g., double-drop). Existence of `self` guarantees that `self.idx < leaf.len`, so the
        // value is initialized.
        unsafe {
            self.node.with_mut(|leaf| {
                // Add a hole before taking the value:
                leaf.holes = new_holes;
                leaf.vals.get_unchecked(self.idx as usize).assume_init_read()
            })
        }
    }

    /// Fills a "hole" at this slice created by a call to [`take_value_and_leave_hole`]
    ///
    /// ## Panics
    ///
    /// This method will panic if there is no such hole.
    ///
    /// [`take_value_and_leave_hole`]: Self::take_value_and_leave_hole
    pub fn fill_hole(&mut self, val: S) {
        // SAFETY: `self.idx < self.node.leaf().len && self.node.leaf() <= u8::MAX`, so the +1
        // can't overflow, and stays non-zero.
        let idx_plus_one = unsafe { NonZeroU8::new_unchecked(self.idx + 1) };

        let holes = self.node.leaf().holes;
        let new_holes = if holes[1] == Some(idx_plus_one) {
            [None, holes[0]]
        } else if holes[0] == Some(idx_plus_one) {
            [None, holes[1]]
        } else {
            panic!("cannot put slice back: no hole at index")
        };

        unsafe {
            self.node.with_mut(|leaf| {
                leaf.vals.get_mut_unchecked(self.idx as usize).write(val);
                leaf.holes = new_holes;
            })
        }
    }

    /// Calls the function with a mutable reference to the slice
    ///
    /// ## Panics
    ///
    /// This method internally creates a temporary hole in the node, meaning that this method will
    /// panic if there is already more than one hole.
    pub fn with_slice<R>(&mut self, func: impl FnOnce(&mut S) -> R) -> R {
        let mut slice = self.take_value_and_leave_hole();
        let output = func(&mut slice);
        self.fill_hole(slice);
        output
    }
}

// ty::Unknown, borrow::SliceRef
//  * borrow_slice
//  * range
impl<I, S, P, const M: usize> SliceHandle<ty::Unknown, borrow::SliceRef, I, S, P, M>
where
    P: RleTreeConfig<I, S, M>,
{
    /// Produces a borrow on the slice referenced by this handle, for as long a lifetime is needed
    ///
    /// ## Safety
    ///
    /// The caller must ensure that -- for the lifetime `'s` -- the tree is not modified or
    /// dropped.
    pub unsafe fn borrow_slice<'r>(&self) -> &'r S {
        if self.is_hole() {
            panic_invalid_state();
        }

        // SAFETY: the `get_unchecked` + `assume_init_ref` combo requires that the slice at index
        // `self.idx` is valid and initialized. The existence of a `SliceHandle` guarantees this
        // for its values.
        let key_ref: &S =
            unsafe { self.node.leaf().vals.get_unchecked(self.idx as usize).assume_init_ref() };

        // extend the lifetime of `key_ref`
        //
        // SAFETY: guaranteed by caller.
        unsafe { &*(key_ref as *const S) }
    }

    /// Returns the range of indexes covered by the slice
    pub fn range(&self) -> Range<I>
    where
        I: Index,
    {
        let mut pos = self.key_pos();
        let size = self.slice_size();

        let mut node = self.node.borrow();

        while let Ok((parent, child_idx)) = node.into_parent() {
            // SAFETY: `child_pos` requires that `child_idx` is a valid child index, which is
            // guaranteed because parent information is always correct when `P = AllowSliceRefs`
            pos = pos.add_left(unsafe { parent.child_pos(child_idx) });
            node = parent.erase_type();
        }

        pos..pos.add_right(size)
    }

    /// Creates a [`SliceHandle`] from the raw parts
    ///
    /// ## Safety
    ///
    /// THIS FUNCTION IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPR-
    ///
    /// Anyways. Basically you have to pinkie promise that all of the arguments are valid, that the
    /// [`SliceHandle`] will not be accessed (a) after changes to its node or (b) before all other
    /// changes within the [`RleTree`] have finished.
    ///
    /// [`RleTree`]: super::RleTree
    pub(super) unsafe fn from_raw_parts(ptr: NodePtr<I, S, P, M>, height: u8, idx: u8) -> Self {
        let node = NodeHandle { ptr, height, borrow: PhantomData };
        // SAFETY: Guaranteed by the caller.
        unsafe { weak_assert!(idx < node.leaf().len()) };

        SliceHandle { node, idx }
    }
}
