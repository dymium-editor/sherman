//! Management of nodes

use std::alloc::{self, Layout};
use std::fmt::{self, Debug};
use std::mem;
use std::num::NonZeroU16;
use std::ops::Range;
use std::ptr::NonNull;

use super::borrow::{self, Borrow, Borrowed};
use crate::{DirectionalSub, Index};

pub type Pointer<I, S> = NonNull<Node<I, S>>;
pub type HandleUniqueOwned<I, S> = NodeHandle<borrow::UniqueOwned<Node<I, S>>>;
pub type HandleMut<'t, I, S> = NodeHandle<borrow::Mut<'t, Node<I, S>>>;
pub type HandleImmut<'t, I, S> = NodeHandle<borrow::Immut<'t, Node<I, S>>>;

/// Reference to a node in the tree
pub(super) struct NodeHandle<B>
where
    B: NodeBorrow,
{
    inner: Borrowed<B>,
}

pub(super) trait NodeBorrow: Borrow<Value = Node<Self::Index, Self::Slice>> {
    type Index;
    type Slice;

    type UniqueOwned: NodeBorrow;
    type Mut<'t>: NodeBorrow
    where
        Self::Index: 't,
        Self::Slice: 't;
}

impl<B, I, S> NodeBorrow for B
where
    B: Borrow<Value = Node<I, S>>,
{
    type Index = I;
    type Slice = S;

    type UniqueOwned = borrow::UniqueOwned<Node<Self::Index, Self::Slice>>;
    type Mut<'t>
        = borrow::Mut<'t, Node<Self::Index, Self::Value>>
    where
        I: 't,
        S: 't;
}

borrowable! {
    mod field;

    pub(super) struct Node<I, S> {
        /// Parent pointer, always present so that operations can be done in a loop, rather than
        /// recursively or with an explicit stack.
        parent: Option<(NonNull<Node<I, S>>, Side)>,

        /// The full size of the subtree rooted at this node.
        subtree_size: I,

        /// `Slice` value that this node represents
        ///
        /// This value is always `Some(_)`, except during operations where we temporarily remove it to
        /// be able to call [`Slice::try_join`] on it.
        value: Option<S>,

        /// Total height of the subtree rooted at this node, starting at one.
        height: NonZeroU16,

        /// Left-hand child of this node
        lhs: Option<NodeHandle<borrow::UniqueOwned<Node<I, S>>>>,
        /// Right-hand child of this node
        rhs: Option<NodeHandle<borrow::UniqueOwned<Node<I, S>>>>,
    }
}

// Helper macro for accessing borrowed fields of `Node`, to avoid needing to type out the generics.
macro_rules! f {
    (B::$field:ident) => {
        field::$field<B::Index, B::Slice>
    };
    (Node::$field:ident) => {
        field::$field<I, S>
    }
}

impl<B: NodeBorrow> Debug for NodeHandle<B>
where
    B::Index: Debug,
    B::Slice: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        struct DebugHex<T>(T);
        impl<T: fmt::LowerHex> Debug for DebugHex<T> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::LowerHex::fmt(&self.0, f)
            }
        }

        let mut s = f.debug_struct("Node");
        s.field("addr", &DebugHex(self.addr()));
        let parent = self
            .inner
            .get::<f![B::parent]>()
            .as_ref()
            .map(|(ptr, side)| (DebugHex(ptr.as_ptr().addr()), side));
        s.field("parent", &format_args!("{parent:?}"));
        s.field("subtree_size", self.inner.get::<f![B::subtree_size]>());
        s.field("height", self.inner.get::<f![B::height]>());
        s.field("lhs", self.inner.get::<f![B::lhs]>());
        s.field("value", &format_args!("{:?}", self.inner.get::<f![B::value]>()));
        s.field("rhs", self.inner.get::<f![B::rhs]>());
        s.finish()
    }
}

/// This is relevant as a micro-optimization because the CPU will generally load entire cache lines
/// at a time into the L1/L2 cache,
const CACHE_LINE_SIZE: usize = 64;

// SAFETY: implementations must guarantee that pointers returned by alloc() are properly aligned
// and valid for reads / writes. We guarantee this by using a pointer from alloc::alloc(), which
// makes those guarantees.
unsafe impl<I, S> borrow::Allocatable for Node<I, S> {
    fn alloc() -> NonNull<Self> {
        // Required for the allocation later, should be removed at compile-time.
        assert!(mem::size_of::<Self>() != 0);

        let layout = Layout::new::<Self>()
            .align_to(CACHE_LINE_SIZE)
            .unwrap_or_else(|_| panic!("allocation would overflow `usize::MAX`"));

        // SAFETY: `alloc` may produce UB if `layout` has a size of zero. We checked above that the
        // size of `T` (and therefore the size of the layout) is non-zero.
        let maybe_null_ptr = unsafe { alloc::alloc(layout) } as *mut Self;

        NonNull::new(maybe_null_ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout))
    }

    unsafe fn free(ptr: NonNull<Self>) {
        let layout = Layout::new::<Self>()
            .align_to(CACHE_LINE_SIZE)
            .expect("previously used Layout be recreatable");

        // SAFETY: `dealloc` requires that the pointer refer to a current allocation (guaranteed by
        // caller) and that the layout is the same as what was originally used (which we can see is the
        // case by comparing with `alloc_aligned`.
        unsafe { alloc::dealloc(ptr.as_ptr() as *mut u8, layout) };
    }

    fn pre_drop(this: &mut Borrowed<borrow::UniqueOwned<Node<I, S>>>) {
        let mut handle = NodeHandle { inner: this.borrow_mut() };
        if let Some(n) = handle.take_lhs() {
            drop_owned_node(n);
        }
        if let Some(n) = handle.take_rhs() {
            drop_owned_node(n);
        }
    }
}

impl<I, S> NodeHandle<borrow::UniqueOwned<Node<I, S>>> {
    pub(super) fn alloc_new(slice: S, size: I) -> Self {
        NodeHandle {
            inner: Borrowed::alloc_new(Node {
                parent: None,
                subtree_size: size,
                value: Some(slice),
                height: NonZeroU16::new(1).unwrap(),
                lhs: None,
                rhs: None,
            }),
        }
    }
}

// Helper for dropping `Node<I, S>` to avoid recursion
fn drop_owned_node<I, S>(mut root: HandleUniqueOwned<I, S>) {
    'replace_root: loop {
        let mut node = root.borrow_mut();
        'traverse_left: loop {
            let mut leftmost = loop {
                match node.into_lhs() {
                    Ok(n) => node = n,
                    Err(n) => break n,
                }
            };

            'replace_leftmost: loop {
                let rhs = leftmost.take_rhs();
                if let Some((mut parent, _)) = leftmost.into_parent() {
                    // The parent node's left-hand child is `leftmost`. Having taken `rhs` from it,
                    // `leftmost` currently has no children, so dropping it alone will not recurse.
                    let lhs = parent.take_lhs();
                    drop(lhs);

                    if let Some(n) = rhs {
                        node = parent.insert_lhs(n);
                        // The RHS child might have a left-hand child, so we should continue.
                        continue 'traverse_left;
                    } else {
                        // We just removed the left-hand child from the parent and didn't replace
                        // it, so the parent is now the leftmost child.
                        leftmost = parent;
                        continue 'replace_leftmost;
                    }
                } else {
                    // `leftmost` is currently the root, and currently has no left-hand child.
                    // We should replace the root with rhs, if it exists.
                    drop(root);
                    match rhs {
                        Some(n) => {
                            root = n;
                            continue 'replace_root;
                        }
                        None => return,
                    }
                }
            }
        }
    }
}

//
// Public API
//

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Side {
    Lhs,
    Rhs,
}

impl<B: NodeBorrow> NodeHandle<B> {
    pub(super) fn addr(&self) -> usize {
        self.inner.addr()
    }

    pub(super) fn has_parent(&self) -> bool {
        self.inner.get::<f![B::parent]>().is_some()
    }

    pub(super) fn parent_addr(&self) -> Option<usize> {
        self.inner.get::<f![B::parent]>().as_ref().map(|(ptr, _)| ptr.as_ptr().addr())
    }

    /// Borrows the node, returning an immutable handle to itself with same lifetime as `&self`.
    pub(super) fn borrow<'a>(&'a self) -> NodeHandle<borrow::Immut<'a, Node<B::Index, B::Slice>>> {
        NodeHandle { inner: self.inner.borrow() }
    }

    /// Mutably borrows the node, returning an mutable handle to itself with the same lifetime as
    /// `&mut self`.
    pub(super) fn borrow_mut<'a>(
        &'a mut self,
    ) -> NodeHandle<borrow::Mut<'a, Node<B::Index, B::Slice>>>
    where
        B: borrow::BorrowAsMut,
    {
        NodeHandle { inner: self.inner.borrow_mut() }
    }

    /// Returns the height of the subtree rooted at this node
    ///
    /// Empty children have zero height, so the return value is necessarily non-zero.
    pub(super) fn height(&self) -> NonZeroU16 {
        *self.inner.get::<f![B::height]>()
    }

    /// Updates this node's value of the subtree height
    pub(super) fn set_height(&mut self, height: NonZeroU16)
    where
        B: borrow::BorrowAsMut,
    {
        *self.inner.get_mut::<f![B::height]>() = height;
    }

    /// Returns the height of the subtree rooted at this node's left-hand child
    ///
    /// If this node has no left-hand child, then the return value will be zero.
    pub(super) fn lhs_height(&self) -> u16 {
        self.lhs().map(|n| n.height().get()).unwrap_or(0)
    }

    /// Returns the height of the subtree rooted at this node's right-hand child
    ///
    /// If this node has no right-hand child, then the return value will be zero.
    pub(super) fn rhs_height(&self) -> u16 {
        self.rhs().map(|n| n.height().get()).unwrap_or(0)
    }

    /// Returns the size of the subtree rooted at this node *as given by this handle*.
    ///
    /// Note that this may return an incorrect value if `self` was reborrowed and its subtree size
    /// was changed.
    pub(super) fn subtree_size(&self) -> B::Index
    where
        B::Index: Copy,
    {
        *self.inner.get::<f![B::subtree_size]>()
    }

    /// Sets the recorded size of the subtree rooted at this node, only for this handle.
    ///
    /// Note that if this is not a `UniqueOwned` handle, the underlying node will have to be
    /// udpated for the change to be reflected.
    pub(super) fn set_subtree_size(&mut self, size: B::Index)
    where
        B: borrow::BorrowAsMut,
    {
        *self.inner.get_mut::<f![B::subtree_size]>() = size;
    }

    /// Returns the range of this node's subtree that's taken up only by the value for this node
    ///
    /// When searching within the subtree rooted at this node, an index less than
    /// `value_range().start` means it's in the left-hand subtree, and an index greater than or
    /// equal to `value_range().end` means it's in the right-hand subtree.
    pub(super) fn value_range(&self) -> Range<B::Index>
    where
        B::Index: Index,
    {
        let lhs_size = self.lhs().map(NodeHandle::subtree_size).unwrap_or(Index::ZERO);
        let rhs_size = self.rhs().map(NodeHandle::subtree_size).unwrap_or(Index::ZERO);

        let start = lhs_size;
        let end = self.subtree_size().sub_right(rhs_size);

        start..end
    }

    /// Removes the "slice" value from the node, returning it.
    ///
    /// This is should only be used for temporary operations that require ownership, like
    /// [`Slice::try_join`].
    ///
    /// Once done, you should call [`set_value`] to place it back.
    ///
    /// # Panics
    ///
    /// This method panics if the slice value was previously taken without being put back via
    /// [`set_value`].
    pub(super) fn take_value(&mut self) -> B::Slice
    where
        B: borrow::BorrowAsMut,
    {
        let v = self.inner.get_mut::<f![B::value]>().take();
        match v {
            Some(v) => v,
            None => panic!("internal error: cannot `take_value()` that is already absent"),
        }
    }

    /// Replaces the "slice" value that was previously removed by [`take_value`].
    ///
    /// # Panics
    ///
    /// This method panics if the slice value is already present in the node.
    pub(super) fn set_value(&mut self, value: B::Slice)
    where
        B: borrow::BorrowAsMut,
    {
        let old = self.inner.get_mut::<f![B::value]>().replace(value);
        if old.is_some() {
            panic!("internal error: cannot `set_value()` that is already present");
        }
    }

    /// Removes and returns the left-hand child of this node, if there is one
    pub(super) fn take_lhs(
        &mut self,
    ) -> Option<NodeHandle<borrow::UniqueOwned<Node<B::Index, B::Slice>>>>
    where
        B: borrow::BorrowAsMut,
    {
        let mut node_opt = self.inner.get_mut::<f![B::lhs]>().take();
        if let Some(n) = node_opt.as_mut() {
            // Reset the parent pointer so that upward tree traversals are sound.
            *n.inner.get_mut::<f![B::parent]>() = None;
        }
        node_opt
    }

    /// Removes and returns the right-hand child of this node, if there is one
    pub(super) fn take_rhs(
        &mut self,
    ) -> Option<NodeHandle<borrow::UniqueOwned<Node<B::Index, B::Slice>>>>
    where
        B: borrow::BorrowAsMut,
    {
        let mut node_opt = self.inner.get_mut::<f![B::rhs]>().take();
        if let Some(n) = node_opt.as_mut() {
            // Reset the parent pointer so that upward tree traversals are sound.
            *n.inner.get_mut::<f![B::parent]>() = None;
        }
        node_opt
    }
}

impl<'t, I, S> NodeHandle<borrow::Immut<'t, Node<I, S>>> {
    /// Copies the `NodeHandle`
    pub(super) fn reborrow(&self) -> Self {
        NodeHandle { inner: self.inner.reborrow() }
    }

    /// Returns a reference to the slice value for the node
    ///
    /// # Panics
    ///
    /// This method panics if the slice value is not currently present, e.g. due to a prior call to
    /// [`take_value`] without a corresponding [`set_value`].
    pub(super) fn value(&self) -> &'t S {
        match self.inner.reborrow().into_ref::<f![Node::value]>() {
            Some(v) => v,
            None => panic!(
                "internal error: `value` should not be None except during temporary operations"
            ),
        }
    }

    /// Produces a reference to this node's parent
    pub(super) fn into_parent(self) -> Option<(Self, Side)> {
        let parent = self.inner.into_ref::<f![Node::parent]>();
        parent.map(|(p, side)| {
            // SAFETY: `p` is properly aligned (like all node pointers are), and the invariants of
            // the tree guarantee that `p` still points to a valid `Node`.
            let inner = unsafe { <Borrowed<borrow::Immut<_>>>::from_non_null(p) };
            (NodeHandle { inner }, side)
        })
    }

    /// Returns an immutable handle to the left-hand child
    pub(super) fn into_lhs(self) -> Option<Self> {
        self.inner.into_ref::<f![Node::lhs]>().as_ref().map(|n| n.borrow())
    }

    /// Returns an immutable handle to the right-hand child
    pub(super) fn into_rhs(self) -> Option<Self> {
        self.inner.into_ref::<f![Node::rhs]>().as_ref().map(|n| n.borrow())
    }
}

impl<'t, I, S> NodeHandle<borrow::Mut<'t, Node<I, S>>> {
    /// Creates a reference from a raw `NonNull`
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by a previous call to [`NodeHandle::ptr`], and the
    /// borrow must be valid for the lifetime of that pointer. It is the caller's responsibility to
    /// ensure that Rust's aliasing requirements are satisfied.
    pub(super) unsafe fn from_ptr(pointer: Pointer<I, S>) -> Self {
        // SAFETY: the pointer must be properly aligned, point to a value of `Node<I, S>`, and be
        // valid for the lifetime. That's all guaranteed by the caller.
        unsafe {
            NodeHandle {
                inner: <Borrowed<borrow::Mut<_>>>::from_non_null(pointer),
            }
        }
    }

    /// Returns the raw `NonNull` represented by this handle
    pub(super) fn ptr(&self) -> Pointer<I, S> {
        self.inner.as_ptr()
    }

    /// Produces a reference to this node's parent
    pub(super) fn into_parent(self) -> Option<(Self, Side)> {
        let parent = self.inner.into_mut::<f![Node::parent]>();
        parent.map(|(p, side)| {
            // SAFETY: `p` is properly aligned (like all node pointers are), and the invariants of
            // the tree guarantee that `p` still points to a valid `Node`.
            let inner = unsafe { <Borrowed<borrow::Mut<_>>>::from_non_null(p) };
            (NodeHandle { inner }, side)
        })
    }

    /// Returns a mutable handle to the left-hand child, or `self` if there is no such child
    ///
    /// The returned handle has the same lifetime as `self`.
    pub(super) fn into_lhs(self) -> Result<Self, Self> {
        match self.inner.try_into_mut::<f![Node::lhs], _>() {
            Ok(lhs) => Ok(lhs.borrow_mut()),
            Err(this) => Err(NodeHandle { inner: this }),
        }
    }

    /// Returns a mutable handle to the right-hand child, or `self` if there is no such child
    ///
    /// The returned handle has the same lifetime as `self`.
    pub(super) fn into_rhs(self) -> Result<Self, Self> {
        match self.inner.try_into_mut::<f![Node::rhs], _>() {
            Ok(rhs) => Ok(rhs.borrow_mut()),
            Err(this) => Err(NodeHandle { inner: this }),
        }
    }

    /// Sets the left-hand child to `lhs`, returning a mutable handle to the new child
    ///
    /// **NOTE:** This node's internal accounting for the relative height of each child *will not
    /// be updated*; you must call `fix()` on this node, or `fix_{lhs,rhs}()` on the parent to
    /// properly rebalance the tree.
    ///
    /// **NOTE:** This node's subtree size *will not be updated*; you must call
    /// `set_subtree_size()` on this node to update its size after adding the new child.
    ///
    /// # Panics
    ///
    /// This method panics if this node already has a left-hand child.
    pub(super) fn insert_lhs(self, mut lhs: NodeHandle<borrow::UniqueOwned<Node<I, S>>>) -> Self {
        let this_ptr = self.inner.as_ptr();

        let this_lhs = self.inner.into_mut::<f![Node::lhs]>();

        if this_lhs.is_some() {
            panic!("internal error: cannot `insert_lhs()` that is already present")
        }

        *lhs.inner.get_mut::<f![Node::parent]>() = Some((this_ptr, Side::Lhs));

        this_lhs.insert(lhs).borrow_mut()
    }

    /// Sets the right-hand child to `rhs`, returning a mutable handle to the new child
    ///
    /// **NOTE:** The subtree rooted at `self` may be left unbalanced until you call `fix()` on
    /// this node, or `fix_{lhs,rhs}()` on the parent.
    ///
    /// # Panics
    ///
    /// This method panics if this node already has a right-hand child.
    pub(super) fn insert_rhs(self, mut rhs: NodeHandle<borrow::UniqueOwned<Node<I, S>>>) -> Self {
        let this_ptr = self.inner.as_ptr();

        let this_rhs = self.inner.into_mut::<f![Node::rhs]>();

        if this_rhs.is_some() {
            panic!("internal error: cannot `insert_rhs()` that is already present")
        }

        *rhs.inner.get_mut::<f![Node::parent]>() = Some((this_ptr, Side::Rhs));

        this_rhs.insert(rhs).borrow_mut()
    }
}

//
// Internal helpers
//
impl<B: NodeBorrow> NodeHandle<B> {
    fn lhs(&self) -> Option<&NodeHandle<borrow::UniqueOwned<Node<B::Index, B::Slice>>>> {
        self.inner.get::<f![B::lhs]>().as_ref()
    }

    fn rhs(&self) -> Option<&NodeHandle<borrow::UniqueOwned<Node<B::Index, B::Slice>>>> {
        self.inner.get::<f![B::rhs]>().as_ref()
    }
}
