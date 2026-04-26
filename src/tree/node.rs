//! Management of nodes

#[cfg(test)]
use std::fmt::{self, Debug};
use std::num::{NonZeroU16, NonZeroUsize};
use std::ops::Range;
use std::ptr::NonNull;

use super::borrow::{self, Borrow, Borrowed};
use super::rc::{BorrowState, Redirect, RefCount};
#[expect(unused)] // only added for docs
use crate::Slice;
use crate::param::{self, RleTreeConfig, SupportsUpdate};
use crate::{DirectionalSub, Index};

pub(super) type Pointer<I, S, P> = NonNull<Node<I, S, P>>;
pub(super) type HandleOwned<I, S, P> = NodeHandle<borrow::Owned<Node<I, S, P>>>;
pub(super) type HandleWeak<I, S, P> = NodeHandle<borrow::Weak<Node<I, S, P>>>;
pub(super) type HandleUniqueOwned<I, S, P> = NodeHandle<borrow::UniqueOwned<Node<I, S, P>>>;
pub(super) type HandleMut<'t, I, S, P> = NodeHandle<borrow::Mut<'t, Node<I, S, P>>>;
pub(crate) type HandleImmut<'t, I, S, P> = NodeHandle<borrow::Immut<'t, Node<I, S, P>>>;
pub(crate) type HandleWeakImmut<'t, I, S, P> = NodeHandle<borrow::WeakImmut<'t, Node<I, S, P>>>;

/// Reference to a node in the tree
pub struct NodeHandle<B>
where
    B: NodeBorrow,
{
    inner: Borrowed<B>,
}

pub trait NodeBorrow: Borrow<Value = Node<Self::Index, Self::Slice, Self::Param>> {
    type Index;
    type Slice;
    type Param: RleTreeConfig<Self::Index, Self::Slice>;

    type UniqueOwned: NodeBorrow;
    type Mut<'t>: NodeBorrow
    where
        Self::Index: 't,
        Self::Slice: 't;
}

impl<B, I, S, P> NodeBorrow for B
where
    B: Borrow<Value = Node<I, S, P>>,
    P: RleTreeConfig<I, S>,
{
    type Index = I;
    type Slice = S;
    type Param = P;

    type UniqueOwned = borrow::UniqueOwned<Node<Self::Index, Self::Slice, Self::Param>>;
    type Mut<'t>
        = borrow::Mut<'t, Node<Self::Index, Self::Slice, Self::Param>>
    where
        I: 't,
        S: 't;
}

borrowable! {
    mod field;

    #[repr(align(64))] // micro-optimization: align to cache line size
    pub struct Node<I, S, P>
    where
        P: RleTreeConfig<I, S>,
    {
        /// Parent pointer, always present so that operations can be done in a loop, rather than
        /// recursively or with an explicit stack.
        parent: Option<(NonNull<Node<I, S, P>>, Side)>,

        (!mut) strong_count: <P as RleTreeConfig<I, S>>::StrongCount,
        (!mut) weak_count: <P as RleTreeConfig<I, S>>::WeakCount,
        (!mut) borrow_state: <P as RleTreeConfig<I, S>>::BorrowState,
        redirect: <P as RleTreeConfig<I, S>>::Redirect,

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
        lhs: Option<Borrowed<borrow::Owned<Node<I, S, P>>>>,
        /// Right-hand child of this node
        rhs: Option<Borrowed<borrow::Owned<Node<I, S, P>>>>,
    }
}

// Helper macro for accessing borrowed fields of `Node`, to avoid needing to type out the generics.
macro_rules! f {
    (B::$field:ident) => {
        field::$field<B::Index, B::Slice, B::Param>
    };
    (Node::$field:ident) => {
        field::$field<I, S, P>
    }
}

#[cfg(test)]
impl<B: NodeBorrow> Debug for NodeHandle<B>
where
    B: borrow::BorrowAsImmut,
    B::Index: Debug,
    B::Slice: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::MaybeDebug;

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
        if B::Param::COW {
            s.field("strong_count", self.inner.get::<f![B::strong_count]>().fallible_debug());
        }
        if B::Param::REFS {
            s.field(
                "borrow_state",
                &format_args!("{:?}", self.inner.get::<f![B::borrow_state]>().fallible_debug()),
            );
        }
        s.field("subtree_size", self.inner.get::<f![B::subtree_size]>());
        s.field("height", self.inner.get::<f![B::height]>());
        s.field("lhs", &self.borrow().into_lhs());
        s.field("value", &format_args!("{:?}", self.inner.get::<f![B::value]>()));
        s.field("rhs", &self.borrow().into_rhs());
        s.finish()
    }
}

// SAFETY: implementations must guarantee that pointers returned by alloc() are properly aligned
// and valid for reads / writes. We guarantee this by using a pointer from alloc::alloc(), which
// makes those guarantees.
unsafe impl<I, S, P> borrow::Borrowable for Node<I, S, P>
where
    P: RleTreeConfig<I, S>,
{
    type StrongCountField = f![Node::strong_count];
    type WeakCountField = f![Node::weak_count];
    type BorrowStateField = f![Node::borrow_state];

    fn drop_strong(this: &mut Borrowed<borrow::UniqueOwned<Node<I, S, P>>>) {
        let mut handle = NodeHandle { inner: this.borrow_mut() };
        _ = handle.inner.get_mut::<f![Node::value]>().take();
        if let Some(n) = handle.take_lhs_unique_or_discard() {
            drop_owned_node(n);
        }
        if let Some(n) = handle.take_rhs_unique_or_discard() {
            drop_owned_node(n);
        }
    }

    fn drop_weak(this: &mut Borrowed<borrow::Weak<Node<I, S, P>>>) {
        let h = NodeHandle { inner: this.access() };
        if let Some(n) = h.take_redirect() {
            drop_weak_redirect(n);
        }
    }
}

impl<I, S, P> HandleUniqueOwned<I, S, P>
where
    P: RleTreeConfig<I, S>,
{
    pub(super) fn alloc_new(slice: S, size: I) -> Self {
        NodeHandle {
            inner: Borrowed::alloc_new(Node {
                parent: None,
                strong_count: RefCount::one(),
                weak_count: RefCount::one(),
                borrow_state: BorrowState::new(),
                redirect: Redirect::empty(),
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
fn drop_owned_node<I, S, P>(mut root: HandleUniqueOwned<I, S, P>)
where
    P: RleTreeConfig<I, S>,
{
    'replace_root: loop {
        let mut node = root.borrow_mut();
        'traverse_left: loop {
            let mut leftmost = loop {
                match node.into_lhs_unique_or_discard() {
                    Ok(n) => node = n,
                    Err(n) => break n,
                }
            };

            'replace_leftmost: loop {
                let rhs = leftmost.take_rhs_unique_or_discard();
                if let Some((mut parent, _)) = leftmost.into_parent() {
                    // The parent node's left-hand child is `leftmost`. Having taken `rhs` from it,
                    // `leftmost` currently has no children, so dropping it alone will not recurse.
                    let lhs = parent.take_lhs_unique_or_discard();
                    drop(lhs);

                    if let Some(n) = rhs {
                        node = parent.insert_into_lhs(n);
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

// Helper for dropping the redirect chain from weak handles, to avoid recursion
fn drop_weak_redirect<I, S, P>(start: HandleWeak<I, S, P>)
where
    P: RleTreeConfig<I, S>,
{
    // All while weak reference we're looking at is unique, keep following the redirect chain to
    // clean up the values. Because strong references implicitly hold a shared weak reference, if
    // we have a weak handle and there is only a single weak reference, then we know that we have
    // the only surviving reference to that node.
    //
    // To avoid recursion, we should pull that node's `redirect` out before dropping it, so we can
    // proceed to the next.
    let mut next = Some(start);
    while let Some(n) = next {
        // Check uniqueness *first*, before trying to upgrade our reference. Otherwise, if not
        // unique, we might fail because the node is already mutably borrowed, which we really
        // should not do in a destructor.
        if !n.inner.get_immut::<f![Node::weak_count]>().is_unique() {
            break;
        }

        // Because our handle on the node is unique, it's sound to remove the redirect. Nothing
        // else will need it.
        let h = NodeHandle { inner: n.inner.access() };
        next = h.take_redirect();
    }
}

//
// Public API
//

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(test, derive(Debug))]
pub enum Side {
    Lhs,
    Rhs,
}

impl<B: NodeBorrow> NodeHandle<B> {
    /// Returns a `NonZeroUsize` with the address of the `NonNull` represented by this handle
    pub(super) fn addr(&self) -> NonZeroUsize {
        self.ptr().addr()
    }

    /// Returns the raw `NonNull` represented by this handle
    pub(super) fn ptr(&self) -> Pointer<B::Index, B::Slice, B::Param> {
        self.inner.as_ptr()
    }

    pub(super) fn parent(&self) -> Option<(Pointer<B::Index, B::Slice, B::Param>, Side)>
    where
        B: borrow::BorrowAsImmut,
    {
        *self.inner.get::<f![B::parent]>()
    }

    pub(super) fn parent_addr(&self) -> Option<NonZeroUsize>
    where
        B: borrow::BorrowAsImmut,
    {
        self.parent().map(|(p, _)| p.addr())
    }

    /// Borrows the node, returning an immutable handle to itself with same lifetime as `&self`.
    pub(super) fn borrow<'a>(
        &'a self,
    ) -> NodeHandle<borrow::Immut<'a, Node<B::Index, B::Slice, B::Param>>>
    where
        B: borrow::BorrowAsImmut,
    {
        NodeHandle { inner: self.inner.borrow() }
    }

    /// Creates a weak reference to the underlying node
    pub(super) fn weak(&self) -> HandleWeak<B::Index, B::Slice, B::Param>
    where
        B: borrow::BorrowAsImmut,
    {
        if !B::Param::REFS {
            panic!("internal error: cannot create weak references when !P::REFS");
        }

        NodeHandle { inner: self.inner.weak() }
    }

    /// Replaces the node's `redirect` field with `None` and returns its current handle, if any
    pub(super) fn take_redirect(&self) -> Option<HandleWeak<B::Index, B::Slice, B::Param>>
    where
        B: borrow::BorrowAsImmut,
    {
        self.inner.get::<f![B::redirect]>().replace(None)
    }

    /// Sets the node's `redirect` field
    pub(super) fn set_redirect(&self, weak: HandleWeak<B::Index, B::Slice, B::Param>)
    where
        B: borrow::BorrowAsImmut,
    {
        self.inner.get::<f![B::redirect]>().replace(Some(weak));
    }

    pub(super) fn write_redirect(
        &mut self,
        redirect: <B::Param as RleTreeConfig<B::Index, B::Slice>>::Redirect,
    ) where
        B: borrow::BorrowAsMut,
    {
        *self.inner.borrow_mut().get_mut::<f![B::redirect]>() = redirect;
    }

    /// Mutably borrows the node, returning an mutable handle to itself with the same lifetime as
    /// `&mut self`.
    pub(super) fn borrow_mut<'a>(
        &'a mut self,
    ) -> NodeHandle<borrow::Mut<'a, Node<B::Index, B::Slice, B::Param>>>
    where
        B: borrow::BorrowAsMut,
    {
        NodeHandle { inner: self.inner.borrow_mut() }
    }

    /// Returns the height of the subtree rooted at this node
    ///
    /// Empty children have zero height, so the return value is necessarily non-zero.
    pub(super) fn height(&self) -> NonZeroU16
    where
        B: borrow::BorrowAsImmut,
    {
        *self.inner.get::<f![B::height]>()
    }

    /// Updates this node's value of the subtree height
    pub(super) fn set_height(&mut self, height: NonZeroU16)
    where
        B: borrow::BorrowAsMut,
    {
        *self.inner.borrow_mut().get_mut::<f![B::height]>() = height;
    }

    /// Returns the height of the subtree rooted at this node's left-hand child
    ///
    /// If this node has no left-hand child, then the return value will be zero.
    pub(super) fn lhs_height(&self) -> u16
    where
        B: borrow::BorrowAsImmut,
    {
        self.borrow().into_lhs().map(|n| n.height().get()).unwrap_or(0)
    }

    /// Returns the height of the subtree rooted at this node's right-hand child
    ///
    /// If this node has no right-hand child, then the return value will be zero.
    pub(super) fn rhs_height(&self) -> u16
    where
        B: borrow::BorrowAsImmut,
    {
        self.borrow().into_rhs().map(|n| n.height().get()).unwrap_or(0)
    }

    /// Returns the size of the subtree rooted at this node *as given by this handle*.
    ///
    /// Note that this may return an incorrect value if `self` was reborrowed and its subtree size
    /// was changed.
    pub(super) fn subtree_size(&self) -> B::Index
    where
        B: borrow::BorrowAsImmut,
        B::Index: Copy,
    {
        *self.inner.get::<f![B::subtree_size]>()
    }

    /// Sets the recorded size of the subtree rooted at this node, only for this handle.
    ///
    /// Note that if this is not a `UniqueOwned` handle, the underlying node will have to be
    /// updated for the change to be reflected.
    pub(super) fn set_subtree_size(&mut self, size: B::Index)
    where
        B: borrow::BorrowAsMut,
    {
        *self.inner.borrow_mut().get_mut::<f![B::subtree_size]>() = size;
    }

    /// Returns the range of this node's subtree that's taken up only by the value for this node
    ///
    /// When searching within the subtree rooted at this node, an index less than
    /// `value_range().start` means it's in the left-hand subtree, and an index greater than or
    /// equal to `value_range().end` means it's in the right-hand subtree.
    pub(super) fn value_range(&self) -> Range<B::Index>
    where
        B: borrow::BorrowAsImmut,
        B::Index: Index,
    {
        let lhs_size = self.borrow().into_lhs().map(|n| n.subtree_size()).unwrap_or(Index::ZERO);
        let rhs_size = self.borrow().into_rhs().map(|n| n.subtree_size()).unwrap_or(Index::ZERO);

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
    ///
    /// [`set_value`]: Self::set_value
    pub(super) fn take_value(&mut self) -> B::Slice
    where
        B: borrow::BorrowAsMut,
    {
        let v = self.inner.borrow_mut().get_mut::<f![B::value]>().take();
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
    ///
    /// [`take_value`]: Self::take_value
    pub(super) fn set_value(&mut self, value: B::Slice)
    where
        B: borrow::BorrowAsMut,
    {
        let old = self.inner.borrow_mut().get_mut::<f![B::value]>().replace(value);
        if old.is_some() {
            panic!("internal error: cannot `set_value()` that is already present");
        }
    }

    /// Removes and returns the left-hand child of this node, if there is one
    pub(super) fn take_lhs(&mut self) -> Option<HandleOwned<B::Index, B::Slice, B::Param>>
    where
        B: borrow::BorrowAsMut,
    {
        let mut node = NodeHandle {
            inner: self.inner.borrow_mut().get_mut::<f![B::lhs]>().take()?,
        };

        // Optimistically reset the parent.
        match node.inner.try_as_unique() {
            Ok(n) => *n.borrow_mut().get_mut::<f![B::parent]>() = None,
            Err(()) => assert!(B::Param::COW, "internal error: !P::COW and not unique"),
        }
        Some(node)
    }

    /// Like `take_lhs`, but also discards the node if is not unique.
    pub(super) fn take_lhs_unique_or_discard(
        &mut self,
    ) -> Option<HandleUniqueOwned<B::Index, B::Slice, B::Param>>
    where
        B: borrow::BorrowAsMut,
    {
        self.take_lhs().and_then(|n| n.into_unique_or_discard())
    }

    /// Removes and returns the right-hand child of this node, if there is one
    pub(super) fn take_rhs(
        &mut self,
    ) -> Option<NodeHandle<borrow::Owned<Node<B::Index, B::Slice, B::Param>>>>
    where
        B: borrow::BorrowAsMut,
    {
        let mut node = NodeHandle {
            inner: self.inner.borrow_mut().get_mut::<f![B::rhs]>().take()?,
        };
        // If this tree is NOT copy-on-write, then we should guarantee early that the parent
        // pointer is correct. With COW, that might result in over-eager cloning, so we'll defer
        // that until later points where we actually use it.
        if let Ok(n) = node.inner.try_as_unique() {
            *n.borrow_mut().get_mut::<f![B::parent]>() = None;
        }
        Some(node)
    }

    /// Like `take_rhs`, but also discards the node if is not unique.
    pub(super) fn take_rhs_unique_or_discard(
        &mut self,
    ) -> Option<HandleUniqueOwned<B::Index, B::Slice, B::Param>>
    where
        B: borrow::BorrowAsMut,
    {
        self.take_rhs().and_then(|n| n.into_unique_or_discard())
    }
}

fn shallow_clone_with_parent<I, S, P>(
    this: Borrowed<borrow::Immut<'_, Node<I, S, P>>>,
    parent: Option<(Pointer<I, S, P>, Side)>,
) -> Node<I, S, P>
where
    P: RleTreeConfig<I, S> + SupportsUpdate<I, S>,
{
    Node {
        parent,
        strong_count: RefCount::one(),
        weak_count: RefCount::one(),
        borrow_state: BorrowState::new(),
        redirect: Redirect::empty(),
        subtree_size: P::copy_index(this.get::<f![Node::subtree_size]>()),
        value: match this.get::<f![Node::value]>() {
            None => None,
            Some(s) => Some(P::clone_slice(s)),
        },
        height: *this.get::<f![Node::height]>(),
        lhs: this.get::<f![Node::lhs]>().as_ref().map(|n| n.shallow_clone()),
        rhs: this.get::<f![Node::rhs]>().as_ref().map(|n| n.shallow_clone()),
    }
}

impl<I, S, P: RleTreeConfig<I, S>> HandleUniqueOwned<I, S, P> {
    /// Erases this handle's "uniqueness" and downgrades it to just "owned"
    pub(super) fn erase(self) -> HandleOwned<I, S, P> {
        NodeHandle { inner: self.inner.erase() }
    }
}

impl<I, S, P: RleTreeConfig<I, S>> HandleOwned<I, S, P> {
    /// If this handle is unique, cast it as uniquely owned.
    /// Otherwise, copy the node.
    pub(super) fn into_unique(self) -> HandleUniqueOwned<I, S, P>
    where
        P: SupportsUpdate<I, S>,
    {
        // SAFETY: `into_unique` requires that strong/weak count are equal to 1 after cloning,
        // which we satisfy here.
        let mut inner = unsafe { self.inner.into_unique(|b| shallow_clone_with_parent(b, None)) };
        // Parent pointers aren't guaranteed to be accurate for Owned nodes with COW.
        // Reset that here.
        if P::COW {
            *inner.borrow_mut().get_mut::<f![Node::parent]>() = None;
        }

        NodeHandle { inner }
    }

    /// If this handle is unique, cast it as uniquely owned.
    /// Otherwise, discard it.
    fn into_unique_or_discard(self) -> Option<HandleUniqueOwned<I, S, P>> {
        self.inner.into_unique_or_discard().map(|inner| NodeHandle { inner })
    }

    /// If this handle is unique, cast it as uniquely owned.
    /// Otherwise, return `Err(self)`.
    pub(super) fn try_into_unique(self) -> Result<HandleUniqueOwned<I, S, P>, Self> {
        match self.inner.try_into_unique() {
            Ok(inner) => Ok(NodeHandle { inner }),
            Err(inner) => Err(NodeHandle { inner }),
        }
    }

    /// Creates a "shallow" clone of this node
    ///
    /// # Panics
    ///
    /// Note that this method panics if `!P::COW`.
    pub(super) fn shallow_clone(&self) -> HandleOwned<I, S, P> {
        if !P::COW {
            panic!("internal error: `NodeHandle::shallow_clone` must only be called with `P::COW`");
        }

        NodeHandle { inner: self.inner.shallow_clone() }
    }
}

impl<I, S> HandleWeak<I, S, param::EnableRefs> {
    /// Returns a borrowed handle on the node
    fn access(&self) -> HandleWeakImmut<'_, I, S, param::EnableRefs> {
        NodeHandle { inner: self.inner.access() }
    }

    /// Follows this weak handle across redirections, returning the node at the end
    ///
    /// Any nodes traversed along the way will be updated in-place to point to the end as well.
    pub(super) fn walk(self) -> Self {
        let mut last = self.clone();
        loop {
            let b = last.access();
            match b.take_redirect() {
                // We got to the end of the redirection chain, nothing more to do.
                None => break,
                Some(w) => {
                    // Place a copy of the weak handle back, so we keep from disturbing the
                    // redirect chain for now
                    b.set_redirect(w.clone());
                    // .. and then continue to the next node:
                    drop(b);
                    last = w;
                }
            }
        }

        // Now that we know where we'll end up, traverse those nodes again, but this time update
        // their redirects in-place to point to the final node. That way, future accesses won't
        // have to do the same traversal.
        let mut head = self;
        if head.ptr() != last.ptr() {
            loop {
                let b = head.access();
                // Get the next node:
                let next =
                    b.take_redirect().expect("`redirect` should be `Some(_)` because head != last");

                // If `next` already matches the end of the chain, we're done.
                if next.ptr() == last.ptr() {
                    b.set_redirect(next);
                    break;
                }

                // ... otherwise, update this node to point to the end, and keep following:
                b.set_redirect(last.clone());
                drop(b);
                head = next;
            }
        }

        // Everything updated, return the final node
        last
    }

    /// Returns a new weak handle pointing to the same node
    pub(super) fn clone(&self) -> Self {
        let inner = self.inner.access().weak();
        NodeHandle { inner }
    }

    /// Returns the pointer for the root of this handle's tree, plus the absolute range represented
    /// by this node's value.
    pub(super) fn root(&self) -> (Pointer<I, S, param::EnableRefs>, Range<I>)
    where
        I: Index,
    {
        let mut node = self.access();
        let Range { start: init_start, end: init_end } = node.value_range();

        let mut value_start = init_start;

        while let Some((parent, side)) = node.parent() {
            // SAFETY: The existence of `node` guarantees that the parent pointer is valid, IFF it
            // exists, so it's safe to produce a reference to it for now. Then, the weak-immutable
            // borrow prevents future mutable access, so we can rely on it existing as we continue
            // to (re)borrow up the tree (specifically because we have *overlapping* borrows,
            // rather than releasing & reborrowing).
            let p = unsafe { <Borrowed<borrow::WeakImmut<_>>>::from_non_null(parent) };
            let p = NodeHandle { inner: p };

            // Update `value_start`
            match side {
                Side::Lhs => (),
                Side::Rhs => {
                    let rhs_start = p.value_range().end;
                    value_start = value_start.add_left(rhs_start);
                }
            }

            node = p;
        }

        // Got to the root, done traversing upwards
        let value_end = value_start.add_right(init_end.sub_left(init_start));
        (node.ptr(), value_start..value_end)
    }

    /// Creates an immutable handle to this node
    ///
    /// # Safety
    ///
    /// The root of the tree must not be mutably borrowed during the lifetime `'t`.
    pub(super) unsafe fn upgrade_ref<'t>(&self) -> HandleImmut<'t, I, S, param::EnableRefs> {
        // SAFETY: the existence of `self` guarantees that the pointer is valid for now; the caller
        // guarantees that it will remain valid.
        let inner = unsafe { <Borrowed<borrow::Immut<_>>>::from_non_null(self.ptr()) };
        NodeHandle { inner }
    }
}

impl<'t, I, S, P: RleTreeConfig<I, S>> HandleImmut<'t, I, S, P> {
    /// Copies the `NodeHandle`
    pub(super) fn reborrow(&self) -> Self {
        NodeHandle { inner: self.inner.reborrow() }
    }

    /// Returns a reference to the slice value for the node
    ///
    /// # Panics
    ///
    /// This method panics if the slice value is not currently present, e.g. due to a prior call to
    /// [`take_value`](Self::take_value) without a corresponding [`set_value`](Self::set_value).
    pub(super) fn value(&self) -> &'t S {
        match self.inner.reborrow().into_ref::<f![Node::value]>() {
            Some(v) => v,
            None => panic!(
                "internal error: `value` should not be None except during temporary operations"
            ),
        }
    }

    /// Returns an immutable handle to the left-hand child
    pub(super) fn into_lhs(self) -> Option<Self> {
        self.inner
            .into_ref::<f![Node::lhs]>()
            .as_ref()
            .map(|n| NodeHandle { inner: n.borrow() })
    }

    /// Returns an immutable handle to the right-hand child
    pub(super) fn into_rhs(self) -> Option<Self> {
        self.inner
            .into_ref::<f![Node::rhs]>()
            .as_ref()
            .map(|n| NodeHandle { inner: n.borrow() })
    }
}

impl<'t, I, S, P: RleTreeConfig<I, S>> HandleMut<'t, I, S, P> {
    pub(super) fn has_parent(&self) -> bool {
        self.inner.get::<f![Node::parent]>().is_some()
    }

    /// Creates a reference from a raw `NonNull`
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by a previous call to [`NodeHandle::ptr`], and the
    /// borrow must be valid for the lifetime of that pointer. It is the caller's responsibility to
    /// ensure that Rust's aliasing requirements are satisfied.
    pub(super) unsafe fn from_ptr(pointer: Pointer<I, S, P>) -> Self {
        // SAFETY: the pointer must be properly aligned, point to a value of `Node<I, S>`, and be
        // valid for the lifetime. That's all guaranteed by the caller.
        unsafe {
            NodeHandle {
                inner: <Borrowed<borrow::Mut<_>>>::from_non_null(pointer),
            }
        }
    }

    /// Produces a reference to this node's parent
    pub(super) fn into_parent(self) -> Option<(Self, Side)> {
        let parent = self.inner.get::<f![Node::parent]>();
        parent.map(|(p, side)| {
            // SAFETY: `p` is properly aligned (like all node pointers are), and the invariants of
            // the tree guarantee that `p` still points to a valid `Node` - either this is not COW
            // and it's always valid, or it is a COW tree and we guarantee during downward
            // traversal of mutable borrows that we update parent pointers.
            let inner = unsafe { <Borrowed<borrow::Mut<_>>>::from_non_null(p) };
            (NodeHandle { inner }, side)
        })
    }

    /// Returns a mutable handle to the left-hand child, or `self` if there is no such child
    ///
    /// The returned handle has the same lifetime as `self`.
    pub(super) fn into_lhs(self) -> Result<Self, Self>
    where
        P: SupportsUpdate<I, S>,
    {
        let parent = Some((self.ptr(), Side::Lhs));
        // SAFETY: `try_into_some_mut` requires that `shallow_clone_with_parent` sets strong & weak
        // ref counts to 1, which it does.
        let result = unsafe {
            self.inner
                .try_into_some_mut::<f![Node::lhs], _>(|h| shallow_clone_with_parent(h, parent))
        };
        match result {
            Ok(mut child) => {
                if P::COW {
                    // With COW enabled, we only guarantee parent pointers are correct for mutable
                    // or unique references, so we must correct it during traversal.
                    *child.get_mut::<f![Node::parent]>() = parent;
                }
                Ok(NodeHandle { inner: child })
            }
            Err(this) => Err(NodeHandle { inner: this }),
        }
    }

    // Specialized version of `into_lhs` just for the drop impl.
    fn into_lhs_unique_or_discard(self) -> Result<Self, Self> {
        let parent = Some((self.ptr(), Side::Lhs));

        match self.inner.try_into_some_mut_or_discard::<f![Node::lhs]>() {
            Ok(mut inner) => {
                // If COW is enabled, make sure we overwrite the parent pointer.
                if P::COW {
                    *inner.get_mut::<f![Node::parent]>() = parent;
                }
                Ok(NodeHandle { inner })
            }
            Err(inner) => Err(NodeHandle { inner }),
        }
    }

    /// Returns a mutable handle to the right-hand child, or `self` if there is no such child
    ///
    /// The returned handle has the same lifetime as `self`.
    pub(super) fn into_rhs(self) -> Result<Self, Self>
    where
        P: SupportsUpdate<I, S>,
    {
        let parent = Some((self.ptr(), Side::Rhs));
        // SAFETY: `try_into_some_mut` requires that `shallow_clone_with_parent` sets strong & weak
        // ref counts to 1, which it does.
        let result = unsafe {
            self.inner
                .try_into_some_mut::<f![Node::rhs], _>(|h| shallow_clone_with_parent(h, parent))
        };
        match result {
            Ok(mut child) => {
                if P::COW {
                    // With COW enabled, we only guarantee parent pointers are correct for mutable
                    // or unique references, so we must correct it during traversal.
                    *child.get_mut::<f![Node::parent]>() = parent;
                }
                Ok(NodeHandle { inner: child })
            }
            Err(this) => Err(NodeHandle { inner: this }),
        }
    }

    /// Sets the left-hand child to `lhs`
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
    pub(super) fn insert_lhs(&mut self, mut lhs: HandleOwned<I, S, P>) {
        let parent = (self.ptr(), Side::Lhs);

        let this_lhs = self.inner.get_mut::<f![Node::lhs]>();
        if this_lhs.is_some() {
            panic!("internal error: cannot `insert_lhs()` that is already present");
        }

        // Optimistically set `lhs`'s parent to match this node.
        match lhs.inner.try_as_unique() {
            Ok(n) => *n.borrow_mut().get_mut::<f![Node::parent]>() = Some(parent),
            Err(()) => assert!(P::COW, "internal error: !P::COW and not unique"),
        }

        *this_lhs = Some(lhs.inner);
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
    pub(super) fn insert_into_lhs(mut self, mut lhs: HandleUniqueOwned<I, S, P>) -> Self {
        let parent = (self.ptr(), Side::Lhs);

        // Use a mutable borrow to check this LHS at the start, so that if there's any issue with
        // reentrant borrowing, we hit that up-front, rather than after setting new LHS's parent.
        let this_lhs = self.inner.get_mut::<f![Node::lhs]>();
        if this_lhs.is_some() {
            panic!("internal error: cannot `insert_lhs()` that is already present");
        }

        // Set `lhs`'s parent to match this node.
        *lhs.inner.borrow_mut().get_mut::<f![Node::parent]>() = Some(parent);

        let lhs_inner = self.inner.insert_into_some_unique::<f![Node::lhs]>(lhs.inner);
        NodeHandle { inner: lhs_inner }
    }

    /// Sets the right-hand child to `rhs`
    ///
    /// **NOTE:** The subtree rooted at `self` may be left unbalanced until you call `fix()` on
    /// this node, or `fix_{lhs,rhs}()` on the parent.
    ///
    /// **NOTE:** This node's subtree size *will not be updated*; you must call
    /// `set_subtree_size()` on this node to update its size after adding the new child.
    ///
    /// # Panics
    ///
    /// This method panics if this node already has a right-hand child.
    pub(super) fn insert_rhs(&mut self, mut rhs: HandleOwned<I, S, P>) {
        let parent = (self.ptr(), Side::Rhs);
        let this_rhs = self.inner.get_mut::<f![Node::rhs]>();

        if this_rhs.is_some() {
            panic!("internal error: cannot `insert_rhs()` that is already present");
        }

        // Optimistically set `rhs`'s parent to match this node.
        match rhs.inner.try_as_unique() {
            Ok(n) => *n.borrow_mut().get_mut::<f![Node::parent]>() = Some(parent),
            Err(()) => assert!(P::COW, "internal error: !P::COW and not unique"),
        }

        *this_rhs = Some(rhs.inner);
    }

    /// Sets the right-hand child to `rhs`, returning a mutable handle to the new child
    ///
    /// **NOTE:** The subtree rooted at `self` may be left unbalanced until you call `fix()` on
    /// this node, or `fix_{lhs,rhs}()` on the parent.
    ///
    /// **NOTE:** This node's subtree size *will not be updated*; you must call
    /// `set_subtree_size()` on this node to update its size after adding the new child.
    ///
    /// # Panics
    ///
    /// This method panics if this node already has a right-hand child.
    pub(super) fn insert_into_rhs(mut self, mut rhs: HandleUniqueOwned<I, S, P>) -> Self {
        let parent = (self.ptr(), Side::Rhs);

        // Use a mutable borrow to check this RHS at the start, so that if there's any issue with
        // reentrant borrowing, we hit that up-front, rather than after setting new RHS's parent.
        let this_rhs = self.inner.get_mut::<f![Node::rhs]>();
        if this_rhs.is_some() {
            panic!("internal error: cannot `insert_rhs()` that is already present");
        }

        // Set `rhs`'s parent to match this node.
        *rhs.inner.borrow_mut().get_mut::<f![Node::parent]>() = Some(parent);

        let rhs_inner = self.inner.insert_into_some_unique::<f![Node::rhs]>(rhs.inner);
        NodeHandle { inner: rhs_inner }
    }
}

pub(super) struct StackHandleImmut<'a, I: 'a, S: 'a, P: RleTreeConfig<I, S>> {
    stack: <P as RleTreeConfig<I, S>>::BorrowStack<'a>,
}

#[cfg(test)]
impl<'a, I: 'a, S: 'a, P> Debug for StackHandleImmut<'a, I, S, P>
where
    P: RleTreeConfig<I, S>,
    <P as RleTreeConfig<I, S>>::BorrowStack<'a>: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.stack, f)
    }
}

impl<'a, I: 'a, S: 'a, P: RleTreeConfig<I, S>> StackHandleImmut<'a, I, S, P> {
    pub(super) fn new_root(handle: HandleImmut<'a, I, S, P>) -> Self {
        StackHandleImmut { stack: BorrowStack::from_root(handle) }
    }

    pub(super) fn reborrow(&self) -> HandleImmut<'a, I, S, P> {
        self.stack.current()
    }

    pub(super) fn into_child(mut self, handle: HandleImmut<'a, I, S, P>, side: Side) -> Self {
        self.stack.push_child(handle, side);
        self
    }

    pub(super) fn into_parent(mut self) -> Option<(Self, Side)> {
        let parent = self.stack.pop();
        parent.map(|(_, side)| (self, side))
    }
}

/// Abstraction for tree traversal with `Immut` handles
///
/// For non-COW trees, we can trust that parent pointers are accurate, so we just use the single
/// handlle. But for COW-enabled trees, parent pointers might not be accurate, so as we *descend*
/// the tree we must track the nodes with mismatched parent pointers and restore them during
/// traversal back up the tree.
///
/// # Safety
///
/// Implementors guarantee that
pub trait BorrowStack<'a> {
    type Index;
    type Slice;
    type Param: RleTreeConfig<Self::Index, Self::Slice>;

    fn from_root(root: HandleImmut<'a, Self::Index, Self::Slice, Self::Param>) -> Self;

    fn current(&self) -> HandleImmut<'a, Self::Index, Self::Slice, Self::Param>;

    fn push_child(
        &mut self,
        child: HandleImmut<'a, Self::Index, Self::Slice, Self::Param>,
        side: Side,
    );
    fn pop(&mut self) -> Option<(HandleImmut<'a, Self::Index, Self::Slice, Self::Param>, Side)>;
}

impl<'a, I, S, P: RleTreeConfig<I, S>> BorrowStack<'a> for HandleImmut<'a, I, S, P> {
    type Index = I;
    type Slice = S;
    type Param = P;

    fn from_root(root: Self) -> Self {
        assert!(!P::COW, "internal error: cannot use HandleImmut as BorrowStack with P::COW");
        let has_parent = root.inner.get::<f![Node::parent]>().is_some();
        assert!(!has_parent);
        root
    }

    fn current(&self) -> Self {
        self.reborrow()
    }

    fn push_child(&mut self, child: Self, _side: Side) {
        *self = child;
    }

    fn pop(&mut self) -> Option<(Self, Side)> {
        assert!(!P::COW, "internal error: cannot use HandleImmut as BorrowStack with P::COW");
        let parent = self.reborrow().inner.into_ref::<f![Node::parent]>();
        parent.map(|(p, side)| {
            // SAFETY: `p` is properly aligned (like all node pointers are), and because we
            // guaranteed above that !P::COW, the tree invariants ensure `p` still points to a
            // valid `Node`.
            let inner = unsafe { <Borrowed<borrow::Immut<_>>>::from_non_null(p) };
            let handle = NodeHandle { inner };
            *self = handle.reborrow();
            (handle, side)
        })
    }
}

#[cfg_attr(test, derive(Debug))]
pub struct ImmutStack<'a, I, S, P: RleTreeConfig<I, S>> {
    sections: Vec<ImmutStackSection<'a, I, S, P>>,
    current: HandleImmut<'a, I, S, P>,
}

#[cfg_attr(test, derive(Debug))]
struct ImmutStackSection<'a, I, S, P: RleTreeConfig<I, S>> {
    parent: Option<(Pointer<I, S, P>, Side)>,
    child: HandleImmut<'a, I, S, P>,
}

impl<'a, I, S, P: RleTreeConfig<I, S>> BorrowStack<'a> for ImmutStack<'a, I, S, P> {
    type Index = I;
    type Slice = S;
    type Param = P;

    fn from_root(root: HandleImmut<'a, I, S, P>) -> Self {
        let mut sections = Vec::new();

        let has_parent = root.inner.get::<f![Node::parent]>().is_some();
        if has_parent {
            sections.push(ImmutStackSection { parent: None, child: root.reborrow() });
        }
        ImmutStack { sections, current: root }
    }

    fn current(&self) -> HandleImmut<'a, I, S, P> {
        self.current.reborrow()
    }

    fn push_child(&mut self, child: HandleImmut<'a, I, S, P>, side: Side) {
        let child_parent = child.inner.get::<f![Node::parent]>();
        match child_parent {
            // Child's parent pointer matches this one. We can assume
            &Some((ptr, s)) if ptr == self.current.ptr() && s == side => (),
            _ => {
                self.sections.push(ImmutStackSection {
                    parent: Some((self.current.ptr(), side)),
                    child: child.reborrow(),
                });
            }
        }

        self.current = child;
    }

    fn pop(&mut self) -> Option<(HandleImmut<'a, I, S, P>, Side)> {
        let parent = match self.sections.last() {
            Some(s) if s.child.ptr() == self.current.ptr() => {
                let parent = s.parent;
                _ = self.sections.pop();
                parent
            }
            _ => *self.current.inner.get::<f![Node::parent]>(),
        };

        parent.map(|(p, side)| {
            // SAFETY: `p` is properly aligned (like all node pointers are), and our behavior in
            // `push_child` guarantees that either `current`'s parent matches what we observed on
            // the downward traversal (in which case, we directly used its parent above), or it
            // didn't and the most recent entry in `self.sections` had the correct parent (in which
            // case, we used that parent instead).
            let inner = unsafe { <Borrowed<borrow::Immut<_>>>::from_non_null(p) };
            let handle = NodeHandle { inner };
            self.current = handle.reborrow();
            (handle, side)
        })
    }
}
