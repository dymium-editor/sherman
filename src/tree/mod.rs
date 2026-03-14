#![allow(clippy::type_complexity)]

use std::fmt::{self, Debug};
use std::ops::{ControlFlow, Range};

use crate::{Index, Slice};

#[macro_use]
mod borrow;

mod entry;
mod fix;
mod node;

pub(crate) mod tests;

use entry::SliceEntry;
use node::Side;

// @commit-fail redo these docs
/// Run-length encoded, balanced binary search tree
///
/// Although this type is quite complex, it arises out of the simplest version of the problems it
/// solves. That is:
///
/// > Given a large `Vec<T>` with `r` runs of identical values, how do we represent the data with
/// > size `O(r)`, supporting `O(log(r))` insert and delete
///
/// The general idea turns out to be pretty ok: a binary search tree with (a) keys as the offset of
/// the run's start index from the parent node's and (b) values as the length and content of each
/// run. Insertion and deletion -- which require updating the position of *every* value at a
/// greater index -- still only require `O(log(r))` updates because all the positions are relative,
/// except for the root.
///
/// Unfortunately, in order to squeeze more functionality out of this type, we gradually added more
/// and more features to it.
///
/// ## Ok, so why is `RleTree` so complicated?
///
/// Well, here's the thing. This crate was specifically made for use in a text editor, where every
/// time you find *one* use case for a run-length encoded tree, it turns out there's another one
/// just around the corner. We started with "*tag each byte in a file with the edit that last
/// touched it*" and moved to "*we can represent the file content itself*", to eventually "*hey, with
/// special index types, this can apply to line/column number pairs!*".
///
/// So the simplest version of this type would just represent a mapping `usize -> T`, where `T`
/// implements `PartialEq + Clone` and comparisons are handled by the tree. But to provide more
/// flexibility, we instead have a mapping `I -> S`, where `I` implements [`Index`] (but is nearly
/// always `usize`) and `S` implements [`Slice`], which provides utilities for joining and splitting
/// runs (instead of with `PartialEq` and `Clone`).
pub struct RleTree<I, S> {
    root: Option<Root<I, S>>,
}

/// The (owned) root node of an `RleTree`
pub(super) struct Root<I, S> {
    handle: node::HandleUniqueOwned<I, S>,
}

#[cfg(not(feature = "nightly"))]
impl<I, S> Drop for RleTree<I, S> {
    fn drop(&mut self) {}
}

#[cfg(feature = "nightly")]
unsafe impl<#[may_dangle] I, #[may_dangle] S> Drop for RleTree<I, S> {
    fn drop(&mut self) {}
}

impl<I: Debug, S: Debug> Debug for Root<I, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = f.debug_struct("Root");
        s.field("node", &self.handle);
        s.finish()
    }
}

impl<I, S> RleTree<I, S>
where
    I: Index,
    S: Slice<I>,
{
    /// Creates a new, empty `RleTree`.
    pub const fn new_empty() -> Self {
        RleTree { root: None }
    }

    /// Creates an `RleTree` initialized to contain only the initial slice of the given size
    ///
    /// ## Panics
    ///
    /// This method will panic if `size` is not greater than zero -- i.e., if `size <= I::ZERO`.
    pub fn new(slice: S, size: I) -> Self {
        if size <= I::ZERO {
            panic!("cannot create new slice with non-positive size {size:?}");
        }

        let root = Root { handle: node::NodeHandle::alloc_new(slice, size) };

        RleTree { root: Some(root) }
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn root(&self) -> Option<&Root<I, S>> {
        self.root.as_ref()
    }

    /// Asserts that the tree is balanced
    #[cfg(any(test, feature = "fuzz"))]
    pub(crate) fn validate_balance(&self) {
        if let Some(root) = self.root.as_ref() {
            fix::validate_balance(root.handle.borrow());
        }
    }

    /// Returns the total size of the tree -- i.e., the sum of the sizes of all the slices
    pub fn size(&self) -> I {
        match &self.root {
            Some(r) => r.handle.subtree_size(),
            None => I::ZERO,
        }
    }

    /// Returns an object with information about the slice containing the index
    ///
    /// Through the returned [`SliceEntry`], both the slice `S` and the range of values covered
    /// `Range<I>` can be retrieved.
    ///
    /// ## Panics
    ///
    /// This method will panic if `idx` is out of bounds -- i.e., if it is less than `I::ZERO` or
    /// greater than `self.size()`.
    pub fn get(&self, idx: I) -> SliceEntry<'_, I, S> {
        if idx < I::ZERO {
            panic!("index {idx:?} out of bounds, less than zero");
        } else if idx >= self.size() {
            panic!("index {idx:?} out of bounds for size {:?}", self.size());
        }

        let Some(root) = self.root.as_ref() else {
            crate::panic_internal_error_or_bad_index::<I>(
                "`self.root` should be `Some` if `0 <= idx < size`",
            );
        };

        let mut node = root.handle.borrow();
        let mut target = idx;

        let (range, offset_in_range) = loop {
            match search_step(node.borrow(), target) {
                SearchResult::Lhs { offset } => {
                    target = offset;
                    node = node
                        .into_lhs()
                        .expect("`SearchResult::Lhs` implies the left-hand child should exist");
                }
                SearchResult::RhsEdge => {
                    target = I::ZERO;
                    node = node.into_rhs().expect(
                        "`SearchResult::RhsEdge` implies the right-hand child should exist",
                    );
                }
                SearchResult::Rhs { offset } => {
                    target = offset;
                    node = node
                        .into_rhs()
                        .expect("`SearchResult::Rhs` implies the right-hand child should exist");
                }
                SearchResult::LhsEdge => break (node.value_range(), I::ZERO),
                SearchResult::Value { range, offset_in_range } => break (range, offset_in_range),
            }
        };

        // Found the value!
        // To reconstruct the *absolute* positions of the slice, we can subtract
        // offset from idx to get the absolute position of range.start (and therefore
        // range.end as well.
        let abs_start = idx.sub_right(offset_in_range);
        let abs_end = abs_start.add_right(range.end.sub_left(range.start));
        SliceEntry { range: abs_start..abs_end, slice: node }
    }

    /// Inserts the slice at position `idx`, shifting all later entries by `size`
    ///
    /// If there is any entry that contains `idx`, it will be split and encompass `slice` on either
    /// side after the insertion (unless `slice` joins with either/both sides).
    ///
    /// ## Panics
    ///
    /// This method will panic if `idx` is *greater* than [`self.size()`]. An index equal to the
    /// current size of the tree is explicitly allowed. It will also panic if the size of the new
    /// slice is not greater than zero -- i.e. if `size <= I::ZERO`.
    pub fn insert(&mut self, idx: I, slice: S, size: I) {
        if idx < I::ZERO {
            panic!("index {idx:?} out of bounds, less than zero");
        } else if idx > self.size() {
            panic!("index {idx:?} out of bounds for size {:?}", self.size());
        } else if size <= I::ZERO {
            panic!("cannot insert new slice with non-positive size {size:?}");
        }

        let mut root = match self.root.take() {
            Some(r) => r,
            None => {
                // This tree is completely empty, so we can actually just initialize it to just
                // contain the value we want, and return. Given that `self.size()` must be zero, we
                // know that `idx` is also zero.
                *self = RleTree::new(slice, size);
                return;
            }
        };

        run_insert(
            root.handle.borrow_mut(),
            None,
            false,
            DownwardInsertState::new(idx, slice, size),
        );
        root.handle = fix::fix_owned(root.handle);
        self.root = Some(root);
    }
}

#[derive(Debug)]
enum SearchResult<I> {
    Lhs { offset: I },
    LhsEdge,
    Value { range: Range<I>, offset_in_range: I },
    RhsEdge,
    Rhs { offset: I },
}

fn search_step<I: Index, S>(node: node::HandleImmut<I, S>, target: I) -> SearchResult<I> {
    let value_range = node.value_range();

    if target < value_range.start {
        SearchResult::Lhs { offset: target }
    } else if target > value_range.end {
        SearchResult::Rhs { offset: target.sub_left(value_range.end) }
    } else if target == value_range.start {
        SearchResult::LhsEdge
    } else if target == value_range.end {
        SearchResult::RhsEdge
    } else {
        SearchResult::Value {
            offset_in_range: target.sub_left(value_range.start),
            range: value_range,
        }
    }
}

struct DownwardInsertState<I, S> {
    target: I,
    fst_value: InsertionValue<I, S>,
    snd_value: Option<InsertionValue<I, S>>,
    allow_joining: bool,
    already_split_once: bool,
}

impl<I: Debug, S: Debug> Debug for DownwardInsertState<I, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = f.debug_struct("DownwardInsertState");
        s.field("target", &self.target);
        s.field("fst_value", &format_args!("{:?}", self.fst_value));
        s.field("snd_value", &format_args!("{:?}", self.snd_value));
        s.field("allow_joining", &self.allow_joining);
        s.field("already_split_once", &self.already_split_once);
        s.finish()
    }
}

#[derive(Debug)]
struct InsertionValue<I, S> {
    slice: S,
    size: I,
}

#[derive(Debug)]
struct UpwardInsertState<I> {
    old_size: I,
}

/// Traverses down the tree, inserts the value, and traverses back up the tree to adjust the
/// subtree sizes of each node.
///
/// Upward traversal will terminate at `root`, and `root`'s subtree size will be updated to match.
fn run_insert<'t, I: Index, S: Slice<I>>(
    root: node::HandleMut<'t, I, S>,
    root_subtree_size: Option<I>,
    mut force_edge_rhs: bool,
    state: DownwardInsertState<I, S>,
) -> node::HandleMut<'t, I, S> {
    let root_addr = root.addr();

    let mut down_state = state;
    let mut node = root;

    let mut up_state = loop {
        let (n, cf) = if force_edge_rhs {
            force_edge_rhs = false;
            down_state.step_edge_rhs(node)
        } else {
            down_state.step(node)
        };
        node = n;
        match cf {
            ControlFlow::Continue(s) => down_state = s,
            ControlFlow::Break(up_state) => break up_state,
        }
    };

    while node.addr() != root_addr {
        let parent_addr = node.parent_addr().expect("parent addr should be Some(_)");

        node = fix::fix_mut(node);

        let override_size = match parent_addr == root_addr {
            true => root_subtree_size,
            false => None,
        };
        (node, up_state) = up_state.step(node, override_size);
    }

    if node.parent_addr().is_some() {
        node = fix::fix_mut(node);
    }

    node
}

impl<I: Index, S: Slice<I>> DownwardInsertState<I, S> {
    fn new(target: I, slice: S, size: I) -> Self {
        DownwardInsertState {
            target,
            fst_value: InsertionValue { slice, size },
            snd_value: None,
            allow_joining: true,
            already_split_once: false,
        }
    }

    fn step(
        self,
        node: node::HandleMut<I, S>,
    ) -> (node::HandleMut<I, S>, ControlFlow<UpwardInsertState<I>, Self>) {
        match search_step(node.borrow(), self.target) {
            SearchResult::Lhs { offset } => match node.into_lhs() {
                // Recurse into LHS child:
                Ok(child) => (child, ControlFlow::Continue(Self { target: offset, ..self })),
                // No LHS, insert a new one.
                Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                    "search step returned LHS without LHS child",
                ),
            },
            SearchResult::Rhs { offset } => match node.into_rhs() {
                // Recurse into RHS child:
                Ok(child) => (child, ControlFlow::Continue(Self { target: offset, ..self })),
                // No RHS, even though our search told us we were outside the range of this value.
                Err(_) => crate::panic_internal_error_or_bad_index::<I>(
                    "search step returned RHS without RHS child",
                ),
            },
            SearchResult::LhsEdge => self.step_edge_lhs(node),
            SearchResult::RhsEdge => self.step_edge_rhs(node),
            SearchResult::Value { range, offset_in_range } => {
                self.step_split_value(node, range, offset_in_range)
            }
        }
    }

    /// Sub-case of `step` that handles the case where the target index is at the edge of the node
    /// and its LHS child (if there is one).
    fn step_edge_lhs(
        mut self,
        mut node: node::HandleMut<I, S>,
    ) -> (node::HandleMut<I, S>, ControlFlow<UpwardInsertState<I>, Self>) {
        // Target is on the edge of this node and LHS subtree; try joining the slice with this
        // node.
        if self.allow_joining {
            debug_assert!(self.snd_value.is_none());

            match self.fst_value.slice.try_join(node.take_value()) {
                Err((slice, this_value)) => {
                    // Couldn't join; put the node's value back.
                    node.set_value(this_value);
                    self.fst_value.slice = slice;
                }
                Ok(mut join_value) => {
                    // Successfully joined with this node. We potentially could still join with a
                    // node to the left, if there is one. Any left-hand node higher up the tree
                    // would have already been attempted and failed, so we just need to traverse
                    // down the tree.

                    let old_subtree_size = node.subtree_size();
                    let new_subtree_size;
                    {
                        let range = node.value_range();
                        let lhs_size = range.start;
                        let value_size = range.end.sub_left(range.start);
                        let rhs_size = old_subtree_size.sub_left(range.end);

                        new_subtree_size = lhs_size
                            .add_right(self.fst_value.size)
                            .add_right(value_size)
                            .add_right(rhs_size);
                    }

                    join_value = Self::try_join_traverse_lhs(node.borrow_mut(), join_value);

                    node.set_value(join_value);
                    node.set_subtree_size(new_subtree_size);
                    return (
                        node,
                        ControlFlow::Break(UpwardInsertState { old_size: old_subtree_size }),
                    );
                }
            }
        }

        // Couldn't join; recurse into the child or insert a new one.
        match node.into_lhs() {
            // Recurse into LHS child:
            Ok(child) => (child, ControlFlow::Continue(Self { target: self.target, ..self })),
            // No LHS, insert a new one:
            Err(n) => {
                let mut new_lhs =
                    node::NodeHandle::alloc_new(self.fst_value.slice, self.fst_value.size);

                if let Some(snd_value) = self.snd_value {
                    new_lhs
                        .borrow_mut()
                        .insert_rhs(node::NodeHandle::alloc_new(snd_value.slice, snd_value.size));
                    new_lhs.set_subtree_size(self.fst_value.size.add_right(snd_value.size));
                    new_lhs = fix::fix_owned(new_lhs);
                }

                let child = n.insert_lhs(new_lhs);

                (child, ControlFlow::Break(UpwardInsertState { old_size: I::ZERO }))
            }
        }
    }

    /// Sub-case of `step` that handles the case where the target index is at the edge of the node
    /// and its RHS child (if there is one).
    fn step_edge_rhs(
        mut self,
        mut node: node::HandleMut<I, S>,
    ) -> (node::HandleMut<I, S>, ControlFlow<UpwardInsertState<I>, Self>) {
        // Target is on the edge of this node and RHS subtree; try joining the slice
        // with this node.
        if self.allow_joining {
            debug_assert!(self.snd_value.is_none());

            match node.take_value().try_join(self.fst_value.slice) {
                Err((this_value, slice)) => {
                    // Couldn't join; put the node's value back.
                    node.set_value(this_value);
                    self.fst_value.slice = slice;
                }
                Ok(mut join_value) => {
                    // Successfully joined with this node. If we haven't already tried joining with
                    // the node immediately to the right, we need to traverse the tree to try_join
                    // again.

                    let old_subtree_size = node.subtree_size();
                    let new_subtree_size;
                    {
                        let range = node.value_range();
                        let lhs_size = range.start;
                        let value_size = range.end.sub_left(range.start);
                        let rhs_size = old_subtree_size.sub_left(range.end);

                        new_subtree_size = lhs_size
                            .add_right(value_size)
                            .add_right(self.fst_value.size)
                            .add_right(rhs_size);
                    }

                    join_value = Self::try_join_traverse_rhs(node.borrow_mut(), join_value);

                    node.set_value(join_value);
                    node.set_subtree_size(new_subtree_size);
                    return (
                        node,
                        ControlFlow::Break(UpwardInsertState { old_size: old_subtree_size }),
                    );
                }
            }
        }

        // Couldn't join; recurse into the child or insert a new one.
        match node.into_rhs() {
            // Recurse into RHS child:
            Ok(child) => (child, ControlFlow::Continue(Self { target: I::ZERO, ..self })),
            // No RHS, insert a new one:
            Err(n) => {
                let mut new_rhs =
                    node::NodeHandle::alloc_new(self.fst_value.slice, self.fst_value.size);

                if let Some(snd_value) = self.snd_value {
                    new_rhs
                        .borrow_mut()
                        .insert_rhs(node::NodeHandle::alloc_new(snd_value.slice, snd_value.size));
                    new_rhs.set_subtree_size(self.fst_value.size.add_right(snd_value.size));
                }

                let child = n.insert_rhs(new_rhs);

                (child, ControlFlow::Break(UpwardInsertState { old_size: I::ZERO }))
            }
        }
    }

    /// Tries to join `root_value` with the node immediately left of `subtree_root`, via its
    /// left-hand child (i.e., the RIGHT-MOST node in the subtree rooted at its left-hand child).
    ///
    /// Returns the result of attempting to join with `root_value`.
    fn try_join_traverse_lhs(subtree_root: node::HandleMut<I, S>, root_value: S) -> S {
        let root_addr = subtree_root.addr();

        let mut immediate_lhs = match subtree_root.into_lhs() {
            Err(_) => return root_value,
            Ok(child) => child,
        };

        loop {
            // To get the *immediate* left-hand node, we have to first get the left-hand child, and
            // then keep following right-hand children until we get to the end.
            match immediate_lhs.into_rhs() {
                // Has a right-hand child! Keep going down the tree.
                Ok(c) => immediate_lhs = c,
                // No more right-hand children; must be this node!
                // Try joining.
                Err(n) => {
                    immediate_lhs = n;
                    break;
                }
            }
        }

        let final_value; // <- if we successfully join.

        let lhs_value = immediate_lhs.take_value();
        match lhs_value.try_join(root_value) {
            // Couldn't join, put the values back:
            Err((lhs, root_value)) => {
                immediate_lhs.set_value(lhs);
                return root_value;
            }
            // Did join -- we have complex operations ahead. More below.
            Ok(v) => final_value = v,
        }

        // Joined! Let's remove this lower node, replacing it with its own left-hand child, if
        // there is one - because we already know it doesn't have a right-hand child.
        let mut lower_lhs = immediate_lhs.take_lhs();

        let lower_lhs_size = lower_lhs.as_ref().map(|lhs| lhs.subtree_size()).unwrap_or(I::ZERO);
        let removed_size = immediate_lhs.subtree_size().sub_left(lower_lhs_size);

        let mut replaced_empty_node = false;

        // Traverse back up the tree, until we get back to `subtree_root`.
        // On the first parent, we'll need to reinsert `lower_lhs`.
        let mut upward_child = immediate_lhs;
        loop {
            let (mut parent, side) = upward_child
                .into_parent()
                .expect("internal error: bad traversal: node must have a parent");

            match side {
                // On the left-hand side of the parent? That means we must *already* be at
                // `subtree_root`. Replace the LHS if we haven't already, and we're done.
                Side::Lhs => {
                    assert!(parent.addr() == root_addr);

                    if !replaced_empty_node {
                        // remove the node we took the value from.
                        drop(parent.take_lhs());
                        // reinsert the node's left-hand child, if it had one:
                        if let Some(lhs) = lower_lhs.take() {
                            parent.borrow_mut().insert_lhs(lhs);
                        }
                    }

                    return final_value;
                }
                // Right-hand side of the parent means there's more recursion we'll have to do.
                // Let's fix the immediate parent if we need to, and then continue upwards.
                Side::Rhs => {
                    if !replaced_empty_node {
                        // remove the node we took the value from.
                        drop(parent.take_rhs());
                        // reinsert the node's left-hand child, if it had one:
                        if let Some(lhs) = lower_lhs.take() {
                            parent.borrow_mut().insert_rhs(lhs);
                        }
                        replaced_empty_node = true;
                    }

                    let new_subtree_size = parent.subtree_size().sub_right(removed_size);
                    parent.set_subtree_size(new_subtree_size);

                    parent = fix::fix_mut(parent);

                    // recurse upwards
                    upward_child = parent;
                }
            }
        }
    }

    /// Tries to join `root_value` with the node immediately right of `subtree_root`, via its
    /// right-hand child (i.e., the LEFT-MOST node in the subtree rooted at its right-hand child).
    ///
    /// Returns the result of attempting to join with `root_value`.
    fn try_join_traverse_rhs(subtree_root: node::HandleMut<I, S>, root_value: S) -> S {
        let root_addr = subtree_root.addr();

        let mut immediate_rhs = match subtree_root.into_rhs() {
            Err(_) => return root_value,
            Ok(child) => child,
        };

        loop {
            // To get the *immediate* right-hand node, we have to first get the right-hand child, and
            // then keep following left-hand children until we get to the end.
            match immediate_rhs.into_lhs() {
                // Has a left-hand child! Keep going down the tree.
                Ok(c) => immediate_rhs = c,
                // No more left-hand children; must be this node!
                // Try joining.
                Err(n) => {
                    immediate_rhs = n;
                    break;
                }
            }
        }

        let final_value; // <- if we successfully join.

        let rhs_value = immediate_rhs.take_value();
        match root_value.try_join(rhs_value) {
            // Couldn't join, put the values back
            Err((root_value, rhs)) => {
                immediate_rhs.set_value(rhs);
                return root_value;
            }
            // Did join -- we have complex operations ahead. More below.
            Ok(v) => final_value = v,
        }

        // Joined! Let's remove this lower node, replacing it with its own left-hand child, if
        // there is one - because we already know it doesn't have a right-hand child.
        let mut lower_rhs = immediate_rhs.take_rhs();

        let lower_rhs_size = lower_rhs.as_ref().map(|rhs| rhs.subtree_size()).unwrap_or(I::ZERO);
        let removed_size = immediate_rhs.subtree_size().sub_right(lower_rhs_size);

        let mut replaced_empty_node = false;

        // Traverse back up the tree, until we get back to `subtree_root`.
        // On the first parent, we'll need to reinsert `lower_lhs`.
        let mut upward_child = immediate_rhs;
        loop {
            let (mut parent, side) = upward_child
                .into_parent()
                .expect("internal error: bad traversal: node must have a parent");

            match side {
                // On the right-hand side of the parent? That means we must *already* be at
                // `subtree_root`. Replace the RHS if we haven't already, and we're done.
                Side::Rhs => {
                    assert!(parent.addr() == root_addr);

                    if !replaced_empty_node {
                        // remove the node we took the value from.
                        drop(parent.take_rhs());
                        // reinsert the node's left-hand child, if it had one:
                        if let Some(rhs) = lower_rhs.take() {
                            parent.borrow_mut().insert_rhs(rhs);
                        }
                    }

                    return final_value;
                }
                // Left-hand side of the parent means there's more recursion we'll have to do.
                // Let's fix the immediate parent if we need to, then continue upwards.
                Side::Lhs => {
                    if !replaced_empty_node {
                        // Remove the node we took the value from.
                        drop(parent.take_lhs());
                        // reinsert the node's right-hand child, if it had one:
                        if let Some(rhs) = lower_rhs.take() {
                            parent.borrow_mut().insert_lhs(rhs);
                        }
                        replaced_empty_node = true;
                    }

                    let new_subtree_size = parent.subtree_size().sub_left(removed_size);
                    parent.set_subtree_size(new_subtree_size);

                    parent = fix::fix_mut(parent);

                    // recurse upwards
                    upward_child = parent;
                }
            }
        }
    }

    fn step_split_value(
        self,
        mut node: node::HandleMut<I, S>,
        range: Range<I>,
        offset_in_range: I,
    ) -> (node::HandleMut<I, S>, ControlFlow<UpwardInsertState<I>, Self>) {
        // At this point: the value is in the middle of the range, so we need to split this
        // node.
        // We'll end up with something like:
        //
        //            ┏━━━━━━━━━━━┓
        //            ┃ this node ┃
        //            ┗━━━━━━━━━━━┛
        //                  ⇓
        //   ┏━━━━━━━━━━┱───────┲━━━━━━━━━━┓
        //   ┃ this lhs ┃ slice ┃ this rhs ┃
        //   ┗━━━━━━━━━━┹───────┺━━━━━━━━━━┛
        //
        // and `slice` can join with either (or both!) of the parts of the original node.
        // If `slice` joins with only one, we'll have to insert one value.
        // If `slice` joins with neither, we'll have to insert two values (!!)

        if self.already_split_once {
            // When we've already perfored a single split, the 1-2 new values will be inserted
            // EXACTLY at the right-hand edge of where the preexisting value was, so we should
            // never find that we need to split another value.
            //
            // So if we get to here (needing to perform a split) finding that we ALREADY split
            // once, then something has gone wrong.
            crate::panic_internal_error_or_bad_index::<I>("would double-split on insert");
        }

        let original_size = node.subtree_size();
        let node_lhs_size = range.start;
        let node_rhs_size = original_size.sub_left(range.end);

        let (split_lhs, split_rhs) = node.take_value().split_at(offset_in_range);
        let split_lhs_size = offset_in_range;
        let split_rhs_size = range.end.sub_left(range.start).sub_left(offset_in_range);

        let replacement: S;
        let replacement_size: I;
        let to_insert: Option<(InsertionValue<I, S>, Option<InsertionValue<I, S>>)>;

        // Try joining `slice` to lhs:
        match split_lhs.try_join(self.fst_value.slice) {
            Ok(new_value) => {
                // Joined with LHS. Try to re-join with RHS.
                match new_value.try_join(split_rhs) {
                    Ok(final_value) => {
                        // Successfully joined all three pieces. Nothing left to do.
                        replacement = final_value;
                        replacement_size =
                            split_lhs_size.add_right(self.fst_value.size).add_right(split_rhs_size);
                        to_insert = None;
                    }
                    Err((lhs, rhs)) => {
                        // Joined LHS+slice but not RHS. We'll have to re-insert it.
                        replacement = lhs;
                        replacement_size = split_lhs_size.add_right(self.fst_value.size);
                        to_insert =
                            Some((InsertionValue { slice: rhs, size: split_rhs_size }, None));
                    }
                }
            }
            Err((lhs, slice)) => {
                // Couldn't join with LHS. Try joining with RHS.
                replacement = lhs;
                replacement_size = split_lhs_size;

                match slice.try_join(split_rhs) {
                    Ok(new_value) => {
                        // Joined slice+RHS but not with LHS. We'll have to re-insert
                        // slice+RHS.
                        to_insert = Some((
                            InsertionValue {
                                slice: new_value,
                                size: self.fst_value.size.add_right(split_rhs_size),
                            },
                            None,
                        ));
                    }
                    Err((slice, rhs)) => {
                        to_insert = Some((
                            InsertionValue { slice, size: self.fst_value.size },
                            Some(InsertionValue { slice: rhs, size: split_rhs_size }),
                        ));
                    }
                }
            }
        }

        // Put everything back, maybe continue inserting.
        node.set_value(replacement);

        let old_subtree_size = original_size;
        let insertion_size = match &to_insert {
            None => I::ZERO,
            Some((fst_value, None)) => fst_value.size,
            Some((fst_value, Some(snd_value))) => fst_value.size.add_right(snd_value.size),
        };

        let new_subtree_size = node_lhs_size
            .add_right(replacement_size)
            .add_right(insertion_size)
            .add_right(node_rhs_size);
        node.set_subtree_size(new_subtree_size);

        if let Some((fst_value, snd_value)) = to_insert {
            // Insert the new value(s) - recursion will be bounded by the check at the top of THIS
            // function, which checks that we don't perform a second split.
            // `run_insert` will stop at this node.
            node = run_insert(
                node,
                Some(new_subtree_size),
                true,
                DownwardInsertState {
                    target: new_subtree_size.sub_right(node_rhs_size),
                    fst_value,
                    snd_value,
                    allow_joining: false, // already checked joining above.
                    already_split_once: true,
                },
            );
        }

        (node, ControlFlow::Break(UpwardInsertState { old_size: old_subtree_size }))
    }
}

impl<I: Index> UpwardInsertState<I> {
    fn step<S: Slice<I>>(
        self,
        node: node::HandleMut<I, S>,
        override_parent_subtree_size: Option<I>,
    ) -> (node::HandleMut<I, S>, Self) {
        let lower_old_subtree_size = self.old_size;
        let lower_new_subtree_size = node.subtree_size();
        let lower_addr = node.addr();

        let Some((mut parent, side)) = node.into_parent() else {
            panic!("internal error: tried to `UpwardInsertState::step` a node with no parent");
        };

        match side {
            Side::Lhs => {
                assert_eq!(parent.borrow().into_lhs().map(|n| n.addr()), Some(lower_addr));

                let old_parent_size = parent.subtree_size();
                let new_parent_size = override_parent_subtree_size.unwrap_or_else(|| {
                    old_parent_size
                        .sub_left(lower_old_subtree_size)
                        .add_left(lower_new_subtree_size)
                });
                parent.set_subtree_size(new_parent_size);

                (parent, UpwardInsertState { old_size: old_parent_size })
            }
            Side::Rhs => {
                assert_eq!(parent.borrow().into_rhs().map(|n| n.addr()), Some(lower_addr));

                let old_parent_size = parent.subtree_size();
                let new_parent_size = override_parent_subtree_size.unwrap_or_else(|| {
                    old_parent_size
                        .sub_right(lower_old_subtree_size)
                        .add_right(lower_new_subtree_size)
                });
                parent.set_subtree_size(new_parent_size);

                (parent, UpwardInsertState { old_size: old_parent_size })
            }
        }
    }
}
