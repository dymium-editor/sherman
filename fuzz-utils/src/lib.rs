use arbitrary::{Arbitrary, Unstructured};
use sherman::param::{self, RleTreeConfig};
use sherman::range::{EndBound, StartBound};
use sherman::{Index, RleTree, Slice};
use std::fmt::{self, Debug, Display, Formatter};
use std::mem;
use std::ops::Range;

const BASIC_VARIANTS: u8 = 2;

/// A basic command, applicable to all [`RleTree`] parameterizations
#[derive(Clone)]
pub enum BasicCommand<I, S> {
    Iter {
        id: TreeId,
        start: StartBound<I>,
        end: EndBound<I>,
        /// Access pattern for elements in the iterator
        access: Vec<IterDirection>,
    },
    Insert {
        id: TreeId,
        index: I,
        slice: S,
        size: I,
    },
}

#[derive(Debug, Copy, Clone, Arbitrary)]
pub struct TreeId(usize);

impl Display for TreeId {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Copy, Clone, Arbitrary)]
pub enum IterDirection {
    Forward,
    Backward,
}

const SLICE_REF_VARIANTS: u8 = 0;

/// Command applicable only to [`RleTree`]s parameterized with [`param::AllowSliceRefs`]
#[derive(Clone)]
pub enum SliceRefCommand<I, S> {
    Basic(BasicCommand<I, S>),
}

const COW_VARIANTS: u8 = 0;

/// Command applicable only to [`RleTree`]s parameterized with [`param::AllowCow`]
#[derive(Clone)]
pub enum CowCommand<I, S> {
    Basic(BasicCommand<I, S>),
}

/// Sequence of [`BasicCommand`]s, [`SliceRefCommand`]s, or [`CowCommand`]s
pub struct CommandSequence<C> {
    pub cmds: Vec<C>,
}

impl<C: Debug> Debug for CommandSequence<C> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let init_id = TreeId(0);
        f.write_str("#[test]\n")?;
        f.write_str("fn test_case() {\n")?;
        writeln!(f, "    let mut tree_{init_id} = RleTree::new_empty();")?;
        for c in &self.cmds {
            c.fmt(f)?;
        }
        f.write_str("}")
    }
}

impl<I: Debug, S: Debug> Debug for BasicCommand<I, S> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Iter { id, start, end, access } => {
                f.write_str("    {\n")?;
                let start_fmt = match start {
                    StartBound::Unbounded => String::new(),
                    StartBound::Included(i) => format!("{i:?}"),
                };
                let end_fmt = match end {
                    EndBound::Unbounded => String::new(),
                    EndBound::Included(i) => format!("={i:?}"),
                    EndBound::Excluded(i) => format!("{i:?}"),
                };
                writeln!(f, "        let mut iter = tree_{id}.iter({start_fmt}..{end_fmt});")?;
                for a in access {
                    let method = match a {
                        IterDirection::Forward => "next",
                        IterDirection::Backward => "next_back",
                    };

                    writeln!(f, "        let _ = iter.{method}();")?;
                }

                f.write_str("    }\n")
            }
            Self::Insert { id, index, slice, size } => {
                writeln!(f, "    tree_{id}.insert({index:?}, {slice:?}, {size:?});")
            },
        }
    }
}

impl<I: Debug, S: Debug> Debug for SliceRefCommand<I, S> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Basic(c) => c.fmt(f),
        }
    }
}

impl<I: Debug, S: Debug> Debug for CowCommand<I, S> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Basic(c) => c.fmt(f),
        }
    }
}

impl<C> CommandSequence<C> {
    pub fn map<D, F: FnMut(C) -> D>(self, f: F) -> CommandSequence<D> {
        CommandSequence {
            cmds: self.cmds.into_iter().map(f).collect(),
        }
    }
}

impl<I, S> BasicCommand<I, S> {
    #[rustfmt::skip]
    pub fn map_index<J, F: FnMut(I) -> J>(self, mut f: F) -> BasicCommand<J, S> {
        match self {
            Self::Iter { id, start, end, access } => BasicCommand::Iter {
                id,
                start: match start {
                    StartBound::Included(i) => StartBound::Included(f(i)),
                    StartBound::Unbounded => StartBound::Unbounded,
                },
                end: match end {
                    EndBound::Included(i) => EndBound::Included(f(i)),
                    EndBound::Excluded(i) => EndBound::Excluded(f(i)),
                    EndBound::Unbounded => EndBound::Unbounded,
                },
                access,
            },
            Self::Insert { id, index, slice, size } => BasicCommand::Insert {
                id,
                index: f(index),
                slice,
                size: f(size),
            }
        }
    }

    #[rustfmt::skip]
    pub fn map_slice<T, F: FnMut(S) -> T>(self, mut f: F) -> BasicCommand<I, T> {
        match self {
            Self::Iter { id, start, end, access } => BasicCommand::Iter {
                id,
                start,
                end,
                access,
            },
            Self::Insert { id, index, slice, size } => BasicCommand::Insert {
                id,
                index,
                slice: f(slice),
                size,
            }
        }
    }
}

impl<I, S> SliceRefCommand<I, S> {
    #[rustfmt::skip]
    pub fn map_index<J, F: FnMut(I) -> J>(self, f: F) -> SliceRefCommand<J, S> {
        match self {
            Self::Basic(c) => SliceRefCommand::Basic(c.map_index(f)),
        }
    }

    #[rustfmt::skip]
    pub fn map_slice<T, F: FnMut(S) -> T>(self, f: F) -> SliceRefCommand<I, T> {
        match self {
            Self::Basic(c) => SliceRefCommand::Basic(c.map_slice(f)),
        }
    }
}

impl<I, S> CowCommand<I, S> {
    #[rustfmt::skip]
    pub fn map_index<J, F: FnMut(I) -> J>(self, f: F) -> CowCommand<J, S> {
        match self {
            Self::Basic(c) => CowCommand::Basic(c.map_index(f)),
        }
    }

    #[rustfmt::skip]
    pub fn map_slice<T, F: FnMut(S) -> T>(self, f: F) -> CowCommand<I, T> {
        match self {
            Self::Basic(c) => CowCommand::Basic(c.map_slice(f)),
        }
    }
}

impl<'d, C: ArbitraryCommand<'d>> Arbitrary<'d> for CommandSequence<C> {
    fn arbitrary(u: &mut Unstructured<'d>) -> arbitrary::Result<Self> {
        let num_trees = 1;
        let mut cmds = Vec::new();

        while !u.is_empty() {
            let id = TreeId(u.int_in_range(0..=num_trees - 1)?);
            let variant = u.int_in_range(0..=C::VARIANTS - 1)?;
            cmds.push(C::arbitrary(u, variant, id)?);
        }

        Ok(CommandSequence { cmds })
    }
}

pub trait ArbitraryCommand<'d>: Sized {
    const VARIANTS: u8;

    fn arbitrary(u: &mut Unstructured<'d>, variant: u8, id: TreeId) -> arbitrary::Result<Self>;
}

#[derive(Arbitrary)]
enum FuzzStartBound<I> {
    Included(I),
    Unbounded,
}

#[derive(Arbitrary)]
enum FuzzEndBound<I> {
    Included(I),
    Excluded(I),
    Unbounded,
}

impl<'d, I: Arbitrary<'d>, S: Arbitrary<'d>> ArbitraryCommand<'d> for BasicCommand<I, S> {
    const VARIANTS: u8 = BASIC_VARIANTS;

    fn arbitrary(u: &mut Unstructured<'d>, variant: u8, id: TreeId) -> arbitrary::Result<Self> {
        match variant {
            // iter
            0 => Ok(Self::Iter {
                id,
                start: match u.arbitrary()? {
                    FuzzStartBound::Included(i) => StartBound::Included(i),
                    FuzzStartBound::Unbounded => StartBound::Unbounded,
                },
                end: match u.arbitrary()? {
                    FuzzEndBound::Included(i) => EndBound::Included(i),
                    FuzzEndBound::Excluded(i) => EndBound::Excluded(i),
                    FuzzEndBound::Unbounded => EndBound::Unbounded,
                },
                access: u.arbitrary()?,
            }),
            // insert
            1 => Ok(Self::Insert {
                id,
                index: u.arbitrary()?,
                slice: u.arbitrary()?,
                size: u.arbitrary()?,
            }),
            _ => unreachable!("bad BasicCommand variant {variant}"),
        }
    }
}

impl<'d, I: Arbitrary<'d>, S: Arbitrary<'d>> ArbitraryCommand<'d> for SliceRefCommand<I, S> {
    const VARIANTS: u8 = BASIC_VARIANTS + SLICE_REF_VARIANTS;

    fn arbitrary(u: &mut Unstructured<'d>, variant: u8, id: TreeId) -> arbitrary::Result<Self> {
        match variant {
            v if v < BASIC_VARIANTS => BasicCommand::arbitrary(u, v, id).map(Self::Basic),
            _ => unreachable!("bad SliceRefCommand variant {variant}"),
        }
    }
}

impl<'d, I: Arbitrary<'d>, S: Arbitrary<'d>> ArbitraryCommand<'d> for CowCommand<I, S> {
    const VARIANTS: u8 = BASIC_VARIANTS + COW_VARIANTS;

    fn arbitrary(u: &mut Unstructured<'d>, variant: u8, id: TreeId) -> arbitrary::Result<Self> {
        match variant {
            v if v < BASIC_VARIANTS => BasicCommand::arbitrary(u, v, id).map(Self::Basic),
            _ => unreachable!("bad CowCommand variant {variant}"),
        }
    }
}

/// Ongoing state for executing commands to an [`RleTree`] and mock implementation
pub struct RunnerState<I, S, P: RleTreeConfig<I, S>, const M: usize> {
    trees: Vec<(Mock<I, S>, RleTree<I, S, P, M>)>,
}

impl<I, S, P, const M: usize> RunnerState<I, S, P, M>
where
    I: Index,
    S: Debug + Clone + PartialEq + Slice<I>,
    P: RleTreeConfig<I, S> + param::SupportsInsert<I, S>,
{
    /// Creates a new, blank `RunnerState` to run a series of commands
    pub fn init() -> Self {
        RunnerState {
            trees: vec![(Mock::new(), RleTree::new_empty())],
        }
    }

    /// Runs the command, returning an error if no more commands can be run
    pub fn run_basic_cmd(
        &mut self,
        cmd: &BasicCommand<I, S>,
        too_big: impl Fn(I, I) -> bool,
    ) -> Result<(), ()> {
        match cmd {
            BasicCommand::Iter {
                id,
                start,
                end,
                access,
            } => {
                let (mock, tree) = &mut self.trees[id.0];
                let mut mock_iter = mock.try_iter(*start, *end)?;
                let mut tree_iter = tree.iter((*start, *end));
                for a in access {
                    match a {
                        IterDirection::Forward => {
                            let mock_item = mock_iter.next();
                            let tree_item =
                                tree_iter.next().map(|s| (s.range(), s.size(), s.slice()));

                            assert_eq!(mock_item, tree_item);
                        }
                        IterDirection::Backward => {
                            let mock_item = mock_iter.next_back();
                            let tree_item = tree_iter
                                .next_back()
                                .map(|s| (s.range(), s.size(), s.slice()));

                            assert_eq!(mock_item, tree_item);
                        }
                    }
                }
                Ok(())
            }
            BasicCommand::Insert {
                id,
                index,
                slice,
                size,
            } => {
                let (mock, tree) = &mut self.trees[id.0];
                if too_big(mock.size(), *size) {
                    return Err(());
                }

                mock.try_insert(*index, slice.clone(), *size)?;
                tree.insert(*index, slice.clone(), *size);

                // After inserting, check that the contents match what we're expecting.
                let mock_items = mock
                    .try_iter(StartBound::Unbounded, EndBound::Unbounded)
                    .unwrap()
                    .collect::<Vec<_>>();
                let tree_items = tree
                    .iter(..)
                    .map(|s| (s.range(), s.size(), s.slice()))
                    .collect::<Vec<_>>();

                assert_eq!(mock_items, tree_items);

                Ok(())
            }
        }
    }
}

impl<I, S, const M: usize> RunnerState<I, S, param::AllowSliceRefs, M>
where
    I: Index,
    S: Debug + Clone + PartialEq + Slice<I>,
{
    pub fn run_slice_ref_cmd(
        &mut self,
        cmd: &SliceRefCommand<I, S>,
        too_big: impl Fn(I, I) -> bool,
    ) -> Result<(), ()> {
        match cmd {
            SliceRefCommand::Basic(c) => self.run_basic_cmd(c, too_big),
        }
    }
}

impl<I, S, const M: usize> RunnerState<I, S, param::AllowCow, M>
where
    I: Index,
    S: Debug + Clone + PartialEq + Slice<I>,
{
    pub fn run_cow_cmd(
        &mut self,
        cmd: &CowCommand<I, S>,
        too_big: impl Fn(I, I) -> bool,
    ) -> Result<(), ()> {
        match cmd {
            CowCommand::Basic(c) => self.run_basic_cmd(c, too_big),
        }
    }
}

/// A mock, inefficient implementation of the [`RleTree`] interface
#[derive(Debug, Clone)]
pub struct Mock<I, S> {
    // list of end positions and the slices in the run
    runs: Vec<(I, S)>,
}

impl<I: Index, S: Slice<I>> Mock<I, S> {
    fn new() -> Self {
        Mock { runs: Vec::new() }
    }

    fn size(&self) -> I {
        self.runs.last().map(|(i, _)| *i).unwrap_or(I::ZERO)
    }

    fn try_insert(&mut self, index: I, mut slice: S, size: I) -> Result<(), ()> {
        if index > self.size() || size == I::ZERO {
            return Err(());
        }

        // find the insertion point
        let mut idx = match self.runs.binary_search_by_key(&index, |(i, _)| *i) {
            Err(i) => i,
            Ok(i) => i + 1,
        };

        let key_start = if idx < self.runs.len() {
            Some(
                idx.checked_sub(1)
                    .map(|i| self.runs[i].0)
                    .unwrap_or(I::ZERO),
            )
        } else {
            None
        };

        // If the index is greater than the start of the key it's in, then we need to split
        // that key
        if let Some(s) = key_start && s < index {
            // let (key_end, mut lhs) = self.runs.remove(idx);
            let pos_in_key = index.sub_left(s);
            let rhs = self.runs[idx].1.split_at(pos_in_key);
            let rhs_end = mem::replace(&mut self.runs[idx].0, pos_in_key);
            self.runs[idx].0 = pos_in_key;
            self.runs.insert(idx + 1, (rhs_end, rhs));
        }

        let mut base_pos = index;
        let mut old_size = I::ZERO;
        let mut new_size = size;

        // insert at the the point between this key and the one before

        if let Some(p) = idx.checked_sub(1) {
            let (lhs_end, lhs) = self.runs.remove(p);
            assert_eq!(lhs_end, index);
            match lhs.try_join(slice) {
                Err((lhs, s)) => {
                    self.runs.insert(p, (lhs_end, lhs));
                    slice = s;
                }
                Ok(new) => {
                    let lhs_start = p.checked_sub(1).map(|i| self.runs[i].0).unwrap_or(I::ZERO);
                    let lhs_size = lhs_end.sub_left(lhs_start);
                    old_size = old_size.add_left(lhs_size);
                    new_size = new_size.add_left(lhs_size);
                    base_pos = lhs_start;
                    slice = new;
                    idx = p;
                }
            }
        }

        // `idx` is already the right-hand node, because `index` is equal to the end of `lhs`
        if idx < self.runs.len() {
            let (rhs_end, rhs) = self.runs.remove(idx);
            match rhs.try_join(slice) {
                Err((s, rhs)) => {
                    self.runs.insert(idx + 1, (rhs_end, rhs));
                    slice = s;
                }
                Ok(new) => {
                    let rhs_start = idx
                        .checked_sub(1)
                        .map(|i| self.runs[i].0)
                        .unwrap_or(I::ZERO);
                    let rhs_size = rhs_end.sub_left(rhs_start);
                    old_size = old_size.add_right(rhs_size);
                    new_size = new_size.add_right(rhs_size);
                    slice = new;
                }
            }
        }

        self.runs.insert(idx, (base_pos.add_right(new_size), slice));

        let diff = base_pos
            .add_right(new_size)
            .sub_left(base_pos.add_right(old_size));

        for (i, _) in self.runs.get_mut(idx + 1..).unwrap_or(&mut []) {
            *i = i.add_left(diff);
        }

        Ok(())
    }

    fn try_iter(&self, start: StartBound<I>, end: EndBound<I>) -> Result<MockIter<'_, I, S>, ()> {
        let start_pos;

        let fwd_idx = match start {
            StartBound::Unbounded => {
                start_pos = I::ZERO;
                0
            }
            StartBound::Included(i) => {
                start_pos = i;
                if i > self.size() {
                    return Err(());
                }

                match self.runs.binary_search_by_key(&i, |(i, _)| *i) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                }
            }
        };

        let bkwd_idx = match end {
            EndBound::Included(i) if i < start_pos || i >= self.size() => return Err(()),
            EndBound::Excluded(i) if i <= start_pos || i > self.size() => return Err(()),

            EndBound::Unbounded => self.runs.len(),
            EndBound::Included(i) => match self.runs.binary_search_by_key(&i, |(i, _)| *i) {
                Ok(i) => i + 1,
                Err(i) => i,
            },
            EndBound::Excluded(i) => match self.runs.binary_search_by_key(&i, |(i, _)| *i) {
                Ok(i) => i,
                Err(i) => i,
            },
        };

        Ok(MockIter {
            runs: &self.runs,
            fwd_idx,
            bkwd_idx,
        })
    }
}

pub struct MockIter<'t, I, S> {
    runs: &'t [(I, S)],
    fwd_idx: usize,
    bkwd_idx: usize,
}

impl<'t, I: Index, S> Iterator for MockIter<'t, I, S> {
    type Item = (Range<I>, I, &'t S);

    fn next(&mut self) -> Option<Self::Item> {
        if self.fwd_idx >= self.bkwd_idx {
            return None;
        }

        let start = self
            .fwd_idx
            .checked_sub(1)
            .map(|i| self.runs[i].0)
            .unwrap_or(I::ZERO);

        let &(end, ref slice) = &self.runs[self.fwd_idx];
        self.fwd_idx += 1;

        Some((start..end, end.sub_left(start), slice))
    }
}

impl<'t, I: Index, S> DoubleEndedIterator for MockIter<'t, I, S> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.fwd_idx >= self.bkwd_idx {
            return None;
        }

        self.bkwd_idx -= 1;
        let start = self
            .bkwd_idx
            .checked_sub(1)
            .map(|i| self.runs[i].0)
            .unwrap_or(I::ZERO);

        let &(end, ref slice) = &self.runs[self.bkwd_idx];

        Some((start..end, end.sub_left(start), slice))
    }
}
