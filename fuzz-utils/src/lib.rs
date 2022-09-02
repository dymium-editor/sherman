use arbitrary::{Arbitrary, Unstructured};
use sherman::mock::Mock;
use sherman::param::{self, RleTreeConfig};
use sherman::range::{EndBound, StartBound};
use sherman::RleTree;
use std::fmt::{self, Debug, Display, Formatter};
use std::ops::Range;
use std::panic::{self, UnwindSafe};

fn expect_might_panic<R, F: UnwindSafe + FnOnce() -> R>(f: F) -> Result<R, ()> {
    // set a custom hook that does nothing, so we don't print panic information every time the mock
    // implementation panics
    panic::set_hook(Box::new(|_| {}));

    let result = panic::catch_unwind(f).map_err(|_| ());

    // remove our custom hook
    let _ = panic::take_hook();

    result
}

const BASIC_VARIANTS: u8 = 2;

/// A basic command, applicable to all [`RleTree`] parameterizations
#[derive(Clone)]
pub enum BasicCommand<I, S> {
    Iter {
        id: TreeId,
        start: StartBound<I>,
        end: EndBound<I>,
        /// Access pattern for elements in the iterator, only if it shouldn't panic
        access: Result<Vec<(IterDirection, Option<IterEntry<I, S>>)>, ()>,
    },
    Insert {
        id: TreeId,
        index: I,
        slice: S,
        size: I,
        panics: bool,
    },
}

/// Owned iterator item
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IterEntry<I, S> {
    range: Range<I>,
    size: I,
    slice: S,
}

impl<I, S: Clone> IterEntry<I, S> {
    fn from_triple((range, size, slice): (Range<I>, I, &S)) -> Self {
        IterEntry {
            range,
            size,
            slice: slice.clone(),
        }
    }
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
                let start_fmt = match start {
                    StartBound::Unbounded => String::new(),
                    StartBound::Included(i) => format!("{i:?}"),
                };
                let end_fmt = match end {
                    EndBound::Unbounded => String::new(),
                    EndBound::Included(i) => format!("={i:?}"),
                    EndBound::Excluded(i) => format!("{i:?}"),
                };

                let call = format!("tree_{id}.iter({start_fmt}..{end_fmt})");

                if let Ok(access) = access {
                    f.write_str("    {\n")?;

                    let maybe_mut = match access.is_empty() {
                        true => "_", // add an underscore to mark `iter` as unused
                        false => "mut ",
                    };

                    writeln!(f, "        let {maybe_mut}iter = {call};")?;
                    for (a, entry) in access {
                        let method = match a {
                            IterDirection::Forward => "next",
                            IterDirection::Backward => "next_back",
                        };

                        match entry {
                            None => writeln!(f, "        assert!(iter.{method}().is_none());")?,
                            Some(IterEntry { range, size, slice }) => {
                                f.write_str("        {\n")?;
                                writeln!(f, "            let item = iter.{method}().unwrap();")?;
                                writeln!(f, "            assert_eq!(item.range(), {range:?});")?;
                                writeln!(f, "            assert_eq!(item.size(), {size:?});")?;
                                writeln!(f, "            assert_eq!(item.slice(), &{slice:?});")?;
                                f.write_str("        }\n")?;
                            }
                        }
                    }

                    f.write_str("    }\n")
                } else {
                    f.write_str("    assert!(std::panic::catch_unwind(move || {\n")?;
                    writeln!(f, "        let _ = {call};")?;
                    f.write_str("    }).is_err());\n")
                }
            }
            Self::Insert { id, index, slice, size, panics: false } => {
                writeln!(f, "    tree_{id}.insert({index:?}, {slice:?}, {size:?});")
            },
            Self::Insert { id, index, slice, size, panics: true } => {
                f.write_str("    assert!(std::panic::catch_unwind(move || {\n")?;
                writeln!(f, "        tree_{id}.insert({index:?}, {slice:?}, {size:?})")?;
                f.write_str("    }).is_err());\n")
            }
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
                access: access.map(|vals| {
                    vals
                        .into_iter()
                        .map(|(dir, entry)| (dir, entry.map(|e| e.map_index(&mut f))))
                        .collect()
                }),
            },
            Self::Insert { id, index, slice, size, panics } => BasicCommand::Insert {
                id,
                index: f(index),
                slice,
                size: f(size),
                panics,
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
                access: access.map(|vals| {
                    vals
                        .into_iter()
                        .map(|(dir, entry)| (dir, entry.map(|e| e.map_slice(&mut f))))
                        .collect()
                }),
            },
            Self::Insert { id, index, slice, size, panics } => BasicCommand::Insert {
                id,
                index,
                slice: f(slice),
                size,
                panics,
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

impl<I, S> IterEntry<I, S> {
    fn map_index<J, F: FnMut(I) -> J>(self, mut f: F) -> IterEntry<J, S> {
        IterEntry {
            range: f(self.range.start)..f(self.range.end),
            size: f(self.size),
            slice: self.slice,
        }
    }

    fn map_slice<T, F: FnMut(S) -> T>(self, mut f: F) -> IterEntry<I, T> {
        IterEntry {
            range: self.range,
            size: self.size,
            slice: f(self.slice),
        }
    }
}

impl<'d, C: ArbitraryCommand<'d>> Arbitrary<'d> for CommandSequence<C>
where
    C::Index: sherman::Index,
    C::Slice: sherman::Slice<C::Index>,
{
    fn arbitrary(u: &mut Unstructured<'d>) -> arbitrary::Result<Self> {
        let mut cmds: Vec<C> = Vec::new();

        let mut trees = vec![Mock::new_empty()];

        while !u.is_empty() && !cmds.last().map(|c| c.should_panic()).unwrap_or(false) {
            let id = TreeId(u.int_in_range(0..=trees.len() - 1)?);
            let variant = u.int_in_range(0..=C::VARIANTS - 1)?;
            cmds.push(C::arbitrary(u, variant, id, &mut trees)?);
        }

        Ok(CommandSequence { cmds })
    }
}

pub trait ArbitraryCommand<'d>: Sized {
    const VARIANTS: u8;

    type Index;
    type Slice;

    /// Returns true if the expected result of this command is a panic
    fn should_panic(&self) -> bool;

    /// Creates a new command and executes it on the provided mock trees
    fn arbitrary(
        u: &mut Unstructured<'d>,
        variant: u8,
        id: TreeId,
        mocks: &mut Vec<Mock<Self::Index, Self::Slice>>,
    ) -> arbitrary::Result<Self>;
}

impl<'d, I, S> ArbitraryCommand<'d> for BasicCommand<I, S>
where
    I: Arbitrary<'d> + UnwindSafe + sherman::Index,
    S: Arbitrary<'d> + UnwindSafe + sherman::Slice<I> + Clone,
{
    const VARIANTS: u8 = BASIC_VARIANTS;

    type Index = I;
    type Slice = S;

    fn should_panic(&self) -> bool {
        match self {
            Self::Iter { access, .. } => access.is_err(),
            Self::Insert { panics, .. } => *panics,
        }
    }

    fn arbitrary(
        u: &mut Unstructured<'d>,
        variant: u8,
        id: TreeId,
        mocks: &mut Vec<Mock<I, S>>,
    ) -> arbitrary::Result<Self> {
        match variant {
            // iter
            0 => {
                let mock = mocks.remove(id.0);

                let start: StartBound<I> = u.arbitrary()?;
                let end: EndBound<I> = u.arbitrary()?;
                let access_directions: Vec<IterDirection> = u.arbitrary()?;

                let access = expect_might_panic(move || {
                    let mut iter = mock.iter((start, end));
                    let mut vals = Vec::new();
                    for direction in access_directions {
                        let entry = match direction {
                            IterDirection::Forward => iter.next(),
                            IterDirection::Backward => iter.next_back(),
                        };

                        vals.push((direction, entry.map(IterEntry::from_triple)));
                    }

                    (mock, vals)
                });

                Ok(Self::Iter {
                    id,
                    start,
                    end,
                    access: match access {
                        Ok((mock, vals)) => {
                            mocks.insert(id.0, mock);
                            Ok(vals)
                        }
                        Err(()) => Err(()),
                    },
                })
            }
            // insert
            1 => {
                let mut mock = mocks.remove(id.0);

                let index = u.arbitrary()?;
                let slice: S = u.arbitrary()?;
                let size = u.arbitrary()?;

                let slice_cloned = slice.clone();
                let result = expect_might_panic(move || {
                    mock.insert(index, slice_cloned, size);
                    mock
                });

                Ok(Self::Insert {
                    id,
                    index,
                    slice,
                    size,
                    panics: match result {
                        Ok(mock) => {
                            mocks.insert(id.0, mock);
                            false
                        }
                        Err(()) => true,
                    },
                })
            }
            _ => unreachable!("bad BasicCommand variant {variant}"),
        }
    }
}

impl<'d, I, S> ArbitraryCommand<'d> for SliceRefCommand<I, S>
where
    I: Arbitrary<'d> + UnwindSafe + sherman::Index,
    S: Arbitrary<'d> + UnwindSafe + sherman::Slice<I> + Clone,
{
    const VARIANTS: u8 = BASIC_VARIANTS + SLICE_REF_VARIANTS;

    type Index = I;
    type Slice = S;

    fn should_panic(&self) -> bool {
        match self {
            Self::Basic(c) => c.should_panic(),
        }
    }

    fn arbitrary(
        u: &mut Unstructured<'d>,
        variant: u8,
        id: TreeId,
        mocks: &mut Vec<Mock<I, S>>,
    ) -> arbitrary::Result<Self> {
        match variant {
            v if v < BASIC_VARIANTS => BasicCommand::arbitrary(u, v, id, mocks).map(Self::Basic),
            _ => unreachable!("bad SliceRefCommand variant {variant}"),
        }
    }
}

impl<'d, I, S> ArbitraryCommand<'d> for CowCommand<I, S>
where
    I: Arbitrary<'d> + UnwindSafe + sherman::Index,
    S: Arbitrary<'d> + UnwindSafe + sherman::Slice<I> + Clone,
{
    const VARIANTS: u8 = BASIC_VARIANTS + COW_VARIANTS;

    type Index = I;
    type Slice = S;

    fn should_panic(&self) -> bool {
        match self {
            Self::Basic(c) => c.should_panic(),
        }
    }

    fn arbitrary(
        u: &mut Unstructured<'d>,
        variant: u8,
        id: TreeId,
        mocks: &mut Vec<Mock<I, S>>,
    ) -> arbitrary::Result<Self> {
        match variant {
            v if v < BASIC_VARIANTS => BasicCommand::arbitrary(u, v, id, mocks).map(Self::Basic),
            _ => unreachable!("bad CowCommand variant {variant}"),
        }
    }
}

/// Ongoing state for executing commands to an [`RleTree`] and mock implementation
pub struct RunnerState<I, S, P: RleTreeConfig<I, S>, const M: usize> {
    trees: Vec<RleTree<I, S, P, M>>,
}

impl<I, S, P, const M: usize> RunnerState<I, S, P, M>
where
    I: UnwindSafe + sherman::Index,
    S: UnwindSafe + sherman::Slice<I> + Debug + Clone + PartialEq,
    P: RleTreeConfig<I, S> + param::SupportsInsert<I, S>,
{
    /// Creates a new, blank `RunnerState` to run a series of commands
    pub fn init() -> Self {
        RunnerState {
            trees: vec![RleTree::new_empty()],
        }
    }

    /// Runs the command
    pub fn run_basic_cmd(&mut self, cmd: &BasicCommand<I, S>) {
        match cmd {
            BasicCommand::Iter {
                id,
                start,
                end,
                access,
            } => {
                let tree = self.trees.remove(id.0);
                if let Ok(access) = access {
                    let mut iter = tree.iter((*start, *end));
                    for (dir, entry) in access {
                        let item = match dir {
                            IterDirection::Forward => iter.next(),
                            IterDirection::Backward => iter.next_back(),
                        };

                        let item_entry = item
                            .map(|e| (e.range(), e.size(), e.slice()))
                            .map(IterEntry::from_triple);
                        assert_eq!(entry, &item_entry);
                    }

                    self.trees.insert(id.0, tree);
                } else {
                    let (start, end) = (*start, *end);
                    let panicked = expect_might_panic(move || {
                        let _ = tree.iter((start, end));
                    })
                    .is_err();

                    assert!(panicked);
                }
            }
            BasicCommand::Insert {
                id,
                index,
                slice,
                size,
                panics,
            } => {
                let mut tree = self.trees.remove(id.0);

                if !panics {
                    tree.insert(*index, slice.clone(), *size);
                    self.trees.insert(id.0, tree);
                } else {
                    let (index, slice, size) = (*index, slice.clone(), *size);
                    let panicked = expect_might_panic(move || {
                        tree.insert(index, slice, size);
                    })
                    .is_err();

                    assert!(panicked);
                }
            }
        }
    }
}

impl<I, S, const M: usize> RunnerState<I, S, param::AllowSliceRefs, M>
where
    I: UnwindSafe + sherman::Index,
    S: UnwindSafe + sherman::Slice<I> + Debug + Clone + PartialEq,
{
    pub fn run_slice_ref_cmd(&mut self, cmd: &SliceRefCommand<I, S>) {
        match cmd {
            SliceRefCommand::Basic(c) => self.run_basic_cmd(c),
        }
    }
}

impl<I, S, const M: usize> RunnerState<I, S, param::AllowCow, M>
where
    I: UnwindSafe + sherman::Index,
    S: UnwindSafe + sherman::Slice<I> + Debug + Clone + PartialEq,
{
    pub fn run_cow_cmd(&mut self, cmd: &CowCommand<I, S>) {
        match cmd {
            CowCommand::Basic(c) => self.run_basic_cmd(c),
        }
    }
}
