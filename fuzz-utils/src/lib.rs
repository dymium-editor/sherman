use arbitrary::{Arbitrary, Unstructured};
use sherman::mock::{self, Mock};
use sherman::param::{self, RleTreeConfig};
use sherman::range::{EndBound, StartBound};
use sherman::{RleTree, SliceRef};
use std::fmt::{self, Debug, Display, Formatter};
use std::ops::Range;
use std::panic::{self, RefUnwindSafe, UnwindSafe};
use std::sync::Mutex;

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
        /// Access pattern for elements in the iterator, only if the initial call shouldn't panic
        access: Result<Vec<(IterDirection, Option<IterEntry<I, S>>)>, ()>,
    },
    Get {
        id: TreeId,
        index: I,
        // if the call doesn't panic, the range and value of the slice
        info: Result<(Range<I>, S), ()>,
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
        Display::fmt(&self.0, f)
    }
}

#[derive(Debug, Copy, Clone, Arbitrary)]
pub enum IterDirection {
    Forward,
    Backward,
}

const SLICE_REF_VARIANTS: u8 = 7;

/// Command applicable only to [`RleTree`]s parameterized with [`param::AllowSliceRefs`]
#[derive(Clone)]
pub enum SliceRefCommand<I, S> {
    Basic(BasicCommand<I, S>),
    MakeRef {
        id: TreeId,
        index: I,
        // if accessing at the index shoudn't panic, the `RefId`
        ref_id: Result<RefId, ()>,
    },
    InsertRef {
        id: TreeId,
        index: I,
        slice: S,
        size: I,
        // if the insertion shouldn't panic, the returned `RefId`
        ref_id: Result<RefId, ()>,
    },
    CloneRef {
        src_id: RefId,
        new_id: RefId,
    },
    CheckRefValid {
        id: TreeId,
        ref_id: RefId,
        valid: bool,
    },
    CheckRefRange {
        id: TreeId,
        ref_id: RefId,
        // if the call shouldn't panic, the range
        range: Result<Range<I>, ()>,
    },
    CheckRefSlice {
        id: TreeId,
        ref_id: RefId,
        // if the call shouldn't panic, the slice
        slice: Result<S, ()>,
    },
    DropRef {
        ref_id: RefId,
    },
}

#[derive(Debug, Copy, Clone)]
pub struct RefId(usize);

impl Display for RefId {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

const COW_VARIANTS: u8 = 2;

/// Command applicable only to [`RleTree`]s parameterized with [`param::AllowCow`]
#[derive(Clone)]
pub enum CowCommand<I, S> {
    Basic(BasicCommand<I, S>),
    ShallowClone { src_id: TreeId, new_id: TreeId },
    DropTree { id: TreeId },
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
                    f.write_str("    assert!(std::panic::catch_unwind(|| {\n")?;
                    writeln!(f, "        let _ = {call};")?;
                    f.write_str("    }).is_err());\n")
                }
            }
            Self::Get { id, index, info: Ok((range, slice)) } => {
                f.write_str("    {\n")?;
                writeln!(f, "        let entry = tree_{id}.get({index:?});")?;
                writeln!(f, "        assert_eq!(entry.range(), {range:?});")?;
                writeln!(f, "        assert_eq!(entry.slice(), &{slice:?});")?;
                f.write_str("    }\n")
            }
            Self::Get { id, index, info: Err(()) } => {
                f.write_str("    assert!(std::panic::catch_unwind(|| {\n")?;
                writeln!(f, "        tree_{id}.get({index:?});")?;
                f.write_str("    }).is_err());\n")
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
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Basic(c) => c.fmt(f),
            Self::MakeRef { id, index, ref_id: Ok(ref_id) } => {
                writeln!(f, "    let ref_{ref_id} = tree_{id}.get({index:?}).make_ref();")
            },
            Self::MakeRef { id, index, ref_id: Err(()) } => {
                f.write_str("    assert!(std::panic::catch_unwind(|| {\n")?;
                writeln!(f, "        tree_{id}.get({index:?}).make_ref()\n")?;
                f.write_str("    }).is_err());\n")
            },
            Self::InsertRef { id, index, slice, size, ref_id: Ok(ref_id) } => {
                writeln!(
                    f,
                    "    let ref_{ref_id} = tree_{id}.insert_ref({index:?}, {slice:?}, {size:?});"
                )
            },
            Self::InsertRef { id, index, slice, size, ref_id: Err(()) } => {
                f.write_str("    assert!(std::panic::catch_unwind(move || {\n")?;
                writeln!(f, "        tree_{id}.insert_ref({index:?}, {slice:?}, {size:?})")?;
                f.write_str("    }).is_err());\n")
            },
            Self::CloneRef { src_id, new_id } => {
                writeln!(f, "    let ref_{new_id} = ref_{src_id}.clone();")
            },
            Self::CheckRefValid { id: _, ref_id, valid: true } => {
                writeln!(f, "    assert!(ref_{ref_id}.is_valid());")
            },
            Self::CheckRefValid { id: _, ref_id, valid: false } => {
                writeln!(f, "    assert!(!ref_{ref_id}.is_valid());")
            },
            Self::CheckRefRange { id: _, ref_id, range: Ok(range) } => {
                writeln!(f, "    assert_eq!(ref_{ref_id}.range(), {range:?});")
            },
            Self::CheckRefRange { id: _, ref_id, range: Err(()) } => {
                f.write_str("    assert!(std::panic::catch_unwind(move || {\n")?;
                writeln!(f, "        ref_{ref_id}.range()")?;
                f.write_str("    }).is_err());\n")
            },
            Self::CheckRefSlice { id: _, ref_id, slice: Ok(slice) } => {
                writeln!(f, "    assert_eq!(&*ref_{ref_id}.borrow_slice(), &{slice:?});")
            },
            Self::CheckRefSlice { id: _, ref_id, slice: Err(()) } => {
                f.write_str("    assert!(std::panic::catch_unwind(move || {\n")?;
                writeln!(f, "        ref_{ref_id}.borrow_slice()")?;
                f.write_str("    }).is_err());\n")
            },
            Self::DropRef { ref_id } => writeln!(f, "    drop(ref_{ref_id})"),
        }
    }
}

impl<I: Debug, S: Debug> Debug for CowCommand<I, S> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Basic(c) => c.fmt(f),
            Self::ShallowClone { src_id, new_id } => f.write_fmt(format_args!(
                "    let mut tree_{new_id} = tree_{src_id}.shallow_clone();\n",
            )),
            Self::DropTree { id } => f.write_fmt(format_args!("    drop(tree_{id});\n")),
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
            Self::Get { id, index, info } => BasicCommand::Get {
                id,
                index: f(index),
                info: info.map(|(range, slice)| (f(range.start)..f(range.end), slice)),
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
            Self::Get { id, index, info } => BasicCommand::Get {
                id,
                index,
                info: info.map(|(range, slice)| (range, f(slice))),
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
    pub fn map_index<J, F: FnMut(I) -> J>(self, mut f: F) -> SliceRefCommand<J, S> {
        match self {
            Self::Basic(c) => SliceRefCommand::Basic(c.map_index(f)),
            Self::MakeRef { id, index, ref_id } => SliceRefCommand::MakeRef {
                id,
                index: f(index),
                ref_id,
            },
            Self::InsertRef { id, index, slice, size, ref_id } => SliceRefCommand::InsertRef {
                id,
                index: f(index),
                slice,
                size: f(size),
                ref_id,
            },
            Self::CloneRef { src_id, new_id } => SliceRefCommand::CloneRef { src_id, new_id },
            Self::CheckRefValid { id, ref_id, valid } => {
                SliceRefCommand::CheckRefValid { id, ref_id, valid }
            }
            Self::CheckRefRange { id, ref_id, range } => SliceRefCommand::CheckRefRange {
                id,
                ref_id,
                range: range.map(|r| f(r.start)..f(r.end)),
            },
            Self::CheckRefSlice { id, ref_id, slice } => {
                SliceRefCommand::CheckRefSlice { id, ref_id, slice }
            }
            Self::DropRef { ref_id } => SliceRefCommand::DropRef { ref_id },
        }
    }

    #[rustfmt::skip]
    pub fn map_slice<T, F: FnMut(S) -> T>(self, mut f: F) -> SliceRefCommand<I, T> {
        match self {
            Self::Basic(c) => SliceRefCommand::Basic(c.map_slice(f)),
            Self::MakeRef { id, index, ref_id } => {
                SliceRefCommand::MakeRef { id, index, ref_id }
            }
            Self::InsertRef { id, index, slice, size, ref_id } => SliceRefCommand::InsertRef {
                id,
                index,
                slice: f(slice),
                size,
                ref_id,
            },
            Self::CloneRef { src_id, new_id } => SliceRefCommand::CloneRef { src_id, new_id },
            Self::CheckRefValid { id, ref_id, valid } => {
                SliceRefCommand::CheckRefValid { id, ref_id, valid }
            }
            Self::CheckRefRange { id, ref_id, range } => {
                SliceRefCommand::CheckRefRange { id, ref_id, range }
            }
            Self::CheckRefSlice { id, ref_id, slice } => SliceRefCommand::CheckRefSlice {
                id,
                ref_id,
                slice: slice.map(|s| f(s)),
            },
            Self::DropRef { ref_id } => SliceRefCommand::DropRef { ref_id },
        }
    }
}

impl<I, S> CowCommand<I, S> {
    #[rustfmt::skip]
    pub fn map_index<J, F: FnMut(I) -> J>(self, f: F) -> CowCommand<J, S> {
        match self {
            Self::Basic(c) => CowCommand::Basic(c.map_index(f)),
            Self::ShallowClone { src_id, new_id } => CowCommand::ShallowClone { src_id, new_id },
            Self::DropTree { id } => CowCommand::DropTree { id },
        }
    }

    #[rustfmt::skip]
    pub fn map_slice<T, F: FnMut(S) -> T>(self, f: F) -> CowCommand<I, T> {
        match self {
            Self::Basic(c) => CowCommand::Basic(c.map_slice(f)),
            Self::ShallowClone { src_id, new_id } => CowCommand::ShallowClone { src_id, new_id },
            Self::DropTree { id } => CowCommand::DropTree { id },
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

        let mut trees = vec![Some(Mock::new_empty())];
        let mut refs: Vec<Option<mock::Ref>> = Vec::new();
        let mut num_trees = 1;
        let mut num_refs = 0;

        while !u.is_empty() && num_trees != 0 {
            let id = TreeId(choose_sparse_index(u, num_trees, &trees)?);
            let variant = u.int_in_range(0..=C::VARIANTS - 1)?;
            cmds.push(C::arbitrary(
                u,
                variant,
                id,
                &mut num_trees,
                &mut num_refs,
                &mut trees,
                &mut refs,
            )?);
        }

        Ok(CommandSequence { cmds })
    }
}

fn choose_sparse_index<T>(
    u: &mut Unstructured,
    count: usize,
    vals: &[Option<T>],
) -> arbitrary::Result<usize> {
    let mut idx = u.choose_index(count)?;
    let mut i = 0;
    while i <= idx {
        if vals[i].is_none() {
            idx += 1;
        }
        i += 1;
    }
    Ok(idx)
}

pub trait ArbitraryCommand<'d>: Sized {
    const VARIANTS: u8;

    type Index;
    type Slice;

    /// Creates a new command and executes it on the provided mock trees
    fn arbitrary(
        u: &mut Unstructured<'d>,
        variant: u8,
        id: TreeId,
        count: &mut usize,
        refcount: &mut usize,
        mocks: &mut Vec<Option<Mock<Self::Index, Self::Slice>>>,
        refs: &mut Vec<Option<mock::Ref>>,
    ) -> arbitrary::Result<Self>;
}

impl<'d, I, S> ArbitraryCommand<'d> for BasicCommand<I, S>
where
    I: Arbitrary<'d> + UnwindSafe + RefUnwindSafe + sherman::Index,
    S: Arbitrary<'d> + UnwindSafe + RefUnwindSafe + sherman::Slice<I> + Clone,
{
    const VARIANTS: u8 = BASIC_VARIANTS;

    type Index = I;
    type Slice = S;

    fn arbitrary(
        u: &mut Unstructured<'d>,
        variant: u8,
        id: TreeId,
        count: &mut usize,
        _rc: &mut usize,
        mocks: &mut Vec<Option<Mock<I, S>>>,
        _refs: &mut Vec<Option<mock::Ref>>,
    ) -> arbitrary::Result<Self> {
        match variant {
            // iter
            0 => {
                let mock = mocks[id.0].as_ref().unwrap();

                let start: StartBound<I> = u.arbitrary()?;
                let end: EndBound<I> = u.arbitrary()?;
                let access_directions: Vec<IterDirection> = u.arbitrary()?;

                let result = expect_might_panic(|| {
                    let mut iter = mock.iter((start, end));
                    let mut vals = Vec::new();
                    for direction in access_directions {
                        let entry = match direction {
                            IterDirection::Forward => iter.next(),
                            IterDirection::Backward => iter.next_back(),
                        };

                        vals.push((direction, entry.map(IterEntry::from_triple)));
                    }

                    vals
                });

                Ok(Self::Iter {
                    id,
                    start,
                    end,
                    access: result,
                })
            }
            // get
            1 => {
                let mock = mocks[id.0].as_ref().unwrap();

                let index = u.arbitrary()?;

                let result = expect_might_panic(|| {
                    let (range, slice) = mock.get(index);
                    let slice = slice.clone();
                    (range, slice)
                });

                Ok(Self::Get {
                    id,
                    index,
                    info: result,
                })
            }
            // insert
            2 => {
                let mut mock = mocks[id.0].take().unwrap();
                *count -= 1;

                let index = u.arbitrary()?;
                let slice: S = u.arbitrary()?;
                let size = u.arbitrary()?;

                let slice_cloned = slice.clone();
                let result = expect_might_panic(move || {
                    mock.insert(index, slice_cloned, size, false);
                    mock
                });

                Ok(Self::Insert {
                    id,
                    index,
                    slice,
                    size,
                    panics: match result {
                        Ok(mock) => {
                            mocks[id.0] = Some(mock);
                            *count += 1;
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
    I: Arbitrary<'d> + UnwindSafe + RefUnwindSafe + sherman::Index,
    S: Arbitrary<'d> + UnwindSafe + RefUnwindSafe + sherman::Slice<I> + Clone,
{
    const VARIANTS: u8 = BASIC_VARIANTS + SLICE_REF_VARIANTS;

    type Index = I;
    type Slice = S;

    fn arbitrary(
        u: &mut Unstructured<'d>,
        variant: u8,
        id: TreeId,
        count: &mut usize,
        refcount: &mut usize,
        mocks: &mut Vec<Option<Mock<I, S>>>,
        refs: &mut Vec<Option<mock::Ref>>,
    ) -> arbitrary::Result<Self> {
        match variant {
            v if v < BASIC_VARIANTS => {
                BasicCommand::arbitrary(u, v, id, count, refcount, mocks, refs).map(Self::Basic)
            }
            // make ref
            v if v == BASIC_VARIANTS => {
                let mock = mocks[id.0].take().unwrap();

                let index = u.arbitrary()?;

                let mock_mux = Mutex::new(mock);
                let result = expect_might_panic(|| mock_mux.lock().unwrap().make_ref(index));

                let mock = match mock_mux.into_inner() {
                    Ok(m) => m,
                    Err(poisoned) => poisoned.into_inner(),
                };
                mocks[id.0] = Some(mock);

                Ok(Self::MakeRef {
                    id,
                    index,
                    ref_id: result.map(|mock_ref| {
                        let ref_id = RefId(refs.len());
                        refs.push(Some(mock_ref));
                        *refcount += 1;
                        ref_id
                    }),
                })
            }
            // insert_ref
            v if v == BASIC_VARIANTS + 1 => {
                let mut mock = mocks[id.0].take().unwrap();
                *count -= 1;

                let index = u.arbitrary()?;
                let slice: S = u.arbitrary()?;
                let size = u.arbitrary()?;

                let slice_cloned = slice.clone();
                let result = expect_might_panic(move || {
                    let mock_ref = mock.insert(index, slice_cloned, size, true);
                    (mock, mock_ref)
                });

                Ok(Self::InsertRef {
                    id,
                    index,
                    slice,
                    size,
                    ref_id: result.map(|(mock, mock_ref)| {
                        mocks[id.0] = Some(mock);
                        *count += 1;
                        let ref_id = RefId(refs.len());
                        refs.push(Some(mock_ref.unwrap()));
                        *refcount += 1;
                        ref_id
                    }),
                })
            }
            // clone ref
            v if v == BASIC_VARIANTS + 2 => {
                let src_id = RefId(choose_sparse_index(u, *refcount, &refs)?);
                let new_id = RefId(refs.len());
                refs.push(Some(refs[src_id.0].unwrap()));

                Ok(Self::CloneRef { src_id, new_id })
            }
            // check ref.is_valid()
            v if v == BASIC_VARIANTS + 3 => {
                let ref_id = RefId(choose_sparse_index(u, *refcount, &refs)?);
                let mock = mocks[id.0].as_ref().unwrap();
                let mock_ref = refs[ref_id.0].as_ref().unwrap();
                let valid = mock.ref_slice(mock_ref).is_some();

                Ok(Self::CheckRefValid { id, ref_id, valid })
            }
            // check ref.borrow_slice()
            v if v == BASIC_VARIANTS + 4 => {
                let ref_id = RefId(choose_sparse_index(u, *refcount, &refs)?);
                let mock = mocks[id.0].as_ref().unwrap();
                let mock_ref = refs[ref_id.0].as_ref().unwrap();

                let range = mock.ref_range(mock_ref);
                if range.is_none() {
                    refs[ref_id.0] = None;
                    *refcount -= 1;
                }

                Ok(Self::CheckRefRange {
                    id,
                    ref_id,
                    range: range.ok_or(()),
                })
            }
            // check ref.range()
            v if v == BASIC_VARIANTS + 5 => {
                let ref_id = RefId(choose_sparse_index(u, *refcount, &refs)?);
                let mock = mocks[id.0].as_ref().unwrap();
                let mock_ref = refs[ref_id.0].as_ref().unwrap();

                let slice = mock.ref_slice(mock_ref).cloned();
                if slice.is_none() {
                    refs[ref_id.0] = None;
                    *refcount -= 1;
                }

                Ok(Self::CheckRefSlice {
                    id,
                    ref_id,
                    slice: slice.ok_or(()),
                })
            }
            // drop ref
            v if v == BASIC_VARIANTS + 6 => {
                let ref_id = RefId(choose_sparse_index(u, *refcount, &refs)?);
                refs[ref_id.0] = None;
                *refcount -= 1;
                Ok(Self::DropRef { ref_id })
            }
            _ => unreachable!("bad SliceRefCommand variant {variant}"),
        }
    }
}

impl<'d, I, S> ArbitraryCommand<'d> for CowCommand<I, S>
where
    I: Arbitrary<'d> + UnwindSafe + RefUnwindSafe + sherman::Index,
    S: Arbitrary<'d> + UnwindSafe + RefUnwindSafe + sherman::Slice<I> + Clone,
{
    const VARIANTS: u8 = BASIC_VARIANTS + COW_VARIANTS;

    type Index = I;
    type Slice = S;

    fn arbitrary(
        u: &mut Unstructured<'d>,
        variant: u8,
        id: TreeId,
        count: &mut usize,
        rc: &mut usize,
        mocks: &mut Vec<Option<Mock<I, S>>>,
        refs: &mut Vec<Option<mock::Ref>>,
    ) -> arbitrary::Result<Self> {
        match variant {
            v if v < BASIC_VARIANTS => {
                BasicCommand::arbitrary(u, v, id, count, rc, mocks, refs).map(Self::Basic)
            }
            // shallow clone
            v if v == BASIC_VARIANTS => {
                let new_id = TreeId(mocks.len());
                mocks.push(mocks[id.0].clone());
                *count += 1;
                Ok(Self::ShallowClone { src_id: id, new_id })
            }
            // drop tree
            v if v == BASIC_VARIANTS + 1 => {
                mocks[id.0].take();
                *count -= 1;
                Ok(Self::DropTree { id })
            }
            _ => unreachable!("bad CowCommand variant {variant}"),
        }
    }
}

/// Ongoing state for executing commands to an [`RleTree`] and mock implementation
pub struct RunnerState<I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    trees: Vec<Option<RleTree<I, S, P, M>>>,
    refs: Vec<Option<SliceRef<I, S, M>>>,
}

impl<I, S, P, const M: usize> RunnerState<I, S, P, M>
where
    I: UnwindSafe + RefUnwindSafe + sherman::Index,
    S: UnwindSafe + RefUnwindSafe + sherman::Slice<I> + Debug + Clone + PartialEq,
    P: RleTreeConfig<I, S, M> + param::SupportsInsert<I, S, M>,
{
    /// Creates a new, blank `RunnerState` to run a series of commands
    pub fn init() -> Self {
        RunnerState {
            trees: vec![Some(RleTree::new_empty())],
            refs: Vec::new(),
        }
    }

    /// Runs the command
    #[rustfmt::skip]
    pub fn run_basic_cmd(&mut self, cmd: &BasicCommand<I, S>) {
        match cmd {
            BasicCommand::Iter { id, start, end, access: Ok(access) } => {
                let tree = self.trees[id.0].as_ref().unwrap();
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
            }
            BasicCommand::Iter { id, start, end, access: Err(()) } => {
                let tree = self.trees[id.0].as_ref().unwrap();
                let panicked = expect_might_panic(|| {
                    let _ = tree.iter((*start, *end));
                })
                .is_err();

                assert!(panicked);
            }
            BasicCommand::Get { id, index, info: Ok((range, slice)) } => {
                let tree = self.trees[id.0].as_ref().unwrap();
                let entry = tree.get(*index);
                assert_eq!((entry.range(), entry.slice()), (range.clone(), slice));
            },
            BasicCommand::Get { id, index, info: Err(()) } => {
                let tree = self.trees[id.0].as_ref().unwrap();
                let index = *index;
                let panicked = expect_might_panic(|| tree.get(index)).is_err();
                assert!(panicked);
            },
            BasicCommand::Insert { id, index, slice, size, panics: false } => {
                let tree = self.trees[id.0].as_mut().unwrap();

                tree.insert(*index, slice.clone(), *size);
                tree.validate();
            }
            BasicCommand::Insert { id, index, slice, size, panics: true } => {
                let mut tree = self.trees[id.0].take().unwrap();

                let panicked = expect_might_panic(move || {
                    tree.insert(*index, slice.clone(), *size);
                })
                .is_err();

                assert!(panicked);
            }
        }
    }
}

impl<I, S, const M: usize> RunnerState<I, S, param::AllowSliceRefs, M>
where
    I: UnwindSafe + RefUnwindSafe + sherman::Index,
    S: UnwindSafe + RefUnwindSafe + sherman::Slice<I> + Debug + Clone + PartialEq,
{
    #[rustfmt::skip]
    pub fn run_slice_ref_cmd(&mut self, cmd: &SliceRefCommand<I, S>) {
        match cmd {
            SliceRefCommand::Basic(c) => self.run_basic_cmd(c),
            SliceRefCommand::MakeRef { id, index, ref_id: Ok(_) } => {
                let tree = self.trees[id.0].as_ref().unwrap();
                self.refs.push(Some(tree.get(*index).make_ref()));
            },
            SliceRefCommand::MakeRef { id, index, ref_id: Err(()) } => {
                let tree = self.trees[id.0].as_ref().unwrap();
                let panicked = expect_might_panic(|| {
                    let _ = tree.get(*index); // calls to `.get(...).make_ref()` only panic on
                                              // failure from `get(...)`, not `make_ref`
                })
                .is_err();

                assert!(panicked);
            },
            SliceRefCommand::InsertRef { id, index, slice, size, ref_id: Ok(_) } => {
                let tree = self.trees[id.0].as_mut().unwrap();
                self.refs.push(Some(tree.insert_ref(*index, slice.clone(), *size)));
                tree.validate();
            },
            SliceRefCommand::InsertRef { id, index, slice, size, ref_id: Err(()) } => {
                let mut tree = self.trees[id.0].take().unwrap();
                let panicked = expect_might_panic(move || {
                    let _ = tree.insert_ref(*index, slice.clone(), *size);
                })
                .is_err();

                assert!(panicked);
            },
            SliceRefCommand::CloneRef { src_id, new_id: _ } => {
                let new_ref = self.refs[src_id.0].as_ref().unwrap().clone();
                self.refs.push(Some(new_ref));
            }
            SliceRefCommand::CheckRefValid { id: _, ref_id, valid } => {
                let r = self.refs[ref_id.0].as_ref().unwrap();
                assert_eq!(r.is_valid(), *valid);
            },
            SliceRefCommand::CheckRefRange { id: _, ref_id, range: Ok(range) } => {
                let r = self.refs[ref_id.0].as_ref().unwrap();
                assert_eq!(r.range(), range.clone());
            },
            SliceRefCommand::CheckRefRange { id: _, ref_id, range: Err(()) } => {
                let r = self.refs[ref_id.0].as_ref().unwrap();
                assert!(r.try_range().is_none());
            },
            SliceRefCommand::CheckRefSlice { id: _, ref_id, slice: Ok(slice) } => {
                let r = self.refs[ref_id.0].as_ref().unwrap();
                assert_eq!(&*r.borrow_slice(), slice);
            },
            SliceRefCommand::CheckRefSlice { id: _, ref_id, slice: Err(()) } => {
                let r = self.refs[ref_id.0].as_ref().unwrap();
                assert!(r.try_borrow_slice().is_none());
            },
            SliceRefCommand::DropRef { ref_id } => drop(self.refs[ref_id.0].take()),
        }
    }
}

impl<I, S, const M: usize> RunnerState<I, S, param::AllowCow, M>
where
    I: UnwindSafe + RefUnwindSafe + sherman::Index,
    S: UnwindSafe + RefUnwindSafe + sherman::Slice<I> + Debug + Clone + PartialEq,
{
    pub fn run_cow_cmd(&mut self, cmd: &CowCommand<I, S>) {
        match cmd {
            CowCommand::Basic(c) => self.run_basic_cmd(c),
            CowCommand::ShallowClone { src_id, .. } => {
                let old_tree = self.trees[src_id.0].take().unwrap();
                let new_tree = old_tree.shallow_clone();
                old_tree.validate();
                new_tree.validate();

                self.trees[src_id.0] = Some(old_tree);
                self.trees.push(Some(new_tree));
            }
            CowCommand::DropTree { id } => drop(self.trees[id.0].take()),
        }
    }
}
