#![no_main]
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use sherman::{param::NoFeatures, Constant};
use sherman_fuzz_utils::{BasicCommand, CommandSequence, RunnerState};

#[derive(Debug)]
struct UpperLetter(char);

impl<'d> Arbitrary<'d> for UpperLetter {
    fn arbitrary(u: &mut Unstructured<'d>) -> arbitrary::Result<Self> {
        Ok(UpperLetter((b'A' + u.int_in_range(0_u8..=25)?) as char))
    }
}

fuzz_target!(|cmds: CommandSequence<BasicCommand<usize, UpperLetter>>| {
    let cmds = cmds.map(|cmd| cmd.map_slice(|c| Constant(c.0)));

    let mut runner: RunnerState<usize, Constant<char>, NoFeatures, 3> = RunnerState::init();

    for c in cmds.cmds {
        if runner
            .run_basic_cmd(&c, |x, y| {
                x.checked_add(y).is_none() || x + y > isize::MAX as usize
            })
            .is_err()
        {
            return;
        }
    }
});
