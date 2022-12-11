#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use sherman::{param::NoFeatures, Constant};
use sherman_fuzz_utils::{BasicCommand, CommandSequence, ExtraRunnerInfo, RunnerState};
use std::fmt::{self, Debug, Formatter};

#[derive(Copy, Clone, PartialEq, Eq)]
struct UpperLetter(char);

// Forward the Debug implementation to the inner `char` so that printing the fuzz input can be more
// easily replicated.
impl Debug for UpperLetter {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'d> Arbitrary<'d> for UpperLetter {
    fn arbitrary(u: &mut Unstructured<'d>) -> arbitrary::Result<Self> {
        Ok(UpperLetter((b'A' + u.int_in_range(0_u8..=25)?) as char))
    }
}

fuzz_target!(|cmds: CommandSequence<BasicCommand<u8, Constant<UpperLetter>>>| {
    let cmds = cmds.map(|cmd| cmd.map_slice(|c| Constant(c.0)));

    let mut runner: RunnerState<u8, Constant<UpperLetter>, NoFeatures, 3> = RunnerState::init();
    let extra = ExtraRunnerInfo::new_basic();

    for c in cmds.cmds {
        runner.run_basic_cmd(&c, &extra);
    }
});
