#![no_main]

use libfuzzer_sys::fuzz_target;
use sherman::Constant;
use sherman::fuzz::{BasicOperation, OpSequence, UpperLetter};

type Input = OpSequence<BasicOperation<u8, Constant<UpperLetter>>>;

fuzz_target!(|seq: Input| {
    seq.execute_assert();
});
