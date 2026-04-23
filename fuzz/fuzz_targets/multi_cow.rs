#![no_main]

use libfuzzer_sys::fuzz_target;
use sherman::Constant;
use sherman::fuzz::{MultiCowOperation, OpSequence, UpperLetter};

type Input = OpSequence<MultiCowOperation<u8, Constant<UpperLetter>>>;

fuzz_target!(|seq: Input| {
    seq.execute_assert();
});
