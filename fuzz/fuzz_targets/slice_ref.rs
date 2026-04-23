#![no_main]

use libfuzzer_sys::fuzz_target;
use sherman::Constant;
use sherman::fuzz::{OpSequence, SliceRefOperation, UpperLetter};

type Input = OpSequence<SliceRefOperation<u8, Constant<UpperLetter>>>;

fuzz_target!(|seq: Input| {
    seq.execute_assert();
});
