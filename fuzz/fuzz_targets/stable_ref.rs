#![no_main]

use libfuzzer_sys::fuzz_target;
use sherman::Constant;
use sherman::fuzz::{OpSequence, StableRefOperation, UpperLetter};

type Input = OpSequence<StableRefOperation<u8, Constant<UpperLetter>>>;

fuzz_target!(|seq: Input| {
    seq.execute_assert();
});
