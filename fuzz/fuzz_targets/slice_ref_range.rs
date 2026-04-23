#![no_main]

use libfuzzer_sys::fuzz_target;
use sherman::fuzz::{CharRange, OpSequence, SliceRefOperation};

type Input = OpSequence<SliceRefOperation<u8, CharRange>>;

fuzz_target!(|seq: Input| {
    seq.execute_assert();
});
