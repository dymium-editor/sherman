#![no_main]

use libfuzzer_sys::fuzz_target;
use sherman::fuzz::{CharRange, MultiCowOperation, OpSequence};

type Input = OpSequence<MultiCowOperation<u8, CharRange>>;

fuzz_target!(|seq: Input| {
    seq.execute_assert();
});
