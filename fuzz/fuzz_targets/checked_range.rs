#![no_main]

use libfuzzer_sys::fuzz_target;
use sherman::fuzz::{CharRange, CheckedIndexOperation, OpSequence};

type Input = OpSequence<CheckedIndexOperation<u8, CharRange>>;

fuzz_target!(|seq: Input| {
    seq.execute_assert();
});
