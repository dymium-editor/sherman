#![no_main]

use libfuzzer_sys::fuzz_target;
use sherman::fuzz::{BasicOperation, CharRange, OpSequence};

type Input = OpSequence<BasicOperation<u8, CharRange>>;

fuzz_target!(|seq: Input| {
    seq.execute_assert();
});
