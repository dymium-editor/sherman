MIRIFLAGS := '-Zmiri-tree-borrows'

fmt:
    cargo +nightly fmt

# Run unit tests (without miri)
test:
    RUST_BACKTRACE=1 cargo +nightly test --features=nightly

# Run unit tests with miri
miri-test:
    MIRIFLAGS={{ MIRIFLAGS }} cargo +nightly miri test

# List fuzzing targets
fuzz-list:
    cargo +nightly fuzz list

# Clean the fuzz corpus
fuzz-clean-corpus:
    rm -rf fuzz/corpus/
    rm -rf fuzz/artifacts/

# Run a fuzzing target
fuzz target:
    cargo +nightly fuzz run {{target}}
