_default:
    @just -l

# Format code
fmt:
    cargo +nightly fmt

# Lint (with clippy)
lint:
    cargo +stable clippy
    cargo +nightly clippy --features=nightly,fuzz
    cargo +stable clippy --profile=test
    cargo +nightly clippy --profile=test --features=nightly,fuzz

# Run unit tests (without miri)
test:
    RUST_BACKTRACE=1 cargo +nightly test --features=nightly

# Run unit tests with miri
miri-test:
    cargo +nightly miri test

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
