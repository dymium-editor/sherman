_default:
    @just -l

# Format code
fmt:
    cargo +nightly fmt

# Lint (with clippy & rustdoc)
lint:
    cargo +stable clippy
    cargo +nightly clippy --all-features
    cargo +stable clippy --profile=test
    cargo +nightly clippy --profile=test --all-features
    cargo +stable doc --no-deps
    cargo +nightly doc --no-deps --all-features
    cargo +stable doc --document-private-items --no-deps
    cargo +nightly doc --document-private-items --no-deps --all-features

# Generate default documentation
doc-minimal:
    cargo doc --no-deps

# Generate complete documentation
doc-full:
    cargo +nightly doc --no-deps --document-private-items --all-features

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
