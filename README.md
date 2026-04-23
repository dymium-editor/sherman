# Sherman — a featureful tree type

This library provides a single type — `RleTree`. It is a tree representation of a series of
[run-length encoded] values, supporting O(log n) insert & removal, and generic over the index and
value types.

[run-length encoded]: https://en.wikipedia.org/wiki/Run-length_encoding

## Notable features

* Values are retrieved by global index with efficient "shift" operations, where insertions and
    removals shift all values after them in O(log n) time.
* Values are run-length encoded — individual entries in the tree represent a uniform range of
    indexes
    * ... but values may be any type that satisfies some basic operations ("split" and "maybe
        join"), which opens up many more possibilities.
* Stable references -- the position and values of a prior insertion can be fetched in O(log n)
    time *(conflicts with COW)*
* Lock-free concurrent {copy,clone}-on-write -- `RleTree`s can be shared across threads, with
    concurrent writes copying or cloning only the path down to the changed node(s). *(conflicts with
    stable references)*

These features are all largely zero-cost when not explicitly enabled.

## Documentation

See <https://docs.rs/sherman> for the most recent release, or run `cargo doc --no-deps --open`
locally for the latest development version.

## Development & testing

> ![NOTE]
> Most local development operations require a nightly toolchain.

Development & testing operations are managed with [`just`](https://just.systems/):

* `just fmt` for formatting
* `just lint` for linting with clippy
* `just test` for running unit tests
* `just miri-test` for running unit tests with [Miri](https://github.com/rust-lang/miri/)
* `just fuzz <target>` for fuzzing a particular target (see the list with `just fuzz-list`)

## Miscellanea

### Naming

This library is named after [General Sherman], a really big tree.

[General Sherman]: https://en.wikipedia.org/wiki/General_Sherman_(tree)
