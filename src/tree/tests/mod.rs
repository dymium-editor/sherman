//! Various collections of unit tests
//!
//! The bulk of the "we're confident this works" comes from fuzzing, but each failure from fuzzing
//! is used as a new unit tests, so there's a few of them here as well.

/// "Basic" fuzz tests -- just insertion, and without any extra features
///
/// Generated from the `no_features_basic_slices` fuzz target.
mod basic;
/// Fuzz tests for COW-enabled trees
///
/// Generated from the `cow_basic_slices` fuzz target.
mod cow;
/// Manually-written test cases
mod manual;
