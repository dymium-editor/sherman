//! Various collections of unit tests
//!
//! The bulk of our confidence in correctness comes from fuzzing, but each failure from fuzzing is
//! used as a new unit tests, so that there's a moderate corpus here as well.

/// Manually written test cases
mod manual;

/// Tests from the `basic` fuzz target
mod fuzz_basic;
