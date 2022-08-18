//! A small handful of helper macros

#[cfg(test)]
use std::sync::atomic::AtomicBool;

// Macro that is `unreachable!` in debug mode, but `unreachable_unchecked` in release mode.
macro_rules! weak_unreachable {
    ($($tt:tt)*) => {{
        if cfg!(debug_assertions) {
            unreachable!($($tt)*);
        } else {
            std::hint::unreachable_unchecked()
        }
    }};
}

// Helper macro so that a bunch of our unsafe functions (particular in `mod node`) can have their
// invariants checked in debug mode, but *assumed* in release mode.
macro_rules! weak_assert {
    ($cond:expr $(, $($tt:tt)+)?) => {{
        if !$cond {
            // We use `if cfg!(...)` here so that any formatting arguments will be marked as used
            // by the macro, even in release mode (where they are definitely not used).
            if cfg!(debug_assertions) {
                panic!(weak_assert!(@panic_args $cond $(, $($tt)+)?));
            } else {
                std::hint::unreachable_unchecked()
            }
        }
    }};
    (@panic_args $cond:expr $(,)?) => { concat!("debug assertion failed: ", stringify!($cond)) };
    (@panic_args $cond:expr, $($tt:tt)+) => { $($tt)+ };
}

#[cfg(test)]
pub(crate) static DEBUG: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
macro_rules! enable_debug {
    () => {{
        $crate::macros::DEBUG.store(true, std::sync::atomic::Ordering::SeqCst);
    }};
}

#[cfg(test)]
macro_rules! disable_debug {
    () => {{
        $crate::macros::DEBUG.store(false, std::sync::atomic::Ordering::SeqCst);
    }};
}

macro_rules! debug_println {
    ($($args:tt)*) => {
        #[cfg(test)]
        {
            if $crate::macros::DEBUG.load(std::sync::atomic::Ordering::SeqCst) {
                println!($($args)*);
            }
        };
    };
    ($($args:tt)*) => {};
}
