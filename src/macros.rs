//! A small handful of helper macros

#[cfg(test)]
use std::sync::atomic::AtomicBool;

#[cfg(test)]
#[allow(dead_code)] // this is only used when actively debugging.
pub(crate) static DEBUG: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
#[allow(unused_macros)]
macro_rules! enable_debug {
    () => {{
        $crate::macros::DEBUG.store(true, std::sync::atomic::Ordering::SeqCst);
    }};
}

#[cfg(test)]
#[allow(unused_macros)]
macro_rules! disable_debug {
    () => {{
        $crate::macros::DEBUG.store(false, std::sync::atomic::Ordering::SeqCst);
    }};
}

#[allow(unused_macros)]
macro_rules! debug_println {
    ($($args:tt)*) => {
        if cfg!(test) {
            #[cfg(test)]
            let debug = $crate::macros::DEBUG.load(std::sync::atomic::Ordering::SeqCst);
            #[cfg(not(test))]
            let debug = false;
            if debug {
                println!($($args)*);
            }
        };
    };
    ($($args:tt)*) => {};
}
