//! Tools for pretty-printing rust types and expressions

use std::fmt;

// TODO: docs
pub trait RustType {
    fn write_rust_type(f: &mut fmt::Formatter) -> fmt::Result;

    fn display_rust_type() -> impl fmt::Display {
        DisplayFn(Self::write_rust_type)
    }
}

// TODO: docs
pub trait RustExpr {
    fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result;

    fn display_rust_expr(&self) -> impl '_ + fmt::Display {
        DisplayFn(|f: &mut fmt::Formatter| self.write_rust_expr(f))
    }
}

struct DisplayFn<F>(F);

impl<F: Fn(&mut fmt::Formatter) -> fmt::Result> fmt::Display for DisplayFn<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (self.0)(f)
    }
}

macro_rules! rust_type_nongeneric {
    ($( $( $ty:ident ),+ $(,)? )?) => {
        $($(
        impl RustType for $ty {
            fn write_rust_type(f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(stringify!($ty))
            }
        }
        )+)?
    }
}

rust_type_nongeneric! {
    bool,
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    char,
    str, String,
}

macro_rules! rust_expr_debug {
    ($( $( $ty:ident ),+ $(,)? )?) => {
        $($(
        impl RustExpr for $ty {
            fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Debug::fmt(self, f)
            }
        }
        )+)?
    }
}

rust_expr_debug! {
    bool,
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    char,
}

impl RustExpr for &str {
    fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl RustExpr for String {
    fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_str().write_rust_expr(f)?;
        f.write_str(".to_owned()")
    }
}

impl<T: RustExpr> RustExpr for Option<T> {
    fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            None => f.write_str("None"),
            Some(v) => f.write_fmt(format_args!("Some({})", v.display_rust_expr())),
        }
    }
}

impl<T: RustExpr, E: RustExpr> RustExpr for Result<T, E> {
    fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ok(v) => f.write_fmt(format_args!("Ok({})", v.display_rust_expr())),
            Err(e) => f.write_fmt(format_args!("Err({})", e.display_rust_expr())),
        }
    }
}

impl<T: RustExpr> RustExpr for std::ops::Range<T> {
    fn write_rust_expr(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!(
            "{start}..{end}",
            start = self.start.display_rust_expr(),
            end = self.end.display_rust_expr()
        ))
    }
}
