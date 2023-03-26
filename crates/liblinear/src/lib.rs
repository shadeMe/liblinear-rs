//! # liblinear
//!
//! `liblinear` is a Rust wrapper for the [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
//! C/C++ machine learning library.
#[macro_use]
extern crate num_derive;

pub mod errors;
mod ffi;
pub mod model;
pub mod parameter;
pub mod solver;
pub mod util;

/// The version of the bundled liblinear C-library.
pub fn liblinear_version() -> i32 {
    unsafe { ffi::liblinear_version }
}

/// Toggles the log output liblinear prints to the program's `stdout`.
pub fn toggle_liblinear_stdout_output(state: bool) {
    unsafe {
        match state {
            true => ffi::set_print_string_function(None),
            false => ffi::set_print_string_function(Some(ffi::silence_liblinear_stdout)),
        }
    }
}
