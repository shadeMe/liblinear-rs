//! # liblinear
//!
//! `liblinear` is a Rust wrapper for the [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
//! C/C++ machine learning library.
//!
//! ## Usage
//! ```
//! use liblinear::{
//! Model,
//! Parameters,
//! TrainingInput,
//! PredictionInput,
//! solver::L2R_LR,
//! model::traits::*,
//! parameter::traits::*,
//! solver::traits::*,
//! };
//!
//! let x: Vec<Vec<(u32, f64)>> = vec![
//! vec![(1, 0.1), (3, 0.2)],
//! vec![(3, 9.9)],
//! vec![(1, 0.2), (2, 3.2)],
//! ];
//! let y = vec![0.0, 1.0, 0.0];
//!
//! let mut params = Parameters::<L2R_LR>::default();
//! params
//! .bias(0f64)
//! .stopping_tolerance(0.1f64)
//! .constraints_violation_cost(0.1f64);
//!
//! let model = Model::train(&TrainingInput::from_sparse_features(y, x).unwrap(), &params).unwrap();
//!
//! let predicted_class = model
//! .predict(&PredictionInput::from_sparse_features(vec![(3u32, 9.9f64)]).unwrap())
//! .unwrap();
//! println!("{}",predicted_class);
//! ```
#[macro_use]
extern crate num_derive;

pub mod errors;
mod ffi;
pub mod model;
pub mod parameter;
pub mod solver;
pub mod util;

pub use model::Model;
pub use parameter::Parameters;
pub use util::{PredictionInput, TrainingInput, TrainingInstance};

/// The version of the bundled LIBLINEAR C/C++ library.
pub fn liblinear_version() -> i32 {
    unsafe { ffi::liblinear_version }
}

/// Toggles the log output LIBLINEAR prints to the program's `stdout`.
pub fn toggle_liblinear_stdout_output(state: bool) {
    unsafe {
        match state {
            true => ffi::set_print_string_function(None),
            false => ffi::set_print_string_function(Some(ffi::silence_liblinear_stdout)),
        }
    }
}
