//! # liblinear
//!
//! `liblinear` is a Rust wrapper for the [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
//! C/C++ machine learning library.
//!
//! Use [Builder](struct.Builder.html) to instantiate a [LibLinearModel](trait.LibLinearModel.html).

pub mod errors;
mod ffi;
pub mod model;
pub mod parameter;
pub mod problem;
pub mod util;

use errors::ModelError;
pub use ffi::FeatureNode;
use model::{LibLinearCrossValidator, LibLinearModel, Model};
use parameter::ParameterBuilder;
use problem::ProblemBuilder;

/// Primary model builder. Functions as the entry point into the API.
#[derive(Clone)]
pub struct Builder {
    problem_builder: ProblemBuilder,
    parameter_builder: ParameterBuilder,
}

impl Default for Builder {
    /// Creates a new instance of the builder.
    fn default() -> Self {
        Self {
            ..Default::default()
        }
    }
}

impl Builder {
    /// Builder for the model's linear problem.
    pub fn problem(&mut self) -> &mut ProblemBuilder {
        &mut self.problem_builder
    }

    /// Builder for the model's tunable parameters.
    pub fn parameters(&mut self) -> &mut ParameterBuilder {
        &mut self.parameter_builder
    }

    /// Builds a [LibLinearCrossValidator](trait.LibLinearCrossValidator.html) instance with the given problem and parameters.
    pub fn build_cross_validator(self) -> Result<impl LibLinearCrossValidator, ModelError> {
        Ok(Model::from_input(
            self.problem_builder
                .build()
                .map_err(|e| ModelError::IllegalArgument(e.to_string()))?,
            self.parameter_builder
                .build()
                .map_err(|e| ModelError::IllegalArgument(e.to_string()))?,
            false,
        )?)
    }

    /// Builds a [LibLinearModel](trait.LibLinearModel.html) instance with the given problem and parameters.
    pub fn build_model(self) -> Result<impl LibLinearModel, ModelError> {
        Ok(Model::from_input(
            self.problem_builder
                .build()
                .map_err(|e| ModelError::IllegalArgument(e.to_string()))?,
            self.parameter_builder
                .build()
                .map_err(|e| ModelError::IllegalArgument(e.to_string()))?,
            true,
        )?)
    }
}

/// Helper struct to serialize and deserialize [LibLinearModel](trait.LibLinearModel.html) instances.
pub struct Serializer;

impl Serializer {
    /// Loads a model from disk.
    ///
    /// The loaded model will have no associated [LibLinearProblem](trait.LibLinearProblem.html) instance.
    /// With the exception of the solver type, all parameters in the associated [LibLinearParameter](trait.LibLinearParameter.html) instance
    /// will be reset to their default values.
    pub fn load_model(path_to_serialized_model: &str) -> Result<impl LibLinearModel, ModelError> {
        Ok(Model::from_serialized_file(path_to_serialized_model)?)
    }

    /// Saves a model to disk.
    ///
    /// Convenience method that calls `save_to_disk` on the model instance.
    pub fn save_model(
        path_to_serialized_model: &str,
        model: &impl LibLinearModel,
    ) -> Result<(), ModelError> {
        Ok(model.save_to_disk(path_to_serialized_model)?)
    }
}

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
