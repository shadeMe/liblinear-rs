//! Error enumerations.

use thiserror::Error;

/// Errors related to input/training data.
#[derive(Debug, Error)]
pub enum TrainingInputError {
    /// The LIBSVM data file was not found/couldn't be read.
    #[error("io error: {0}")]
    IoError(String),

    /// The LIBSVM data file has invalid/incomplete entries that do not conform to the expected format.
    #[error("parse error: {0}")]
    ParseError(String),

    /// The training data is invalid/incomplete.
    ///
    /// This can occur if the input features are missing, or if there is mismatch between the
    /// target values and the source features, or if the data is incorrect.
    #[error("data error: {0}")]
    DataError(String),
}

/// Errors related to test/prediction data.
#[derive(Debug, Error)]
pub enum PredictionInputError {
    /// The prediction data is invalid/incomplete/missing.
    #[error("data error: {0}")]
    DataError(String),
}

/// Errors related to a model's parameters.
#[derive(Debug, Error)]
pub enum ParameterError {
    /// One or more of the model's parameters are either incomplete or invalid.
    #[error("parameter error: {0}")]
    InvalidParameters(String),
}

/// Errors related to a model's input/training data.
#[derive(Debug, Error)]
pub enum ProblemError {
    /// The model's input/training data is either incomplete or invalid.
    #[error("input data error: {0}")]
    InvalidTrainingData(String),
}

/// Errors raised by a model's API call.
#[derive(Debug, Error)]
pub enum ModelError {
    /// The model's internal state is invalid.
    ///
    /// This can occur if the model's parameters or input data were not initialized correctly.
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// The model cannot be saved to/loaded from disk.
    ///
    /// This can occur if the serialized data was not found, or if the model is in an indeterminate
    /// state after deserialization.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// The model couldn't be applied to the test/prediction data.
    #[error("prediction error: {0}")]
    PredictionError(String),

    /// One or more of the arguments passed to the API call are invalid.
    #[error("illegal argument: {0}")]
    IllegalArgument(String),

    /// The model encountered an unexpected internal error.
    #[error("unknown error: {0}")]
    UnknownError(String),
}
