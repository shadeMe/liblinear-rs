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

/// Errors raised by a model's API call.
#[derive(Debug, Error)]
pub enum ModelError {
    /// One or more of the model's parameters are either incomplete or invalid.
    #[error("parameter error: {0}")]
    InvalidParameters(String),

    /// The model cannot be cast to another type.
    ///
    /// This can occur if the source and target solver types are not the same.
    #[error("invalid conversion: {0}")]
    InvalidConversion(String),

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
}
