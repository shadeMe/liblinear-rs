//! Utility structs and functions for reading/generating training
//! and prediction data.

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::str::FromStr;

use crate::errors::PredictionInputError;
use crate::errors::TrainingInputError;

/// A tuple of a (sparse) vector of features and their corresponding gold-standard label/target value.
#[derive(Default, Clone)]
pub struct TrainingInstance {
    features: Vec<(u32, f64)>,
    label: f64,
    last_feature_index: u32,
}

impl TrainingInstance {
    /// A list of tuples that encode a feature index and its corresponding feature value.
    pub fn features(&self) -> &Vec<(u32, f64)> {
        &self.features
    }

    /// The target value.
    ///
    /// Target values are either integers (in classification) or real numbers (in regression).
    pub fn label(&self) -> f64 {
        self.label
    }
}

impl FromStr for TrainingInstance {
    type Err = TrainingInputError;

    /// Converts a string representing a single training instance in the LIBSVM data format.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // assumes that the string is a valid line in the libSVM data format
        let mut last_feature_index = 0u32;
        let splits: Vec<&str> = s.split(" ").collect();
        match splits.len() {
            0 => {
                return Err(TrainingInputError::ParseError("Empty line".to_owned()));
            }
            1 => {
                return Err(TrainingInputError::ParseError(
                    "No features found".to_owned(),
                ));
            }
            _ => {
                let mut label = 0f64;
                let mut features: Vec<(u32, f64)> = Vec::default();

                for (idx, token) in splits.iter().enumerate() {
                    if token.is_empty() {
                        continue;
                    }

                    match idx {
                        0 => {
                            label = token.parse::<f64>().map_err(|_| {
                                TrainingInputError::ParseError(format!(
                                    "Couldn't parse output label '{}'",
                                    token
                                ))
                            })?
                        }
                        _ => {
                            let pair: Vec<&str> = token.split(":").collect();
                            if pair.len() != 2 {
                                return Err(TrainingInputError::ParseError(format!(
                                    "Couldn't feature pair '{}'",
                                    token
                                )));
                            }

                            features.push((
                                pair[0].parse::<u32>().map_err(|_| {
                                    TrainingInputError::ParseError(format!(
                                        "Couldn't parse feature index '{}'",
                                        pair[0]
                                    ))
                                })?,
                                pair[1].parse::<f64>().map_err(|_| {
                                    TrainingInputError::ParseError(format!(
                                        "Couldn't parse feature value '{}'",
                                        pair[1]
                                    ))
                                })?,
                            ));

                            let parsed_feature_index = features.last().unwrap().0;
                            if parsed_feature_index == 0 {
                                return Err(TrainingInputError::DataError(
                                    "Invalid feature index '0'".to_owned(),
                                ));
                            } else if parsed_feature_index < last_feature_index {
                                return Err(TrainingInputError::DataError(
                                    "Feature indices must be ascending".to_owned(),
                                ));
                            }

                            last_feature_index = parsed_feature_index;
                        }
                    }
                }

                Ok(TrainingInstance {
                    features,
                    label,
                    last_feature_index,
                })
            }
        }
    }
}

/// Input data for [LibLinearModel](trait.LibLinearModel.html) and [LibLinearCrossValidator](trait.LibLinearCrossValidator.html).
#[derive(Default, Clone)]
pub struct TrainingInput {
    instances: Vec<TrainingInstance>,
    last_feature_index: u32,
}

impl TrainingInput {
    #[doc(hidden)]
    pub fn last_feature_index(&self) -> u32 {
        self.last_feature_index
    }

    #[doc(hidden)]
    pub fn yield_data(self) -> Vec<TrainingInstance> {
        self.instances
    }

    /// Number of training instances.
    pub fn len_data(&self) -> usize {
        self.instances.len()
    }

    /// Dimensionality of the feature vector.
    pub fn len_features(&self) -> usize {
        self.last_feature_index as usize
    }

    /// Returns a reference to the training instance at the given index.
    pub fn get(&self, index: usize) -> Option<&TrainingInstance> {
        self.instances.get(index)
    }

    /// Create a new instance from a LIBSVM training data file.
    ///
    /// Each line in the data file represents a training instance and has the following format:
    ///
    /// `<target_value> <feature_index>:<feature_value> <feature_index>:<feature_value>...`
    ///
    /// Feature indices start from 1 and increase monotonically. However, they do not need to be continuous.
    pub fn from_libsvm_file(path: &str) -> Result<TrainingInput, TrainingInputError> {
        let mut out = TrainingInput::default();
        let reader = BufReader::new(
            File::open(path).map_err(|io_err| TrainingInputError::IoError(io_err.to_string()))?,
        );

        for line in reader.lines() {
            let new_training_instance = TrainingInstance::from_str(
                line.map_err(|io_err| TrainingInputError::IoError(io_err.to_string()))?
                    .as_str(),
            )?;

            out.last_feature_index = new_training_instance.last_feature_index;
            out.instances.push(new_training_instance);
        }

        Ok(out)
    }

    /// Create a new instance from a dense vector of features and their corresponding target values.
    pub fn from_dense_features(
        labels: Vec<f64>,
        features: Vec<Vec<f64>>,
    ) -> Result<TrainingInput, TrainingInputError> {
        if labels.len() != features.len() {
            return Err(TrainingInputError::DataError(
                "Mismatch between number of training instances and output labels".to_owned(),
            ));
        } else if labels.len() == 0 || features.len() == 0 {
            return Err(TrainingInputError::DataError(
                "No input/output data".to_owned(),
            ));
        }

        let last_feature_index = features.len() as u32;

        Ok(TrainingInput {
            instances: features
                .iter()
                .map(|feats| {
                    feats
                        .iter()
                        .zip(1..=feats.len())
                        .map(|(v, i)| (i as u32, *v))
                        .collect::<Vec<(u32, f64)>>()
                })
                .zip(labels.iter())
                .map(|(features, label)| TrainingInstance {
                    last_feature_index,
                    features,
                    label: *label,
                })
                .collect(),
            last_feature_index,
        })
    }

    /// Create a new instance from a sparse vector of features and their corresponding target values.
    ///
    /// The feature vector must be a list of tuples that encode a feature index and its corresponding feature value.
    pub fn from_sparse_features(
        labels: Vec<f64>,
        features: Vec<Vec<(u32, f64)>>,
    ) -> Result<TrainingInput, TrainingInputError> {
        if labels.len() != features.len() {
            return Err(TrainingInputError::DataError(
                "Mismatch between number of training instances and output labels".to_owned(),
            ));
        } else if labels.len() == 0 || features.len() == 0 {
            return Err(TrainingInputError::DataError(
                "No input/output data".to_owned(),
            ));
        }

        let last_feature_index = features.iter().fold(0u32, |acc, feats| {
            feats
                .iter()
                .fold(acc, |acc, (i, _v)| if *i > acc { *i } else { acc })
        });

        Ok(TrainingInput {
            instances: features
                .into_iter()
                .zip(labels.iter())
                .map(|(features, label)| TrainingInstance {
                    last_feature_index,
                    features,
                    label: *label,
                })
                .collect(),
            last_feature_index,
        })
    }
}

/// Test data for [LibLinearModel](trait.LibLinearModel.html).
#[derive(Default, Clone)]
pub struct PredictionInput {
    features: Vec<(u32, f64)>,
    last_feature_index: u32,
}

impl PredictionInput {
    /// A list of tuples that encode a feature index and its corresponding feature value.
    pub fn features(&self) -> &Vec<(u32, f64)> {
        &self.features
    }

    #[doc(hidden)]
    pub fn yield_data(self) -> Vec<(u32, f64)> {
        self.features
    }

    #[doc(hidden)]
    pub fn last_feature_index(&self) -> u32 {
        self.last_feature_index
    }

    /// Create a new instance from a dense vector of features.
    pub fn from_dense_features(
        features: Vec<f64>,
    ) -> Result<PredictionInput, PredictionInputError> {
        if features.is_empty() {
            return Err(PredictionInputError::DataError("No input data".to_owned()));
        }

        let last_feature_index = features.len() as u32;

        Ok(PredictionInput {
            features: features
                .iter()
                .zip(1..=features.len())
                .map(|(v, i)| (i as u32, *v))
                .collect::<Vec<(u32, f64)>>(),
            last_feature_index,
        })
    }

    /// Create a new instance from a sparse vector of features.
    ///
    /// The feature vector must be a list of tuples that encode a feature index and its corresponding feature value.
    pub fn from_sparse_features(
        features: Vec<(u32, f64)>,
    ) -> Result<PredictionInput, PredictionInputError> {
        if features.len() == 0 {
            return Err(PredictionInputError::DataError("No input data".to_owned()));
        }

        let last_feature_index = features
            .iter()
            .fold(0u32, |acc, (i, _v)| if *i > acc { *i } else { acc });

        Ok(PredictionInput {
            features,
            last_feature_index,
        })
    }
}
