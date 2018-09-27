use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::str::FromStr;

#[derive(Debug, Fail)]
pub enum TrainingInputError {
    /// File read/write errors
    #[fail(display = "io error: {}", e)]
    IoError { e: String },
    /// Parsing errors
    #[fail(display = "parse error: {}", e)]
    ParseError { e: String },
    /// No data, mismatch between output and input, invalid data
    #[fail(display = "data error: {}", e)]
    DataError { e: String },
}

/// A tuple of a (sparse)vector of features and their corresponding gold-standard label
#[derive(Default, Clone)]
pub struct TrainingInstance {
    features: Vec<(u32, f64)>,
    label: f64,
    last_feature_index: u32,
}

impl TrainingInstance {
    pub fn features(&self) -> &Vec<(u32, f64)> {
        &self.features
    }
    pub fn label(&self) -> f64 {
        self.label
    }
}

impl FromStr for TrainingInstance {
    type Err = TrainingInputError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // assumes that the string is a valid line in the libSVM data format
        let mut last_feature_index = 0u32;
        let splits: Vec<&str> = s.split(" ").collect();
        match splits.len() {
            0 => {
                return Err(TrainingInputError::ParseError {
                    e: "Empty line".to_string(),
                })
            }
            1 => {
                return Err(TrainingInputError::ParseError {
                    e: "No features found".to_string(),
                })
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
                                TrainingInputError::ParseError {
                                    e: format!("Couldn't parse output label '{}'", token)
                                        .to_string(),
                                }
                            })?
                        }
                        _ => {
                            let pair: Vec<&str> = token.split(":").collect();
                            if pair.len() != 2 {
                                return Err(TrainingInputError::ParseError {
                                    e: format!("Couldn't feature pair '{}'", token).to_string(),
                                });
                            }

                            features.push((
                                pair[0].parse::<u32>().map_err(|_| {
                                    TrainingInputError::ParseError {
                                        e: format!("Couldn't parse feature index '{}'", pair[0])
                                            .to_string(),
                                    }
                                })?,
                                pair[1].parse::<f64>().map_err(|_| {
                                    TrainingInputError::ParseError {
                                        e: format!("Couldn't parse feature value '{}'", pair[1])
                                            .to_string(),
                                    }
                                })?,
                            ));

                            let parsed_feature_index = features.last().unwrap().0;
                            if parsed_feature_index == 0 {
                                return Err(TrainingInputError::DataError {
                                    e: "Invalid feature index '0'".to_string(),
                                });
                            } else if parsed_feature_index < last_feature_index {
                                return Err(TrainingInputError::DataError {
                                    e: "Feature indices must be ascending".to_string(),
                                });
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

#[derive(Default)]
pub struct TrainingInput {
    instances: Vec<TrainingInstance>,
    last_feature_index: u32,
}

impl TrainingInput {
    pub fn last_feature_index(&self) -> u32 {
        self.last_feature_index
    }
    pub fn yield_data(self) -> Vec<TrainingInstance> {
        self.instances
    }
    pub fn len_data(&self) -> usize {
        self.instances.len()
    }
    pub fn len_features(&self) -> usize {
	    self.last_feature_index as usize
    }

    pub fn from_libsvm_file(path: &str) -> Result<TrainingInput, TrainingInputError> {
        let mut out = TrainingInput::default();
        let reader = BufReader::new(File::open(path).map_err(|io_err| {
            TrainingInputError::IoError {
                e: io_err.to_string(),
            }
        })?);

        for line in reader.lines() {
            let mut new_training_instance = TrainingInstance::from_str(
                line.map_err(|io_err| TrainingInputError::IoError {
                    e: io_err.to_string(),
                })?
                    .as_str(),
            )?;

            out.last_feature_index = new_training_instance.last_feature_index;
            out.instances.push(new_training_instance);
        }

        Ok(out)
    }

    pub fn from_dense_features(
        labels: Vec<f64>,
        features: Vec<Vec<f64>>,
    ) -> Result<TrainingInput, TrainingInputError> {
        if labels.len() != features.len() {
            return Err(TrainingInputError::DataError {
                e: "Mismatch between number of training instances and output labels".to_string(),
            });
        } else if labels.len() == 0 || features.len() == 0 {
            return Err(TrainingInputError::DataError {
                e: "No input/output data".to_string(),
            });
        }

        let last_feature_index = (features.len() + 1) as u32;

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
}

#[derive(Debug, Fail)]
pub enum PredictionInputError {
    /// No data, mismatch between output and input, invalid data
    #[fail(display = "data error: {}", e)]
    DataError { e: String },
}

#[derive(Default)]
pub struct PredictionInput {
    features: Vec<(u32, f64)>,
    last_feature_index: u32,
}

impl PredictionInput {
    pub fn yield_data(self) -> Vec<(u32, f64)> {
        self.features
    }
    pub fn last_feature_index(&self) -> u32 {
        self.last_feature_index
    }
    pub fn from_dense_features(
        features: Vec<f64>,
    ) -> Result<PredictionInput, PredictionInputError> {
        if features.len() == 0 {
            return Err(PredictionInputError::DataError {
                e: "No input data".to_string(),
            });
        }

        let last_feature_index = (features.len() + 1) as u32;

        Ok(PredictionInput {
            features: features
                .iter()
                .zip(1..=features.len())
                .map(|(v, i)| (i as u32, *v))
                .collect::<Vec<(u32, f64)>>(),
            last_feature_index,
        })
    }
}
