use crate::{errors::ProblemError, ffi, util::TrainingInput, FeatureNode};

/// Represents a one-to-one mapping of source features to target values.
///
/// Source features are represented as sparse vectors of real numbers. Target values are
/// either integers (in classification) or real numbers (in regression).
pub trait LibLinearProblem: Clone {
    /// The feature vectors of each training instance.
    fn source_features(&self) -> &[Vec<FeatureNode>];

    /// Target labels/values of each training instance.
    fn target_values(&self) -> &[f64];

    /// Bias of the input data.
    fn bias(&self) -> f64;
}

pub(crate) struct Problem {
    backing_store_labels: Vec<f64>,
    backing_store_features: Vec<Vec<FeatureNode>>,
    _backing_store_feature_ptrs: Vec<*const FeatureNode>,
    bound: ffi::Problem,
}

impl Problem {
    fn new(input_data: TrainingInput, bias: f64) -> Self {
        let num_training_instances = input_data.len_data() as i32;
        let num_features = input_data.len_features() as i32;
        let has_bias = bias >= 0f64;
        let last_feature_index = input_data.last_feature_index() as i32;

        let (mut transformed_features, labels): (Vec<Vec<FeatureNode>>, Vec<f64>) =
            input_data.yield_data().iter().fold(
                (Vec::<Vec<FeatureNode>>::default(), Vec::<f64>::default()),
                |(mut feats, mut labels), instance| {
                    feats.push(
                        instance
                            .features()
                            .iter()
                            .map(|(index, value)| FeatureNode {
                                index: *index as i32,
                                value: *value,
                            })
                            .collect(),
                    );
                    labels.push(instance.label());
                    (feats, labels)
                },
            );

        // add feature nodes for non-negative biases and an extra book-end
        transformed_features = transformed_features
            .into_iter()
            .map(|mut v: Vec<FeatureNode>| {
                if has_bias {
                    v.push(FeatureNode {
                        index: last_feature_index + 1,
                        value: bias,
                    });
                }

                v.push(FeatureNode {
                    index: -1,
                    value: 0f64,
                });

                v
            })
            .collect();

        let transformed_feature_ptrs: Vec<*const FeatureNode> =
            transformed_features.iter().map(|e| e.as_ptr()).collect();

        // the pointers passed to ffi::Problem will be valid even after their corresponding Vecs
        // are moved to a different location as they point to the actual backing store on the heap
        Self {
            bound: ffi::Problem {
                l: num_training_instances as i32,
                n: num_features + if has_bias { 1 } else { 0 } as i32,
                y: labels.as_ptr(),
                x: transformed_feature_ptrs.as_ptr(),
                bias,
            },
            backing_store_labels: labels,
            backing_store_features: transformed_features,
            _backing_store_feature_ptrs: transformed_feature_ptrs,
        }
    }

    pub(crate) fn ffi_obj(&self) -> &ffi::Problem {
        &self.bound
    }
}

impl LibLinearProblem for Problem {
    fn source_features(&self) -> &[Vec<FeatureNode>] {
        &self.backing_store_features
    }

    fn target_values(&self) -> &[f64] {
        &self.backing_store_labels
    }

    fn bias(&self) -> f64 {
        self.bound.bias
    }
}

impl Clone for Problem {
    fn clone(&self) -> Self {
        let labels = self.backing_store_labels.clone();
        let transformed_features: Vec<Vec<FeatureNode>> = self.backing_store_features.clone();
        let transformed_feature_ptrs: Vec<*const FeatureNode> =
            transformed_features.iter().map(|e| e.as_ptr()).collect();

        Self {
            bound: ffi::Problem {
                l: self.bound.l,
                n: self.bound.n,
                y: labels.as_ptr(),
                x: transformed_feature_ptrs.as_ptr(),
                bias: self.bound.bias,
            },
            backing_store_labels: labels,
            backing_store_features: transformed_features,
            _backing_store_feature_ptrs: transformed_feature_ptrs,
        }
    }
}

/// Builder for [LibLinearProblem](enum.LibLinearProblem.html).
#[derive(Clone, Debug)]
pub struct ProblemBuilder {
    input_data: Option<TrainingInput>,
    bias: f64,
}

impl Default for ProblemBuilder {
    fn default() -> Self {
        Self {
            input_data: None,
            bias: -1.0,
        }
    }
}

impl ProblemBuilder {
    /// Set input/training data.
    pub fn input_data(&mut self, input_data: TrainingInput) -> &mut Self {
        self.input_data = Some(input_data);
        self
    }

    /// Set bias. If `bias >= 0`, it's appended to the feature vector for every instance.
    /// Must be set to `1.0` for one-class SVM solvers.
    ///
    /// Default: `-1.0`
    pub fn bias(&mut self, bias: f64) -> &mut Self {
        self.bias = bias;
        self
    }

    pub(crate) fn build(self) -> Result<Problem, ProblemError> {
        let input_data = self.input_data.ok_or(ProblemError::InvalidTrainingData(
            "Missing input/training data".to_owned(),
        ))?;

        Ok(Problem::new(input_data, self.bias))
    }
}

/// Super-trait of [LibLinearModel](trait.LibLinearModel.html) and [LibLinearCrossValidator](trait.LibLinearCrossValidator.html).
pub trait HasLibLinearProblem {
    type Output: LibLinearProblem;

    /// The problem associated with the model/cross-validator.
    ///
    /// This will return `None` when called on a model that was deserialized/loaded from disk.
    fn problem(&self) -> Option<&Self::Output>;
}
