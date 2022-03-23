use std::ffi::CString;

use crate::{
    errors::{ModelError, PredictionInputError},
    ffi,
    parameter::{HasLibLinearParameter, LibLinearParameter, Parameter, ParameterBuilder},
    problem::{HasLibLinearProblem, Problem},
    util::PredictionInput,
    FeatureNode,
};

/// Represents a linear model that can be used for prediction.
pub trait LibLinearModel: HasLibLinearProblem + HasLibLinearParameter {
    /// Returns one of the following values:
    ///
    /// * For a classification model, the predicted class is returned.
    /// * For a regression model, the function value of `x` calculated using the model is returned.
    fn predict(&self, features: PredictionInput) -> Result<f64, ModelError>;

    /// Returns a tuple of the following values:
    ///
    /// * A list of decision values. If `k` is the number of classes, each element includes results
    /// of predicting k binary-class SVMs. If `k == 2` and solver is not `MCSVM_CS`, only one decision value
    ///	is returned.
    ///
    ///   The values correspond to the classes returned by the `labels` method.
    ///
    ///
    /// * The class with the highest decision value.
    fn predict_values(&self, features: PredictionInput) -> Result<(Vec<f64>, f64), ModelError>;

    /// Returns a tuple of the following values:
    ///
    /// * A list of probability estimates. each element contains `k` values
    /// indicating the probability that the testing instance is in each class.
    ///
    ///   The values correspond to the classes returned by the `labels` method.
    ///
    ///
    /// * The class with the highest probability.
    ///
    /// Only supports logistic regression solvers.
    fn predict_probabilities(
        &self,
        features: PredictionInput,
    ) -> Result<(Vec<f64>, f64), ModelError>;

    /// Returns the coefficient for the feature with the given index
    /// and the class with the given (label) index.
    ///
    /// Note that while feature indices start from `1`, label indices start from `0`.
    /// If the feature index is not in the valid range, a zero value will be returned.
    ///
    /// For classification models, if the label index is not in the valid range, a zero value will be returned.
    /// For regression and one-class models, the label index is ignored.
    fn feature_coefficient(&self, feature_index: i32, label_index: i32) -> f64;

    /// Returns the bias term corresponding to the class with the given index.
    ///
    /// For classification models, if label index is not in a valid range, a zero value will be returned.
    /// For regression models, the label index is ignored.
    ///
    /// Will return an error if called on one-class SMV models.
    fn label_bias(&self, label_index: i32) -> Result<f64, ModelError>;

    /// Returns the bias term used in one-class SVMs.
    ///
    /// Will return an error if called on non-one-class models.
    fn rho(&self) -> Result<f64, ModelError>;

    /// Returns the bias of the input data with which the model was trained.
    fn bias(&self) -> f64;

    /// Returns the labels/classes learned by the model.
    fn labels(&self) -> &Vec<i32>;

    /// Returns the number of classes of the model.
    ///
    /// For regression models, `2` is returned.
    fn num_classes(&self) -> usize;

    /// Returns the number of features of the input data with which the model was trained.
    fn num_features(&self) -> usize;

    /// Serializes the model and saves it to disk.
    ///
    /// Only serializes the learned model weights, labels and solver type.
    fn save_to_disk(&self, file_path: &str) -> Result<(), ModelError>;
}

/// Represents a linear model that can be used for validation.
pub trait LibLinearCrossValidator: HasLibLinearProblem + HasLibLinearParameter {
    /// Performs k-folds cross-validation and returns the predicted labels.
    ///
    /// Number of folds must be `>= 2`.
    fn cross_validation(&self, folds: i32) -> Result<Vec<f64>, ModelError>;

    /// Performs k-folds cross-validation to find the best cost value (parameter _C_) and regression
    /// loss sensitivity (parameter _p_) and returns a tuple with the following values:
    ///
    /// * The best cost value.
    /// * The accuracy of the best cost value (classification) or mean squared error (regression).
    /// * The best regression loss sensitivity value (only for regression).
    ///
    /// Supported Solvers:
    /// * L2R_LR, L2R_L2LOSS_SVC - Cross validation is conducted many times with values of `_C_
    /// = n * start_C; n = 1, 2, 4, 8...` to find the value with the highest cross validation
    /// accuracy. The procedure stops when the models of all folds become stable or when the cost
    /// reaches the upper-bound `max_cost = 1024`. If `start_cost <= 0`, an appropriately small value is
    /// automatically calculated and used instead.
    ///
    ///
    /// * L2R_L2LOSS_SVR - Cross validation is conducted in a two-fold loop. The outer loop
    /// iterates over the values of `_p_ = n / (20 * max_loss_sensitivity); n = 19, 18, 17...0`. For each value
    /// of _p_, the inner loop performs cross validation with values of `_C_ = n * start_C; n = 1, 2,
    /// 4, 8...` to find the value with the lowest mean squared error. The procedure stops when the
    /// models of all folds become stable or when the cost reaches the upper-bound `max_cost = 1048576`. If
    /// `start_cost <= 0`, an appropriately small value is automatically calculated and used instead.
    ///
    ///   `max_p` is automatically calculated from the problem's training data.
    /// If `start_loss_sensitivity <= 0`, it is set to `max_loss_sensitivity`. Otherwise, the outer
    /// loop starts with the first `_p_ = n / (20 * max_p)` that is `<= start_loss_sensitivity`.
    fn find_optimal_constraints_violation_cost_and_loss_sensitivity(
        &self,
        folds: i32,
        start_cost: f64,
        start_loss_sensitivity: f64,
    ) -> Result<(f64, f64, f64), ModelError>;
}

pub(crate) struct Model {
    problem: Option<Problem>,
    parameter: Parameter,
    backing_store_labels: Vec<i32>,
    bound: *mut ffi::Model,
}

impl Model {
    pub(crate) fn from_input(
        problem: Problem,
        parameter: Parameter,
        train: bool,
    ) -> Result<Self, ModelError> {
        let mut bound: *mut ffi::Model = std::ptr::null_mut();
        if train {
            bound = unsafe { ffi::train(problem.ffi_obj(), parameter.ffi_obj()) };
            if bound.is_null() {
                return Err(ModelError::UnknownError(
                    "train() returned a NULL pointer".to_owned(),
                ));
            }
        }

        let mut backing_store_labels = Vec::<i32>::new();
        unsafe {
            if train {
                for i in 0..(*bound).nr_class {
                    backing_store_labels.push(*(*bound).label.offset(i as isize));
                }
            }
        }

        Ok(Self {
            problem: Some(problem),
            parameter,
            backing_store_labels,
            bound,
        })
    }

    pub(crate) fn from_serialized_file(path_to_serialized_model: &str) -> Result<Self, ModelError> {
        unsafe {
            let file_path_cstr = CString::new(path_to_serialized_model).unwrap();
            let bound = ffi::load_model(file_path_cstr.as_ptr());

            if bound.is_null() {
                return Err(ModelError::SerializationError(
                    "load_model() returned a NULL pointer".to_owned(),
                ));
            }

            let mut backing_store_labels = Vec::<i32>::new();
            for i in 0..(*bound).nr_class {
                backing_store_labels.push(*(*bound).label.offset(i as isize));
            }

            let mut parameter_builder = ParameterBuilder::default();
            parameter_builder.solver_type(std::mem::transmute((*bound).param.solver_type as i8));

            let parameter = parameter_builder
                .build()
                .map_err(|e| ModelError::SerializationError(e.to_string()))?;

            Ok(Self {
                problem: None,
                // solver_type is the only parameter that's serialized to disk
                // init the parameter object with just that and pass the defaults for the rest
                parameter,
                backing_store_labels,
                bound,
            })
        }
    }

    fn preprocess_prediction_input(
        &self,
        prediction_input: PredictionInput,
    ) -> Result<Vec<FeatureNode>, PredictionInputError> {
        assert_ne!(self.bound.is_null(), true);

        let last_feature_index = prediction_input.last_feature_index() as i32;
        if last_feature_index as usize != self.num_features() {
            return Err(PredictionInputError::DataError(format!(
                "Expected {} features, found {} instead",
                self.num_features(),
                last_feature_index
            )));
        }

        let bias = unsafe { (*self.bound).bias };
        let has_bias = bias >= 0f64;
        let mut data: Vec<FeatureNode> = prediction_input
            .yield_data()
            .iter()
            .map(|(index, value)| FeatureNode {
                index: *index as i32,
                value: *value,
            })
            .collect();

        if has_bias {
            data.push(FeatureNode {
                index: last_feature_index + 1,
                value: bias,
            });
        }

        data.push(FeatureNode {
            index: -1,
            value: 0f64,
        });

        Ok(data)
    }
}

impl HasLibLinearProblem for Model {
    type Output = Problem;

    fn problem(&self) -> Option<&Self::Output> {
        self.problem.as_ref()
    }
}

impl HasLibLinearParameter for Model {
    type Output = Parameter;

    fn parameter(&self) -> &Self::Output {
        &self.parameter
    }
}

impl LibLinearModel for Model {
    fn predict(&self, features: PredictionInput) -> Result<f64, ModelError> {
        Ok(self.predict_values(features)?.1)
    }

    fn predict_values(&self, features: PredictionInput) -> Result<(Vec<f64>, f64), ModelError> {
        let transformed_features = self
            .preprocess_prediction_input(features)
            .map_err(|err| ModelError::PredictionError(err.to_string()))?;

        unsafe {
            let mut output_values: Vec<f64> = match (*self.bound).nr_class {
                2 => vec![0f64; 1],
                l => vec![0f64; l as usize],
            };

            let best_class = ffi::predict_values(
                self.bound,
                transformed_features.as_ptr(),
                output_values.as_mut_ptr(),
            );
            Ok((output_values, best_class))
        }
    }

    fn predict_probabilities(
        &self,
        features: PredictionInput,
    ) -> Result<(Vec<f64>, f64), ModelError> {
        let transformed_features = self
            .preprocess_prediction_input(features)
            .map_err(|err| ModelError::PredictionError(err.to_string()))?;

        if !self.parameter.solver_type().is_logistic_regression() {
            return Err(ModelError::PredictionError(
                "Probability output is only supported for logistic regression".to_owned(),
            ));
        }

        unsafe {
            let mut output_probabilities = vec![0f64; (*self.bound).nr_class as usize];

            let best_class = ffi::predict_values(
                self.bound,
                transformed_features.as_ptr(),
                output_probabilities.as_mut_ptr(),
            );
            Ok((output_probabilities, best_class))
        }
    }

    fn feature_coefficient(&self, feature_index: i32, label_index: i32) -> f64 {
        unsafe { ffi::get_decfun_coef(self.bound, feature_index, label_index) }
    }

    fn label_bias(&self, label_index: i32) -> Result<f64, ModelError> {
        if self.parameter().solver_type().is_one_class() {
            return Err(ModelError::InvalidState(
                "Label bias value unavailable for one-class models".to_owned(),
            ));
        }

        unsafe { Ok(ffi::get_decfun_bias(self.bound, label_index)) }
    }

    fn rho(&self) -> Result<f64, ModelError> {
        if !self.parameter().solver_type().is_one_class() {
            return Err(ModelError::InvalidState(
                "Rho value unavailable for non-one-class models".to_owned(),
            ));
        }

        unsafe { Ok(ffi::get_decfun_rho(self.bound)) }
    }

    fn bias(&self) -> f64 {
        unsafe { (*self.bound).bias }
    }

    fn labels(&self) -> &Vec<i32> {
        &self.backing_store_labels
    }

    fn num_classes(&self) -> usize {
        unsafe { (*self.bound).nr_class as usize }
    }

    fn num_features(&self) -> usize {
        unsafe { (*self.bound).nr_feature as usize }
    }

    fn save_to_disk(&self, file_path: &str) -> Result<(), ModelError> {
        unsafe {
            let file_path_cstr = CString::new(file_path).unwrap();
            let result = ffi::save_model(file_path_cstr.as_ptr(), self.bound);
            if result == -1 {
                return Err(ModelError::SerializationError(
                    "save_model() returned -1".to_owned(),
                ));
            }
        }

        Ok(())
    }
}

impl LibLinearCrossValidator for Model {
    fn cross_validation(&self, folds: i32) -> Result<Vec<f64>, ModelError> {
        if folds < 2 {
            return Err(ModelError::IllegalArgument(
                "Number of folds must be >= 2 for cross validator".to_owned(),
            ));
        } else if self.problem.is_none() {
            return Err(ModelError::InvalidState(
                "Invalid problem/parameters for cross validator".to_owned(),
            ));
        }

        unsafe {
            let mut output_labels = vec![0f64; self.problem.as_ref().unwrap().ffi_obj().l as usize];

            ffi::cross_validation(
                self.problem.as_ref().unwrap().ffi_obj(),
                self.parameter.ffi_obj(),
                folds,
                output_labels.as_mut_ptr(),
            );
            Ok(output_labels)
        }
    }

    fn find_optimal_constraints_violation_cost_and_loss_sensitivity(
        &self,
        folds: i32,
        start_cost: f64,
        start_loss_sensitivity: f64,
    ) -> Result<(f64, f64, f64), ModelError> {
        if folds < 2 {
            return Err(ModelError::IllegalArgument(
                "Number of folds must be >= 2 for cross validator".to_owned(),
            ));
        } else if self.problem.is_none() {
            return Err(ModelError::InvalidState(
                "Invalid problem/parameters for cross validator".to_owned(),
            ));
        }

        unsafe {
            let mut best_cost = 0f64;
            let mut best_rate = 0f64;
            let mut best_loss_sensitivity = 0f64;
            ffi::find_parameters(
                self.problem.as_ref().unwrap().ffi_obj(),
                self.parameter.ffi_obj(),
                folds,
                start_cost,
                start_loss_sensitivity,
                &mut best_cost,
                &mut best_loss_sensitivity,
                &mut best_rate,
            );
            Ok((best_cost, best_rate, best_loss_sensitivity))
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            let mut temp = self.bound;
            ffi::free_and_destroy_model(&mut temp);
        }
    }
}
