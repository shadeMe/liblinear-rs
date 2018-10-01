//! # liblinear
//!
//! `liblinear` is a Rust wrapper for the [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
//! C++ machine learning library.

#[macro_use]
extern crate failure;
extern crate num;
#[macro_use]
extern crate num_derive;

use failure::Error;
pub use ffi::FeatureNode;
use num::FromPrimitive;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use util::*;


mod ffi;
pub mod util;

/// Errors related to a model's parameters.
#[derive(Debug, Fail)]
pub enum ParameterError {
	/// The model's parameters are either incomplete or invalid.
    #[fail(display = "parameter error: {}", e)]
	InvalidParameters { #[doc(hidden)] e: String },
}

/// Errors related to a model's input/training data.
#[derive(Debug, Fail)]
pub enum ProblemError {
	/// The model's input/training data is either incomplete or invalid.
    #[fail(display = "input data error: {}", e)]
	InvalidTrainingData { #[doc(hidden)] e: String },
}

/// Errors raised by a model's API call.
#[derive(Debug, Fail)]
pub enum ModelError {
	/// The model's internal state is invalid.
	///
	/// This can occur if the model's parameters or input data were not initialized correctly.
    #[fail(display = "invalid state: {}", e)]
	InvalidState { #[doc(hidden)] e: String },
	/// The model cannot be saved to/loaded from disk.
	///
	/// This can occur if the serialized data was not found, or if the model is in an indeterminate
	/// state after deserialization.
    #[fail(display = "serialization error: {}", e)]
	SerializationError { #[doc(hidden)] e: String },
	/// The model couldn't be applied to the test/prediction data.
	#[fail(display = "prediction error: {}", e)]
	PredictionError { #[doc(hidden)] e: String },
	/// The model encountered an unexpected internal error.
    #[fail(display = "unknown error: {}", e)]
	UnknownError { #[doc(hidden)] e: String },
}

/// Represents a one-to-one mapping of source features to target values.
///
/// Source features are represented as sparse vectors of real numbers. Target values are
/// either integers (in classification) or real numbers (in regression).
pub trait LibLinearProblem: Clone {
	/// The feature vectors of each training instance.
	fn source_features(&self) -> &Vec<Vec<FeatureNode>>;
	/// Target labels/values of each training instance.
	fn target_values(&self) -> &Vec<f64>;
	/// Bias of the input data.
	fn bias(&self) -> f64;
}

#[doc(hidden)]
pub struct Problem {
    backing_store_labels: Vec<f64>,
	backing_store_features: Vec<Vec<FeatureNode>>,
	backing_store_feature_ptrs: Vec<*const FeatureNode>,
    bound: ffi::Problem,
}

impl Problem {
    fn new(input_data: TrainingInput, bias: f64) -> Result<Problem, ParameterError> {
        let num_training_instances = input_data.len_data() as i32;
        let num_features = input_data.len_features() as i32;
        let has_bias = bias >= 0f64;
        let last_feature_index = input_data.last_feature_index() as i32;

	    let (mut transformed_features, labels): (Vec<Vec<FeatureNode>>, Vec<f64>) =
            input_data.yield_data().iter().fold(
                (
	                Vec::<Vec<FeatureNode>>::default(),
	                Vec::<f64>::default(),
                ),
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
        Ok(Problem {
            bound: ffi::Problem {
                l: num_training_instances as i32,
                n: num_features + if has_bias { 1 } else { 0 } as i32,
                y: labels.as_ptr(),
                x: transformed_feature_ptrs.as_ptr(),
                bias,
            },
            backing_store_labels: labels,
            backing_store_features: transformed_features,
            backing_store_feature_ptrs: transformed_feature_ptrs,
        })
    }
}

impl LibLinearProblem for Problem {
	fn source_features(&self) -> &Vec<Vec<FeatureNode>> {
		&self.backing_store_features
	}

	fn target_values(&self) -> &Vec<f64> {
		&self.backing_store_labels
	}

	fn bias(&self) -> f64 {
		self.bound.bias
	}
}

impl Clone for Problem {
    fn clone(&self) -> Self {
	    let labels = self.backing_store_labels.clone();
	    let transformed_features: Vec<Vec<FeatureNode>> = self
            .backing_store_features
            .iter()
            .map(|e| (*e).clone())
            .collect();
	    let transformed_feature_ptrs: Vec<*const FeatureNode> =
            transformed_features.iter().map(|e| e.as_ptr()).collect();

        Problem {
            bound: ffi::Problem {
                l: self.bound.l,
                n: self.bound.n,
                y: labels.as_ptr(),
                x: transformed_feature_ptrs.as_ptr(),
                bias: self.bound.bias,
            },
            backing_store_labels: labels,
            backing_store_features: transformed_features,
            backing_store_feature_ptrs: transformed_feature_ptrs,
        }
    }
}

/// Builder for [LibLinearProblem](enum.LibLinearProblem.html).
pub struct ProblemBuilder {
    input_data: Option<TrainingInput>,
    bias: f64,
}

impl ProblemBuilder {
    fn new() -> ProblemBuilder {
        ProblemBuilder {
            input_data: None,
	        bias: -1.0,
        }
    }

	/// Set input/training data.
	pub fn input_data(&mut self, input_data: TrainingInput) -> &mut Self {
        self.input_data = Some(input_data);
        self
    }
	/// Set bias. If bias is >= 0, it's appended to the feature vector for every instance.
	///
	/// Default: -1.0
    pub fn bias(&mut self, bias: f64) -> &mut Self {
        self.bias = bias;
        self
    }
    fn build(self) -> Result<Problem, Error> {
	    let input_data = self.input_data.ok_or(ProblemError::InvalidTrainingData {
            e: "Missing input/training data".to_string(),
        })?;

        Ok(Problem::new(input_data, self.bias)?)
    }
}

/// Types of generalized linear models supported by liblinear.
///
/// These combine several types of regularization schemes:
/// * L1
/// * L2
///
/// ...and loss functions:
/// * L1-loss for SVM
/// * Regular L2-loss for SVM (hinge-loss)
/// * Logistic loss for logistic regression
#[derive(FromPrimitive)]
#[allow(non_camel_case_types)]
pub enum SolverType {
	/// L2-regularized logistic regression (primal).
	L2R_LR = 0,
	/// L2-regularized L2-loss support vector classification (dual).
	L2R_L2LOSS_SVC_DUAL = 1,
	/// L2-regularized L2-loss support vector classification (primal).
	L2R_L2LOSS_SVC = 2,
	/// L2-regularized L1-loss support vector classification (dual).
	L2R_L1LOSS_SVC_DUAL = 3,
	/// Support vector classification by Crammer and Singer.
	MCSVM_CS = 4,
	/// L1-regularized L2-loss support vector classification.
	L1R_L2LOSS_SVC = 5,
	/// L1-regularized logistic regression.
	L1R_LR = 6,
	/// L2-regularized logistic regression (dual).
	L2R_LR_DUAL = 7,
	/// L2-regularized L2-loss support vector regression (primal).
	L2R_L2LOSS_SVR = 11,
	/// L2-regularized L2-loss support vector regression (dual).
	L2R_L2LOSS_SVR_DUAL = 12,
	/// L2-regularized L1-loss support vector regression (dual).
	L2R_L1LOSS_SVR_DUAL = 13,
}

impl SolverType {
	/// Returns true if the solver is a probabilistic/logistic regression solver.
	///
	/// Supported solvers: L2R_LR, L1R_LR, L2R_LR_DUAL.
	pub fn is_logistic_regression(&self) -> bool {
		match self {
			L2R_LR => true,
			L1R_LR => true,
			L2R_LR_DUAL => true,
			_ => false
		}
	}
	/// Returns true if the solver is a support vector regression solver.
	///
	/// Supported solvers: L2R_L2LOSS_SVR, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL.
	pub fn is_support_vector_regression(&self) -> bool {
		match self {
			L2R_L2LOSS_SVR => true,
			L2R_L2LOSS_SVR_DUAL => true,
			L2R_L1LOSS_SVR_DUAL => true,
			_ => false
		}
	}
	/// Returns true if the solver supports multi-class classification.
	///
	/// Supported solvers: All non-SVR solvers.
	pub fn is_multi_class_classification(&self) -> bool {
		!self.is_support_vector_regression()
	}
}

impl Default for SolverType {
	/// Default: L2R_LR
	fn default() -> Self {
		SolverType::L2R_LR
	}
}

/// Represents the tunable parameters of a model.
pub trait LibLinearParameter: Clone {
	/// Solver used for classification or regression.
    fn solver_type(&self) -> SolverType;
	/// Tolerance of termination criterion for optimization (parameter _e_).
    fn stopping_criterion(&self) -> f64;
	/// Cost of constraints violation (parameter _C_).
	///
	/// Rules the trade-off between regularization and correct classification on data.
	/// It can be seen as the inverse of a regularization constant.
    fn constraints_violation_cost(&self) -> f64;
	/// Sensitivity of loss of support vector regression (parameter _p_).
    fn regression_loss_sensitivity(&self) -> f64;
}

#[doc(hidden)]
pub struct Parameter {
	backing_store_class_cost_penalty_weights: Vec<f64>,
	backing_store_class_cost_penalty_labels: Vec<i32>,
    backing_store_starting_solutions: Vec<f64>,
    bound: ffi::Parameter,
}

impl Parameter {
    fn new(
	    solver: SolverType,
	    eps: f64,
	    cost: f64,
	    p: f64,
	    cost_penalty_weights: Vec<f64>,
	    cost_penalty_labels: Vec<i32>,
	    init_solutions: Vec<f64>,
    ) -> Result<Parameter, ParameterError> {
	    if cost_penalty_weights.len() != cost_penalty_labels.len() {
            return Err(ParameterError::InvalidParameters {
	            e: "Mismatch between cost penalty weights and labels".to_string(),
            });
        }

	    let num_weights = cost_penalty_weights.len() as i32;

	    let param = Parameter {
            bound: ffi::Parameter {
                solver_type: solver as i32,
                eps,
                C: cost,
                nr_weight: num_weights,
	            weight_label: if cost_penalty_labels.is_empty() {
		            ptr::null()
	            } else {
		            cost_penalty_labels.as_ptr()
	            },
	            weight: if cost_penalty_weights.is_empty() {
		            ptr::null()
	            } else {
		            cost_penalty_weights.as_ptr()
	            },
                p,
	            init_sol: if init_solutions.is_empty() {
                    ptr::null()
                } else {
		            init_solutions.as_ptr()
                },
            },
		    backing_store_class_cost_penalty_weights: cost_penalty_weights,
		    backing_store_class_cost_penalty_labels: cost_penalty_labels,
		    backing_store_starting_solutions: init_solutions,
        };

        unsafe {
            let param_error = ffi::check_parameter(ptr::null(), &param.bound);
            if !param_error.is_null() {
                return Err(ParameterError::InvalidParameters {
                    e: CStr::from_ptr(param_error)
                        .to_string_lossy()
                        .to_owned()
                        .to_string(),
                });
            }
        }

        Ok(param)
    }
}

impl LibLinearParameter for Parameter {
    fn solver_type(&self) -> SolverType {
        num::FromPrimitive::from_i32(self.bound.solver_type).unwrap()
    }
    fn stopping_criterion(&self) -> f64 {
        self.bound.eps
    }
    fn constraints_violation_cost(&self) -> f64 {
        self.bound.C
    }
    fn regression_loss_sensitivity(&self) -> f64 {
        self.bound.p
    }
}

impl Clone for Parameter {
    fn clone(&self) -> Self {
	    let weights = self.backing_store_class_cost_penalty_weights.clone();
	    let weight_labels = self.backing_store_class_cost_penalty_labels.clone();
        let init_sol = self.backing_store_starting_solutions.clone();

        Parameter {
            bound: ffi::Parameter {
                solver_type: self.bound.solver_type as i32,
                eps: self.bound.eps,
                C: self.bound.C,
                nr_weight: self.bound.nr_weight,
                weight_label: weight_labels.as_ptr(),
                weight: weights.as_ptr(),
                p: self.bound.p,
                init_sol: init_sol.as_ptr(),
            },
	        backing_store_class_cost_penalty_weights: weights,
	        backing_store_class_cost_penalty_labels: weight_labels,
            backing_store_starting_solutions: init_sol,
        }
    }
}

/// Builder for [LibLinearParameter](enum.LibLinearParameter.html).
pub struct ParameterBuilder {
    solver_type: SolverType,
    epsilon: f64,
    cost: f64,
    p: f64,
	cost_penalty_weights: Vec<f64>,
	cost_penalty_labels: Vec<i32>,
    init_solutions: Vec<f64>,
}

impl ParameterBuilder {
    fn new() -> ParameterBuilder {
        ParameterBuilder {
	        solver_type: SolverType::default(),
	        epsilon: 0.01,
	        cost: 1.0,
	        p: 0.1,
	        cost_penalty_weights: Vec::new(),
	        cost_penalty_labels: Vec::new(),
            init_solutions: Vec::new(),
        }
    }

	/// Set solver type.
	///
	/// Default: [L2R_LR](enum.SolverType.html#variant.L2R_LR)
    pub fn solver_type(&mut self, solver_type: SolverType) -> &mut Self {
        self.solver_type = solver_type;
        self
    }
	/// Set tolerance of termination criterion.
	///
	/// Default: 0.01
	pub fn stopping_criterion(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }
	/// Set cost of constraints violation.
	///
	/// Default: 1.0
	pub fn constraints_violation_cost(&mut self, cost: f64) -> &mut Self {
        self.cost = cost;
        self
    }

	/// Set tolerance margin in regression loss function of SVR. Not used for classification problems.
	///
	/// Default: 0.1
	pub fn regression_loss_sensitivity(&mut self, p: f64) -> &mut Self {
        self.p = p;
        self
    }
	/// Set weights to adjust the cost of constraints violation for specific classes.
	///
	/// Useful when training classifiers on unbalanced input data or with asymmetric mis-classification cost.
	pub fn cost_penalty_weights(&mut self, cost_penalty_weights: Vec<f64>) -> &mut Self {
		self.cost_penalty_weights = cost_penalty_weights;
        self
    }

	/// Set classes that correspond to the weights used to adjust the cost of constraints violation.
	///
	/// Each weight corresponds to a label at the same index.
	pub fn cost_penalty_labels(&mut self, cost_penalty_labels: Vec<i32>) -> &mut Self {
		self.cost_penalty_labels = cost_penalty_labels;
        self
    }
	/// Set initial solution specification for solvers L2R_LR and/or L2R_L2LOSSES_SVC.
    pub fn initial_solutions(&mut self, init_solutions: Vec<f64>) -> &mut Self {
        self.init_solutions = init_solutions;
        self
    }

    fn build(self) -> Result<Parameter, Error> {
        Ok(Parameter::new(
	        self.solver_type,
	        self.epsilon,
	        self.cost,
	        self.p,
	        self.cost_penalty_weights,
	        self.cost_penalty_labels,
	        self.init_solutions,
        )?)
    }
}

/// Super-trait of [LibLinearModel](trait.LibLinearModel.html) and [LibLinearCrossValidator](trait.LibLinearCrossValidator.html)
pub trait HasLibLinearProblem {
	type Output: LibLinearProblem;
	/// The problem associated with the model/cross-validator.
	///
	/// This will return `None` when called on a model that was deserialized/loaded from disk.
	fn problem(&self) -> Option<&Self::Output>;
}

/// Super-trait of [LibLinearModel](trait.LibLinearModel.html) and [LibLinearCrossValidator](trait.LibLinearCrossValidator.html)
pub trait HasLibLinearParameter {
	type Output: LibLinearParameter;
	/// The parameters of the model/cross-validator.
	fn parameter(&self) -> &Self::Output;
}

/// Represents a linear model that can be used for prediction.
pub trait LibLinearModel: HasLibLinearProblem + HasLibLinearParameter {
	/// Returns one of the following values:
	///
	/// * For a classification model, the predicted class is returned.
	/// * For a regression model, the function value of x calculated using the model is returned.
	fn predict(&self, features: PredictionInput) -> Result<f64, ModelError>;

	/// Returns a tuple of the following values:
	///
	/// * A list of decision values. If k is the number of classes, each element includes results
	/// of predicting k binary-class SVMs. If k=2 and solver is not MCSVM_CS, only one decision value
	///	is returned.
	///
	///   The values correspond to the classes returned by the `labels` method.
	/// * The class with the highest decision value.
	fn predict_values(
		&self,
		features: PredictionInput,
	) -> Result<(Vec<f64>, f64), ModelError>;

	/// Returns a tuple of the following values:
	///
	/// * A list of probability estimates. each element contains k values
	/// indicating the probability that the testing instance is in each class.
	///
	///   The values correspond to the classes returned by the `labels` method.
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
	/// Note that while feature indices start from 1, label indices start from 0.
	/// If the feature index is not in the valid range, a zero value will be returned.
	///
	/// For classification models, if the label index is not in the valid range, a zero value will be returned.
	/// For regression models, the label index is ignored.
    fn feature_coefficient(&self, feature_index: i32, label_index: i32) -> f64;

	/// Returns the bias term corresponding to the class with the given index
	///
	/// For classification models, if label index is not in a valid range, a zero value will be returned.
	/// For regression models, the label index is ignored.
    fn label_bias(&self, label_index: i32) -> f64;

	/// Returns the bias of the input data with which the model was trained.
    fn bias(&self) -> f64;

	/// Returns the labels/classes learned by the model.
	fn labels(&self) -> &Vec<i32>;

	/// Returns the number of classes of the model.
	///
	/// For regression models, 2 is returned.
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
	/// Number of folds must be >= 2.
    fn cross_validation(&self, folds: i32) -> Result<Vec<f64>, ModelError>;

	/// Performs k-folds cross-validation to find the best cost value (parameter _C_) within the
	/// closed search range `(start_C, end_C)` and returns a tuple of the following values:
	///
	/// * The best cost value.
	/// * The accuracy of the best cost value.
	///
	/// Cross validation is conducted many times under the following values of _C_:
	/// * `start_C`
	/// * 2 * `start_C`
	/// * 4 * `start_C`
	/// * 8 * `start_C`, and so on
	///
	/// ...and finds the best one with the highest cross validation accuracy. The procedure stops when
	/// the models of all folds become stable or the cost reaches `end_C`.
	///
	/// If `start_C` is <= 0, an appropriately small value is automatically calculated and used instead.
    fn find_optimal_constraints_violation_cost(
        &self,
        folds: i32,
        search_range: (f64, f64),
    ) -> Result<(f64, f64), ModelError>;
}

#[doc(hidden)]
struct Model {
    problem: Option<Problem>,
	parameter: Parameter,
	backing_store_labels: Vec<i32>,
    bound: *mut ffi::Model,
}

impl Model {
    fn from_input(
        problem: Problem,
        parameter: Parameter,
        train: bool,
    ) -> Result<Model, ModelError> {
        let mut bound: *mut ffi::Model = ptr::null_mut();
        if train {
            bound = unsafe { ffi::train(&problem.bound, &parameter.bound) };
            if bound.is_null() {
                return Err(ModelError::UnknownError {
                    e: "train() returned a NULL pointer".to_owned().to_string(),
                });
            }
        }

	    let mut backing_store_labels = Vec::<i32>::new();
	    unsafe {
		    for i in 0..(*bound).nr_class {
			    backing_store_labels.push(*(*bound).label.offset(i as isize));
		    }
	    }

        Ok(Model {
            problem: Some(problem),
	        parameter,
	        backing_store_labels,
            bound,
        })
    }

    fn from_serialized_file(path_to_serialized_model: &str) -> Result<Model, ModelError> {
	    unsafe {
		    let file_path_cstr = CString::new(path_to_serialized_model).unwrap();
		    let bound = ffi::load_model(file_path_cstr.as_ptr());

		    if bound.is_null() {
			    return Err(ModelError::SerializationError {
				    e: "load_model() returned a NULL pointer"
					    .to_owned()
					    .to_string(),
			    });
		    }

		    let mut backing_store_labels = Vec::<i32>::new();
		    for i in 0..(*bound).nr_class {
			    backing_store_labels.push(*(*bound).label.offset(i as isize));
		    }

		    Ok(Model {
			    problem: None,
			    // solver_type is the only parameter that's serialized to disk
			    // init the parameter object with just that and pass the defaults for the rest
			    parameter: Parameter::new(
				    num::FromPrimitive::from_i32((*bound).param.solver_type).unwrap(),
				    0.01,
				    1.0,
				    0.1,
				    Vec::new(),
				    Vec::new(),
				    Vec::new(),
			    ).unwrap(),
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
		    return Err(PredictionInputError::DataError {
			    e: format!(
				    "Expected {} features, found {} instead",
				    self.num_features(),
				    last_feature_index
			    ).to_string(),
		    });
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

	fn predict_values(
		&self,
		features: PredictionInput,
	) -> Result<(Vec<f64>, f64), ModelError> {
		let transformed_features = self.preprocess_prediction_input(features).map_err(|err| ModelError::PredictionError {
			e: format!("{}", err).to_string()
		})?;
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
		let transformed_features = self.preprocess_prediction_input(features).map_err(|err| ModelError::PredictionError {
			e: format!("{}", err).to_string()
		})?;

		if !self.parameter.solver_type().is_logistic_regression() {
			return Err(ModelError::PredictionError {
				e: "Probability output is only supported for logistic regression".to_string()
			});
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

    fn label_bias(&self, label_index: i32) -> f64 {
        unsafe { ffi::get_decfun_bias(self.bound, label_index) }
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
				return Err(ModelError::SerializationError {
					e: "save_model() returned -1".to_owned().to_string(),
				});
			}
		}

		Ok(())
	}
}

impl LibLinearCrossValidator for Model {
    fn cross_validation(&self, folds: i32) -> Result<Vec<f64>, ModelError> {
	    assert!(folds >= 2);

	    if self.problem.is_none() {
            return Err(ModelError::InvalidState {
                e: "Invalid problem/parameters for cross validator"
                    .to_owned()
                    .to_string(),
            });
        }

        unsafe {
            let mut output_labels = vec![0f64; self.problem.as_ref().unwrap().bound.l as usize];

            ffi::cross_validation(
	            &self.problem.as_ref().unwrap().bound,
	            &self.parameter.bound,
	            folds,
	            output_labels.as_mut_ptr(),
            );
            Ok(output_labels)
        }
    }

    fn find_optimal_constraints_violation_cost(
        &self,
        folds: i32,
        search_range: (f64, f64),
    ) -> Result<(f64, f64), ModelError> {
	    assert!(folds >= 2);

	    if self.problem.is_none() {
            return Err(ModelError::InvalidState {
                e: "Invalid problem/parameters for cross validator"
                    .to_owned()
                    .to_string(),
            });
        }

        unsafe {
            let mut best_cost = 0f64;
            let mut best_rate = 0f64;
            ffi::find_parameter_C(
	            &self.problem.as_ref().unwrap().bound,
	            &self.parameter.bound,
	            folds,
	            search_range.0,
	            search_range.1,
	            &mut best_cost,
	            &mut best_rate,
            );
            Ok((best_cost, best_rate))
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


/// Primary model builder. Functions as the entry point into the API.
pub struct Builder {
    problem_builder: ProblemBuilder,
    parameter_builder: ParameterBuilder,
}

impl Builder {
	/// Creates a new instance of the builder.
    pub fn new() -> Builder {
        Builder {
            problem_builder: ProblemBuilder::new(),
            parameter_builder: ParameterBuilder::new(),
        }
    }

	/// Builder for the model's linear problem.
    pub fn problem(&mut self) -> &mut ProblemBuilder {
        &mut self.problem_builder
    }
	/// Builder for the model's tunable parameters.
    pub fn parameters(&mut self) -> &mut ParameterBuilder {
        &mut self.parameter_builder
    }
	/// Builds a [LibLinearCrossValidator](trait.LibLinearCrossValidator.html) instance with the given problem and parameters.
    pub fn build_cross_validator(self) -> Result<impl LibLinearCrossValidator, Error> {
        Ok(Model::from_input(
            self.problem_builder.build()?,
            self.parameter_builder.build()?,
            false,
        )?)
    }
	/// Builds a [LibLinearModel](trait.LibLinearModel.html) instance with the given problem and parameters.
    pub fn build_model(self) -> Result<impl LibLinearModel, Error> {
        Ok(Model::from_input(
            self.problem_builder.build()?,
            self.parameter_builder.build()?,
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
    pub fn load_model(path_to_serialized_model: &str) -> Result<impl LibLinearModel, Error> {
        Ok(Model::from_serialized_file(path_to_serialized_model)?)
    }

	/// Saves a model to disk.
	///
	/// Convenience method that calls `save_to_disk` on the model instance.
    pub fn save_model(
	    path_to_serialized_model: &str,
	    model: &impl LibLinearModel,
    ) -> Result<(), Error> {
        Ok(model.save_to_disk(path_to_serialized_model)?)
    }
}

/// The version of the bundled liblinear C-library.
pub fn liblinear_version() -> i32 {
    unsafe { ffi::liblinear_version }
}
