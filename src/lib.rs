#[macro_use]
extern crate failure;
extern crate num;
#[macro_use]
extern crate num_derive;

use failure::Error;
pub use ffi::SolverType;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use util::*;

mod ffi;
pub mod metrics;
pub mod util;

#[derive(Debug, Fail)]
pub enum ParameterError {
    /// Invalid/incomplete parameter values
    #[fail(display = "parameter error: {}", e)]
    InvalidParameters { e: String },
}

#[derive(Debug, Fail)]
pub enum ProblemError {
    /// Invalid/missing training input
    #[fail(display = "input data error: {}", e)]
    InvalidTrainingData { e: String },
}

#[derive(Debug, Fail)]
pub enum ModelError {
    /// Invalid/missing internal state
    #[fail(display = "invalid state: {}", e)]
    InvalidState { e: String },
    /// Missing file, invalid serialized data
    #[fail(display = "serialization error: {}", e)]
    SerializationError { e: String },
    /// Unexpected/unspecified error
    #[fail(display = "unknown error: {}", e)]
    UnknownError { e: String },
}

pub trait LibLinearProblem: Clone {}

#[doc(hidden)]
pub struct Problem {
    backing_store_labels: Vec<f64>,
    backing_store_features: Vec<Vec<ffi::FeatureNode>>,
    backing_store_feature_ptrs: Vec<*const ffi::FeatureNode>,
    bound: ffi::Problem,
}

impl Problem {
    fn new(input_data: TrainingInput, bias: f64) -> Result<Problem, ParameterError> {
        let num_training_instances = input_data.len_data() as i32;
        let num_features = input_data.len_features() as i32;
        let has_bias = bias >= 0f64;
        let last_feature_index = input_data.last_feature_index() as i32;

        let (mut transformed_features, labels): (Vec<Vec<ffi::FeatureNode>>, Vec<f64>) =
            input_data.yield_data().iter().fold(
                (
                    Vec::<Vec<ffi::FeatureNode>>::default(),
                    Vec::<f64>::default(),
                ),
                |(mut feats, mut labels), instance| {
                    feats.push(
                        instance
                            .features()
                            .iter()
                            .map(|(index, value)| ffi::FeatureNode {
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
            .map(|mut v: Vec<ffi::FeatureNode>| {
                if has_bias {
                    v.push(ffi::FeatureNode {
	                    index: last_feature_index + 1,
                        value: bias,
                    });
                }

                v.push(ffi::FeatureNode {
                    index: -1,
                    value: 0f64,
                });

                v
            })
            .collect();

	    let transformed_feature_ptrs: Vec<*const ffi::FeatureNode> =
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

impl LibLinearProblem for Problem {}

impl Clone for Problem {
    fn clone(&self) -> Self {
	    let labels = self.backing_store_labels.clone();
	    let transformed_features: Vec<Vec<ffi::FeatureNode>> = self
            .backing_store_features
            .iter()
            .map(|e| (*e).clone())
            .collect();
	    let transformed_feature_ptrs: Vec<*const ffi::FeatureNode> =
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
	pub fn input_data(&mut self, input_data: TrainingInput) -> &mut Self {
        self.input_data = Some(input_data);
        self
    }
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

pub trait LibLinearParameter: Clone {
    fn solver_type(&self) -> SolverType;
    fn stopping_criterion(&self) -> f64;
    fn constraints_violation_cost(&self) -> f64;
    fn regression_loss_sensitivity(&self) -> f64;
}

#[doc(hidden)]
pub struct Parameter {
    backing_store_weights: Vec<f64>,
    backing_store_weight_labels: Vec<i32>,
    backing_store_starting_solutions: Vec<f64>,
    bound: ffi::Parameter,
}

impl Parameter {
    fn new(
        solver: SolverType,
        eps: f64,
        cost: f64,
        p: f64,
        weights: Vec<f64>,
        weight_labels: Vec<i32>,
        init_sol: Vec<f64>,
    ) -> Result<Parameter, ParameterError> {
	    if !init_sol.is_empty() && !weights.is_empty() && weights.len() != init_sol.len() {
            return Err(ParameterError::InvalidParameters {
                e: "Mismatch between number of initial solutions and weights".to_string(),
            });
        }

        let num_weights = weights.len() as i32;

	    let param = Parameter {
            bound: ffi::Parameter {
                solver_type: solver as i32,
                eps,
                C: cost,
                nr_weight: num_weights,
	            weight_label: if weight_labels.is_empty() {
		            ptr::null()
	            } else {
		            weight_labels.as_ptr()
	            },
	            weight: if weights.is_empty() {
		            ptr::null()
	            } else {
		            weights.as_ptr()
	            },
                p,
                init_sol: if init_sol.is_empty() {
                    ptr::null()
                } else {
                    init_sol.as_ptr()
                },
            },
            backing_store_weights: weights,
            backing_store_weight_labels: weight_labels,
            backing_store_starting_solutions: init_sol,
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
        let weights = self.backing_store_weights.clone();
        let weight_labels = self.backing_store_weight_labels.clone();
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
            backing_store_weights: weights,
            backing_store_weight_labels: weight_labels,
            backing_store_starting_solutions: init_sol,
        }
    }
}

pub struct ParameterBuilder {
    solver_type: SolverType,
    epsilon: f64,
    cost: f64,
    p: f64,
    weights: Vec<f64>,
    weight_labels: Vec<i32>,
    init_solutions: Vec<f64>,
}

impl ParameterBuilder {
    fn new() -> ParameterBuilder {
        ParameterBuilder {
            solver_type: SolverType::L2R_LR,
	        epsilon: 0.01,
	        cost: 1.0,
	        p: 0.1,
            weights: Vec::new(),
            weight_labels: Vec::new(),
            init_solutions: Vec::new(),
        }
    }

    pub fn solver_type(&mut self, solver_type: SolverType) -> &mut Self {
        self.solver_type = solver_type;
        self
    }

    pub fn epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    pub fn cost(&mut self, cost: f64) -> &mut Self {
        self.cost = cost;
        self
    }

    pub fn loss_sensitivity(&mut self, p: f64) -> &mut Self {
        self.p = p;
        self
    }

    pub fn weights(&mut self, weights: Vec<f64>) -> &mut Self {
        self.weights = weights;
        self
    }

    pub fn weight_labels(&mut self, weight_labels: Vec<i32>) -> &mut Self {
        self.weight_labels = weight_labels;
        self
    }

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
            self.weights,
            self.weight_labels,
            self.init_solutions,
        )?)
    }
}

pub enum ModelType {
    CLASSIFICATION,
    REGRESSION,
}

pub trait HasLibLinearProblem {
	type Output: LibLinearProblem;
	fn problem(&self) -> Option<&Self::Output>;
}

pub trait HasLibLinearParameter {
	type Output: LibLinearParameter;
	fn parameter(&self) -> Option<&Self::Output>;
}

pub trait LibLinearModel: HasLibLinearProblem + HasLibLinearParameter {
    fn model_type(&self) -> Result<ModelType, ModelError>;

	fn predict(&self, features: PredictionInput) -> Result<f64, PredictionInputError>;
	fn predict_values(
		&self,
		features: PredictionInput,
	) -> Result<(f64, Vec<f64>), PredictionInputError>;
	fn predict_probabilities(
		&self,
		features: PredictionInput,
	) -> Result<(f64, Vec<f64>), PredictionInputError>;

    fn feature_coefficient(&self, feature_index: i32, label_index: i32) -> f64;
    fn label_bias(&self, label_index: i32) -> f64;
    fn bias(&self) -> f64;
    fn num_classes(&self) -> usize;
    fn num_features(&self) -> usize;

	fn save_to_disk(&self, file_path: &str) -> Result<(), ModelError>;
}

pub trait LibLinearCrossValidator: HasLibLinearProblem + HasLibLinearParameter {
    fn cross_validation(&self, folds: i32) -> Result<Vec<f64>, ModelError>;
    fn find_optimal_constraints_violation_cost(
        &self,
        folds: i32,
        search_range: (f64, f64),
    ) -> Result<(f64, f64), ModelError>;
}

struct Model {
    problem: Option<Problem>,
    parameter: Option<Parameter>,
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

        Ok(Model {
            problem: Some(problem),
            parameter: Some(parameter),
            bound,
        })
    }

    fn from_serialized_file(path_to_serialized_model: &str) -> Result<Model, ModelError> {
        let bound =
            unsafe { ffi::load_model(CString::new(path_to_serialized_model).unwrap().as_ptr()) };
        if bound.is_null() {
            return Err(ModelError::SerializationError {
                e: "load_model() returned a NULL pointer"
                    .to_owned()
                    .to_string(),
            });
        }

        Ok(Model {
            problem: None,
            parameter: None,
            bound,
        })
    }

    fn preprocess_prediction_input(
	    &self,
	    prediction_input: PredictionInput,
    ) -> Result<Vec<ffi::FeatureNode>, PredictionInputError> {
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
        let mut data: Vec<ffi::FeatureNode> = prediction_input
            .yield_data()
            .iter()
            .map(|(index, value)| ffi::FeatureNode {
                index: *index as i32,
                value: *value,
            })
            .collect();

        if has_bias {
            data.push(ffi::FeatureNode {
	            index: last_feature_index + 1,
                value: bias,
            });
        }

        data.push(ffi::FeatureNode {
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

	fn parameter(&self) -> Option<&Self::Output> {
		self.parameter.as_ref()
    }
}

impl LibLinearModel for Model {
    fn model_type(&self) -> Result<ModelType, ModelError> {
        unsafe {
            if !self.bound.is_null() {
                if ffi::check_probability_model(self.bound) == 1 {
                    return Ok(ModelType::CLASSIFICATION);
                } else if ffi::check_regression_model(self.bound) == 1 {
                    return Ok(ModelType::REGRESSION);
                }
            }
        }

        Err(ModelError::InvalidState {
            e: "unknown model type".to_owned().to_string(),
        })
    }

	fn predict(&self, features: PredictionInput) -> Result<f64, PredictionInputError> {
		Ok(self.predict_values(features)?.0)
    }

	fn predict_values(
		&self,
		features: PredictionInput,
	) -> Result<(f64, Vec<f64>), PredictionInputError> {
		let transformed_features = self.preprocess_prediction_input(features)?;
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
	        Ok((best_class, output_values))
        }
    }

	fn predict_probabilities(
		&self,
		features: PredictionInput,
	) -> Result<(f64, Vec<f64>), PredictionInputError> {
		let transformed_features = self.preprocess_prediction_input(features)?;
        unsafe {
            let mut output_probabilities = vec![0f64; (*self.bound).nr_class as usize];

            let best_class = ffi::predict_values(
                self.bound,
                transformed_features.as_ptr(),
                output_probabilities.as_mut_ptr(),
            );
	        Ok((best_class, output_probabilities))
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
        if self.problem.is_none() || self.parameter.is_none() {
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
                &self.parameter.as_ref().unwrap().bound,
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
        if self.problem.is_none() || self.parameter.is_none() {
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
                &self.parameter.as_ref().unwrap().bound,
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

pub struct Builder {
    problem_builder: ProblemBuilder,
    parameter_builder: ParameterBuilder,
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            problem_builder: ProblemBuilder::new(),
            parameter_builder: ParameterBuilder::new(),
        }
    }

    pub fn problem(&mut self) -> &mut ProblemBuilder {
        &mut self.problem_builder
    }

    pub fn parameters(&mut self) -> &mut ParameterBuilder {
        &mut self.parameter_builder
    }

    pub fn build_cross_validator(self) -> Result<impl LibLinearCrossValidator, Error> {
        Ok(Model::from_input(
            self.problem_builder.build()?,
            self.parameter_builder.build()?,
            false,
        )?)
    }

    pub fn build_model(self) -> Result<impl LibLinearModel, Error> {
        Ok(Model::from_input(
            self.problem_builder.build()?,
            self.parameter_builder.build()?,
            true,
        )?)
    }
}

pub struct Serializer;

impl Serializer {
    pub fn load_model(path_to_serialized_model: &str) -> Result<impl LibLinearModel, Error> {
        Ok(Model::from_serialized_file(path_to_serialized_model)?)
    }

    pub fn save_model(
	    path_to_serialized_model: &str,
	    model: &impl LibLinearModel,
    ) -> Result<(), Error> {
        Ok(model.save_to_disk(path_to_serialized_model)?)
    }
}

pub fn liblinear_version() -> i32 {
    unsafe { ffi::liblinear_version }
}
