use failure::Error;
use ffi;
pub use ffi::SolverType;
use num;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use util::predict::*;
use util::train::*;

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

pub trait LibLinearProblem: Clone {
    fn bias(&self) -> f64;
    fn num_features(&self) -> usize;
}

#[doc(hidden)]
pub struct Problem {
    num_features: i32,
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
                        index: last_feature_index,
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

        let mut transformed_feature_ptrs: Vec<*const ffi::FeatureNode> =
            transformed_features.iter().map(|e| e.as_ptr()).collect();

        // the pointers passed to ffi::Problem will be valid even after their corresponding Vecs
        // are moved to a different location as they point to the actual backing store on the heap
        Ok(Problem {
            num_features,
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
    fn bias(&self) -> f64 {
        self.bound.bias
    }
    fn num_features(&self) -> usize {
        self.num_features as usize
    }
}

impl Clone for Problem {
    fn clone(&self) -> Self {
        let mut labels = self.backing_store_labels.clone();
        let mut transformed_features: Vec<Vec<ffi::FeatureNode>> = self
            .backing_store_features
            .iter()
            .map(|e| (*e).clone())
            .collect();
        let mut transformed_feature_ptrs: Vec<*const ffi::FeatureNode> =
            transformed_features.iter().map(|e| e.as_ptr()).collect();

        Problem {
            num_features: self.num_features,
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
            bias: 0.0,
        }
    }
	fn input_data(&mut self, input_data: TrainingInput) -> &mut Self {
		self.input_data = Some(input_data);
		self
	}
    pub fn bias(&mut self, bias: f64) -> &mut Self {
        self.bias = bias;
        self
    }
	fn build(self) -> Result<Problem, Error> {
		let mut input_data = self.input_data.ok_or(ProblemError::InvalidTrainingData {
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
	    C: f64,
	    p: f64,
	    weights: Vec<f64>,
	    weight_labels: Vec<i32>,
	    init_sol: Vec<f64>,
    ) -> Result<Parameter, ParameterError> {
        if weights.len() == 0 || weight_labels.len() == 0 {
	        return Err(ParameterError::InvalidParameters {
                e: "No weights/weight labels".to_string(),
            });
        } else if weights.len() != weight_labels.len() {
	        return Err(ParameterError::InvalidParameters {
                e: "Mismatch between number of labels and weights".to_string(),
            });
        } else if !init_sol.is_empty() && weights.len() != init_sol.len() {
	        return Err(ParameterError::InvalidParameters {
                e: "Mismatch between number of initial solutions and weights".to_string(),
            });
        }

        let num_weights = weights.len() as i32;

        let mut param = Parameter {
            bound: ffi::Parameter {
                solver_type: solver as i32,
                eps,
                C,
                nr_weight: num_weights,
                weight_label: weight_labels.as_ptr(),
                weight: weights.as_ptr(),
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
        let mut weights = self.backing_store_weights.clone();
        let mut weight_labels = self.backing_store_weight_labels.clone();
        let mut init_sol = self.backing_store_starting_solutions.clone();

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
            epsilon: 0.0,
            cost: 0.0,
            p: 0.0,
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

#[doc(hidden)]
pub trait SerializableModel {
	fn save_to_disk(&self, file_path: &str) -> Result<(), ModelError>;
}

pub trait LibLinearModel: SerializableModel {
	type Parameter: LibLinearParameter;
	type Problem: LibLinearProblem;

	fn model_type(&self) -> Result<ModelType, ModelError>;
	fn parameters(&self) -> Option<&Self::Parameter>;
	fn problem(&self) -> Option<&Self::Problem>;

	fn predict(&self, features: PredictionInput) -> f64;
	fn predict_values(&self, features: PredictionInput) -> (f64, Vec<f64>);
	fn predict_probabilities(&self, features: PredictionInput) -> (f64, Vec<f64>);

	fn feature_coefficient(&self, feature_index: i32, label_index: i32) -> f64;
	fn label_bias(&self, label_index: i32) -> f64;
}

pub trait LibLinearCrossValidator {
	fn cross_validation(&self, folds: i32) -> Vec<f64>;
	fn find_optimal_constraints_violation_cost(
		&self,
		folds: i32,
		search_range: (f64, f64),
	) -> (f64, f64);
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
	) -> Vec<ffi::FeatureNode> {
		assert_ne!(self.bound.is_null(), true);

		let last_feature_index = prediction_input.last_feature_index() as i32;
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
				index: last_feature_index,
				value: bias,
			});
		}

		data.push(ffi::FeatureNode {
			index: -1,
			value: 0f64,
		});

		data
	}
}

impl SerializableModel for Model {
	fn save_to_disk(&self, file_path: &str) -> Result<(), ModelError> {
		unsafe {
			let result = ffi::save_model(CString::new(file_path).unwrap().as_ptr(), self.bound);
			if result == -1 {
				return Err(ModelError::SerializationError {
					e: "save_model() returned -1".to_owned().to_string(),
				});
			}
		}

		Ok(())
	}
}

impl LibLinearModel for Model {
	type Parameter = Parameter;
	type Problem = Problem;

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


	fn parameters(&self) -> Option<&Self::Parameter> {
		self.parameter.as_ref()
	}

	fn problem(&self) -> Option<&Self::Problem> {
		self.problem.as_ref()
	}

	fn predict(&self, features: PredictionInput) -> f64 {
		let transformed_features = self.preprocess_prediction_input(features);
		unimplemented!()
	}

	fn predict_values(&self, features: PredictionInput) -> (f64, Vec<f64>) {
		unimplemented!()
	}

	fn predict_probabilities(&self, features: PredictionInput) -> (f64, Vec<f64>) {
		unimplemented!()
	}

	fn feature_coefficient(&self, feature_index: i32, label_index: i32) -> f64 {
		unimplemented!()
	}

	fn label_bias(&self, label_index: i32) -> f64 {
		unimplemented!()
	}
}

impl LibLinearCrossValidator for Model {
	fn cross_validation(&self, folds: i32) -> Vec<f64> {
		unimplemented!()
	}

	fn find_optimal_constraints_violation_cost(
		&self,
		folds: i32,
		search_range: (f64, f64),
	) -> (f64, f64) {
		unimplemented!()
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

	pub fn save_model(path_to_serialized_model: &str, model: &LibLinearModel<Parameter=Parameter, Problem=Problem>) -> Result<(), Error> {
		Ok(model.save_to_disk(path_to_serialized_model)?)
	}
}
