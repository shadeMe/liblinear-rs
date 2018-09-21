use failure::Error;
use ffi;
pub use ffi::SolverType as SolverType;
use itertools::Itertools;
use num;
use std;
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use util::train::*;

#[derive(Debug, Fail)]
pub enum ClassifierError {
    /// Parameter errors
    #[fail(display = "parameter error: {}", e)]
    InvalidParameters { e: String },
}

pub trait LibLinearProblem: Clone {
    fn bias(&self) -> f64;
    fn num_features(&self) -> usize;
}

struct Problem {
    num_features: i32,
    backing_store_labels: Vec<f64>,
    backing_store_features: Vec<Vec<ffi::FeatureNode>>,
    backing_store_feature_ptrs: Vec<*const ffi::FeatureNode>,
    bound: ffi::Problem,
}

impl Problem {
    fn new(input_data: TrainingInput, bias: f64) -> Result<Problem, ClassifierError> {
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
    input_data: Result<TrainingInput, TrainingInputError>,
    bias: f64,
}

impl ProblemBuilder {
    fn from_dense_features(labels: Vec<f64>, dense_features: Vec<Vec<f64>>) -> ProblemBuilder {
        ProblemBuilder {
            input_data: TrainingInput::from_dense_features(labels, dense_features),
            bias: 0.0,
        }
    }

    fn from_libsvm_data(path_to_libsvm_data: &str) -> ProblemBuilder {
        ProblemBuilder {
            input_data: TrainingInput::from_libsvm_file(path_to_libsvm_data),
            bias: 0.0,
        }
    }

    pub fn bias(&mut self, bias: f64) -> &mut Self {
        self.bias = bias;
        self
    }

    pub fn build(self) -> Result<impl LibLinearProblem, Error> {
        Ok(Problem::new(self.input_data?, self.bias)?)
    }
}


pub trait LibLinearParameter: Clone {
    fn solver_type(&self) -> SolverType;
    fn stopping_criterion(&self) -> f64;
    fn constraints_violation_cost(&self) -> f64;
    fn regression_loss_sensitivity(&self) -> f64;
}

struct Parameter {
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
    ) -> Result<Parameter, ClassifierError> {
        if weights.len() == 0 || weight_labels.len() == 0 {
            return Err(ClassifierError::InvalidParameters {
                e: "No weights/weight labels".to_string(),
            });
        } else if weights.len() != weight_labels.len() {
            return Err(ClassifierError::InvalidParameters {
                e: "Mismatch between number of labels and weights".to_string(),
            });
        } else if !init_sol.is_empty() && weights.len() != init_sol.len() {
            return Err(ClassifierError::InvalidParameters {
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
                return Err(ClassifierError::InvalidParameters {
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

    pub fn build(self) -> Result<impl LibLinearParameter, Error> {
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

pub struct Classifier {
    problem: Option<Problem>,
    parameter: Option<Parameter>,
    model: *mut ffi::Model,
}

impl Classifier {}
