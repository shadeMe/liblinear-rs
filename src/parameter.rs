use std::ffi::CStr;

use crate::{errors::ParameterError, ffi};

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
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

    /// One-class support vector machine (dual).
    ONECLASS_SVM = 21,
}

impl SolverType {
    /// Returns true if the solver is a probabilistic/logistic regression solver.
    ///
    /// Supported solvers: `L2R_LR`, `L1R_LR`, `L2R_LR_DUAL`.
    pub fn is_logistic_regression(&self) -> bool {
        match self {
            SolverType::L2R_LR | SolverType::L1R_LR | SolverType::L2R_LR_DUAL => true,
            _ => false,
        }
    }

    /// Returns true if the solver is a support vector regression solver.
    ///
    /// Supported solvers: `L2R_L2LOSS_SVR`, `L2R_L2LOSS_SVR_DUAL`, `L2R_L1LOSS_SVR_DUAL`.
    pub fn is_support_vector_regression(&self) -> bool {
        match self {
            SolverType::L2R_L2LOSS_SVR
            | SolverType::L2R_L2LOSS_SVR_DUAL
            | SolverType::L2R_L1LOSS_SVR_DUAL => true,
            _ => false,
        }
    }

    /// Returns true if the solver supports multi-class classification.
    ///
    /// Supported solvers: All non-SVR solvers.
    pub fn is_multi_class_classification(&self) -> bool {
        !self.is_support_vector_regression()
    }

    /// Returns true if the solver is a one-class solver.
    pub fn is_one_class(&self) -> bool {
        *self == SolverType::ONECLASS_SVM
    }
}

impl Default for SolverType {
    /// Default: `L2R_LR`
    fn default() -> Self {
        SolverType::L2R_LR
    }
}

/// Represents the tunable parameters of a model.
pub trait LibLinearParameter: Clone {
    /// Solver used for classification or regression.
    fn solver_type(&self) -> SolverType;

    /// Tolerance of termination criterion for optimization (parameter _e_).
    fn stopping_tolerance(&self) -> f64;

    /// Cost of constraints violation (parameter _c_).
    ///
    /// Rules the trade-off between regularization and correct classification on data.
    /// It can be seen as the inverse of a regularization constant.
    fn constraints_violation_cost(&self) -> f64;

    /// Fraction of data that is to be classified as outliers (parameter _n_).
    ///
    /// Only applicable to the one-class SVM solver.
    fn outlier_ratio(&self) -> f64;

    /// Sensitivity of loss of support vector regression (parameter _p_).
    fn regression_loss_sensitivity(&self) -> f64;

    /// Regularize bias during training (parameter _r_).
    fn regularize_bias(&self) -> bool;
}

pub(crate) struct Parameter {
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
        nu: f64,
        cost_penalty_weights: Vec<f64>,
        cost_penalty_labels: Vec<i32>,
        init_solutions: Vec<f64>,
        regularize_bias: bool,
    ) -> Result<Self, ParameterError> {
        if cost_penalty_weights.len() != cost_penalty_labels.len() {
            return Err(ParameterError::InvalidParameters(
                "Mismatch between cost penalty weights and labels".to_owned(),
            ));
        }

        let num_weights = cost_penalty_weights.len() as i32;

        let param = Self {
            bound: ffi::Parameter {
                solver_type: solver as i32,
                eps,
                C: cost,
                nr_weight: num_weights,
                weight_label: if cost_penalty_labels.is_empty() {
                    std::ptr::null()
                } else {
                    cost_penalty_labels.as_ptr()
                },
                weight: if cost_penalty_weights.is_empty() {
                    std::ptr::null()
                } else {
                    cost_penalty_weights.as_ptr()
                },
                p,
                nu,
                init_sol: if init_solutions.is_empty() {
                    std::ptr::null()
                } else {
                    init_solutions.as_ptr()
                },
                regularize_bias: if regularize_bias { 1 } else { 0 },
            },
            backing_store_class_cost_penalty_weights: cost_penalty_weights,
            backing_store_class_cost_penalty_labels: cost_penalty_labels,
            backing_store_starting_solutions: init_solutions,
        };

        unsafe {
            let param_error = ffi::check_parameter(std::ptr::null(), &param.bound);
            if !param_error.is_null() {
                return Err(ParameterError::InvalidParameters(
                    CStr::from_ptr(param_error).to_string_lossy().to_string(),
                ));
            }
        }

        Ok(param)
    }

    pub(crate) fn ffi_obj(&self) -> &ffi::Parameter {
        &self.bound
    }
}

impl LibLinearParameter for Parameter {
    fn solver_type(&self) -> SolverType {
        unsafe { std::mem::transmute(self.bound.solver_type as i8) }
    }

    fn stopping_tolerance(&self) -> f64 {
        self.bound.eps
    }

    fn constraints_violation_cost(&self) -> f64 {
        self.bound.C
    }

    fn regression_loss_sensitivity(&self) -> f64 {
        self.bound.p
    }

    fn regularize_bias(&self) -> bool {
        if self.bound.regularize_bias == 1 {
            true
        } else {
            false
        }
    }

    fn outlier_ratio(&self) -> f64 {
        self.bound.nu
    }
}

impl Clone for Parameter {
    fn clone(&self) -> Self {
        let weights = self.backing_store_class_cost_penalty_weights.clone();
        let weight_labels = self.backing_store_class_cost_penalty_labels.clone();
        let init_sol = self.backing_store_starting_solutions.clone();

        Self {
            bound: ffi::Parameter {
                solver_type: self.bound.solver_type as i32,
                eps: self.bound.eps,
                C: self.bound.C,
                nr_weight: self.bound.nr_weight,
                weight_label: weight_labels.as_ptr(),
                weight: weights.as_ptr(),
                p: self.bound.p,
                nu: self.bound.nu,
                init_sol: init_sol.as_ptr(),
                regularize_bias: self.bound.regularize_bias,
            },
            backing_store_class_cost_penalty_weights: weights,
            backing_store_class_cost_penalty_labels: weight_labels,
            backing_store_starting_solutions: init_sol,
        }
    }
}

/// Builder for [LibLinearParameter](enum.LibLinearParameter.html).
#[derive(Clone, Debug)]
pub struct ParameterBuilder {
    solver_type: SolverType,
    epsilon: f64,
    cost: f64,
    p: f64,
    nu: f64,
    cost_penalty_weights: Vec<f64>,
    cost_penalty_labels: Vec<i32>,
    init_solutions: Vec<f64>,
    regularize_bias: bool,
}

impl Default for ParameterBuilder {
    fn default() -> Self {
        Self {
            solver_type: SolverType::default(),
            epsilon: 0.01,
            cost: 1.0,
            p: 0.1,
            nu: 0.5,
            cost_penalty_weights: Vec::new(),
            cost_penalty_labels: Vec::new(),
            init_solutions: Vec::new(),
            regularize_bias: true,
        }
    }
}

impl ParameterBuilder {
    /// Set solver type.
    ///
    /// Default: `[L2R_LR](enum.SolverType.html#variant.L2R_LR)`
    pub fn solver_type(&mut self, solver_type: SolverType) -> &mut Self {
        self.solver_type = solver_type;
        self
    }

    /// Set tolerance of termination criterion.
    ///
    /// Default: `0.01`
    pub fn stopping_tolerance(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Set cost of constraints violation.
    ///
    /// Default: `1.0`
    pub fn constraints_violation_cost(&mut self, cost: f64) -> &mut Self {
        self.cost = cost;
        self
    }

    /// Set tolerance margin in regression loss function of SVR. Not used for classification problems.
    ///
    /// Default: `0.1
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

    /// Set initial solution specification for solvers [L2R_LR](enum.SolverType.html#variant.L2R_LR) and/or [L2R_L2LOSSES_SVC](enum.SolverType.html#variant.L2R_L2LOSSES_SVC).
    pub fn initial_solutions(&mut self, init_solutions: Vec<f64>) -> &mut Self {
        self.init_solutions = init_solutions;
        self
    }

    /// Regularize bias during training.
    ///
    /// If set to `false`, the bias value must be set to `1`.
    ///
    /// Default: `true`
    pub fn regularize_bias(&mut self, regularize: bool) -> &mut Self {
        self.regularize_bias = regularize;
        self
    }

    /// Fraction of data that is to be classified as outliers.
    ///
    /// Only applicable to the one-class SVM solver.
    ///
    /// Default: `0.5`
    pub fn outlier_ratio(&mut self, nu: f64) -> &mut Self {
        self.nu = nu;
        self
    }

    pub(crate) fn build(self) -> Result<Parameter, ParameterError> {
        Parameter::new(
            self.solver_type,
            self.epsilon,
            self.cost,
            self.p,
            self.nu,
            self.cost_penalty_weights,
            self.cost_penalty_labels,
            self.init_solutions,
            self.regularize_bias,
        )
    }
}

/// Super-trait of [LibLinearModel](trait.LibLinearModel.html) and [LibLinearCrossValidator](trait.LibLinearCrossValidator.html).
pub trait HasLibLinearParameter {
    type Output: LibLinearParameter;

    /// The parameters of the model/cross-validator.
    fn parameter(&self) -> &Self::Output;
}
