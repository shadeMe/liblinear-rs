//! Types and traits that wrap hyper parameters.

use std::marker::PhantomData;

use crate::{
    errors::ModelError,
    solver::{
        traits::{
            CanDisableBiasRegularization, IsSingleClassSolver, IsSupportVectorRegressionSolver,
            IsTrainableSolver, Solver, SupportsInitialSolutions,
        },
        SolverOrdinal,
    },
};

use self::traits::{
    SetBiasRegularization, SetInitialSolutions, SetOutlierRatio, SetRegressionLossSensitivity,
};

/// Traits implemented by [`Parameters`].
pub mod traits {
    /// Implemented for parameters with solvers that implement the
    /// [`SupportsInitialSolutions`](crate::solver::traits::SupportsInitialSolutions) trait.
    pub trait SetInitialSolutions {
        /// Set the initial solution specification.
        fn initial_solutions(&mut self, init_solutions: Vec<f64>) -> &mut Self;
    }

    /// Implemented for parameters with solvers that implement the
    /// [`CanDisableBiasRegularization`](crate::solver::traits::CanDisableBiasRegularization) trait.
    pub trait SetBiasRegularization {
        /// Toggle bias regularization during training.
        ///
        /// If set to `false`, the bias value will automatically be set to `1`.
        ///
        /// Default: `true`
        fn bias_regularization(&mut self, bias_regularization: bool) -> &mut Self;
    }

    /// Implemented for parameters with solvers that implement the
    /// [`IsSingleClassSolver`](crate::solver::traits::IsSingleClassSolver) trait.
    pub trait SetOutlierRatio {
        /// Set the fraction of data that is to be classified as outliers (parameter `nu`).
        ///
        /// Default: `0.5`
        fn outlier_ratio(&mut self, nu: f64) -> &mut Self;
    }

    /// Implemented for parameters with solvers that implement the
    /// [`IsSupportVectorRegressionSolver`](crate::solver::traits::IsSupportVectorRegressionSolver) trait.
    pub trait SetRegressionLossSensitivity {
        /// Set the tolerance margin/loss sensitivity of support vector regression (parameter `p`).
        ///
        /// Default: `0.1
        fn regression_loss_sensitivity(&mut self, p: f64) -> &mut Self;
    }
}

/// Represents the tunable parameters of a LIBLINEAR model.
///
/// This struct is generic on the [`Solver`](crate::solver::traits::Solver) trait and
/// its descendents, using them to implement solver-specific functionality.
#[derive(Debug, Clone)]
pub struct Parameters<SolverT> {
    pub(crate) _solver: PhantomData<SolverT>,
    pub(crate) epsilon: f64,
    pub(crate) cost: f64,
    pub(crate) p: f64,
    pub(crate) nu: f64,
    pub(crate) cost_penalty: Vec<(i32, f64)>,
    pub(crate) initial_solutions: Vec<f64>,
    pub(crate) bias: f64,
    pub(crate) regularize_bias: bool,
}

impl<SolverT> Default for Parameters<SolverT> {
    fn default() -> Self {
        Self {
            _solver: PhantomData,
            epsilon: 0.01,
            cost: 1.0,
            p: 0.1,
            nu: 0.5,
            cost_penalty: Vec::new(),
            initial_solutions: Vec::new(),
            bias: -1f64,
            regularize_bias: true,
        }
    }
}

impl<SolverT> Parameters<SolverT>
where
    SolverT: IsTrainableSolver,
{
    /// Set tolerance of termination criterion for optimization (parameter `e`).
    ///
    /// Default: `0.01`
    pub fn stopping_tolerance(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Set cost of constraints violation (parameter `C`).
    //
    /// Rules the trade-off between regularization and correct classification on data.
    /// It can be seen as the inverse of a regularization constant.
    ///
    /// Default: `1.0`
    pub fn constraints_violation_cost(&mut self, cost: f64) -> &mut Self {
        self.cost = cost;
        self
    }

    /// Set weights to adjust the cost of constraints violation for specific classes. Each element
    /// is a tuple where the first value is the label and the second its corresponding weight penalty.
    ///
    /// Useful when training classifiers on unbalanced input data or with asymmetric mis-classification cost.
    pub fn cost_penalty(&mut self, cost_penalty: Vec<(i32, f64)>) -> &mut Self {
        self.cost_penalty = cost_penalty;
        self
    }

    /// Set the bias of the training data. If `bias >= 0`, it's appended to the feature vector of each training data instance.
    ///
    /// Default: `-1.0`
    pub fn bias(&mut self, bias: f64) -> &mut Self {
        self.bias = bias;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), ModelError> {
        if self.epsilon <= 0f64 {
            return Err(ModelError::InvalidParameters(format!(
                "epsilon must be > 0, but got '{}'",
                self.epsilon
            )));
        }

        if self.cost <= 0f64 {
            return Err(ModelError::InvalidParameters(format!(
                "constraints violation cost must be > 0, but got '{}'",
                self.cost
            )));
        }

        if self.p < 0f64 {
            return Err(ModelError::InvalidParameters(format!(
                "regression loss sensitivity must be >= 0, but got '{}'",
                self.p
            )));
        }

        if self.bias >= 0f64 && <SolverT as Solver>::ordinal() == SolverOrdinal::ONECLASS_SVM {
            return Err(ModelError::InvalidParameters(format!(
                "bias term must be < 0 for single-class SVM, but got '{}'",
                self.bias
            )));
        }

        if !self.regularize_bias && self.bias != 1f64 {
            return Err(ModelError::InvalidParameters(format!(
                "bias term must be `1.0` when regularization is disabled, but got '{}'",
                self.bias
            )));
        }

        Ok(())
    }
}

impl<SolverT> SetInitialSolutions for Parameters<SolverT>
where
    SolverT: IsTrainableSolver + SupportsInitialSolutions,
{
    fn initial_solutions(&mut self, initial_solutions: Vec<f64>) -> &mut Self {
        self.initial_solutions = initial_solutions;
        self
    }
}

impl<SolverT> SetBiasRegularization for Parameters<SolverT>
where
    SolverT: IsTrainableSolver + CanDisableBiasRegularization,
{
    fn bias_regularization(&mut self, bias_regularization: bool) -> &mut Self {
        self.regularize_bias = bias_regularization;
        if !self.regularize_bias {
            self.bias = 1f64;
        }
        self
    }
}

impl<SolverT> SetOutlierRatio for Parameters<SolverT>
where
    SolverT: IsTrainableSolver + IsSingleClassSolver,
{
    fn outlier_ratio(&mut self, nu: f64) -> &mut Self {
        self.nu = nu;
        self
    }
}

impl<SolverT> SetRegressionLossSensitivity for Parameters<SolverT>
where
    SolverT: IsTrainableSolver + IsSupportVectorRegressionSolver,
{
    fn regression_loss_sensitivity(&mut self, p: f64) -> &mut Self {
        self.p = p;
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{errors::ModelError, solver};

    use super::traits::*;
    use super::Parameters;

    #[test]
    fn test_parameter_runtime_validation() {
        let mut params = Parameters::<solver::L2R_LR>::default();
        params.stopping_tolerance(-1f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::L1R_LR>::default();
        params.constraints_violation_cost(0f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::L2R_L2LOSS_SVR>::default();
        params.regression_loss_sensitivity(-1f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::ONECLASS_SVM>::default();
        params.bias(10f64).outlier_ratio(1f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::L1R_L2LOSS_SVC>::default();
        params.bias_regularization(false).bias(10f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::L2R_L2LOSS_SVR>::default();
        params
            .cost_penalty(Vec::new())
            .initial_solutions(Vec::new());

        let mut params = Parameters::<solver::L2R_L2LOSS_SVR>::default();
        params.cost_penalty(Vec::new());
    }
}
