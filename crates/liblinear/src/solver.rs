//! Types of generalized linear models supported by LIBLINEAR.
//!
//! These combine several types of regularization schemes:
//! * L1
//! * L2
//!
//! ...and loss functions:
//! * L1-loss for SVM
//! * Regular L2-loss for SVM (hinge-loss)
//! * Logistic loss for logistic regression
//!
//! Solvers are represented as a unique structs that implement specific
//! marker traits to indicate their capabilities. These are used by the
//! generic types [`Model`](crate::model::Model) and [`Parameters`](crate::parameter::Parameters)
//! to ensure at compile-time that a model with a specific solver can only
//! invoke its supported functionality.

use liblinear_macros::{
    CanDisableBiasRegularization, IsLogisticRegressionSolver, IsNonSingleClassSolver,
    IsSingleClassSolver, IsTrainableSolver, SupportsInitialSolutions, SupportsParameterSearch,
};

/// Traits implemented by solvers.
pub mod traits {
    use super::SolverOrdinal;

    /// Trait implemented by all solver types.
    pub trait Solver {
        fn ordinal() -> SolverOrdinal;
    }

    /// Marker trait for probablistic/logistic regression solvers.
    pub trait IsLogisticRegressionSolver: Solver {}

    /// Marker trait for single-class solvers.
    pub trait IsSingleClassSolver: Solver {}

    /// Marker trait for non-single-class solvers.
    pub trait IsNonSingleClassSolver: Solver {}

    /// Marker trait for solvers that support model training.
    pub trait IsTrainableSolver: Solver {}

    /// Marker trait for solvers that support toggling bias regularization.
    pub trait CanDisableBiasRegularization: IsTrainableSolver {}

    /// Marker trait for solvers that support initial solution specification.
    pub trait SupportsInitialSolutions: IsTrainableSolver {}

    /// Marker trait for solvers that support searching for optimal hyperparameters.
    pub trait SupportsParameterSearch: IsTrainableSolver {}
}

/// FFI ordinal that identifies the type.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, FromPrimitive)]
pub enum SolverOrdinal {
    #[doc(hidden)]
    /// The `UNKNOWN` variant is used as a sentinel value
    /// by non-trainable solvers.
    UNKNOWN = -1,
    L2R_LR = 0,
    L2R_L2LOSS_SVC_DUAL = 1,
    L2R_L2LOSS_SVC = 2,
    L2R_L1LOSS_SVC_DUAL = 3,
    MCSVM_CS = 4,
    L1R_L2LOSS_SVC = 5,
    L1R_LR = 6,
    L2R_LR_DUAL = 7,
    L2R_L2LOSS_SVR = 11,
    L2R_L2LOSS_SVR_DUAL = 12,
    L2R_L1LOSS_SVR_DUAL = 13,
    ONECLASS_SVM = 21,
}

/// L2-regularized logistic regression (primal).
#[allow(non_camel_case_types)]
#[derive(
    Copy,
    Clone,
    Debug,
    IsLogisticRegressionSolver,
    IsNonSingleClassSolver,
    IsTrainableSolver,
    CanDisableBiasRegularization,
    SupportsInitialSolutions,
    SupportsParameterSearch,
)]
pub struct L2R_LR;
impl traits::Solver for L2R_LR {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L2R_LR
    }
}

/// L2-regularized L2-loss support vector classification (dual).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, IsNonSingleClassSolver, IsTrainableSolver)]
pub struct L2R_L2LOSS_SVC_DUAL;
impl traits::Solver for L2R_L2LOSS_SVC_DUAL {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L2R_L2LOSS_SVC_DUAL
    }
}

/// L2-regularized L2-loss support vector classification (primal).
#[allow(non_camel_case_types)]
#[derive(
    Copy,
    Clone,
    Debug,
    IsNonSingleClassSolver,
    IsTrainableSolver,
    CanDisableBiasRegularization,
    SupportsInitialSolutions,
    SupportsParameterSearch,
)]
pub struct L2R_L2LOSS_SVC;
impl traits::Solver for L2R_L2LOSS_SVC {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L2R_L2LOSS_SVC
    }
}

/// L2-regularized L1-loss support vector classification (dual).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, IsNonSingleClassSolver, IsTrainableSolver)]
pub struct L2R_L1LOSS_SVC_DUAL;
impl traits::Solver for L2R_L1LOSS_SVC_DUAL {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L2R_L1LOSS_SVC_DUAL
    }
}

/// Support vector classification by Crammer and Singer.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, IsNonSingleClassSolver, IsTrainableSolver)]
pub struct MCSVM_CS;
impl traits::Solver for MCSVM_CS {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::MCSVM_CS
    }
}

/// L1-regularized L2-loss support vector classification.
#[allow(non_camel_case_types)]
#[derive(
    Copy, Clone, Debug, IsNonSingleClassSolver, IsTrainableSolver, CanDisableBiasRegularization,
)]
pub struct L1R_L2LOSS_SVC;
impl traits::Solver for L1R_L2LOSS_SVC {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L1R_L2LOSS_SVC
    }
}

/// L1-regularized logistic regression.
#[allow(non_camel_case_types)]
#[derive(
    Copy,
    Clone,
    Debug,
    IsLogisticRegressionSolver,
    IsTrainableSolver,
    IsNonSingleClassSolver,
    CanDisableBiasRegularization,
)]
pub struct L1R_LR;
impl traits::Solver for L1R_LR {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L1R_LR
    }
}

/// L2-regularized logistic regression (dual).
#[allow(non_camel_case_types)]
#[derive(
    Copy, Clone, Debug, IsLogisticRegressionSolver, IsTrainableSolver, IsNonSingleClassSolver,
)]
pub struct L2R_LR_DUAL;
impl traits::Solver for L2R_LR_DUAL {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L2R_LR_DUAL
    }
}

/// L2-regularized L2-loss support vector regression (primal).
#[allow(non_camel_case_types)]
#[derive(
    Copy,
    Clone,
    Debug,
    IsNonSingleClassSolver,
    IsTrainableSolver,
    CanDisableBiasRegularization,
    SupportsInitialSolutions,
    SupportsParameterSearch,
)]
pub struct L2R_L2LOSS_SVR;
impl traits::Solver for L2R_L2LOSS_SVR {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L2R_L2LOSS_SVR
    }
}

/// L2-regularized L2-loss support vector regression (dual).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, IsNonSingleClassSolver, IsTrainableSolver)]
pub struct L2R_L2LOSS_SVR_DUAL;
impl traits::Solver for L2R_L2LOSS_SVR_DUAL {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L2R_L2LOSS_SVR_DUAL
    }
}

/// L2-regularized L1-loss support vector regression (dual).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, IsNonSingleClassSolver, IsTrainableSolver)]
pub struct L2R_L1LOSS_SVR_DUAL;
impl traits::Solver for L2R_L1LOSS_SVR_DUAL {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::L2R_L1LOSS_SVR_DUAL
    }
}

/// One-class support vector machine (dual).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, IsSingleClassSolver, IsTrainableSolver)]
pub struct ONECLASS_SVM;
impl traits::Solver for ONECLASS_SVM {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::ONECLASS_SVM
    }
}

/// Generic solver that supports basic model operations.
///
/// This is useful when loading a model from disk whose
/// solver kind is not known ahead of time.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct GenericSolver;
impl traits::Solver for GenericSolver {
    fn ordinal() -> SolverOrdinal {
        SolverOrdinal::UNKNOWN
    }
}

#[cfg(test)]
mod tests {
    use crate::solver::{
        traits::Solver, GenericSolver, SolverOrdinal, L1R_L2LOSS_SVC, L1R_LR, L2R_L1LOSS_SVC_DUAL,
        L2R_L1LOSS_SVR_DUAL, L2R_L2LOSS_SVC, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVR,
        L2R_L2LOSS_SVR_DUAL, L2R_LR, L2R_LR_DUAL, MCSVM_CS, ONECLASS_SVM,
    };

    #[test]
    fn test_solver_ordinals() {
        assert_eq!(L2R_LR::ordinal(), SolverOrdinal::L2R_LR);
        assert_eq!(
            L2R_L2LOSS_SVC_DUAL::ordinal(),
            SolverOrdinal::L2R_L2LOSS_SVC_DUAL
        );
        assert_eq!(L2R_L2LOSS_SVC::ordinal(), SolverOrdinal::L2R_L2LOSS_SVC);
        assert_eq!(
            L2R_L1LOSS_SVC_DUAL::ordinal(),
            SolverOrdinal::L2R_L1LOSS_SVC_DUAL
        );
        assert_eq!(MCSVM_CS::ordinal(), SolverOrdinal::MCSVM_CS);
        assert_eq!(L1R_L2LOSS_SVC::ordinal(), SolverOrdinal::L1R_L2LOSS_SVC);
        assert_eq!(L1R_LR::ordinal(), SolverOrdinal::L1R_LR);
        assert_eq!(L2R_LR_DUAL::ordinal(), SolverOrdinal::L2R_LR_DUAL);
        assert_eq!(L2R_L2LOSS_SVR::ordinal(), SolverOrdinal::L2R_L2LOSS_SVR);
        assert_eq!(
            L2R_L2LOSS_SVR_DUAL::ordinal(),
            SolverOrdinal::L2R_L2LOSS_SVR_DUAL
        );
        assert_eq!(
            L2R_L1LOSS_SVR_DUAL::ordinal(),
            SolverOrdinal::L2R_L1LOSS_SVR_DUAL
        );
        assert_eq!(ONECLASS_SVM::ordinal(), SolverOrdinal::ONECLASS_SVM);
        assert_eq!(GenericSolver::ordinal(), SolverOrdinal::UNKNOWN);
    }
}
