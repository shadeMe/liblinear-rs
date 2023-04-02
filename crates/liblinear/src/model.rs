//! Types and traits that wrap models.

use std::{ffi::CStr, marker::PhantomData};

use self::traits::ModelBase;
use crate::{
    errors::{ModelError, PredictionInputError},
    ffi::{self, FeatureNode},
    parameter::Parameters,
    solver::{
        traits::{
            IsLogisticRegressionSolver, IsNonSingleClassSolver, IsSingleClassSolver,
            IsTrainableSolver, Solver, SupportsParameterSearch,
        },
        GenericSolver, SolverOrdinal, L1R_L2LOSS_SVC, L1R_LR, L2R_L1LOSS_SVC_DUAL,
        L2R_L1LOSS_SVR_DUAL, L2R_L2LOSS_SVC, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVR,
        L2R_L2LOSS_SVR_DUAL, L2R_LR, L2R_LR_DUAL, MCSVM_CS, ONECLASS_SVM,
    },
    util::{PredictionInput, TrainingInput},
};

/// Traits implemented by [`Model`].
pub mod traits {
    use crate::{
        errors::ModelError,
        parameter::Parameters,
        solver::SolverOrdinal,
        util::{PredictionInput, TrainingInput},
    };

    /// Common methods implemented by all [`Model`](super::Model)s.
    pub trait ModelBase {
        /// Returns one of the following values:
        ///
        /// * For a classification model, the predicted class is returned.
        /// * For a regression model, the function value calculated using the model is returned.
        fn predict(&self, features: &PredictionInput) -> Result<f64, ModelError>;

        /// Returns a tuple of the following values:
        ///
        /// * A list of decision values. If `k` is the number of classes, each element includes results
        /// of predicting k binary-class SVMs. If `k == 2` and solver is not `MCSVM_CS`, only the first
        /// decision value is valid.
        ///
        ///   The values correspond to the classes returned by the `labels` method.
        ///
        ///
        /// * The class with the highest decision value.
        fn predict_values(&self, features: &PredictionInput)
            -> Result<(Vec<f64>, f64), ModelError>;

        /// Returns the coefficient for the feature with the given index
        /// and the class with the given (label) index.
        ///
        ///
        /// Note that while feature indices start from `1`, label indices start from `0`.
        /// For regression and one-class models, the label index is ignored.
        fn feature_coefficient(
            &self,
            feature_index: u32,
            label_index: u32,
        ) -> Result<f64, ModelError>;

        /// Returns the bias of the input data with which the model was trained.
        fn bias(&self) -> f64;

        /// Returns the labels/classes learned by the model.
        fn labels(&self) -> &Vec<i32>;

        /// Returns the number of classes of the model.
        ///
        /// For regression models and one-class SVMs, `2` is returned.
        fn num_classes(&self) -> u32;

        /// Returns the number of features of the input data with which the model was trained.
        fn num_features(&self) -> u32;

        /// Returns the solver used by the model.
        fn solver(&self) -> SolverOrdinal;
    }

    /// Methods implemented by models with solvers that implement the
    /// [`IsTrainableSolver`](crate::solver::traits::IsTrainableSolver) trait.
    pub trait TrainableModel<SolverT>: Sized {
        /// Trains a new model with the provided input and parameters.
        fn train(
            training_data: &TrainingInput,
            params: &Parameters<SolverT>,
        ) -> Result<Self, ModelError>;

        /// Performs k-folds cross-validation with the provided input and
        /// parameters and returns the predicted labels.
        ///
        /// Number of folds must be `>= 2`.
        fn cross_validation(
            training_data: &TrainingInput,
            params: &Parameters<SolverT>,
            folds: u32,
        ) -> Result<Vec<f64>, ModelError>;
    }

    /// Methods implemented by models with solvers that implement the
    /// [`IsLogisticRegressionSolver`](crate::solver::traits::IsLogisticRegressionSolver) trait.
    pub trait LogisticRegressionModel {
        /// Returns a tuple of the following values:
        ///
        /// * A list of probability estimates. If `k` is the number of classes,
        /// each element contains `k` values indicating the probability that the
        /// testing instance is in each class.
        ///
        ///   The values correspond to the classes returned by the `labels` method.
        ///
        ///
        /// * The class with the highest probability.
        fn predict_probabilities(
            &self,
            features: &PredictionInput,
        ) -> Result<(Vec<f64>, f64), ModelError>;
    }

    /// Methods implemented by models with solvers that implement the
    /// [`IsSingleClassSolver`](crate::solver::traits::IsSingleClassSolver) trait.
    pub trait SingleClassModel {
        /// Returns the bias term used in one-class SVMs.
        fn rho(&self) -> f64;
    }

    /// Methods implemented by models with solvers that implement the
    /// [`IsNonSingleClassSolver`](crate::solver::traits::IsNonSingleClassSolver) trait.
    pub trait NonSingleClassModel {
        /// Returns the bias term corresponding to the class with the given index.
        ///
        /// For classification models, if label index is not in a valid range, an !!!ERROR value will be returned.
        /// For regression models, the label index is ignored.
        fn label_bias(&self, label_index: u32) -> Result<f64, ModelError>;
    }

    /// Methods implemented by models with solvers that implement the
    /// [`SupportsParameterSearch`](crate::solver::traits::SupportsParameterSearch) trait.
    pub trait ParameterSearchableModel<SolverT> {
        /// Performs k-folds cross-validation with the provided input and
        /// parameters to find the best cost value (parameter `C`) and regression
        /// loss sensitivity (parameter `p`) and returns a tuple with the following values:
        ///
        /// * The best cost value.
        /// * The accuracy of the best cost value (classification) or mean squared error (regression).
        /// * The best regression loss sensitivity value (only for regression).
        ///
        /// Supported Solvers:
        /// * L2R_LR, L2R_L2LOSS_SVC - Cross validation is conducted many times with values of `C
        /// = n * start_C; n = 1, 2, 4, 8...` to find the value with the highest cross validation
        /// accuracy. The procedure stops when the models of all folds become stable or when the cost
        /// reaches the upper-bound `max_cost = 1024`. If `start_cost <= 0`, an appropriately small value is
        /// automatically calculated and used instead.
        ///
        ///
        /// * L2R_L2LOSS_SVR - Cross validation is conducted in a two-fold loop. The outer loop
        /// iterates over the values of `p = n / (20 * max_loss_sensitivity); n = 19, 18, 17...0`. For each value
        /// of `p`, the inner loop performs cross validation with values of `C = n * start_C; n = 1, 2,
        /// 4, 8...` to find the value with the lowest mean squared error. The procedure stops when the
        /// models of all folds become stable or when the cost reaches the upper-bound `max_cost = 1048576`. If
        /// `start_cost <= 0`, an appropriately small value is automatically calculated and used instead.
        ///
        ///   `max_p` is automatically calculated from the problem's training data.
        /// If `start_loss_sensitivity <= 0`, it is set to `max_loss_sensitivity`. Otherwise, the outer
        /// loop starts with the first `p = n / (20 * max_p)` that is `<= start_loss_sensitivity`.
        fn find_optimal_constraints_violation_cost_and_loss_sensitivity(
            training_data: &TrainingInput,
            params: &Parameters<SolverT>,
            folds: u32,
            start_cost: f64,
            start_loss_sensitivity: f64,
        ) -> Result<(f64, f64, f64), ModelError>;
    }
}

/// Backing store to hold parameters and input data during model training.
#[derive(Default)]
struct BackingStore {
    labels: Vec<f64>,
    features: Vec<Vec<FeatureNode>>,
    _feature_ptrs: Vec<*const FeatureNode>,
    cost_penalty_weights: Vec<f64>,
    cost_penalty_labels: Vec<i32>,
    initial_solutions: Vec<f64>,
}

/// Represents a LIBLINEAR model object with a specific [solver](crate::solver).
///
/// This struct is generic on the [`Solver`](crate::solver::traits::Solver) trait and
/// its descendents, using them to implement solver-specific functionality.
pub struct Model<SolverT> {
    _solver: PhantomData<SolverT>,
    training_storage: Option<BackingStore>,
    learned_labels: Vec<i32>,
    c_obj: *mut ffi::Model,
}

impl<SolverT> Model<SolverT>
where
    SolverT: Solver,
{
    fn prepare_training_input(
        input_data: &TrainingInput,
        bias: f64,
        backing_store: &mut BackingStore,
    ) -> ffi::Problem {
        let num_training_instances = input_data.len() as i32;
        let num_features = input_data.dim() as i32;
        let has_bias = bias >= 0f64;
        let last_feature_index = input_data.dim() as i32;

        let (mut transformed_features, labels): (Vec<Vec<FeatureNode>>, Vec<f64>) =
            input_data.instances().iter().fold(
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

        backing_store.labels = labels;
        backing_store.features = transformed_features;
        backing_store._feature_ptrs = transformed_feature_ptrs;

        // the pointers passed to ffi::Problem will be valid even after their corresponding Vecs
        // are moved to a different location as they point to the actual backing store on the heap
        ffi::Problem {
            l: num_training_instances as i32,
            n: num_features + if has_bias { 1 } else { 0 } as i32,
            y: backing_store.labels.as_ptr(),
            x: backing_store._feature_ptrs.as_ptr(),
            bias,
        }
    }

    fn prepare_parameters(
        params: &Parameters<SolverT>,
        backing_store: &mut BackingStore,
    ) -> Result<ffi::Parameter, ModelError>
    where
        SolverT: IsTrainableSolver,
    {
        params.validate()?;

        let (cost_penalty_labels, cost_penalty_weights): (Vec<_>, Vec<_>) =
            params.cost_penalty.iter().copied().unzip();
        let num_weights = cost_penalty_labels.len() as i32;

        backing_store.cost_penalty_labels = cost_penalty_labels;
        backing_store.cost_penalty_weights = cost_penalty_weights;
        backing_store.initial_solutions = params.initial_solutions.clone();

        Ok(ffi::Parameter {
            solver_type: <SolverT as Solver>::ordinal() as i32,
            eps: params.epsilon,
            C: params.cost,
            nr_weight: num_weights,
            weight_label: if backing_store.cost_penalty_labels.is_empty() {
                std::ptr::null()
            } else {
                backing_store.cost_penalty_labels.as_ptr()
            },
            weight: if backing_store.cost_penalty_weights.is_empty() {
                std::ptr::null()
            } else {
                backing_store.cost_penalty_weights.as_ptr()
            },
            p: params.p,
            nu: params.nu,
            init_sol: if backing_store.initial_solutions.is_empty() {
                std::ptr::null()
            } else {
                backing_store.initial_solutions.as_ptr()
            },
            regularize_bias: if params.regularize_bias { 1 } else { 0 },
        })
    }

    fn prepare_prediction_input(
        &self,
        prediction_input: &PredictionInput,
    ) -> Result<Vec<FeatureNode>, PredictionInputError> {
        let last_feature_index = prediction_input.dim();
        if last_feature_index != self.num_features() {
            return Err(PredictionInputError::DataError(format!(
                "Expected {} features, found {} instead",
                self.num_features(),
                last_feature_index
            )));
        }

        let bias = unsafe { (*self.c_obj).bias };
        let has_bias = bias >= 0f64;
        let mut data: Vec<FeatureNode> = prediction_input
            .features()
            .iter()
            .map(|(index, value)| FeatureNode {
                index: *index as i32,
                value: *value,
            })
            .collect();

        if has_bias {
            data.push(FeatureNode {
                index: (last_feature_index + 1) as i32,
                value: bias,
            });
        }

        data.push(FeatureNode {
            index: -1,
            value: 0f64,
        });

        Ok(data)
    }

    fn generate_model_internals(
        training_data: &TrainingInput,
        params: &Parameters<SolverT>,
    ) -> Result<(ffi::Problem, ffi::Parameter, BackingStore), ModelError>
    where
        SolverT: IsTrainableSolver,
    {
        let mut training_storage = BackingStore::default();
        let problem =
            Self::prepare_training_input(training_data, params.bias, &mut training_storage);
        let parameter = Self::prepare_parameters(params, &mut training_storage)?;

        let param_check_err_msg = unsafe { ffi::check_parameter(&problem, &parameter) };
        // This should always pass as we perform the requisite checks in Rust-land.
        unsafe {
            assert!(
                param_check_err_msg.is_null(),
                "ffi::check_parameter() returned an error: {}",
                CStr::from_ptr(param_check_err_msg).to_string_lossy()
            );
        }

        Ok((problem, parameter, training_storage))
    }
}

impl<SolverT> traits::ModelBase for Model<SolverT>
where
    SolverT: Solver,
{
    fn predict(&self, features: &PredictionInput) -> Result<f64, ModelError> {
        Ok(self.predict_values(features)?.1)
    }

    fn predict_values(&self, features: &PredictionInput) -> Result<(Vec<f64>, f64), ModelError> {
        let transformed_features = self
            .prepare_prediction_input(features)
            .map_err(|err| ModelError::PredictionError(err.to_string()))?;

        unsafe {
            // The C-API is pretty lousy when it comes to bounds checking, so we'll conservatively
            // allocate the output buffer to fit the total number of classes.
            let mut output_values = vec![0f64; (*self.c_obj).nr_class as usize];

            let best_class = ffi::predict_values(
                self.c_obj,
                transformed_features.as_ptr(),
                output_values.as_mut_ptr(),
            );
            Ok((output_values, best_class))
        }
    }

    fn feature_coefficient(&self, feature_index: u32, label_index: u32) -> Result<f64, ModelError> {
        if feature_index == 0
            || feature_index > self.num_features()
            || feature_index - 1 > self.num_features()
        {
            return Err(ModelError::IllegalArgument(format!(
                "expected 1 <= feature index <= {}, but got '{}'",
                self.num_features(),
                feature_index
            )));
        } else if label_index >= self.num_classes() {
            return Err(ModelError::IllegalArgument(format!(
                "expected 0 <= label index < {}, but got '{}'",
                self.num_classes(),
                label_index
            )));
        }

        unsafe {
            Ok(ffi::get_decfun_coef(
                self.c_obj,
                feature_index as i32,
                label_index as i32,
            ))
        }
    }

    fn bias(&self) -> f64 {
        unsafe { (*self.c_obj).bias }
    }

    fn labels(&self) -> &Vec<i32> {
        &self.learned_labels
    }

    fn num_classes(&self) -> u32 {
        unsafe { (*self.c_obj).nr_class as u32 }
    }

    fn num_features(&self) -> u32 {
        unsafe { (*self.c_obj).nr_feature as u32 }
    }

    fn solver(&self) -> SolverOrdinal {
        let ffi_ordinal = unsafe { (*self.c_obj).param.solver_type };
        let ordinal: SolverOrdinal = num::FromPrimitive::from_i32(ffi_ordinal)
            .unwrap_or_else(|| panic!("unknown model solver with ordinal '{}'", ffi_ordinal));
        // This shouldn't happen as the sentinel value is only used internally by our crate.
        assert!(
            ordinal != SolverOrdinal::UNKNOWN,
            "sentinel value for ordinal found in model"
        );
        ordinal
    }
}

impl<SolverT> traits::TrainableModel<SolverT> for Model<SolverT>
where
    SolverT: IsTrainableSolver,
{
    fn train(
        training_data: &TrainingInput,
        params: &Parameters<SolverT>,
    ) -> Result<Self, ModelError> {
        let (problem, parameter, training_storage) =
            Self::generate_model_internals(training_data, params)?;

        let c_obj = unsafe { ffi::train(&problem, &parameter) };
        assert!(!c_obj.is_null(), "ffi::train() returned a NULL pointer");

        let mut learned_labels = Vec::<i32>::new();
        unsafe {
            for i in 0..(*c_obj).nr_class {
                learned_labels.push(*(*c_obj).label.offset(i as isize));
            }
        }

        Ok(Self {
            _solver: PhantomData,
            training_storage: Some(training_storage),
            learned_labels,
            c_obj,
        })
    }

    fn cross_validation(
        training_data: &TrainingInput,
        params: &Parameters<SolverT>,
        folds: u32,
    ) -> Result<Vec<f64>, ModelError> {
        if folds < 2 {
            return Err(ModelError::IllegalArgument(format!(
                "number of folds must be >= 2 for cross validation, but got '{}'",
                folds
            )));
        }

        let (problem, parameter, _training_storage) =
            Self::generate_model_internals(training_data, params)?;

        let mut output_labels = vec![0f64; problem.l as usize];
        unsafe {
            ffi::cross_validation(
                &problem,
                &parameter,
                folds as i32,
                output_labels.as_mut_ptr(),
            );
        }
        Ok(output_labels)
    }
}

impl<SolverT> traits::LogisticRegressionModel for Model<SolverT>
where
    SolverT: IsLogisticRegressionSolver,
{
    fn predict_probabilities(
        &self,
        features: &PredictionInput,
    ) -> Result<(Vec<f64>, f64), ModelError> {
        let transformed_features = self
            .prepare_prediction_input(features)
            .map_err(|err| ModelError::PredictionError(err.to_string()))?;

        unsafe {
            let mut output_probabilities = vec![0f64; (*self.c_obj).nr_class as usize];

            let best_class = ffi::predict_probability(
                self.c_obj,
                transformed_features.as_ptr(),
                output_probabilities.as_mut_ptr(),
            );
            Ok((output_probabilities, best_class))
        }
    }
}

impl<SolverT> traits::SingleClassModel for Model<SolverT>
where
    SolverT: IsSingleClassSolver,
{
    fn rho(&self) -> f64 {
        unsafe { ffi::get_decfun_rho(self.c_obj) }
    }
}

impl<SolverT> traits::NonSingleClassModel for Model<SolverT>
where
    SolverT: IsNonSingleClassSolver,
{
    fn label_bias(&self, label_index: u32) -> Result<f64, ModelError> {
        if label_index >= self.num_classes() {
            return Err(ModelError::IllegalArgument(format!(
                "expected 0 <= label index < {}, but got '{}'",
                self.num_classes(),
                label_index
            )));
        }
        unsafe { Ok(ffi::get_decfun_bias(self.c_obj, label_index as i32)) }
    }
}

impl<SolverT> traits::ParameterSearchableModel<SolverT> for Model<SolverT>
where
    SolverT: SupportsParameterSearch,
{
    fn find_optimal_constraints_violation_cost_and_loss_sensitivity(
        training_data: &TrainingInput,
        params: &Parameters<SolverT>,
        folds: u32,
        start_cost: f64,
        start_loss_sensitivity: f64,
    ) -> Result<(f64, f64, f64), ModelError> {
        if folds < 2 {
            return Err(ModelError::IllegalArgument(format!(
                "number of folds must be >= 2 for cross validation, but got '{}'",
                folds
            )));
        }

        let (problem, parameter, _training_storage) =
            Self::generate_model_internals(training_data, params)?;
        let mut best_cost = 0f64;
        let mut best_rate = 0f64;
        let mut best_loss_sensitivity = 0f64;
        unsafe {
            ffi::find_parameters(
                &problem,
                &parameter,
                folds as i32,
                start_cost,
                start_loss_sensitivity,
                &mut best_cost,
                &mut best_loss_sensitivity,
                &mut best_rate,
            );
        }
        Ok((best_cost, best_rate, best_loss_sensitivity))
    }
}

impl<SolverT> Drop for Model<SolverT> {
    fn drop(&mut self) {
        unsafe {
            let mut temp = self.c_obj;
            if !temp.is_null() {
                ffi::free_and_destroy_model(&mut temp);
            }
        }
    }
}

/// Helper methods for serializing and deserializing [`Model`]s.
pub mod serde {
    use std::{ffi::CString, marker::PhantomData};

    use crate::{
        errors::ModelError,
        ffi,
        solver::{traits::IsTrainableSolver, GenericSolver},
    };

    use super::Model;

    /// Loads a serialized model from disk.
    ///
    /// The output [`Model`] instance support basic model operations.
    /// To expose solver-specific functionality, it can be converted
    /// into other `Model` types using the [`TryInto`] trait.
    pub fn load_model_from_disk(
        path_to_serialized_model: &str,
    ) -> Result<Model<GenericSolver>, ModelError> {
        let file_path_cstr = CString::new(path_to_serialized_model).unwrap();
        let c_obj = unsafe { ffi::load_model(file_path_cstr.as_ptr()) };
        if c_obj.is_null() {
            return Err(ModelError::SerializationError(
                "ffi::load_model() returned a NULL pointer - check `stderr` for more information"
                    .to_owned(),
            ));
        }

        let mut learned_labels = Vec::<i32>::new();
        unsafe {
            for i in 0..(*c_obj).nr_class {
                learned_labels.push(*(*c_obj).label.offset(i as isize));
            }
        }

        Ok(Model {
            _solver: PhantomData,
            training_storage: None,
            learned_labels,
            c_obj,
        })
    }

    /// Saves a trained model to disk.
    pub fn save_model_to_disk<SolverT>(
        model: &Model<SolverT>,
        file_path: &str,
    ) -> Result<(), ModelError>
    where
        SolverT: IsTrainableSolver,
    {
        let file_path_cstr = CString::new(file_path).unwrap();
        let result = unsafe { ffi::save_model(file_path_cstr.as_ptr(), model.c_obj) };
        if result == -1 {
            return Err(ModelError::SerializationError(
                "ffi::save_model() returned '-1' - check `stderr` for more information".to_owned(),
            ));
        }

        Ok(())
    }
}

macro_rules! impl_tryfrom_for_solver {
    ($solver: ident) => {
        impl TryFrom<Model<GenericSolver>> for Model<$solver> {
            type Error = ModelError;

            fn try_from(mut value: Model<GenericSolver>) -> Result<Self, Self::Error> {
                {
                    let model = &value as &dyn ModelBase;
                    if model.solver() != $solver::ordinal() {
                        return Err(ModelError::InvalidConversion(format!(
                            "conversion only possible into a model with '{:?}' solver",
                            model.solver()
                        )));
                    }
                }

                assert!(
                    value.training_storage.is_none(),
                    "model with generic solver has backing store"
                );

                let c_obj = value.c_obj;
                value.c_obj = std::ptr::null_mut();

                Ok(Model {
                    _solver: std::marker::PhantomData,
                    training_storage: None,
                    learned_labels: value.learned_labels.clone(),
                    c_obj,
                })
            }
        }
    };
}

impl_tryfrom_for_solver!(L2R_LR);
impl_tryfrom_for_solver!(L2R_L2LOSS_SVC_DUAL);
impl_tryfrom_for_solver!(L2R_L2LOSS_SVC);
impl_tryfrom_for_solver!(L2R_L1LOSS_SVC_DUAL);
impl_tryfrom_for_solver!(MCSVM_CS);
impl_tryfrom_for_solver!(L1R_L2LOSS_SVC);
impl_tryfrom_for_solver!(L1R_LR);
impl_tryfrom_for_solver!(L2R_LR_DUAL);
impl_tryfrom_for_solver!(L2R_L2LOSS_SVR);
impl_tryfrom_for_solver!(L2R_L2LOSS_SVR_DUAL);
impl_tryfrom_for_solver!(L2R_L1LOSS_SVR_DUAL);
impl_tryfrom_for_solver!(ONECLASS_SVM);
