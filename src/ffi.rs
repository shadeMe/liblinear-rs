use std::os::raw::c_char;

#[repr(C)]
pub struct LLFeatureNode {
	pub index: i32,
	pub value: f64,
}

#[repr(C)]
pub struct LLProblem {
	pub l: i32,
	pub n: i32,
	pub y: *const f64,
	pub x: *const *const LLFeatureNode,
	pub bias: f64,
}

pub enum LLSolverType {
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
}

#[allow(non_snake_case)]
#[repr(C)]
pub struct LLParameter {
	pub solver_type: i32,

	pub eps: f64,
	pub C: f64,
	pub nr_weight: i32,
	pub weight_label: *const i32,
	pub weight: *const f64,
	pub p: f64,
	pub init_sol: *const f64,
}

#[repr(C)]
pub struct LLModel {
	pub param: LLParameter,
	pub nr_class: i32,
	pub nr_feature: i32,
	pub w: *mut f64,
	pub label: *mut i32,
	pub bias: f64,
}

extern "C" {
	pub static liblinear_version: i32;

	pub fn train(prob: *const LLProblem, param: *const LLParameter) -> *mut LLModel;
	pub fn cross_validation(
		prob: *const LLProblem,
		param: *const LLParameter,
		nr_fold: i32,
		target: *mut f64,
	);
	pub fn find_parameter_C(
		prob: *const LLProblem,
		param: *const LLParameter,
		nr_fold: i32,
		start_C: f64,
		max_C: f64,
		best_C: *mut f64,
		best_rate: *mut f64,
	);

	pub fn predict_values(
		model_: *const LLModel,
		x: *const LLFeatureNode,
		dec_values: *mut f64,
	) -> f64;
	pub fn predict(model_: *const LLModel, x: *const LLFeatureNode) -> f64;
	pub fn predict_probability(
		model_: *const LLModel,
		x: *const LLFeatureNode,
		prob_estimates: *mut f64,
	);

	pub fn save_model(model_file_name: *const c_char, model_: *const LLModel) -> i32;
	pub fn load_model(model_file_name: *const c_char) -> *mut LLModel;

	pub fn get_nr_feature(model_: *const LLModel) -> i32;
	pub fn get_nr_class(model_: *const LLModel) -> i32;
	pub fn get_labels(model_: *const LLModel, label: *mut i32);
	pub fn get_decfun_coef(model_: *const LLModel, feat_idx: i32, label_idx: i32) -> f64;
	pub fn get_decfun_bias(model_: *const LLModel, label_idx: i32) -> f64;

	pub fn free_model_content(model_: *mut LLModel);
	pub fn free_and_destroy_model(model_: *mut *mut LLModel);
	pub fn destroy_param(param: *mut LLParameter);

	pub fn check_parameter(prob: *const LLProblem, param: *const LLParameter) -> *const c_char;
	pub fn check_probability_model(model_: *const LLModel) -> i32;
	pub fn check_regression_model(model_: *const LLModel) -> i32;
}
