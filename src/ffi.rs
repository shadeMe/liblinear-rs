use std::os::raw::c_char;


#[repr(C)]
pub struct FeatureNode {
    pub index: i32,
    pub value: f64,
}

#[repr(C)]
pub struct Problem {
    pub l: i32,
    pub n: i32,
    pub y: *const f64,
    pub x: *const *const FeatureNode,
    pub bias: f64,
}

pub enum SolverType {
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
pub struct Parameter {
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
pub struct Model {
    pub param: Parameter,
    pub nr_class: i32,
    pub nr_feature: i32,
    pub w: *mut f64,
    pub label: *mut i32,
    pub bias: f64,
}

extern "C" {
    pub static liblinear_version: i32;

    pub fn train(prob: *const Problem, param: *const Parameter) -> *mut Model;
    pub fn cross_validation(
        prob: *const Problem,
        param: *const Parameter,
        nr_fold: i32,
        target: *mut f64,
    );
    pub fn find_parameter_C(
        prob: *const Problem,
        param: *const Parameter,
        nr_fold: i32,
        start_C: f64,
        max_C: f64,
        best_C: *mut f64,
        best_rate: *mut f64,
    );

    pub fn predict_values(model_: *const Model, x: *const FeatureNode, dec_values: *mut f64)
        -> f64;
    pub fn predict(model_: *const Model, x: *const FeatureNode) -> f64;
    pub fn predict_probability(
        model_: *const Model,
        x: *const FeatureNode,
        prob_estimates: *mut f64,
    );

    pub fn save_model(model_file_name: *const c_char, model_: *const Model) -> i32;
    pub fn load_model(model_file_name: *const c_char) -> *mut Model;

    pub fn get_nr_feature(model_: *const Model) -> i32;
    pub fn get_nr_class(model_: *const Model) -> i32;
    pub fn get_labels(model_: *const Model, label: *mut i32);
    pub fn get_decfun_coef(model_: *const Model, feat_idx: i32, label_idx: i32) -> f64;
    pub fn get_decfun_bias(model_: *const Model, label_idx: i32) -> f64;

    pub fn free_model_content(model_: *mut Model);
    pub fn free_and_destroy_model(model_: *mut *mut Model);
    pub fn destroy_param(param: *mut Parameter);

    pub fn check_parameter(prob: *const Problem, param: *const Parameter) -> *const c_char;
    pub fn check_probability_model(model_: *const Model) -> i32;
    pub fn check_regression_model(model_: *const Model) -> i32;
}
