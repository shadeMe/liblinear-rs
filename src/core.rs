use ffi;
use std;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use util::train;

#[derive(Debug, Fail)]
pub enum ClassifierError {
    /// Parameter errors
    #[fail(display = "parameter error: {}", e)]
    InvalidParameters { e: String },
}


struct Problem {
    backing_store_labels: Vec<f64>,
    backing_store_features: Vec<Vec<ffi::FeatureNode>>,
    backing_store_feature_ptrs: Vec<*const ffi::FeatureNode>,
    bound: ffi::Problem,
}

impl Problem {
    fn new(input_data: train::TrainingInput, bias: f64) -> Result<Problem, ClassifierError> {
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
                            .map(|(index, value)| ffi::FeatureNode { index: *index as i32, value: *value })
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
                    v.push(ffi::FeatureNode { index: last_feature_index, value: bias });
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

struct Parameter {
    backing_store_weights: Vec<f64>,
    backing_store_weight_labels: Vec<i32>,
    backing_store_starting_solutions: Vec<f64>,
    bound: ffi::Parameter,
}

impl Parameter {
    fn new(
        solver: ffi::SolverType,
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
                init_sol: init_sol.as_ptr(),
            },
            backing_store_weights: weights,
            backing_store_weight_labels: weight_labels,
            backing_store_starting_solutions: init_sol,
        };

        unsafe {
            let param_error = ffi::check_parameter(ptr::null::<ffi::Problem>(), &param.bound);
            if param_error != ptr::null::<c_char>() {
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

pub enum Solver {

}

struct Classifier {
    problem: Option<Problem>,
    parameter: Option<Parameter>,
    model: *mut ffi::Model,
}

pub trait Trainer {
    fn
}

pub trait Builder {
    fn train()
}

