use approx::abs_diff_eq;
use liblinear::{
    model::Model,
    model::{serde, traits::*},
    parameter::Parameters,
    solver::{L1R_LR, L2R_L2LOSS_SVC_DUAL, L2R_LR, MCSVM_CS},
    util::{PredictionInput, TrainingInput},
};
use parsnip::categorical_accuracy;

fn create_default_training_data() -> TrainingInput {
    TrainingInput::from_libsvm_file("tests/data/heart_scale")
        .expect("couldn't read training data from disk")
}

#[test]
fn test_version() {
    assert_eq!(liblinear::liblinear_version(), 246);
}

#[test]
fn test_training_input_libsvm_data() {
    let libsvm_data = create_default_training_data();
    assert_eq!(libsvm_data.len(), 270);
    assert_eq!(libsvm_data.dim(), 13);

    {
        let instance_218 = libsvm_data.get(217).unwrap();
        assert_eq!(instance_218.label(), -1f64);
        assert_eq!(instance_218.features().get(3).unwrap().0, 4);
        assert_eq!(instance_218.features().get(3).unwrap().1, -1f64);
        assert_eq!(instance_218.features().get(4).unwrap().1, -0.538813f64);
    }

    let mut params = Parameters::<L1R_LR>::default();
    params.stopping_tolerance(0.11f64).bias(12f64);
    let model = Model::train(&libsvm_data, &params).expect("couldn't train model");

    assert_eq!(model.num_classes(), 2);
    let class = model
        .predict(
            &PredictionInput::from_dense_features(vec![
                -0.5, -1.0, 0.333333, -0.660377, -0.351598, -1.0, 1.0, 0.541985, 1.0, -1.0, -1.0,
                -1.0, -1.0,
            ])
            .expect("couldn't generate input from dense features"),
        )
        .expect("couldn't predict using model");
    assert_eq!(class, -1f64);
}

#[test]
fn test_model_sparse_data() {
    let x: Vec<Vec<(u32, f64)>> = vec![
        vec![(1, 0.1), (3, 0.2)],
        vec![(3, 9.9)],
        vec![(1, 0.2), (2, 3.2)],
    ];
    let y = vec![0.0, 1.0, 0.0];

    let mut params = Parameters::<L2R_LR>::default();
    params
        .bias(0f64)
        .stopping_tolerance(0.1f64)
        .constraints_violation_cost(0.1f64)
        .regression_loss_sensitivity(1f64);

    let model = Model::train(
        &TrainingInput::from_sparse_features(y, x).expect("couldn't generate training data"),
        &params,
    )
    .expect("couldn't train model");
    assert_eq!(model.num_classes(), 2);

    let class = model
        .predict(
            &PredictionInput::from_sparse_features(vec![(3u32, 9.9f64)])
                .expect("couldn't generate input from sparse features"),
        )
        .expect("couldn't predict using model");
    assert_eq!(class, 1f64);
}

#[test]
fn test_model_dense_data() {
    let x = vec![
        vec![1.1, 0.0, 8.4],
        vec![0.9, 1.0, 9.1],
        vec![1.2, 1.0, 9.0],
    ];
    let y = vec![0.0, 1.0, 3.0];

    let mut params = Parameters::<MCSVM_CS>::default();
    params.bias(0f64);

    let model = Model::train(
        &TrainingInput::from_dense_features(y, x).expect("couldn't generate training data"),
        &params,
    )
    .expect("couldn't train model");
    assert_eq!(model.num_classes(), 3);

    let class = model
        .predict(
            &PredictionInput::from_dense_features(vec![1.2, 1.0, 9.0])
                .expect("couldn't generate input from sparse features"),
        )
        .expect("couldn't predict using model");
    assert_eq!(class, 3f64);
}

#[test]
fn test_model_save_load() {
    let libsvm_data = create_default_training_data();
    let mut params = Parameters::<L2R_LR>::default();
    params.bias(10.2);
    let model = Model::train(&libsvm_data, &params).expect("couldn't train model");

    let model_labels = model.labels().clone();
    assert_eq!(model.num_classes(), 2);

    serde::save_model_to_disk(&model, "tests/data/heart_scale.dat")
        .expect("couldn't save model to disk");
    let model = serde::load_model_from_disk("tests/data/heart_scale.dat")
        .expect("couldn't load model from disk");

    assert_eq!(model.num_classes(), 2);
    assert_eq!(model.bias(), 10.2);
    assert_eq!(model.labels(), &model_labels);

    let _: Model<L2R_LR> = model
        .try_into()
        .expect("couldn't convert model to correct type");

    let model = serde::load_model_from_disk("tests/data/heart_scale.dat")
        .expect("couldn't load model from disk");
    let converted: Result<Model<L2R_L2LOSS_SVC_DUAL>, _> = model.try_into();
    assert_eq!(converted.is_err(), true);
}

#[test]
fn test_cross_validator() {
    liblinear::toggle_liblinear_stdout_output(false);

    let libsvm_data = create_default_training_data();
    let params = Parameters::<L2R_LR>::default();

    let ground_truth = vec![
        1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0,
        1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
        1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
        1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
        1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0,
        1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
        1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
        1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
        -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
        -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
        1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
    ]
    .iter()
    .map(|e| *e as i32)
    .collect::<Vec<i32>>();

    let predicted = Model::cross_validation(&libsvm_data, &params, 4)
        .expect("couldn't cross validate")
        .iter()
        .map(|e| *e as i32)
        .collect::<Vec<i32>>();

    // RHS was taken from the output of liblinear's bundled trainer program
    let _ = abs_diff_eq!(
        categorical_accuracy(&predicted, &ground_truth).unwrap(),
        0.8148148
    );

    let (best_c, acc, best_p) =
        Model::find_optimal_constraints_violation_cost_and_loss_sensitivity(
            &libsvm_data,
            &params,
            4,
            0.0,
            0.0,
        )
        .expect("couldn't perform parameter search");
    let _ = abs_diff_eq!(best_c, 0.00390625);
    let _ = abs_diff_eq!(acc, 0.8407407);
    assert_eq!(best_p, -1f64);
}
