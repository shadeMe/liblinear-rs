use approx::abs_diff_eq;
use liblinear::parameter::LibLinearParameter;
use liblinear::{
    model::{LibLinearCrossValidator, LibLinearModel},
    parameter::{HasLibLinearParameter, SolverType},
    util::{PredictionInput, TrainingInput},
    Builder, Serializer,
};
use parsnip::categorical_accuracy;

fn create_default_model_builder() -> Builder {
    let libsvm_data = TrainingInput::from_libsvm_file("tests/data/heart_scale").unwrap();

    let mut model_builder = Builder::default();
    model_builder.problem().input_data(libsvm_data);
    model_builder.parameters().solver_type(SolverType::L1R_LR);
    model_builder
}

#[test]
fn test_version() {
    assert_eq!(liblinear::liblinear_version(), 244);
}

#[test]
fn test_training_input_libsvm_data() {
    let libsvm_data = TrainingInput::from_libsvm_file("tests/data/heart_scale").unwrap();
    assert_eq!(libsvm_data.len_data(), 270);
    assert_eq!(libsvm_data.len_features(), 13);

    {
        let instance_218 = libsvm_data.get(217).unwrap();
        assert_eq!(instance_218.label(), -1f64);
        assert_eq!(instance_218.features().get(3).unwrap().0, 4);
        assert_eq!(instance_218.features().get(3).unwrap().1, -1f64);
        assert_eq!(instance_218.features().get(4).unwrap().1, -0.538813f64);
    }

    let mut model_builder = liblinear::Builder::default();
    model_builder.problem().input_data(libsvm_data).bias(0f64);
    model_builder.parameters().solver_type(SolverType::L1R_LR);

    let model = model_builder.build_model();
    assert_eq!(model.is_ok(), true);

    let model = model.unwrap();
    assert_eq!(model.num_classes(), 2);

    let class = model
        .predict(
            PredictionInput::from_dense_features(vec![
                -0.5, -1.0, 0.333333, -0.660377, -0.351598, -1.0, 1.0, 0.541985, 1.0, -1.0, -1.0,
                -1.0, -1.0,
            ])
            .unwrap(),
        )
        .unwrap();
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

    let mut model_builder = liblinear::Builder::default();
    model_builder
        .problem()
        .input_data(TrainingInput::from_sparse_features(y, x).unwrap())
        .bias(0f64);
    model_builder
        .parameters()
        .solver_type(SolverType::L2R_LR)
        .stopping_tolerance(0.1f64)
        .constraints_violation_cost(0.1f64)
        .regression_loss_sensitivity(1f64);

    let model = model_builder.build_model();
    assert_eq!(model.is_ok(), true);

    let model = model.unwrap();
    assert_eq!(model.num_classes(), 2);

    let class = model
        .predict(PredictionInput::from_sparse_features(vec![(3u32, 9.9f64)]).unwrap())
        .unwrap();
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

    let mut model_builder = liblinear::Builder::default();
    model_builder
        .problem()
        .input_data(TrainingInput::from_dense_features(y, x).unwrap())
        .bias(0f64);
    model_builder.parameters().solver_type(SolverType::MCSVM_CS);

    let model = model_builder.build_model();
    assert_eq!(model.is_ok(), true);

    let model = model.unwrap();
    assert_eq!(model.num_classes(), 3);

    let class = model
        .predict(PredictionInput::from_dense_features(vec![1.2, 1.0, 9.0]).unwrap())
        .unwrap();
    assert_eq!(class, 3f64);
}

#[test]
fn test_model_save_load() {
    let mut model_builder = create_default_model_builder();
    model_builder.problem().bias(10.2);

    let model = model_builder.build_model().unwrap();
    let model_labels = model.labels().clone();
    assert_eq!(model.num_classes(), 2);
    assert_eq!(
        model.parameter().solver_type().is_logistic_regression(),
        true
    );
    assert_eq!(
        Serializer::save_model("tests/data/heart_scale.dat", &model).is_ok(),
        true
    );
    let model = Serializer::load_model("tests/data/heart_scale.dat");
    assert_eq!(model.is_ok(), true);
    let model = model.unwrap();
    assert_eq!(model.num_classes(), 2);
    assert_eq!(model.bias(), 10.2);
    assert_eq!(model.labels(), &model_labels);
}

#[test]
fn test_cross_validator() {
    liblinear::toggle_liblinear_stdout_output(false);

    let mut model_builder = create_default_model_builder();
    model_builder.parameters().solver_type(SolverType::L2R_LR);
    let cross_validator = model_builder.build_cross_validator().unwrap();

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
    let predicted = cross_validator
        .cross_validation(4)
        .unwrap()
        .iter()
        .map(|e| *e as i32)
        .collect::<Vec<i32>>();

    // RHS was taken from the output of liblinear's bundled trainer program
    let _ = abs_diff_eq!(
        categorical_accuracy(&predicted, &ground_truth).unwrap(),
        0.8148148
    );

    let (best_c, acc, best_p) = cross_validator
        .find_optimal_constraints_violation_cost_and_loss_sensitivity(4, 0.0, 0.0)
        .unwrap();
    let _ = abs_diff_eq!(best_c, 0.00390625);
    let _ = abs_diff_eq!(acc, 0.8407407);
    assert_eq!(best_p, -1f64);
}
