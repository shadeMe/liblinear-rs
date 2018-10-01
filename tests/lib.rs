extern crate approx;
extern crate liblinear;
extern crate parsnip;

use liblinear::*;
use liblinear::util::*;
use parsnip::*;

fn create_default_model_builder() -> Builder {
	let libsvm_data = util::TrainingInput::from_libsvm_file("tests/data/heart_scale").unwrap();

	let mut model_builder = liblinear::Builder::new();
	model_builder.problem().input_data(libsvm_data);
	model_builder.parameters().solver_type(SolverType::L1R_LR);
	model_builder
}

#[test]
fn test_version() {
	assert_eq!(liblinear_version(), 220);
}

#[test]
fn test_training_input_libsvm_data() {
	let libsvm_data = util::TrainingInput::from_libsvm_file("tests/data/heart_scale").unwrap();
	assert_eq!(libsvm_data.len_data(), 270);
	assert_eq!(libsvm_data.len_features(), 13);

	{
		let instance_218 = libsvm_data.get(217).unwrap();
		assert_eq!(instance_218.label(), -1f64);
		assert_eq!(instance_218.features().get(3).unwrap().0, 4);
		assert_eq!(instance_218.features().get(3).unwrap().1, -1f64);
		assert_eq!(instance_218.features().get(4).unwrap().1, -0.538813f64);
	}

	let mut model_builder = liblinear::Builder::new();
	model_builder.problem().input_data(libsvm_data).bias(0f64);
	model_builder.parameters().solver_type(SolverType::L1R_LR);

	let model = model_builder.build_model();
	assert_eq!(model.is_ok(), true);

	let model = model.unwrap();
	assert_eq!(model.num_classes(), 2);

	let class = model
		.predict(
			util::PredictionInput::from_dense_features(vec![
				-0.5, -1.0, 0.333333, -0.660377, -0.351598, -1.0, 1.0, 0.541985, 1.0, -1.0, -1.0,
				-1.0, -1.0,
			]).unwrap(),
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

	let mut model_builder = liblinear::Builder::new();
	model_builder
		.problem()
		.input_data(util::TrainingInput::from_sparse_features(y, x).unwrap())
		.bias(0f64);
	model_builder
		.parameters()
		.solver_type(SolverType::L2R_LR)
		.stopping_criterion(0.1f64)
		.constraints_violation_cost(0.1f64)
		.regression_loss_sensitivity(1f64);

	let model = model_builder.build_model();
	assert_eq!(model.is_ok(), true);

	let model = model.unwrap();
	assert_eq!(model.num_classes(), 2);

	let class = model
		.predict(util::PredictionInput::from_sparse_features(vec![(3u32, 9.9f64)]).unwrap())
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

	let mut model_builder = liblinear::Builder::new();
	model_builder
		.problem()
		.input_data(util::TrainingInput::from_dense_features(y, x).unwrap())
		.bias(0f64);
	model_builder.parameters().solver_type(SolverType::MCSVM_CS);

	let model = model_builder.build_model();
	assert_eq!(model.is_ok(), true);

	let model = model.unwrap();
	assert_eq!(model.num_classes(), 3);

	let class = model
		.predict(util::PredictionInput::from_dense_features(vec![1.1, 0.0, 8.4]).unwrap())
		.unwrap();
	assert_eq!(class, 0f64);
}

#[test]
fn test_model_save_load() {
	let mut model_builder = create_default_model_builder();
	model_builder.problem().bias(10.2);

	let model = model_builder.build_model().unwrap();
	assert_eq!(model.num_classes(), 2);

	assert_eq!(
		Serializer::save_model("tests/data/heart_scale.dat", &model).is_ok(),
		true
	);
	let model = Serializer::load_model("tests/data/heart_scale.dat");
	assert_eq!(model.is_ok(), true);
	let model = model.unwrap();
	assert_eq!(model.num_classes(), 2);
	assert_eq!(model.bias(), 10.2);
}

#[test]
fn test_cross_validator() {
	let mut model_builder = create_default_model_builder();
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
	].iter().map(|e| *e as i32).collect::<Vec<i32>>();
	let predicted = cross_validator.cross_validation(4).unwrap().iter().map(|e| *e as i32).collect::<Vec<i32>>();

	println!("Accuracy: {}", categorical_accuracy(&predicted, &ground_truth).unwrap());
	println!("Precision: {}", precision(&predicted, &ground_truth, Average::Macro).unwrap());
	println!("Recall: {}", recall(&predicted, &ground_truth, Average::Macro).unwrap());
}
