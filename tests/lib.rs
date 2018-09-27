extern crate liblinear;

use liblinear::*;

#[test]
fn test_version() {
	assert_eq!(liblinear_version(), 220);
}

#[test]
fn test_training_input() {
	let libsvm_data = util::TrainingInput::from_libsvm_file("tests/data/heart_scale").unwrap();
	assert_eq!(libsvm_data.len_data(), 270);
	assert_eq!(libsvm_data.len_features(), 13);

	let instances = libsvm_data.yield_data();
	let instance_218 = instances.get(217).unwrap();
	assert_eq!(instance_218.label(), -1f64);
	assert_eq!(instance_218.features().get(3).unwrap().0, 4);
	assert_eq!(instance_218.features().get(3).unwrap().1, -1f64);
	assert_eq!(instance_218.features().get(4).unwrap().1, -0.538813f64);
}