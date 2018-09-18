use core::*;
use util::*;
use ffi::*;

#[derive(Default)]
pub struct Trainer {
	input_data: Result<train::TrainingInput, train::TrainingInputError>,
	bias: f64,
}

impl ProblemBuilder {
	pub fn new() -> ProblemBuilder {
		ProblemBuilder::default()
	}

	pub fn bias(&mut self, bias: f64) -> &mut self {
		self.bias = bias;
		self
	}

	pub fn input(&mut self, labels: Vec<f64>, dense_features: Vec<Vec<f64>>) -> &mut self {
		self.input_data = train::TrainingInput::from_dense_features(labels, dense_features);

		self
	}

	pub fn input(&mut self, path_to_libsvm_data: &str) -> &mut self {

	}
}

pub struct ClassifierBuilder {

}

