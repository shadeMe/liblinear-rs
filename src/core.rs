use ffi;

#[derive(Debug, PartialEq)]
pub enum ErrorKind {
	InvalidData(String),
}

pub struct Problem {
	pub bound: ffi::LLProblem
}

impl Problem {
	pub fn new(labels: Vec<f64>, features: Vec<Vec<f64>>, bias: f64) -> Result<Problem, ErrorKind> {
		if labels.len() != features.len() {
			return Err(ErrorKind::InvalidData(
				"Mismatch between number of training instances and output labels".to_string(),
			));
		} else if labels.len() == 0 || features.len() == 0 {
			return Err(ErrorKind::InvalidData("No input data".to_string()));
		}

		let num_training_instances = labels.len() as i32;
		let num_features = features[0].len() as i32;
		let has_bias = bias >= 0f64;

		let mut transformed_features: Vec<*const ffi::LLFeatureNode> = features
			.iter()
			.zip(1..=num_features)
			.map(|(v, i)| {
				v.iter()
					.map(|v| ffi::LLFeatureNode {
						index: i as i32,
						value: *v,
					})
					.collect()
			})
			.map(|mut v: Vec<ffi::LLFeatureNode>| {
				if has_bias {
					let index = (v.len() + 1) as i32;
					v.push(ffi::LLFeatureNode { index, value: bias });
				}

				v.push(ffi::LLFeatureNode {
					index: -1,
					value: 0f64,
				});
				v.as_ptr()
			})
			.collect();

		/* ### TODO don't we have to save the transformed input and output labels in the struct?
					otherwise the backing store vecs will be dropped at the end of this function?
		*/
		let out = Ok(Problem {
			bound: ffi::LLProblem {
				l: num_training_instances as i32,
				n: num_features + if has_bias { 1 } else { 0 } as i32,
				y: labels.as_ptr(),
				x: transformed_features.as_ptr(),
				bias,
			},
		});

		::std::mem::drop(labels);

		out
	}
}

pub struct Model {
	bound: *mut ffi::LLModel,
}

impl Model {}

#[cfg(test)]
mod tests {
	use super::*;

	fn take_ownership(x: Vec<f64>) -> *const f64 {
		x.as_ptr()
	}

	#[test]
	fn test_simple_drop() {
		let mut x = vec![
			1.0,
			1.0,
			1.0
		];
		let ptr = take_ownership(x);
		unsafe {
			assert_eq!(1.0, *ptr.offset(2));
		}
	}

	#[test]
	fn test_problem_drop() {
		let mut x = vec![
			vec![0.1, 0.2, 0.3, 0.4],
			vec![0.5, 0.6, 0.7, 0.8],
			vec![0.9, 1.0, 1.1, 1.2],
		];
		let mut y = vec![
			1.0,
			2.0,
			2.0
		];
		let prob = Problem::new(y, x, 1.2).unwrap();
		unsafe {
			assert_eq!(2.0, *prob.bound.y.offset(2));
		}
	}
}