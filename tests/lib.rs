extern crate liblinear;

use liblinear::*;

#[test]
fn test_version() {
	assert_eq!(liblinear_version(), 220);
}