extern crate gcc;
extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
	println!("cargo:rustc-flags=-l dylib=stdc++");
	gcc::compile_library("liblinear.a", &["liblinear/linear.cpp"]);

	let bindings = bindgen::Builder::default()
		.header("liblinear/linear.h")
		.generate()
		.expect("Unable to generate bindings!");

	let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
	bindings.write_to_file(out_path.join("bindings.rs")).expect("Couldn't write bindings!");
}