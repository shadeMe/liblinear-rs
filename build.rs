extern crate gcc;
extern crate bindgen;

fn main() {
	println!("cargo:rustc-flags=-l dylib=stdc++");

	gcc::Build::new()
		.file("liblinear/linear.cpp")
		.compile("liblinear.a");

//	let bindings = bindgen::Builder::default()
//		.header("liblinear/linear.h")
//		.generate()
//		.expect("Unable to generate bindings!");

//	bindings.write_to_file("src/bindings/mod.rs").expect("Couldn't write bindings!");
}