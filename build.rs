extern crate gcc;

fn main() {
    println!("cargo:rustc-flags=-l dylib=stdc++");

    gcc::Build::new()
        .file("liblinear/linear.cpp")
        .compile("liblinear.a");
}
