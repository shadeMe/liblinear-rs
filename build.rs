extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .cpp_link_stdlib("stdc++")
        .flag("-O3")
        .flag("-fPIC")
        .file("liblinear/linear.cpp")
        .file("liblinear/tron.cpp")
        .file("liblinear/blas/daxpy.c")
        .file("liblinear/blas/ddot.c")
        .file("liblinear/blas/dnrm2.c")
        .file("liblinear/blas/dscal.c")
        .include("liblinear")
        .include("liblinear/blas")
        .compile("liblinear.a");
}
