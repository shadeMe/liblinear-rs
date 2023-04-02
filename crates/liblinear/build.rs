extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .flag("-O2")
        .file("vendor/liblinear/linear.cpp")
        .file("vendor/liblinear/newton.cpp")
        .file("vendor/liblinear/blas/daxpy.c")
        .file("vendor/liblinear/blas/ddot.c")
        .file("vendor/liblinear/blas/dnrm2.c")
        .file("vendor/liblinear/blas/dscal.c")
        .include("vendor/liblinear")
        .include("vendor/liblinear/blas")
        .compile("liblinear.a");
}
