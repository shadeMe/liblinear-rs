[![Latest Version]][crates.io]
[![deps.svg]][deps]
[![docs]][docs.rs]
![MIT]

# liblinear-rs
Rust bindings for the [liblinear](https://github.com/cjlin1/liblinear) C/C++ library.
Provides a thin (but rustic) wrapper around the original C-interface exposed by the library.


# Usage
Use the `liblinear::Builder` API to train a model on sparse features and
predict the class of a new instance.


```rust
use liblinear::*;

let x: Vec<Vec<(u32, f64)>> = vec![
        vec![(1, 0.1), (3, 0.2)],
        vec![(3, 9.9)],
        vec![(1, 0.2), (2, 3.2)],
    ];
let y = vec![0.0, 1.0, 0.0];

let mut model_builder = liblinear::Builder::default();
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

let model = model_builder.build_model().unwrap();
assert_eq!(model.num_classes(), 2);

let predicted_class = model
    .predict(util::PredictionInput::from_sparse_features(vec![(1u32, 2.2f64)]).unwrap())
    .unwrap();
println!(predicted_class);
```

More examples can be found in the bundled unit tests.


# Changelog
1.0.0 - Update liblinear to v230 (breaking changes), minor changes and fixes.
0.1.1 - Added readme, minor documentation fixes.
0.1.0 - Initial release.


[Latest Version]: https://img.shields.io/crates/v/liblinear.svg
[crates.io]: https://crates.io/crates/liblinear
[MIT]: https://img.shields.io/badge/license-MIT-blue.svg
[docs]: https://docs.rs/liblinear/badge.svg
[docs.rs]: https://docs.rs/crate/liblinear/
[deps]: https://deps.rs/repo/github/shademe/liblinear-rs
[deps.svg]: https://deps.rs/repo/github/shademe/liblinear-rs/status.svg