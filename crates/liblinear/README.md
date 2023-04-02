[![Latest Version]][crates.io]
[![deps.svg]][deps]
[![docs]][docs.rs]
![MIT]

# liblinear

Rust bindings for the [LIBLINEAR](https://github.com/cjlin1/liblinear) C/C++ library.
Provides a thin (but rustic) wrapper around the original C-interface exposed by the library.

# Usage

```rust
use liblinear::{
    Model,
    Parameters,
    TrainingInput,
    PredictionInput,
    solver::L2R_LR,
    model::traits::*,
    parameter::traits::*,
    solver::traits::*,
};

let x: Vec<Vec<(u32, f64)>> = vec![
    vec![(1, 0.1), (3, 0.2)],
    vec![(3, 9.9)],
    vec![(1, 0.2), (2, 3.2)],
];
let y = vec![0.0, 1.0, 0.0];

let mut params = Parameters::<L2R_LR>::default();
params
    .bias(0f64)
    .stopping_tolerance(0.1f64)
    .constraints_violation_cost(0.1f64);

let model = Model::train(&TrainingInput::from_sparse_features(y, x).unwrap(), &params).unwrap();

let predicted_class = model
    .predict(&PredictionInput::from_sparse_features(vec![(3u32, 9.9f64)]).unwrap())
    .unwrap();
println!("{}", predicted_class);
```

Please refer to the [API docs][docs.rs] for more information.

# Changelog

Please refer to the detailed [changelog](CHANGELOG.md).

[latest version]: https://img.shields.io/crates/v/liblinear.svg
[crates.io]: https://crates.io/crates/liblinear
[mit]: https://img.shields.io/badge/license-MIT-blue.svg
[docs]: https://docs.rs/liblinear/badge.svg
[docs.rs]: https://docs.rs/crate/liblinear/
[deps]: https://deps.rs/repo/github/shademe/liblinear-rs
[deps.svg]: https://deps.rs/repo/github/shademe/liblinear-rs/status.svg
