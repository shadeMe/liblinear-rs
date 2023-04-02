# Changelog

## 2.0.0

### Breaking

- Rewrite public API to utilize compile-time validation of models and parameters.
- Replace `failure` with `thiserror`.
- Update LIBLINEAR to `v246`.
- Expose one-class SVM solver.
- Reorganize as `cargo` workspace.
  - Add `liblinear-macros` crate.

### Non-breaking

- Use `-O2` optimizations when compiling LIBLINEAR.

## 1.0.0

### Breaking

- Update LIBLINEAR to `v230`.
- `ParameterError` - Add variant `IllegalArgument`.
- `LibLinearCrossValidator` - Change `find_optimal_constraints_violation_cost` to `find_optimal_constraints_violation_cost_and_loss_sensitivity` signature.

### Non-breaking

- Fix build failures on macOS by automatically selecting the C++ standard library when compiling LIBLINEAR.
- Update dependencies.

## 0.1.1

- Added readme.
- Minor documentation fixes.

## 0.1.0

- Initial release.
