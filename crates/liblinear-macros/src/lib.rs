use proc_macro::{self, TokenStream};
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(IsLogisticRegressionSolver)]
pub fn derive_is_logistic_regression_solver(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, .. } = parse_macro_input!(input);
    let output = quote! {
        impl traits::IsLogisticRegressionSolver for #ident {}
    };
    output.into()
}

#[proc_macro_derive(IsSingleClassSolver)]
pub fn derive_is_single_class_solver(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, .. } = parse_macro_input!(input);
    let output = quote! {
        impl traits::IsSingleClassSolver for #ident {}
    };
    output.into()
}

#[proc_macro_derive(IsNonSingleClassSolver)]
pub fn derive_is_non_single_class_solver(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, .. } = parse_macro_input!(input);
    let output = quote! {
        impl traits::IsNonSingleClassSolver for #ident {}
    };
    output.into()
}

#[proc_macro_derive(IsTrainableSolver)]
pub fn derive_is_trainable_solver(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, .. } = parse_macro_input!(input);
    let output = quote! {
        impl traits::IsTrainableSolver for #ident {}
    };
    output.into()
}

#[proc_macro_derive(CanDisableBiasRegularization)]
pub fn derive_can_disable_bias_regularization(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, .. } = parse_macro_input!(input);
    let output = quote! {
        impl traits::CanDisableBiasRegularization for #ident {}
    };
    output.into()
}

#[proc_macro_derive(SupportsInitialSolutions)]
pub fn derive_supports_initial_solutions(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, .. } = parse_macro_input!(input);
    let output = quote! {
        impl traits::SupportsInitialSolutions for #ident {}
    };
    output.into()
}

#[proc_macro_derive(SupportsParameterSearch)]
pub fn derive_supports_parameter_search(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, .. } = parse_macro_input!(input);
    let output = quote! {
        impl traits::SupportsParameterSearch for #ident {}
    };
    output.into()
}
