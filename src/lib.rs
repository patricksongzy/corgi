#![warn(missing_docs)]

//! Machine learning, and automatic differentiation implementation.

pub mod numbers;
#[macro_use]
pub mod array;
pub mod dense;
pub mod layer;
pub mod model;
