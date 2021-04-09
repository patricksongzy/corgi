//! An activation function is a non-linear function applied to the output of a layer.

use crate::array::*;

use std::sync::Arc;

/// An activation function, which is applied to the output of a layer, and implements the differentiable
/// activation operation.
pub type Activation = Arc<dyn Fn(Array) -> Array>;
