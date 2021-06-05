//! Activation functions are differentiable non-linearities applied to the output of layers.

use crate::array::*;

use std::sync::Arc;

/// An activation function, which is applied to the output of a layer, and implements the differentiable
/// activation operation.
pub type Activation = Arc<dyn Fn(Array) -> Array>;

/// Creates a ReLU activation function closure.
pub fn make_relu() -> Activation {
    Arc::new(|x| x.relu())
}

/// Creates a sigmoid activation function closure.
pub fn make_sigmoid() -> Activation {
    Arc::new(|x| x.sigmoid())
}

/// Creates a softmax activation function closure.
pub fn make_softmax() -> Activation {
    Arc::new(|x| x.softmax())
}
