//! Activation functions are differentiable non-linearities applied to the output of layers.

use crate::array::*;

/// An activation function, which is applied to the output of a layer, and implements the differentiable
/// activation operation.
pub type Activation = Box<dyn Fn(Array) -> Array>;

/// Creates a ReLU activation function closure.
pub fn relu() -> Activation {
    Box::new(|x| x.relu())
}

/// Creates a sigmoid activation function closure.
pub fn sigmoid() -> Activation {
    Box::new(|x| x.sigmoid())
}

/// Creates a softmax activation function closure.
pub fn softmax() -> Activation {
    Box::new(|x| x.softmax())
}
