//! A collection of functions for which neural networks require.

use crate::array::*;
use crate::numbers::*;

use std::sync::Arc;

/// Initializers initialize the parameters of a model.
pub mod initializer {
    use super::*;

    use rand::Rng;

    /// A parameter initializer, which intializes parameters based on the input size.
    pub type Initializer = Arc<dyn Fn(Float) -> Float>;

    /// Creates a He initializer closure.
    pub fn make_he() -> Initializer {
        Arc::new(|x| {
            let stddev = (2.0 / x).sqrt();
            rand::thread_rng().gen_range(-stddev..=stddev)
        })
    }
}

/// Activation functions are differentiable non-linearities applied to the output of layers.
pub mod activation {
    use super::*;

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
}

/// Cost functions compute the loss given a target, and are used for the backward pass.
pub mod cost {
    use super::*;

    /// A cost function, which computes the loss given a target. The cost function takes in the output
    /// as the first argument, and the target as the second.
    pub type CostFunction = Arc<dyn Fn(&Array, &Array) -> Array>;

    /// Creates a mean square error loss closure.
    pub fn make_mse() -> CostFunction {
        Arc::new(|output, target| {
            let length: usize = output.dimensions().iter().product();
            (1.0 / length as Float) * &(target - output).powf(2.0)
        })
    }

    /// Creates a cross-entropy loss closure.
    pub fn make_cross_entropy() -> CostFunction {
        Arc::new(|output, target| &(-target) * &output.ln())
    }
}
