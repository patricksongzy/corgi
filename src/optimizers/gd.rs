//! A gradient descent optimizer, which updates based on a learning rate.

use crate::array::*;
use crate::numbers::*;
use crate::optimizer::Optimizer;

/// A gradient descent optimizer, which stores the parameters it updates, and the learning rate.
pub struct GradientDescent {
    learning_rate: Float,
}

impl GradientDescent {
    /// Creates a new gradient descent optimizer, which updates based on the learning rate.
    pub fn new(learning_rate: Float) -> GradientDescent {
        GradientDescent { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    // TODO it is possible that an incorrect parameters are passed in
    fn update(&self, parameters: Vec<&mut Array>) {
        for parameter in parameters {
            let mut gradient = parameter.gradient();
            *parameter =
                parameter.untracked() - (gradient.untracked() * self.learning_rate).untracked();
        }
    }
}
