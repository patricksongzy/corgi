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
    fn update(&self, parameters: Vec<&mut Array>) {
        for parameter in parameters {
            let gradient = parameter.gradient();
            if let Some(x) = gradient {
                parameter.stop_tracking();
                // use crate::blas;
                // let mut next = parameter.values().clone();
                // blas::daxpy_blas(-self.learning_rate, &x.values(), &mut next);
                // *parameter = Arrays::new((parameter.dimensions(), next)).tracked();
                // *parameter = &*parameter + &(-self.learning_rate * &x);
                *parameter = Array::daxpy(-self.learning_rate, &x, &*parameter);
                parameter.start_tracking();
                *parameter.gradient_mut() = None;
            }
        }
    }
}
