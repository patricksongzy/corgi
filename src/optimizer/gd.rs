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
        // let (values, gradients): (Vec<_>, Vec<_>) = parameters.iter().map(|p| {
        //     let gradient = p.gradient();
        //     if let Some(x) = gradient {
        //         (p.values().clone(), x.values().clone())
        //     } else {
        //         (p.values().clone(), vec![0.0; p.values().len()])
        //     }
        // }).unzip();

        // let (mut values, gradients): (Vec<Float>, Vec<Float>) = (values.into_iter().flatten().collect(), gradients.into_iter().flatten().collect());

        // Array::axpy(-self.learning_rate, &gradients, &values);
        // use crate::blas::daxpy_blas;
        // daxpy_blas(-self.learning_rate, &gradients, &mut values);

        for parameter in parameters {
            // *parameter = Arrays::new((parameter.dimensions(), values.drain(0..parameter.values().len()).collect()));
            let gradient = parameter.gradient();
            if let Some(x) = gradient {
                parameter.stop_tracking();
                *parameter = Array::axpy(-self.learning_rate, &x, &*parameter);
                parameter.start_tracking();
                *parameter.gradient_mut() = None;
            }
        }
    }
}
