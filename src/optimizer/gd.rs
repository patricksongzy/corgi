//! A gradient descent optimizer, which updates based on a learning rate.

use crate::array::*;
use crate::numbers::*;
use crate::optimizer::Optimizer;

#[cfg(feature = "blas")]
use crate::blas::daxpy_blas;

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
        let mut frozen = Vec::new();
        let mut parameter_values = Vec::new();
        let mut parameter_gradients = Vec::new();
        parameters
            .iter()
            .filter(|p| {
                let gradient = p.gradient();
                if gradient.is_none() {
                    frozen.push(true);
                    false
                } else {
                    frozen.push(false);
                    true
                }
            })
            .for_each(|p| {
                parameter_values.extend(p.values());
                parameter_gradients.extend(p.replace_gradient().unwrap().values());
            });

        #[cfg(not(feature = "blas"))]
        parameter_values
            .iter_mut()
            .zip(parameter_gradients)
            .for_each(|(x, g): (&mut Float, Float)| {
                *x -= self.learning_rate * g;
            });
        #[cfg(feature = "blas")]
        daxpy_blas(
            -self.learning_rate,
            &parameter_gradients,
            &mut parameter_values,
        );

        parameters
            .into_iter()
            .zip(frozen)
            .filter(|(_, f)| !f)
            .for_each(|(p, _)| {
                *p = Array::from((
                    p.dimensions().clone(),
                    parameter_values
                        .drain(0..p.values().len())
                        .collect::<Vec<Float>>(),
                ))
                .tracked();
            });
    }
}
