//! Cost functions compute the loss given a target, and are used for the backward pass.

use crate::array::*;
use crate::numbers::*;

/// A cost function, which computes the loss given a target. The cost function takes in the output
/// as the first argument, and the target as the second.
pub type CostFunction = Box<dyn Fn(&Array, &Array) -> Array>;

/// Creates a mean square error loss closure.
pub fn mse() -> CostFunction {
    Box::new(|output, target| {
        let length: usize = output.dimensions().iter().product();
        (1.0 / length as Float) * &(target - output).powf(2.0)
    })
}

/// Creates a cross-entropy loss closure.
pub fn cross_entropy() -> CostFunction {
    Box::new(|output, target| {
        let batch_size: usize = output.dimensions()[0];
        (1.0 / batch_size as Float) * &(&(-target) * &output.ln())
    })
}
