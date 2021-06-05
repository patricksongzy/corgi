//! Implementations of gradient descent optimizers, to optimize the parameters of a model.

pub mod gd;

use crate::array::Array;

/// An optimizer, which updates the parameters of a model.
pub trait Optimizer {
    /// Updates the parameters. It is critical that the order of the parameters remains the same between calls.
    fn update(&self, parameters: Vec<&mut Array>);
}
