//! An optimizer updates the parameters of a model.

use crate::array::Array;

/// An optimizer, which updates the parameters of a model.
pub trait Optimizer {
    /// Updates the parameters.
    fn update(&self, parameters: Vec<&mut Array>);
}
