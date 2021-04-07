//! A layer of a neural network.

use crate::array::*;

/// A layer of a neural network.
pub trait Layer {
    /// Completes the forward pass of the layer.
    fn forward(&self, x: Array) -> Array;

    /// Updates the parameters of the layer.
    fn update(&mut self);
}
