//! A max pooling neural network layer, which applies y = x.pool().

use crate::array::*;
use crate::layer::Layer;

/// A max pooling neural network layer, storing the parameters of the layer.
pub struct Pool {
    pool_dimensions: (usize, usize),
    stride_dimensions: (usize, usize),
}

impl Pool {
    /// Constructs a new convolutional layer, with given dimensions.
    /// The filter dimensions are filter count by image depth by filter rows by filter columns.
    pub fn new(
        pool_dimensions: (usize, usize),
        stride_dimensions: (usize, usize),
    ) -> Pool {
        Pool {
            pool_dimensions,
            stride_dimensions,
        }
    }
}

impl Layer for Pool{
    fn forward(&self, input: Array) -> Array {
        input.pool(self.pool_dimensions, self.stride_dimensions)
    }

    fn parameters(&mut self) -> Vec<&mut Array> {
        Vec::new()
    }
}
