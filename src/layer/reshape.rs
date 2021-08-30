//! A max pooling neural network layer, which applies y = x.pool().

use crate::array::*;
use crate::layer::Layer;

/// A max pooling neural network layer, storing the parameters of the layer.
pub struct Reshape {
    reshape_dimensions: Vec<usize>,
    reshape_length: usize,
}

impl Reshape {
    /// Constructs a new convolutional layer, with given dimensions.
    /// The filter dimensions are filter count by image depth by filter rows by filter columns.
    pub fn new(
        reshape_dimensions: Vec<usize>,
    ) -> Reshape {
        let reshape_length = reshape_dimensions.iter().product();
        Reshape {
            reshape_dimensions,
            reshape_length,
        }
    }
}

impl Layer for Reshape {
    fn forward(&self, input: Array) -> Array {
        let batch_size = input.values().len() / self.reshape_length;
        let mut dimensions = vec![batch_size];
        dimensions.extend(self.reshape_dimensions.iter());
        input.reshape(&dimensions)
    }

    fn parameters(&mut self) -> Vec<&mut Array> {
        Vec::new()
    }
}
