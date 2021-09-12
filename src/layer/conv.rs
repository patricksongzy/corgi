//! A convolutional neural network layer, which applies y = activation(x.conv(filters) + b).

use crate::activation::Activation;
use crate::array::*;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::numbers::*;

/// A convolutional neural network layer, storing the parameters of the layer.
pub struct Conv<'a> {
    stride_dimensions: (usize, usize),
    filters: Array,
    biases: Array,
    activation: Option<&'a Activation>,
}

impl<'a> Conv<'a> {
    /// Constructs a new convolutional layer, with given dimensions.
    /// The filter dimensions are filter count by image depth by filter rows by filter columns.
    pub fn new(
        filter_dimensions: (usize, usize, usize, usize),
        stride_dimensions: (usize, usize),
        initializer: &'_ Initializer,
        activation: Option<&'a Activation>,
    ) -> Conv<'a> {
        let (filter_count, image_depth, filter_rows, filter_cols) = filter_dimensions;

        let filter_dimensions = vec![filter_count, image_depth, filter_rows, filter_cols];
        let filter_size = filter_dimensions.iter().product();
        let input_size = image_depth * filter_rows * filter_cols;

        Conv {
            stride_dimensions,
            filters: Array::from((
                filter_dimensions,
                (0..filter_size)
                    .map(|_| (*initializer)(input_size as Float))
                    .collect::<Vec<Float>>(),
            ))
            .tracked(),
            biases: Array::from((
                vec![filter_count, 1, 1],
                (0..filter_count)
                    .map(|_| (*initializer)(input_size as Float))
                    .collect::<Vec<Float>>(),
            ))
            .tracked(),
            activation,
        }
    }
}

impl Layer for Conv<'_> {
    fn forward(&self, input: Array) -> Array {
        let result = &input.conv(&self.filters, self.stride_dimensions) + &self.biases;
        match &self.activation {
            Some(f) => f(result),
            None => result,
        }
    }

    fn parameters(&mut self) -> Vec<&mut Array> {
        vec![&mut self.filters, &mut self.biases]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{activation, cost, initializer, model::Model, optimizer::gd::GradientDescent};

    #[test]
    fn test_gradient() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let learning_rate = 0.0;

        let batch_size = 2;
        let (image_depth, image_rows, image_cols) = (3, 9, 9);
        let image_dimensions = vec![batch_size, image_depth, image_rows, image_cols];
        let output_dimensions = vec![batch_size, 16, 4, 4];
        let input_size = image_dimensions.iter().product();
        let output_size = output_dimensions.iter().product();

        let initializer = initializer::he();
        let relu = activation::relu();
        let mse = cost::mse();
        let gd = GradientDescent::new(learning_rate);

        // 3x9x9 -> 16x4x4
        let mut l1 = Conv::new(
            (16, image_depth, 3, 3),
            (2, 2),
            &initializer,
            Some(&relu),
        );
        let mut model = Model::new(vec![&mut l1], &gd, &mse);

        let input = Array::from((
            image_dimensions,
            (0..input_size)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect::<Vec<Float>>(),
        ));

        let target = Array::from((
            output_dimensions,
            (0..output_size)
                .map(|_| rng.gen_range(0.0..1.0))
                .collect::<Vec<Float>>(),
        ));

        model.test_gradient(&mse, input, target);
    }
}
