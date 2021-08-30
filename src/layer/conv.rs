//! A convolutional neural network layer, which applies y = activation(x.conv(filters) + b).

use crate::activation::Activation;
use crate::array::*;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::numbers::*;

/// A convolutional neural network layer, storing the parameters of the layer.
pub struct Conv {
    stride_dimensions: (usize, usize),
    filters: Array,
    biases: Array,
    activation: Option<Activation>,
}

impl Conv {
    /// Constructs a new convolutional layer, with given dimensions.
    /// The filter dimensions are filter count by image depth by filter rows by filter columns.
    pub fn new(
        filter_dimensions: (usize, usize, usize, usize),
        stride_dimensions: (usize, usize),
        initializer: &Initializer,
        activation: Option<Activation>,
    ) -> Conv {
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

impl Layer for Conv {
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
    use crate::{activation, initializer};

    #[test]
    fn test_smoke() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let learning_rate = 0.01;

        let (image_depth, image_rows, image_cols) = (3, 9, 9);
        let image_dimensions = vec![image_depth, image_rows, image_cols];
        let output_dimensions = vec![1, 2, 2];
        let input_size = image_dimensions.iter().product();
        let output_size = output_dimensions.iter().product();

        let initializer = initializer::he();
        let activation = activation::relu();
        let mut l1 = Conv::new(
            (16, image_depth, 3, 3),
            (2, 2),
            &initializer,
            Some(activation),
        );
        let mut l2 = Conv::new((1, 16, 2, 2), (2, 2), &initializer, None);

        for _ in 0..8 {
            let input = Array::from((
                image_dimensions.clone(),
                (0..input_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect::<Vec<Float>>(),
            ));

            let target = Array::from((
                output_dimensions.clone(),
                (0..output_size)
                    .map(|i| input[i] / 2.0)
                    .collect::<Vec<Float>>(),
            ));

            let r1 = l1.forward(input);
            let r2 = l2.forward(r1);

            let error = (&target - &r2).powf(2.0);

            error.backward(None);

            let mut parameters = l1.parameters();
            parameters.append(&mut l2.parameters());

            for parameter in parameters {
                let gradient = parameter.gradient().to_owned().unwrap();
                parameter.stop_tracking();
                *parameter = &*parameter - &(&gradient * learning_rate);
                parameter.start_tracking();
                *parameter.gradient_mut() = None;
            }
        }
    }
}
