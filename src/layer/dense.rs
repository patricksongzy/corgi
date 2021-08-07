//! A fully-connected neural network layer, which applies y = activation(Ax + b).

use crate::activation::Activation;
use crate::array::*;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::numbers::*;

/// A fully-connected neural network layer, storing the parameters of the layer.
pub struct Dense<'a> {
    weights: Array,
    biases: Array,
    activation: Option<&'a Activation>,
}

impl<'a> Dense<'a> {
    /// Constructs a new dense layer, with a given input, and output size.
    pub fn new(
        input_size: usize,
        output_size: usize,
        initializer: &'_ Initializer,
        activation: Option<&'a Activation>,
    ) -> Dense<'a> {
        Dense {
            weights: Array::from((
                vec![output_size, input_size],
                (0..input_size * output_size)
                    .map(|_| (*initializer)(input_size as Float))
                    .collect::<Vec<Float>>(),
            ))
            .tracked(),
            biases: Array::from((
                vec![output_size],
                (0..output_size)
                    .map(|_| (*initializer)(input_size as Float))
                    .collect::<Vec<Float>>(),
            ))
            .tracked(),
            activation,
        }
    }
}

impl Layer for Dense<'_> {
    fn forward(&self, input: Array) -> Array {
        let result = Array::matmul((&input, false), (&self.weights, true), Some(&self.biases));
        match &self.activation {
            Some(f) => f(result),
            None => result,
        }
    }

    fn parameters(&mut self) -> Vec<&mut Array> {
        vec![&mut self.weights, &mut self.biases]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::activation;
    use crate::initializer;

    #[test]
    fn test_smoke() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let learning_rate = 0.01;
        let input_size = 1;
        let hidden_size = 16;
        let output_size = 1;
        let initializer = initializer::he();
        let sigmoid = activation::sigmoid();
        let mut l1 = Dense::new(input_size, hidden_size, &initializer, Some(&sigmoid));
        let mut l2 = Dense::new(hidden_size, output_size, &initializer, None);

        for _ in 0..8 {
            let x = rng.gen_range(-1.0..1.0);
            let input = arr![arr![x]];
            let target = x.exp();

            let r1 = l1.forward(input);
            let r2 = l2.forward(r1);

            let error = (&arr![target] - &r2).powf(2.0);

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
