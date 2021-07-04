//! A fully-connected neural network layer, which applies y = activation(Ax + b).

use crate::activation::Activation;
use crate::array::*;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::numbers::*;

/// A fully-connected neural network layer, storing the parameters of the layer.
pub struct Dense {
    weights: Array,
    biases: Array,
    activation: Option<Activation>,
}

impl Dense {
    /// Constructs a new dense layer, with a given input, and output size.
    pub fn new(
        input_size: usize,
        output_size: usize,
        initializer: Initializer,
        activation: Option<Activation>,
    ) -> Dense {
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

impl Layer for Dense {
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

    use std::sync::Arc;

    #[test]
    fn test_smoke() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let learning_rate = 0.01;
        let input_size = 1;
        let hidden_size = 16;
        let output_size = 1;
        let initializer = Arc::new(|x: Float| {
            let range = 1.0 / x.sqrt();
            rand::thread_rng().gen_range(-range..=range)
        });
        let sigmoid = Arc::new(|x: Array| x.sigmoid());
        let mut l1 = Dense::new(
            input_size,
            hidden_size,
            initializer.clone(),
            Some(sigmoid.clone()),
        );
        let mut l2 = Dense::new(hidden_size, output_size, initializer.clone(), None);

        for _ in 0..8 {
            let x = rng.gen_range(-1.0..1.0);
            let input = arr![arr![x]];
            let target = x.exp();

            let r1 = l1.forward(input);
            let r2 = l2.forward(r1);

            let mut error = (&arr![target] - &r2).powf(2.0);

            error.backward(None);

            let mut parameters = l1.parameters();
            parameters.append(&mut l2.parameters());

            for parameter in parameters {
                let gradient = parameter.gradient().unwrap();
                parameter.stop_tracking();
                *parameter = &*parameter - &(&gradient * learning_rate);
                parameter.start_tracking();
                *parameter.gradient_mut() = None;
            }
        }
    }
}
