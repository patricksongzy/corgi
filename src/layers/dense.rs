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
        // TODO this should not be in `dense.rs`
        // TODO He Initialisation
        Dense {
            weights: Arrays::new((
                vec![output_size, input_size],
                (0..input_size * output_size)
                    .map(|_| (*initializer)(input_size as Float))
                    .collect::<Vec<Float>>(),
            )),
            biases: Arrays::new(vec![0.0; output_size]),
            activation,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, x: Array) -> Array {
        let y = &Array::matmul((&self.weights, false), (&x, false)) + &self.biases;
        match &self.activation {
            Some(f) => f(y),
            None => y,
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
    fn test_backward() {
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

            let mut gw1 = l1.weights.gradient();
            let mut gb1 = l1.biases.gradient();
            let mut gw2 = l2.weights.gradient();
            let mut gb2 = l2.biases.gradient();

            l1.weights = l1.weights.untracked() - (gw1.untracked() * learning_rate).untracked();
            l1.biases = l1.biases.untracked() - (gb1.untracked() * learning_rate).untracked();
            l2.weights = l2.weights.untracked() - (gw2.untracked() * learning_rate).untracked();
            l2.biases = l2.biases.untracked() - (gb2.untracked() * learning_rate).untracked();

            *l1.weights.gradient_mut() = None;
            *l1.biases.gradient_mut() = None;
            *l2.weights.gradient_mut() = None;
            *l2.biases.gradient_mut() = None;
        }
    }
}
