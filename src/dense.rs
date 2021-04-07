//! A fully-connected neural network layer, which applies y = activation(Ax + b).

use crate::array::*;
use crate::layer::Layer;
use crate::numbers::*;

/// A fully-connected neural network layer, storing the parameters of the layer.
pub struct Dense {
    weights: Array,
    biases: Array,
    lr: Float,
    activation: bool,
}

impl Dense {
    /// Constructs a new dense layer, with a given input, and output size.
    pub fn new(input_size: usize, output_size: usize, lr: Float, activation: bool) -> Dense {
        // TODO this should not be in `dense.rs`
        // TODO He Initialisation
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let range = 1.0 / (input_size as Float).sqrt();
        Dense {
            weights: Arrays::new((
                vec![output_size, input_size],
                (0..input_size * output_size)
                    .map(|_| rng.gen_range(-range..=range))
                    .collect::<Vec<Float>>(),
            )),
            biases: Arrays::new(vec![0.0; output_size]),
            lr,
            activation,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, x: Array) -> Array {
        let y = &Array::matmul((&self.weights, false), (&x, false)) + &self.biases;
        if self.activation {
            y.sigmoid()
        } else {
            y
        }
    }

    fn update(&mut self) {
        let mut weights_gradient = self.weights.gradient();
        let mut biases_gradient = self.biases.gradient();
        // update the parameters, using an untracked update since we are not interested in the
        // derivative of our update
        self.weights =
            self.weights.untracked() + (weights_gradient.untracked() * -self.lr).untracked();
        self.biases =
            self.biases.untracked() + (biases_gradient.untracked() * -self.lr).untracked();
        // clear the gradients for the next update
        *self.weights.gradient_mut() = None;
        *self.biases.gradient_mut() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let lr = 0.01;
        let input_size = 1;
        let hidden_size = 16;
        let output_size = 1;

        let mut l1 = Dense::new(input_size, hidden_size, lr, true);
        let mut l2 = Dense::new(hidden_size, output_size, lr, false);

        for _ in 0..8 {
            let x = rng.gen_range(-1.0..1.0);
            let input = arr![arr![x]];
            let target = x.exp();

            let r1 = l1.forward(input);
            let r2 = l2.forward(r1);

            let mut error = (&arr![target] - &r2).powf(2.0);

            error.backward(None);

            l1.update();
            l2.update();
        }
    }
}
