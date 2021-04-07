//! A supervised neural network model, which computes a forward pass, and updates parameters based on a target.

use crate::numbers::*;
use crate::array::*;
use crate::layer::Layer;

/// A neural network model, containing the layers of the model, and the outputs.
pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    output: Option<Array>,
}

impl Model {
    /// Constructs a new model given the layers.
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Model {
        Model {
            layers,
            output: None,
        }
    }

    /// Completes the forward pass of a model.
    pub fn forward(&mut self, mut x: Array) -> Array {
        for layer in &self.layers {
            x = layer.forward(x)
        }

        self.output = Some(x.clone());
        x
    }

    /// Completes the backward pass of a model, and updates parameters.
    pub fn backward(&mut self, target: Array) -> Float {
        let mut error = (&target - self.output.as_ref().unwrap()).powf(2.0);
        error.backward(None);

        for layer in &mut self.layers {
            layer.update();
        }

        error.sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::Dense;

    #[test]
    fn test_model() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let lr = 0.01;
        let input_size = 1;
        let hidden_size = 16;
        let output_size = 1;
        let l1 = Dense::new(input_size, hidden_size, lr, true);
        let l2 = Dense::new(hidden_size, output_size, lr, false);
        let mut model = Model::new(vec![Box::new(l1), Box::new(l2)]);

        for _ in 0..1024 {
            let x = rng.gen_range(-1.0..1.0);
            let input = arr![arr![x]];
            let target = x.exp();

            let result = model.forward(input);
            let loss = model.backward(arr![target]);

            println!(
                "in: {}, out: {}, target: {}, loss: {}",
                x, result[0], target, loss
            );
        }
    }
}
