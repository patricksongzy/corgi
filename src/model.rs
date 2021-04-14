//! A supervised neural network model, which computes a forward pass, and updates parameters based on a target.

use crate::array::*;
use crate::layer::Layer;
use crate::numbers::*;
use crate::optimizer::Optimizer;

/// A neural network model, containing the layers of the model, and the outputs.
pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    output: Option<Array>,
    optimizer: Box<dyn Optimizer>,
}

impl Model {
    /// Constructs a new model given the layers.
    pub fn new(layers: Vec<Box<dyn Layer>>, optimizer: Box<dyn Optimizer>) -> Model {
        Model {
            layers,
            output: None,
            optimizer,
        }
    }

    /// Computes the forward pass of a model.
    pub fn forward(&mut self, mut x: Array) -> Array {
        for layer in &self.layers {
            x = layer.forward(x)
        }

        self.output = Some(x.clone());
        x
    }

    /// Computes the backward pass of a model, and updates parameters.
    pub fn backward(&mut self, target: Array) -> Float {
        let output = self.output.as_ref().unwrap();
        let dimensions = output.dimensions();
        let batch_size = if dimensions.len() > 1 {
            dimensions[dimensions.len() - 2]
        } else {
            1
        };

        let mut error = (1.0 / batch_size as Float) * &(&target - output).powf(2.0);
        error.backward(None);

        self.update();

        error.sum()
    }

    fn update(&mut self) {
        let parameters = self
            .layers
            .iter_mut()
            .map(|l| l.parameters())
            .flatten()
            .collect();
        self.optimizer.update(parameters);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dense::Dense;
    use crate::optimizers::gd::GradientDescent;

    use std::sync::Arc;

    #[test]
    fn test_model() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let learning_rate = 0.01;
        let batch_size = 32;
        let input_size = 2;
        let hidden_size = 16;
        let output_size = 2;
        let initializer = Arc::new(|x: Float| {
            let range = 1.0 / x.sqrt();
            rand::thread_rng().gen_range(-range..=range)
        });
        let sigmoid = Arc::new(|x: Array| x.sigmoid());
        let gd = GradientDescent::new(learning_rate);
        let l1 = Dense::new(input_size, hidden_size, initializer.clone(), Some(sigmoid));
        let l2 = Dense::new(hidden_size, output_size, initializer.clone(), None);
        let mut model = Model::new(vec![Box::new(l1), Box::new(l2)], Box::new(gd));

        for _ in 0..8 {
            let mut input = vec![0.0; input_size * batch_size];
            let mut target = vec![0.0; output_size * batch_size];
            for j in 0..batch_size {
                let x: Float = rng.gen_range(-1.0..1.0);
                let y: Float = rng.gen_range(-1.0..1.0);
                input[input_size * j] = x;
                input[input_size * j + 1] = y;
                target[output_size * j] = x.exp();
                target[output_size * j + 1] = x.exp() + y.sin();
            }

            let input = Arrays::new((vec![batch_size, input_size], input));
            let target = Arrays::new((vec![batch_size, output_size], target));

            let _result = model.forward(input.clone());
            let loss = model.backward(target.clone());

            println!("loss: {}", loss);
        }
    }
}
