//! A supervised neural network model, which computes a forward pass, and updates parameters based on a target.

use crate::array::*;
use crate::layer::Layer;
use crate::nn_functions::cost::CostFunction;
use crate::numbers::*;
use crate::optimizer::Optimizer;

/// A neural network model, containing the layers of the model, and the outputs.
pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    output: Option<Array>,
    optimizer: Box<dyn Optimizer>,
    cost: CostFunction,
}

impl Model {
    /// Constructs a new model given the layers.
    pub fn new(
        layers: Vec<Box<dyn Layer>>,
        optimizer: Box<dyn Optimizer>,
        cost: CostFunction,
    ) -> Model {
        Model {
            layers,
            output: None,
            optimizer,
            cost,
        }
    }

    /// Computes the forward pass of a model.
    /// The input should have the dimensions batch size by input size.
    pub fn forward(&mut self, mut input: Array) -> Array {
        for layer in &self.layers {
            input = layer.forward(input)
        }

        self.output = Some(input.clone());
        input
    }

    /// Computes the backward pass of a model, and updates parameters.
    pub fn backward(&mut self, target: Array) -> Float {
        let output = self.output.as_ref().unwrap();
        let mut error = (self.cost)(&output, &target);
        error.backward(None);

        self.update();

        error.sum_all()
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
    use crate::nn_functions::{activation, cost, initializer};
    use crate::optimizers::gd::GradientDescent;

    use rand::Rng;

    #[test]
    fn test_model() {
        let mut rng = rand::thread_rng();
        let learning_rate = 0.01;
        let batch_size = 32;
        let input_size = 2;
        let hidden_size = 16;
        let output_size = 2;
        let initializer = initializer::make_he();
        let sigmoid = activation::make_sigmoid();
        let mse = cost::make_mse();
        let gd = GradientDescent::new(learning_rate);
        let l1 = Dense::new(input_size, hidden_size, initializer.clone(), Some(sigmoid));
        let l2 = Dense::new(hidden_size, output_size, initializer.clone(), None);
        let mut model = Model::new(vec![Box::new(l1), Box::new(l2)], Box::new(gd), mse);

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
