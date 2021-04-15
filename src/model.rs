//! A supervised neural network model, which computes a forward pass, and updates parameters based on a target.

use crate::array::*;
use crate::layer::Layer;
use crate::nn::cost::CostFunction;
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

        error.sum_all()
    }

    /// Updates all parameters of the model.
    pub fn update(&mut self) {
        let parameters = Model::parameters(&mut self.layers);
        self.optimizer.update(parameters);
    }

    /// Retrieves the parameters of every layer in the model.
    fn parameters(layers: &mut Vec<Box<dyn Layer>>) -> Vec<&mut Array> {
        layers.iter_mut().map(|l| l.parameters()).flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dense::Dense;
    use crate::nn::{activation, cost, initializer};
    use crate::optimizers::gd::GradientDescent;

    use rand::Rng;

    use std::sync::Arc;

    #[test]
    fn test_gradient() {
        #[cfg(feature = "f32")]
        let epsilon = 1e-2;
        #[cfg(not(feature = "f32"))]
        let epsilon = 1e-7;
        let learning_rate = 1.0;
        let input_size = 1;
        let hidden_size = 16;
        let output_size = 1;
        let initializer = initializer::make_he();
        let sigmoid = activation::make_sigmoid();
        let mse = cost::make_mse();
        let gd = GradientDescent::new(learning_rate);
        let l1 = Dense::new(input_size, hidden_size, initializer.clone(), Some(sigmoid));
        let l2 = Dense::new(hidden_size, output_size, initializer.clone(), None);
        let mut model = Model::new(vec![Box::new(l1), Box::new(l2)], Box::new(gd), Arc::clone(&mse));

        let (x, y) = (5.0, 6.0);
        model.forward(arr![x]);
        model.backward(arr![y]);

        // due to borrow checking, we need to keep re-borrowing, and dropping the parameters
        let parameters = Model::parameters(&mut model.layers);
        let length = parameters.len();
        std::mem::drop(parameters);
        for i in 0..length {
            let parameters = Model::parameters(&mut model.layers);
            let value_length = parameters[i].values().len();
            let dimensions = Arc::new(parameters[i].dimensions().clone());
            let gradient = parameters[i].gradient().unwrap().clone();
            std::mem::drop(parameters);

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for j in 0..value_length {
                let mut delta = vec![0.0; value_length];
                delta[j] = epsilon;
                let delta = Arrays::new((Arc::clone(&dimensions), Arc::new(delta)));

                let mut parameters = Model::parameters(&mut model.layers);
                let parameter = &mut parameters[i];
                parameter.stop_tracking();
                **parameter = &**parameter + &delta;
                parameter.start_tracking();
                std::mem::drop(parameters);

                let result_plus = model.forward(arr![x]);
                let error_plus = (mse)(&result_plus, &arr![y]).sum_all();

                let mut delta = vec![0.0; value_length];
                delta[j] = -2.0 * epsilon;
                let delta = Arrays::new((Arc::clone(&dimensions), Arc::new(delta)));

                let mut parameters = Model::parameters(&mut model.layers);
                let parameter = &mut parameters[i];
                parameter.stop_tracking();
                **parameter = &**parameter + &delta;
                parameter.start_tracking();
                std::mem::drop(parameters);

                let result_minus = model.forward(arr![x]);
                let error_minus = (mse)(&result_minus, &arr![y]).sum_all();

                let mut delta = vec![0.0; value_length];
                delta[j] = epsilon;
                let delta = Arrays::new((Arc::clone(&dimensions), Arc::new(delta)));

                let mut parameters = Model::parameters(&mut model.layers);
                let parameter = &mut parameters[i];
                parameter.stop_tracking();
                **parameter = &**parameter + &delta;
                parameter.start_tracking();
                std::mem::drop(parameters);

                let numerical_gradient = (error_plus - error_minus) / (2.0 * epsilon);
                numerator += ((gradient[j] - numerical_gradient).abs()).powf(2.0);
                denominator += ((gradient[j] + numerical_gradient).abs()).powf(2.0);
            }

            numerator = numerator.sqrt();
            denominator = denominator.sqrt();

            let norm = numerator / denominator;

            println!("{}", norm);
            assert!(norm < epsilon);
        }
    }

    #[test]
    fn test_model() {
        let mut rng = rand::thread_rng();
        let learning_rate = 1.0;
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
            model.update();

            println!("loss: {}", loss);
        }
    }
}
