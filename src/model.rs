//! A supervised neural network model, which computes a forward pass, and updates parameters based on a target.

use crate::array::*;
use crate::cost::CostFunction;
use crate::layer::Layer;
use crate::numbers::*;
use crate::optimizer::Optimizer;

/// A neural network model, containing the layers of the model, and the outputs.
pub struct Model<'a> {
    layers: Vec<&'a mut dyn Layer>,
    output: Option<Array>,
    optimizer: &'a dyn Optimizer,
    cost: &'a CostFunction,
}

impl<'a> Model<'a> {
    /// Constructs a new model given the layers.
    pub fn new(
        layers: Vec<&'a mut dyn Layer>,
        optimizer: &'a dyn Optimizer,
        cost: &'a CostFunction,
    ) -> Model<'a> {
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
        let error = (self.cost)(&output, &target);
        error.backward(None);

        error.sum_all()
    }

    /// Updates all parameters of the model.
    pub fn update(&mut self) {
        let optimizer = self.optimizer;
        let parameters = self.parameters();
        optimizer.update(parameters);
    }

    /// Retrieves the parameters of every layer in the model.
    fn parameters(&mut self) -> Vec<&mut Array> {
        self.layers
            .iter_mut()
            .map(|l| l.parameters())
            .flatten()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::conv::Conv;
    use crate::layer::dense::Dense;
    use crate::optimizer::gd::GradientDescent;
    use crate::{activation, cost, initializer};

    use rand::Rng;

    fn test_gradient(mut model: Model<'_>, cost: &CostFunction, input: Array, target: Array) {
        #[cfg(feature = "f32")]
        let epsilon = 0.1;
        #[cfg(not(feature = "f32"))]
        let epsilon = 1e-7;

        model.forward(input.clone());
        model.backward(target.clone());

        // due to borrow checking, we need to keep re-borrowing, and dropping the parameters
        let parameters = model.parameters();
        let length = parameters.len();
        std::mem::drop(parameters);
        for i in 0..length {
            let parameters = model.parameters();
            let value_length = parameters[i].values().len();
            let dimensions = parameters[i].dimensions().to_vec();
            let gradient = parameters[i].gradient().to_owned().unwrap().clone();
            std::mem::drop(parameters);

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for j in 0..value_length {
                let mut delta = vec![0.0; value_length];
                delta[j] = epsilon;
                let delta = Array::from((dimensions.clone(), delta));

                let mut parameters = model.parameters();
                let parameter = &mut parameters[i];
                parameter.stop_tracking();
                **parameter = &**parameter + &delta;
                parameter.start_tracking();
                std::mem::drop(parameters);

                let result_plus = model.forward(input.clone());
                let error_plus = (cost)(&result_plus, &target.clone()).sum_all();

                let mut delta = vec![0.0; value_length];
                delta[j] = -2.0 * epsilon;
                let delta = Array::from((dimensions.clone(), delta));

                let mut parameters = model.parameters();
                let parameter = &mut parameters[i];
                parameter.stop_tracking();
                **parameter = &**parameter + &delta;
                parameter.start_tracking();
                std::mem::drop(parameters);

                let result_minus = model.forward(input.clone());
                let error_minus = (cost)(&result_minus, &target).sum_all();

                let mut delta = vec![0.0; value_length];
                delta[j] = epsilon;
                let delta = Array::from((dimensions.clone(), delta));

                let mut parameters = model.parameters();
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
    fn test_dense_gradient() {
        let learning_rate = 0.0;
        let input_size = 2;
        let hidden_size = 16;
        let output_size = 2;
        let initializer = initializer::he();
        let sigmoid = activation::sigmoid();
        let softmax = activation::softmax();
        let cross_entropy = cost::cross_entropy();
        let gd = GradientDescent::new(learning_rate);
        let mut l1 = Dense::new(input_size, hidden_size, &initializer, Some(&sigmoid));
        let mut l2 = Dense::new(hidden_size, output_size, &initializer, Some(&softmax));
        let model = Model::new(vec![&mut l1, &mut l2], &gd, &cross_entropy);

        let (x, y, z, w) = (0.5, -0.25, 0.0, 1.0);
        test_gradient(model, &cross_entropy, arr![x, y], arr![z, w]);
    }

    #[test]
    fn test_conv_gradient() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let learning_rate = 0.0;

        let (image_depth, image_rows, image_cols) = (3, 9, 9);
        let image_dimensions = vec![image_depth, image_rows, image_cols];
        let output_dimensions = vec![1, 2, 2];
        let input_size = image_dimensions.iter().product();
        let output_size = output_dimensions.iter().product();

        let initializer = initializer::he();
        let activation = activation::relu();
        let mse = cost::mse();
        let gd = GradientDescent::new(learning_rate);

        let mut l1 = Conv::new(
            (16, image_depth, 3, 3),
            (2, 2),
            &initializer,
            Some(activation),
        );
        let mut l2 = Conv::new((1, 16, 2, 2), (2, 2), &initializer, None);
        let model = Model::new(vec![&mut l1, &mut l2], &gd, &mse);

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

        test_gradient(model, &mse, input, target);
    }

    #[test]
    fn test_model() {
        let mut rng = rand::thread_rng();
        let learning_rate = 0.1;
        let batch_size = 32;
        let input_size = 2;
        let hidden_size = 16;
        let output_size = 2;
        let initializer = initializer::he();
        let relu = activation::relu();
        let mse = cost::mse();
        let gd = GradientDescent::new(learning_rate);
        let mut l1 = Dense::new(input_size, hidden_size, &initializer, Some(&relu));
        let mut l2 = Dense::new(hidden_size, output_size, &initializer, None);
        let mut model = Model::new(vec![&mut l1, &mut l2], &gd, &mse);

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

            let input = Array::from((vec![batch_size, input_size], input));
            let target = Array::from((vec![batch_size, output_size], target));

            let _result = model.forward(input.clone());
            let loss = model.backward(target.clone());
            model.update();

            println!("loss: {}", loss);
        }
    }
}
