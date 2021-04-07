#[macro_use]
extern crate corgi;

use corgi::array::*;
use corgi::layer::Layer;
use corgi::model::Model;
use corgi::numbers::*;

pub struct Dense {
    weights: Array,
    biases: Array,
    lr: Float,
    activation: bool,
}

impl Dense {
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

fn main() {
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
        let x: Float = rng.gen_range(-1.0..1.0);
        let target = x.exp();
        let input = arr![arr![x]];
        let result = model.forward(input);
        let loss = model.backward(arr![target]);
        println!(
            "in: {}, out: {}, target: {}, loss: {}",
            x, result[0], target, loss
        );
    }
}
