use crate::array::*;
use crate::layer::Layer;
use crate::numbers::*;

struct Dense {
    weights: Array,
    biases: Array,
    activation: bool,
}

impl Dense {
    fn new(input_size: usize, output_size: usize, activation: bool) -> Dense {
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

    fn update(&self, target: Array) {}
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

        let mut l1 = Dense::new(input_size, hidden_size, true);
        let mut l2 = Dense::new(hidden_size, output_size, false);

        for _ in 0..1024 {
            let x = rng.gen_range(-1.0..1.0);
            let input = arr![arr![x]];
            let target = x.exp();

            let r1 = l1.forward(input);
            let r2 = l2.forward(r1);

            let mut error = (&arr![target] - &r2).powf(2.0);
            // Mean Square Error
            let loss = error.sum();

            println!(
                "in: {}, out: {}, target: {}, loss: {}",
                x, r2[0], target, loss
            );

            error.backward(None);

            let mut gw2 = l2.weights.gradient();
            let mut gb2 = l2.biases.gradient();
            let mut gw1 = l1.weights.gradient();
            let mut gb1 = l1.biases.gradient();

            // update the parameters, using an untracked update since we are not interested in the
            // derivative of our update
            l2.weights = l2.weights.untracked() + (gw2.untracked() * -lr).untracked();
            l2.biases = l2.biases.untracked() + (gb2.untracked() * -lr).untracked();
            l1.weights = l1.weights.untracked() + (gw1.untracked() * -lr).untracked();
            l1.biases = l1.biases.untracked() + (gb1.untracked() * -lr).untracked();

            // clear the gradients for the next update
            *l2.weights.gradient_mut() = None;
            *l2.biases.gradient_mut() = None;
            *l1.weights.gradient_mut() = None;
            *l1.biases.gradient_mut() = None;
        }
    }
}
