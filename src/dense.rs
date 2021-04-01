use crate::numbers::*;
use crate::array::*;

use std::sync::Arc;

struct Dense {
    weights: Array,
    biases: Array,
}

impl Dense {
    fn new(input_size: usize, output_size: usize) -> Dense {
        // TODO this should not be in `dense.rs`
        // TODO He Initialisation
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let range = 1.0 / (input_size as Float).sqrt();
        Dense {
            weights: Arrays::new((Arc::new(vec![output_size, input_size]), Arc::new((0..input_size * output_size)
                .map(|_| rng.gen_range(-range..=range)).collect::<Vec<Float>>()))),
            biases: Arrays::new(vec![0.0; output_size])
        }
    }

    fn forward(&self, x: Array) -> Array {
        &Array::matmul(&self.weights, &x, false, true) + &self.biases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let input_size = 128;
        let hidden_size = 256;
        let output_size = 16;
        let l1 = Dense::new(input_size, hidden_size);
        let l2 = Dense::new(hidden_size, output_size);
        let r1 = l1.forward(Arrays::new((0..input_size).map(|_| rng.gen_range(0.0..1.0)).collect::<Vec<Float>>()));
        let mut r2 = l2.forward(r1);
        r2.backward(None);
        println!("{:?}", l1.weights.gradient());
        println!("{:?}", l1.biases.gradient());
    }
}
