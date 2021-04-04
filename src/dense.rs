use crate::numbers::*;
use crate::array::*;

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
            weights: Arrays::new((vec![output_size, input_size], (0..input_size * output_size)
                .map(|_| rng.gen_range(-range..=range)).collect::<Vec<Float>>())),
            biases: Arrays::new(vec![0.0; output_size])
        }
    }

    fn forward(&self, x: Array) -> Array {
        &Array::matmul(&self.weights, &x, false, false) + &self.biases
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
        let hidden_size = 8;
        let output_size = 1;
        let mut l1 = Dense::new(input_size, hidden_size);
        let mut l2 = Dense::new(hidden_size, output_size);
        for _ in 0..1000 {
            // let input = Arrays::new((vec![input_size, 1], (0..input_size).map(|_| rng.gen_range(0.0..1.0))
            //     .collect::<Vec<Float>>()));
            let x = rng.gen_range(0.0..1.0);
            let input = arr![arr![x]];
            let r1 = l1.forward(input);
            let mut r2 = l2.forward(r1);

            let target = 0.5 * x + 0.5;
            let loss = (target - r2[0]).powf(2.0);
            let delta = 2.0 * (r2[0] - target);
            r2.backward(Some(arr![arr![delta]]));
            println!("in: {}, out: {}, target: {}, loss: {}", x, r2[0], target, loss);

            let mut gw2 = l2.weights.gradient();
            let mut gb2 = l2.biases.gradient();
            let mut gw1 = l1.weights.gradient();
            let mut gb1 = l1.biases.gradient();

            l2.weights = l2.weights.untracked() + (gw2.untracked() * -lr).untracked();
            l2.biases = l2.biases.untracked() + (gb2.untracked() * -lr).untracked();
            l1.weights = l1.weights.untracked() + (gw1.untracked() * -lr).untracked();
            l1.biases = l1.biases.untracked() + (gb1.untracked() * -lr).untracked();

            *l2.weights.gradient_mut() = None;
            *l2.biases.gradient_mut() = None;
            *l1.weights.gradient_mut() = None;
            *l1.biases.gradient_mut() = None;
        }
    }
}
