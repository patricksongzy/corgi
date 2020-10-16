use crate::numbers::*;
use crate::array::*;

use std::ops::Index;

struct Tensor {
    matrix: Array,
}

impl Tensor {
    #[inline]
    pub fn new(matrix: Array) -> Self {
        Self { matrix }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tensor = Tensor::new(
            array![array![
                array![0.0], array![1.0]
            ],
            array![
                array![2.0], array![3.0]
            ],
            array![
                array![4.0], array![5.0]
            ]]
        );

        assert_eq!(tensor.matrix.dimensions, vec![3, 2, 1]);
        assert_eq!(tensor.matrix.values, (0..=5).map(|x| x as Float).collect::<Vec<Float>>());
    }
}

