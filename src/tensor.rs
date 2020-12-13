use crate::numbers::*;
use crate::array::*;

use std::ops;

struct Tensor {
    matrix: Array,
}

impl Tensor {
    #[inline]
    pub fn new(matrix: Array) -> Self {
        Self { matrix }
    }
}

impl<'a, 'b> ops::Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    #[inline]
    fn add(self, other: &Tensor) -> Tensor {
        Tensor::new(Arrays::new(self.matrix.values.iter().zip(&other.matrix.values).map(|(x, y)| x + y).collect::<Vec<Float>>()))
    }
}

impl<'a, 'b> ops::Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    #[inline]
    fn mul(self, other: &Tensor) -> Tensor {
        Tensor::new(Arrays::new(self.matrix.values.iter().zip(&other.matrix.values).map(|(x, y)| x * y).collect::<Vec<Float>>()))    
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

    #[test]
    fn test_arithmetic() {
        let a = Tensor::new(
            array![array![
                array![0.0, 1.0], array![2.0, 3.0]
            ],
            array![
                array![4.0, 5.0], array![6.0, 7.0]
            ]]
        );

        let b = Tensor::new(
            array![array![
                array![2.0, 4.0], array![6.0, 8.0]
            ],
            array![
                array![10.0, 12.0], array![14.0, 16.0]
            ]]
        );

        let sum = Tensor::new(
            array![array![
                array![2.0, 5.0], array![8.0, 11.0]
            ],
            array![
                array![14.0, 17.0], array![20.0, 23.0]
            ]]
        );

        let product = Tensor::new(
            array![array![
                array![0.0, 4.0], array![12.0, 24.0]
            ],
            array![
                array![40.0, 60.0], array![84.0, 112.0]
            ]]
        );

        assert_eq!((&a + &b).matrix.values, sum.matrix.values);
        assert_eq!((&a * &b).matrix.values, product.matrix.values);
    }
}

