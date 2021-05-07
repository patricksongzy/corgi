#![deny(missing_docs)]

//! Machine learning, and dynamic automatic differentiation implementation.

#[cfg(feature = "blas")]
extern crate libc;

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod numbers;
#[macro_use]
pub mod array;
#[cfg(feature = "blas")]
pub mod blas;
pub mod layer;
pub mod layers;
pub mod model;
pub mod nn;
pub mod optimizer;
pub mod optimizers;

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use array::*;
    use numbers::*;

    #[test]
    fn test_op() {
        let mul: array::ForwardOp = Arc::new(|x: &[&Array]| {
            Array::from((
                x[0].dimensions(),
                x[0].values()
                    .iter()
                    .zip(x[1].values())
                    .map(|(x, y)| x * y)
                    .collect::<Vec<Float>>(),
            ))
        });

        let mul_clone = Arc::clone(&mul);
        let backward_op: array::BackwardOp = Arc::new(move |children: &mut Vec<Array>, is_tracked: &[bool], delta: &Array| {
            vec![
                if is_tracked[0] {
                    Some(Array::op(&vec![&children[1], delta], Arc::clone(&mul_clone), None))
                } else {
                    None
                },
                if is_tracked[1] {
                    Some(Array::op(&vec![&children[0], delta], Arc::clone(&mul_clone), None))
                } else {
                    None
                }
            ]
        });

        let a = arr![1.0, 2.0, 3.0].tracked();
        let b = arr![3.0, 2.0, 1.0].tracked();
        let mut product = Array::op(&vec![&a, &b], mul, Some(backward_op));
        assert_eq!(product, arr![3.0, 4.0, 3.0]);
        product.backward(None);
        assert_eq!(product.gradient().unwrap(), arr![1.0, 1.0, 1.0]);
        assert_eq!(b.gradient().unwrap(), arr![1.0, 2.0, 3.0]);
        assert_eq!(a.gradient().unwrap(), arr![3.0, 2.0, 1.0]);
    }
}
