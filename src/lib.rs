#![deny(missing_docs)]

//! Machine learning, and dynamic automatic differentiation implementation.

extern crate mimalloc;

#[cfg(feature = "blas")]
extern crate cblas_sys;
#[cfg(feature = "netlib")]
extern crate netlib_src;
#[cfg(feature = "openblas")]
extern crate openblas_src;

#[cfg(test)]
#[macro_use]
extern crate approx;

use mimalloc::MiMalloc;

pub mod numbers;
#[macro_use]
pub mod array;
pub mod activation;
#[cfg(feature = "blas")]
pub mod blas;
pub mod cost;
pub mod initializer;
pub mod layer;
pub mod model;
pub mod optimizer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(test)]
mod tests {
    use super::*;

    use std::rc::Rc;

    use array::*;
    use numbers::*;

    #[test]
    fn test_op() {
        let mul: array::ForwardOp = Rc::new(|x: &[&Array]| {
            Array::from((
                x[0].dimensions().to_vec(),
                x[0].values()
                    .iter()
                    .zip(x[1].values())
                    .map(|(x, y)| x * y)
                    .collect::<Vec<Float>>(),
            ))
        });

        let mul_clone = Rc::clone(&mul);
        let backward_op: array::BackwardOp = Rc::new(move |children, is_tracked, delta| {
            vec![
                if is_tracked[0] {
                    Some(Array::op(
                        &vec![&children[1], delta],
                        Rc::clone(&mul_clone),
                        None,
                    ))
                } else {
                    None
                },
                if is_tracked[1] {
                    Some(Array::op(
                        &vec![&children[0], delta],
                        Rc::clone(&mul_clone),
                        None,
                    ))
                } else {
                    None
                },
            ]
        });

        let a = arr![1.0, 2.0, 3.0].tracked();
        let b = arr![3.0, 2.0, 1.0].tracked();
        let product = Array::op(&vec![&a, &b], mul, Some(backward_op));
        assert_eq!(product, arr![3.0, 4.0, 3.0]);
        product.backward(None);
        assert_eq!(product.gradient().to_owned().unwrap(), arr![1.0, 1.0, 1.0]);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![1.0, 2.0, 3.0]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![3.0, 2.0, 1.0]);
    }
}
