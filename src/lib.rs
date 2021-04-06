#![warn(missing_docs)]

//! Machine learning, and automatic differentiation implementation.

pub mod numbers;
#[macro_use]
pub mod array;

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use array::*;
    use numbers::*;

    #[test]
    fn test_op() {
        let op: array::ForwardOp = Arc::new(|x: &[&Array]| {
            Arrays::new((
                x[0].dimensions(),
                x[0].values()
                    .iter()
                    .zip(x[1].values())
                    .map(|(x, y)| x * y)
                    .collect::<Vec<Float>>(),
            ))
        });

        let op_clone = Arc::clone(&op);
        let backward_op: array::BackwardOp = Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
            vec![
                Array::op(
                    &vec![c[1].untracked(), x.untracked()],
                    &Vec::new(),
                    Arc::clone(&op_clone),
                    None,
                ),
                Array::op(
                    &vec![c[0].untracked(), x.untracked()],
                    &Vec::new(),
                    Arc::clone(&op_clone),
                    None,
                ),
            ]
        });

        let a = arr![1.0, 2.0, 3.0];
        let b = arr![3.0, 2.0, 1.0];
        let mut product = Array::op(
            &vec![&a, &b],
            &vec![a.clone(), b.clone()],
            op,
            Some(backward_op),
        );
        assert_eq!(product, arr![3.0, 4.0, 3.0]);
        product.backward(None);
        assert_eq!(product.gradient(), arr![1.0, 1.0, 1.0]);
        assert_eq!(b.gradient(), arr![1.0, 2.0, 3.0]);
        assert_eq!(a.gradient(), arr![3.0, 2.0, 1.0]);
    }
}
