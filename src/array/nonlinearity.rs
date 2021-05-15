use crate::array::*;
use crate::numbers::*;

impl Array {
    /// Computes the ReLU of the array, defined as max(0, x) for all elements x in the array.
    pub fn relu(&self) -> Array {
        let (values, derivative) = self
            .values
            .iter()
            .map(|&x| if x > 0.0 { (x, 1.0) } else { (0.0, 0.0) })
            .unzip();

        let result = Array::from((Arc::clone(&self.dimensions), Arc::new(values)));
        let derivative = Array::from((Arc::clone(&self.dimensions), Arc::new(derivative)));

        if !self.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(move |_, _, x| vec![Some(&derivative * x)]);

            result
                .with_children(vec![self.clone()])
                .with_backward_op(backward_op)
        }
    }

    /// Computes the sigmoid operation on each value of the array.
    pub fn sigmoid(&self) -> Array {
        let values = Arc::new(
            self.values
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect::<Vec<Float>>(),
        );

        let cached = Arc::clone(&values);
        let result = Array::from((Arc::clone(&self.dimensions), values));

        if !self.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(move |c, _, x| {
                let values = arithmetic::mul_values(
                    &cached.iter().map(|v| v * (1.0 - v)).collect::<Vec<Float>>(),
                    &x.values,
                );
                vec![Some(Array::from((
                    Arc::clone(&c[0].dimensions),
                    Arc::new(values),
                )))]
            });

            result
                .with_children(vec![self.clone()])
                .with_backward_op(backward_op)
        }
    }

    /// Computes the softmax of the array.
    pub fn softmax(&self) -> Array {
        let exponentials = self.exp();
        &exponentials / &exponentials.sum(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arr;

    #[test]
    fn test_softmax() {
        let a = arr![
            arr![(2.0 as Float).ln(), (2.0 as Float).ln()],
            arr![(1.0 as Float).ln(), (1.0 as Float).ln()]
        ]
        .tracked();
        let b = arr![arr![3.0, 5.0], arr![2.0, 5.0]].tracked();
        let c = a.softmax().tracked();

        let mut result = &c * &b;
        assert_eq!(result, arr![arr![1.5, 2.5], arr![1.0, 2.5]]);

        result.backward(None);
        assert_eq!(c.gradient().unwrap(), arr![arr![3.0, 5.0], arr![2.0, 5.0]]);
        assert_eq!(b.gradient().unwrap(), arr![arr![0.5, 0.5], arr![0.5, 0.5]]);
        assert_eq!(
            a.gradient().unwrap(),
            arr![arr![-0.5, 0.5], arr![-0.75, 0.75]]
        );
    }

    #[test]
    fn test_relu() {
        let a = arr![1.0, -2.0, 0.0].tracked();

        let mut result = a.relu();
        assert_eq!(result, arr![1.0, 0.0, 0.0]);

        result.backward(None);
        assert_eq!(a.gradient().unwrap(), arr![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sigmoid() {
        let a = arr![arr![(3.0 as Float).ln()]].tracked();
        let b = arr![arr![5.0]].tracked();
        let c = a.sigmoid().tracked();

        let mut result = &c * &b;
        assert_relative_eq!(result, arr![arr![3.75]]);

        result.backward(None);
        assert_relative_eq!(c.gradient().unwrap(), arr![arr![5.0]]);
        assert_relative_eq!(b.gradient().unwrap(), arr![arr![0.75]]);
        assert_relative_eq!(a.gradient().unwrap(), arr![arr![0.9375]]);
    }
}
