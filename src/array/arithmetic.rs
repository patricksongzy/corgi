use crate::array::*;
use crate::numbers::*;

pub(crate) fn scale_values(a: &[Float], s: Float) -> Vec<Float> {
    a.iter().map(|x| x * s).collect::<Vec<Float>>()
}

pub(crate) fn mul_values(a: &[Float], b: &[Float]) -> Vec<Float> {
    a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<Float>>()
}

impl Array {
    /// Computes the reciprocal of each value in the array.
    pub fn reciprocal(&self) -> Array {
        let values = Arc::new(self.values.iter().map(|x| 1.0 / x).collect());
        let result = Array::from((Arc::clone(&self.dimensions), values));

        if !self.is_tracked {
            result
        } else {
            let backward_op: BackwardOp =
                Arc::new(|c, _, x| vec![Some(&(-&c[0].reciprocal().powf(2.0)) * x)]);

            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }

    /// Raises the array to the specified exponent.
    pub fn powf(&self, exponent: Float) -> Array {
        let values = self
            .values
            .iter()
            .map(|x| x.powf(exponent))
            .collect::<Vec<Float>>();

        let result = Array::from((Arc::clone(&self.dimensions), Arc::new(values)));

        if !self.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(move |c, _, x| vec![Some(&(&c[0] * 2.0) * x)]);

            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }

    /// Computes the natural logarithm of all values of the array.
    pub fn ln(&self) -> Array {
        let values = Arc::new(self.values.iter().map(|x| x.ln()).collect());
        let result = Array::from((Arc::clone(&self.dimensions), values));

        if !self.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(|c, _, x| vec![Some(x * &c[0].reciprocal())]);

            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }

    /// Computes the exponential of all values of the array.
    pub fn exp(&self) -> Array {
        let values = Arc::new(self.values.iter().map(|x| x.exp()).collect());

        let cached = Arc::clone(&values);
        let result = Array::from((Arc::clone(&self.dimensions), values));

        if !self.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(move |c, _, x| {
                vec![Some(Array::from((
                    Arc::clone(&c[0].dimensions),
                    Arc::new(mul_values(&x.values, &cached)),
                )))]
            });

            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }

    /// Sums along the last `skip_size` dimensions.
    pub fn sum(&self, skip_size: usize) -> Array {
        if skip_size == 0 {
            return self.clone();
        }

        let op: SlicedOp = Box::new(move |output_slice: &mut [Float], arrays: Vec<&[Float]>| {
            output_slice[0] = arrays[0].iter().sum();
        });

        let leading_count = self.dimensions.len().saturating_sub(skip_size);
        let target_dimensions: Vec<usize> = self
            .dimensions
            .iter()
            .copied()
            .take(leading_count)
            .chain(vec![1; skip_size])
            .collect();

        let output_values = Array::sliced_op(
            vec![self],
            &op,
            &self.dimensions,
            &target_dimensions,
            skip_size,
        );

        let output_dimensions = target_dimensions[0..leading_count + 1].to_vec();
        let result = Array::from((output_dimensions, output_values));

        if !self.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(move |c, _, x| {
                let op: SlicedOp = Box::new(move |output_slice, arrays| {
                    for output in output_slice.iter_mut() {
                        *output = arrays[0][0];
                    }
                });

                let output_values = Array::sliced_op(
                    vec![&x],
                    &op,
                    &target_dimensions,
                    &c[0].dimensions,
                    skip_size,
                );

                vec![Some(Array::from((
                    Arc::clone(&c[0].dimensions),
                    Arc::new(output_values),
                )))]
            });

            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }

    /// Sums all the values of the array.
    pub fn sum_all(&self) -> Float {
        self.values.iter().sum()
    }
}

impl<'a, 'b> ops::Add<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn add(self, other: &Array) -> Self::Output {
        let dimensions = element_wise_dimensions(&self.dimensions, &other.dimensions);

        let op: SlicedOp = Box::new(move |output_slice: &mut [Float], arrays: Vec<&[Float]>| {
            for (i, output) in output_slice.iter_mut().enumerate() {
                *output = arrays[0][i] + arrays[1][i];
            }
        });

        let output_values = Array::sliced_op(vec![self, other], &op, &dimensions, &dimensions, 0);
        let result = Array::from((Arc::new(dimensions), Arc::new(output_values)));

        if !self.is_tracked && !other.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(move |_, t, x| {
                vec![
                    if t[0] {
                        Some(Array::from((
                            Arc::clone(&x.dimensions),
                            Arc::clone(&x.values),
                        )))
                    } else {
                        None
                    },
                    if t[1] {
                        Some(Array::from((
                            Arc::clone(&x.dimensions),
                            Arc::clone(&x.values),
                        )))
                    } else {
                        None
                    },
                ]
            });

            result
                .with_children(vec![self.clone(), other.clone()])
                .with_backward_op(Some(backward_op))
        }
    }
}

impl<'a, 'b> ops::Sub<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn sub(self, other: &Array) -> Self::Output {
        self + &(-other)
    }
}

impl<'a> ops::Neg for &'a Array {
    type Output = Array;

    #[inline]
    fn neg(self) -> Self::Output {
        let result = Array::from((
            Arc::clone(&self.dimensions),
            Arc::new(scale_values(&self.values, -1.0)),
        ));

        if !self.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(move |_, _, x| vec![Some(-x)]);
            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }
}

impl ops::Mul<&Array> for Float {
    type Output = Array;

    #[inline]
    fn mul(self, other: &Array) -> Self::Output {
        other * self
    }
}

impl<'a> ops::Mul<Float> for &'a Array {
    type Output = Array;

    #[inline]
    fn mul(self, other: Float) -> Self::Output {
        let result = Array::from((
            Arc::clone(&self.dimensions),
            Arc::new(scale_values(&self.values, other)),
        ));

        if !self.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(move |_, _, x| vec![Some(x * other)]);
            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }
}

impl<'a, 'b> ops::Mul<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn mul(self, other: &Array) -> Self::Output {
        let dimensions = element_wise_dimensions(&self.dimensions, &other.dimensions);
        let op: SlicedOp = Box::new(move |output_slice: &mut [Float], arrays: Vec<&[Float]>| {
            for (i, output) in output_slice.iter_mut().enumerate() {
                *output = arrays[0][i] * arrays[1][i];
            }
        });

        let dimensions = Arc::new(dimensions);
        let output_values = Array::sliced_op(vec![self, other], &op, &dimensions, &dimensions, 0);
        let result = Array::from((Arc::clone(&dimensions), Arc::new(output_values)));

        if !self.is_tracked && !other.is_tracked {
            result
        } else {
            let backward_op: BackwardOp = Arc::new(move |c, t, x| {
                vec![
                    if t[0] { Some(&c[1] * x) } else { None },
                    if t[1] { Some(&c[0] * x) } else { None },
                ]
            });

            result
                .with_children(vec![self.clone(), other.clone()])
                .with_backward_op(Some(backward_op))
        }
    }
}

impl<'a, 'b> ops::Div<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn div(self, other: &Array) -> Self::Output {
        self * &other.reciprocal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arr;

    #[test]
    fn test_arithmetic() {
        let a = arr![
            arr![arr![0.0, 1.0], arr![2.0, 3.0]],
            arr![arr![4.0, 5.0], arr![6.0, 7.0]]
        ];

        let b = arr![
            arr![arr![2.0, 4.0], arr![6.0, 8.0]],
            arr![arr![10.0, 12.0], arr![14.0, 16.0]]
        ];

        let sum_expect = arr![
            arr![arr![2.0, 5.0], arr![8.0, 11.0]],
            arr![arr![14.0, 17.0], arr![20.0, 23.0]]
        ];

        let product_expect = arr![
            arr![arr![0.0, 4.0], arr![12.0, 24.0]],
            arr![arr![40.0, 60.0], arr![84.0, 112.0]]
        ];

        let sum = &a + &b;
        let product = &a * &b;

        assert_eq!(sum, sum_expect);
        assert_eq!(product, product_expect);
    }

    #[test]
    fn test_neg() {
        let a = arr![1.0, 2.0, 3.0].tracked();
        let b = arr![7.0, 8.0, 9.0].tracked();

        let mut product = (&(-&a) * &b).tracked();
        assert_eq!(product, arr![-7.0, -16.0, -27.0]);

        product.backward(None);
        assert_eq!(product.gradient().unwrap(), arr![1.0, 1.0, 1.0]);
        assert_eq!(b.gradient().unwrap(), arr![-1.0, -2.0, -3.0]);
        assert_eq!(a.gradient().unwrap(), arr![-7.0, -8.0, -9.0]);
    }

    #[test]
    fn test_sub() {
        let a = arr![1.0].tracked();
        let b = arr![3.0].tracked();

        let mut result = &a - &b;
        assert_eq!(result, arr![-2.0]);

        result.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![-1.0]);
        assert_eq!(a.gradient().unwrap(), arr![1.0]);
    }

    #[test]
    fn test_div() {
        let a = arr![arr![2.0, 4.0], arr![1.0, 2.0]].tracked();
        let b = arr![arr![2.0, 2.0], arr![2.0, 1.0]].tracked();

        let mut result = &a / &b;
        assert_eq!(result, arr![arr![1.0, 2.0], arr![0.5, 2.0]]);

        result.backward(None);
        assert_eq!(
            b.gradient().unwrap(),
            arr![arr![-0.5, -1.0], arr![-0.25, -2.0]]
        );
        assert_eq!(a.gradient().unwrap(), arr![arr![0.5, 0.5], arr![0.5, 1.0]]);
    }

    #[test]
    fn test_ln() {
        let a = arr![(1.0 as Float).exp(), (2.0 as Float).exp()];

        let result = a.ln();
        assert_relative_eq!(result, arr![1.0, 2.0]);
    }

    #[test]
    fn test_ln_backward() {
        let a = arr![1.0, 2.0].tracked();

        let mut result = a.ln();
        result.backward(None);
        assert_relative_eq!(a.gradient().unwrap(), arr![1.0, 0.5]);
    }

    #[test]
    fn test_exp() {
        let a = arr![
            (1.0 as Float).ln(),
            (2.0 as Float).ln(),
            (2.0 as Float).ln()
        ]
        .tracked();
        let b = arr![2.0, 4.0, 6.0].tracked();

        let mut result = &a.exp() * &b;
        assert_eq!(result, arr![2.0, 8.0, 12.0]);

        result.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![1.0, 2.0, 2.0]);
        assert_eq!(a.gradient().unwrap(), arr![2.0, 8.0, 12.0]);
    }

    #[test]
    fn test_sum() {
        let a = arr![
            arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]],
            arr![arr![9.0, 8.0, 7.0], arr![7.0, 6.0, 5.0]]
        ]
        .tracked();
        assert_eq!(
            a.sum(1),
            arr![arr![arr![6.0], arr![15.0]], arr![arr![24.0], arr![18.0]]]
        );
        assert_eq!(a.sum(2), arr![arr![21.0], arr![42.0]]);

        let mut result = 2.0 * &a.sum(2);
        result.backward(None);

        let gradient_expect = arr![
            arr![arr![2.0, 2.0, 2.0], arr![2.0, 2.0, 2.0]],
            arr![arr![2.0, 2.0, 2.0], arr![2.0, 2.0, 2.0]]
        ];
        assert_eq!(a.gradient().unwrap(), gradient_expect);
    }

    #[test]
    fn test_backward_sum() {
        let a = arr![1.0, 2.0, 3.0].tracked();
        let b = arr![3.0, 2.0, 1.0].tracked();
        let c = a.sum(0).tracked();
        let mut result = &c * &b;

        result.backward(None);
        assert_eq!(a.gradient().unwrap(), arr![3.0, 2.0, 1.0]);
        assert_eq!(b.gradient().unwrap(), arr![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_powf() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]].tracked();
        let b = arr![arr![3.0, 2.0, 1.0], arr![6.0, 5.0, 4.0]].tracked();
        let c = a.powf(2.0).tracked();

        let mut result = &c * &b;
        assert_eq!(result, arr![arr![3.0, 8.0, 9.0], arr![96.0, 125.0, 144.0]]);

        result.backward(None);
        assert_eq!(
            c.gradient().unwrap(),
            arr![arr![3.0, 2.0, 1.0], arr![6.0, 5.0, 4.0]]
        );
        assert_eq!(
            b.gradient().unwrap(),
            arr![arr![1.0, 4.0, 9.0], arr![16.0, 25.0, 36.0]]
        );
        assert_eq!(
            a.gradient().unwrap(),
            arr![arr![6.0, 8.0, 6.0], arr![48.0, 50.0, 48.0]]
        );
    }

    #[test]
    fn test_backward_div_sum() {
        let a = arr![arr![2.0, 4.0, 2.0]].tracked();

        let mut result = &a / &a.sum(1);
        assert_eq!(result, arr![arr![0.25, 0.5, 0.25]]);

        result.backward(None);
        assert_eq!(a.gradient().unwrap(), arr![arr![0.0, 0.0, 0.0]]);
    }

    #[test]
    fn test_mul_broadcast() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![3.0, 2.0, 1.0]].tracked();
        let b = arr![1.0, 2.0, 3.0].tracked();

        let mut result = &a * &b;
        assert_eq!(result, arr![arr![1.0, 4.0, 9.0], arr![3.0, 4.0, 3.0]]);

        result.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![4.0, 4.0, 4.0]);
        assert_eq!(
            a.gradient().unwrap(),
            arr![arr![1.0, 2.0, 3.0], arr![1.0, 2.0, 3.0]]
        );
    }
}
