use crate::array::*;
use crate::numbers::*;

#[cfg(feature = "blas")]
use crate::blas::{daxpy_blas, matmul_blas};

impl Array {
    /// Computes the element-wise `alpha * x + y`, for each matching dimension not multiplied.
    pub fn axpy(alpha: Float, x: &Array, y: &Array) -> Array {
        #[cfg(not(feature = "blas"))]
        return &(alpha * x) + y;
        #[cfg(feature = "blas")]
        {
            let dimensions = element_wise_dimensions(&x.dimensions, &y.dimensions);

            let op: SlicedOp =
                Box::new(move |output_slice: &mut [Float], arrays: Vec<&[Float]>| {
                    output_slice.clone_from_slice(&arrays[1]);
                    daxpy_blas(alpha, arrays[0], output_slice);
                });

            let output_values = Array::sliced_op(vec![x, y], &op, &dimensions, &dimensions, 1);
            let result = Array::from((Arc::new(dimensions), Arc::new(output_values)));

            if !x.is_tracked && !y.is_tracked {
                result
            } else {
                let backward_op: BackwardOp = Arc::new(move |c, t, x| {
                    vec![
                        if t[0] {
                            Some(
                                -2.0 * &Array::from((
                                    Arc::clone(&x.dimensions),
                                    Arc::clone(&x.values),
                                )),
                            )
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
                    .with_children(vec![x.clone(), y.clone()])
                    .with_backward_op(Some(backward_op))
            }
        }
    }

    /// Computes a matrix multiplication on two matrices, storing the result in `values`.
    ///
    /// # Arguments
    ///
    /// `values` - The result values slice.
    /// `matmul_dimensions` - The dimensions to compute from: `(output_rows, output_cols, sum_len)`.
    /// `a` - The LHS matrix, and whether to transpose it: `(a, a_transpose)`.
    /// `b` - The RHS matrix, and whether to transpose it: `(b, b_transpose)`.
    #[cfg(not(feature = "blas"))]
    fn matmul_slice(
        values: &mut [Float],
        matmul_dimensions: (usize, usize, usize),
        a: (&[Float], bool),
        b: (&[Float], bool),
    ) {
        let (output_rows, output_cols, sum_len) = matmul_dimensions;
        let (a, a_transpose) = a;
        let (b, b_transpose) = b;
        for r in 0..output_rows {
            for j in 0..output_cols {
                let mut sum = 0.0;
                for k in 0..sum_len {
                    let a_index = if a_transpose {
                        k * output_rows + r
                    } else {
                        r * sum_len + k
                    };

                    let b_index = if b_transpose {
                        j * sum_len + k
                    } else {
                        k * output_cols + j
                    };

                    sum += a[a_index] * b[b_index];
                }

                values[r * output_cols + j] = sum;
            }
        }
    }

    /// Computes matrix multiplications on two arrays, for each matching dimension not multiplied.
    ///
    /// # Arguments
    ///
    /// `a` - The LHS matrix, and whether to transpose it: `(a, a_transpose)`.
    /// `b` - The RHS matrix, and whether to transpose it: `(b, b_transpose)`.
    fn matmul_values(a: (&Array, bool), b: (&Array, bool)) -> Array {
        let (a, a_transpose) = a;
        let (b, b_transpose) = b;

        let input_dimensions = if a.dimensions.len() >= b.dimensions.len() {
            &a.dimensions
        } else {
            &b.dimensions
        };

        // TODO OpenCL
        let output_rows = if a.dimensions.len() < 2 && (!a_transpose || b.dimensions.len() < 2) {
            1
        } else {
            a.dimensions[a.dimensions.len() - if a_transpose { 1 } else { 2 }]
        };

        let output_cols = if b.dimensions.len() < 2 && (b_transpose || a.dimensions.len() < 2) {
            1
        } else {
            b.dimensions[b.dimensions.len() - if b_transpose { 2 } else { 1 }]
        };

        let sum_len = {
            let a_index = if a_transpose { 2 } else { 1 };
            let b_index = if b_transpose { 1 } else { 2 };

            if a.dimensions.len() < a_index {
                if b.dimensions.len() >= b_index {
                    b.dimensions[b.dimensions.len() - b_index]
                } else {
                    1
                }
            } else {
                let sum_len = a.dimensions[a.dimensions.len() - a_index];
                if b.dimensions.len() >= b_index
                    && sum_len != b.dimensions[b.dimensions.len() - b_index]
                {
                    panic!(
                        "error: the dimensions {:?}, and {:?} are not compatible",
                        a.dimensions, b.dimensions
                    );
                } else {
                    sum_len
                }
            }
        };

        let leading_count = input_dimensions.len().saturating_sub(2);
        let output_dimensions: Vec<usize> = input_dimensions
            .iter()
            .copied()
            .take(leading_count)
            .chain(if input_dimensions.len() < 2 {
                vec![output_cols]
            } else {
                vec![output_rows, output_cols]
            })
            .collect();

        let op: SlicedOp = Box::new(move |output_slice: &mut [Float], arrays: Vec<&[Float]>| {
            #[cfg(feature = "blas")]
            matmul_blas(
                (output_rows, output_cols, sum_len),
                (arrays[0], a_transpose),
                (arrays[1], b_transpose),
                output_slice,
            );
            #[cfg(not(feature = "blas"))]
            Array::matmul_slice(
                output_slice,
                (output_rows, output_cols, sum_len),
                (arrays[0], a_transpose),
                (arrays[1], b_transpose),
            );
        });

        let output_values =
            Array::sliced_op(vec![a, b], &op, &input_dimensions, &output_dimensions, 2);
        let result = Array::from((output_dimensions, output_values));

        if !a.is_tracked && !b.is_tracked {
            result
        } else {
            let backward_a = Box::new(move |c: &mut Vec<Array>, x: &Array| {
                if a_transpose {
                    Array::matmul_values((&c[1], b_transpose), (x, true))
                } else {
                    Array::matmul_values((x, false), (&c[1], !b_transpose))
                }
            });

            let backward_b = Box::new(move |c: &mut Vec<Array>, x: &Array| {
                if b_transpose {
                    Array::matmul_values((&x, true), (&c[0], a_transpose))
                } else {
                    Array::matmul_values((&c[0], !a_transpose), (&x, false))
                }
            });

            let backward_op: BackwardOp = Arc::new(move |c, t, x| {
                vec![
                    if t[0] { Some(backward_a(c, x)) } else { None },
                    if t[1] { Some(backward_b(c, x)) } else { None },
                ]
            });

            result
                .with_children(vec![a.clone(), b.clone()])
                .with_backward_op(Some(backward_op))
        }
    }

    /// Computes matrix multiplications on two arrays, for each matching dimension not multiplied.
    ///
    /// # Arguments
    ///
    /// * `a` - The LHS matrix, and whether to transpose it: `(a, a_transpose)`.
    /// * `b` - The RHS matrix, and whether to transpose it: `(b, b_transpose)`.
    pub fn matmul(a: (&Array, bool), b: (&Array, bool)) -> Array {
        Array::matmul_values(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arr;

    #[test]
    fn test_axpy() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![3.0, 2.0, 1.0]].tracked();
        let b = arr![arr![5.0, 6.0, 7.0], arr![9.0, 8.0, 7.0]].tracked();
        let mut result = Array::axpy(-2.0, &a, &b);
        assert_eq!(result, arr![arr![3.0, 2.0, 1.0], arr![3.0, 4.0, 5.0]]);

        result.backward(None);
        assert_eq!(
            b.gradient().unwrap(),
            arr![arr![1.0, 1.0, 1.0], arr![1.0, 1.0, 1.0]]
        );
        assert_eq!(
            a.gradient().unwrap(),
            arr![arr![-2.0, -2.0, -2.0], arr![-2.0, -2.0, -2.0]]
        );
    }

    #[test]
    fn test_matmul() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]].tracked();
        let b = arr![arr![5.0, 3.0], arr![2.0, 6.0], arr![1.0, 2.0]].tracked();

        let matmul_expect = arr![arr![12.0, 21.0], arr![36.0, 54.0]];

        let mut result = Array::matmul((&a, false), (&b, false));
        assert_eq!(result, matmul_expect);

        result.backward(None);
        assert_eq!(
            b.gradient().unwrap(),
            arr![arr![5.0, 5.0], arr![7.0, 7.0], arr![9.0, 9.0]]
        );
        assert_eq!(
            a.gradient().unwrap(),
            arr![arr![8.0, 8.0, 3.0], arr![8.0, 8.0, 3.0]]
        );
    }

    #[test]
    fn test_matmul_transpose() {
        let a = arr![arr![1.0, 4.0], arr![2.0, 5.0], arr![3.0, 6.0]].tracked();
        let b = arr![arr![5.0, 3.0], arr![2.0, 6.0], arr![1.0, 2.0]].tracked();

        let matmul_expect = arr![arr![12.0, 21.0], arr![36.0, 54.0]];

        let mut result = Array::matmul((&a, true), (&b, false));
        assert_eq!(result, matmul_expect);

        result.backward(None);
        assert_eq!(
            b.gradient().unwrap(),
            arr![arr![5.0, 5.0], arr![7.0, 7.0], arr![9.0, 9.0]]
        );
        assert_eq!(
            a.gradient().unwrap(),
            arr![arr![8.0, 8.0], arr![8.0, 8.0], arr![3.0, 3.0]]
        );
    }

    #[test]
    fn test_matmul_broadcast() {
        let a = arr![
            arr![arr![1.0, 2.0, 3.0], arr![3.0, 2.0, 1.0]],
            arr![arr![4.0, 5.0, 6.0], arr![7.0, 8.0, 9.0]]
        ]
        .tracked();
        let b = arr![arr![1.0, 2.0, 3.0]].tracked();

        let mut result = Array::matmul((&a, false), (&b, true));
        assert_eq!(
            result,
            arr![arr![arr![14.0], arr![10.0]], arr![arr![32.0], arr![50.0]]]
        );

        result.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![arr![15.0, 17.0, 19.0]]);
        assert_eq!(
            a.gradient().unwrap(),
            arr![
                arr![arr![1.0, 2.0, 3.0], arr![1.0, 2.0, 3.0]],
                arr![arr![1.0, 2.0, 3.0], arr![1.0, 2.0, 3.0]]
            ]
        );
    }

    #[test]
    fn test_matmul_broadcast_vec() {
        let a = arr![1.0, 2.0].tracked();
        let b = arr![arr![2.0, 4.0]].tracked();

        let mut result = Array::matmul((&a, true), (&b, false));
        assert_eq!(result, arr![arr![2.0, 4.0], arr![4.0, 8.0]]);

        result.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![arr![3.0, 3.0]]);
        assert_eq!(a.gradient().unwrap(), arr![6.0, 6.0]);
    }

    #[test]
    fn test_matmul_broadcast_dense() {
        let w = arr![arr![1.0, 2.0], arr![8.0, 2.0]].tracked();
        let x = arr![5.0, 3.0];

        let mut result = Array::matmul((&x, false), (&w, true));
        result.backward(None);
        assert_eq!(w.gradient().unwrap(), arr![arr![5.0, 3.0], arr![5.0, 3.0]]);
    }

    #[test]
    fn test_matmul_vec() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]].tracked();
        let b = arr![arr![1.0, 2.0, 3.0]].tracked();
        let c = arr![arr![1.0], arr![2.0], arr![3.0]].tracked();

        let result = Array::matmul((&a, false), (&b, true));
        assert_eq!(result, arr![arr![14.0], arr![32.0]]);

        let result = Array::matmul((&b, false), (&a, true));
        assert_eq!(result, arr![arr![14.0, 32.0]]);

        let result = Array::matmul((&b, false), (&c, false));
        assert_eq!(result, arr![arr![14.0]]);
    }

    #[test]
    fn test_matmul_single() {
        let a = arr![1.0, 2.0, 3.0];
        let b = arr![3.0, 2.0, 1.0];
        let result = Array::matmul((&a, false), (&b, false));
        assert_eq!(result, arr![10.0]);
    }

    #[test]
    fn test_matmul_multi() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];
        let b = arr![arr![1.0], arr![2.0], arr![3.0]];
        let c = arr![arr![1.0, 2.0, 3.0]];

        let result = Array::matmul(
            (&Array::matmul((&a, false), (&b, false)), false),
            (&c, false),
        );
        assert_eq!(result, arr![arr![14.0, 28.0, 42.0], arr![32.0, 64.0, 96.0]]);
    }

    #[test]
    fn test_matmul_nd() {
        let a = arr![
            arr![
                arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]],
                arr![arr![6.0, 5.0, 4.0], arr![3.0, 2.0, 1.0]]
            ],
            arr![
                arr![arr![9.0, 8.0, 7.0], arr![4.0, 5.0, 6.0]],
                arr![arr![6.0, 7.0, 8.0], arr![3.0, 2.0, 1.0]]
            ]
        ];

        let b = arr![
            arr![
                arr![arr![5.0, 3.0], arr![2.0, 6.0], arr![1.0, 2.0]],
                arr![arr![3.0, 6.0], arr![2.0, 5.0], arr![1.0, 4.0]]
            ],
            arr![
                arr![arr![5.0, 3.0], arr![2.0, 6.0], arr![8.0, 7.0]],
                arr![arr![8.0, 6.0], arr![5.0, 3.0], arr![4.0, 7.0]]
            ]
        ];

        let matmul_expect = arr![
            arr![
                arr![arr![12.0, 21.0], arr![36.0, 54.0]],
                arr![arr![32.0, 77.0], arr![14.0, 32.0]]
            ],
            arr![
                arr![arr![117.0, 124.0], arr![78.0, 84.0]],
                arr![arr![115.0, 113.0], arr![38.0, 31.0]]
            ]
        ];

        let result = Array::matmul((&a, false), (&b, false));
        assert_eq!(result, matmul_expect);
    }

    #[test]
    fn test_backward_matmul_vec() {
        let a = arr![arr![1.0, 2.0, 3.0]].tracked();
        let b = arr![arr![9.0, 8.0, 7.0]].tracked();

        let mut result = Array::matmul((&a, false), (&b, true));

        result.backward(None);
        assert_eq!(a.gradient().unwrap(), arr![arr![9.0, 8.0, 7.0]]);
        assert_eq!(b.gradient().unwrap(), arr![arr![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_backward_matmul_vec_multi() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]].tracked();
        let b = arr![arr![1.0], arr![2.0], arr![3.0]].tracked();
        let c = arr![arr![7.0], arr![8.0]].tracked();

        let mut result = &Array::matmul((&a, false), (&b, false)) + &c;

        result.backward(None);
        assert_eq!(c.gradient().unwrap(), arr![arr![1.0], arr![1.0]]);
        assert_eq!(b.gradient().unwrap(), arr![arr![5.0], arr![7.0], arr![9.0]]);
        assert_eq!(
            a.gradient().unwrap(),
            arr![arr![1.0, 2.0, 3.0], arr![1.0, 2.0, 3.0]]
        );
    }

    #[test]
    fn test_backward_matmul_multi() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]].tracked();
        let b = arr![arr![1.0], arr![2.0], arr![3.0]].tracked();
        let c = arr![arr![1.0, 2.0, 3.0]].tracked();

        let mut result = Array::matmul(
            (&Array::matmul((&a, false), (&b, false)), false),
            (&c, false),
        );

        result.backward(None);
        assert_eq!(c.gradient().unwrap(), arr![arr![46.0, 46.0, 46.0]]);
        assert_eq!(
            b.gradient().unwrap(),
            arr![arr![30.0], arr![42.0], arr![54.0]]
        );
        assert_eq!(
            a.gradient().unwrap(),
            arr![arr![6.0, 12.0, 18.0], arr![6.0, 12.0, 18.0]]
        );
    }
}
