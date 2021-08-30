use crate::array::*;

impl Array {
    /// Unrolls an image into rows of blocks with the length of the filter.
    fn unroll_blocks(
        image: &Array,
        stride_dimensions: (usize, usize),
        filter_dimensions: (usize, usize),
    ) -> Array {
        let dimension_count = image.dimensions.len();
        let (stride_rows, stride_cols) = stride_dimensions;
        let (filter_rows, filter_cols) = filter_dimensions;

        let image_depth = image.dimensions[dimension_count - 3];
        let image_rows = image.dimensions[dimension_count - 2];
        let image_cols = image.dimensions[dimension_count - 1];

        let image_dimensions = (image_depth, image_rows, image_cols);

        // the number of values in strided to
        let row_stride_count = (image_rows - filter_rows) / stride_rows + 1;
        let col_stride_count = (image_cols - filter_cols) / stride_cols + 1;

        // the number of unrolled rows
        let unrolled_count = row_stride_count * col_stride_count;
        // the length of each unrolled row
        let unrolled_size = filter_rows * filter_cols;

        let output_dimensions: Vec<usize> = image
            .dimensions
            .iter()
            .cloned()
            .take(dimension_count - 3)
            .chain(vec![unrolled_count, image_depth * unrolled_size])
            .collect();

        let op: SlicedOp = Box::new(move |output_slice, arrays| {
            let mut output_index = 0;
            for r in 0..row_stride_count {
                for c in 0..col_stride_count {
                    for k in 0..image_depth {
                        for m in 0..filter_rows {
                            // the filter row position plus the stride row position
                            let row_index = m + stride_rows * r;
                            for n in 0..filter_cols {
                                // the filter col position plus the stride col position
                                let col_index = n + stride_cols * c;
                                let input_index =
                                    col_index + image_cols * (row_index + image_rows * k);
                                output_slice[output_index] = arrays[0][input_index];
                                output_index += 1;
                            }
                        }
                    }
                }
            }
        });

        let result = Array::sliced_op(
            vec![image],
            &op,
            None,
            &image.dimensions,
            &output_dimensions,
            3,
            0,
        );

        if !image.is_tracked.get() {
            result
        } else {
            let backward_op: BackwardOp = Rc::new(move |_, t, x| {
                vec![if t[0] {
                    Some(Array::roll_blocks(
                        &x,
                        image_dimensions,
                        stride_dimensions,
                        filter_dimensions,
                    ))
                } else {
                    None
                }]
            });
            result
                .with_backward_op(backward_op)
                .with_children(vec![image.clone()])
        }
    }

    /// Inverse of unrolling the blocks.
    fn roll_blocks(
        unrolled: &Array,
        image_dimensions: (usize, usize, usize),
        stride_dimensions: (usize, usize),
        filter_dimensions: (usize, usize),
    ) -> Array {
        let dimension_count = unrolled.dimensions.len();
        let (image_depth, image_rows, image_cols) = image_dimensions;
        let (stride_rows, stride_cols) = stride_dimensions;
        let (filter_rows, filter_cols) = filter_dimensions;

        // the number of unrolled rows
        let unrolled_count = unrolled.dimensions[dimension_count - 2];
        // the length of each unrolled row
        let unrolled_size = filter_rows * filter_cols;

        // the number of values in strided to
        let col_stride_count = (image_cols - filter_cols) / stride_cols + 1;

        let leading_dimensions = unrolled
            .dimensions
            .iter()
            .copied()
            .take(dimension_count - 2);

        let output_dimensions: Vec<usize> = leading_dimensions
            .chain(vec![image_depth, image_rows, image_cols])
            .collect();

        let op: SlicedOp = Box::new(move |output_slice, arrays| {
            for i in 0..unrolled_count {
                let stride_offset = {
                    let stride_row_index = i / col_stride_count;
                    let stride_col_index = i % col_stride_count;

                    let stride_row_offset = image_cols * stride_rows * stride_row_index;
                    let stride_col_offset = stride_cols * stride_col_index;

                    stride_row_offset + stride_col_offset
                };

                for j in 0..unrolled_size * image_depth {
                    let current_depth = j / unrolled_size;
                    let filter_index = j % unrolled_size;

                    let input_index = j + unrolled_size * image_depth * i;
                    let output_index = {
                        let depth_offset = image_rows * image_cols * current_depth;

                        let filter_row_index = filter_index / filter_cols;
                        let filter_col_index = filter_index % filter_cols;
                        let filter_row_offset = image_cols * filter_row_index;

                        filter_col_index + filter_row_offset + depth_offset + stride_offset
                    };

                    output_slice[output_index] = arrays[0][input_index];
                }
            }
        });

        let result = Array::sliced_op(
            vec![unrolled],
            &op,
            None,
            &unrolled.dimensions,
            &output_dimensions,
            3,
            0,
        );

        if !unrolled.is_tracked.get() {
            result
        } else {
            let backward_op: BackwardOp = Rc::new(move |_, t, x| {
                vec![if t[0] {
                    Some(Array::unroll_blocks(
                        &x,
                        stride_dimensions,
                        filter_dimensions,
                    ))
                } else {
                    None
                }]
            });
            result
                .with_backward_op(backward_op)
                .with_children(vec![unrolled.clone()])
        }
    }

    /// Transforms arrays of the form (output rows * output cols, depth) to (depth, output rows, output cols).
    fn expand_conv(&self, stride_counts: (usize, usize)) -> Array {
        let (row_stride_count, col_stride_count) = stride_counts;
        let filter_count = self.dimensions[self.dimensions.len() - 1];

        let values_length = self.values.len();
        // the stride between two convolution outputs
        let stride = values_length / filter_count;
        let mut result = vec![0.0; values_length];
        let mut result_index = 0;
        for k in 0..filter_count {
            for i in 0..stride {
                result[result_index] = self.values[k + filter_count * i];
                result_index += 1;
            }
        }

        let output_dimensions: Vec<usize> = self
            .dimensions
            .iter()
            .take(self.dimensions.len() - 2)
            .copied()
            .chain(vec![filter_count, row_stride_count, col_stride_count])
            .collect();

        let result = Array::from((output_dimensions, result));

        if !self.is_tracked.get() {
            result
        } else {
            let backward_op: BackwardOp = Rc::new(move |c, _, x| {
                let mut result = vec![0.0; values_length];
                let mut delta_index = 0;
                for k in 0..filter_count {
                    for i in 0..stride {
                        result[k + filter_count * i] = x.values[delta_index];
                        delta_index += 1;
                    }
                }

                vec![Some(Array::from((c[0].dimensions.clone(), result)))]
            });

            result
                .with_backward_op(backward_op)
                .with_children(vec![self.clone()])
        }
    }

    /// Computes the image convolution of the array with the filter.
    pub fn conv(&self, filters: &Array, stride_dimensions: (usize, usize)) -> Array {
        let dimension_count = self.dimensions.len();
        let filter_dimension_count = filters.dimensions.len();
        // the number of dimensions for the unrolled image
        let unrolled_dimension_count = dimension_count - 1;

        assert!(
            dimension_count >= 3 && filter_dimension_count >= 3,
            "error: cannot convolve with fewer than 3 dimensions"
        );

        let (stride_rows, stride_cols) = stride_dimensions;

        let (image_depth, image_rows, image_cols) = (
            self.dimensions[dimension_count - 3],
            self.dimensions[dimension_count - 2],
            self.dimensions[dimension_count - 1],
        );

        let (filter_rows, filter_cols) = (
            filters.dimensions[filter_dimension_count - 2],
            filters.dimensions[filter_dimension_count - 1],
        );

        let filter_dimensions = (filter_rows, filter_cols);

        let row_stride_count = (image_rows - filter_rows) / stride_rows + 1;
        let col_stride_count = (image_cols - filter_cols) / stride_cols + 1;

        // convert image dimensions to (unrolled count, unrolled size * image depth)
        let unrolled = Array::unroll_blocks(&self, stride_dimensions, filter_dimensions);
        let unrolled_size = unrolled.dimensions[unrolled_dimension_count - 1] / image_depth;

        // combine last three filter dimensions to single row to (filter count, unrolled size * image depth)
        let filter_matrix_dimensions = filters
            .dimensions
            .iter()
            .cloned()
            .take(filter_dimension_count.saturating_sub(3))
            .chain(vec![unrolled_size * image_depth])
            .collect();

        let filter_matrix = filters.reshape(filter_matrix_dimensions);

        // convert unrolled dimensions to (unrolled count, filter count)
        let convolved = Array::matmul((&unrolled, false), (&filter_matrix, true), None);
        // convert convolved dimensions to (filter count, row stride count, col stride count)
        convolved.expand_conv((row_stride_count, col_stride_count))
    }

    // pub fn pool(&self, pool_dimensions: (usize, usize)) {
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arr;

    #[test]
    fn test_expand_conv() {
        let a = arr![arr![1.0, 4.0], arr![2.0, 5.0], arr![3.0, 6.0]].tracked();
        let expanded = a.expand_conv((1, 3));
        assert_eq!(
            expanded,
            arr![arr![arr![1.0, 2.0, 3.0]], arr![arr![4.0, 5.0, 6.0]]]
        );

        expanded.backward(Some(expanded.clone()));
        assert_eq!(a.gradient().to_owned().unwrap(), a.clone());
    }

    #[test]
    fn test_rolling() {
        let a = arr![arr![
            arr![1.0, 2.0, 3.0],
            arr![4.0, 5.0, 6.0],
            arr![7.0, 8.0, 9.0]
        ]];
        let result = Array::unroll_blocks(&a, (1, 1), (2, 2));
        assert_eq!(
            result,
            arr![
                arr![1.0, 2.0, 4.0, 5.0],
                arr![2.0, 3.0, 5.0, 6.0],
                arr![4.0, 5.0, 7.0, 8.0],
                arr![5.0, 6.0, 8.0, 9.0]
            ]
        );
        let rolled = Array::roll_blocks(&result, (1, 3, 3), (1, 1), (2, 2));
        assert_eq!(rolled, a);
    }

    #[test]
    fn test_rolling_rect() {
        let a = arr![arr![
            arr![1.0, 2.0, 3.0, 4.0],
            arr![5.0, 6.0, 7.0, 8.0],
            arr![9.0, 10.0, 11.0, 12.0]
        ]];
        let result = Array::unroll_blocks(&a, (1, 1), (2, 3));
        assert_eq!(
            result,
            arr![
                arr![1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
                arr![2.0, 3.0, 4.0, 6.0, 7.0, 8.0],
                arr![5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
                arr![6.0, 7.0, 8.0, 10.0, 11.0, 12.0]
            ]
        );
        let rolled = Array::roll_blocks(&result, (1, 3, 4), (1, 1), (2, 3));
        assert_eq!(rolled, a);
    }

    #[test]
    fn test_rolling_strided() {
        let a = arr![
            arr![arr![1.0, 2.0, 3.0, 4.0], arr![5.0, 6.0, 7.0, 8.0]],
            arr![arr![9.0, 10.0, 11.0, 12.0], arr![13.0, 14.0, 15.0, 16.0]]
        ];
        let result = Array::unroll_blocks(&a, (1, 2), (1, 2));
        assert_eq!(
            result,
            arr![
                arr![1.0, 2.0, 9.0, 10.0],
                arr![3.0, 4.0, 11.0, 12.0],
                arr![5.0, 6.0, 13.0, 14.0],
                arr![7.0, 8.0, 15.0, 16.0]
            ],
        );
        let rolled = Array::roll_blocks(&result, (2, 2, 4), (1, 2), (1, 2));
        assert_eq!(rolled, a);
    }

    #[test]
    fn test_conv() {
        let a = arr![arr![
            arr![1.0, 2.0, 3.0],
            arr![4.0, 5.0, 6.0],
            arr![7.0, 8.0, 9.0]
        ]];

        let filters = arr![arr![arr![3.0, 5.0], arr![2.0, 6.0]]];
        let conv = a.conv(&filters, (1, 1));
        assert_eq!(conv, arr![arr![arr![51.0, 67.0], arr![99.0, 115.0]]]);
    }

    #[test]
    fn test_conv_filter_broadcast() {
        let a = arr![arr![arr![
            arr![1.0, 2.0, 3.0],
            arr![4.0, 5.0, 6.0],
            arr![7.0, 8.0, 9.0]
        ]]];

        let filters = arr![arr![arr![3.0, 5.0], arr![2.0, 6.0]]];
        let conv = a.conv(&filters, (1, 1));
        assert_eq!(conv, arr![arr![arr![arr![51.0, 67.0], arr![99.0, 115.0]]]]);
    }

    #[test]
    fn test_conv_multi() {
        let input_dimensions = vec![3, 9, 9];
        let input_size = input_dimensions.iter().product();
        let input_values: Vec<Float> = (0..input_size).map(|x| (x % 3) as Float).collect();

        let f1_dimensions = vec![16, 3, 3, 3];
        let f1_size = f1_dimensions.iter().product();
        let f1_values: Vec<Float> = (0..f1_size).map(|x| (x % 2) as Float).collect();

        let f2_dimensions = vec![1, 16, 2, 2];
        let f2_size = f2_dimensions.iter().product();
        let f2_values: Vec<Float> = (0..f2_size).map(|x| (x % 5) as Float).collect();

        let stride_dimensions = (2, 2);

        let a = Array::from((input_dimensions, input_values)).tracked();
        let f1 = Array::from((f1_dimensions, f1_values)).tracked();
        let f2 = Array::from((f2_dimensions, f2_values)).tracked();

        let b = a.conv(&f1, stride_dimensions);
        let result = b.conv(&f2, stride_dimensions);

        result.backward(None);

        assert_eq!(f2.gradient().to_owned().unwrap()[f2_size - 1], 58.0);
        assert_eq!(f1.gradient().to_owned().unwrap()[f1_size - 1], 32.0);
        assert_eq!(
            b.gradient().to_owned().unwrap()[b.gradient().to_owned().unwrap().values.len() - 1],
            3.0
        );
        assert_eq!(a.gradient().to_owned().unwrap()[input_size - 1], 15.0);
    }

    #[test]
    fn test_conv_strided() {
        let a = arr![
            arr![arr![1.0, 2.0, 3.0, 4.0], arr![5.0, 6.0, 7.0, 8.0]],
            arr![arr![9.0, 10.0, 11.0, 12.0], arr![13.0, 14.0, 15.0, 16.0]]
        ]
        .tracked();

        let filters = arr![
            arr![arr![arr![3.0, 5.0]], arr![arr![1.0, 3.0]]],
            arr![arr![arr![1.0, 3.0]], arr![arr![2.0, 8.0]]],
            arr![arr![arr![1.0, 3.0]], arr![arr![2.0, 8.0]]]
        ]
        .tracked();

        let conv = a.conv(&filters, (1, 2));
        assert_eq!(
            conv,
            arr![
                arr![arr![52.0, 76.0], arr![100.0, 124.0]],
                arr![arr![105.0, 133.0], arr![161.0, 189.0]],
                arr![arr![105.0, 133.0], arr![161.0, 189.0]]
            ]
        );

        conv.backward(Some(arr![arr![
            arr![arr![1.0, 2.0], arr![3.0, 4.0]],
            arr![arr![5.0, 6.0], arr![7.0, 8.0]],
            arr![arr![9.0, 10.0], arr![11.0, 12.0]]
        ]]));

        assert_eq!(
            a.gradient().to_owned().unwrap(),
            arr![
                arr![arr![17.0, 47.0, 22.0, 58.0], arr![27.0, 69.0, 32.0, 80.0]],
                arr![
                    arr![29.0, 115.0, 34.0, 134.0],
                    arr![39.0, 153.0, 44.0, 172.0]
                ]
            ]
        );
        assert_eq!(
            filters.gradient().to_owned().unwrap(),
            arr![
                arr![arr![arr![50.0, 60.0]], arr![arr![130.0, 140.0]]],
                arr![arr![arr![114.0, 140.0]], arr![arr![322.0, 348.0]]],
                arr![arr![arr![178.0, 220.0]], arr![arr![514.0, 556.0]]]
            ]
        );
    }
}
