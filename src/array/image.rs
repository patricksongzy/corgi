use crate::array::*;

impl Array {
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
                            for n in 0..filter_cols {
                                let input_index = (n + stride_cols * c)
                                    + image_cols * ((m + stride_rows * r) + image_rows * k);
                                output_slice[output_index] = arrays[0][input_index];
                                output_index += 1;
                            }
                        }
                    }
                }
            }
        });

        Array::sliced_op(
            vec![image],
            &op,
            None,
            &image.dimensions,
            &output_dimensions,
            3,
            0,
        )
    }

    fn roll_blocks(
        unrolled: &Array,
        image_dimensions: (usize, usize, usize),
        stride_dimensions: (usize, usize),
        filter_dimensions: (usize, usize),
    ) -> Array {
        let dimension_count = unrolled.dimensions.len();
        let (image_depth, image_rows, image_cols) = image_dimensions;
        let (_, stride_cols) = stride_dimensions;
        let (filter_rows, filter_cols) = filter_dimensions;

        // the number of unrolled rows
        let unrolled_count = unrolled.dimensions[dimension_count - 2];
        // the length of each unrolled row
        let unrolled_size = unrolled.dimensions[dimension_count - 1] / image_depth;

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
            for i in 0..image_depth {
                let depth_offset = i * image_rows * image_cols;
                let skipped = i * filter_rows * filter_cols;
                for j in 0..unrolled_count {
                    let stride_offset =
                        stride_cols * (j % col_stride_count) + image_cols * (j / col_stride_count);
                    for k in 0..unrolled_size {
                        let filter_offset = (k % filter_cols) + image_cols * (k / filter_cols);
                        output_slice[stride_offset + filter_offset + depth_offset] =
                            arrays[0][k + skipped + unrolled_size * image_depth * j];
                    }
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

        Array::from((Arc::new(output_dimensions), result.values))
    }

    /// Computes the image convolution of the array with the filter.
    pub fn conv(&self, filter: &Array, stride_dimensions: (usize, usize)) -> Array {
        let dimension_count = self.dimensions.len();
        let filter_dimension_count = filter.dimensions.len();
        let unrolled_dimension_count = dimension_count - 1;

        if dimension_count < 3 || filter_dimension_count < 3 {
            panic!("error: cannot convolve with fewer than 3 dimensions");
        }

        let (stride_rows, stride_cols) = stride_dimensions;

        let (image_depth, image_rows, image_cols) = (
            self.dimensions[dimension_count - 3],
            self.dimensions[dimension_count - 2],
            self.dimensions[dimension_count - 1],
        );

        let (filter_rows, filter_cols) = (
            filter.dimensions[filter_dimension_count - 2],
            filter.dimensions[filter_dimension_count - 1],
        );

        let filter_dimensions = (filter_rows, filter_cols);

        let row_stride_count = (image_rows - filter_rows) / stride_rows + 1;
        let col_stride_count = (image_cols - filter_cols) / stride_cols + 1;

        let unrolled = Array::unroll_blocks(&self, stride_dimensions, filter_dimensions);
        let unrolled_size = unrolled.dimensions[unrolled_dimension_count - 1] / image_depth;

        let filter_matrix_dimensions = filter
            .dimensions
            .iter()
            .cloned()
            .take(filter_dimension_count.saturating_sub(3))
            .chain(vec![unrolled_size * image_depth])
            .collect();

        let filter_matrix = Array::from((
            Arc::new(filter_matrix_dimensions),
            Arc::clone(&filter.values),
        ));

        let convolved = Array::matmul((&unrolled, false), (&filter_matrix, true), None);
        let skip_size = convolved.values.len() / image_depth;
        let mut result = vec![0.0; convolved.values.len()];
        let mut result_index = 0;
        for k in 0..image_depth {
            for i in 0..skip_size {
                result[result_index] = convolved[k + image_depth * i];
                result_index += 1;
            }
        }

        let output_dimensions: Vec<usize> = self
            .dimensions
            .iter()
            .take(dimension_count - 3)
            .copied()
            .chain(vec![image_depth, row_stride_count, col_stride_count])
            .collect();

        Array::from((output_dimensions, result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arr;

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

        let filter = arr![arr![arr![3.0, 5.0], arr![2.0, 6.0]]];
        let conv = a.conv(&filter, (1, 1));
        assert_eq!(conv, arr![arr![arr![51.0, 67.0], arr![99.0, 115.0]]]);
    }

    #[test]
    fn test_conv_filter_broadcast() {
        let a = arr![arr![arr![
            arr![1.0, 2.0, 3.0],
            arr![4.0, 5.0, 6.0],
            arr![7.0, 8.0, 9.0]
        ]]];

        let filter = arr![arr![arr![3.0, 5.0], arr![2.0, 6.0]]];
        let conv = a.conv(&filter, (1, 1));
        assert_eq!(conv, arr![arr![arr![arr![51.0, 67.0], arr![99.0, 115.0]]]]);
    }

    #[test]
    fn test_conv_strided() {
        let a = arr![
            arr![arr![1.0, 2.0, 3.0, 4.0], arr![5.0, 6.0, 7.0, 8.0]],
            arr![arr![9.0, 10.0, 11.0, 12.0], arr![13.0, 14.0, 15.0, 16.0]]
        ];

        let filter = arr![
            arr![arr![arr![3.0, 5.0]], arr![arr![1.0, 3.0]]],
            arr![arr![arr![1.0, 3.0]], arr![arr![2.0, 8.0]]]
        ];
        let conv = a.conv(&filter, (1, 2));
        assert_eq!(
            conv,
            arr![
                arr![arr![52.0, 76.0], arr![100.0, 124.0]],
                arr![arr![105.0, 133.0], arr![161.0, 189.0]]
            ]
        );
    }
}
