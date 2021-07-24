//! A wrapper for CBLAS.
//! Reference used: https://doc.rust-lang.org/nomicon/ffi.html

#[cfg(not(feature = "f32"))]
use cblas_sys::{cblas_daxpy, cblas_dgemm};
#[cfg(feature = "f32")]
use cblas_sys::{cblas_saxpy, cblas_sgemm};

use cblas_sys::CBLAS_LAYOUT;
use cblas_sys::CBLAS_TRANSPOSE;

use crate::numbers::*;

use std::convert::TryInto;

use self::CBLAS_LAYOUT::*;
use self::CBLAS_TRANSPOSE::*;

pub(crate) fn daxpy_blas(alpha: Float, x: &[Float], y: &mut [Float]) {
    unsafe {
        #[cfg(not(feature = "f32"))]
        cblas_daxpy(
            y.len().try_into().unwrap(),
            alpha,
            x.as_ptr(),
            1,
            y.as_mut_ptr(),
            1,
        );
        #[cfg(feature = "f32")]
        cblas_saxpy(
            y.len().try_into().unwrap(),
            alpha,
            x.as_ptr(),
            1,
            y.as_mut_ptr(),
            1,
        );
    }
}

/// Performs a matrix multiplication on two matrices, storing the result in `values`.
///
/// # Arguments
///
/// `values` - The result values slice.
/// `matmul_dimensions` - The dimensions to compute from: `(output_rows, output_cols, sum_len)`.
/// `a` - The LHS matrix, and whether to transpose it: `(a, a_transpose)`.
/// `b` - The RHS matrix, and whether to transpose it: `(b, b_transpose)`.
pub(crate) fn matmul_blas(
    matmul_dimensions: (usize, usize, usize),
    a: (&[Float], bool),
    b: (&[Float], bool),
    values: &mut [Float],
) {
    let (a, a_transpose) = a;
    let (b, b_transpose) = b;
    let (output_rows, output_cols, sum_len) = matmul_dimensions;
    let (output_rows, output_cols, sum_len) = (
        output_rows.try_into().unwrap(),
        output_cols.try_into().unwrap(),
        sum_len.try_into().unwrap(),
    );

    let (a_transpose, lda) = if a_transpose {
        (CblasTrans, output_rows)
    } else {
        (CblasNoTrans, sum_len)
    };

    let (b_transpose, ldb) = if b_transpose {
        (CblasTrans, sum_len)
    } else {
        (CblasNoTrans, output_cols)
    };

    unsafe {
        #[cfg(not(feature = "f32"))]
        cblas_dgemm(
            CblasRowMajor,
            a_transpose,
            b_transpose,
            output_rows,
            output_cols,
            sum_len,
            1.0,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            1.0,
            values.as_mut_ptr(),
            output_cols,
        );
        #[cfg(feature = "f32")]
        cblas_sgemm(
            CblasRowMajor,
            a_transpose,
            b_transpose,
            output_rows,
            output_cols,
            sum_len,
            1.0,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            1.0,
            values.as_mut_ptr(),
            output_cols,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daxpy() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![5.0, 6.0, 7.0];
        daxpy_blas(-2.0, &x, &mut y);

        assert_eq!(y, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_dgemm() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0; 4];
        matmul_blas((2, 2, 3), (&a, false), (&b, false), &mut c);

        assert_eq!(c, vec![20.0, 14.0, 56.0, 41.0]);
    }
}
