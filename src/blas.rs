//! A wrapper for CBLAS.
//! Reference used: https://doc.rust-lang.org/nomicon/ffi.html

use crate::numbers::*;

use std::convert::TryInto;

#[cfg(not(feature = "f32"))]
use libc::c_double;
#[cfg(feature = "f32")]
use libc::c_float;

use libc::c_int;

#[repr(C)]
enum CBLAS_LAYOUT {
    RowMajor = 101,
}

#[repr(C)]
enum CBLAS_TRANSPOSE {
    NoTrans = 111,
    Trans = 112,
}

use self::CBLAS_LAYOUT::*;
use self::CBLAS_TRANSPOSE::*;

#[link(name = "cblas")]
extern "C" {
    #[cfg(not(feature = "f32"))]
    fn cblas_daxpy(
        n: c_int,
        alpha: c_double,
        x: *const c_double,
        inc_x: c_int,
        y: *mut c_double,
        inc_y: c_int,
    );

    #[cfg(feature = "f32")]
    fn cblas_saxpy(
        n: c_int,
        alpha: c_float,
        x: *const c_float,
        inc_x: c_int,
        y: *mut c_float,
        inc_y: c_int,
    );

    #[cfg(not(feature = "f32"))]
    fn cblas_dgemm(
        layout: CBLAS_LAYOUT,
        trans_a: CBLAS_TRANSPOSE,
        trans_b: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );

    #[cfg(feature = "f32")]
    fn cblas_sgemm(
        layout: CBLAS_LAYOUT,
        trans_a: CBLAS_TRANSPOSE,
        trans_b: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );
}

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
        (Trans, output_rows)
    } else {
        (NoTrans, sum_len)
    };

    let (b_transpose, ldb) = if b_transpose {
        (Trans, sum_len)
    } else {
        (NoTrans, output_cols)
    };

    unsafe {
        #[cfg(not(feature = "f32"))]
        cblas_dgemm(
            RowMajor,
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
            RowMajor,
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
