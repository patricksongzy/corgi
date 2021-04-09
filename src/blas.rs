//! A wrapper for CBLAS.
//! Reference used: https://doc.rust-lang.org/nomicon/ffi.html

use crate::numbers::*;

use std::convert::TryInto;

use libc::{c_double, c_int};

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
}

/// Performs a matrix multiplication on two matrices, storing the result in `values`.
///
/// # Arguments
///
/// `values` - The result values slice.
/// `matmul_dimensions` - The dimensions to compute from: `(output_rows, output_cols, sum_len)`.
/// `a` - The LHS matrix, and whether to transpose it: `(a, a_transpose)`.
/// `b` - The RHS matrix, and whether to transpose it: `(b, b_transpose)`.
pub fn matmul_blas(
    values: &mut [Float],
    matmul_dimensions: (usize, usize, usize),
    a: (&[Float], bool),
    b: (&[Float], bool),
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
            0.0,
            values.as_mut_ptr(),
            output_cols,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgemm() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0; 4];
        matmul_blas(&mut c, (2, 2, 3), (&a, false), (&b, false));

        assert_eq!(c, vec![20.0, 14.0, 56.0, 41.0]);
    }
}
