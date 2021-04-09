//! An n-dimensional array, with automatic differentation.
//! # Examples
//! See the README for more examples.
//! ```
//! # #[macro_use]
//! # extern crate corgi;
//! use corgi::array::*;
//!
//! # fn main() {
//! let a = arr![5.0];
//! let b = arr![2.0];
//! let mut c = arr![0.0];
//!
//! for _ in 0..10 {
//!     c = &c + &(&a * &b);
//!     if c[0] > 50.0 {
//!         c = &c * &a;
//!     }
//! }
//!
//! c.backward(None);
//! assert_eq!(c, arr![195300.0]);
//! assert_eq!(c.gradient(), arr![1.0]);
//! assert_eq!(b.gradient(), arr![97650.0]);
//! assert_eq!(a.gradient(), arr![232420.0]);
//! # }
//! ```

#[cfg(feature = "blas")]
use crate::blas::matmul_blas;
use crate::numbers::*;

use std::ops;
use std::ops::Index;

use std::fmt;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

// TODO more ergonomic Array creation
/// Helper trait to construct `Array` structs.
pub trait Arrays {
    /// Constructs a new `Array`.
    fn new(self) -> Array;
}

/// Implementation to construct `Array` structs by flattening other contained `Array` structs, and keeping
/// their dimensions.
///
/// # Examples
///
/// ```
/// # #[macro_use]
/// # extern crate corgi;
/// # use corgi::array::*;
/// # fn main () {
/// let a = Arrays::new(vec![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]]);
/// assert_eq!(a[vec![1, 2]], 6.0);
/// # }
/// ```
impl Arrays for Vec<Array> {
    fn new(self) -> Array {
        // check if any of the contained array dimensions mismatch
        let is_dimensions_valid = match self.split_first() {
            Some((first, elements)) => elements
                .iter()
                .all(|item| *item.dimensions == *first.dimensions),
            None => true,
        };

        if !is_dimensions_valid {
            panic!("error: contained array dimensions must all be the same");
        }

        let mut dimensions = vec![self.len()];
        dimensions.append(&mut (*self.first().unwrap().dimensions).clone());

        // take ownership if possible, but clone otherwise
        let values = self
            .into_iter()
            .map(|array| Arc::try_unwrap(array.values).unwrap_or_else(|x| (*x).clone()))
            .flatten()
            .collect::<Vec<Float>>();

        Arrays::new((dimensions, values))
    }
}

/// Implementation to construct `Array` structs by using `Vec<Float>` as the values, and by keeping flat dimensions.
///
/// # Examples
///
/// ```
/// # extern crate corgi;
/// # use corgi::array::*;
/// # fn main () {
/// let a = Arrays::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// assert_eq!(a[vec![5]], 6.0);
/// # }
/// ```
impl Arrays for Vec<Float> {
    fn new(self) -> Array {
        Arrays::new((vec![self.len()], self))
    }
}

/// Implementation to construct `Array` structs by using `Vec<usize>` as the dimensions, and filling values with zeros.
///
/// # Examples
///
/// ```
/// # extern crate corgi;
/// # use corgi::array::*;
/// # fn main () {
/// let a = Arrays::new(vec![3, 2, 3]);
/// assert_eq!(a[vec![2, 1, 1]], 0.0);
/// # }
/// ```
impl Arrays for Vec<usize> {
    fn new(self) -> Array {
        let product = self.iter().product();
        Arrays::new((self, vec![0.0; product]))
    }
}

/// Implementation to construct `Array` structs by using `Vec<usize>` as the dimensions, and `Vec<Float>`
/// as the values.
///
/// # Examples
///
/// ```
/// # extern crate corgi;
/// # use corgi::array::*;
/// # fn main () {
/// let a = Arrays::new((vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
/// assert_eq!(a[vec![1, 2]], 6.0);
/// # }
/// ```
impl Arrays for (Vec<usize>, Vec<Float>) {
    fn new(self) -> Array {
        let (dimensions, values) = self;
        Arrays::new((Arc::new(dimensions), Arc::new(values)))
    }
}

/// Implementation to construct `Array` structs by using `Arc<Vec<usize>>` as the dimensions, and `Arc<Vec<Float>>`
/// as the values.
///
/// # Examples
///
/// ```
/// # extern crate corgi;
/// # use std::sync::Arc;
/// # use corgi::array::*;
/// # fn main () {
/// let a = Arrays::new((Arc::new(vec![2, 3]), Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
/// assert_eq!(a[vec![1, 2]], 6.0);
/// # }
/// ```
impl Arrays for (Arc<Vec<usize>>, Arc<Vec<Float>>) {
    fn new(self) -> Array {
        let (dimensions, values) = self;
        Array {
            dimensions,
            values,
            children: Arc::new(Mutex::new(Vec::new())),
            consumer_count: Arc::new(AtomicUsize::new(0)),
            backward_op: None,
            tx: Arc::new(Mutex::new(None)),
            untracked: false,
            gradient: Arc::new(Mutex::new(None)),
        }
    }
}

/// The forward operation computes an operation with respect to inputs.
pub type ForwardOp = Arc<dyn Fn(&[&Array]) -> Array + Send + Sync>;
/// The backward operation computes deltas with respect to inputs.
pub type BackwardOp = Arc<dyn Fn(&mut Vec<Array>, &mut Array) -> Vec<Option<Array>> + Send + Sync>;

// TODO add flag to not store gradient
/// An n-dimensional differentiable array. Stored in row-major order.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate corgi;
///
/// use corgi::array::*;
///
/// # fn main() {
/// let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];
/// let b = arr![arr![3.0, 2.0, 1.0], arr![6.0, 5.0, 4.0]];
///
/// let mut p = &a * &b;
/// p.backward(None);
/// # }
/// ```
pub struct Array {
    dimensions: Arc<Vec<usize>>,
    values: Arc<Vec<Float>>,
    children: Arc<Mutex<Vec<Array>>>,
    consumer_count: Arc<AtomicUsize>,
    backward_op: Option<BackwardOp>,
    tx: Arc<Mutex<Option<Sender<Array>>>>,
    untracked: bool,
    gradient: Arc<Mutex<Option<Array>>>,
}

// TODO look into `arr![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];`
/// Creates an `Array`, which is row-major, with either:
/// * Contained arrays:
/// ```
/// # #[macro_use]
/// # extern crate corgi;
/// # use corgi::array::*;
/// # fn main() {
/// let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];
/// assert_eq!(a[vec![1, 2]], 6.0);
/// # }
/// ```
/// * Contained `Vec<Float>`
/// ```
/// # #[macro_use]
/// # extern crate corgi;
/// # use corgi::array::*;
/// # fn main() {
/// let a = arr![1.0, 2.0, 3.0];
/// assert_eq!(a[vec![1]], 2.0);
/// # }
/// ```
#[macro_export]
macro_rules! arr {
    ( $( $x:expr ),* ) => {
        {
            let mut values = Vec::new();

            $(
                values.push($x);
            )*

            Arrays::new(values)
        }
    };
}

// TODO error handling
impl Array {
    /// Returns a copy of the dimensions of the array.
    pub fn dimensions(&self) -> Vec<usize> {
        self.dimensions.to_vec()
    }

    /// Returns a read-only reference to the values of the array in row-major order.
    pub fn values(&self) -> &Vec<Float> {
        &*self.values
    }

    /// Returns a copy of the gradient of the array.
    ///
    /// # Panics
    ///
    /// Panics if no gradient exists.
    pub fn gradient(&self) -> Array {
        self.gradient.lock().unwrap().clone().unwrap()
    }

    /// Returns a guard for the gradient option of the array.
    ///
    /// # Panics
    ///
    /// Panics if unable to obtain a lock on the Mutex.
    pub fn gradient_mut(&mut self) -> MutexGuard<Option<Array>> {
        self.gradient.lock().unwrap()
    }

    /// Prevents tracking of operations for the backward pass, meaning the backward pass will skip computation of
    /// gradients for the current Array, and any children Arrays.
    ///
    /// This does not persist through threads, meaning calls in any `backward_op` will only temporarily set the
    /// `Array` to untracked. This is recommended, unless higher order derivatives are needed.
    ///
    /// Note that there currently is no way to re-enable tracking, apart from cloning an array.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate corgi;
    /// # use corgi::array::*;
    /// # fn main () {
    /// let mut a = arr![1.0, 2.0, 3.0];
    /// let b = arr![3.0, 2.0, 1.0];
    /// // only the gradients for `b`, and `c` will be computed, and stored
    /// let mut c = a.untracked() * &b;
    /// c.backward(None);
    /// assert_eq!(c.gradient(), arr![1.0, 1.0, 1.0]);
    /// assert_eq!(b.gradient(), arr![1.0, 2.0, 3.0]);
    /// # }
    /// ```
    pub fn untracked(&mut self) -> &Array {
        self.untracked = true;
        self
    }

    /// Adds `Vec<Array>` as the children of a vector.
    fn with_children(mut self, children: Vec<Array>) -> Array {
        self.children = Arc::new(Mutex::new(children));
        self
    }

    /// Sets the backward operation of the array for the backward pass.
    fn with_backward_op(mut self, backward_op: Option<BackwardOp>) -> Array {
        self.backward_op = backward_op;
        self
    }

    /// Performs a matrix multiplication on two matrices, storing the result in `values`.
    ///
    /// # Arguments
    ///
    /// `values` - The result values slice.
    /// `matmul_dimensions` - The dimensions to compute from: `(output_rows, output_cols, sum_len)`.
    /// `a` - The LHS matrix, and whether to transpose it: `(a, a_transpose)`.
    /// `b` - The RHS matrix, and whether to transpose it: `(b, b_transpose)`.
    #[cfg(not(feature = "blas"))]
    fn matmul_flat(
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
                    // TODO cleanup
                    sum += a[if a_transpose {
                        k * output_rows + r
                    } else {
                        r * sum_len + k
                    }] * b[if b_transpose {
                        j * sum_len + k
                    } else {
                        k * output_cols + j
                    }];
                }

                values[r * output_cols + j] = sum;
            }
        }
    }

    /// Performs matrix multiplications on two arrays, for each matching dimension not multiplied.
    ///
    /// # Arguments
    ///
    /// `a` - The LHS matrix, and whether to transpose it: `(a, a_transpose)`.
    /// `b` - The RHS matrix, and whether to transpose it: `(b, b_transpose)`.
    fn matmul_values(a: (&Array, bool), b: (&Array, bool)) -> Array {
        let (a, a_transpose) = a;
        let (b, b_transpose) = b;

        // TODO broadcasting
        // TODO OpenCL
        if a.dimensions.len() != b.dimensions.len() {
            panic!(
                "error: the dimensions {:?}, and {:?} are not compatible",
                a.dimensions, b.dimensions
            );
        }

        let mut indices = vec![0; a.dimensions.len().saturating_sub(2)];

        // TODO clean up
        let output_rows = if a.dimensions.len() < 2 {
            1
        } else {
            a.dimensions[a.dimensions.len() - if a_transpose { 1 } else { 2 }]
        };
        let output_cols = if b.dimensions.len() < 2 {
            1
        } else {
            b.dimensions[b.dimensions.len() - if b_transpose { 2 } else { 1 }]
        };
        let sum_len = if a.dimensions.len() < 2 && a_transpose {
            1
        } else {
            a.dimensions[a.dimensions.len() - if a_transpose { 2 } else { 1 }]
        };

        let output_dimensions: Vec<usize> = a
            .dimensions
            .iter()
            .copied()
            .take(indices.len())
            .chain(if a.dimensions.len() < 2 {
                vec![output_cols]
            } else {
                vec![output_rows, output_cols]
            })
            .collect();

        let output_length = output_dimensions.iter().product();
        let mut output_values = vec![0.0; output_length];

        let product = a.dimensions.iter().rev().skip(2).product();
        for _ in 0..product {
            let offset = flatten_indices(
                indices
                    .iter()
                    .copied()
                    .chain(vec![0; a.dimensions.len() - indices.len()])
                    .collect(),
                &a.dimensions,
            );
            let output_offset = flatten_indices(
                indices
                    .iter()
                    .copied()
                    .chain(vec![0; output_dimensions.len() - indices.len()])
                    .collect(),
                &output_dimensions,
            );

            let output_slice =
                &mut output_values[output_offset..output_offset + output_rows * output_cols];
            let a_slice = &a.values()[offset..offset + sum_len * output_rows];
            let b_slice = &b.values()[offset..offset + sum_len * output_cols];

            #[cfg(feature = "blas")]
            matmul_blas(
                output_slice,
                (output_rows, output_cols, sum_len),
                (a_slice, a_transpose),
                (b_slice, b_transpose),
            );
            #[cfg(not(feature = "blas"))]
            Array::matmul_flat(
                output_slice,
                (output_rows, output_cols, sum_len),
                (a_slice, a_transpose),
                (b_slice, b_transpose),
            );

            for j in 0..indices.len() {
                let current = indices.len() - j - 1;
                if indices[current] == a.dimensions[current] - 1 {
                    indices[current] = 0;
                } else {
                    indices[current] += 1;
                    break;
                }
            }
        }

        let result = Arrays::new((output_dimensions, output_values));

        if a.untracked && b.untracked {
            result
        } else {
            let backward_a = Box::new(move |c: &mut Vec<Array>, x: &mut Array| {
                if a_transpose {
                    Array::matmul_values((c[1].untracked(), b_transpose), (x.untracked(), true))
                } else {
                    Array::matmul_values((x.untracked(), false), (c[1].untracked(), !b_transpose))
                }
            });

            let backward_b = Box::new(move |c: &mut Vec<Array>, x: &mut Array| {
                if b_transpose {
                    Array::matmul_values((x.untracked(), true), (c[0].untracked(), a_transpose))
                } else {
                    Array::matmul_values((c[0].untracked(), !a_transpose), (x.untracked(), false))
                }
            });

            let backward_op: BackwardOp = if a.untracked {
                Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
                    vec![None, Some(backward_b(c, x))]
                })
            } else if b.untracked {
                Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
                    vec![Some(backward_a(c, x)), None]
                })
            } else {
                Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
                    vec![Some(backward_a(c, x)), Some(backward_b(c, x))]
                })
            };

            result
                .with_children(vec![a.clone(), b.clone()])
                .with_backward_op(Some(backward_op))
        }
    }

    /// Performs matrix multiplications on two arrays, for each matching dimension not multiplied.
    ///
    /// # Arguments
    ///
    /// * `a` - The LHS matrix, and whether to transpose it: `(a, a_transpose)`.
    /// * `b` - The RHS matrix, and whether to transpose it: `(b, b_transpose)`.
    pub fn matmul(a: (&Array, bool), b: (&Array, bool)) -> Array {
        Array::matmul_values(a, b)
    }

    /// Performs an operation on arrays.
    ///
    /// # Arguments
    ///
    /// * `arrays` - The arrays to perform the operations on.
    /// * `op` - The `ForwardOp`, which takes in the arrays, and outputs another Array.
    /// * `backward_op` - The `BackwardOp`, which takes in the arrays, and the delta, and outputs a
    /// new delta, with respect to each input. It is recommended that any array operations here are
    /// untracked, unless interested in higher order derivatives.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate corgi;
    /// # use std::sync::Arc;
    /// # use corgi::numbers::*;
    /// # use corgi::array::*;
    /// # fn main () {
    /// let op: ForwardOp = Arc::new(|x: &[&Array]| {
    ///     Arrays::new((x[0].dimensions(), x[0].values().iter().zip(x[1].values()).map(|(x, y)| x * y).collect::<Vec<Float>>()))
    /// });
    ///
    /// let op_clone = Arc::clone(&op);
    /// let backward_op: BackwardOp = Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
    ///     vec![
    ///         Some(Array::op(&vec![c[1].untracked(), x.untracked()], Arc::clone(&op_clone), None)),
    ///         Some(Array::op(&vec![c[0].untracked(), x.untracked()], Arc::clone(&op_clone), None)),
    ///     ]
    /// });
    ///
    /// let a = arr![1.0, 2.0, 3.0];
    /// let b = arr![3.0, 2.0, 1.0];
    ///
    /// let mut product = Array::op(&vec![&a, &b], op, Some(backward_op));
    /// assert_eq!(product, arr![3.0, 4.0, 3.0]);
    /// product.backward(None);
    /// assert_eq!(product.gradient(), arr![1.0, 1.0, 1.0]);
    /// assert_eq!(b.gradient(), arr![1.0, 2.0, 3.0]);
    /// assert_eq!(a.gradient(), arr![3.0, 2.0, 1.0]);
    /// # }
    /// ```
    pub fn op(arrays: &[&Array], op: ForwardOp, backward_op: Option<BackwardOp>) -> Array {
        let result = op(arrays);
        result
            .with_children(arrays.iter().map(|v| (*v).clone()).collect())
            .with_backward_op(backward_op)
    }

    /// Raises the array to the specified exponent.
    pub fn powf(&self, exponent: Float) -> Array {
        let values = self
            .values
            .iter()
            .map(|x| x.powf(exponent))
            .collect::<Vec<Float>>();

        let result = Arrays::new((Arc::clone(&self.dimensions), Arc::new(values)));

        if self.untracked {
            result
        } else {
            let backward_op = Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
                vec![Some((c[0].untracked() * 2.0).untracked() * x.untracked())]
            });

            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }

    /// Performs the sigmoid operation on each value of the array.
    pub fn sigmoid(&self) -> Array {
        let values = Arc::new(
            self.values
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect::<Vec<Float>>(),
        );
        let cached = Arc::clone(&values);

        let result = Arrays::new((Arc::clone(&self.dimensions), values));

        if self.untracked {
            result
        } else {
            let backward_op = Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
                let values = mul_values(
                    &cached.iter().map(|v| v * (1.0 - v)).collect::<Vec<Float>>(),
                    &x.values,
                );
                vec![Some(Arrays::new((
                    Arc::clone(&c[0].dimensions),
                    Arc::new(values),
                )))]
            });

            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }

    /// Sums the values of the array.
    pub fn sum(&self) -> Float {
        self.values.iter().sum()
    }

    /// Prepares a graph for the backward pass by traversing the graph to update consumer counts.
    fn propagate_consumers(&mut self) {
        for child in &mut *self.children.lock().unwrap() {
            child.consumer_count.fetch_add(1, Ordering::SeqCst);
            child.propagate_consumers();
        }
    }

    /// Awaits for deltas from all consumers, then continues the backward pass.
    ///
    /// # Panics
    ///
    /// Panics if the current node has no consumers (is an end node).
    fn await_results(&mut self, rx: Receiver<Array>, delta: Array) {
        let mut consumer_count = self.consumer_count.load(Ordering::SeqCst);
        if consumer_count == 0 {
            self.backward(Some(delta));
            return;
        }

        let mut delta = Arc::try_unwrap(delta.values).unwrap_or_else(|x| (*x).clone());
        consumer_count -= 1;
        let sum = |acc: &mut Vec<Float>, x: &Vec<Float>| {
            acc.iter_mut().zip(x).for_each(|(s, x)| *s += *x);
        };

        while consumer_count > 0 {
            let received = rx.recv().unwrap();
            consumer_count -= 1;
            sum(&mut delta, &received.values);
        }

        *self.tx.lock().unwrap() = None;
        self.consumer_count.store(0, Ordering::SeqCst);

        let delta = Arrays::new((Arc::clone(&self.dimensions), Arc::new(delta)));
        self.backward(Some(delta));
    }

    /// Performs the backward pass, computing gradients for all descendants, and propagating consumer counts if requested.
    ///
    /// # Panics
    ///
    /// Panics if the current node has children, but is not a differentiable function (is not a leaf).
    pub fn backward(&mut self, delta: Option<Array>) {
        let mut delta = match delta {
            Some(x) => x,
            None => {
                self.propagate_consumers();
                Arrays::new((
                    Arc::clone(&self.dimensions),
                    Arc::new(vec![1.0; self.values.len()]),
                ))
            }
        };

        match &self.backward_op {
            Some(x) => {
                let mut children_guard = self.children.lock().unwrap();
                let delta = (*x)(&mut children_guard, &mut delta);
                let mut handles = Vec::new();
                // start a new thread which will wait on all consumers
                for (i, delta) in delta.into_iter().enumerate() {
                    if let Some(delta) = delta {
                        let mut tx_guard = children_guard[i].tx.lock().unwrap();
                        match &*tx_guard {
                            Some(x) => {
                                x.send(delta).unwrap();
                            }
                            None => {
                                let mut child = children_guard[i].clone();

                                let (tx, rx) = channel();
                                *tx_guard = Some(tx);
                                handles.push(thread::spawn(move || {
                                    child.await_results(rx, delta);
                                }));
                            }
                        }
                    }
                }

                // wait for all threads to finish
                for handle in handles {
                    handle
                        .join()
                        .expect("error: could not join backward pass threads");
                }
            }
            None => {
                if self.children.lock().unwrap().len() != 0 {
                    panic!("error: operation is not differentiable")
                }
            }
        }

        let mut gradient_guard = self.gradient.lock().unwrap();
        match &mut *gradient_guard {
            Some(x) => *gradient_guard = Some(x.untracked() + delta.untracked()),
            None => *gradient_guard = Some(delta),
        }
    }
}

impl Clone for Array {
    fn clone(&self) -> Array {
        let backward_op = match &self.backward_op {
            Some(x) => Some(Arc::clone(&x)),
            None => None,
        };

        Array {
            dimensions: Arc::clone(&self.dimensions),
            values: Arc::clone(&self.values),
            children: Arc::clone(&self.children),
            consumer_count: Arc::clone(&self.consumer_count),
            backward_op,
            tx: Arc::clone(&self.tx),
            untracked: false,
            gradient: self.gradient.clone(),
        }
    }
}

impl PartialEq for Array {
    fn eq(&self, other: &Array) -> bool {
        *self.dimensions == *other.dimensions && *self.values == *other.values
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array")
            .field("dimensions", &*self.dimensions)
            .field("values", &*self.values)
            .finish()
    }
}

impl Index<usize> for Array {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.values.len() {
            panic!("error: invalid index supplied");
        }

        &self.values[index]
    }
}

impl Index<Vec<usize>> for Array {
    type Output = Float;

    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        &self.values[flatten_indices(indices, &*self.dimensions)]
    }
}

/// Converts indices by dimension to a single flattened index.
fn flatten_indices(indices: Vec<usize>, dimensions: &[usize]) -> usize {
    let is_indices_valid = indices.len() == dimensions.len()
        && indices
            .iter()
            .zip(dimensions)
            .filter(|&(i, d)| *i >= *d)
            .peekable()
            .peek()
            .is_none();

    if !is_indices_valid {
        panic!("error: invalid indices supplied")
    }

    let mut iter = indices.iter();
    let first = iter.next().unwrap();

    // dimensions will always have at least one element
    iter.zip(dimensions.iter().skip(1))
        .fold(*first, |acc, (i, d)| acc * d + i)
}

fn add_values(a: &[Float], b: &[Float]) -> Vec<Float> {
    a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<Float>>()
}

fn scale_values(a: &[Float], s: Float) -> Vec<Float> {
    a.iter().map(|x| x * s).collect::<Vec<Float>>()
}

fn mul_values(a: &[Float], b: &[Float]) -> Vec<Float> {
    a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<Float>>()
}

impl<'a, 'b> ops::Add<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn add(self, other: &Array) -> Self::Output {
        // TODO broadcasting, checking for valid dimensions
        let result = Arrays::new((
            Arc::clone(&self.dimensions),
            Arc::new(add_values(&self.values, &other.values)),
        ));

        if self.untracked && other.untracked {
            result
        } else {
            let backward_op: BackwardOp = if self.untracked {
                Arc::new(move |_: &mut Vec<Array>, x: &mut Array| {
                    vec![
                        None,
                        Some(Arrays::new((
                            Arc::clone(&x.dimensions),
                            Arc::clone(&x.values),
                        ))),
                    ]
                })
            } else if other.untracked {
                Arc::new(move |_: &mut Vec<Array>, x: &mut Array| {
                    vec![
                        Some(Arrays::new((
                            Arc::clone(&x.dimensions),
                            Arc::clone(&x.values),
                        ))),
                        None,
                    ]
                })
            } else {
                Arc::new(move |_: &mut Vec<Array>, x: &mut Array| {
                    vec![
                        Some(Arrays::new((
                            Arc::clone(&x.dimensions),
                            Arc::clone(&x.values)
                        )));
                        2
                    ]
                })
            };

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
        let result = Arrays::new((
            Arc::clone(&self.dimensions),
            Arc::new(scale_values(&self.values, -1.0)),
        ));
        if self.untracked {
            result
        } else {
            let backward_op =
                Arc::new(move |_: &mut Vec<Array>, x: &mut Array| vec![Some(-x.untracked())]);
            result
                .with_children(vec![self.clone()])
                .with_backward_op(Some(backward_op))
        }
    }
}

impl<'a> ops::Mul<Float> for &'a Array {
    type Output = Array;

    #[inline]
    fn mul(self, other: Float) -> Self::Output {
        let result = Arrays::new((
            Arc::clone(&self.dimensions),
            Arc::new(scale_values(&self.values, other)),
        ));

        if self.untracked {
            result
        } else {
            let backward_op = Arc::new(move |_: &mut Vec<Array>, x: &mut Array| {
                vec![Some(x.untracked() * other)]
            });
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
        // TODO broadcasting, checking for valid dimensions
        let result = Arrays::new((
            Arc::clone(&self.dimensions),
            Arc::new(mul_values(&self.values, &other.values)),
        ));

        if self.untracked && other.untracked {
            result
        } else {
            let backward_op: BackwardOp = if self.untracked {
                Arc::new(|c: &mut Vec<Array>, x: &mut Array| {
                    vec![None, Some(c[0].untracked() * x.untracked())]
                })
            } else if other.untracked {
                Arc::new(|c: &mut Vec<Array>, x: &mut Array| {
                    vec![Some(c[1].untracked() * x.untracked()), None]
                })
            } else {
                Arc::new(|c: &mut Vec<Array>, x: &mut Array| {
                    vec![
                        Some(c[1].untracked() * x.untracked()),
                        Some(c[0].untracked() * x.untracked()),
                    ]
                })
            };

            result
                .with_children(vec![self.clone(), other.clone()])
                .with_backward_op(Some(backward_op))
        }
    }
}

// TODO test f32
// TODO test calling backward, doing more computation, then calling backward again
// TODO test with multiple calls to backward
// TODO implement higher-order derivatives
// TODO test with higher-order derivatives
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let matrix = arr![
            arr![arr![0.0], arr![1.0]],
            arr![arr![2.0], arr![3.0]],
            arr![arr![4.0], arr![5.0]]
        ];

        assert_eq!(*matrix.dimensions, vec![3, 2, 1]);
        assert_eq!(
            *matrix.values,
            (0..6).map(|x| x as Float).collect::<Vec<Float>>()
        );
    }

    #[test]
    fn test_zeros() {
        let matrix = Arrays::new(vec![3, 2, 3]);
        assert_eq!(*matrix.dimensions, vec![3, 2, 3]);
        assert_eq!(
            *matrix.values,
            (0..18).map(|_| 0 as Float).collect::<Vec<Float>>()
        );
    }

    #[test]
    fn test_new_clone() {
        let a = arr![1.0, 2.0, 3.0];
        let b = arr![a.clone()];
        let c = arr![a.clone(), a.clone()];

        assert_eq!(b, arr![arr![1.0, 2.0, 3.0]]);
        assert_eq!(c, arr![arr![1.0, 2.0, 3.0], arr![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_ne() {
        let a = arr![1.0, 2.0, 3.0];
        let b = arr![2.0, 2.0, 3.0];

        assert_ne!(a, b);

        let c = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];
        let d = arr![arr![1.0, 2.0], arr![3.0, 4.0], arr![5.0, 6.0]];

        assert_ne!(c, d);
    }

    #[test]
    #[should_panic]
    fn test_invalid_dimensions() {
        arr![arr![arr![0.0], arr![1.0]], arr![arr![2.0, 3.0], arr![4.0]]];
    }

    #[test]
    fn test_access() {
        let matrix = arr![
            arr![arr![0.0, 1.0, 2.0], arr![3.0, 4.0, 5.0]],
            arr![arr![6.0, 7.0, 8.0], arr![9.0, 10.0, 11.0]],
            arr![arr![12.0, 13.0, 14.0], arr![15.0, 16.0, 17.0]]
        ];

        assert_eq!(matrix[vec![1, 1, 2]], 11.0);
    }

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
    fn test_matmul() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];

        let b = arr![arr![5.0, 3.0], arr![2.0, 6.0], arr![1.0, 2.0]];

        let matmul_expect = arr![arr![12.0, 21.0], arr![36.0, 54.0]];

        let mut result = Array::matmul((&a, false), (&b, false));
        assert_eq!(result, matmul_expect);

        result.backward(None);
        assert_eq!(result.gradient(), arr![arr![1.0, 1.0], arr![1.0, 1.0]]);
        assert_eq!(
            b.gradient(),
            arr![arr![5.0, 5.0], arr![7.0, 7.0], arr![9.0, 9.0]]
        );
        assert_eq!(a.gradient(), arr![arr![8.0, 8.0, 3.0], arr![8.0, 8.0, 3.0]]);
    }

    #[test]
    fn test_matmul_transpose() {
        let a = arr![arr![1.0, 4.0], arr![2.0, 5.0], arr![3.0, 6.0]];

        let b = arr![arr![5.0, 3.0], arr![2.0, 6.0], arr![1.0, 2.0]];

        let matmul_expect = arr![arr![12.0, 21.0], arr![36.0, 54.0]];

        let mut result = Array::matmul((&a, true), (&b, false));
        assert_eq!(result, matmul_expect);

        result.backward(None);
        assert_eq!(result.gradient(), arr![arr![1.0, 1.0], arr![1.0, 1.0]]);
        assert_eq!(
            b.gradient(),
            arr![arr![5.0, 5.0], arr![7.0, 7.0], arr![9.0, 9.0]]
        );
        assert_eq!(
            a.gradient(),
            arr![arr![8.0, 8.0], arr![8.0, 8.0], arr![3.0, 3.0]]
        );
    }

    #[test]
    fn test_matmul_vec() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];
        let b = arr![arr![1.0, 2.0, 3.0]];
        let c = arr![arr![1.0], arr![2.0], arr![3.0]];

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
    fn test_propagate() {
        let a = arr![5.0];
        let b = arr![2.0];

        let product = &a * &b;
        let mut sum = &product + &a;

        sum.propagate_consumers();
        assert_eq!(product.consumer_count.load(Ordering::Relaxed), 1);
        assert_eq!(b.consumer_count.load(Ordering::Relaxed), 1);
        assert_eq!(a.consumer_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_propagate_continue() {
        let a = arr![1.0, 2.0];
        let mut b = arr![5.0, 6.0];

        b = &b * &a;
        b.propagate_consumers();
        assert_eq!(a.consumer_count.load(Ordering::Relaxed), 1);
        a.consumer_count.store(0, Ordering::Relaxed);
        b.consumer_count.store(0, Ordering::Relaxed);

        b = &b * &a;
        b.propagate_consumers();
        assert_eq!(a.consumer_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_backward_op() {
        let a = arr![5.0];
        let b = arr![2.0];

        let product = &a * &b;
        let result =
            (*product.backward_op.unwrap())(&mut vec![a.clone(), b.clone()], &mut arr![1.0]);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result
                .iter()
                .map(|x| x.clone().unwrap())
                .collect::<Vec<Array>>(),
            vec![arr![2.0], arr![5.0]]
        );
        assert!(!a.untracked);
        assert!(!b.untracked);
        assert!(!product.untracked);
    }

    #[test]
    fn test_backward_untracked() {
        let mut a = arr![5.0];
        let b = arr![2.0];

        let mut product = a.untracked() * &b;
        product.backward(None);

        assert_eq!(product.gradient(), arr![1.0]);
        assert_eq!(b.gradient(), arr![5.0]);
    }

    #[test]
    fn test_backward_untracked_both() {
        let mut a = arr![5.0];
        let mut b = arr![2.0];

        let mut product = a.untracked() * b.untracked();
        product.backward(None);
        assert_eq!(product.gradient(), arr![1.0]);
        assert_eq!(*b.gradient.lock().unwrap(), None);
        assert_eq!(*a.gradient.lock().unwrap(), None);
    }

    #[test]
    fn test_backward_neg() {
        let a = arr![1.0, 2.0, 3.0];
        let b = arr![7.0, 8.0, 9.0];

        let mut product = &(-&a) * &b;
        assert_eq!(product, arr![-7.0, -16.0, -27.0]);
        product.backward(None);
        assert_eq!(product.gradient(), arr![1.0, 1.0, 1.0]);
        assert_eq!(b.gradient(), arr![-1.0, -2.0, -3.0]);
        assert_eq!(a.gradient(), arr![-7.0, -8.0, -9.0]);
    }

    #[test]
    fn test_backward_sub() {
        let a = arr![1.0];
        let b = arr![3.0];

        let mut result = &a - &b;
        assert_eq!(result, arr![-2.0]);
        result.backward(None);
        assert_eq!(result.gradient(), arr![1.0]);
        assert_eq!(b.gradient(), arr![-1.0]);
        assert_eq!(a.gradient(), arr![1.0]);
    }

    #[test]
    fn test_backward_powf() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];
        let b = arr![arr![3.0, 2.0, 1.0], arr![6.0, 5.0, 4.0]];
        let c = a.powf(2.0);

        let mut result = &c * &b;
        assert_eq!(result, arr![arr![3.0, 8.0, 9.0], arr![96.0, 125.0, 144.0]]);
        result.backward(None);
        assert_eq!(
            result.gradient(),
            arr![arr![1.0, 1.0, 1.0], arr![1.0, 1.0, 1.0]]
        );
        assert_eq!(c.gradient(), arr![arr![3.0, 2.0, 1.0], arr![6.0, 5.0, 4.0]]);
        assert_eq!(
            b.gradient(),
            arr![arr![1.0, 4.0, 9.0], arr![16.0, 25.0, 36.0]]
        );
        assert_eq!(
            a.gradient(),
            arr![arr![6.0, 8.0, 6.0], arr![48.0, 50.0, 48.0]]
        );
    }

    #[test]
    fn test_backward_sigmoid() {
        let a = arr![arr![(3.0 as Float).ln()]];
        let b = arr![arr![5.0]];
        let c = a.sigmoid();

        let mut result = &c * &b;
        assert_eq!(result, arr![arr![3.75]]);
        result.backward(None);
        assert_eq!(result.gradient(), arr![arr![1.0]]);
        assert_eq!(c.gradient(), arr![arr![5.0]]);
        assert_eq!(b.gradient(), arr![arr![0.75]]);
        assert_eq!(a.gradient(), arr![arr![0.9375]]);
    }

    #[test]
    fn test_backward_matmul_vec() {
        let a = arr![arr![1.0, 2.0, 3.0]];
        let b = arr![arr![9.0, 8.0, 7.0]];

        let mut result = Array::matmul((&a, false), (&b, true));
        result.backward(None);
    }

    #[test]
    fn test_backward_matmul_vec_multi() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];

        let b = arr![arr![1.0], arr![2.0], arr![3.0]];

        let c = arr![arr![7.0], arr![8.0]];

        let mut result = &Array::matmul((&a, false), (&b, false)) + &c;
        result.backward(None);
        assert_eq!(result.gradient(), arr![arr![1.0], arr![1.0]]);
        assert_eq!(c.gradient(), arr![arr![1.0], arr![1.0]]);
        assert_eq!(b.gradient(), arr![arr![5.0], arr![7.0], arr![9.0]]);
        assert_eq!(a.gradient(), arr![arr![1.0, 2.0, 3.0], arr![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_backward_matmul_multi() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];

        let b = arr![arr![1.0], arr![2.0], arr![3.0]];

        let c = arr![arr![1.0, 2.0, 3.0]];

        let mut result = Array::matmul(
            (&Array::matmul((&a, false), (&b, false)), false),
            (&c, false),
        );
        result.backward(None);
        assert_eq!(
            result.gradient(),
            arr![arr![1.0, 1.0, 1.0], arr![1.0, 1.0, 1.0]]
        );
        assert_eq!(c.gradient(), arr![arr![46.0, 46.0, 46.0]]);
        assert_eq!(b.gradient(), arr![arr![30.0], arr![42.0], arr![54.0]]);
        assert_eq!(
            a.gradient(),
            arr![arr![6.0, 12.0, 18.0], arr![6.0, 12.0, 18.0]]
        );
    }

    #[test]
    fn test_backward_repeat() {
        let a = arr![arr![1.0, 2.0, 3.0]];
        for _ in 0..5 {
            let mut b = &a * &a;
            b.backward(None);
        }
    }

    #[test]
    fn test_backward_single() {
        let a = arr![5.0];
        let b = arr![2.0];

        let mut product = &a * &b;
        product.backward(None);
        assert_eq!(product.gradient(), arr![1.0]);
        assert_eq!(b.gradient(), arr![5.0]);
        assert_eq!(a.gradient(), arr![2.0]);
    }

    #[test]
    fn test_backward_control_flow() {
        let a = arr![5.0];
        let b = arr![2.0];
        let mut c = arr![0.0];

        for _ in 0..10 {
            c = &c + &(&a * &b);
            if c[0] > 50.0 {
                c = &c * &a;
            }
        }

        c.backward(None);
        assert_eq!(c, arr![195300.0]);
        assert_eq!(c.gradient(), arr![1.0]);
        assert_eq!(b.gradient(), arr![97650.0]);
        assert_eq!(a.gradient(), arr![232420.0]);
    }

    #[test]
    fn test_backward_delta() {
        let a = arr![5.0];
        let b = arr![2.0];

        let mut product = &a * &b;
        product.backward(Some(arr![5.0]));
        assert_eq!(product.gradient(), arr![5.0]);
        assert_eq!(b.gradient(), arr![25.0]);
        assert_eq!(a.gradient(), arr![10.0]);
    }

    #[test]
    fn test_backward_dimensions() {
        let a = arr![arr![5.0, 2.0], arr![3.0, 1.0]];
        let b = arr![arr![6.0, 3.0], arr![7.0, 8.0]];

        let mut product = &a * &b;
        product.backward(None);
        assert_eq!(product.gradient(), arr![arr![1.0, 1.0], arr![1.0, 1.0]]);
        assert_eq!(b.gradient(), arr![arr![5.0, 2.0], arr![3.0, 1.0]]);
        assert_eq!(a.gradient(), arr![arr![6.0, 3.0], arr![7.0, 8.0]]);
    }

    #[test]
    fn test_backward_multi() {
        let a = arr![5.0, 2.0];
        let b = arr![6.0, 3.0];
        let c = &a * &b;
        let d = &c + &a;
        let mut e = &a * &d;
        e.backward(None);

        assert_eq!(e.gradient(), arr![1.0, 1.0]);
        assert_eq!(d.gradient(), arr![5.0, 2.0]);
        assert_eq!(c.gradient(), arr![5.0, 2.0]);
        assert_eq!(b.gradient(), arr![25.0, 4.0]);
        assert_eq!(a.gradient(), arr![70.0, 16.0]);
    }

    #[test]
    fn test_backward_intermediate() {
        let a = arr![1.0, 2.0];
        let b = arr![5.0, 3.0];
        let c = &(&(&a * &b) + &a) * &b;
        let mut product = &c * &a;
        product.backward(None);

        assert_eq!(product.gradient(), arr![1.0, 1.0]);
        assert_eq!(c.gradient(), arr![1.0, 2.0]);
        assert_eq!(b.gradient(), arr![11.0, 28.0]);
        assert_eq!(a.gradient(), arr![60.0, 48.0]);
    }

    #[test]
    fn test_backward_reassign() {
        let a = arr![1.0, 2.0];
        let mut b = arr![5.0, 6.0];

        b = &b + &a;
        b = &b * &a;
        b.backward(None);

        assert_eq!(b.gradient(), arr![1.0, 1.0]);
        assert_eq!(a.gradient(), arr![7.0, 10.0]);
    }

    #[test]
    fn test_backward_continue() {
        // TODO race condition here? fails intermittently
        let a = arr![1.0, 2.0];
        let mut b = arr![5.0, 6.0];

        b = &b * &a;
        assert_eq!(b, arr![5.0, 12.0]);

        b.backward(None);
        assert_eq!(b.gradient(), arr![1.0, 1.0]);
        assert_eq!(a.gradient(), arr![5.0, 6.0]);

        b = &b * &a;
        assert_eq!(b, arr![5.0, 24.0]);

        b.backward(None);
        assert_eq!(b.gradient(), arr![1.0, 1.0]);
        assert_eq!(a.gradient(), arr![15.0, 30.0]);
    }
}
