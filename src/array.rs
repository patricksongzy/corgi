use crate::numbers::*;

use std::ops;
use std::ops::Index;

use std::fmt;
use std::mem;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::mpsc::channel;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;
use std::thread;

// TODO more ergonomic Array creation
/// Helper trait to construct `Array` structs.
pub trait Arrays {
    /// Constructs a new `Array`.
    fn new(self) -> Array;
}

/// Implementation to construct `Array` structs by flattening other contained `Array` structs.
impl Arrays for Vec<Array> {
    /// Constructs a new `Array`, by flattening the contained `Array` structs, and keeping their dimensions.
    fn new(self) -> Array {
        // check if any of the contained array dimensions mismatch
        let is_dimensions_valid = match self.split_first() {
            Some((first, elements)) => elements.iter().all(|item| *item.dimensions == *first.dimensions),
            None => true,
        };

        if !is_dimensions_valid {
            panic!("error: contained array dimensions must all be the same");
        }

        let mut dimensions = vec![self.len()];
        dimensions.append(&mut (*self.first().unwrap().dimensions).clone());

        // take ownership if possible, but clone otherwise
        let values = self.into_iter().map(|array| Arc::try_unwrap(array.values).unwrap_or_else(|x| (*x).clone()))
            .flatten().collect::<Vec<Float>>();

        Arrays::new((dimensions, values))
    }
}

/// Implementation to construct `Array` structs by using `Vec<Float>` as the values, and by keeping flat dimensions.
impl Arrays for Vec<Float> {
    fn new(self) -> Array {
        Arrays::new((vec![self.len()], self))
    }
}

/// Implementation to construct `Array` structs by using `Vec<usize>` as the dimensions, and filling values with zeros.
impl Arrays for Vec<usize> {
    fn new(self) -> Array {
        let product = self.iter().fold(1, |acc, x| acc * x);
        Arrays::new((self, vec![0.0; product]))
    }
}

/// Implementation to construct `Array` structs by using `Vec<usize>` as the dimensions, and `Vec<Float>`
/// as the values.
impl Arrays for (Vec<usize>, Vec<Float>) {
    fn new(self) -> Array {
        let (dimensions, values) = self;
        Arrays::new((Arc::new(dimensions), Arc::new(values)))
    }
}

/// Implementation to construct `Array` structs by using `Arc<Vec<usize>>` as the dimensions, and `Arc<Vec<Float>>`
/// as the values.
impl Arrays for (Arc<Vec<usize>>, Arc<Vec<Float>>) {
    fn new(self) -> Array {
        let (dimensions, values) = self;
        Array {
            dimensions,
            values: values,
            children: Arc::new(Mutex::new(Vec::new())),
            consumer_count: Arc::new(Mutex::new(0)),
            backward_op: None,
            tx: Arc::new(Mutex::new(None)),
            untracked: false,
            gradient: Arc::new(Mutex::new(None)),
        }
    }
}

// TODO add flag to not store gradient
// TODO add poisoned flag, if Array has been modified
/// An n-dimensional differentiable Array.
///
/// # Examples
/// ```
/// #[macro_use]
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
    consumer_count: Arc<Mutex<usize>>,
    backward_op: Option<Arc<dyn Fn(&mut Vec<Array>, &mut Array) -> Vec<Array> + Send + Sync>>,
    tx: Arc<Mutex<Option<Sender<Array>>>>,
    untracked: bool,
    gradient: Arc<Mutex<Option<Array>>>,
}
 
// TODO look into `arr![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];`
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
    /// Returns a copy of the gradient.
    ///
    /// # Panics
    ///
    /// Panics if no gradient exists.
    pub fn gradient(&self) -> Array {
        self.gradient.lock().unwrap().clone().unwrap()
    }

    /// Returns a guard for the gradient option.
    ///
    /// # Panics
    ///
    /// Panics if unable to obtain a lock on the Mutex.
    pub fn gradient_mut(&mut self) -> MutexGuard<Option<Array>> {
        self.gradient.lock().unwrap()
    }

    /// Prevents tracking of operations for the backward pass.
    pub fn untracked(&mut self) -> &Array {
        self.untracked = true;
        self
    }

    /// Tracks operations for the backward pass.
    pub fn tracked(&mut self) -> &Array {
        self.untracked = false;
        self
    }

    /// Adds `Vec<Array>` as the children of a vector.
    fn with_children(mut self, children: Vec<Array>) -> Array {
        self.children = Arc::new(Mutex::new(children));
        self
    }

    /// Sets the backward operation of the array for the backward pass.
    fn with_backward_op(mut self, backward_op: Option<Arc<dyn Fn(&mut Vec<Array>, &mut Array) -> Vec<Array> + Send + Sync>>) -> Array {
        self.backward_op = backward_op;
        self
    }

    /// Performs a matrix multiplication on two arrays.
    fn matmul_flat(values: &mut Vec<Float>, output_rows: usize, output_cols: usize, sum_len: usize, offset: usize,
                   output_offset: usize, a: &Array, b: &Array, a_transpose: bool, b_transpose: bool) {
        for r in 0..output_rows {
            for j in 0..output_cols {
                let mut sum = 0.0;

                for k in 0..sum_len {
                    // TODO cleanup
                    sum += a[offset + if a_transpose { k * output_rows + r } else { r * sum_len + k }]
                        * b[offset + if b_transpose { j * sum_len + k } else { k * output_cols + j }];
                }

                values[output_offset + r * output_cols + j] = sum;
            }
        }
    }

    /// Performs matrix multiplications on two arrays, for each matching dimension not multiplied.
    fn matmul_values(a: &Array, b: &Array, a_transpose: bool, b_transpose: bool, has_backward: bool) -> Array {
        // TODO broadcasting
        // TODO use BLAS, and take slice of floats instead
        if a.dimensions.len() != b.dimensions.len() {
            panic!("error: the dimensions {:?}, and {:?} are not compatible", a.dimensions, b.dimensions);
        }

        let mut indices = vec![0; a.dimensions.len().checked_sub(2).unwrap_or(0)];

        // TODO clean up
        let output_rows = if a.dimensions.len() < 2 { 1 } else { a.dimensions[a.dimensions.len()
            - if a_transpose { 1 } else { 2 }] };
        let output_cols = if b.dimensions.len() < 2 { 1 } else { b.dimensions[b.dimensions.len()
            - if b_transpose { 2 } else { 1 }] };
        let sum_len = if a.dimensions.len() < 2 && a_transpose { 1 }
            else { a.dimensions[a.dimensions.len() - if a_transpose { 2 } else { 1 }]};

        let output_dimensions: Vec<usize> = a.dimensions.iter().copied().take(indices.len())
            .chain(if a.dimensions.len() < 2 { vec![output_cols] } else { vec![output_rows, output_cols] }).collect();

        let output_length = output_dimensions.iter().fold(1, |acc, x| acc * x);
        let mut output_values = vec![0.0; output_length];

        let product = a.dimensions.iter().rev().skip(2).fold(1, |acc, x| acc * x);
        for _ in 0..product {
            Array::matmul_flat(&mut output_values, output_rows, output_cols, sum_len,
                flatten_indices(indices.iter().copied().chain(vec![0; a.dimensions.len() - indices.len()]).collect(), &a.dimensions),
                flatten_indices(indices.iter().copied().chain(vec![0; output_dimensions.len() - indices.len()]).collect(), &output_dimensions),
                a, b, a_transpose, b_transpose);

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

        if has_backward {
            let backward_op = Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
                let delta_a = if a_transpose {
                    Array::matmul_values(&c[1], x, b_transpose, true, false)
                } else {
                    Array::matmul_values(x, &c[1], false, !b_transpose, false)
                };

                let delta_b = if b_transpose {
                    Array::matmul_values(x, &c[0], true, a_transpose, false)
                } else {
                    Array::matmul_values(&c[0], x, !a_transpose, false, false)
                };

                vec![delta_a, delta_b]
            });

            result.with_children(vec![a.clone(), b.clone()]).with_backward_op(Some(backward_op))
        } else {
            result
        }
    }

    /// Performs matrix multiplications on two arrays, for each matching dimension not multiplied.
    pub fn matmul(a: &Array, b: &Array, a_transpose: bool, b_transpose: bool) -> Array {
        Array::matmul_values(a, b, a_transpose, b_transpose, true)
    }

    /// Raises the array to the specified exponent.
    pub fn powf(&self, exponent: Float) -> Array {
        let values = self.values.iter().map(|x| x.powf(exponent)).collect::<Vec<Float>>();
        let backward_op = Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
            vec![(c[0].untracked() * 2.0).untracked() * x.untracked()]
        });
        
        let result = Arrays::new((Arc::clone(&self.dimensions), Arc::new(values)));

        if self.untracked { result } else { result.with_children(vec![self.clone()]).with_backward_op(Some(backward_op)) }
    }

    /// Performs the sigmoid operation on each value of the array.
    pub fn sigmoid(&self) -> Array {
        let values = Arc::new(self.values.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect::<Vec<Float>>());
        let cached = Arc::clone(&values);
        let backward_op = Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
            let values = mul_values(&cached.iter().map(|v| v * (1.0 - v)).collect::<Vec<Float>>(), &x.values);
            vec![Arrays::new((Arc::clone(&c[0].dimensions), Arc::new(values)))]
        });

        let result = Arrays::new((Arc::clone(&self.dimensions), values));

        if self.untracked { result } else { result.with_children(vec![self.clone()]).with_backward_op(Some(backward_op)) }
    }

    /// Sums the values of the array.
    pub fn sum(&self) -> Float {
        self.values.iter().sum()
    }

    /// Prepares a graph for the backward pass by traversing the graph to update consumer counts.
    fn propagate_consumers(&mut self) {
        for child in &mut *self.children.lock().unwrap() {
            *child.consumer_count.lock().unwrap() += 1;
            child.propagate_consumers();
        }
    }

    /// Awaits for deltas from all consumers, then continues the backward pass.
    /// 
    /// # Panics
    ///
    /// Panics if the current node has no consumers (is an end node).
    fn await_results(&mut self, rx: Receiver<Array>, delta: Array) {
        let mut consumer_count = self.consumer_count.lock().unwrap();

        if *consumer_count == 0 {
            panic!("error: cannot await results from end node");
        }

        let mut delta = Arc::try_unwrap(delta.values).unwrap_or_else(|x| (*x).clone());
        *consumer_count -= 1;
        let sum = |acc: &mut Vec<Float>, x: &Vec<Float>| {
            acc.iter_mut().zip(x).for_each(|(s, x)| *s += *x);
        };

        while *consumer_count > 0 {
            let received = rx.recv().unwrap();
            *consumer_count -= 1;
            sum(&mut delta, &received.values);
        }

        *self.tx.lock().unwrap() = None;
        mem::drop(consumer_count);

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
                Arrays::new((Arc::clone(&self.dimensions), Arc::new(vec![1.0; self.values.len()])))
            },
        };

        match &self.backward_op {
            Some(x) => {
                let mut children_guard = self.children.lock().unwrap();
                let delta = (*x)(&mut children_guard, &mut delta);
                let mut handles = Vec::new();
                // start a new thread which will wait on all consumers
                for (i, delta) in delta.into_iter().enumerate() {
                    let mut tx_guard = children_guard[i].tx.lock().unwrap();
                    match &*tx_guard {
                        Some(x) => {
                            x.send(delta).unwrap();
                        },
                        None => {
                            let mut child = children_guard[i].clone();

                            let (tx, rx) = channel();
                            *tx_guard = Some(tx);
                            handles.push(thread::spawn(move|| {
                                child.await_results(rx, delta);
                            }));
                        },
                    }
                }

                // wait for all threads to finish
                for handle in handles {
                    handle.join().unwrap();
                }
            },
            None => {
                if self.children.lock().unwrap().len() != 0 {
                    panic!("error: operation is not differentiable")
                }
            },
        }

        let mut gradient_guard = self.gradient.lock().unwrap();
        match &mut *gradient_guard {
            Some(x) => *gradient_guard = Some(x.untracked() + delta.untracked()),
            None => *gradient_guard = Some(delta)
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
        f.debug_struct("Array").field("dimensions", &*self.dimensions).field("values", &*self.values).finish()
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
fn flatten_indices(indices: Vec<usize>, dimensions: &Vec<usize>) -> usize {
    let is_indices_valid = indices.len() == dimensions.len()
        && !indices.iter().zip(dimensions).filter(|&(i, d)| *i >= *d).peekable().peek().is_some();

    if !is_indices_valid {
        panic!("error: invalid indices supplied")
    } 

    let mut iter = indices.iter();
    let first = iter.next().unwrap();

    // dimensions will always have at least one element
    iter.zip(dimensions.iter().skip(1)).fold(*first, |acc, (i, d)| acc * d + i)
}

fn add_values(a: &Vec<Float>, b: &Vec<Float>) -> Vec<Float> {
    a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<Float>>()
}

fn scale_values(a: &Vec<Float>, s: Float) -> Vec<Float> {
    a.iter().map(|x| x * s).collect::<Vec<Float>>()
}

fn mul_values(a: &Vec<Float>, b: &Vec<Float>) -> Vec<Float> {
    a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<Float>>()
}

impl<'a, 'b> ops::Add<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn add(self, other: &Array) -> Self::Output {
        // TODO broadcasting, checking for valid dimensions
        let result = Arrays::new((Arc::clone(&self.dimensions), Arc::new(add_values(&self.values, &other.values)))); 
        if self.untracked && other.untracked {
            result
        } else {
            let mut delta_count = 1;
            let children = if self.untracked {
                vec![other.clone()]
            } else if other.untracked {
                vec![self.clone()]
            } else {
                delta_count = 2;
                vec![self.clone(), other.clone()]
            };

            let backward_op = Arc::new(move |_: &mut Vec<Array>, x: &mut Array| vec![Arrays::new((Arc::clone(&x.dimensions),
                Arc::clone(&x.values))); delta_count]);
            result.with_children(children).with_backward_op(Some(backward_op))
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
        let result = Arrays::new((Arc::clone(&self.dimensions), Arc::new(scale_values(&self.values, -1.0))));
        if self.untracked {
            result
        } else {
            let backward_op = Arc::new(move |_: &mut Vec<Array>, x: &mut Array| vec![-x.untracked()]);
            result.with_children(vec![self.clone()]).with_backward_op(Some(backward_op))
        }
    }
}

impl<'a> ops::Mul<Float> for &'a Array {
    type Output = Array;

    #[inline]
    fn mul(self, other: Float) -> Self::Output {
        let result = Arrays::new((Arc::clone(&self.dimensions), Arc::new(scale_values(&self.values, other))));
        if self.untracked {
            result
        } else {
            let backward_op = Arc::new(move |_: &mut Vec<Array>, x: &mut Array| vec![x.untracked() * other]);
            result.with_children(vec![self.clone()]).with_backward_op(Some(backward_op))
        }
    }
}

impl<'a, 'b> ops::Mul<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn mul(self, other: &Array) -> Self::Output {
        // TODO broadcasting, checking for valid dimensions
        let result = Arrays::new((Arc::clone(&self.dimensions), Arc::new(mul_values(&self.values, &other.values))));

        if self.untracked && other.untracked {
            result
        } else {
            let mut delta_count = 1;
            let (children, backward_op): (Vec<Array>, Arc<dyn Fn(&mut Vec<Array>, &mut Array) -> Vec<Array> + Send + Sync>) = if self.untracked {
                // let backward_op = Arc::new(|c: &mut Vec<Array>, x: &mut Array| vec![Arrays::new((Arc::clone(&c[1].dimensions),
                //     Arc::new(mul_values(&c[0].values, &x.values))))]);
                let backward_op = Arc::new(|c: &mut Vec<Array>, x: &mut Array| vec![c[0].untracked() * x.untracked()]);
                (vec![other.clone()], backward_op)
            } else if other.untracked {
                // let backward_op = Arc::new(|c: &mut Vec<Array>, x: &mut Array| vec![Arrays::new((Arc::clone(&c[0].dimensions),
                //     Arc::new(mul_values(&c[1].values, &x.values))))]);
                let backward_op = Arc::new(|c: &mut Vec<Array>, x: &mut Array| vec![c[1].untracked() * x.untracked()]);
                (vec![self.clone()], backward_op)
            } else {
                delta_count = 2;
                // let backward_op = Arc::new(|c: &Vec<Array>, x: &Array| vec![Arrays::new((Arc::clone(&c[0].dimensions),
                //     Arc::new(mul_values(&c[1].values, &x.values)))), Arrays::new((Arc::clone(&c[1].dimensions),
                //     Arc::new(mul_values(&c[0].values, &x.values))))]);
                let backward_op = Arc::new(|c: &mut Vec<Array>, x: &mut Array| vec![c[1].untracked() * x.untracked(), c[0].untracked() * x.untracked()]);
                (vec![self.clone(), other.clone()], backward_op)
            };

            result.with_children(children).with_backward_op(Some(backward_op))

        }
    }
}

// TODO test with array modification before backward call (poisoned)
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
        let matrix = arr![arr![
            arr![0.0], arr![1.0]
        ],
        arr![
            arr![2.0], arr![3.0]
        ],
        arr![
            arr![4.0], arr![5.0]
        ]];
        
        assert_eq!(*matrix.dimensions, vec![3, 2, 1]);
        assert_eq!(*matrix.values, (0..6).map(|x| x as Float).collect::<Vec<Float>>());
    }

    #[test]
    fn test_zeros() {
        let matrix = Arrays::new(vec![3, 2, 3]);
        assert_eq!(*matrix.dimensions, vec![3, 2, 3]);
        assert_eq!(*matrix.values, (0..18).map(|_| 0 as Float).collect::<Vec<Float>>());
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
        arr![arr![ 
            arr![0.0], arr![1.0]
        ],
        arr![
            arr![2.0, 3.0], arr![4.0]
        ]];                              
    }
    
    #[test]
    fn test_access() {
        let matrix = arr![arr![
            arr![0.0, 1.0, 2.0], arr![3.0, 4.0, 5.0]
        ],
        arr![
            arr![6.0, 7.0, 8.0], arr![9.0, 10.0, 11.0]
        ],
        arr![
            arr![12.0, 13.0, 14.0], arr![15.0, 16.0, 17.0]
        ]];

        assert_eq!(matrix[vec![1, 1, 2]], 11.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = arr![arr![
            arr![0.0, 1.0], arr![2.0, 3.0]
        ],
        arr![
            arr![4.0, 5.0], arr![6.0, 7.0]
        ]];

        let b = arr![arr![
            arr![2.0, 4.0], arr![6.0, 8.0]
        ],
        arr![
            arr![10.0, 12.0], arr![14.0, 16.0]
        ]];

        let sum_expect = arr![arr![
            arr![2.0, 5.0], arr![8.0, 11.0]
        ],
        arr![
            arr![14.0, 17.0], arr![20.0, 23.0]
        ]];

        let product_expect = arr![arr![
            arr![0.0, 4.0], arr![12.0, 24.0]
        ],
        arr![
            arr![40.0, 60.0], arr![84.0, 112.0]
        ]];

        let sum = &a + &b;
        let product = &a * &b;

        assert_eq!(sum, sum_expect);
        assert_eq!(product, product_expect);
    }

    #[test]
    fn test_matmul() {
        let a = arr![
            arr![1.0, 2.0, 3.0],
            arr![4.0, 5.0, 6.0]
        ];

        let b = arr![
            arr![5.0, 3.0],
            arr![2.0, 6.0],
            arr![1.0, 2.0]
        ];
        
        let matmul_expect = arr![
            arr![12.0, 21.0],
            arr![36.0, 54.0]
        ];

        let mut result = Array::matmul(&a, &b, false, false);
        assert_eq!(result, matmul_expect);

        result.backward(None);
        assert_eq!(result.gradient(), arr![arr![1.0, 1.0], arr![1.0, 1.0]]);
        assert_eq!(b.gradient(), arr![arr![5.0, 5.0], arr![7.0, 7.0], arr![9.0, 9.0]]);
        assert_eq!(a.gradient(), arr![arr![8.0, 8.0, 3.0], arr![8.0, 8.0, 3.0]]);
    }

    #[test]
    fn test_matmul_transpose() {
        let a = arr![
            arr![1.0, 4.0],
            arr![2.0, 5.0],
            arr![3.0, 6.0]
        ];

        let b = arr![
            arr![5.0, 3.0],
            arr![2.0, 6.0],
            arr![1.0, 2.0]
        ];
        
        let matmul_expect = arr![
            arr![12.0, 21.0],
            arr![36.0, 54.0]
        ];

        let mut result = Array::matmul(&a, &b, true, false);
        assert_eq!(result, matmul_expect);

        result.backward(None);
        assert_eq!(result.gradient(), arr![arr![1.0, 1.0], arr![1.0, 1.0]]);
        assert_eq!(b.gradient(), arr![arr![5.0, 5.0], arr![7.0, 7.0], arr![9.0, 9.0]]);
        assert_eq!(a.gradient(), arr![arr![8.0, 8.0], arr![8.0, 8.0], arr![3.0, 3.0]]);
    }

    #[test]
    fn test_matmul_vec() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];
        let b = arr![arr![1.0, 2.0, 3.0]];
        let c = arr![arr![1.0], arr![2.0], arr![3.0]]; 

        let result = Array::matmul(&a, &b, false, true);
        assert_eq!(result, arr![arr![14.0], arr![32.0]]);

        let result = Array::matmul(&b, &a, false, true);
        assert_eq!(result, arr![arr![14.0, 32.0]]);

        let result = Array::matmul(&b, &c, false, false);
        assert_eq!(result, arr![arr![14.0]]);
    }

    #[test]
    fn test_matmul_single() {
        let a = arr![1.0, 2.0, 3.0];
        let b = arr![3.0, 2.0, 1.0];
        let result = Array::matmul(&a, &b, false, false);
        assert_eq!(result, arr![10.0]);
    }

    #[test]
    fn test_matmul_multi() {
        let a = arr![
            arr![1.0, 2.0, 3.0],
            arr![4.0, 5.0, 6.0]
        ];
        
        let b = arr![arr![1.0], arr![2.0], arr![3.0]];
        
        let c = arr![arr![1.0, 2.0, 3.0]];

        let result = Array::matmul(&Array::matmul(&a, &b, false, false), &c, false, false);
        assert_eq!(result, arr![arr![14.0, 28.0, 42.0], arr![32.0, 64.0, 96.0]]);
    }

    #[test]
    fn test_matmul_nd() {
        let a = arr![
            arr![arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]], arr![arr![6.0, 5.0, 4.0], arr![3.0, 2.0, 1.0]]],
            arr![arr![arr![9.0, 8.0, 7.0], arr![4.0, 5.0, 6.0]], arr![arr![6.0, 7.0, 8.0], arr![3.0, 2.0, 1.0]]]
        ];

        let b = arr![
            arr![arr![arr![5.0, 3.0], arr![2.0, 6.0], arr![1.0, 2.0]], arr![arr![3.0, 6.0], arr![2.0, 5.0], arr![1.0, 4.0]]],
            arr![arr![arr![5.0, 3.0], arr![2.0, 6.0], arr![8.0, 7.0]], arr![arr![8.0, 6.0], arr![5.0, 3.0], arr![4.0, 7.0]]]
        ];

        let matmul_expect = arr![
            arr![arr![arr![12.0, 21.0], arr![36.0, 54.0]], arr![arr![32.0, 77.0], arr![14.0, 32.0]]],
            arr![arr![arr![117.0, 124.0], arr![78.0, 84.0]], arr![arr![115.0, 113.0],arr![38.0, 31.0]]]
        ];

        let result = Array::matmul(&a, &b, false, false);
        assert_eq!(result, matmul_expect);
    }

    #[test]
    fn test_propagate_consumers() {
        let a = arr![5.0];
        let b = arr![2.0];

        let product = &a * &b;
        let mut sum = &product + &a;

        sum.propagate_consumers();
        assert_eq!(*product.consumer_count.lock().unwrap(), 1);
        assert_eq!(*b.consumer_count.lock().unwrap(), 1);
        assert_eq!(*a.consumer_count.lock().unwrap(), 2);
    }

    #[test]
    fn test_backward_op() {
        let a = arr![5.0];
        let b = arr![2.0];

        let product = &a * &b;
        let result = (*product.backward_op.unwrap())(&mut vec![a.clone(), b.clone()], &mut arr![1.0]);
        assert_eq!(result.len(), 2);
        assert_eq!(result, vec![arr![2.0], arr![5.0]]);
        assert!(!a.untracked);
        assert!(!b.untracked);
        assert!(!product.untracked);
    }

    #[test]
    fn test_backward_untracked() {
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
        assert_eq!(result.gradient(), arr![arr![1.0, 1.0, 1.0], arr![1.0, 1.0, 1.0]]);
        assert_eq!(c.gradient(), arr![arr![3.0, 2.0, 1.0], arr![6.0, 5.0, 4.0]]);
        assert_eq!(b.gradient(), arr![arr![1.0, 4.0, 9.0], arr![16.0, 25.0, 36.0]]);
        assert_eq!(a.gradient(), arr![arr![6.0, 8.0, 6.0], arr![48.0, 50.0, 48.0]]);
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
        
        let mut result = Array::matmul(&a, &b, false, true);
        result.backward(None);
    }

    #[test]
    fn test_backward_matmul_vec_multi() {
        let a = arr![
            arr![1.0, 2.0, 3.0],
            arr![4.0, 5.0, 6.0]
        ];

        let b = arr![arr![1.0], arr![2.0], arr![3.0]];

        let c = arr![arr![7.0], arr![8.0]];

        let mut result = &Array::matmul(&a, &b, false, false) + &c;
        result.backward(None);
        assert_eq!(result.gradient(), arr![arr![1.0], arr![1.0]]);
        assert_eq!(c.gradient(), arr![arr![1.0], arr![1.0]]);
        assert_eq!(b.gradient(), arr![arr![5.0], arr![7.0], arr![9.0]]);
        assert_eq!(a.gradient(), arr![arr![1.0, 2.0, 3.0], arr![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_backward_matmul_multi() {
        let a = arr![
            arr![1.0, 2.0, 3.0],
            arr![4.0, 5.0, 6.0]
        ];
        
        let b = arr![arr![1.0], arr![2.0], arr![3.0]];
        
        let c = arr![arr![1.0, 2.0, 3.0]];

        let mut result = Array::matmul(&Array::matmul(&a, &b, false, false), &c, false, false);
        result.backward(None);
        assert_eq!(result.gradient(), arr![arr![1.0, 1.0, 1.0], arr![1.0, 1.0, 1.0]]);
        assert_eq!(c.gradient(), arr![arr![46.0, 46.0, 46.0]]);
        assert_eq!(b.gradient(), arr![arr![30.0], arr![42.0], arr![54.0]]);
        assert_eq!(a.gradient(), arr![arr![6.0, 12.0, 18.0], arr![6.0, 12.0, 18.0]]);
    }

    #[test]
    fn test_backward_repeat() {
        let a = arr![arr![1.0, 2.0, 3.0]];
        // TODO implement test
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
        assert_eq!(*product.consumer_count.lock().unwrap(), 0);
        assert_eq!(*b.consumer_count.lock().unwrap(), 0);
        assert_eq!(*a.consumer_count.lock().unwrap(), 0);
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
    fn test_backward_dimensions() {
        let a = arr![arr![5.0, 2.0], arr![3.0, 1.0]];
        let b = arr![arr![6.0, 3.0], arr![7.0, 8.0]];
        let mut c = &a * &b;
        c.backward(None);

        assert_eq!(c.gradient(), arr![arr![1.0, 1.0], arr![1.0, 1.0]]);
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
    fn test_backward_poisoned() {
        // TODO modify array before backward is called
    }
}

