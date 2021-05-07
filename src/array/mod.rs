//! An n-dimensional array, with automatic differentation.
//! # Examples
//! See the README for more examples.
//! ```
//! # #[macro_use]
//! # extern crate corgi;
//! use corgi::array::*;
//!
//! # fn main() {
//! let a = arr![5.0].tracked();
//! let b = arr![2.0].tracked();
//! let mut c = arr![0.0].tracked();
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
//! assert_eq!(c.gradient().unwrap(), arr![1.0]);
//! assert_eq!(b.gradient().unwrap(), arr![97650.0]);
//! assert_eq!(a.gradient().unwrap(), arr![232420.0]);
//! # }
//! ```

mod arithmetic;
mod linalg;

use crate::numbers::*;

use approx::{AbsDiffEq, RelativeEq};

use std::convert::{From, Into};

use std::ops;
use std::ops::Index;

use std::fmt;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

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
/// let a = Array::from(vec![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]]);
/// assert_eq!(a[vec![1, 2]], 6.0);
/// # }
/// ```
impl From<Vec<Array>> for Array {
    fn from(contents: Vec<Array>) -> Self {
        // check if any of the contained array dimensions mismatch
        let is_dimensions_valid = match contents.split_first() {
            Some((first, elements)) => elements
                .iter()
                .all(|item| *item.dimensions == *first.dimensions),
            None => true,
        };

        if !is_dimensions_valid {
            panic!("error: contained array dimensions must all be the same");
        }

        let mut dimensions = vec![contents.len()];
        dimensions.append(&mut (*contents.first().unwrap().dimensions).clone());

        // take ownership if possible, but clone otherwise
        let values = contents
            .into_iter()
            .map(|array| Arc::try_unwrap(array.values).unwrap_or_else(|x| (*x).clone()))
            .flatten()
            .collect::<Vec<Float>>();

        Array::from((dimensions, values))
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
/// let a = Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// assert_eq!(a[vec![5]], 6.0);
/// # }
/// ```
impl From<Vec<Float>> for Array {
    fn from(values: Vec<Float>) -> Self {
        Array::from((vec![values.len()], values))
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
/// let a = Array::from(vec![3, 2, 3]);
/// assert_eq!(a[vec![2, 1, 1]], 0.0);
/// # }
/// ```
impl From<Vec<usize>> for Array {
    fn from(dimensions: Vec<usize>) -> Self {
        let product = dimensions.iter().product();
        Array::from((dimensions, vec![0.0; product]))
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
/// let a = Array::from((vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
/// assert_eq!(a[vec![1, 2]], 6.0);
/// # }
/// ```
impl From<(Vec<usize>, Vec<Float>)> for Array {
    fn from(items: (Vec<usize>, Vec<Float>)) -> Self {
        let (dimensions, values) = items;
        Array::from((Arc::new(dimensions), Arc::new(values)))
    }
}

impl From<(Arc<Vec<usize>>, Arc<Vec<Float>>)> for Array {
    fn from(items: (Arc<Vec<usize>>, Arc<Vec<Float>>)) -> Self {
        let (dimensions, values) = items;

        let is_dimensions_valid = dimensions.iter().all(|d| *d >= 1);
        if !is_dimensions_valid {
            panic!("error: invalid dimensions {:?}", dimensions);
        }

        let is_values_valid = dimensions.iter().product::<usize>() == values.len();
        if !is_values_valid {
            panic!("error: dimensions, and values must be of the same length");
        }

        Array {
            dimensions,
            values,
            children: Arc::new(Mutex::new(Vec::new())),
            consumer_count: Arc::new(AtomicUsize::new(0)),
            backward_op: None,
            tx: Arc::new(Mutex::new(None)),
            is_tracked: false,
            keep_gradient: false,
            gradient: Arc::new(Mutex::new(None)),
        }
    }
}

impl Into<Vec<Float>> for Array {
    fn into(self) -> Vec<Float> {
        Arc::try_unwrap(self.values).unwrap()
    }
}

/// The sliced operation computes an operation with respect to slices on a mutable output slice.
type SlicedOp = Box<dyn Fn(&mut [Float], Vec<&[Float]>)>;
/// The forward operation computes an operation with respect to inputs.
pub type ForwardOp = Arc<dyn Fn(&[&Array]) -> Array + Send + Sync>;
/// The backward operation computes deltas with respect to inputs.
pub type BackwardOp =
    Arc<dyn Fn(&mut Vec<Array>, &[bool], &Array) -> Vec<Option<Array>> + Send + Sync>;

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
    is_tracked: bool,
    keep_gradient: bool,
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

            Array::from(values)
        }
    };
}

/// Computes the element-wise dimensions to broadcast to.
fn element_wise_dimensions(x: &Arc<Vec<usize>>, y: &Arc<Vec<usize>>) -> Vec<usize> {
    let (mut longer, other) = if x.len() > y.len() {
        ((**x).clone(), y)
    } else {
        ((**y).clone(), x)
    };

    let is_dimensions_valid = longer
        .iter()
        .rev()
        .zip(other.iter().rev())
        .all(|(&l, &o)| l == o || l == 1 || o == 1);
    if !is_dimensions_valid {
        panic!(
            "error: multiplication dimensions, {:?}, and {:?} must be matching",
            *x, *y
        );
    }

    for (l, o) in longer.iter_mut().rev().zip(other.iter().rev()) {
        *l = std::cmp::max(*l, *o);
    }

    longer
}

// TODO better error handling
impl Array {
    /// Returns a copy of the dimensions of the array.
    pub fn dimensions(&self) -> Vec<usize> {
        self.dimensions.to_vec()
    }

    /// Returns an immutable reference to the values of the array in row-major order.
    pub fn values(&self) -> &Vec<Float> {
        &*self.values
    }

    /// Returns a copy of the gradient of the array.
    ///
    /// # Panics
    ///
    /// Panics if unable to obtain a lock on the Mutex.
    pub fn gradient(&self) -> Option<Array> {
        let gradient_option = self.gradient.lock().unwrap();
        match gradient_option.as_ref() {
            Some(x) => Some(x.clone()),
            None => None,
        }
    }

    /// Returns a guard for the gradient option of the array.
    ///
    /// # Panics
    ///
    /// Panics if unable to obtain a lock on the Mutex.
    pub fn gradient_mut(&mut self) -> MutexGuard<Option<Array>> {
        self.gradient.lock().unwrap()
    }

    /// Enables tracking of operations for the backward pass, meaning the backward pass will compute, and store
    /// gradients for the current array, and any children arrays which are tracked.
    ///
    /// An operation with any positive number of tracked children will always output a tracked array.
    ///
    /// This does not persist through threads, or through being set on a clone, meaning any tracked clones will not
    /// affect tracking of the array, apart from clones of the clone.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate corgi;
    /// # use corgi::array::*;
    /// # fn main () {
    /// // only the gradient for `b`, will be stored
    /// let mut a = arr![1.0, 2.0, 3.0].untracked();
    /// let b = arr![3.0, 2.0, 1.0].tracked();
    /// let mut c = &a * &b;
    /// c.backward(None);
    /// assert_eq!(b.gradient().unwrap(), arr![1.0, 2.0, 3.0]);
    /// # }
    /// ```
    pub fn tracked(mut self) -> Array {
        self.keep_gradient = true;
        self.is_tracked = true;
        self
    }

    /// Starts tracking operations for a mutable reference to an array.
    pub fn start_tracking(&mut self) {
        self.is_tracked = true;
    }

    /// Prevents tracking of operations for the backward pass, meaning the backward pass will skip computation of
    /// gradients for the current array, and any children arrays.
    ///
    /// Any operation with every child untracked will always output an untracked array, and will not store any
    /// subgraph information.
    ///
    /// This does not persist through threads, or through being set on a clone, meaning any tracked clones will not
    /// affect tracking of the array, apart from clones of the clone.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate corgi;
    /// # use corgi::array::*;
    /// # fn main () {
    /// // only the gradient for `b`, will be stored
    /// let mut a = arr![1.0, 2.0, 3.0].untracked();
    /// let b = arr![3.0, 2.0, 1.0].tracked();
    /// let mut c = &a * &b;
    /// c.backward(None);
    /// assert_eq!(b.gradient().unwrap(), arr![1.0, 2.0, 3.0]);
    /// # }
    /// ```
    pub fn untracked(mut self) -> Array {
        self.keep_gradient = false;
        self.is_tracked = false;
        self
    }

    /// Stops tracking operations for a mutable reference to an array. Useful for temporarily updating parameters
    /// without requiring their gradients.
    pub fn stop_tracking(&mut self) {
        self.is_tracked = false;
    }

    /// Adds `Vec<Array>` as the children of a vector.
    fn with_children(mut self, children: Vec<Array>) -> Array {
        self.children = Arc::new(Mutex::new(children));
        self.tracked()
    }

    /// Sets the backward operation of the array for the backward pass.
    fn with_backward_op(mut self, backward_op: Option<BackwardOp>) -> Array {
        self.backward_op = backward_op;
        self
    }

    /// Awaits for deltas from all consumers, then continues the backward pass.
    ///
    /// # Panics
    ///
    /// Panics if the current node has no consumers (is an end node).
    fn await_results(&mut self, rx: Receiver<Array>, delta: Array) {
        let mut consumer_count = self.consumer_count.load(Ordering::Relaxed);
        if consumer_count == 0 {
            self.backward(Some(delta));
            return;
        }

        let mut delta = Arc::try_unwrap(delta.flatten_to(Arc::clone(&self.dimensions)).values)
            .unwrap_or_else(|x| (*x).clone());
        consumer_count -= 1;
        let sum = |acc: &mut Vec<Float>, x: &Vec<Float>| {
            acc.iter_mut().zip(x).for_each(|(s, x)| *s += *x);
        };

        while consumer_count > 0 {
            let received = rx.recv().unwrap();
            consumer_count -= 1;
            sum(
                &mut delta,
                &received.flatten_to(Arc::clone(&self.dimensions)).values,
            );
        }

        self.consumer_count.store(0, Ordering::Relaxed);
        *self.tx.lock().unwrap() = None;

        let delta = Array::from((Arc::clone(&self.dimensions), Arc::new(delta)));
        self.backward(Some(delta));
    }

    /// Propagates the number of consumers to each array in the graph.
    fn propagate_consumers(&mut self) {
        for child in &mut *self.children.lock().unwrap() {
            if child.is_tracked {
                child.consumer_count.fetch_add(1, Ordering::Relaxed);
                // don't double-count consumers
                if child.consumer_count.load(Ordering::Relaxed) == 1 {
                    child.propagate_consumers();
                }
            }
        }
    }

    /// Computes the backward pass, computing gradients for all descendants, and propagating consumer counts if requested.
    ///
    /// # Panics
    ///
    /// Panics if the current node has children, but is not a differentiable function (is not a leaf).
    pub fn backward(&mut self, delta: Option<Array>) {
        let mut delta = match delta {
            Some(x) => x,
            None => {
                self.propagate_consumers();
                Array::from((
                    Arc::clone(&self.dimensions),
                    Arc::new(vec![1.0; self.values.len()]),
                ))
            }
        };

        match &self.backward_op {
            Some(x) => {
                let mut children_guard = self.children.lock().unwrap();

                // TODO some closure to simplify tracked, and untracked operations
                let is_tracked: Vec<bool> = children_guard
                    .iter_mut()
                    .map(|c| {
                        let is_tracked = c.is_tracked;
                        c.stop_tracking();
                        is_tracked
                    })
                    .collect();

                let delta = (*x)(&mut children_guard, &is_tracked, &mut delta);

                children_guard
                    .iter_mut()
                    .zip(is_tracked)
                    .filter(|(_, t)| *t)
                    .for_each(|(c, _)| c.start_tracking());

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

        if self.children.lock().unwrap().len() == 0 || self.keep_gradient {
            let mut gradient_guard = self.gradient.lock().unwrap();
            match &mut *gradient_guard {
                Some(x) => *gradient_guard = Some(&*x + &delta),
                None => *gradient_guard = Some(delta),
            }
        }
    }

    /// Computes an operation on slices of arrays with a stride given by the products of each dimensions skipped.
    /// This is useful for broadcasting arrays to compatible dimensions.
    ///
    /// Takes in a vector of arrays, slicing them to meet `input_dimensions`, while skipping the
    /// last `skip_size` dimensions, and performing the operation on the slices, which must have
    /// length of the product of the skipped dimensions.
    ///
    /// # Panics
    ///
    /// Panics if unable to broadcast the arrays to `input_dimensions`.
    fn sliced_op(
        arrays: Vec<&Array>,
        op: &SlicedOp,
        input_dimensions: &[usize],
        output_dimensions: &[usize],
        skip_size: usize,
    ) -> Vec<Float> {
        let is_dimensions_valid = arrays.iter().all(|a| {
            a.dimensions
                .iter()
                .rev()
                .skip(skip_size)
                .zip(input_dimensions.iter().rev().skip(skip_size))
                .all(|(x, y)| *x == 1 || *x == *y)
        });

        if !is_dimensions_valid {
            panic!("error: unable to broadcast arrays to target dimensions");
        }

        // total length of the leading values
        let leading_length = input_dimensions.iter().rev().skip(skip_size).product();
        // total length of the output
        let output_length = output_dimensions.iter().product();

        // count of leading dimensions
        let leading_count = input_dimensions.len().saturating_sub(skip_size);
        // total length of the slicesslice
        let group_lengths: Vec<usize> = arrays
            .iter()
            .map(|a| a.dimensions.iter().rev().take(skip_size).product())
            .collect();
        let output_group_length: usize = output_dimensions.iter().skip(leading_count).product();

        let mut indices = vec![0; leading_count];
        let mut output_values = vec![0.0; output_length];
        for _ in 0..leading_length {
            // add zero indices to the skipped dimensions
            let chained = indices
                .iter()
                .copied()
                .chain(vec![0; input_dimensions.len() - indices.len()])
                .collect::<Vec<usize>>();
            let slices = arrays
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let offset = flatten_indices(&chained, &a.dimensions);
                    &a.values[offset..offset + group_lengths[i]]
                })
                .collect();

            let output_offset = flatten_indices(&chained, &output_dimensions);
            let output_slice =
                &mut output_values[output_offset..output_offset + output_group_length];

            op(output_slice, slices);

            for j in 0..indices.len() {
                let current = indices.len() - j - 1;
                if indices[current] == input_dimensions[current] - 1 {
                    indices[current] = 0;
                } else {
                    indices[current] += 1;
                    break;
                }
            }
        }

        output_values
    }

    /// Flattens the array by summing along dimensions to match the target dimensions.
    fn flatten_to(self, dimensions: Arc<Vec<usize>>) -> Array {
        if self.dimensions == dimensions {
            return self;
        }

        let op: SlicedOp = Box::new(move |output_slice: &mut [Float], arrays: Vec<&[Float]>| {
            for (i, output) in output_slice.iter_mut().enumerate() {
                *output += arrays[0][i];
            }
        });

        let output_values = Array::sliced_op(vec![&self], &op, &self.dimensions, &*dimensions, 0);
        Array::from((dimensions, Arc::new(output_values)))
    }

    /// Computes an operation on arrays.
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
    /// let mul: ForwardOp = Arc::new(|x: &[&Array]| {
    ///     Array::from((x[0].dimensions(), x[0].values().iter().zip(x[1].values()).map(|(x, y)| x * y).collect::<Vec<Float>>()))
    /// });
    ///
    /// let mul_clone = Arc::clone(&mul);
    /// let backward_op: BackwardOp = Arc::new(move |children: &mut Vec<Array>, is_tracked: &[bool], delta: &Array| {
    ///     vec![
    ///         if is_tracked[0] {
    ///             Some(Array::op(&[&children[1], delta], Arc::clone(&mul_clone), None))
    ///         } else {
    ///             None
    ///         },
    ///         if is_tracked[1] {
    ///             Some(Array::op(&[&children[0], delta], Arc::clone(&mul_clone), None))
    ///         } else {
    ///             None
    ///         }
    ///     ]
    /// });
    ///
    /// let a = arr![1.0, 2.0, 3.0].tracked();
    /// let b = arr![3.0, 2.0, 1.0].tracked();
    /// let mut product = Array::op(&vec![&a, &b], mul, Some(backward_op));
    /// assert_eq!(product, arr![3.0, 4.0, 3.0]);
    /// product.backward(None);
    /// assert_eq!(product.gradient().unwrap(), arr![1.0, 1.0, 1.0]);
    /// assert_eq!(b.gradient().unwrap(), arr![1.0, 2.0, 3.0]);
    /// assert_eq!(a.gradient().unwrap(), arr![3.0, 2.0, 1.0]);
    /// # }
    /// ```
    pub fn op(arrays: &[&Array], op: ForwardOp, backward_op: Option<BackwardOp>) -> Array {
        let result = op(arrays);
        result
            .with_children(arrays.iter().map(|v| (*v).clone()).collect())
            .with_backward_op(backward_op)
    }

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
                .with_backward_op(Some(backward_op))
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
                .with_backward_op(Some(backward_op))
        }
    }

    /// Computes the softmax of the array.
    pub fn softmax(&self) -> Array {
        let exponentials = self.exp();
        &exponentials / &exponentials.sum(1)
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
            is_tracked: self.is_tracked,
            keep_gradient: self.keep_gradient,
            gradient: self.gradient.clone(),
        }
    }
}

impl PartialEq for Array {
    fn eq(&self, other: &Array) -> bool {
        *self.dimensions == *other.dimensions && *self.values == *other.values
    }
}

impl AbsDiffEq for Array {
    type Epsilon = <Float as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> <Float as AbsDiffEq>::Epsilon {
        Float::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Array, epsilon: <Float as AbsDiffEq>::Epsilon) -> bool {
        *self.dimensions == *other.dimensions
            && self
                .values
                .iter()
                .zip(other.values.iter())
                .all(|(x, y)| Float::abs_diff_eq(x, y, epsilon))
    }
}

impl RelativeEq for Array {
    fn default_max_relative() -> <Float as AbsDiffEq>::Epsilon {
        Float::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Array,
        epsilon: <Float as AbsDiffEq>::Epsilon,
        max_relative: <Float as AbsDiffEq>::Epsilon,
    ) -> bool {
        *self.dimensions == *other.dimensions
            && self
                .values
                .iter()
                .zip(other.values.iter())
                .all(|(x, y)| Float::relative_eq(x, y, epsilon, max_relative))
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
            panic!(
                "error: the index {} is not compatible with the dimensions {:?}",
                index, self.dimensions
            );
        }

        &self.values[index]
    }
}

impl Index<Vec<usize>> for Array {
    type Output = Float;

    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        &self.values[flatten_indices(&indices, &*self.dimensions)]
    }
}

/// Converts indices by dimension to a single flattened index.
fn flatten_indices(indices: &[usize], dimensions: &[usize]) -> usize {
    let is_indices_valid = indices.len() >= dimensions.len()
        && indices
            .iter()
            .rev()
            .zip(dimensions.iter().rev())
            .filter(|&(i, d)| *i >= *d && *d != 1)
            .peekable()
            .peek()
            .is_none();

    if !is_indices_valid {
        panic!(
            "error: the indices {:?} are not compatible with the dimensions {:?}",
            indices, dimensions
        )
    }

    let mut iter = indices.iter().skip(indices.len() - dimensions.len());
    let first = iter.next().unwrap();

    // dimensions will always have at least one element
    iter.zip(dimensions.iter().skip(1))
        .filter(|&(i, d)| *i < *d || *d != 1)
        .fold(*first, |acc, (i, d)| acc * d + i)
}

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
        let matrix = Array::from(vec![3, 2, 3]);
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
    fn test_consumers_drop() {
        let a = arr![1.0, 2.0, 3.0].tracked();
        let _b = &a * &a;
        let mut c = &a * &a;

        c.backward(None);

        assert_eq!(a.gradient().unwrap(), arr![2.0, 4.0, 6.0]);
    }

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

    #[test]
    fn test_propagate() {
        let a = arr![5.0].tracked();
        let b = arr![2.0].tracked();

        let product = &a * &b;
        let mut sum = &product + &a;

        sum.propagate_consumers();
        assert_eq!(product.consumer_count.load(Ordering::Relaxed), 1);
        assert_eq!(b.consumer_count.load(Ordering::Relaxed), 1);
        assert_eq!(a.consumer_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_propagate_continue() {
        let a = arr![1.0, 2.0].tracked();
        let mut b = arr![5.0, 6.0].tracked();

        b = &b * &a;
        b.propagate_consumers();
        assert!(a.is_tracked);
        assert!(b.is_tracked);
        assert_eq!(a.consumer_count.load(Ordering::Relaxed), 1);

        a.consumer_count.store(0, Ordering::Relaxed);

        b = &b * &a;
        b.propagate_consumers();
        assert_eq!(a.consumer_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_backward_op() {
        let a = arr![5.0].tracked();
        let b = arr![2.0].tracked();

        let product = &a * &b;
        let result = (*product.backward_op.unwrap())(
            &mut vec![a.clone(), b.clone()],
            &[true, true],
            &mut arr![1.0],
        );

        assert_eq!(result.len(), 2);
        assert_eq!(
            result
                .iter()
                .map(|x| x.clone().unwrap())
                .collect::<Vec<Array>>(),
            vec![arr![2.0], arr![5.0]]
        );

        assert!(a.is_tracked);
        assert!(b.is_tracked);
        assert!(product.is_tracked);
    }

    #[test]
    fn test_backward_untracked_both() {
        let a = arr![5.0];
        let b = arr![2.0];

        let mut product = &a * &b;

        product.backward(None);
        assert_eq!(product.gradient().unwrap(), arr![1.0]);
        assert_eq!(*b.gradient.lock().unwrap(), None);
        assert_eq!(*a.gradient.lock().unwrap(), None);
    }

    #[test]
    fn test_backward_repeat() {
        let a = arr![arr![1.0, 2.0, 3.0]].tracked();

        for _ in 0..5 {
            let mut b = &a * &a;
            b.backward(None);
        }

        assert_eq!(a.gradient().unwrap(), arr![arr![10.0, 20.0, 30.0]]);
    }

    #[test]
    fn test_backward_single() {
        let a = arr![5.0].tracked();
        let b = arr![2.0].tracked();

        let mut product = &a * &b;

        product.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![5.0]);
        assert_eq!(a.gradient().unwrap(), arr![2.0]);
    }

    #[test]
    fn test_backward_control_flow() {
        let a = arr![5.0].tracked();
        let b = arr![2.0].tracked();
        let mut c = arr![0.0].tracked();

        for _ in 0..10 {
            c = &c + &(&a * &b);
            if c[0] > 50.0 {
                c = &c * &a;
            }
        }

        assert_eq!(c, arr![195300.0]);

        c.backward(None);
        assert_eq!(c.gradient().unwrap(), arr![1.0]);
        assert_eq!(b.gradient().unwrap(), arr![97650.0]);
        assert_eq!(a.gradient().unwrap(), arr![232420.0]);
    }

    #[test]
    fn test_backward_delta() {
        let a = arr![5.0].tracked();
        let b = arr![2.0].tracked();

        let mut product = &a * &b;

        product.backward(Some(arr![5.0]));
        assert_eq!(b.gradient().unwrap(), arr![25.0]);
        assert_eq!(a.gradient().unwrap(), arr![10.0]);
    }

    #[test]
    fn test_backward_dimensions() {
        let a = arr![arr![5.0, 2.0], arr![3.0, 1.0]].tracked();
        let b = arr![arr![6.0, 3.0], arr![7.0, 8.0]].tracked();

        let mut product = &a * &b;

        product.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![arr![5.0, 2.0], arr![3.0, 1.0]]);
        assert_eq!(a.gradient().unwrap(), arr![arr![6.0, 3.0], arr![7.0, 8.0]]);
    }

    #[test]
    fn test_backward_multi() {
        let a = arr![5.0, 2.0].tracked();
        let b = arr![6.0, 3.0].tracked();
        let c = (&a * &b).tracked();
        let d = (&c + &a).tracked();
        let mut e = (&a * &d).tracked();

        e.backward(None);
        assert_eq!(e.gradient().unwrap(), arr![1.0, 1.0]);
        assert_eq!(d.gradient().unwrap(), arr![5.0, 2.0]);
        assert_eq!(c.gradient().unwrap(), arr![5.0, 2.0]);
        assert_eq!(b.gradient().unwrap(), arr![25.0, 4.0]);
        assert_eq!(a.gradient().unwrap(), arr![70.0, 16.0]);
    }

    #[test]
    fn test_backward_intermediate() {
        let a = arr![1.0, 2.0].tracked();
        let b = arr![5.0, 3.0].tracked();
        let c = (&(&(&a * &b) + &a) * &b).tracked();
        let mut product = &c * &a;

        product.backward(None);
        assert_eq!(c.gradient().unwrap(), arr![1.0, 2.0]);
        assert_eq!(b.gradient().unwrap(), arr![11.0, 28.0]);
        assert_eq!(a.gradient().unwrap(), arr![60.0, 48.0]);
    }

    #[test]
    fn test_backward_reassign() {
        let a = arr![1.0, 2.0].tracked();
        let mut b = arr![5.0, 6.0].tracked();

        b = &b + &a;
        b = &b * &a;

        b.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![1.0, 1.0]);
        assert_eq!(a.gradient().unwrap(), arr![7.0, 10.0]);
    }

    #[test]
    fn test_backward_continue() {
        let a = arr![1.0, 2.0].tracked();
        let mut b = arr![5.0, 6.0].tracked();

        b = &b * &a;
        assert_eq!(b, arr![5.0, 12.0]);

        b.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![1.0, 1.0]);
        assert_eq!(a.gradient().unwrap(), arr![5.0, 6.0]);

        b = &b * &a;
        assert_eq!(b, arr![5.0, 24.0]);

        b.backward(None);
        assert_eq!(b.gradient().unwrap(), arr![1.0, 1.0]);
        assert_eq!(a.gradient().unwrap(), arr![15.0, 30.0]);
    }

    #[test]
    fn test_backward_untracked() {
        let a = arr![1.0, 2.0].tracked();
        let b = arr![2.0, 1.0];
        let mut product = &a * &b;

        product.backward(None);
        assert!(b.gradient().is_none());
        assert!(a.gradient().is_some());
    }

    #[test]
    fn test_backward_untracked_clone() {
        let a = arr![1.0, 2.0].tracked();
        let b = arr![2.0, 1.0];
        let product = &a * &b;
        let mut clone = product.clone();

        clone.backward(None);
        assert!(b.gradient().is_none());
        assert!(a.gradient().is_some());
    }
}
