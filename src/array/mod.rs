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
//! assert_eq!(c.gradient().to_owned().unwrap(), arr![1.0]);
//! assert_eq!(b.gradient().to_owned().unwrap(), arr![97650.0]);
//! assert_eq!(a.gradient().to_owned().unwrap(), arr![232420.0]);
//! # }
//! ```

mod arithmetic;
mod image;
mod linalg;
mod nonlinearity;

use crate::numbers::*;

use approx::{AbsDiffEq, RelativeEq};

use std::cmp;
use std::convert::{From, Into};
use std::fmt;

use std::ops;
use std::ops::Index;

use std::cell::{Cell, Ref, RefCell, RefMut};
use std::rc::Rc;

/// The sliced operation computes an operation with respect to slices on a mutable output slice.
type SlicedOp = Box<dyn Fn(&mut [Float], &[&[Float]])>;
/// The forward operation computes an operation with respect to inputs.
pub type ForwardOp = Rc<dyn Fn(&[&Array]) -> Array>;
/// The backward operation computes deltas with respect to inputs.
pub type BackwardOp = Rc<dyn Fn(&[Array], &[bool], &Array) -> Vec<Option<Array>>>;

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
    dimensions: Vec<usize>,
    values: Rc<Vec<Float>>,
    children: Rc<Vec<Array>>,
    backward_op: Option<BackwardOp>,
    is_tracked: Cell<bool>,
    keep_gradient: Cell<bool>,
    consumer_count: Rc<Cell<usize>>,
    delta: Rc<Cell<Option<Array>>>,
    gradient: Rc<RefCell<Option<Array>>>,
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

        assert!(
            is_dimensions_valid,
            "error: contained array dimensions must all be the same"
        );

        let mut dimensions = vec![contents.len()];
        dimensions.extend(&contents.first().unwrap().dimensions);

        // take ownership if possible, but clone otherwise
        let values: Vec<Float> = contents
            .into_iter()
            .map(|array| Rc::try_unwrap(array.values).unwrap_or_else(|x| (*x).clone()))
            .flatten()
            .collect();

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
        Array::from((dimensions, Rc::new(values)))
    }
}

impl From<(Vec<usize>, Rc<Vec<Float>>)> for Array {
    fn from(items: (Vec<usize>, Rc<Vec<Float>>)) -> Self {
        let (dimensions, values) = items;

        let is_dimensions_valid = dimensions.iter().all(|d| *d >= 1);
        assert!(
            is_dimensions_valid,
            "error: invalid dimensions {:?}",
            dimensions
        );

        let is_values_valid = dimensions.iter().product::<usize>() == values.len();
        assert!(
            is_values_valid,
            "error: dimensions, and values must be of the same length"
        );

        Array {
            dimensions,
            values,
            children: Rc::new(Vec::new()),
            backward_op: None,
            is_tracked: Cell::new(false),
            keep_gradient: Cell::new(false),
            consumer_count: Rc::new(Cell::new(0)),
            delta: Rc::new(Cell::new(None)),
            gradient: Rc::new(RefCell::new(None)),
        }
    }
}

impl Into<Vec<Float>> for Array {
    fn into(self) -> Vec<Float> {
        Rc::try_unwrap(self.values).unwrap()
    }
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
fn element_wise_dimensions(x: &[usize], y: &[usize]) -> Vec<usize> {
    let (mut longer, other) = if x.len() > y.len() {
        (x.to_owned(), y)
    } else {
        (y.to_owned(), x)
    };

    for (l, o) in longer.iter_mut().rev().zip(other.iter().rev()) {
        assert!(
            *l == *o || *l == 1 || *o == 1,
            "error: element-wise operation dimensions, {:?}, and {:?} must be matching",
            x,
            y
        );

        *l = cmp::max(*l, *o);
    }

    longer
}

// TODO better error handling
impl Array {
    /// Returns a copy of the dimensions of the array.
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Returns an immutable reference to the values of the array in row-major order.
    pub fn values(&self) -> &[Float] {
        &*self.values
    }

    /// Returns a reference to the gradient option of the array.
    pub fn gradient(&self) -> Ref<Option<Array>> {
        self.gradient.borrow()
    }

    /// Returns a mutable reference to the gradient option of the array.
    pub fn gradient_mut(&self) -> RefMut<Option<Array>> {
        self.gradient.borrow_mut()
    }

    /// Returns the owned gradient option of the array, replacing it with nothing.
    pub fn replace_gradient(&self) -> Option<Array> {
        self.gradient.replace(None)
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
    /// assert_eq!(b.gradient().to_owned().unwrap(), arr![1.0, 2.0, 3.0]);
    /// # }
    /// ```
    pub fn tracked(self) -> Array {
        self.keep_gradient.replace(true);
        self.is_tracked.replace(true);
        self
    }

    /// Starts tracking operations for a mutable reference to an array, returning the previous value.
    pub fn start_tracking(&self) -> bool {
        self.is_tracked.replace(true)
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
    /// assert_eq!(b.gradient().to_owned().unwrap(), arr![1.0, 2.0, 3.0]);
    /// # }
    /// ```
    pub fn untracked(self) -> Array {
        self.keep_gradient.replace(false);
        self.is_tracked.replace(false);
        self
    }

    /// Stops tracking operations for a mutable reference to an array, returning the previous value.
    /// Useful for temporarily updating parameters without requiring their gradients.
    pub fn stop_tracking(&self) -> bool {
        self.is_tracked.replace(false)
    }

    /// Adds `Vec<Array>` as the children of a vector.
    fn with_children(mut self, children: Vec<Array>) -> Array {
        self.children = Rc::new(children);
        self.tracked()
    }

    /// Sets the backward operation of the array for the backward pass.
    fn with_backward_op(mut self, backward_op: BackwardOp) -> Array {
        self.backward_op = Some(backward_op);
        self
    }

    /// Propagates the number of consumers to each array in the graph.
    fn propagate_consumers(&self) {
        for child in self.children.iter() {
            if child.is_tracked.get() {
                let child_consumer_count = child.consumer_count.get();
                child.consumer_count.set(child_consumer_count + 1);
                // don't double-count consumers
                if child_consumer_count == 0 {
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
    pub fn backward(&self, delta: Option<Array>) {
        let mut delta = {
            if let Some(x) = self.delta.take() {
                x
            } else {
                self.propagate_consumers();
                match delta {
                    Some(x) => x,
                    None => Array::from((
                        self.dimensions.clone(),
                        Rc::new(vec![1.0; self.values.len()]),
                    )),
                }
            }
        };

        match &self.backward_op {
            Some(x) => {
                let is_tracked: Vec<bool> =
                    self.children.iter().map(|c| c.stop_tracking()).collect();

                let delta = (*x)(&self.children, &is_tracked, &mut delta);

                self.children
                    .iter()
                    .zip(is_tracked)
                    .filter(|(_, t)| *t)
                    .for_each(|(c, _)| {
                        c.start_tracking();
                    });

                for (i, delta) in delta.into_iter().enumerate() {
                    if let Some(delta) = delta {
                        let child = &self.children[i];
                        {
                            match child.delta.take() {
                                Some(x) => child.delta.set(Some(&x + &delta)),
                                None => child.delta.set(Some(delta.flatten_to(&child.dimensions))),
                            }
                        }

                        let child_consumer_count = child.consumer_count.get();
                        child.consumer_count.set(child_consumer_count - 1);
                        if child_consumer_count == 1 {
                            child.backward(None);
                        }
                    }
                }
            }
            None => {
                assert!(
                    self.children.is_empty(),
                    "error: backward pass called on non-differentiable operation"
                );
            }
        }

        if self.children.is_empty() || self.keep_gradient.get() {
            let mut gradient = self.gradient.borrow_mut();
            match &mut *gradient {
                Some(x) => *gradient = Some(&*x + &delta),
                None => *gradient = Some(delta),
            }
        }
    }

    /// Computes an operation on slices of arrays with a stride given by the products of each dimensions skipped.
    /// This is useful for broadcasting arrays to compatible dimensions.
    ///
    /// Takes in a vector of arrays, slicing them to meet `input_dimensions`, while skipping the
    /// last `op_dimension_count` dimensions, and performing the operation on the slices, which must have
    /// length of the product of the skipped dimensions.
    ///
    /// # Arguments
    ///
    /// * `arrays` - The arrays to perform the operations on.
    /// * `op` - The `SlicedOp`, which takes in slices of the arrays, and modifies the output slice.
    /// * `input_dimensions` - The target dimensions to broadcast inputs to.
    /// * `output_dimensions` - The output dimensions of the operation.
    /// * `op_dimension_count` - The number of dimensions which form the input slices.
    /// * `flatten_count` - The number of dimensions which are flattened in the output.
    ///
    /// # Panics
    ///
    /// Panics if unable to broadcast the arrays to `input_dimensions`.
    fn sliced_op(
        arrays: Vec<&Array>,
        op: &SlicedOp,
        backward_op: Option<BackwardOp>,
        input_dimensions: &[usize],
        output_dimensions: &[usize],
        op_dimension_count: usize,
        flatten_count: usize,
    ) -> Array {
        let is_dimensions_valid = arrays.iter().all(|v| {
            v.dimensions
                .iter()
                .rev()
                .skip(op_dimension_count)
                .zip(input_dimensions.iter().rev().skip(op_dimension_count))
                .all(|(x, y)| *x == 1 || *x == *y)
        });

        assert!(
            is_dimensions_valid,
            "error: unable to broadcast arrays to target dimensions"
        );

        // count of leading dimensions
        let leading_count = input_dimensions.len().saturating_sub(op_dimension_count);
        let output_leading_count = output_dimensions.len().saturating_sub(op_dimension_count);

        let unbroadcasted_group_length: usize = input_dimensions
            .iter()
            .rev()
            .take(op_dimension_count)
            .product();
        let mut slices: Vec<&[Float]> = Vec::with_capacity(arrays.len());
        let mut broadcasted_slices: Vec<(&Array, usize, *mut &[Float])> = Vec::new();
        let mut unbroadcasted_slices: Vec<(&Array, *mut &[Float])> = Vec::new();
        for array in &arrays {
            let index = slices.len();
            // total length of the slices
            if array.dimensions == input_dimensions {
                let slice = &array.values[0..unbroadcasted_group_length];
                slices.push(slice);
                unbroadcasted_slices.push((array, &mut slices[index]));
            } else {
                let group_length = array
                    .dimensions
                    .iter()
                    .rev()
                    .take(op_dimension_count)
                    .product();
                let slice = &array.values[0..group_length];
                slices.push(slice);
                broadcasted_slices.push((array, group_length, &mut slices[index]));
            }
        }

        // total length of the output
        let output_length = output_dimensions.iter().product();
        let output_group_length: usize = output_dimensions.iter().skip(leading_count).product();
        let mut output_values = vec![0.0; output_length];
        let mut output_slice = &mut output_values[0..output_group_length];

        if leading_count == 0 {
            op(output_slice, &slices);
        } else {
            // total length of the leading values
            let leading_length = input_dimensions
                .iter()
                .rev()
                .skip(op_dimension_count)
                .product();

            let mut indices = vec![0; cmp::max(input_dimensions.len(), output_dimensions.len())];
            let mut flat_unbroadcasted_index = 0;
            let mut flat_broadcasted_indices = vec![0; broadcasted_slices.len()];
            let mut output_offset = 0;
            for _ in 0..leading_length {
                op(output_slice, &slices);

                for (i, (x, d)) in indices
                    .iter_mut()
                    .zip(input_dimensions)
                    .enumerate()
                    .rev()
                    .skip(op_dimension_count)
                {
                    // increment the first dimension that is not about to overflow
                    if *x == *d - 1 {
                        *x = 0;
                    } else {
                        // do not increment the index if we have broadcasted by extending the array's dimensions (if the current dimension is greater than the array's dimensions)
                        // or if the dimension is broadcasted from 1
                        for (index, &(array, group_length, slice)) in
                            flat_broadcasted_indices.iter_mut().zip(&broadcasted_slices)
                        {
                            if i < array.dimensions.len().saturating_sub(op_dimension_count)
                                && array.dimensions[i] != 1
                            {
                                *index += group_length;
                                unsafe {
                                    // we may use unsafe code, since we are just modifying slices
                                    *slice = &array.values[*index..*index + group_length];
                                }
                            }
                        }

                        if i < leading_count && input_dimensions[i] != 1 {
                            flat_unbroadcasted_index += unbroadcasted_group_length;
                            for &(array, slice) in &unbroadcasted_slices {
                                unsafe {
                                    // we may use unsafe code, since we are just modifying slices
                                    *slice = &array.values[flat_unbroadcasted_index
                                        ..flat_unbroadcasted_index + unbroadcasted_group_length];
                                }
                            }
                        }

                        if i < output_leading_count && output_dimensions[i] != 1
                        {
                            output_offset += output_group_length;
                            output_slice = &mut output_values
                                [output_offset..output_offset + output_group_length];
                        }

                        *x += 1;
                        break;
                    }
                }
            }
        }

        let mut output_dimensions = output_dimensions.to_vec();
        if flatten_count > 0 {
            let flattened_dimensions = output_dimensions
                [output_dimensions.len() - flatten_count..output_dimensions.len()]
                .iter()
                .product();

            let flattened_length = output_dimensions.len() - flatten_count + 1;
            output_dimensions.truncate(flattened_length);
            output_dimensions[flattened_length - 1] = flattened_dimensions;
        }

        let result = Array::from((output_dimensions, output_values));
        if let Some(backward_op) = backward_op {
            result
                .with_children(arrays.into_iter().cloned().collect())
                .with_backward_op(backward_op)
        } else {
            result
        }
    }

    /// Flattens the array by summing along dimensions to match the target dimensions.
    fn flatten_to(self, dimensions: &[usize]) -> Array {
        if self.dimensions == dimensions {
            self
        } else {
            let flatten_dimension_count = self.dimensions.len().saturating_sub(dimensions.len());

            let op: SlicedOp = Box::new(move |output_slice, arrays| {
                let stride = output_slice.len();
                for (i, output) in output_slice.iter_mut().enumerate() {
                    *output += arrays[0].iter().skip(i).step_by(stride).sum::<Float>();
                }
            });

            Array::sliced_op(
                vec![&self],
                &op,
                None,
                &self.dimensions,
                &*dimensions,
                flatten_dimension_count + 1,
                0,
            )
        }
    }

    /// Computes an operation on arrays.
    ///
    /// # Arguments
    ///
    /// * `arrays` - The arrays to perform the operations on.
    /// * `op` - The `ForwardOp`, which takes in the arrays, and outputs another array.
    /// * `backward_op` - The `BackwardOp`, which takes in the arrays, and the delta, and outputs a
    /// new delta, with respect to each input. It is recommended that any array operations here are
    /// untracked, unless interested in higher order derivatives.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate corgi;
    /// # use std::rc::Rc;
    /// # use corgi::numbers::*;
    /// # use corgi::array::*;
    /// # fn main () {
    /// let mul: ForwardOp = Rc::new(|x: &[&Array]| {
    ///     Array::from((x[0].dimensions().to_vec(), x[0].values().iter().zip(x[1].values()).map(|(x, y)| x * y).collect::<Vec<Float>>()))
    /// });
    ///
    /// let mul_clone = Rc::clone(&mul);
    /// let backward_op: BackwardOp = Rc::new(move |children, is_tracked, delta| {
    ///     vec![
    ///         if is_tracked[0] {
    ///             Some(Array::op(&[&children[1], delta], Rc::clone(&mul_clone), None))
    ///         } else {
    ///             None
    ///         },
    ///         if is_tracked[1] {
    ///             Some(Array::op(&[&children[0], delta], Rc::clone(&mul_clone), None))
    ///         } else {
    ///             None
    ///         }
    ///     ]
    /// });
    ///
    /// let a = arr![1.0, 2.0, 3.0].tracked();
    /// let b = arr![3.0, 2.0, 1.0].tracked();
    /// let product = Array::op(&vec![&a, &b], mul, Some(backward_op));
    /// assert_eq!(product, arr![3.0, 4.0, 3.0]);
    /// product.backward(None);
    /// assert_eq!(product.gradient().to_owned().unwrap(), arr![1.0, 1.0, 1.0]);
    /// assert_eq!(b.gradient().to_owned().unwrap(), arr![1.0, 2.0, 3.0]);
    /// assert_eq!(a.gradient().to_owned().unwrap(), arr![3.0, 2.0, 1.0]);
    /// # }
    /// ```
    pub fn op(arrays: &[&Array], op: ForwardOp, backward_op: Option<BackwardOp>) -> Array {
        let result = op(arrays);
        if let Some(backward_op) = backward_op {
            result
                .with_children(arrays.iter().map(|v| (*v).clone()).collect())
                .with_backward_op(backward_op)
        } else {
            result
        }
    }
}

impl Clone for Array {
    fn clone(&self) -> Array {
        let backward_op = match &self.backward_op {
            Some(x) => Some(Rc::clone(&x)),
            None => None,
        };

        Array {
            dimensions: self.dimensions.clone(),
            values: Rc::clone(&self.values),
            children: Rc::clone(&self.children),
            backward_op,
            is_tracked: Cell::new(self.is_tracked.get()),
            keep_gradient: Cell::new(self.keep_gradient.get()),
            consumer_count: Rc::clone(&self.consumer_count),
            delta: Rc::clone(&self.delta),
            gradient: Rc::clone(&self.gradient),
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Array")
            .field("dimensions", &self.dimensions)
            .field("values", &*self.values)
            .field("consumers", &*self.consumer_count)
            .field("tracked", &self.is_tracked)
            .finish()
    }
}

impl Index<usize> for Array {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < self.values.len(),
            "error: the index {} is not compatible with the dimensions {:?}",
            index,
            self.dimensions
        );

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
    // dimensions will always have at least one element
    let mut iter = indices.iter().skip(indices.len() - dimensions.len());
    let first = iter.next().unwrap();
    iter.zip(dimensions.iter().skip(1))
        .filter(|&(_, d)| *d != 1)
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

        assert_eq!(matrix.dimensions, vec![3, 2, 1]);
        assert_eq!(
            *matrix.values,
            (0..6).map(|x| x as Float).collect::<Vec<Float>>()
        );
    }

    #[test]
    fn test_zeros() {
        let matrix = Array::from(vec![3, 2, 3]);
        assert_eq!(matrix.dimensions, vec![3, 2, 3]);
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
        let c = &a * &a;

        c.backward(None);

        assert_eq!(a.gradient().to_owned().unwrap(), arr![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_propagate() {
        let a = arr![5.0].tracked();
        let b = arr![2.0].tracked();

        let product = &a * &b;
        let sum = &product + &a;

        sum.propagate_consumers();
        assert_eq!(product.consumer_count.get(), 1);
        assert_eq!(b.consumer_count.get(), 1);
        assert_eq!(a.consumer_count.get(), 2);
    }

    #[test]
    fn test_propagate_continue() {
        let a = arr![1.0, 2.0].tracked();
        let mut b = arr![5.0, 6.0].tracked();

        b = &b * &a;
        b.propagate_consumers();
        assert!(a.is_tracked.get());
        assert!(b.is_tracked.get());
        assert_eq!(a.consumer_count.get(), 1);

        a.consumer_count.set(0);

        b = &b * &a;
        b.propagate_consumers();
        assert_eq!(a.consumer_count.get(), 2);
    }

    // TODO should this panic?
    #[test]
    #[should_panic]
    fn test_sliced_op() {
        let a = arr![1.0];
        let op: SlicedOp = Box::new(move |output_slice, arrays| {
            output_slice[0] = arrays[0][0] + arrays[0][1];
        });

        Array::sliced_op(vec![&a], &op, None, &[2], &[1], 0, 0);
    }

    #[test]
    fn test_flatten_to() {
        let a = arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]];
        // TODO error checking
        let result = a.flatten_to(&[3]);

        assert_eq!(result, arr![5.0, 7.0, 9.0]);

        let b = arr![
            arr![arr![1.0, 2.0, 3.0], arr![4.0, 5.0, 6.0]],
            arr![arr![7.0, 8.0, 9.0], arr![10.0, 11.0, 12.0]]
        ];
        let result = b.flatten_to(&[3]);
        assert_eq!(result, arr![22.0, 26.0, 30.0]);
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

        assert!(a.is_tracked.get());
        assert!(b.is_tracked.get());
        assert!(product.is_tracked.get());
    }

    #[test]
    fn test_backward_untracked_both() {
        let a = arr![5.0];
        let b = arr![2.0];

        let product = &a * &b;

        product.backward(None);
        assert_eq!(product.gradient().to_owned().unwrap(), arr![1.0]);
        assert!(b.gradient().is_none());
        assert!(a.gradient().is_none());
    }

    #[test]
    fn test_backward_repeat() {
        let a = arr![arr![1.0, 2.0, 3.0]].tracked();

        for _ in 0..5 {
            let b = &a * &a;
            b.backward(None);
        }

        assert_eq!(
            a.gradient().to_owned().unwrap(),
            arr![arr![10.0, 20.0, 30.0]]
        );
    }

    #[test]
    fn test_backward_single() {
        let a = arr![5.0].tracked();
        let b = arr![2.0].tracked();

        let product = &a * &b;

        product.backward(None);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![5.0]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![2.0]);
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
        assert_eq!(c.gradient().to_owned().unwrap(), arr![1.0]);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![97650.0]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![232420.0]);
    }

    #[test]
    fn test_backward_delta() {
        let a = arr![5.0].tracked();
        let b = arr![2.0].tracked();

        let product = &a * &b;

        product.backward(Some(arr![5.0]));
        assert_eq!(b.gradient().to_owned().unwrap(), arr![25.0]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![10.0]);
    }

    #[test]
    fn test_backward_dimensions() {
        let a = arr![arr![5.0, 2.0], arr![3.0, 1.0]].tracked();
        let b = arr![arr![6.0, 3.0], arr![7.0, 8.0]].tracked();

        let product = &a * &b;

        product.backward(None);
        assert_eq!(
            b.gradient().to_owned().unwrap(),
            arr![arr![5.0, 2.0], arr![3.0, 1.0]]
        );
        assert_eq!(
            a.gradient().to_owned().unwrap(),
            arr![arr![6.0, 3.0], arr![7.0, 8.0]]
        );
    }

    #[test]
    fn test_backward_multi() {
        let a = arr![5.0, 2.0].tracked();
        let b = arr![6.0, 3.0].tracked();
        let c = (&a * &b).tracked();
        let d = (&c + &a).tracked();
        let e = (&a * &d).tracked();

        e.backward(None);
        assert_eq!(e.gradient().to_owned().unwrap(), arr![1.0, 1.0]);
        assert_eq!(d.gradient().to_owned().unwrap(), arr![5.0, 2.0]);
        assert_eq!(c.gradient().to_owned().unwrap(), arr![5.0, 2.0]);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![25.0, 4.0]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![70.0, 16.0]);
    }

    #[test]
    fn test_backward_intermediate() {
        let a = arr![1.0, 2.0].tracked();
        let b = arr![5.0, 3.0].tracked();
        let c = (&(&(&a * &b) + &a) * &b).tracked();
        let product = &c * &a;

        product.backward(None);
        assert_eq!(c.gradient().to_owned().unwrap(), arr![1.0, 2.0]);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![11.0, 28.0]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![60.0, 48.0]);
    }

    #[test]
    fn test_backward_drop() {
        let a = arr![6.0].tracked();
        let b = arr![7.0].tracked();
        let c = arr![8.0].tracked();
        let _d = arr![9.0].tracked();
        let e = &(&a + &b) * &c;
        std::mem::drop(a);
        let p = &e * &b;
        p.backward(None);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![160.0]);
    }

    #[test]
    fn test_backward_reassign() {
        let a = arr![1.0, 2.0].tracked();
        let mut b = arr![5.0, 6.0].tracked();

        b = &b + &a;
        b = &b * &a;

        b.backward(None);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![1.0, 1.0]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![7.0, 10.0]);
    }

    #[test]
    fn test_backward_continue() {
        let a = arr![1.0, 2.0].tracked();
        let mut b = arr![5.0, 6.0].tracked();

        b = &b * &a;
        assert_eq!(b, arr![5.0, 12.0]);

        b.backward(None);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![1.0, 1.0]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![5.0, 6.0]);

        b = &b * &a;
        assert_eq!(b, arr![5.0, 24.0]);

        b.backward(None);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![1.0, 1.0]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![15.0, 30.0]);
    }

    #[test]
    fn test_backward_untracked() {
        let a = arr![1.0, 2.0].tracked();
        let b = arr![2.0, 1.0];
        let product = &a * &b;

        product.backward(None);
        assert!(b.gradient().is_none());
        assert!(a.gradient().is_some());
    }

    #[test]
    fn test_backward_untracked_clone() {
        let a = arr![1.0, 2.0].tracked();
        let b = arr![2.0, 1.0];
        let product = &a * &b;
        let clone = product.clone();

        clone.backward(None);
        assert!(b.gradient().is_none());
        assert!(a.gradient().is_some());
    }
}
