//! Initializers initialize the parameters of a model.

use crate::numbers::*;

use rand::Rng;

/// A parameter initializer, which intializes parameters based on the input size.
pub type Initializer = Box<dyn Fn(Float) -> Float>;

/// Creates a He initializer closure.
pub fn he() -> Initializer {
    Box::new(|x| {
        let stddev = (2.0 / x).sqrt();
        rand::thread_rng().gen_range(-stddev..=stddev)
    })
}
