//! Initializers initialize the parameters of a model.

use crate::numbers::*;

use rand::Rng;

use std::sync::Arc;

/// A parameter initializer, which intializes parameters based on the input size.
pub type Initializer = Arc<dyn Fn(Float) -> Float>;

/// Creates a He initializer closure.
pub fn make_he() -> Initializer {
    Arc::new(|x| {
        let stddev = (2.0 / x).sqrt();
        rand::thread_rng().gen_range(-stddev..=stddev)
    })
}
