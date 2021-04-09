//! A weight initializer.

use crate::numbers::*;

use std::sync::Arc;

/// A weight initializer, which intializes weights based on the input size.
pub type Initializer = Arc<dyn Fn(Float) -> Float>;
