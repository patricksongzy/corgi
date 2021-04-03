use crate::numbers::*;
use crate::array::*;

trait Layer {
    fn forward(x: Array) -> Array;
}
