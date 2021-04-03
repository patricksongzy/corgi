use crate::numbers::*;
use crate::array::*;

trait Model {
    fn forward(x: Array) -> Array;
}
