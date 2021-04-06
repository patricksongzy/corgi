use crate::numbers::*;
use crate::array::*;

trait Model {
    fn forward(&self, x: Array) -> Array;
}
