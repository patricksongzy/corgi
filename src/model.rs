use crate::array::*;
use crate::numbers::*;

trait Model {
    fn forward(&self, x: Array) -> Array;
}
