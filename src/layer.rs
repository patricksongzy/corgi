use crate::numbers::*;
use crate::array::*;

pub trait Layer {
    fn forward(&self, x: Array) -> Array;

    fn update(&self, target: Array);
}
