use crate::array::*;
use crate::numbers::*;

pub trait Layer {
    fn forward(&self, x: Array) -> Array;

    fn update(&self, target: Array);
}
