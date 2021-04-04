//! Floating point type wrapper, which may be changed to `f32` when the feature "f32" is active.

#[cfg(feature = "f32")]
/// The single-precision floating point type.
pub type Float = f32;
#[cfg(not(feature = "f32"))]
/// The double-precision floating point type.
pub type Float = f64;
